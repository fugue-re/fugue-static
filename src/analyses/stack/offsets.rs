use std::collections::{BTreeMap, BTreeSet};

use fugue::ir::convention::Convention;
use fugue::ir::il::ecode::{Stmt, Var, Expr, BinOp};
use good_lp::*;

use crate::graphs::AsEntityGraph;
use crate::graphs::entity::VertexIndex;
use crate::models::{Block, Phi};
use crate::models::cfg::BranchKind;
use crate::traits::stmt::*;
use crate::types::{EntityRef, Identifiable, IntoEntityRef, Located, SimpleVar};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhiOrStmt<'a> {
    Phi(EntityRef<'a, Located<Phi>>),
    Stmt(EntityRef<'a, Located<Stmt>>),
}

impl<'a> PhiOrStmt<'a> {
    pub fn is_phi(&self) -> bool {
        matches!(self, Self::Phi(_))
    }

    pub fn is_stmt(&self) -> bool {
        matches!(self, Self::Stmt(_))
    }
}

impl<'a> From<EntityRef<'a, Located<Phi>>> for PhiOrStmt<'a> {
    fn from(phi: EntityRef<'a, Located<Phi>>) -> Self {
        Self::Phi(phi)
    }
}

impl<'a> From<EntityRef<'a, Located<Stmt>>> for PhiOrStmt<'a> {
    fn from(stmt: EntityRef<'a, Located<Stmt>>) -> Self {
        Self::Stmt(stmt)
    }
}

pub struct StackOffsets<'a>(BTreeMap<PhiOrStmt<'a>, i64>);

fn constant(expr: &Expr) -> Option<i64> {
    if let Expr::Val(v) = expr {
        v.signed_cast(v.bits()).to_i64()
    } else {
        None
    }
}

impl<'a> StackOffsets<'a> {
    pub fn analyse_with<G>(g: &'a G, tracked: &Var, convention: &Convention) -> Option<Self>
    where G: AsEntityGraph<'a, Block, BranchKind> {
        let g = g.entity_graph();

        let mut vars = BTreeMap::<VertexIndex<_>, _>::default();
        let mut offsets = BTreeMap::default();

        let mut lp_vars = variables!();
        let mut constraints = Vec::default();

        let tracked = SimpleVar::from(tracked);
        let extra_pop = convention.default_prototype().extra_pop() as f64;

        let is_tracked = |expr: &Expr| { matches!(expr, Expr::Var(v) if SimpleVar::from(v) == tracked) };

        let roots = g.root_entities()
            .into_iter().map(|(id, _, _)| *id)
            .collect::<BTreeSet<_>>();

        for vx in g.reverse_post_order().into_iter() {
            let b = g.entity(vx);

            let (b_in, b_out) = if let Some(b_inout) = vars.get(&vx) {
                *b_inout
            } else {
                let b_in = lp_vars.add(variable().integer());
                let b_out = lp_vars.add(variable().integer());
                vars.insert(vx, (b_in, b_out));
                (b_in, b_out)
            };

            let in_c = if roots.contains(&b.id()) { // in is 0
                constraint!(b_in == 0)
            } else {
                constraint!(b_in >= -0xffff)
            };

            let out_c = if b.last().is_return() { // out is extra_pop
                constraint!(b_out == extra_pop)
            } else {
                constraint!(b_out >= -0xffff)
            };

            constraints.push(in_c);
            constraints.push(out_c);

            let mut shift = 0i64;

            for phi in b.phis() {
                let r = phi.into_entity_ref();
                offsets.insert(PhiOrStmt::from(r), Expression::from(b_in + shift as f64));
            }

            for op in b.operations() {
                let r = op.into_entity_ref();
                let e = offsets.entry(PhiOrStmt::from(r)).or_insert(Expression::from(b_in + shift as f64));

                match &**op.value() {
                    Stmt::Assign(v, exp) if SimpleVar::from(v) == tracked => {
                        // look at used in exp, if only in terms of v, then build shift
                        match exp {
                            Expr::Val(ref bv) => { shift = bv.signed_cast(bv.bits()).to_i64()?; },
                            Expr::BinOp(BinOp::SUB, l, r) if is_tracked(l) => if let Some(v) = constant(&**r) {
                                shift = shift - v;
                            },
                            Expr::BinOp(op, l, r) => if let Some(v) = {
                                if is_tracked(l) {
                                    constant(&**r)
                                } else if is_tracked(r) {
                                    constant(&**l)
                                } else {
                                    continue
                                }
                            } {
                                if *op == BinOp::ADD {
                                    shift = shift + v;
                                } else if *op == BinOp::AND {
                                    shift = shift & v;
                                } else if *op == BinOp::OR {
                                    shift = shift | v;
                                }
                            },
                            _ => (),
                        }
                    }
                    _ => (),
                };

                *e = Expression::from(b_in + shift as f64);
            }

            if shift == 0 {
                constraints.push(constraint!(b_out - b_in >= shift as f64));
            } else {
                constraints.push(constraint!(b_out - b_in == 0));
            }

            // RPO should ensure preds have already been visited
            for (px, _)  in g.predecessors(vx) {
                let p_in = if let Some(p_inout) = vars.get(&px) {
                    p_inout.0
                } else {
                    let p_in = lp_vars.add(variable().integer());
                    let p_out = lp_vars.add(variable().integer());
                    vars.insert(px, (p_in, p_out));
                    p_in
                };

                constraints.push(constraint!(p_in - b_out == 0));
            }
        }

        let problem = vars.values().fold(0.into(), |v, &(pi, po)| pi + po + v);
        let mut model = lp_vars
            .minimise(problem)
            .using(default_solver);

        for c in constraints.into_iter() {
            model = model.with(c);
        }

        let solution = model.solve().ok()?;

        Some(StackOffsets(
            offsets.into_iter().map(|(k, v)| (k, solution.eval(v) as i64)).collect()
        ))
    }

    pub fn offset_at<T, E>(&self, entity: E) -> Option<i64>
    where
        T: Clone + 'a,
        E: IntoEntityRef<'a, T = T>,
        PhiOrStmt<'a>: From<EntityRef<'a, T>>
    {
        self.0.get(&entity.into_entity_ref().into()).copied()
    }
}
