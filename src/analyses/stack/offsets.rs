use std::collections::{BTreeMap, BTreeSet};

use fugue::ir::convention::Convention;
use fugue::ir::il::ecode::{BinOp, Expr, Stmt, Var};
use good_lp::*;

use crate::graphs::entity::VertexIndex;
use crate::graphs::AsEntityGraph;
use crate::models::cfg::BranchKind;
use crate::models::{Block, Phi};
use crate::traits::stmt::*;
use crate::types::{EntityRef, Identifiable, IntoEntityRef, Located, SimpleVar};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Debug, Clone)]
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
    where
        G: AsEntityGraph<'a, Block, BranchKind>,
    {
        let g = g.entity_graph();

        let mut vars = BTreeMap::<VertexIndex<_>, _>::default();
        let mut offsets = BTreeMap::default();

        let mut lp_vars = variables!();
        let mut constraints = Vec::default();

        let tracked = SimpleVar::from(tracked);
        let extra_pop = convention.default_prototype().extra_pop() as f64;

        let is_tracked =
            |expr: &Expr| matches!(expr, Expr::Var(v) if SimpleVar::from(v) == tracked);

        let roots = g
            .root_entities()
            .into_iter()
            .map(|(id, _, _)| *id)
            .collect::<BTreeSet<_>>();

        for (i, vx) in g.reverse_post_order().into_iter().enumerate() {
            let b = g.entity(vx);

            let (b_in, b_out) = if let Some(b_inout) = vars.get(&vx) {
                *b_inout
            } else {
                let b_in = lp_vars.add(variable().integer().name(format!("in{}", i)));
                let b_out = lp_vars.add(variable().integer().name(format!("out{}", i)));
                vars.insert(vx, (b_in, b_out));
                (b_in, b_out)
            };

            let in_c = if roots.contains(&b.id()) {
                // in is 0
                constraint!(b_in == 0)
            } else {
                constraint!(b_in >= -0xffff)
            };

            let out_c = if b.last().is_return() {
                // out is extra_pop
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

            let mut is_call = false;

            for op in b.operations() {
                let r = op.into_entity_ref();
                let e = offsets
                    .entry(PhiOrStmt::from(r))
                    .or_insert(Expression::from(b_in + shift as f64));

                match &**op.value() {
                    Stmt::Assign(v, exp) => {
                        if SimpleVar::from(v) == tracked {
                            // look at used in exp, if only in terms of v, then build shift
                            match exp {
                                Expr::Val(ref bv) => {
                                    shift = bv.signed_cast(bv.bits()).to_i64()?;
                                }
                                Expr::BinOp(BinOp::SUB, l, r) if is_tracked(l) => {
                                    if let Some(v) = constant(&**r) {
                                        shift -= v;
                                    }
                                }
                                Expr::BinOp(op, l, r) => {
                                    if let Some(v) = {
                                        if is_tracked(l) {
                                            constant(&**r)
                                        } else if is_tracked(r) {
                                            constant(&**l)
                                        } else {
                                            continue;
                                        }
                                    } {
                                        match op {
                                            BinOp::ADD => {
                                                shift += v;
                                            }
                                            BinOp::AND => {
                                                shift &= v;
                                            }
                                            BinOp::OR => {
                                                shift |= v;
                                            }
                                            _ => (),
                                        }
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                    _ => (),
                }

                *e = if op.is_call() {
                    is_call = true;
                    shift += extra_pop as i64;
                    // we know that this will be the last op of the block
                    // hence we can say that the call is assigned the adjusted
                    // value
                    Expression::from(b_out)
                } else {
                    Expression::from(b_in + shift as f64)
                };
            }

            if is_call {
                log::trace!(
                    "{} - {} >= {}",
                    lp_vars.display(&b_out),
                    lp_vars.display(&b_in),
                    shift
                );
                constraints.push(constraint!(b_out - b_in >= shift as f64));
            } else {
                log::trace!(
                    "{} - {} == {}",
                    lp_vars.display(&b_out),
                    lp_vars.display(&b_in),
                    shift
                );
                constraints.push(constraint!(b_out - b_in == shift as f64));
            }

            // RPO should ensure preds have already been visited
            for (px, _) in g.predecessors(vx) {
                let p_out = if let Some(p_inout) = vars.get(&px) {
                    p_inout.1
                } else {
                    let p_in = lp_vars.add(variable().integer());
                    let p_out = lp_vars.add(variable().integer());
                    vars.insert(px, (p_in, p_out));
                    p_out
                };

                log::trace!(
                    "{} - {} == 0",
                    lp_vars.display(&b_in),
                    lp_vars.display(&p_out)
                );
                constraints.push(constraint!(b_in - p_out == 0));
            }
        }

        let problem = vars.values().fold(0.into(), |v, &(pi, po)| pi + po + v);
        let mut model = lp_vars.minimise(problem).using(default_solver);

        for c in constraints.into_iter() {
            model = model.with(c);
        }

        model.set_parameter("log", "0");

        let solution = model.solve().ok()?;

        Some(StackOffsets(
            offsets
                .into_iter()
                .map(|(k, v)| (k, solution.eval(v) as i64))
                .collect(),
        ))
    }

    pub fn offset_at<T, E>(&self, entity: E) -> Option<i64>
    where
        T: Clone + 'a,
        E: IntoEntityRef<'a, T = T>,
        PhiOrStmt<'a>: From<EntityRef<'a, T>>,
    {
        self.0.get(&entity.into_entity_ref().into()).copied()
    }
}

#[cfg(test)]
mod test {
    use crate::models::{Lifter, Project};
    use crate::traits::oracle::database_oracles;
    use crate::types::{EntityIdMapping, Locatable};
    use fugue::db::Database;
    use fugue::ir::il::traits::*;
    use fugue::ir::LanguageDB;

    use super::*;

    #[test]
    fn test_sample2() -> Result<(), Box<dyn std::error::Error>> {
        env_logger::init();

        let ldb = LanguageDB::from_directory_with("./processors", true)?;
        let db = Database::from_file("./tests/sample2.fdb", &ldb)?;

        let translator = db.default_translator();
        let convention = translator.compiler_conventions()["gcc"].clone();
        let lifter = Lifter::new(translator, convention);

        let mut project = Project::new("sample2", lifter);
        let (bo, fo) = database_oracles(&db);

        project.set_block_oracle(bo);
        project.set_function_oracle(fo);

        for seg in db.segments().values() {
            if seg.is_code() && !seg.is_external() {
                project.add_region_mapping_with(
                    seg.name(),
                    seg.address(),
                    seg.endian(),
                    seg.bytes(),
                );
            }
        }

        let sample2 = db.function("sample2").unwrap();
        let fid = project.add_function(sample2.address()).unwrap();

        let sample2f = project.lookup_by_id(fid).unwrap();
        let cfg = sample2f.cfg_with(&*project, &*project);

        let offs = StackOffsets::analyse_with(
            &cfg,
            &project.lifter().stack_pointer(),
            &project.lifter().convention(),
        );

        assert!(offs.is_some());

        for (k, v) in offs.unwrap().0.iter() {
            match k {
                PhiOrStmt::Phi(k) => println!(
                    "{} {} // {}",
                    k.location(),
                    (****k).display_with(Some(project.lifter().translator())),
                    v
                ),
                PhiOrStmt::Stmt(k) => println!(
                    "{} {} // {}",
                    k.location(),
                    (****k).display_with(Some(project.lifter().translator())),
                    v
                ),
            }
        }

        Ok(())
    }
}
