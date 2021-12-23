use std::collections::{BTreeMap, BTreeSet};

use fugue::bv::BitVec;
use fugue::ir::convention::Convention;
use fugue::ir::il::ecode::{BinOp, Expr, Stmt, Var};
use fugue::ir::il::traits::*;
use good_lp::*;

use crate::graphs::entity::VertexIndex;
use crate::graphs::{AsEntityGraph, AsEntityGraphMut};
use crate::models::cfg::BranchKind;
use crate::models::Block;
use crate::traits::stmt::*;
use crate::types::{Entity, Id, Identifiable, Locatable, Located, SimpleVar};

/*
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
*/

#[derive(Debug, Clone)]
pub struct StackOffsets {
    tracked: Var,
    statements: BTreeMap<Id<Located<Stmt>>, (i64, i64)>,
    adjustments: BTreeMap<Id<Block>, i64>,
}

fn constant(expr: &Expr) -> Option<i64> {
    if let Expr::Val(v) = expr {
        v.signed_cast(v.bits()).to_i64()
    } else {
        None
    }
}

impl StackOffsets {
    fn analyse_aux<'a, G>(g: &G, tracked: &Var, convention: &Convention) -> Option<Self>
    where
        G: AsEntityGraph<'a, Block, BranchKind>,
    {
        let g = g.entity_graph();

        let mut vars = BTreeMap::<VertexIndex<_>, _>::default();
        let mut stmts = BTreeMap::default();
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
            let mut is_call = false;

            for op in b.operations() {
                let mut sft = stmts.entry(op.id()).or_insert((
                    Expression::from(b_in + shift as f64),
                    Expression::from(b_in + shift as f64),
                ));

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

                if op.is_call() {
                    // we know that this will be the last op of the block
                    // hence we can say that the call is assigned the adjusted
                    // value
                    is_call = true;
                    sft.1 = Expression::from(b_out);
                    offsets.insert(b.id(), Expression::from(b_out - (b_in + shift as f64)));
                    shift += extra_pop as i64;
                } else {
                    sft.1 = Expression::from(b_in + shift as f64);
                }
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

        Some(StackOffsets {
            tracked: **tracked,
            statements: stmts
                .into_iter()
                .map(|(id, (e_in, e_out))| {
                    let s_in = solution.eval(e_in) as i64;
                    let s_out = solution.eval(e_out) as i64;
                    (id, (s_in, s_out))
                })
                .collect(),
            adjustments: offsets
                .into_iter()
                .map(|(k, v)| (k, solution.eval(v) as i64))
                .collect(),
        })
    }

    pub fn analyse<'a, G>(g: &G, tracked: &Var, convention: &Convention) -> Self
    where
        G: AsEntityGraph<'a, Block, BranchKind>,
    {
        Self::analyse_aux(g, tracked, convention).unwrap_or_else(|| Self {
            tracked: *tracked,
            statements: Default::default(),
            adjustments: Default::default(),
        })
    }

    pub fn apply<'a, G>(&self, g: &mut G)
    where
        G: AsEntityGraphMut<'a, Block, BranchKind>,
    {
        let g = g.entity_graph_mut();
        for (bid, adjustment) in self.adjustments.iter() {
            let bx = g.entity_vertex(*bid).unwrap();
            let blk = g.entity_mut(bx).to_mut();
            let loc = blk.last().location();
            let mut nblk = Block::empty(loc.clone());
            nblk.operations_mut().push(Entity::new(
                "stmt",
                Located::new(
                    loc,
                    Stmt::Assign(
                        self.tracked,
                        Expr::int_add(
                            self.tracked,
                            BitVec::from_i64(*adjustment, self.tracked.bits()),
                        ),
                    ),
                ),
            ));

            let mut loc_tgts = vec![nblk.id().into()];
            std::mem::swap(blk.next_blocks_mut(), &mut loc_tgts);
            *nblk.next_blocks_mut() = loc_tgts;

            let sx = g.add_entity(nblk);
            let succs = g.successors(bx).into_iter().collect::<Vec<_>>();
            g.add_vertex_relation(bx, sx, BranchKind::Fall);

            for (succ, ex) in succs {
                let ev = g.remove_edge(ex);
                g.add_vertex_relation(sx, succ, ev.unwrap());
            }
        }
    }

    pub fn offsets_for(&self, stmt: Id<Located<Stmt>>) -> (i64, i64) {
        self.statements.get(&stmt).copied().unwrap_or((0, 0))
    }

    pub fn adjustment_for(&self, blk: Id<Block>) -> i64 {
        self.adjustments.get(&blk).copied().unwrap_or(0)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&Id<Block>, i64)> {
        self.adjustments.iter().map(|(k, v)| (k, *v))
    }
}

#[cfg(test)]
mod test {
    use crate::analyses::expressions::symbolic::{SymExprs, SymExprsProp, SymPropFold};
    use crate::analyses::reaching_definitions::ReachingDefinitions;
    use crate::analyses::stack::variables::StackRename;
    use crate::models::{Lifter, Project};
    use crate::traits::oracle::database_oracles;
    use crate::traits::{Substitutor, VisitMut};
    use crate::types::EntityIdMapping;
    use fugue::db::Database;
    use fugue::ir::il::traits::*;
    use fugue::ir::il::Location;
    use fugue::ir::{AddressSpaceId, LanguageDB};
    use itertools::Itertools;

    use crate::transforms::SSA;
    use crate::visualise::AsDot;

    use super::*;

    #[test]
    fn test_sample2() -> Result<(), Box<dyn std::error::Error>> {
        env_logger::try_init().ok();

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

        let sample2 = db.function("__libc_csu_init").unwrap();
        let fid = project.add_function(sample2.address()).unwrap();

        let sample2f = project.lookup_by_id(fid).unwrap();
        let mut cfg = sample2f.cfg_with(&*project, &*project);

        let offs = StackOffsets::analyse(
            &cfg,
            &project.lifter().stack_pointer(),
            &project.lifter().convention(),
        );

        /*
        let mut prop = SymExprs::new(project.lifter().translator());

        offs.apply(&mut cfg);
        cfg.ssa();
        cfg.propagate_expressions(&mut prop);

        // only perform subst if forall v in fv(subst). v in reaching-defs(subst)

        let subst = Substitutor::new(prop.propagator());
        //let mut fsubst = SubstLoadStore::new(&mut subst);
        //fsubst.apply_graph(&mut cfg);

        struct RenameStack<'a> {
            tracked: SimpleVar<'static>,
            stack_space: AddressSpaceId,
            subst: Substitutor<Location, BitVec, Var, SymExprsProp<'a>>,
        }

        impl<'a, 'ecode> VisitMut<'ecode, Location, BitVec, Var> for RenameStack<'a> {
            fn visit_expr_mut(&mut self, expr: &'ecode mut Expr) {
                match expr {
                    Expr::UnRel(op, ref mut expr) => self.visit_expr_unrel_mut(*op, expr),
                    Expr::UnOp(op, ref mut expr) => self.visit_expr_unop_mut(*op, expr),
                    Expr::BinRel(op, ref mut lexpr, ref mut rexpr) => {
                        self.visit_expr_binrel_mut(*op, lexpr, rexpr)
                    }
                    Expr::BinOp(op, ref mut lexpr, ref mut rexpr) => {
                        self.visit_expr_binop_mut(*op, lexpr, rexpr)
                    }
                    Expr::Cast(ref mut expr, ref mut cast) => self.visit_expr_cast_mut(expr, cast),
                    Expr::Load(ref mut lexpr, size, _) => {
                        let mut nlexpr = lexpr.clone();
                        self.subst.apply_expr(&mut nlexpr);

                        match &mut *nlexpr {
                            Expr::BinOp(op, ref mut lexpr, ref mut rexpr) => if *op == BinOp::ADD {
                                match (&mut **lexpr, &mut **rexpr) {
                                    (Expr::Var(ref sp), Expr::Val(ref sft)) |
                                        (Expr::Val(ref sft), Expr::Var(ref sp)) if SimpleVar::from(sp) == self.tracked => {
                                            let val = sft.to_i64().unwrap();
                                            let var = Var::new(self.stack_space, val as u64, *size, 0);
                                            *expr = Expr::from(var);
                                        },
                                        _ => self.visit_expr_mut(lexpr),
                                }
                            } else {
                                self.visit_expr_mut(lexpr)
                            },
                            _ => self.visit_expr_mut(lexpr),
                        }
                    },
                    Expr::ExtractHigh(ref mut expr, bits) => self.visit_expr_extract_high_mut(expr, *bits),
                    Expr::ExtractLow(ref mut expr, bits) => self.visit_expr_extract_low_mut(expr, *bits),
                    Expr::Extract(ref mut expr, lsb, msb) => self.visit_expr_extract_mut(expr, *lsb, *msb),
                    Expr::Concat(ref mut lexpr, ref mut rexpr) => self.visit_expr_concat_mut(lexpr, rexpr),
                    Expr::IfElse(ref mut cond, ref mut texpr, ref mut fexpr) => self.visit_expr_ite_mut(cond, texpr, fexpr),
                    Expr::Call(ref mut branch_target, ref mut args, bits) => {
                        self.visit_expr_call_mut(branch_target, args, *bits)
                    }
                    Expr::Intrinsic(ref name, ref mut args, bits) => {
                        self.visit_expr_intrinsic_mut(name, args, *bits)
                    }
                    Expr::Val(_) => (),
                    Expr::Var(_) => (),
                }
            }

            fn visit_stmt_mut(&mut self, stmt: &'ecode mut Stmt) {
                match stmt {
                    Stmt::Assign(_, ref mut expr) => {
                        self.visit_expr_mut(expr)
                    },
                    Stmt::Call(ref mut bt, ref mut args) => self.visit_stmt_call_mut(bt, args),
                    Stmt::Branch(ref mut bt) => self.visit_stmt_branch_mut(bt),
                    Stmt::CBranch(ref mut c, ref mut bt) => self.visit_stmt_cbranch_mut(c, bt),
                    Stmt::Intrinsic(name, ref mut args) => self.visit_stmt_intrinsic_mut(&*name, args),
                    Stmt::Return(ref mut bt) => self.visit_stmt_return_mut(bt),
                    Stmt::Skip => (),
                    Stmt::Store(ref mut lexpr, ref mut roexpr, size, _) => {
                        self.visit_expr_mut(roexpr);

                        let mut nlexpr = lexpr.clone();
                        self.subst.apply_expr(&mut nlexpr);

                        match &mut nlexpr {
                            Expr::BinOp(op, ref mut lexpr, ref mut rexpr) => if *op == BinOp::ADD {
                                match (&mut **lexpr, &mut **rexpr) {
                                    (Expr::Var(ref sp), Expr::Val(ref sft)) |
                                    (Expr::Val(ref sft), Expr::Var(ref sp)) if SimpleVar::from(sp) == self.tracked => {
                                        let val = sft.to_i64().unwrap();
                                        let var = Var::new(self.stack_space, val as u64, *size, 0);
                                        *stmt = Stmt::Assign(var, roexpr.clone());
                                    },
                                    _ => self.visit_expr_mut(lexpr),
                                }
                            } else {
                                self.visit_expr_mut(lexpr)
                            },
                            _ => self.visit_expr_mut(lexpr)
                        }
                    }
                }
            }
        }

        let mut rs = RenameStack {
            tracked: project.stack_pointer().into(),
            stack_space: project.lifter().stack_space_id(),
            subst,
        };

        for bx in cfg.reverse_post_order().into_iter() {
            let blk = cfg.entity_mut(bx).to_mut();
            for op in blk.operations_mut() {
                rs.visit_stmt_mut(op);
            }
        }

        cfg.ssa();

        println!(
            "{}",
            cfg.dot_with(
                |_, e| {
                    format!(
                        "{}",
                        e.display_with(project.lifter().translator().into()),
                    )
                },
                |_| "".to_string()
            )
        );
        */
        cfg.ssa();

        let rs = StackRename::new(&offs, &project);
        rs.apply(&mut cfg);

        let rd = ReachingDefinitions::new(&cfg);

        for vx in cfg.reverse_post_order() {
            let blk = cfg.entity(vx);
            for op in blk.operations() {
                let (sft_in, sft_out) = offs.statements[&op.id()];
                print!(
                    "{} {} ({}, {})",
                    op.location(),
                    op.display_with(Some(project.lifter().translator())),
                    sft_in,
                    sft_out,
                );

                let reaching = rd.all_reaching_vars(op);
                if let Some(rd) = reaching {
                    println!(
                        " [{}]",
                        rd.filter(|(v, _)| v.space().is_register())
                            .map(|(v, _)| v.display_with(Some(project.lifter().translator())))
                            .format(", ")
                    );
                } else {
                    println!(" []");
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_lojax() -> Result<(), Box<dyn std::error::Error>> {
        env_logger::try_init().ok();

        let ldb = LanguageDB::from_directory_with("./processors", true)?;
        let db = Database::from_file("./tests/lojax.fdb", &ldb)?;

        let translator = db.default_translator();
        let convention = translator.compiler_conventions()["windows"].clone();
        let lifter = Lifter::new(translator, convention);

        let mut project = Project::new("sample2", lifter);
        let (bo, fo) = database_oracles(&db);

        project.set_block_oracle(bo);
        project.set_function_oracle(fo);

        for seg in db.segments().values() {
            project.add_region_mapping_with(seg.name(), seg.address(), seg.endian(), seg.bytes());
        }

        let sample2 = db.function("sub_DF4").unwrap();
        let fid = project.add_function(sample2.address()).unwrap();

        let sample2f = project.lookup_by_id(fid).unwrap();
        let mut cfg = sample2f.cfg_with(&*project, &*project);

        let offs = StackOffsets::analyse(
            &cfg,
            &project.lifter().stack_pointer(),
            &project.lifter().convention(),
        );

        cfg.ssa();

        let rs = StackRename::new(&offs, &project);
        rs.apply(&mut cfg);

        let rd = ReachingDefinitions::new(&cfg);

        for vx in cfg.reverse_post_order() {
            let blk = cfg.entity(vx);
            for op in blk.operations() {
                let (sft_in, sft_out) = offs.statements[&op.id()];
                print!(
                    "{} {} ({}, {})",
                    op.location(),
                    op.display_with(Some(project.lifter().translator())),
                    sft_in,
                    sft_out,
                );

                let reaching = rd.all_reaching_vars(op);
                if let Some(rd) = reaching {
                    println!(
                        " [{}]",
                        rd.filter(|(v, _)| v.space().is_register())
                            .map(|(v, _)| v.display_with(Some(project.lifter().translator())))
                            .format(", ")
                    );
                } else {
                    println!(" []");
                }
            }
        }

        Ok(())

        /*
        let mut prop = SymExprs::new(project.lifter().translator());

        offs.apply(&mut cfg);

        cfg.ssa();
        cfg.propagate_expressions(&mut prop);

        let mut subst = Substitutor::new(prop.propagator());
        subst.apply_graph(&mut cfg);

        cfg.ssa();

        println!(
            "{}",
            cfg.dot_with(
                |_, e| {
                    format!(
                        "{}",
                        e.display_with(project.lifter().translator().into()),
                    )
                },
                |_| "".to_string()
            )
        );

        Ok(())
        */
    }
}
