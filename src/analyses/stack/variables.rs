use fugue::bv::BitVec;
use fugue::ir::il::Location;
use fugue::ir::il::ecode::{BinOp, Expr, Stmt, Var};

use crate::analyses::expressions::symbolic::{SymPropFold, SymExprs, SymExprsProp};
use crate::types::SimpleVar;
use crate::models::{Block, CFG, Project};
use crate::traits::{VisitMut, Substitutor};
use crate::transforms::SSA;
use fugue::ir::AddressSpaceId;


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


pub struct StackRename<'a> {
    tracked: SimpleVar<'static>,
    stack_space: AddressSpaceId,
    sym_exprs: SymExprs<'a>,
}

impl<'a> StackRename<'a> {
    // NOTE: expects stack offset corrections have been applied
    // and that g is in SSA form
    pub fn new<'g>(g: &CFG<'g, Block>, p: &'a Project) -> Self {
        let mut prop = SymExprs::new(p.lifter().translator());

        g.propagate_expressions(&mut prop);

        Self {
            tracked: p.lifter().stack_pointer().into(),
            stack_space: p.lifter().stack_space_id(),
            sym_exprs: prop,
        }
    }

    // NOTE: ensures that cfg is in SSA form
    pub fn apply<'g>(&self, cfg: &mut CFG<Block>) {
        let subst = Substitutor::new(self.sym_exprs.propagator());

        let mut rs = RenameStack {
            tracked: self.tracked.clone(),
            stack_space: self.stack_space,
            subst,
        };

        for bx in cfg.reverse_post_order().into_iter() {
            let blk = cfg.entity_mut(bx).to_mut();
            for op in blk.operations_mut() {
                rs.visit_stmt_mut(op);
            }
        }

        cfg.ssa();
    }
}
