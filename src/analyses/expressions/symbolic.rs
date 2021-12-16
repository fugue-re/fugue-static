use fugue::bv::BitVec;
use fugue::ir::Translator;
use fugue::ir::il::Location;
use fugue::ir::il::ecode::{Expr, Stmt, Var};

use std::collections::BTreeMap;

use crate::models::block::Block;
use crate::models::cfg::CFG;
use crate::traits::*;
use crate::transforms::egraph::Rewriter;

pub struct SymProp<'a> {
    translator: &'a Translator,
    mapping: BTreeMap<Var, Expr>,
}

impl<'a, 'ecode> VisitMut<'ecode, Location, BitVec, Var> for SymProp<'a> {
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
            Expr::Load(ref mut expr, size, space) => {
                self.visit_expr_load_mut(expr, *size, *space)
            }
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
            Expr::Var(var) => if let Some(nexpr) = self.mapping.get(var) {
                *expr = nexpr.clone();
            }
        }
        *expr = Rewriter::default().simplify_expr(self.translator, expr);
    }

    fn visit_stmt_mut(&mut self, stmt: &'ecode mut Stmt) {
        match stmt {
            Stmt::Assign(var, ref mut expr) => {
                self.visit_expr_mut(expr);
                self.mapping.insert(*var, expr.clone());
            },
            Stmt::Call(ref mut bt, ref mut args) => self.visit_stmt_call_mut(bt, args),
            Stmt::Branch(ref mut bt) => self.visit_stmt_branch_mut(bt),
            Stmt::CBranch(ref mut c, ref mut bt) => self.visit_stmt_cbranch_mut(c, bt),
            Stmt::Intrinsic(name, ref mut args) => self.visit_stmt_intrinsic_mut(&*name, args),
            Stmt::Store(ref mut t, ref mut s, sz, spc) => self.visit_stmt_store_mut(t, s, *sz, *spc),
            Stmt::Return(ref mut bt) => self.visit_stmt_return_mut(bt),
            Stmt::Skip => (),
        }
    }
}

impl<'a> SymProp<'a> {
    pub fn new(translator: &'a Translator) -> Self {
        Self {
            translator,
            mapping: Default::default(),
        }
    }

    pub fn propagate_expressions<T>(&mut self, t: &mut T)
    where T: SymPropFold {
        t.propagate_expressions(self);
    }
}

pub trait SymPropFold {
    fn propagate_expressions(&mut self, p: &mut SymProp);
}

impl<'a> SymPropFold for CFG<'a, Block> {
    fn propagate_expressions(&mut self, p: &mut SymProp) {
        // TODO: this should run to a fixed point!!
        let rpo = self.reverse_post_order();
        for _ in 0..3 {
            for vx in rpo.iter() {
                let blk = self.entity_mut(*vx);
                blk.to_mut().propagate_expressions(p);
            }
        }
    }
}

impl SymPropFold for Block {
    fn propagate_expressions(&mut self, p: &mut SymProp) {
        for phi in self.phis() {
            // if all incoming vars are the same, then we can propagate
            let vars = phi.assign();
            let evar = p.mapping.get(&vars[0]);
            if let Some(expr) = evar {
                if vars[1..].iter().map(|v| p.mapping.get(v)).all(|e| e == evar) {
                    let expr = expr.to_owned();
                    p.mapping.insert(*phi.var(), expr.into());
                }
            }
        }

        for op in self.operations_mut() {
            op.propagate_expressions(p);
        }
    }
}

impl SymPropFold for Stmt {
    fn propagate_expressions(&mut self, p: &mut SymProp) {
        p.visit_stmt_mut(self);
    }
}

impl SymPropFold for Expr {
    fn propagate_expressions(&mut self, p: &mut SymProp) {
        p.visit_expr_mut(self);
    }
}
