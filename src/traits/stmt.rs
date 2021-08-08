use fugue::ir::il::ecode::{Expr, Stmt, Var};

use std::collections::HashSet;

use crate::traits::Visit;

pub trait StmtExt {
    fn is_branch(&self) -> bool;
    fn is_jump(&self) -> bool;
    fn is_cond(&self) -> bool;
    fn is_call(&self) -> bool;
    fn is_return(&self) -> bool;

    fn defined_variables(&self) -> HashSet<&Var>;
    fn used_variables(&self) -> HashSet<&Var>;
}

impl StmtExt for Stmt {
    fn is_branch(&self) -> bool {
        matches!(self,
                 Stmt::Branch(_) |
                 Stmt::CBranch(_, _) |
                 Stmt::Call(_) |
                 Stmt::Return(_))
    }

    fn is_jump(&self) -> bool {
        matches!(self, Stmt::Branch(_) | Stmt::CBranch(_, _))
    }

    fn is_cond(&self) -> bool {
        matches!(self, Stmt::CBranch(_, _))
    }

    fn is_call(&self) -> bool {
        matches!(self, Stmt::Call(_))
    }

    fn is_return(&self) -> bool {
        matches!(self, Stmt::Return(_))
    }

    fn defined_variables(&self) -> HashSet<&Var> {
        struct VisitDefs<'ecode>(HashSet<&'ecode Var>);

        impl<'ecode> Visit<'ecode> for VisitDefs<'ecode> {
            fn visit_stmt_assign(&mut self, var: &'ecode Var, _expr: &'ecode Expr) {
                self.0.insert(var);
            }
        }

        let mut visitor = VisitDefs(HashSet::default());

        visitor.visit_stmt(self);

        visitor.0
    }

    fn used_variables(&self) -> HashSet<&Var> {
        struct VisitUses<'ecode>(HashSet<&'ecode Var>);

        impl<'ecode> Visit<'ecode> for VisitUses<'ecode> {
            fn visit_expr_var(&mut self, var: &'ecode Var) {
                self.0.insert(var);
            }
        }

        let mut visitor = VisitUses(HashSet::default());

        visitor.visit_stmt(self);

        visitor.0
    }
}
