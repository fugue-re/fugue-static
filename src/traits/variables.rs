use fugue::ir::il::ecode::{Expr, Stmt, Var};
use std::collections::HashSet;

use crate::traits::Visit;

pub trait Variables {
    fn defined_variables(&self) -> HashSet<&Var> {
        let mut vars = HashSet::default();
        self.defined_variables_with(&mut vars);
        vars
    }

    fn used_variables(&self) -> HashSet<&Var> {
        let mut vars = HashSet::default();
        self.used_variables_with(&mut vars);
        vars
    }

    fn defined_and_used_variables(&self) -> (HashSet<&Var>, HashSet<&Var>) {
        (self.defined_variables(), self.used_variables())
    }

    fn defined_variables_with<'ecode>(&'ecode self, vars: &mut HashSet<&'ecode Var>);
    fn used_variables_with<'ecode>(&'ecode self, vars: &mut HashSet<&'ecode Var>);

    fn defined_and_used_variables_with<'ecode>(&'ecode self, defs: &mut HashSet<&'ecode Var>, uses: &mut HashSet<&'ecode Var>) {
        self.defined_variables_with(defs);
        self.used_variables_with(uses);
    }
}

impl Variables for Stmt {
    fn defined_variables_with<'ecode>(&'ecode self, vars: &mut HashSet<&'ecode Var>) {
        struct VisitDefs<'a, 'ecode>(&'a mut HashSet<&'ecode Var>);

        impl<'a, 'ecode> Visit<'ecode> for VisitDefs<'a, 'ecode> {
            fn visit_stmt_assign(&mut self, var: &'ecode Var, _expr: &'ecode Expr) {
                self.0.insert(var);
            }
        }

        let mut visitor = VisitDefs(vars);
        visitor.visit_stmt(self);
    }

    fn used_variables_with<'ecode>(&'ecode self, vars: &mut HashSet<&'ecode Var>) {
        struct VisitUses<'a, 'ecode>(&'a mut HashSet<&'ecode Var>);

        impl<'a, 'ecode> Visit<'ecode> for VisitUses<'a, 'ecode> {
            fn visit_expr_var(&mut self, var: &'ecode Var) {
                self.0.insert(var);
            }
        }

        let mut visitor = VisitUses(vars);
        visitor.visit_stmt(self);
    }
}
