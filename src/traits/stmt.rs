use std::marker::PhantomData;
use fugue::ir::il::ecode::{Expr, Stmt, Var};
use crate::traits::{Variables, ValueRefCollector, ValueMutCollector, Visit, VisitMut};

pub trait StmtExt {
    fn is_branch(&self) -> bool;
    fn is_jump(&self) -> bool;
    fn is_cond(&self) -> bool;
    fn is_call(&self) -> bool;
    fn is_intrinsic(&self) -> bool;
    fn is_return(&self) -> bool;
}

impl StmtExt for Stmt {
    fn is_branch(&self) -> bool {
        matches!(self,
                 Stmt::Branch(_) |
                 Stmt::CBranch(_, _) |
                 Stmt::Call(_, _) |
                 Stmt::Intrinsic(_, _) |
                 Stmt::Return(_))
    }

    fn is_jump(&self) -> bool {
        matches!(self, Stmt::Branch(_) | Stmt::CBranch(_, _))
    }

    fn is_cond(&self) -> bool {
        matches!(self, Stmt::CBranch(_, _))
    }

    fn is_call(&self) -> bool {
        matches!(self, Stmt::Call(_, _))
    }

    fn is_intrinsic(&self) -> bool {
        matches!(self, Stmt::Intrinsic(_, _))
    }

    fn is_return(&self) -> bool {
        matches!(self, Stmt::Return(_))
    }
}

struct VisitVars<'a, 'ecode, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, C> Visit<'ecode> for VisitVars<'a, 'ecode, C>
where C: ValueRefCollector<'ecode, Var> {
    fn visit_var(&mut self, var: &'ecode Var) {
        self.0.insert_ref(var);
    }
}

struct VisitVarsMut<'a, 'ecode, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, C> VisitMut<'ecode> for VisitVarsMut<'a, 'ecode, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {
        self.0.insert_mut(var);
    }
}

struct VisitDefs<'a, 'ecode, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, C> Visit<'ecode> for VisitDefs<'a, 'ecode, C>
where C: ValueRefCollector<'ecode, Var> {
    fn visit_stmt_assign(&mut self, var: &'ecode Var, _expr: &'ecode Expr) {
        self.0.insert_ref(var);
    }
}

struct VisitDefsMut<'a, 'ecode, C>(&'a mut C, PhantomData<&'ecode mut Var>);
impl<'a, 'ecode, C> VisitMut<'ecode> for VisitDefsMut<'a, 'ecode, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, _expr: &'ecode mut Expr) {
        self.0.insert_mut(var);
    }
}

struct VisitUses<'a, 'ecode, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, C> Visit<'ecode> for VisitUses<'a, 'ecode, C>
where C: ValueRefCollector<'ecode, Var> {
    fn visit_expr_var(&mut self, var: &'ecode Var) {
        self.0.insert_ref(var);
    }
}

struct VisitUsesMut<'a, 'ecode, C>(&'a mut C, PhantomData<&'ecode mut Var>);
impl<'a, 'ecode, C> VisitMut<'ecode> for VisitUsesMut<'a, 'ecode, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        self.0.insert_mut(var);
    }
}

struct VisitDefUsesMut<'a, 'ecode, C>(&'a mut C, &'a mut C, PhantomData<&'ecode mut Var>);
impl<'a, 'ecode, C> VisitMut<'ecode> for VisitDefUsesMut<'a, 'ecode, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, expr: &'ecode mut Expr) {
        // defs
        self.0.insert_mut(var);
        self.visit_expr_mut(expr);
    }

    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        // uses
        self.1.insert_mut(var);
    }
}

impl<'ecode> Variables<'ecode> for Stmt {
    fn all_variables_with<C>(&'ecode self, vars: &mut C)
    where C: ValueRefCollector<'ecode, Var> {
        VisitVars(vars, PhantomData).visit_stmt(self)
    }

    fn all_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where C: ValueMutCollector<'ecode, Var> {
        VisitVarsMut(vars, PhantomData).visit_stmt_mut(self)
    }

    fn defined_variables_with<C>(&'ecode self, vars: &mut C)
    where C: ValueRefCollector<'ecode, Var> {
        VisitDefs(vars, PhantomData).visit_stmt(self)
    }

    fn defined_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where C: ValueMutCollector<'ecode, Var> {
        VisitDefsMut(vars, PhantomData).visit_stmt_mut(self)
    }

    fn used_variables_with<C>(&'ecode self, vars: &mut C)
    where C: ValueRefCollector<'ecode, Var> {
        VisitUses(vars, PhantomData).visit_stmt(self)
    }

    fn used_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where C: ValueMutCollector<'ecode, Var> {
        VisitUsesMut(vars, PhantomData).visit_stmt_mut(self)
    }

    fn defined_and_used_variables_mut_with<C>(&'ecode mut self, defs: &mut C, uses: &mut C)
    where C: ValueMutCollector<'ecode, Var> {
        VisitDefUsesMut(defs, uses, PhantomData).visit_stmt_mut(self)
    }
}
