use std::marker::PhantomData;
use fugue::ir::il::ecode::{ExprT, StmtT};
use crate::traits::{Variables, ValueRefCollector, ValueMutCollector, Visit, VisitMut};

pub trait StmtExt<Loc, Val, Var> {
    fn is_skip(&self) -> bool;
    fn is_branch(&self) -> bool;
    fn is_jump(&self) -> bool;
    fn is_cond(&self) -> bool;
    fn is_call(&self) -> bool;
    fn is_intrinsic(&self) -> bool;
    fn is_return(&self) -> bool;
    fn has_fall(&self) -> bool;

    fn for_each_variable_def<F: FnMut(&Var)>(&self, f: F);
    fn for_each_variable_def_mut<F: FnMut(&mut Var)>(&mut self, f: F);

    fn for_each_variable_use<F: FnMut(&Var)>(&self, f: F);
    fn for_each_variable_use_mut<F: FnMut(&mut Var)>(&mut self, f: F);
}

impl<Loc, Val, Var> StmtExt<Loc, Val, Var> for StmtT<Loc, Val, Var> {
    fn is_branch(&self) -> bool {
        matches!(self,
                 StmtT::Branch(_) |
                 StmtT::CBranch(_, _) |
                 StmtT::Call(_, _) |
                 StmtT::Intrinsic(_, _) |
                 StmtT::Return(_))
    }

    fn is_jump(&self) -> bool {
        matches!(self, StmtT::Branch(_) | StmtT::CBranch(_, _))
    }

    fn is_cond(&self) -> bool {
        matches!(self, StmtT::CBranch(_, _))
    }

    fn is_call(&self) -> bool {
        matches!(self, StmtT::Call(_, _))
    }

    fn is_intrinsic(&self) -> bool {
        matches!(self, StmtT::Intrinsic(_, _))
    }

    fn has_fall(&self) -> bool {
        !matches!(self, StmtT::Branch(_) | StmtT::Return(_))
    }

    fn is_return(&self) -> bool {
        matches!(self, StmtT::Return(_))
    }

    fn is_skip(&self) -> bool {
        matches!(self, StmtT::Skip)
    }

    fn for_each_variable_def<F: FnMut(&Var)>(&self, mut f: F) {
        if let StmtT::Assign(ref v, _) = self {
            f(v)
        }
    }

    fn for_each_variable_def_mut<F: FnMut(&mut Var)>(&mut self, mut f: F) {
        if let StmtT::Assign(ref mut v, _) = self {
            f(v)
        }
    }

    fn for_each_variable_use<F: FnMut(&Var)>(&self, mut f: F) {
        ForEachUse(&mut f, PhantomData).visit_stmt(self);
    }

    fn for_each_variable_use_mut<F: FnMut(&mut Var)>(&mut self, mut f: F) {
        ForEachUseMut(&mut f, PhantomData).visit_stmt_mut(self);
    }
}

struct ForEachUse<'a, 'ecode, Var, F>(&'a mut F, PhantomData<&'ecode Var>)
    where F: FnMut(&'ecode Var);

impl<'a, 'ecode, Loc, Val, Var, F> Visit<'ecode, Loc, Val, Var> for ForEachUse<'a, 'ecode, Var, F>
where F: FnMut(&'ecode Var) {
    fn visit_expr_var(&mut self, var: &'ecode Var) {
        self.0(var)
    }
}

struct ForEachUseMut<'a, 'ecode, Var, F>(&'a mut F, PhantomData<&'ecode mut Var>)
    where F: FnMut(&'ecode mut Var);

impl<'a, 'ecode, Loc, Val, Var, F> VisitMut<'ecode, Loc, Val, Var> for ForEachUseMut<'a, 'ecode, Var, F>
where F: FnMut(&'ecode mut Var) {
    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        self.0(var)
    }
}

struct VisitVars<'a, 'ecode, Var, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, Loc, Val, Var, C> Visit<'ecode, Loc, Val, Var> for VisitVars<'a, 'ecode, Var, C>
where C: ValueRefCollector<'ecode, Var> {
    fn visit_var(&mut self, var: &'ecode Var) {
        self.0.insert_ref(var);
    }
}

struct VisitVarsMut<'a, 'ecode, Var, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, Loc, Val, Var, C> VisitMut<'ecode, Loc, Val, Var> for VisitVarsMut<'a, 'ecode, Var, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {
        self.0.insert_mut(var);
    }
}

struct VisitDefs<'a, 'ecode, Var, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, Loc, Val, Var, C> Visit<'ecode, Loc, Val, Var> for VisitDefs<'a, 'ecode, Var, C>
where C: ValueRefCollector<'ecode, Var> {
    fn visit_stmt_assign(&mut self, var: &'ecode Var, _expr: &'ecode ExprT<Loc, Val, Var>) {
        self.0.insert_ref(var);
    }
}

struct VisitDefsMut<'a, 'ecode, Var, C>(&'a mut C, PhantomData<&'ecode mut Var>);
impl<'a, 'ecode, Loc, Val, Var, C> VisitMut<'ecode, Loc, Val, Var> for VisitDefsMut<'a, 'ecode, Var, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, _expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.0.insert_mut(var);
    }
}

struct VisitUses<'a, 'ecode, Var, C>(&'a mut C, PhantomData<&'ecode Var>);
impl<'a, 'ecode, Loc, Val, Var, C> Visit<'ecode, Loc, Val, Var> for VisitUses<'a, 'ecode, Var, C>
where C: ValueRefCollector<'ecode, Var> {
    fn visit_expr_var(&mut self, var: &'ecode Var) {
        self.0.insert_ref(var);
    }
}

struct VisitUsesMut<'a, 'ecode, Var, C>(&'a mut C, PhantomData<&'ecode mut Var>);
impl<'a, 'ecode, Loc, Val, Var, C> VisitMut<'ecode, Loc, Val, Var> for VisitUsesMut<'a, 'ecode, Var, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        self.0.insert_mut(var);
    }
}

struct VisitDefUsesMut<'a, 'ecode, Var, C>(&'a mut C, &'a mut C, PhantomData<&'ecode mut Var>);
impl<'a, 'ecode, Loc, Val, Var, C> VisitMut<'ecode, Loc, Val, Var> for VisitDefUsesMut<'a, 'ecode, Var, C>
where C: ValueMutCollector<'ecode, Var> {
    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        // defs
        self.0.insert_mut(var);
        self.visit_expr_mut(expr);
    }

    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        // uses
        self.1.insert_mut(var);
    }
}

impl<'ecode, Loc, Val, Var> Variables<'ecode, Var> for StmtT<Loc, Val, Var> {
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
