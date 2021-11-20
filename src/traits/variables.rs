use fugue::ir::il::ecode::Var;

use crate::traits::{ValueRefCollector, ValueMutCollector};

pub trait Variables<'ecode> {
    fn all_variables<C>(&'ecode self) -> C
    where C: ValueRefCollector<'ecode, Var> {
        let mut vars = C::default();
        self.all_variables_with(&mut vars);
        vars
    }

    fn all_variables_mut<C>(&'ecode mut self) -> C
    where C: ValueMutCollector<'ecode, Var> {
        let mut vars = C::default();
        self.all_variables_mut_with(&mut vars);
        vars
    }

    fn defined_variables<C>(&'ecode self) -> C
    where C: ValueRefCollector<'ecode, Var> {
        let mut vars = C::default();
        self.defined_variables_with(&mut vars);
        vars
    }

    fn defined_variables_mut<C>(&'ecode mut self) -> C
    where C: ValueMutCollector<'ecode, Var> {
        let mut vars = C::default();
        self.defined_variables_mut_with(&mut vars);
        vars
    }

    fn used_variables<C>(&'ecode self) -> C
    where C: ValueRefCollector<'ecode, Var> {
        let mut vars = C::default();
        self.used_variables_with(&mut vars);
        vars
    }

    fn used_variables_mut<C>(&'ecode mut self) -> C
    where C: ValueMutCollector<'ecode, Var> {
        let mut vars = C::default();
        self.used_variables_mut_with(&mut vars);
        vars
    }

    fn defined_and_used_variables<C>(&'ecode self) -> (C, C)
    where C: ValueRefCollector<'ecode, Var> {
        (self.defined_variables(), self.used_variables())
    }

    fn all_variables_with<C>(&'ecode self, vars: &mut C)
        where C: ValueRefCollector<'ecode, Var>;

    fn all_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
        where C: ValueMutCollector<'ecode, Var>;

    fn defined_variables_with<C>(&'ecode self, vars: &mut C)
        where C: ValueRefCollector<'ecode, Var>;

    fn used_variables_with<C>(&'ecode self, vars: &mut C)
        where C: ValueRefCollector<'ecode, Var>;

    fn defined_and_used_variables_with<C>(&'ecode self, defs: &mut C, uses: &mut C)
    where C: ValueRefCollector<'ecode, Var> {
        self.defined_variables_with(defs);
        self.used_variables_with(uses);
    }

    fn defined_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
        where C: ValueMutCollector<'ecode, Var>;

    fn used_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
        where C: ValueMutCollector<'ecode, Var>;

    fn defined_and_used_variables_mut_with<C>(&'ecode mut self, defs: &mut C, uses: &mut C)
        where C: ValueMutCollector<'ecode, Var>;
}
