use fugue::ir::il::ecode::{Expr, Stmt, Var};

use crate::traits::VisitMut;
use crate::traits::{ValueRefCollector, ValueMutCollector};
use crate::types::SimpleVar;

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

pub trait Substitution {
    #[allow(unused_variables)]
    fn rename(&mut self, var: &Var) -> Option<Var> {
        None
    }

    #[allow(unused_variables)]
    fn replace(&mut self, var: &Var) -> Option<Expr> {
        None
    }
}

impl<'ecode, T> VisitMut<'ecode> for T where T: Substitution {
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {
        if let Some(nvar) = self.rename(&*var) {
            *var = nvar;
        }
    }

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
            Expr::Extract(ref mut expr, lsb, msb) => self.visit_expr_extract_mut(expr, *lsb, *msb),
            Expr::Concat(ref mut lexpr, ref mut rexpr) => self.visit_expr_concat_mut(lexpr, rexpr),
            Expr::IfElse(ref mut cond, ref mut texpr, ref mut fexpr) => self.visit_expr_ite_mut(cond, texpr, fexpr),
            Expr::Intrinsic(ref name, ref mut args, bits) => {
                self.visit_expr_intrinsic_mut(name, args, *bits)
            }
            Expr::Val(ref mut bv) => self.visit_expr_val_mut(bv),
            Expr::Var(ref mut var) => if let Some(nexpr) = self.replace(&*var) {
                *expr = nexpr;
            } else {
                self.visit_expr_var_mut(var)
            }
        }
    }
}

pub struct Substitutor<T: Substitution>(T);

impl<T: Substitution> Substitutor<T> {
    pub fn new(subst: T) -> Self {
        Self(subst)
    }

    pub fn apply_stmt<'ecode>(&mut self, stmt: &'ecode mut Stmt) {
        self.0.visit_stmt_mut(stmt);
    }

    pub fn apply_expr<'ecode>(&mut self, expr: &'ecode mut Expr) {
        self.0.visit_expr_mut(expr)
    }
}

pub struct SimpleVarSubst<'a> {
    var: SimpleVar<'a>,
    expr: Expr,
}

impl<'a> SimpleVarSubst<'a> {
    pub fn new(var: &'a Var, expr: Expr) -> Self {
        Self {
            var: SimpleVar::from(var),
            expr,
        }
    }
}

impl<'a> Substitution for SimpleVarSubst<'a> {
    fn replace(&mut self, var: &Var) -> Option<Expr> {
        if SimpleVar::from(var) == self.var {
            Some(self.expr.clone())
        } else {
            None
        }
    }
}
