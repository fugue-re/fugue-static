use fugue::ir::il::ecode::{ExprT, StmtT, Var};
use std::marker::PhantomData;

use crate::graphs::AsEntityGraphMut;
use crate::models::BlockT;
use crate::traits::VisitMut;
use crate::traits::{ValueMutCollector, ValueRefCollector};
use crate::types::SimpleVar;

pub trait Variables<'ecode, Var> {
    fn all_variables<C>(&'ecode self) -> C
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        let mut vars = C::default();
        self.all_variables_with(&mut vars);
        vars
    }

    fn all_variables_mut<C>(&'ecode mut self) -> C
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        let mut vars = C::default();
        self.all_variables_mut_with(&mut vars);
        vars
    }

    fn defined_variables<C>(&'ecode self) -> C
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        let mut vars = C::default();
        self.defined_variables_with(&mut vars);
        vars
    }

    fn defined_variables_mut<C>(&'ecode mut self) -> C
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        let mut vars = C::default();
        self.defined_variables_mut_with(&mut vars);
        vars
    }

    fn used_variables<C>(&'ecode self) -> C
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        let mut vars = C::default();
        self.used_variables_with(&mut vars);
        vars
    }

    fn used_variables_mut<C>(&'ecode mut self) -> C
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        let mut vars = C::default();
        self.used_variables_mut_with(&mut vars);
        vars
    }

    fn defined_and_used_variables<C>(&'ecode self) -> (C, C)
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        (self.defined_variables(), self.used_variables())
    }

    fn all_variables_with<C>(&'ecode self, vars: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>;

    fn all_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>;

    fn defined_variables_with<C>(&'ecode self, vars: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>;

    fn used_variables_with<C>(&'ecode self, vars: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>;

    fn defined_and_used_variables_with<C>(&'ecode self, defs: &mut C, uses: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        self.defined_variables_with(defs);
        self.used_variables_with(uses);
    }

    fn defined_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>;

    fn used_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>;

    fn defined_and_used_variables_mut_with<C>(&'ecode mut self, defs: &mut C, uses: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>;
}

pub trait Substitution<Loc, Val, Var> {
    #[allow(unused_variables)]
    fn rename(&mut self, var: &Var) -> Option<Var> {
        None
    }

    #[allow(unused_variables)]
    fn replace(&mut self, var: &Var) -> Option<ExprT<Loc, Val, Var>> {
        None
    }
}

impl<'ecode, Loc, Val, Var, T> VisitMut<'ecode, Loc, Val, Var> for Substitutor<Loc, Val, Var, T>
where
    T: Substitution<Loc, Val, Var>,
{
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {
        if let Some(nvar) = self.0.rename(&*var) {
            *var = nvar;
        }
    }

    fn visit_expr_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        match expr {
            ExprT::UnRel(op, ref mut expr) => self.visit_expr_unrel_mut(*op, expr),
            ExprT::UnOp(op, ref mut expr) => self.visit_expr_unop_mut(*op, expr),
            ExprT::BinRel(op, ref mut lexpr, ref mut rexpr) => {
                self.visit_expr_binrel_mut(*op, lexpr, rexpr)
            }
            ExprT::BinOp(op, ref mut lexpr, ref mut rexpr) => {
                self.visit_expr_binop_mut(*op, lexpr, rexpr)
            }
            ExprT::Cast(ref mut expr, ref mut cast) => self.visit_expr_cast_mut(expr, cast),
            ExprT::Load(ref mut expr, size, space) => self.visit_expr_load_mut(expr, *size, *space),
            ExprT::ExtractHigh(ref mut expr, bits) => self.visit_expr_extract_high_mut(expr, *bits),
            ExprT::ExtractLow(ref mut expr, bits) => self.visit_expr_extract_low_mut(expr, *bits),
            ExprT::Extract(ref mut expr, lsb, msb) => self.visit_expr_extract_mut(expr, *lsb, *msb),
            ExprT::Concat(ref mut lexpr, ref mut rexpr) => self.visit_expr_concat_mut(lexpr, rexpr),
            ExprT::IfElse(ref mut cond, ref mut texpr, ref mut fexpr) => {
                self.visit_expr_ite_mut(cond, texpr, fexpr)
            }
            ExprT::Call(ref mut branch_target, ref mut args, bits) => {
                self.visit_expr_call_mut(branch_target, args, *bits)
            }
            ExprT::Intrinsic(ref name, ref mut args, bits) => {
                self.visit_expr_intrinsic_mut(name, args, *bits)
            }
            ExprT::Val(ref mut bv) => self.visit_expr_val_mut(bv),
            ExprT::Var(ref mut var) => {
                if let Some(nexpr) = self.0.replace(&*var) {
                    *expr = nexpr;
                } else {
                    self.visit_expr_var_mut(var)
                }
            }
        }
    }
}

pub struct Substitutor<Loc, Val, Var, T: Substitution<Loc, Val, Var>>(
    T,
    PhantomData<(Loc, Val, Var)>,
);

pub struct SubstLoadStore<'a, Loc, Val, Var, T: Substitution<Loc, Val, Var>>(
    &'a mut Substitutor<Loc, Val, Var, T>,
);

impl<'a, Loc, Val, Var, T> SubstLoadStore<'a, Loc, Val, Var, T>
where
    T: Substitution<Loc, Val, Var>,
{
    pub fn new(subst: &'a mut Substitutor<Loc, Val, Var, T>) -> Self {
        Self(subst)
    }

    pub fn apply_stmt<'ecode>(&mut self, stmt: &'ecode mut StmtT<Loc, Val, Var>) {
        self.visit_stmt_mut(stmt);
    }

    pub fn apply_expr<'ecode>(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(expr)
    }
}

impl<'a, Loc, Val, Var, T> SubstLoadStore<'a, Loc, Val, Var, T>
where
    T: Substitution<Loc, Val, Var>,
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    pub fn apply_block<'ecode>(&mut self, block: &'ecode mut BlockT<Loc, Val, Var>) {
        for phi in block.phis_mut() {
            phi.assign_mut()
                .iter_mut()
                .for_each(|var| self.visit_var_mut(var));
        }

        for op in block.operations_mut() {
            self.apply_stmt(op);
        }
    }

    pub fn apply_graph<'ecode, G, E>(&mut self, graph: &mut G)
    where
        G: AsEntityGraphMut<'ecode, BlockT<Loc, Val, Var>, E>,
        Loc: 'ecode,
        Val: 'ecode,
        Var: 'ecode,
    {
        let g = graph.entity_graph_mut();
        for vx in g.reverse_post_order() {
            let blk = g.entity_mut(vx);
            self.apply_block(&mut **blk.to_mut());
        }
    }
}

impl<'a, 'ecode, Loc, Val, Var, T> VisitMut<'ecode, Loc, Val, Var>
    for SubstLoadStore<'a, Loc, Val, Var, T>
where
    T: Substitution<Loc, Val, Var>,
{
    fn visit_expr_load_mut(
        &mut self,
        expr: &'ecode mut ExprT<Loc, Val, Var>,
        _size: usize,
        _space: fugue::ir::AddressSpaceId,
    ) {
        self.0.visit_expr_mut(expr);
    }

    fn visit_stmt_store_mut(
        &mut self,
        loc: &'ecode mut ExprT<Loc, Val, Var>,
        val: &'ecode mut ExprT<Loc, Val, Var>,
        _size: usize,
        _space: fugue::ir::AddressSpaceId,
    ) {
        self.visit_expr_mut(val);
        self.0.visit_expr_mut(loc);
    }
}

impl<Loc, Val, Var, T> Substitutor<Loc, Val, Var, T>
where
    T: Substitution<Loc, Val, Var>,
{
    pub fn new(subst: T) -> Self {
        Self(subst, PhantomData)
    }

    pub fn apply_stmt<'ecode>(&mut self, stmt: &'ecode mut StmtT<Loc, Val, Var>) {
        self.visit_stmt_mut(stmt);
    }

    pub fn apply_expr<'ecode>(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(expr)
    }
}

impl<Loc, Val, Var, T> Substitutor<Loc, Val, Var, T>
where
    T: Substitution<Loc, Val, Var>,
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    pub fn apply_block<'ecode>(&mut self, block: &'ecode mut BlockT<Loc, Val, Var>) {
        for phi in block.phis_mut() {
            phi.assign_mut()
                .iter_mut()
                .for_each(|var| self.visit_var_mut(var));
        }

        for op in block.operations_mut() {
            self.apply_stmt(op);
        }
    }

    pub fn apply_graph<'ecode, G, E>(&mut self, graph: &mut G)
    where
        G: AsEntityGraphMut<'ecode, BlockT<Loc, Val, Var>, E>,
        Loc: 'ecode,
        Val: 'ecode,
        Var: 'ecode,
    {
        let g = graph.entity_graph_mut();
        for vx in g.reverse_post_order() {
            let blk = g.entity_mut(vx);
            self.apply_block(&mut **blk.to_mut());
        }
    }
}

pub struct SimpleVarSubst<'a, Loc, Val> {
    var: SimpleVar<'a>,
    expr: ExprT<Loc, Val, Var>,
}

impl<'a, Loc, Val> SimpleVarSubst<'a, Loc, Val> {
    pub fn new(var: &'a Var, expr: ExprT<Loc, Val, Var>) -> Self {
        Self {
            var: SimpleVar::from(var),
            expr,
        }
    }
}

impl<'a, Loc, Val> Substitution<Loc, Val, Var> for SimpleVarSubst<'a, Loc, Val>
where
    Loc: Clone,
    Val: Clone,
{
    fn replace(&mut self, var: &Var) -> Option<ExprT<Loc, Val, Var>> {
        if SimpleVar::from(var) == self.var {
            Some(self.expr.clone())
        } else {
            None
        }
    }
}
