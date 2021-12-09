/// This analysis normalises the ECode IR to handle implicitly
/// aliased variables which occur as artefacts from the PCode
/// lifting process.
///
/// To do this, we expand variables to their maximal used size
/// and perform operations on views over each variables enclosing
/// region (e.g., assignments to and uses of AL, AH, etc. are in
/// terms of RAX for 64-bit x86).
///
/// NOTE: we assume that the graph is not in SSA form and we make
/// no effort to preserve it.
///
/// This transform should be applied as:
///
/// Lift -> Normalise -> SSA
///

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::iter::FromIterator;

use fugue::ir::{AddressSpaceId, Translator};
use fugue::ir::il::ecode::{ECode, ExprT, StmtT, Var};
use fugue::ir::il::traits::*;

use crate::models::{Block, CFG};
use crate::types::{SimpleVar, VarView, VarViews};
use crate::traits::*;

#[derive(Debug, Clone)]
pub struct VariableAliasNormaliser {
    register_space: AddressSpaceId,
    register_view: VarView,
    all_vars: bool,
}

struct VariableAliasNormaliserVisitor<'v> {
    classes: VarViews<'v>,
    all_vars: bool,
}

impl<'v> VariableAliasNormaliserVisitor<'v> {
    fn new(classes: VarViews<'v>, all_vars: bool) -> Self {
        Self {
            classes,
            all_vars,
        }
    }

    fn should_transform(&self, var: &Var) -> bool {
        self.all_vars || var.space().is_unique() || var.space().is_register()
    }

    /// Assumes that svar.bits() != pvar.bits()
    /// Assumes that either svar completely contains pvar or pvar completely contains svar
    fn resize_expr<Loc, Val, E: Borrow<ExprT<Loc, Val, Var>>>(svar: &SimpleVar, pvar: &SimpleVar, expr: E) -> ExprT<Loc, Val, Var>
    where Loc: Clone,
          Val: BitSize + Clone, {
        let expr = expr.borrow().clone();

        match svar.bits().cmp(&pvar.bits()) {
            Ordering::Greater => if svar.offset() == pvar.offset() { // truncate
                // e.g svar: RAX, pvar: AL
                ExprT::extract_low(expr, pvar.bits())
            } else {
                // e.g. svar: RAX, pvar: AH
                let loff = (pvar.offset() - svar.offset()) as usize * 8;
                let moff = loff + pvar.bits();
                ExprT::extract(expr, loff, moff)
            },
            Ordering::Less => if svar.offset() == pvar.offset() {
                // e.g. svar: AL, pvar: RAX
                let hbits = ExprT::extract_high(***pvar, pvar.bits() - svar.bits());
                ExprT::concat(hbits, expr)
            } else {
                if svar.offset() + (svar.bits() as u64 / 8) == (pvar.bits() as u64 / 8) {
                    // e.g. svar: AH, pvar: AX
                    let lbits = ExprT::extract_low(***pvar, pvar.bits() - svar.bits());
                    ExprT::concat(expr, lbits)
                } else {
                    // e.g. svar: AH, pvar: RAX
                    let shift = (svar.offset() - pvar.offset()) as usize * 8;

                    let hbits = ExprT::extract_high(***pvar, pvar.bits() - svar.bits() - shift);
                    let lbits = ExprT::extract_low(***pvar, shift);

                    ExprT::concat(hbits, ExprT::concat(expr, lbits))
                }
            },
            Ordering::Equal => expr,
        }
    }
}

impl VariableAliasNormaliser {
    pub fn new<T: Borrow<Translator>>(translator: T) -> Self {
        let (register_space, register_view) = VarView::registers(translator);
        Self {
            register_space,
            register_view,
            all_vars: false,
        }
    }

    fn transform_stmt<Loc, Val>(&self, stmt: &mut StmtT<Loc, Val, Var>)
    where
        Loc: Clone,
        Val: BitSize + Clone
    {
        let mut classes = VarViews::from_space(
            self.register_space,
            &self.register_view,
        );

        Self::stmt_variable_aliases(&mut classes, stmt);

        let mut visitor = VariableAliasNormaliserVisitor::new(classes, self.all_vars);

        visitor.visit_stmt_mut(stmt);
    }

    fn transform_ecode(&self, ecode: &mut ECode) {
        let mut classes = VarViews::from_space(
            self.register_space,
            &self.register_view,
        );

        Self::ecode_variable_aliases(&mut classes, ecode);

        let mut visitor = VariableAliasNormaliserVisitor::new(classes, self.all_vars);

        for op in ecode.operations_mut().iter_mut() {
            visitor.visit_stmt_mut(op);
        }
    }

    fn transform_cfg(&self, graph: &mut CFG<Block>) {
        let mut classes = VarViews::from_space(
            self.register_space,
            &self.register_view,
        );

        Self::cfg_variable_aliases(&mut classes, graph);

        let mut visitor = VariableAliasNormaliserVisitor::new(classes, self.all_vars);

        for (_, _, blk) in graph.entities_mut() {
            let nblk = blk.to_mut();
            for op in nblk.operations_mut() {
                visitor.visit_stmt_mut(op.value_mut());
            }
        }
    }

    pub fn all_variables(&mut self, toggle: bool) {
        self.all_vars = toggle;
    }

    /// The goal of this analysis is to minimise the number of inserted
    /// definitions. Therefore, this function only considers aliases with
    /// respect to the CFG being transformed.
    fn cfg_variable_aliases<'v>(view: &mut VarViews<'v>, graph: &CFG<Block>) {
        let mut vars = BTreeSet::new();

        for (_, _, block) in graph.entities() {
            block.all_variables_with(&mut vars);
        }

        view.merge(VarViews::from_iter(vars))
    }

    fn ecode_variable_aliases<'v>(view: &mut VarViews<'v>, ecode: &ECode) {
        let mut vars = BTreeSet::new();

        for op in ecode.operations().iter() {
            op.all_variables_with(&mut vars);
        }

        view.merge(VarViews::from_iter(vars))
    }

    fn stmt_variable_aliases<'v, Loc, Val>(view: &mut VarViews<'v>, stmt: &StmtT<Loc, Val, Var>) {
        let vars = stmt.all_variables::<BTreeSet<_>>();
        view.merge(VarViews::from_iter(vars))
    }
}

impl<'ecode, 'v, Loc, Val> VisitMut<'ecode, Loc, Val, Var> for VariableAliasNormaliserVisitor<'v>
where
    Loc: Clone,
    Val: BitSize + Clone
{
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
            ExprT::Load(ref mut expr, size, space) => {
                self.visit_expr_load_mut(expr, *size, *space)
            }
            ExprT::Extract(ref mut expr, lsb, msb) => self.visit_expr_extract_mut(expr, *lsb, *msb),
            ExprT::Concat(ref mut lexpr, ref mut rexpr) => self.visit_expr_concat_mut(lexpr, rexpr),
            ExprT::IfElse(ref mut cond, ref mut texpr, ref mut fexpr) => self.visit_expr_ite_mut(cond, texpr, fexpr),
            ExprT::Call(ref mut branch_target, ref mut args, bits) => {
                self.visit_expr_call_mut(branch_target, args, *bits)
            }
            ExprT::Intrinsic(ref name, ref mut args, bits) => {
                self.visit_expr_intrinsic_mut(name, args, *bits)
            }
            ExprT::Var(ref mut var) => if self.should_transform(var) {
                let svar = SimpleVar::from(&*var);
                let pvar = self.classes.enclosing(&svar);

                let rvar = **pvar;
                *expr = Self::resize_expr(&pvar, &svar, ExprT::from(rvar));
            },
            ExprT::Val(_) => (),
        }
    }

    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        if self.should_transform(var) {
            let svar = SimpleVar::from(&*var);
            let pvar = self.classes.enclosing(&svar);

            // expand
            self.visit_expr_mut(expr);

            let rvar = Var::new(pvar.space(), pvar.offset(), pvar.bits(), var.generation());
            *expr = Self::resize_expr(&svar, &pvar, &*expr);
            *var = rvar;
        }
    }
}

pub trait NormaliseAliases {
    fn normalise_aliases(&mut self, n: &VariableAliasNormaliser);
}

impl<Loc, Val> NormaliseAliases for StmtT<Loc, Val, Var>
where
    Loc: Clone,
    Val: BitSize + Clone,
{
    fn normalise_aliases(&mut self, n: &VariableAliasNormaliser) {
        n.transform_stmt(self);
    }
}

impl NormaliseAliases for ECode {
    fn normalise_aliases(&mut self, n: &VariableAliasNormaliser) {
        n.transform_ecode(self);
    }
}

impl<'ecode> NormaliseAliases for CFG<'ecode, Block> {
    fn normalise_aliases(&mut self, n: &VariableAliasNormaliser) {
        n.transform_cfg(self);
    }
}

pub struct VariableNormaliser {
    alias_normaliser: VariableAliasNormaliser,
    unique_base: u64,
}

struct VariableNormaliserVisitor {
    mapping: BTreeMap<u64, u64>,
    unique_base: u64,
}

impl VariableNormaliserVisitor {
    fn new(unique_base: u64) -> Self {
        Self {
            mapping: BTreeMap::new(),
            unique_base,
        }
    }
}

impl<'ecode, Loc, Val> VisitMut<'ecode, Loc, Val, Var> for VariableNormaliserVisitor {
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {
        if var.space().is_unique() {
            let mut unique_base = self.unique_base;
            let noffset = self.mapping
                .entry(var.offset())
                .or_insert_with(|| {
                    let offset = unique_base;
                    unique_base += (var.bits() as u64) / 8;
                    offset
                });
            self.unique_base = unique_base;
            *var = Var::new(var.space(), *noffset, var.bits(), var.generation());
        }
    }
}

impl VariableNormaliser {
    pub fn new<T: Borrow<Translator>>(translator: T) -> Self {
        Self {
            alias_normaliser: VariableAliasNormaliser::new(translator),
            unique_base: 0,
        }
    }

    pub fn reset(&mut self) {
        self.unique_base = 0;
    }

    pub fn all_variables(&mut self, toggle: bool) {
        self.alias_normaliser.all_variables(toggle);
    }

    fn transform_ecode(&mut self, ecode: &mut ECode) {
        self.alias_normaliser.transform_ecode(ecode);

        let mut visitor = VariableNormaliserVisitor::new(self.unique_base);

        for op in ecode.operations_mut().iter_mut() {
            visitor.visit_stmt_mut(op);
        }

        self.unique_base = visitor.unique_base;
    }
}

pub trait NormaliseVariables {
    fn normalise_variables(&mut self, normaliser: &mut VariableNormaliser);
}

impl NormaliseVariables for ECode {
    fn normalise_variables(&mut self, normaliser: &mut VariableNormaliser) {
        normaliser.transform_ecode(self);
    }
}

#[cfg(test)]
mod test {
    use fugue::bytes::Endian;
    use fugue::bv::BitVec;
    use fugue::ir::LanguageDB;
    use fugue::ir::il::ecode::Stmt;

    use super::*;

    #[test]
    fn test_var_expander() -> Result<(), Box<dyn std::error::Error>> {
        let ldb = LanguageDB::from_directory_with("./processors", true)?;
        let mut translator = ldb.lookup_default("x86", Endian::Little, 64)
            .unwrap()
            .build()?;

        translator.set_variable_default("addrsize", 2);
        translator.set_variable_default("bit64", 1);
        translator.set_variable_default("opsize", 1);
        translator.set_variable_default("rexprefix", 0);

        let avars = VariableAliasNormaliser::new(&translator);

        let rax = Var::from(*translator.register_by_name("RAX").unwrap());
        let ax = Var::from(*translator.register_by_name("AX").unwrap());
        let ah = Var::from(*translator.register_by_name("AH").unwrap());
        let al = Var::from(*translator.register_by_name("AL").unwrap());

        let is_set_var = |stmt: &Stmt, var| matches!(stmt, Stmt::Assign(v, _) if *v == var);
        let is_set_exp = |stmt: &Stmt, exp| matches!(stmt, Stmt::Assign(_, e) if *e == exp);

        let mut s1 = Stmt::assign(ah, BitVec::from(0xffu8));
        s1.normalise_aliases(&avars);

        assert!(is_set_var(&s1, rax) && is_set_exp(&s1, ExprT::concat(ExprT::extract_high(rax, 48), ExprT::concat(BitVec::from(0xffu8), ExprT::extract_low(rax, 8)))));

        let mut s2 = Stmt::assign(al, BitVec::from(0xffu8));
        s2.normalise_aliases(&avars);

        assert!(is_set_var(&s2, rax) && is_set_exp(&s2, ExprT::concat(ExprT::extract_high(rax, 56), BitVec::from(0xffu8))));

        let mut s3 = Stmt::assign(rax, BitVec::from(0xffu64));
        s3.normalise_aliases(&avars);

        assert!(is_set_var(&s3, rax) && is_set_exp(&s3, ExprT::from(BitVec::from(0xffu64))));

        let mut s4 = Stmt::assign(ax, BitVec::from(0xffu16));
        s4.normalise_aliases(&avars);

        assert!(is_set_var(&s4, rax) && is_set_exp(&s4, ExprT::concat(ExprT::extract_high(rax, 48), BitVec::from(0xffu16))));

        Ok(())
    }
}
