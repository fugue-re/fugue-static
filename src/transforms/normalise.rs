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

use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::iter::FromIterator;

use fugue::ir::il::ecode::{EntityId, Expr, Var};

use crate::models::{Block, CFG};
use crate::types::{SimpleVar, VarViews};
use crate::traits::*;

trait VarClass<'a> {
    fn class_equivalent<V: Into<SimpleVar<'a>>>(&'a self, other: V, classes: &VarViews) -> bool;
}

impl<'a, T> VarClass<'a> for T where T: 'a, &'a T: Into<SimpleVar<'a>> {
    fn class_equivalent<V: Into<SimpleVar<'a>>>(&'a self, other: V, classes: &VarViews) -> bool {
        let sc = classes.enclosing(self);
        let oc = classes.enclosing(other);
        sc == oc
    }
}

#[derive(educe::Educe)]
#[educe(Clone, Default, PartialOrd, PartialEq, Eq)]
pub struct DefinitionMap<'a, 'ecode> {
    mapping: Cow<'a, BTreeMap<SimpleVar<'ecode>, BTreeSet<EntityId>>>,
}

#[derive(Debug, Clone)]
pub struct AliasedDefs {
    classes: VarViews,
    all_vars: bool,
}

impl From<VarViews> for AliasedDefs {
    fn from(classes: VarViews) -> Self {
        Self {
            classes,
            all_vars: false,
        }
    }
}

impl AliasedDefs {
    pub fn new(graph: &CFG<Block>) -> Self {
        Self {
            classes: Self::variable_aliases(graph),
            all_vars: false,
        }
    }

    pub fn new_with(graph: &CFG<Block>, views: VarViews) -> Self {
        Self {
            classes: {
                let mut vars = Self::variable_aliases(graph);
                vars.merge(views);
                vars
            },
            all_vars: false,
        }
    }

    pub fn all_variables(&mut self, toggle: bool) {
        self.all_vars = toggle;
    }

    fn should_transform(&self, var: &Var) -> bool {
        self.all_vars || var.space().is_unique() || var.space().is_register()
    }

    pub fn transform(graph: &mut CFG<Block>) {
        Self::transform_with(graph, Default::default())
    }

    pub fn transform_with(graph: &mut CFG<Block>, views: VarViews) {
        let mut defs = Self::new_with(graph, views);
        for (_, _, blk) in graph.entities_mut() {
            let nblk = blk.to_mut();
            for op in nblk.operations_mut() {
                defs.visit_stmt_mut(op.value_mut());
            }
        }
    }

    /// The goal of this analysis is to minimise the number of inserted
    /// definitions. Therefore, this function only considers aliases with
    /// respect to the CFG being transformed.
    fn variable_aliases(graph: &CFG<Block>) -> VarViews {
        let mut defs = BTreeSet::new();
        let mut uses = BTreeSet::new();

        for (_, _, block) in graph.entities() {
            block.defined_and_used_variables_with(&mut defs, &mut uses);
        }

        VarViews::from_iter(defs.union(&uses).map(|v| *v))
    }

    /// Assumes that svar.bits() != pvar.bits()
    /// Assumes that either svar completely contains pvar or pvar completely contains svar
    fn resize_expr<E: Borrow<Expr>>(svar: &SimpleVar, pvar: &SimpleVar, expr: E) -> Expr {
        let expr = expr.borrow().clone();

        match svar.bits().cmp(&pvar.bits()) {
            Ordering::Greater => if svar.offset() == pvar.offset() { // truncate
                // e.g svar: RAX, pvar: AL
                Expr::extract_low(expr, pvar.bits())
            } else {
                // e.g. svar: RAX, pvar: AH
                let loff = (pvar.offset() - svar.offset()) as usize * 8;
                let moff = loff + pvar.bits();
                Expr::extract(expr, loff, moff)
            },
            Ordering::Less => if svar.offset() == pvar.offset() {
                // e.g. svar: AL, pvar: RAX
                let hbits = Expr::extract_high(***pvar, pvar.bits() - svar.bits());
                Expr::concat(hbits, expr)
            } else {
                if svar.offset() + (svar.bits() as u64 / 8) == (pvar.bits() as u64 / 8) {
                    // e.g. svar: AH, pvar: AX
                    let lbits = Expr::extract_low(***pvar, pvar.bits() - svar.bits());
                    Expr::concat(expr, lbits)
                } else {
                    // e.g. svar: AH, pvar: RAX
                    let shift = (svar.offset() - pvar.offset()) as usize * 8;

                    let hbits = Expr::extract_high(***pvar, pvar.bits() - svar.bits() - shift);
                    let lbits = Expr::extract_low(***pvar, shift);

                    Expr::concat(hbits, Expr::concat(expr, lbits))
                }
            },
            Ordering::Equal => expr,
        }
    }
}

impl<'ecode> VisitMut<'ecode> for AliasedDefs {
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
            Expr::Var(ref mut var) => if self.should_transform(var) {
                let svar = SimpleVar::from(&*var);
                let pvar = self.classes.enclosing(&svar);

                if svar != pvar {
                    let rvar = **pvar;
                    *expr = Self::resize_expr(&pvar, &svar, Expr::from(rvar));
                }
            },
            Expr::Val(_) => (),
        }
    }

    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, expr: &'ecode mut Expr) {
        if self.should_transform(var) {
            let svar = SimpleVar::from(&*var);
            let pvar = self.classes.enclosing(&svar);

            if svar != pvar {
                // expand
                let rvar = Var::new(pvar.space(), pvar.offset(), pvar.bits(), var.generation());
                *expr = Self::resize_expr(&svar, &pvar, &*expr);
                *var = rvar;
            }
        }
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

        let base = VarViews::registers(&translator);
        let mut defs = AliasedDefs::from(base);

        let rax = Var::from(*translator.register_by_name("RAX").unwrap());
        let ax = Var::from(*translator.register_by_name("AX").unwrap());
        let ah = Var::from(*translator.register_by_name("AH").unwrap());
        let al = Var::from(*translator.register_by_name("AL").unwrap());

        let is_set_var = |stmt: &Stmt, var| matches!(stmt, Stmt::Assign(v, _) if *v == var);
        let is_set_exp = |stmt: &Stmt, exp| matches!(stmt, Stmt::Assign(_, e) if *e == exp);

        let mut s1 = Stmt::assign(ah, BitVec::from(0xffu8));
        defs.visit_stmt_mut(&mut s1);

        assert!(is_set_var(&s1, rax) && is_set_exp(&s1, Expr::concat(Expr::extract_high(rax, 48), Expr::concat(BitVec::from(0xffu8), Expr::extract_low(rax, 8)))));

        let mut s2 = Stmt::assign(al, BitVec::from(0xffu8));
        defs.visit_stmt_mut(&mut s2);

        assert!(is_set_var(&s2, rax) && is_set_exp(&s2, Expr::concat(Expr::extract_high(rax, 56), BitVec::from(0xffu8))));

        let mut s3 = Stmt::assign(rax, BitVec::from(0xffu64));
        defs.visit_stmt_mut(&mut s3);

        assert!(is_set_var(&s3, rax) && is_set_exp(&s3, Expr::from(BitVec::from(0xffu64))));

        let mut s4 = Stmt::assign(ax, BitVec::from(0xffu16));
        defs.visit_stmt_mut(&mut s4);

        assert!(is_set_var(&s4, rax) && is_set_exp(&s4, Expr::concat(Expr::extract_high(rax, 48), BitVec::from(0xffu16))));

        let mut e1 = Expr::int_add(ax, BitVec::from(0xffu16));
        defs.visit_expr_mut(&mut e1);

        assert_eq!(e1, Expr::int_add(Expr::extract_low(rax, 16), BitVec::from(0xffu16)));

        let mut e2 = Expr::int_add(ah, BitVec::from(0xffu8));
        defs.visit_expr_mut(&mut e2);

        assert_eq!(e2, Expr::int_add(Expr::extract(rax, 8, 16), BitVec::from(0xffu8)));

        Ok(())
    }
}
