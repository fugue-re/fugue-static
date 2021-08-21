use fugue::bv::BitVec;
use fugue::ir::il::ecode::{BinOp, BinRel, Cast, Expr, Stmt, UnOp, UnRel, Var};

use std::borrow::Cow;

use crate::traits::Visit;

#[derive(Default)]
pub struct ConstEvaluator<'a> {
    value: Option<Cow<'a, BitVec>>,
    variable: Option<&'a Var>,
}

impl<'a> ConstEvaluator<'a> {
    pub fn eval_unary_with<F>(&mut self, expr: &'a Expr, f: F)
    where F: FnOnce(Cow<'a, BitVec>) -> Option<Cow<'a, BitVec>> {
        let mut lv = Self::default();
        lv.visit_expr(expr);

        if let Some(val) = lv.value.and_then(|v| f(v)) {
            self.value = Some(val);
        }
    }

    pub fn eval_binary_with<F>(&mut self, lexpr: &'a Expr, rexpr: &'a Expr, f: F)
    where F: FnOnce(Cow<'a, BitVec>, Cow<'a, BitVec>) -> Option<Cow<'a, BitVec>> {
        let mut lv = Self::default();
        lv.visit_expr(lexpr);

        let mut rv = Self::default();
        rv.visit_expr(rexpr);

        if let Some(val) = lv.value.and_then(|l| rv.value.and_then(|r| f(l, r))) {
            self.value = Some(val);
        }
    }

    pub fn set_variable(&mut self, var: &'a Var) {
        self.variable = Some(var);
    }

    pub fn set_value(&mut self, val: &'a BitVec) {
        self.value = Some(Cow::Borrowed(val));
    }

    pub fn clear_value(&mut self) {
        self.value = None;
    }

    pub fn into_assign(self) -> Option<Stmt> {
        self.variable.cloned().and_then(|var| {
            let value = self.into_value()?;
            Some(Stmt::assign(var, value))
        })
    }

    pub fn into_value(self) -> Option<BitVec> {
        self.value.map(|v| match v {
            Cow::Borrowed(v) => v.clone(),
            Cow::Owned(v) => v,
        })
    }
}

impl<'ecode> Visit<'ecode> for ConstEvaluator<'ecode> {
    fn visit_expr_val(&mut self, bv: &'ecode BitVec) {
        self.set_value(bv);
    }

    fn visit_expr_cast(&mut self, expr: &'ecode Expr, cast: &'ecode Cast) {
        match cast {
            Cast::Bool => self.eval_unary_with(expr, |v| Some(Cow::Owned(if v.is_zero() { BitVec::zero(8) } else { BitVec::one(8) }))),
            Cast::Signed(bits) => self.eval_unary_with(expr, |v| Some(Cow::Owned((&*v).clone().signed().cast(*bits)))),
            Cast::Unsigned(bits) => self.eval_unary_with(expr, |v| Some(Cow::Owned((&*v).clone().unsigned().cast(*bits)))),
            Cast::High(bits) => self.eval_unary_with(expr, |v| Some(Cow::Owned(if *bits >= v.bits() {
                (&*v).clone().unsigned().cast(*bits)
            } else {
                ((&*v).clone().unsigned() >> (v.bits() as u32 - *bits as u32)).cast(*bits)
            }))),
            Cast::Low(bits) => self.eval_unary_with(expr, |v| Some(Cow::Owned((&*v).clone().unsigned().cast(*bits)))),
            Cast::Float(_) => self.clear_value(),
        }
    }

    fn visit_expr_unrel(&mut self, _op: UnRel, _expr: &'ecode Expr) {
        // ignore float ops (is NaN) for now
        self.clear_value()
    }

    fn visit_expr_unop(&mut self, op: UnOp, expr: &'ecode Expr) {
        match op {
            UnOp::NEG => self.eval_unary_with(expr, |v| Some(Cow::Owned(-&*v))),
            UnOp::NOT => self.eval_unary_with(expr, |v| Some(Cow::Owned(!&*v))),
            UnOp::POPCOUNT => self.eval_unary_with(expr, |v| Some(Cow::Owned(BitVec::from(v.count_ones())))),
            _ => self.clear_value(),
        }
    }

    fn visit_expr_binrel(&mut self, op: BinRel, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        match op {
            BinRel::EQ => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l == r { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::NEQ => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l != r { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::LT => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l == r { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::LE => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l == r { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::SLT => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l.signed_cmp(&*r).is_lt() { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::SLE => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l.signed_cmp(&*r).is_le() { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::SBORROW => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l.signed_borrow(&*r) { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::CARRY => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l.carry(&*r) { BitVec::one(8) } else { BitVec::zero(8) }))),
            BinRel::SCARRY => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(if l.signed_carry(&*r) { BitVec::one(8) } else { BitVec::zero(8) }))),
        }
    }

    fn visit_expr_binop(&mut self, op: BinOp, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        match op {
            BinOp::ADD => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l + &*r))),
            BinOp::AND => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l & &*r))),
            BinOp::DIV => self.eval_binary_with(lexpr, rexpr, |l, r| if r.is_zero() { None } else { Some(Cow::Owned(&*l / &*r)) }),
            BinOp::MUL => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l * &*r))),
            BinOp::OR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l | &*r))),
            BinOp::REM => self.eval_binary_with(lexpr, rexpr, |l, r| if r.is_zero() { None } else { Some(Cow::Owned(&*l % &*r)) }),
            BinOp::SAR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(l.signed_shr(&*r)))),
            BinOp::SDIV => self.eval_binary_with(lexpr, rexpr, |l, r| if r.is_zero() { None } else { Some(Cow::Owned(l.signed_div(&*r))) }),
            BinOp::SHL => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l << &*r))),
            BinOp::SHR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l >> &*r))),
            BinOp::SREM => self.eval_binary_with(lexpr, rexpr, |l, r| if r.is_zero() { None } else { Some(Cow::Owned(l.signed_rem(&*r))) }),
            BinOp::SUB => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l - &*r))),
            BinOp::XOR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l ^ &*r))),
        }
    }

    fn visit_stmt(&mut self, stmt: &'ecode Stmt) {
        if let Stmt::Assign(var, expr) = stmt {
            self.set_variable(var);
            self.visit_expr(expr);
        }
    }
}

pub trait ConstExpr {
    fn to_constant(&self) -> Option<BitVec>;
}

impl ConstExpr for Expr {
    fn to_constant(&self) -> Option<BitVec> {
        let mut eval = ConstEvaluator::default();
        eval.visit_expr(self);
        eval.into_value()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add_signed_cast() -> Result<(), Box<dyn std::error::Error>> {
        let expr = Expr::int_add(BitVec::from(10u32), Expr::cast_signed(BitVec::from(0xffff_ffffu32), 32));
        let value = expr.to_constant();

        assert_eq!(value, Some(BitVec::from(9u32)));

        println!("{}", value.unwrap());

        Ok(())
    }
}
