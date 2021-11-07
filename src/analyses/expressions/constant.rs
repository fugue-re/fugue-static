use fugue::bv::BitVec;
use fugue::ir::il::ecode::{BinOp, BinRel, Cast, Expr, Stmt, UnOp, UnRel, Var};

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};

use crate::graphs::AsEntityGraphMut;
use crate::models::block::Block;
use crate::models::cfg::CFG;
use crate::traits::*;

#[derive(Default)]
pub struct ConstEvaluator<'a, 'b> {
    value: Option<Cow<'a, BitVec>>,
    variable: Option<&'a Var>,
    mapping: Cow<'b, BTreeMap<Var, Cow<'a, BitVec>>>,
}

impl<'a, 'b> ConstEvaluator<'a, 'b> {
    pub fn fresh<'c>(&'c self) -> ConstEvaluator<'a, 'c> {
        ConstEvaluator {
            mapping: Cow::Borrowed(self.mapping.as_ref()),
            ..ConstEvaluator::default()
        }
    }

    pub fn eval_unary(&mut self, expr: &'a Expr) -> Option<Cow<'a, BitVec>> {
        let mut v = self.fresh();
        v.visit_expr(expr);
        v.value
    }

    pub fn eval_unary_with<F>(&mut self, expr: &'a Expr, f: F)
    where
        F: FnOnce(Cow<'a, BitVec>) -> Option<Cow<'a, BitVec>>,
    {
        if let Some(val) = self.eval_unary(expr).and_then(|v| f(v)) {
            self.value = Some(val);
        }
    }

    pub fn eval_binary_with<F>(&mut self, lexpr: &'a Expr, rexpr: &'a Expr, f: F)
    where
        F: FnOnce(Cow<'a, BitVec>, Cow<'a, BitVec>) -> Option<Cow<'a, BitVec>>,
    {
        if let Some(val) = self.eval_unary(lexpr).and_then(|l| self.eval_unary(rexpr).and_then(|r| f(l, r))) {
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

impl<'ecode, 'a> Visit<'ecode> for ConstEvaluator<'ecode, 'a> {
    fn visit_expr_val(&mut self, bv: &'ecode BitVec) {
        self.set_value(bv);
    }

    fn visit_var(&mut self, var: &'ecode Var) {
        if let Some(bv) = self.mapping.get(var) {
            self.value = Some(bv.clone());
        } else {
            self.clear_value();
        }
    }

    fn visit_expr_cast(&mut self, expr: &'ecode Expr, cast: &'ecode Cast) {
        match cast {
            Cast::Bool => self.eval_unary_with(expr, |v| {
                Some(Cow::Owned(if v.is_zero() {
                    BitVec::zero(8)
                } else {
                    BitVec::one(8)
                }))
            }),
            Cast::Signed(bits) => self.eval_unary_with(expr, |v| {
                Some(Cow::Owned((&*v).clone().signed().cast(*bits)))
            }),
            Cast::Unsigned(bits) => self.eval_unary_with(expr, |v| {
                Some(Cow::Owned((&*v).clone().unsigned().cast(*bits)))
            }),
            Cast::High(bits) => self.eval_unary_with(expr, |v| {
                Some(Cow::Owned(if *bits >= v.bits() {
                    v.unsigned_cast(*bits)
                } else {
                    (&*v >> (v.bits() as u32 - *bits as u32)).unsigned().cast(*bits)
                }))
            }),
            Cast::Low(bits) => self.eval_unary_with(expr, |v| {
                Some(Cow::Owned(v.unsigned_cast(*bits)))
            }),
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
            UnOp::POPCOUNT => {
                self.eval_unary_with(expr, |v| Some(Cow::Owned(BitVec::from(v.count_ones()))))
            }
            _ => self.clear_value(),
        }
    }

    fn visit_expr_binrel(&mut self, op: BinRel, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        match op {
            BinRel::EQ => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l == r {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::NEQ => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l != r {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::LT => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l < r {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::LE => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l <= r {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::SLT => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l.signed_cmp(&*r).is_lt() {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::SLE => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l.signed_cmp(&*r).is_le() {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::SBORROW => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l.signed_borrow(&*r) {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::CARRY => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l.carry(&*r) {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
            BinRel::SCARRY => self.eval_binary_with(lexpr, rexpr, |l, r| {
                Some(Cow::Owned(if l.signed_carry(&*r) {
                    BitVec::one(8)
                } else {
                    BitVec::zero(8)
                }))
            }),
        }
    }

    fn visit_expr_binop(&mut self, op: BinOp, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        match op {
            BinOp::ADD => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l + &*r))),
            BinOp::AND => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l & &*r))),
            BinOp::DIV => self.eval_binary_with(lexpr, rexpr, |l, r| {
                if r.is_zero() {
                    None
                } else {
                    Some(Cow::Owned(&*l / &*r))
                }
            }),
            BinOp::MUL => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l * &*r))),
            BinOp::OR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l | &*r))),
            BinOp::REM => self.eval_binary_with(lexpr, rexpr, |l, r| {
                if r.is_zero() {
                    None
                } else {
                    Some(Cow::Owned(&*l % &*r))
                }
            }),
            BinOp::SAR => {
                self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(l.signed_shr(&*r))))
            }
            BinOp::SDIV => self.eval_binary_with(lexpr, rexpr, |l, r| {
                if r.is_zero() {
                    None
                } else {
                    Some(Cow::Owned(l.signed_div(&*r)))
                }
            }),
            BinOp::SHL => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l << &*r))),
            BinOp::SHR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l >> &*r))),
            BinOp::SREM => self.eval_binary_with(lexpr, rexpr, |l, r| {
                if r.is_zero() {
                    None
                } else {
                    Some(Cow::Owned(l.signed_rem(&*r)))
                }
            }),
            BinOp::SUB => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l - &*r))),
            BinOp::XOR => self.eval_binary_with(lexpr, rexpr, |l, r| Some(Cow::Owned(&*l ^ &*r))),
        }
    }

    fn visit_expr_concat(&mut self, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        self.eval_binary_with(lexpr, rexpr, |l, r| {
            let bits = l.bits() + r.bits();
            Some(Cow::Owned(
                l.unsigned_cast(bits) << (r.bits() as u32) | r.unsigned_cast(bits),
            ))
        })
    }

    fn visit_expr_extract(&mut self, expr: &'ecode Expr, lsb: usize, msb: usize) {
        self.eval_unary_with(expr, |v| {
            if (msb - lsb) == v.bits() {
                Some(v)
            } else if lsb > 0 {
                Some(Cow::Owned((&*v >> lsb as u32).unsigned_cast((msb - lsb) as usize)))
            } else {
                Some(Cow::Owned(v.unsigned_cast((msb - lsb) as usize)))
            }
        })
    }

    fn visit_expr_ite(&mut self, cond: &'ecode Expr, texpr: &'ecode Expr, fexpr: &'ecode Expr) {
        if let Some(c) = self.eval_unary(cond) {
            self.value = self.eval_unary(if !c.is_zero() { texpr } else { fexpr })
        } else {
            self.clear_value()
        }
    }

    fn visit_stmt(&mut self, stmt: &'ecode Stmt) {
        if let Stmt::Assign(var, expr) = stmt {
            self.set_variable(var);
            self.visit_expr(expr);
        }
    }
}

#[derive(Default)]
pub struct ConstFolder<'a, 'b> {
    mapping: Cow<'b, BTreeMap<Var, Cow<'a, BitVec>>>,
}

impl<'ecode, 'a> VisitMut<'ecode> for ConstFolder<'ecode, 'a> {
    fn visit_expr_mut(&mut self, expr: &'ecode mut Expr) {
        let mut eval = ConstEvaluator {
            mapping: Cow::Borrowed(&*self.mapping),
            ..Default::default()
        };

        eval.visit_expr(expr);

        if let Some(val) = eval.into_value() {
            *expr = Expr::Val(val);
        }
    }
}

#[derive(Default)]
pub struct ConstSubst<'a, 'b> {
    mapping: Cow<'b, BTreeMap<Var, Cow<'a, BitVec>>>,
}

impl<'ecode, 'a> VisitMut<'ecode> for ConstSubst<'ecode, 'a> {
    fn visit_expr_mut(&mut self, expr: &'ecode mut Expr) {
        if let Expr::Var(var) = &*expr {
            if let Some(val) = self.mapping.get(var) {
                *expr = Expr::Val((**val).to_owned())
            }
        }
    }
}

#[derive(Default)]
pub struct ConstPropagator<'a, 'b> {
    mapping: Cow<'b, BTreeMap<Var, Cow<'a, BitVec>>>,
}

pub trait ConstPropagate<'a, 'b> {
    fn propagate_constants(&mut self, propagator: &mut ConstPropagator<'a, 'b>);
}

impl<'a, 'b> ConstPropagate<'a, 'b> for Stmt {
    fn propagate_constants(&mut self, propagator: &mut ConstPropagator<'a, 'b>) {
        propagator.propagate(self)
    }
}

impl<'a, 'b> ConstPropagate<'a, 'b> for Block {
    fn propagate_constants(&mut self, propagator: &mut ConstPropagator<'a, 'b>) {
        propagator.propagate_block(self)
    }
}

impl<'a, 'b> ConstPropagate<'a, 'b> for CFG<'a, Block> {
    fn propagate_constants(&mut self, propagator: &mut ConstPropagator<'a, 'b>) {
        propagator.propagate_graph(self)
    }
}

impl<'a, 'b> ConstPropagator<'a, 'b> {
    pub fn propagate(&mut self, stmt: &mut Stmt) {
        let mut folder = ConstFolder {
            mapping: Cow::Borrowed(&*self.mapping),
            ..Default::default()
        };
        folder.visit_stmt_mut(stmt);

        if let Stmt::Assign(ref var, Expr::Val(ref expr)) = &*stmt {
            self.mapping.to_mut().insert(*var, Cow::Owned((*expr).to_owned()));
        }
    }

    pub fn propagate_block(&mut self, blk: &mut Block) {
        for (v, vs)  in blk.phis() {
            if !self.mapping.contains_key(v) {
                let mut vsf = vs.iter().map(|v| self.mapping.get(v));
                if let Some(bv) = vsf.next().expect("non-empty phi assignment") {
                    for vn in vsf {
                        if !matches!(vn, Some(bvn) if bv == bvn) {
                            continue
                        }
                    }
                    let bv = bv.clone();
                    self.mapping.to_mut().insert(*v, bv);
                }
            }
        }

        for op in blk.operations_mut() {
            self.propagate(op.value_mut());
        }
    }

    pub fn propagate_graph<'g, E, G>(&mut self, g: &mut G)
    where G: AsEntityGraphMut<'g, Block, E> {
        let graph = g.entity_graph_mut();
        let mut rpo = graph.reverse_post_order().map(|(_, v, _)| v).collect::<VecDeque<_>>();
        let mut prop = Self::default();

        while let Some(vx) = rpo.pop_front() {
            let blk = graph.entity_mut(vx);
            prop.propagate_block(blk.to_mut());
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
        let expr = Expr::int_add(
            BitVec::from(10u32),
            Expr::cast_signed(BitVec::from(0xffff_ffffu32), 32),
        );
        let value = expr.to_constant();

        assert_eq!(value, Some(BitVec::from(9u32)));

        println!("{}", value.unwrap());

        Ok(())
    }

    #[test]
    fn test_simplify_concat() {
        let parts = Expr::concat(
            Expr::concat(
                Expr::concat(BitVec::from(3u8), BitVec::from(2u8)),
                BitVec::from(1u8),
            ),
            BitVec::from(0u8),
        );
        let complete = BitVec::from(0x03020100u32);

        println!("{} -> {}", parts, Expr::from(complete.clone()));

        assert_eq!(parts.to_constant(), Some(complete));
    }
}
