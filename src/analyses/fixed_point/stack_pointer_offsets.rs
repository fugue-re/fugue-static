use fugue::bv::BitVec;
use fugue::ir::il::ecode::{Location, Stmt, Var};

use std::collections::BTreeSet;
use std::convert::Infallible;

use crate::analyses::expressions::constant::ConstExpr;
use crate::analyses::fixed_point::FixedPointForward;
use crate::models::cfg::BranchKind;
use crate::models::{Block, CFG};
use crate::traits::variables::SimpleVarSubst;
use crate::traits::*;
use crate::types::SimpleVar;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StackPointerShift {
    Top,
    Shift(BitVec),
    Bottom,
}

impl StackPointerShift {
    pub fn is_shift(&self) -> bool {
        matches!(self, Self::Shift(_))
    }

    pub fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }

    pub fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    fn join(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (_, Self::Shift(ref bv)) => Self::Shift(bv.clone()),
            (Self::Shift(bv), _) => Self::Shift(bv),
            (slf, _) => slf,
        }
    }
}

impl Default for StackPointerShift {
    fn default() -> Self {
        Self::Bottom
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct StackPointerBlockShift {
    start: StackPointerShift,
    finish: StackPointerShift,
}

pub struct StackPointerOffset {
    sp: Var,
    roots: BTreeSet<Location>
}

impl<'ecode> FixedPointForward<'ecode, Block, BranchKind, CFG<'ecode, Block>, StackPointerBlockShift>
    for StackPointerOffset
{
    type Err = Infallible;

    fn join(
        &mut self,
        current: StackPointerBlockShift,
        next: &StackPointerBlockShift,
    ) -> Result<StackPointerBlockShift, Self::Err> {
        let current = current.start.join(&next.finish);
        Ok(StackPointerBlockShift {
            start: current.clone(),
            finish: current,
        })
    }

    fn transfer(
        &mut self,
        entity: &'ecode Block,
        current: Option<StackPointerBlockShift>,
    ) -> Result<StackPointerBlockShift, Self::Err> {
        // if an entry point to CFG then assume that SP == 0
        let mut shift = if self.roots.contains(entity.location()) {
            current.unwrap_or_else(|| StackPointerBlockShift {
                start: StackPointerShift::Shift(BitVec::zero(self.sp.bits())),
                finish: StackPointerShift::Shift(BitVec::zero(self.sp.bits())),
            })
        } else {
            current.unwrap_or_default()
        };

        // NOTE: join will take care of phi node assignment to SP

        for op in entity.operations() {
            if shift.finish.is_top() {
                break;
            }

            match (op.value(), &shift.finish) {
                (Stmt::Assign(ref var, ref expr), StackPointerShift::Shift(ref sft))
                    if SimpleVar::from(var) == SimpleVar::from(self.sp) =>
                {
                    let mut expr = expr.clone();
                    let mut subst =
                        Substitutor::new(SimpleVarSubst::new(&self.sp, sft.clone().into()));
                    subst.apply_expr(&mut expr);

                    if let Some(nsft) = expr.to_constant() {
                        shift.finish = shift.finish.join(&StackPointerShift::Shift(nsft));
                    } else {
                        shift.finish = StackPointerShift::Top; // no idea what it could be (-:
                    }
                }
                _ => (),
            }
        }

        Ok(shift)
    }
}
