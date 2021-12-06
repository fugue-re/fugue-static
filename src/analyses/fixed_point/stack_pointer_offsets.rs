use fugue::bv::BitVec;
use fugue::ir::il::ecode::{Location, Stmt, Var};
use fugue::ir::Translator;

use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::convert::Infallible;
use std::fmt::{self, Display};

use crate::analyses::expressions::constant::ConstExpr;
use crate::analyses::fixed_point::FixedPointForward;
use crate::models::cfg::BranchKind;
use crate::models::{Block, CFG};
use crate::traits::variables::SimpleVarSubst;
use crate::traits::*;
use crate::types::{Locatable, SimpleVar};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StackPointerShift {
    Top,
    Shift(BitVec),
    Bottom,
}

impl Display for StackPointerShift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Top => write!(f, "⊤"),
            Self::Bottom => write!(f, "⊥"),
            Self::Shift(ref bv) => {
                let nv = bv.clone().signed();
                let sig = if nv.is_negative() { "-" } else { "+" };
                let abs = nv.abs();
                write!(f, "SP{}{}", sig, abs)
            }
        }
    }
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
            (Self::Shift(bv1), Self::Shift(ref bv2)) => {
                if bv1 == *bv2 {
                    Self::Shift(bv1)
                } else {
                    Self::Top
                }
            }
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

impl Display for StackPointerBlockShift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "start: {}, finish: {}", self.start, self.finish)
    }
}

pub struct StackPointerOffset {
    sp: Var,
    roots: BTreeSet<Location>,
}

impl StackPointerOffset {
    pub fn new<'ecode, T: Borrow<Translator>>(trans: T, cfg: &CFG<'ecode, Block>) -> Self {
        let trans = trans.borrow();

        // We assume all conventions will use the same SP
        // so we just take the first one. This seems like
        // it should be a pretty safe assumption?

        let conv = trans.compiler_conventions().values().next().unwrap();
        let sp = Var::from(*conv.stack_pointer().varnode());

        Self {
            sp,
            roots: cfg
                .root_entities()
                .into_iter()
                .map(|(_, _, b)| b.location())
                .collect(),
        }
    }
}

impl<'ecode>
    FixedPointForward<'ecode, Block, BranchKind, CFG<'ecode, Block>, StackPointerBlockShift>
    for StackPointerOffset
{
    type Err = Infallible;

    fn join(
        &mut self,
        current: StackPointerBlockShift,
        prev: &StackPointerBlockShift,
    ) -> Result<StackPointerBlockShift, Self::Err> {
        let start = current.start.join(&prev.finish);
        Ok(StackPointerBlockShift {
            finish: start.clone(),
            start,
        })
    }

    fn transfer(
        &mut self,
        entity: &'ecode Block,
        current: Option<StackPointerBlockShift>,
    ) -> Result<StackPointerBlockShift, Self::Err> {
        // if an entry point to CFG then assume that SP == 0
        let mut shift = if self.roots.contains(&entity.location()) {
            current.unwrap_or_else(|| StackPointerBlockShift {
                start: StackPointerShift::Shift(BitVec::zero(self.sp.bits())),
                finish: StackPointerShift::Shift(BitVec::zero(self.sp.bits())),
            })
        } else {
            current.unwrap_or_default()
        };

        // move in -> out
        shift.start = shift.finish.clone();

        // NOTE: join will take care of phi node assignment to SP

        for op in entity.operations() {
            if shift.finish.is_top() {
                break;
            }

            match **op.value() {
                Stmt::Assign(ref var, ref expr)
                    if SimpleVar::from(var) == SimpleVar::from(self.sp) =>
                {
                    let mut expr = expr.clone();

                    if let StackPointerShift::Shift(ref sft) = shift.finish {
                        let mut subst =
                            Substitutor::new(SimpleVarSubst::new(&self.sp, sft.clone().into()));
                        subst.apply_expr(&mut expr);
                    }

                    if let Some(nsft) = expr.to_constant() {
                        shift.finish = StackPointerShift::Shift(nsft);
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
