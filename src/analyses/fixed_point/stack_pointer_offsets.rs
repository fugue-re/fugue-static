use fugue::bv::BitVec;
use fugue::ir::il::ecode::{Expr, Var};

use std::cmp::Ordering;

use crate::analyses::fixed_point::FixedPointForward;
use crate::traits::*;

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
}

impl Default for StackPointerShift {
    fn default() -> Self {
        Self::Top
    }
}