use std::borrow::Cow;
use std::fmt;

use crate::ir::{Expr, Location};
use crate::models::{Block, Function};
use crate::types::Id;

use fugue::ir::il::traits::*;

use hashcons::hashconsing::consign;
use hashcons::Term;

consign! { let TRGT = consign(1024) for BranchTarget; }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BranchTarget {
    Block(Id<Block>),
    Function(Id<Function>),
    Location(Location),
    Computed(Term<Expr>),
}

impl<'target, 'trans> fmt::Display for BranchTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BranchTarget::Block(id) => write!(f, "{}", id),
            BranchTarget::Function(id) => write!(f, "{}", id),
            BranchTarget::Location(loc) => write!(f, "{}", loc),
            BranchTarget::Computed(expr) => write!(f, "{}", expr),
        }
    }
}

pub struct BranchTargetFormatter<'target, 'trans> {
    target: &'target BranchTarget,
    fmt: Cow<'trans, TranslatorFormatter<'trans>>
}

impl<'target, 'trans> fmt::Display for BranchTargetFormatter<'target, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target {
            BranchTarget::Block(id) => {
                write!(f, "<blk: {}>", id.uuid())
            }
            BranchTarget::Function(id) => {
                write!(f, "<fcn: {}>", id.uuid())
            }
            BranchTarget::Location(loc) => {
                write!(f, "{}", loc.display_full(Cow::Borrowed(&*self.fmt)))
            }
            BranchTarget::Computed(expr) => {
                write!(f, "{}", expr.display_full(Cow::Borrowed(&*self.fmt)))
            }
        }
    }
}

impl<'target, 'trans> TranslatorDisplay<'target, 'trans> for BranchTarget {
    type Target = BranchTargetFormatter<'target, 'trans>;

    fn display_full(
        &'target self,
        fmt: Cow<'trans, TranslatorFormatter<'trans>>,
    ) -> Self::Target {
        BranchTargetFormatter {
            target: self,
            fmt,
        }
    }
}

impl BranchTarget {
    pub fn computed<E: Into<Term<Expr>>>(expr: E) -> Term<Self> {
        Self::Computed(expr.into()).into()
    }

    pub fn is_fixed(&self) -> bool {
        !self.is_computed()
    }

    pub fn is_computed(&self) -> bool {
        matches!(self, Self::Computed(_))
    }

    pub fn location<L: Into<Location>>(location: L) -> Term<Self> {
        Self::Location(location.into()).into()
    }
}

impl From<BranchTarget> for Term<BranchTarget> {
    fn from(tgt: BranchTarget) -> Self {
        Term::new(&TRGT, tgt)
    }
}

impl From<Location> for Term<BranchTarget> {
    fn from(t: Location) -> Self {
        Term::new(&TRGT, BranchTarget::Location(t))
    }
}
