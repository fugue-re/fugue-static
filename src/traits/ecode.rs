use fugue::ir::il::ecode::{BranchTarget, ECode, Expr, Location, Stmt};
use std::fmt::{self, Display};

#[derive(Debug, Clone)]
pub enum ECodeTarget {
    IntraIns(Location, bool),
    IntraBlk(Location, bool),
    InterBlk(BranchTarget),
    InterSub(BranchTarget),
    InterRet(BranchTarget, bool),
    Intrinsic,
    Unresolved,
}

impl ECodeTarget {
    pub fn ends_block(&self) -> bool {
        match self {
            Self::Unresolved | Self::InterBlk(_) | Self::InterRet(_, true) => true,
            _ => false,
        }
    }
}

impl Display for ECodeTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unresolved => write!(f, "unresolved flow"),
            Self::IntraIns(loc, _) => write!(f, "intra-instruction flow to {}", loc),
            Self::IntraBlk(loc, _) => write!(f, "intra-block flow to {}", loc),
            Self::InterBlk(tgt) => write!(f, "inter-block flow to {}", tgt),
            Self::InterSub(tgt) => write!(f, "inter-sub-routine flow to {}", tgt),
            Self::InterRet(tgt, _last) => write!(f, "inter-sub-routine flow to {} via return", tgt),
            Self::Intrinsic => write!(f, "intrinsic flow"),
        }
    }
}

pub trait ECodeExt {
    fn branch_targets(&self) -> Vec<(usize, ECodeTarget)>;
}

impl ECodeExt for ECode {
    fn branch_targets(&self) -> Vec<(usize, ECodeTarget)> {
        let address = self.address();
        let naddress = self.address() + self.length();
        let op_count = self.operations().len();

        let mut targets = Vec::new();

        let is_local = |loc: &Location| -> bool { *loc.address() == address };
        let is_fall = |loc: &Location| -> bool { *loc.address() == naddress };

        let nlocation = |i: usize| -> Location { if i >= op_count {
            Location::new(naddress.clone(), i - op_count)
         } else {
            Location::new(address.clone(), i)
         }};

         let nbranch = |i: usize, tgt: &BranchTarget, targets: &mut Vec<(usize, ECodeTarget)>| {
            match tgt {
                BranchTarget::Computed(exp) => if let Expr::Val(ref bv) = exp {
                    if let Some(off) = bv.to_u64() {
                        if off == address.offset() {
                            targets.push((i, ECodeTarget::IntraIns(Location::new(address.clone(), 0), false)));
                        } else if off == naddress.offset() {
                            targets.push((i, ECodeTarget::IntraBlk(Location::new(naddress.clone(), 0), false)));
                        } else {
                            targets.push((i, ECodeTarget::InterBlk(tgt.clone())));
                        }
                    } else {
                        targets.push((i, ECodeTarget::Unresolved));
                    }
                } else {
                    targets.push((i, ECodeTarget::Unresolved));
                },
                BranchTarget::Location(loc) => if is_local(loc) {
                    targets.push((i, ECodeTarget::IntraIns(loc.clone(), false)));
                } else if is_fall(loc) {
                    targets.push((i, ECodeTarget::IntraBlk(loc.clone(), false)));
                } else {
                    targets.push((i, ECodeTarget::InterBlk(tgt.clone())));
                }
            }
         };

         let nfall = |i: usize, fall: Location, targets: &mut Vec<(usize, ECodeTarget)>| {
            targets.push((i, if is_local(&fall) {
                ECodeTarget::IntraIns(fall, true)
            } else {
                ECodeTarget::IntraBlk(fall, true)
            }));
         };

        for (i, stmt) in self.operations().iter().enumerate() {
            let next = nlocation(i + 1);
            match stmt {
                Stmt::Branch(tgt) => {
                    nbranch(i, tgt, &mut targets);
                },
                Stmt::CBranch(_, tgt) => {
                    nbranch(i, tgt, &mut targets);
                    nfall(i, next, &mut targets);
                },
                Stmt::Call(tgt, _) => {
                    targets.push((i, ECodeTarget::InterSub(tgt.clone())));
                    nfall(i, next, &mut targets);
                },
                Stmt::Return(tgt) => {
                    targets.push((i, ECodeTarget::InterRet(tgt.clone(), i + 1 == op_count)));
                },
                Stmt::Intrinsic(_, _) => {
                    targets.push((i, ECodeTarget::Intrinsic));
                    nfall(i, next, &mut targets);
                },
                _  => if i + 1 == op_count {
                    nfall(i, next, &mut targets);
                },
            }
        }

        targets
    }
}
