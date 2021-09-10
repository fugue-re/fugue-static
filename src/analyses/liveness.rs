use crate::analyses::fixed_point::FixedPointBackward;
use crate::models::block::Block;
use crate::traits::collect::ValueRefCollector;
use crate::traits::*;

use fugue::ir::il::ecode::Var;

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::ops::{Deref, DerefMut, Sub};

#[derive(Default)]
struct OnlyLocals<'ecode>(BTreeSet<&'ecode Var>);

impl<'ecode> Deref for OnlyLocals<'ecode> {
    type Target = BTreeSet<&'ecode Var>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ecode> DerefMut for OnlyLocals<'ecode> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'ecode> ValueRefCollector<'ecode, Var> for OnlyLocals<'ecode> {
    #[inline(always)]
    fn insert_ref(&mut self, var: &'ecode Var) {
        if var.space().is_unique() {
            self.0.insert_ref(var);
        }
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        self.0.merge_ref(&mut other.0)
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        self.0.retain_difference_ref(&other.0)
    }
}

#[derive(Default)]
pub struct LocalsLiveness;

impl<'ecode> FixedPointBackward<'ecode, BTreeSet<&'ecode Var>> for LocalsLiveness {
    type Err = Infallible;

    fn join(&mut self, mut current: BTreeSet<&'ecode Var>, next: &BTreeSet<&'ecode Var>) -> Result<BTreeSet<&'ecode Var>, Self::Err> {
        current.extend(next.iter());
        Ok(current)
    }

    fn transfer(&mut self, block: &'ecode Block, current: Option<BTreeSet<&'ecode Var>>) -> Result<BTreeSet<&'ecode Var>, Self::Err> {
        let mut kill = BTreeSet::new();
        let mut gen = BTreeSet::new();

        let mut lkill = OnlyLocals(BTreeSet::new());
        let mut lgen = OnlyLocals(BTreeSet::new());

        for (lvar, rvars) in block.phis() {
            kill.insert_ref(lvar);
            for rvar in rvars {
                gen.insert_ref(rvar);
            }
        }

        for op in block.operations() {
            op.defined_variables_with(&mut lkill);
            op.used_variables_with(&mut lgen);

            gen.extend(lgen.difference(&*lkill));
            kill.append(&mut *lkill);

            lgen.clear();
        }

        Ok(if let Some(out) = current {
            out.sub(&kill).append(&mut gen);
            out
        } else {
            gen
        })
    }
}
