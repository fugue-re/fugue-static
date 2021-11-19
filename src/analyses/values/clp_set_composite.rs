use super::clp::CLP;
use super::finite_set::FiniteSet;

use fugue::bv::BitVec;

#[derive(Debug, Clone)]
pub enum CLPFiniteSet {
    C(CLP),
    S(FiniteSet),
}

const FINITE_THRESHOLD_SIZE: usize = 10;

impl CLPFiniteSet {
    pub fn is_clp(&self) -> bool {
        matches!(self, Self::C(_))
    }

    pub fn is_set(&self) -> bool {
        matches!(self, Self::S(_))
    }

    pub fn into_set(self) -> Self {
        if let Self::C(clp) = self {
            let bits = clp.bits();
            Self::S(FiniteSet::from_iter_with(clp.iter(), bits, false))
        } else {
            self
        }
    }

    pub fn into_clp(self) -> Self {
        if let Self::S(set) = self {
            let bits = set.bits();
            Self::C(CLP::from_iter_with(set.into_iter(), bits, false))
        } else {
            self
        }
    }

    pub fn singleton(bv: BitVec) -> Self {
        Self::S(FiniteSet::singleton(bv))
    }

    pub fn canonise(self) -> Self {
        match self {
            Self::C(ref clp) => {
                let c1 = clp.cardinality();
                let c2 = BitVec::from_usize(FINITE_THRESHOLD_SIZE, c1.bits());
                if c1 > c2 { self } else { self.into_set() }
            },
            Self::S(ref set) => {
                let c1 = set.cardinality();
                let c2 = BitVec::from_usize(FINITE_THRESHOLD_SIZE, c1.bits());
                if c1 > c2 { self.into_clp() } else { self }
            },
        }
    }

    pub fn top(size: usize) -> Self {
        Self::C(CLP::top(size)).canonise()
    }

    pub fn bottom(size: usize) -> Self {
        Self::S(FiniteSet::bottom(size))
    }
}
