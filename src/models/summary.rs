use fugue::bv::BitVec;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StackShift {
    Positive(usize),
    Negative(usize),
}

impl Default for StackShift {
    fn default() -> Self {
        Self::Positive(0)
    }
}

impl StackShift {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Positive(0) | Self::Negative(0))
    }

    pub fn is_negative(&self) -> bool {
        matches!(self, Self::Negative(n) if *n > 0)
    }

    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Positive(n) if *n > 0)
    }

    fn parts(&self) -> (bool, usize) {
        match self {
            Self::Positive(v) => (true, *v),
            Self::Negative(v) => (false, *v),
        }
    }

    pub fn absolute_value(&self) -> usize {
        match self {
            Self::Positive(v) | Self::Negative(v) => *v,
        }
    }

    pub fn apply(&self, value: BitVec) -> BitVec {
        let sz = value.bits();
        let (s, v) = self.parts();
        let bv = BitVec::from_usize(v, sz);
        if s {
            value + bv
        } else {
            value - bv
        }
    }

    pub fn checked_apply(&self, value: BitVec) -> Option<BitVec> {
        let sz = value.bits();
        let (s, v) = self.parts();
        let bv = BitVec::from_usize(v, sz);
        if s {
            let nv = &value + &bv;
            if nv < value {
                None
            } else {
                Some(nv)
            }
        } else {
            let nv = &value - &bv;
            if nv > value {
                None
            } else {
                Some(nv)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Summary<Var> {
    defs: Vec<Var>,
    uses: Vec<Var>,
    stack_access: usize, // range of access in caller's frame
    stack_shift: StackShift, // adjustment made to SP on return
}

impl<Var> Default for Summary<Var> {
    fn default() -> Self {
        Self {
            defs: Vec::new(),
            uses: Vec::new(),
            stack_access: 0,
            stack_shift: StackShift::default(),
        }
    }
}
