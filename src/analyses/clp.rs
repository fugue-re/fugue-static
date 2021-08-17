use fugue::bv::sized::BitVec;

pub struct CLP<const N: usize> {
    lower_bound: BitVec<N>,
    upper_bound: BitVec<N>,
    step: BitVec<N>,
}

impl<const N: usize> CLP<N> {
    fn max_p() -> BitVec<N> {
        BitVec::max_value(true)
    }

    fn max_n() -> BitVec<N> {
        BitVec::min_value(true)
    }

    fn max_d() -> BitVec<N> {
        BitVec::max_value(false)
    }

    pub fn new(signed: bool) -> Self {
        Self {
            lower_bound: BitVec::min_value(signed),
            upper_bound: BitVec::max_value(signed),
            step: BitVec::one(),
        }
    }

    #[inline(always)]
    pub fn signed() -> Self {
        Self::new(true)
    }

    #[inline(always)]
    pub fn unsigned() -> Self {
        Self::new(false)
    }
}
