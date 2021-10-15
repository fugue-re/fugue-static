use fugue::bv::BitVec;
use std::collections::BTreeSet;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiniteSet {
    bits: usize,
    set: BTreeSet<BitVec>,
    signed: bool,
}

impl FiniteSet {
    pub fn new(bits: usize) -> Self {
        Self {
            set: Default::default(),
            signed: false,
            bits,
        }
    }

    pub fn singleton(bv: BitVec) -> Self {
        let bits = bv.bits();
        Self {
            set: {
                let mut set = BTreeSet::new();
                set.insert(bv);
                set
            },
            signed: false,
            bits,
        }
    }

    pub fn from_iter_with<I>(bvs: I, bits: usize, signed: bool) -> Self
    where I: IntoIterator<Item=BitVec> {
        Self {
            set: bvs.into_iter().map(|bv| if signed {
                bv.signed().cast(bits)
            } else {
                bv.unsigned().cast(bits)
            }).collect(),
            signed: false,
            bits,
        }
    }

    pub fn signed(self) -> Self {
        Self {
            set: self.set.into_iter().map(|bv| bv.signed()).collect(),
            signed: true,
            bits: self.bits,
        }
    }

    pub fn min_elem(&self) -> Option<BitVec> {
        // nightly:
        // self.set.first().cloned()
        if self.signed {
            let smallest_s = self.set.iter().next()?;
            Some(self.set.iter()
                .skip_while(|bv| bv.msb())
                .next()
                .unwrap_or(smallest_s)
                .clone())
        } else {
            self.set.iter().next().cloned()
        }
    }

    pub fn max_elem(&self) -> Option<BitVec> {
        // nightly:
        // self.set.last().cloned()
        if self.signed {
            let largest_s = self.set.iter().rev().next()?;
            Some(self.set.iter()
                 .take_while(|bv| bv.msb()) // largest as neg
                 .last()
                 .unwrap_or(largest_s)
                 .clone())
        } else {
            self.set.iter().rev().next().cloned()
        }
    }

    pub fn min_elem_signed(&self) -> Option<BitVec> {
        if self.signed {
            self.set.iter().next().cloned()
        } else {
            let smallest_u = self.set.iter().next()?;
            Some(self.set.iter()
                .rev()
                .take_while(|bv| bv.msb()) // next neg
                .next()
                .unwrap_or(smallest_u)
                .clone())
        }
    }

    pub fn max_elem_signed(&self) -> Option<BitVec> {
        // nightly:
        // self.set.last().cloned()
        if self.signed {
            self.set.iter().rev().next().cloned()
        } else {
            let largest_u = self.set.iter().rev().next()?;
            Some(self.set.iter()
                 .rev()
                 .skip_while(|bv| bv.msb()) // skip all neg
                 .next()
                 .unwrap_or(largest_u)
                 .clone())
        }
    }

    pub fn contains(&self, bv: &BitVec) -> bool {
        assert_eq!(self.bits(), bv.bits());
        self.set.contains(bv)
    }

    fn lift_unop<F: Fn(&BitVec) -> BitVec>(&self, f: F) -> FiniteSet {
        FiniteSet {
            set: self.set.iter().map(f).collect(),
            signed: self.signed,
            bits: self.bits,
        }
    }

    fn lift_binop<F: Fn(&BitVec, &BitVec) -> BitVec>(&self, rhs: &FiniteSet, f: F) -> FiniteSet {
        self.lift2(rhs, |ls, rs| ls.iter().zip(rs.iter()).map(|(l, r)| f(l, r)).collect())
    }

    fn lift2<F>(&self, rhs: &FiniteSet, f: F) -> FiniteSet
    where F: Fn(&BTreeSet<BitVec>, &BTreeSet<BitVec>) -> BTreeSet<BitVec> {
        assert_eq!(self.bits(), rhs.bits());
        assert_eq!(self.signed, rhs.signed);

        FiniteSet {
            set: f(&self.set, &rhs.set),
            signed: self.signed,
            bits: self.bits,
        }
    }

    pub fn intersection(&self, rhs: &FiniteSet) -> FiniteSet {
        self.lift2(rhs, |ls, rs| ls.intersection(rs).map(ToOwned::to_owned).collect())
    }

    pub fn union(&self, rhs: &FiniteSet) -> FiniteSet {
        self.lift2(rhs, |ls, rs| ls.union(rs).map(ToOwned::to_owned).collect())
    }

    pub fn signed_div(&self, rhs: &FiniteSet) -> FiniteSet {
        let mut res = self.lift2(rhs, |ls, rs| {
            ls.iter().zip(rs.iter()).filter_map(|(l, r)| if r.is_zero() {
                Some(l.signed_div(r).signed())
            } else {
                None
            }).collect()
        });
        res.signed = true;
        res
    }

    pub fn signed_rem(&self, rhs: &FiniteSet) -> FiniteSet {
        let mut res = self.lift2(rhs, |ls, rs| {
            ls.iter().zip(rs.iter()).filter_map(|(l, r)| if r.is_zero() {
                Some(l.signed_rem(r).signed())
            } else {
                None
            }).collect()
        });
        res.signed = true;
        res
    }

    pub fn signed_shr(&self, rhs: &FiniteSet) -> FiniteSet {
        let mut res = self.lift_binop(rhs, |ls, rs| ls.signed_shr(rs).signed());
        res.signed = true;
        res
    }

    pub fn nearest_pred(&self, bv: &BitVec) -> Option<BitVec> {
        let v = if self.signed && !bv.is_signed() {
            bv.clone().signed()
        } else {
            bv.clone()
        };

        self.set.iter().fold(None, |mmin, bv_| {
            let diff = &v - bv_;
            if let Some(m) = mmin {
                Some(if m <= diff { m } else { diff })
            } else {
                Some(diff)
            }
        })
    }

    pub fn nearest_succ(&self, bv: &BitVec) -> Option<BitVec> {
        (!self).nearest_pred(&!bv).map(|bv| !bv)
    }

    pub fn splits_by(&self, bv: &BitVec) -> bool {
        if bv.is_zero() {
            return false
        }

        self.set.iter().skip(1)
            .zip(self.set.iter()).map(|(l, r)| l - r)
            .all(|diff| (&diff % bv).is_zero())
    }

    pub fn extract(&self, hi: Option<usize>, lo: Option<usize>) -> FiniteSet {
        let lo = lo.unwrap_or(0);
        let hi = hi.unwrap_or(self.bits - 1);
        let nbits = hi - lo + 1;

        let mut res = self.lift_unop(|bv| {
            let v = if lo > 0 {
                bv.shr(lo as u32)
            } else {
                bv.clone()
            };

            if self.signed {
                v.signed().cast(nbits)
            } else {
                v.cast(nbits)
            }
        });

        res.bits = nbits;
        res
    }

    pub fn signed_cast(&self, bits: usize) -> FiniteSet {
        let mut res = self.lift_unop(|bv| bv.signed_cast(bits));
        res.signed = true;
        res.bits = bits;
        res
    }

    pub fn unsigned_cast(&self, bits: usize) -> FiniteSet {
        let mut res = self.lift_unop(|bv| bv.unsigned_cast(bits));
        res.signed = false;
        res.bits = bits;
        res
    }

    pub fn concat(&self, rhs: &FiniteSet) -> FiniteSet {
        self.lift_binop(rhs, |l, r| {
            let nl = l.unsigned_cast(self.bits() * 2);
            let nr = r.unsigned_cast(self.bits() * 2);
            let nv = (nl << (self.bits() as u32)) | nr;
            if self.signed {
                nv.signed()
            } else {
                nv.unsigned()
            }
        })
    }

    pub fn precedes(&self, rhs: &FiniteSet) -> bool {
        assert_eq!(self.bits(), rhs.bits());
        assert_eq!(self.signed, rhs.signed);

        self.set.is_subset(&rhs.set)
    }

    pub fn join(&self, rhs: &FiniteSet) -> FiniteSet {
        self.union(rhs)
    }

    pub fn meet(&self, rhs: &FiniteSet) -> FiniteSet {
        self.intersection(rhs)
    }

    pub fn bottom(bits: usize) -> FiniteSet {
        FiniteSet {
            set: Default::default(),
            signed: false,
            bits,
        }
    }

    pub fn cardinality(&self) -> BitVec {
        BitVec::from_usize(self.set.len(), self.bits)
    }

    pub fn bits(&self) -> usize {
        self.bits
    }
}

impl Neg for &'_ FiniteSet {
    type Output = FiniteSet;

    fn neg(self) -> Self::Output {
        self.lift_unop(|bv| -bv)
    }
}

impl Add for &'_ FiniteSet {
    type Output = FiniteSet;

    fn add(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l + r)
    }
}

impl Sub for &'_ FiniteSet {
    type Output = FiniteSet;

    fn sub(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l - r)
    }
}

impl Mul for &'_ FiniteSet {
    type Output = FiniteSet;

    fn mul(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l * r)
    }
}

impl Div for &'_ FiniteSet {
    type Output = FiniteSet;

    fn div(self, rhs: Self) -> Self::Output {
        self.lift2(rhs, |ls, rs| {
            ls.iter().zip(rs.iter()).filter_map(|(l, r)| if r.is_zero() {
                Some(l / r)
            } else {
                None
            }).collect()
        })
    }
}

impl Rem for &'_ FiniteSet {
    type Output = FiniteSet;

    fn rem(self, rhs: Self) -> Self::Output {
        self.lift2(rhs, |ls, rs| {
            ls.iter().zip(rs.iter()).filter_map(|(l, r)| if r.is_zero() {
                Some(l % r)
            } else {
                None
            }).collect()
        })
    }
}

impl Shl for &'_ FiniteSet {
    type Output = FiniteSet;

    fn shl(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l << r)
    }
}

impl Shr for &'_ FiniteSet {
    type Output = FiniteSet;

    fn shr(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l >> r)
    }
}

impl Not for &'_ FiniteSet {
    type Output = FiniteSet;

    fn not(self) -> Self::Output {
        self.lift_unop(|bv| !bv)
    }
}

impl BitAnd for &'_ FiniteSet {
    type Output = FiniteSet;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l & r)
    }
}

impl BitOr for &'_ FiniteSet {
    type Output = FiniteSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l | r)
    }
}

impl BitXor for &'_ FiniteSet {
    type Output = FiniteSet;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.lift_binop(rhs, |l, r| l ^ r)
    }
}
