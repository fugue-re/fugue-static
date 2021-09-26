use fugue::bv::BitVec;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Neg, Not, Shl, Shr, Sub};

#[derive(Debug, Clone, Hash)]
pub struct CLP {
    base: BitVec,
    step: BitVec,
    card: BitVec,
}

fn get_and_check_size(clp1: &CLP, clp2: &CLP) -> usize {
    if clp1.bits() != clp2.bits() {
        panic!("input CLPs are of different sizes: {} and {}", clp1.bits(), clp2.bits());
    }
    clp1.bits()
}

impl PartialEq for CLP {
    fn eq(&self, other: &Self) -> bool {
        get_and_check_size(&self, &other);

        let slf = self.canonise();
        let oth = other.canonise();

        slf.base == oth.base &&
            slf.step == oth.step &&
            slf.card == oth.card
    }
}
impl Eq for CLP { }

fn cdiv(bv1: &BitVec, bv2: &BitVec) -> BitVec {
    if (bv1 % bv2).is_zero() {
        bv1 / bv2
    } else {
        (bv1 / bv2).succ()
    }
}

fn mul_exact(bv1: &BitVec, bv2: &BitVec) -> BitVec {
    let bits = bv1.bits() + bv2.bits();
    let bv1 = bv1.clone().cast(bits);
    let bv2 = bv2.clone().cast(bits);
    bv1 * bv2
}

fn extract_lh(bv: &BitVec, high: usize, low: usize) -> BitVec {
    assert!(high > low);

    let bv = if low > 0 {
        bv >> (low as u32)
    } else {
        bv.clone()
    };

    bv.unsigned_cast(high - low)
}

fn factor_2s(w: &BitVec) -> (BitVec, usize) {
    let mut fhi = w.bits() - 1;
    let mut flo = 0;

    while fhi != flo {
        let mid = (fhi + flo) / 2;
        let lo_part = extract_lh(w, mid, flo);
        if lo_part.is_zero() {
            flo = mid + 1;
        } else {
            fhi = mid;
        }
    }

    let lo = fhi;
    let hi = w.bits() - 1 + lo;

    (extract_lh(&w, hi, lo), lo)
}

fn lead_1_bit_run(bv: &BitVec, mut hi: usize, mut lo: usize) -> usize {
    while hi != lo {
        let mid = (hi + lo) / 2;
        let hi_part = extract_lh(bv, hi, mid + 1);
        if (!hi_part).is_zero() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    hi + 1
}

fn dom_size(w: usize, b: usize) -> BitVec {
    BitVec::one(w) << (b as u32)
}

fn rshift_step<F>(f: F, p1: &CLP, p2: &CLP, e2: &BitVec, sz1: usize, sz2: usize) -> BitVec
where F: Fn(&BitVec, &BitVec) -> BitVec {
    assert_eq!(p1.bits(), p2.bits());

    let (_, b1_twos) = factor_2s(&p1.base);
    let (_, s1_twos) = factor_2s(&p1.step);

    let s1_div = BitVec::from_usize(s1_twos, sz1) >= *e2;
    let b1_div = BitVec::from_usize(b1_twos, sz1) >= *e2;
    let b1_leading1s = p1.base.leading_ones();

    if (s1_div && p2.card.is_one()) ||
        (s1_div && b1_div) ||
            (s1_div && BitVec::from_u32(b1_leading1s, sz2) >= *e2) {
        f(&p1.step, &e2).gcd(&(f(&p1.base, &(e2 - &p2.step)) - f(&p1.base, &e2)))
    } else {
        BitVec::one(sz1)
    }
}


impl CLP {
    pub fn new<V>(base: V) -> Self
    where
        V: Into<BitVec>,
    {
        let base = base.into();
        let step = BitVec::one(base.bits());
        let card = BitVec::one(base.bits());
        Self::new_with(base, step, card)
    }

    pub fn new_with<U, V, W>(base: U, step: V, card: W) -> Self
    where
        U: Into<BitVec>,
        V: Into<BitVec>,
        W: Into<BitVec>,
    {
        let base = base.into();
        let step = step.into();
        let card = card.into();

        let width = base.bits();

        let slf = Self {
            base,
            step: step.unsigned_cast(width),
            card: card
                .unsigned_cast(width + 1)
                .min(BitVec::max_value_with(width, false).unsigned_cast(width + 1)),
        };

        assert_eq!(slf.base.bits(), width);
        assert_eq!(slf.step.bits(), width);
        assert_eq!(slf.card.bits(), width + 1);

        slf
    }

    pub fn new_with_width<U, V, W>(base: U, step: V, card: W, width: usize) -> Self
    where
        U: Into<BitVec>,
        V: Into<BitVec>,
        W: Into<BitVec>,
    {
        let base = base.into();
        Self::new_with(base.unsigned_cast(width), step, card)
    }

    fn card_from_bounds(base: &BitVec, step: &BitVec, e: &BitVec) -> BitVec {
        let width = base.bits();

        assert_eq!(step.bits(), width);

        if step.is_zero() {
            BitVec::one(width + 1)
        } else {
            let card = (&(e - base) / step).unsigned_cast(width + 1);
            card.succ()
        }
    }

    pub fn infinite<U, V>(b: U, s: V) -> Self
    where
        U: Into<BitVec>,
        V: Into<BitVec>,
    {
        let b = b.into();
        let s = s.into();

        let width = b.bits();

        assert_eq!(width, s.bits());

        if s.is_zero() {
            Self::new(b)
        } else {
            let (div, twos) = factor_2s(&s);
            let step = s / div;
            let base = &b % &step;
            let card = dom_size(width + 1, width - twos);

            Self { base, step, card }
        }
    }

    pub fn is_infinite(&self) -> bool {
        let width = self.bits();
        let ds = dom_size(2 * width + 1, width);
        let mul = mul_exact(&self.card, &self.step);
        mul >= ds
    }

    pub fn bits(&self) -> usize {
        self.base.bits()
    }

    pub fn bottom(width: usize) -> Self {
        Self::new_with(BitVec::zero(width), BitVec::one(width), BitVec::zero(width))
    }

    pub fn top(width: usize) -> Self {
        Self::infinite(BitVec::zero(width), BitVec::one(width))
    }

    pub fn canonise(&self) -> Self {
        let sz = self.bits();
        let two = BitVec::from_u32(2, sz + 1);

        if self.card.is_zero() {
            Self::bottom(sz)
        } else if self.step.is_zero() || self.card.is_one() {
            Self::new_with(self.base.clone(), BitVec::zero(sz), BitVec::one(sz + 1))
        } else if self.card == two {
            let e = &self.base + &self.step;
            if e >= self.base {
                self.clone()
            } else {
                Self::new_with(e, -&self.step, two)
            }
        } else if self.is_infinite() {
            Self::infinite(self.base.clone(), self.step.clone())
        } else {
            self.clone()
        }
    }

    pub fn cardinality(&self) -> BitVec {
        self.canonise().card
    }

    pub fn finite_end(&self) -> Option<BitVec> {
        let n = self.card.unsigned_cast(self.bits());
        if self.card.is_zero() {
            None
        } else {
            Some(&self.base + &(&self.step * &n.pred()))
        }
    }

    pub fn nearest_pred(&self, bv: &BitVec) -> Option<BitVec> {
        assert_eq!(self.bits(), bv.bits());

        let e = self.finite_end()?;
        if self.step.is_zero() {
            Some(self.base.clone())
        } else {
            let diff = bv - &self.base;
            let rm = &diff % &self.step;
            let end = &e - &self.base;

            if self.is_infinite() {
                Some(bv - &rm)
            } else if diff >= end {
                Some(&end + &self.base)
            } else {
                Some(bv - &rm)
            }
        }
    }

    fn nearest_inf_pred(w: &BitVec, base: &BitVec, step: &BitVec) -> BitVec {
        if step.is_zero() {
            base.clone()
        } else {
            let diff = w - base;
            let rm = &diff % &step;
            w - &rm
        }
    }

    pub fn nearest_succ(&self, bv: &BitVec) -> Option<BitVec> {
        (!self).nearest_pred(&!bv).map(|bv| !bv)
    }

    fn nearest_inf_succ(w: &BitVec, base: &BitVec, step: &BitVec) -> BitVec {
        !Self::nearest_inf_pred(&!w, &!base, step)
    }

    pub fn max_elem(&self) -> Option<BitVec> {
        let max_wd = !BitVec::zero(self.bits());
        self.nearest_pred(&max_wd)
    }

    pub fn min_elem(&self) -> Option<BitVec> {
        let min_wd = BitVec::zero(self.bits());
        self.nearest_succ(&min_wd)
    }

    pub fn max_elem_signed(&self) -> Option<BitVec> {
        let half_way = dom_size(self.bits(), self.bits() - 1);
        let max_dom = half_way.pred();
        self.nearest_pred(&max_dom)
    }

    pub fn min_elem_signed(&self) -> Option<BitVec> {
        let half_way = dom_size(self.bits(), self.bits() - 1);
        self.nearest_succ(&half_way)
    }

    pub fn splits_by(&self, bv: &BitVec) -> bool {
        let p = self.canonise();
        let divides = |a: &BitVec, b: &BitVec| -> bool {
            (b % a).is_zero()
        };

        || -> Option<bool> {
            let min_p = p.min_elem()?;
            let e = p.finite_end()?;
            if p.step.is_zero() {
                Some(true)
            } else {
                Some(
                    divides(bv, &p.step) &&
                    (p.base == min_p || divides(bv, &(&e - &p.base)))
                )

            }
        }().unwrap_or(true)
    }

    pub fn min_separation(&self) -> Option<BitVec> {
        let p = self.canonise();
        let e = p.finite_end()?;
        if p.base == e {
            None
        } else {
            Some((&p.base - &e).min(&e - &p.base).min(p.step))
        }
    }

    pub fn contains(&self, bv: &BitVec) -> bool {
        assert_eq!(self.bits(), bv.bits());

        self.nearest_pred(bv)
            .map(|j| bv == &j)
            .unwrap_or(false)
    }

    pub fn is_top(&self) -> bool {
        self.canonise() == Self::top(self.bits())
    }

    pub fn is_bottom(&self) -> bool {
        self.canonise() == Self::bottom(self.bits())
    }

    pub fn subprogression(&self, other: &CLP) -> bool {
        if self.step.is_zero() {
            let diff = &other.base - &self.base;
            let rm = &diff % &other.step;
            rm.is_zero()
        } else if other.step.is_zero() {
            self.step.is_zero() && self.base == other.base
        } else {
            let (coprime1, _) = factor_2s(&self.step);
            let (coprime2, _) = factor_2s(&other.step);
            let pow21 = &self.step / &coprime1;
            let pow22 = &other.step / &coprime2;
            let root1 = &self.base % &pow21;
            let root2 = &other.base % &pow22;
            pow21 >= pow22 && root1 == root2
        }
    }

    pub fn unwrap(&self) -> CLP {
        || -> Option<CLP> {
            let base = self.min_elem()?;
            let e = self.max_elem()?;
            let step = self.step.clone();
            let card = CLP::card_from_bounds(&base, &step, &e);
            Some(CLP {
                base,
                step,
                card,
            })
        }().unwrap_or_else(|| CLP::bottom(self.bits()))
    }

    pub fn unwrap_signed(&self) -> CLP {
        || -> Option<CLP> {
            let base = self.min_elem_signed()?;
            let e = self.max_elem_signed()?;
            let step = self.step.clone();
            let card = CLP::card_from_bounds(&base, &step, &e);
            Some(CLP {
                base,
                step,
                card,
            })
        }().unwrap_or_else(|| CLP::top(self.bits()))
    }

    fn interval_union(a1: &BitVec, b1: &BitVec, a2: &BitVec, b2: &BitVec) -> (BitVec, BitVec) {
        let sz = a1.bits();

        let b1_ = b1 - a1;
        let a2_ = a2 - a1;
        let b2_ = b2 - a1;
        let zero = BitVec::zero(sz);

        if b1_ >= a2_ && b2_ < a2_ {
            (b1.clone(), b1.pred())
        } else if b1_ >= a2_ && b1_ >= b2_ {
            (a1.clone(), b1.clone())
        } else if b1_ >= a2_ {
            (a1.clone(), b2.clone())
        } else if b2_ < b1_ {
            (a2.clone(), b1.clone())
        } else if b2_ < a2_ {
            (a2.clone(), b2.clone())
        } else if (a2_ - b1_) > (zero - b2_) {
            (a2.clone(), b1.clone())
        } else {
            (a1.clone(), b2.clone())
        }
    }

    pub fn translate(&self, bv: &BitVec) -> CLP {
        Self {
            base: &self.base + bv,
            step: self.step.clone(),
            card: self.card.clone(),
        }
    }

    fn common_step(b1: &BitVec, s1: &BitVec, b2: &BitVec, s2: &BitVec) -> BitVec {
        let diff = if b1 > b2 {
            b1 - b2
        } else {
            b2 - b1
        };

        if s1.is_zero() {
            s2.gcd(&diff)
        } else if s2.is_zero() {
            s1.gcd(&diff)
        } else {
            let gcds = s1.gcd(s2);
            gcds.gcd(&diff)
        }
    }

    pub fn subset(&self, other: &CLP) -> bool {
        let sz = get_and_check_size(self, other);
        let mut p1 = self.canonise();
        let mut p2 = other.canonise();

        let nb2 = -&p2.base;
        p1 = p1.translate(&nb2);
        p2 = p2.translate(&nb2);

        let end1 = p1.finite_end();
        let end2 = p2.finite_end();

        match (end1, end2) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(e1), Some(e2)) => {
                let in_bounds = e1 <= e2 && p1.base <= e2;
                let step_and_overlap = Self::common_step(&p1.base, &p1.step, &BitVec::zero(sz), &p2.step) == p2.step;
                let singleton_elem = p1.card.is_one() && p2.contains(&p1.base);
                singleton_elem || (in_bounds && step_and_overlap)
            }
        }

    }

    pub fn intersection(&self, other: &CLP) -> CLP {
        let sz = get_and_check_size(self, other);

        let mut p1 = self.canonise();
        let mut p2 = other.canonise();

        if p1.base < p2.base {
            std::mem::swap(&mut p1, &mut p2);
        }

        let translation = p1.base.clone();
        let translated_p1 = p1.translate(&-&p1.base);
        let translated_p2 = p2.translate(&-&p1.base);

        p1 = translated_p1;
        p2 = translated_p2;

        let p1_infinite = p1.is_infinite();
        let p2_infinite = p2.is_infinite();

        move || -> Option<CLP> {
            let e1 = p1.finite_end()?;
            let e2 = p2.finite_end()?;

            let step = p1.step.lcm(&p2.step);
            if step.is_zero() {
                if p2.step.is_zero() {
                    if p1.contains(&p2.base) {
                        Some(Self::new(p2.base))
                    } else {
                        None
                    }
                } else {
                    if p2.contains(&p1.base) {
                        Some(Self::new(p1.base))
                    } else {
                        None
                    }
                }
            } else {
                let (x, _) = p1.step.diophantine(&p2.step, &p2.base)?;
                let base = &x * &p1.step;
                let min_e = if p1_infinite {
                    e2
                } else if p2_infinite {
                    e1
                } else {
                    e1.min(e2)
                };

                if base <= min_e {
                    let card = Self::card_from_bounds(&base, &step, &min_e);
                    Some(Self::new_with(base, step, card))
                } else {
                    None
                }
            }

        }().unwrap_or_else(|| Self::bottom(sz)).translate(&translation)
    }

    pub fn overlap(&self, other: &Self) -> bool {
        !self.intersection(other).is_bottom()
    }

    pub fn union(&self, other: &Self) -> Self {
        let p1 = self.canonise();
        let p2 = other.canonise();

        let e1 = if let Some(e1) = p1.finite_end() {
            e1
        } else {
            return p2
        };

        let e2 = if let Some(e2) = p2.finite_end() {
            e2
        } else {
            return p1
        };

        let (base, new_e) = Self::interval_union(&p1.base, &e1, &p2.base, &e2);
        let step = Self::common_step(&p1.base, &p1.step, &p2.base, &p2.step);
        let card = Self::card_from_bounds(&base, &step, &new_e);

        Self::new_with(base, step, card)
    }

    fn compute_lsb(lsb1: isize, lsb2: isize, b1: &BitVec, b2: &BitVec) -> isize {
        if lsb1 < lsb2 {
            let interval = extract_lh(b2, (lsb2 - 1) as usize, lsb1 as usize);
            let (w, bit) = factor_2s(&interval);
            if w.is_zero() {
                lsb2
            } else {
                bit as isize + lsb1
            }
        } else if lsb1 > lsb2 {
            let interval = extract_lh(b1, (lsb1 - 1) as usize, lsb2 as usize);
            let (w, bit) = factor_2s(&interval);
            if w.is_zero() {
                lsb1
            } else {
                bit as isize + lsb2
            }
        } else {
            lsb1
        }
    }

    fn compute_msb(msb1: isize, msb2: isize, b1: &BitVec, b2: &BitVec) -> isize {
        if msb1 > msb2 {
            let interval = extract_lh(b2, msb1 as usize, (msb2 + 1) as usize);
            if let Some(b) = interval.leading_one() {
                b as isize + msb2 + 1
            } else {
                msb2
            }
        } else if msb1 < msb2 {
            let interval = extract_lh(b1, msb2 as usize, (msb1 + 1) as usize);
            if let Some(b) = interval.leading_one() {
                b as isize + msb1 + 1
            } else {
                msb1
            }
        } else {
            msb1
        }
    }

    fn compute_range_sep(msb: isize, msb1: isize, msb2: isize, b1: &BitVec, b2: &BitVec) -> isize {
        if msb1 > msb2 {
            if msb == msb1 {
                lead_1_bit_run(b2, msb1 as usize, (msb2 + 1) as usize) as isize
            } else {
                msb + 1
            }
        } else if msb1 < msb2 {
            if msb == msb2 {
                lead_1_bit_run(b1, msb2 as usize, (msb1 + 1) as usize) as isize
            } else {
                msb + 1
            }
        } else {
            -1
        }
    }

    pub fn iter(&self) -> CLPIter {
        CLPIter {
            clp: self.canonise(),
        }
    }
}

impl Neg for CLP {
    type Output = CLP;

    fn neg(self) -> CLP {
        (&self).neg()
    }
}

impl Neg for &'_ CLP {
    type Output = CLP;

    fn neg(self) -> CLP {
        if let Some(bv) = self.finite_end() {
            CLP {
                base: bv.neg(),
                step: self.step.clone(),
                card: self.card.clone(),

            }
        } else {
            CLP::bottom(self.bits())
        }
    }
}

impl Not for CLP {
    type Output = CLP;

    fn not(self) -> CLP {
        (&self).not()
    }
}

impl Not for &'_ CLP {
    type Output = CLP;

    fn not(self) -> CLP {
        if let Some(bv) = self.finite_end() {
            CLP {
                base: bv.not(),
                step: self.step.clone(),
                card: self.card.clone(),

            }
        } else {
            CLP::bottom(self.bits())
        }
    }
}

impl Add for CLP {
    type Output = CLP;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Add for &'_ CLP {
    type Output = CLP;

    fn add(self, rhs: Self) -> Self::Output {
        let sz = get_and_check_size(self, rhs);
        let e1 = if let Some(e1) = self.finite_end() {
            e1
        } else {
            return CLP::bottom(sz)
        };

        let e2 = if let Some(e2) = rhs.finite_end() {
            e2
        } else {
            return CLP::bottom(sz)
        };

        if self.step.is_zero() || self.card.is_one() {
            rhs.translate(&self.base)
        } else if rhs.step.is_zero() || rhs.card.is_one() {
            self.translate(&rhs.base)
        } else if self.is_infinite() || rhs.is_infinite() {
            CLP::infinite(
                &self.base + &rhs.base,
                self.step.gcd(&rhs.step),
            )
        } else {
            let e1_ = &e1 - &self.base;
            let e2_ = &e2 - &rhs.base;
            let e_ = &e1_ + &e2_;
            let base = &self.base + &rhs.base;
            let step = self.step.gcd(&rhs.step);

            if e_ < e1_ {
                CLP::infinite(base, step)
            } else {
                let card = CLP::card_from_bounds(&BitVec::zero(sz), &step, &e_);
                CLP::new_with(base, step, card)
            }

        }
    }
}

impl Sub for CLP {
    type Output = CLP;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

impl Sub for &'_ CLP {
    type Output = CLP;

    fn sub(self, rhs: Self) -> Self::Output {
        self.add(&rhs.neg())
    }
}

impl Mul for CLP {
    type Output = CLP;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl Mul for &'_ CLP {
    type Output = CLP;

    fn mul(self, rhs: Self) -> Self::Output {
        let sz = get_and_check_size(self, rhs);

        let e1 = if let Some(e1) = self.finite_end() {
            e1
        } else {
            return CLP::bottom(sz)
        };

        let e2 = if let Some(e2) = rhs.finite_end() {
            e2
        } else {
            return CLP::bottom(sz)
        };

        if self.step.is_zero() || self.card.is_one() {
            let base = &rhs.base * &self.base;
            let step = &rhs.step * &self.step;
            CLP::new_with(base, step, rhs.card.clone())
        } else if rhs.step.is_zero() || rhs.card.is_one() {
            let base = &self.base * &rhs.base;
            let step = &self.step * &rhs.step;
            CLP::new_with(base, step, self.card.clone())
        } else {
            let base = mul_exact(&self.base, &rhs.base);
            let e_exact = mul_exact(&e1, &e2);
            let step = mul_exact(&self.base, &rhs.step)
                .gcd(&mul_exact(&rhs.base, &self.step)
                     .gcd(&mul_exact(&self.step, &rhs.step)));
            let end_diff = &e_exact - &base;
            let div_res = &end_diff / &step;
            let card_sz = div_res.bits() + 1;
            let card = div_res.cast(card_sz).succ();
            if self.is_infinite() || rhs.is_infinite() {
                CLP::infinite(base.cast(sz), step.cast(sz))
            } else {
                CLP::new_with_width(base, step, card, sz)
            }
        }
    }
}

impl BitAnd for CLP {
    type Output = CLP;

    fn bitand(self, rhs: Self) -> Self::Output {
        (&self).bitand(&rhs)
    }
}

impl BitAnd for &'_ CLP {
    type Output = CLP;

    fn bitand(self, rhs: Self) -> Self::Output {
        let sz = get_and_check_size(self, rhs);
        let card_two = BitVec::from_u64(2, sz + 1);

        let mut p1 = self.canonise();
        let mut p2 = rhs.canonise();

        if p1.cardinality() > p2.cardinality() {
            std::mem::swap(&mut p1, &mut p2);
        }

        || -> Option<CLP> {
            if p1.card.is_zero() || p2.card.is_zero() {
                None // Some(CLP::bottom(sz))
            } else if p1.card.is_one() && p2.card.is_one() {
                Some(CLP::new(&p1.base & &p2.base))
            } else if p1.card.is_one() && p2.card == card_two {
                let e2 = p2.finite_end()?;
                let base = &p1.base & &p2.base;
                let new_e = &p1.base & & &e2;
                let step = &new_e - &base;
                let card = card_two;
                Some(CLP::new_with(base, step, card))
            } else {
                let min_p1 = p1.min_elem()?;
                let max_p1 = p1.max_elem()?;
                let min_p2 = p2.min_elem()?;
                let max_p2 = p2.max_elem()?;

                let (_, twos_in_s1) = factor_2s(&p1.step);
                let (_, twos_in_s2) = factor_2s(&p2.step);

                let lsb1 = if p1.card.is_one() { sz } else { twos_in_s1 } as isize;
                let lsb2 = if p2.card.is_one() { sz } else { twos_in_s2 } as isize;

                let msb1 = (&min_p1 ^ &max_p1).leading_one()
                    .map(|v| v as isize)
                    .unwrap_or(-1);
                let msb2 = (&min_p2 ^ &max_p2).leading_one()
                    .map(|v| v as isize)
                    .unwrap_or(-1);

                let lsb = CLP::compute_lsb(lsb1, lsb2, &min_p1, &min_p2);
                let msb = CLP::compute_msb(msb1, msb2, &min_p1, &min_p2);

                if lsb > msb {
                    let base = min_p1 & min_p2;
                    Some(CLP::new(base))
                } else {
                    let range_sep = CLP::compute_range_sep(msb, msb1, msb2, &min_p1, &min_p2);
                    let mask = if lsb >= range_sep {
                        BitVec::zero(sz)
                    } else {
                        let ones = BitVec::max_value_with((range_sep - lsb) as usize, false);
                        let sized_ones = ones.cast(sz);
                        sized_ones << (lsb as u32)
                    };

                    let safe_lb = !&mask & (&min_p1 & &min_p2);
                    let safe_ub = (&(&max_p1 & &max_p2) | &mask).min(max_p1).min(max_p2);
                    let twos_step = BitVec::one(sz) << (lsb as u32);
                    let step = if msb1 > msb2 && msb == msb1 && range_sep == lsb {
                        p1.step.max(twos_step)
                    } else if msb2 > msb1 && msb == msb2 && range_sep == lsb {
                        p2.step.max(twos_step)
                    } else {
                        twos_step
                    };

                    let b1_and_b2 = &min_p1 & &min_p2;
                    let frac = cdiv(&(&safe_lb - &b1_and_b2), &step);
                    let base = &b1_and_b2 + &(&step * &frac);
                    let card = CLP::card_from_bounds(&base, &step, &safe_ub);
                    Some(CLP::new_with(base, step, card))
                }
            }
        }().unwrap_or_else(|| CLP::bottom(sz))
    }
}

impl BitOr for CLP {
    type Output = CLP;

    fn bitor(self, rhs: Self) -> Self::Output {
        (&self).bitor(&rhs)
    }
}

impl BitOr for &'_ CLP {
    type Output = CLP;

    fn bitor(self, rhs: Self) -> Self::Output {
        !(!self & !rhs)
    }
}

impl BitXor for CLP {
    type Output = CLP;

    fn bitxor(self, rhs: Self) -> Self::Output {
        (&self).bitxor(&rhs)
    }
}

impl BitXor for &'_ CLP {
    type Output = CLP;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let sz = get_and_check_size(self, rhs);
        let two = CLP::new(BitVec::from_u16(2, sz));
        let approx1 = (self & &!rhs) | (&!self & rhs);
        let approx2 = (self + rhs) - ((self & rhs) * two);
        approx1.intersection(&approx2)
    }
}

impl Shl for CLP {
    type Output = CLP;

    fn shl(self, rhs: Self) -> Self::Output {
        (&self).shl(&rhs)
    }
}

impl Shl for &'_ CLP {
    type Output = CLP;

    fn shl(self, rhs: Self) -> Self::Output {
        let sz1 = self.bits();
        let p2 = rhs.canonise();

        || -> Option<CLP> {
            let e1 = self.finite_end()?;
            let min_p2 = p2.min_elem()?;
            let max_p2 = p2.max_elem()?;

            if max_p2 >= BitVec::from_usize(sz1, max_p2.bits()) {
                Some(CLP::top(sz1))
            } else {
                let max_p2i = max_p2.to_u32().expect("shift amount fits u32");
                let base = &self.base << &min_p2;
                let step = if p2.card.is_one() {
                    &self.step << &min_p2
                } else {
                    &self.base.gcd(&self.step) << &min_p2
                };
                let e_no_wrap = e1.unsigned_cast(e1.bits() + max_p2i as usize) << max_p2i;
                let e_width = e_no_wrap.bits();
                let card = if step.is_zero() {
                    BitVec::one(1)
                } else {
                    let base_ext = base.unsigned_cast(e_width);
                    let step_ext = step.unsigned_cast(e_width);
                    CLP::card_from_bounds(&base_ext, &step_ext, &e_no_wrap)
                };
                Some(CLP::new_with(base, step, card))
            }

        }().unwrap_or_else(|| CLP::bottom(sz1))
    }
}

impl Shr for &'_ CLP {
    type Output = CLP;

    fn shr(self, rhs: Self) -> Self::Output {
        let sz1 = self.bits();
        let sz2 = rhs.bits();

        let p1 = self.canonise().unwrap();
        let p2 = rhs.canonise().unwrap();

        || -> Option<CLP> {
            let e1 = p1.finite_end()?;
            let e2 = p2.finite_end()?;
            let base = &p1.base >> &e2;
            if p1.card.is_one() && p2.card.is_one() {
                Some(CLP::new(base))
            } else {
                let step = rshift_step(|u, v| u >> v, &p1, &p2, &e2, sz1, sz2);
                let card = CLP::card_from_bounds(&base, &step, &(&e1 >> &p2.base));
                Some(CLP::new_with(base, step, card))
            }
        }().unwrap_or_else(|| CLP::bottom(sz1))
    }
}

impl Shr for CLP {
    type Output = CLP;

    fn shr(self, rhs: Self) -> Self::Output {
        (&self).shr(&rhs)
    }
}

pub struct CLPIter {
    clp: CLP,
}

impl Iterator for CLPIter {
    type Item = BitVec;

    fn next(&mut self) -> Option<Self::Item> {
        if self.clp.card.is_zero() {
            None
        } else {
            let mut nbase = &self.clp.base + &self.clp.step;
            std::mem::swap(&mut self.clp.base, &mut nbase);

            self.clp.card = self.clp.card.pred();

            Some(nbase)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(size) = self.clp.card.to_usize() {
            (size, Some(size))
        } else {
            (0, None) // is usize::MAX correct?
        }
    }
}
