use fugue::bv::BitVec;

use std::borrow::Borrow;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

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

fn add_exact(bv1: &BitVec, bv2: &BitVec) -> BitVec {
    let bits = 1 + bv1.bits().max(bv2.bits());
    let bv1 = bv1.clone().cast(bits);
    let bv2 = bv2.clone().cast(bits);
    bv1 + bv2
}

fn mul_exact(bv1: &BitVec, bv2: &BitVec) -> BitVec {
    let bits = bv1.bits() + bv2.bits();
    let bv1 = bv1.clone().cast(bits);
    let bv2 = bv2.clone().cast(bits);
    bv1 * bv2
}

fn shl_exact(bv: &BitVec, bits: usize) -> BitVec {
    bv.unsigned_cast(bv.bits() + bits) << bits as u32
}

fn extract_lh(bv: &BitVec, high: usize, low: usize) -> BitVec {
    assert!(high >= low);

    let bv = if low > 0 {
        bv >> (low as u32)
    } else {
        bv.clone()
    };

    bv.unsigned_cast(1 + high - low)
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

    pub fn range<U, V>(start: U, end: V) -> Self
    where
        U: Into<BitVec>,
        V: Into<BitVec>,
    {
        let mut start = start.into();
        let mut end = end.into();

        let signed = start.is_signed() || end.is_signed();
        let bits = start.bits().max(end.bits());

        if start.bits() != bits {
            start = if signed { start.signed_cast(bits) } else { start.unsigned_cast(bits) }
        }

        if end.bits() != bits {
            end = if signed { end.signed_cast(bits) } else { end.unsigned_cast(bits) }
        }

        if end < start { // empty
            Self::bottom(bits)
        } else {
            let cardn = (&end - &start).abs().unsigned_cast(1 + bits) + BitVec::one(1 + bits);
            Self::new_with(start, BitVec::one(bits), cardn)
        }
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

    // See: Executable Analysis using Abstract Interpretation with Circular Linear Progressions, Section 5.
    // NOTE: we split on unsigned bounds, rather than signed bounds from the paper.
    pub fn split_unsigned(&self) -> (CLP, CLP) {
        let curr = self.canonise();

        let lb = curr.base.clone();
        let ub = curr.finite_end().unwrap();

        if lb < ub {
            (curr.clone(), CLP::bottom(curr.bits()))
        } else {
            let max_p = BitVec::max_value_with(curr.bits(), false).unsigned();

            let p_upper = &lb + &(&curr.step * &(&max_p - &lb).signed_div(&curr.step));
            let p_lower = lb;

            let p_card = (&(&p_upper - &p_lower) / &curr.step).cast(curr.bits() + 1) + BitVec::one(curr.bits() + 1);
            let p_step = curr.step.clone();

            let p = Self {
                base: p_lower,
                step: p_step,
                card: p_card,
            };

            let q_lower = &ub - &(&curr.step * &ub.signed_div(&curr.step));
            let q_upper = ub;

            let q_card = (&(&q_upper - &q_lower) / &curr.step).cast(curr.bits() + 1) + BitVec::one(curr.bits() + 1);

            let q_step = curr.step.clone();

            let q = Self {
                base: q_lower,
                step: q_step,
                card: q_card,
            };

            (p, q)
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
            let base = self.base.clone().unsigned();
            if e >= base {
                Self {
                    base,
                    step: self.step.clone(),
                    card: self.card.clone(),
                }
            } else {
                Self::new_with(e, -&self.step, two)
            }
        } else if self.is_infinite() {
            Self::infinite(self.base.clone().unsigned(), self.step.clone())
        } else {
            Self {
                base: self.base.clone().unsigned(),
                step: self.step.clone(),
                card: self.card.clone(),
            }
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

        p1.base = p1.base.unsigned();
        p2.base = p2.base.unsigned();

        if p1.base < p2.base {
            std::mem::swap(&mut p1, &mut p2);
        }

        let translation = p1.base.clone();

        println!("trans: {}", translation);

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
                println!("p2.base: {}", p2.base);
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
                    println!("base: {}", base);
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

    fn split_at_n(&self, n: &BitVec) -> (CLP, CLP) {
        let p = self.canonise();
        let card1 = CLP::card_from_bounds(&p.base, &p.step, n)
            .min(p.card.clone());
        let card2 = &p.card - &card1;
        let p2_base = Self::nearest_inf_succ(&n.succ(), &p.base, &p.step);
        (
            CLP { base: p.base, step: p.step.clone(), card: card1 },
            CLP { base: p2_base, step: p.step, card: card2 },
        )
    }

    fn extract_exact(&self, width: usize) -> (CLP, CLP) {
        let sz = self.bits();
        assert!(width >= sz);

        let lastn = BitVec::max_value_with(sz, false);
        let (p1, p2) = self.split_at_n(&lastn);
        (
            CLP::new_with_width(p1.base, p1.step, p1.card, width),
            CLP::new_with_width(p2.base, p2.step, p2.card, width),
        )
    }

    fn extract_lo(&self, lo: usize) -> CLP {
        let p = self.canonise();
        let sz = p.bits();

        assert!(lo < sz);

        let rsz = sz - lo;
        if lo == 0 {
            p
        } else {
            || -> Option<CLP> {
                let e = p.finite_end()?;
                let base = extract_lh(&p.base, sz - 1, lo);

                let base_mod2_lo = extract_lh(&p.base, lo - 1, 0);
                let step_mod2_lo = extract_lh(&p.step, lo - 1, 0);

                let max_step_eff = add_exact(&base_mod2_lo, &mul_exact(&step_mod2_lo, &p.card.pred()));
                let carry_bound = dom_size(max_step_eff.bits(), lo);
                if max_step_eff < carry_bound {
                    Some(CLP::new_with(base, extract_lh(&p.step, sz, lo), p.card))
                } else {
                    let e = extract_lh(&e, sz, lo);
                    let step = BitVec::one(rsz);
                    let card = CLP::card_from_bounds(&base, &step, &e);
                    Some(CLP::new_with(base, step, card))
                }
            }().unwrap_or_else(|| CLP::bottom(rsz))
        }
    }

    fn extract_hi(&self, hi: Option<usize>, signed: bool) -> CLP {
        let sz = self.bits();
        let hiv = hi.unwrap_or_else(|| sz - 1);
        if !signed && hiv + 1 >= sz {
            let (r1, r2) = self.extract_exact(hiv + 1);
            r1.union(&r2)
        } else if !signed {
            CLP::new_with_width(self.base.clone(), self.step.clone(), self.card.clone(), hiv + 1)
        } else {
            || -> Option<CLP> {
                let e = self.finite_end()?;
                let hsz = hiv + 1;
                if self.is_infinite() {
                    Some(CLP::infinite(self.base.unsigned_cast(hsz), self.step.unsigned_cast(hiv + 1)))
                } else {
                    let neg_min = BitVec::one(sz) << BitVec::from_usize(sz - 1, sz);
                    let pos_max = neg_min.pred();
                    if self.contains(&neg_min) && self.contains(&pos_max) && (self.base != neg_min || e != pos_max) {
                        Some(CLP::top(hsz))
                    } else {
                        let e_ = &e - &self.base;
                        let new_e = e_.unsigned_cast(hsz);
                        let base = if signed {
                            self.base.signed_cast(hsz)
                        } else {
                            self.base.unsigned_cast(hsz)
                        };
                        let step = self.step.unsigned_cast(hsz);
                        if new_e < e_ {
                            Some(CLP::infinite(base, step))
                        } else {
                            let card = CLP::card_from_bounds(&BitVec::zero(hsz), &step, &new_e);
                            Some(CLP::new_with(base, step, card))
                        }

                    }
                }
            }().unwrap_or_else(|| CLP::bottom(hiv + 1))
        }
    }

    pub fn extract(&self, hi: Option<usize>, lo: Option<usize>, signed: bool) -> CLP {
        let lo = lo.unwrap_or(0);
        let hi = hi.map(|hi| hi - lo);
        self.extract_lo(lo).extract_hi(hi, signed)
    }

    pub fn signed_cast(&self, bits: usize) -> CLP {
        self.extract(Some(bits - 1), None, true)
    }

    pub fn unsigned_cast(&self, bits: usize) -> CLP {
        self.extract(Some(bits - 1), None, false)
    }

    pub fn concat(&self, rhs: &CLP) -> CLP {
        let sz1 = self.bits();
        let sz2 = rhs.bits();
        let sz = sz1 + sz2;
        let p1_base = shl_exact(&self.base, sz2);
        let p1_step = shl_exact(&self.step, sz2);
        let p1 = CLP::new_with(p1_base, p1_step, self.card.clone());
        let p2 = rhs.unsigned_cast(sz);
        p1 + p2
    }

    pub fn signed_shr<Rhs: Borrow<CLP>>(&self, rhs: Rhs) -> CLP {
        let rhs = rhs.borrow();
        let sz1 = self.bits();
        let sz2 = rhs.bits();

        let zero = BitVec::zero(sz1);
        let p1 = self.canonise().unwrap_signed();
        let p2 = rhs.canonise().unwrap();

        || -> Option<CLP> {
            let e1 = p1.finite_end()?;
            let e2 = p2.finite_end()?;

            if p1.card.is_zero() && p2.card.is_one() {
                let base = p1.base.signed_shr(&p2.base);
                Some(CLP::new(base))
            } else {
                let p1_base_signed = p1.base.clone().signed();
                let base = if p1_base_signed >= zero {
                    p1_base_signed.signed_shr(&e2)
                } else {
                    p1_base_signed.signed_shr(&p2.base)
                };

                let step = rshift_step(BitVec::signed_shr, &p1, &p2, &e2, sz1, sz2);

                let e1_signed = e1.signed();
                let new_e = if e1_signed >= zero {
                    e1_signed.signed_shr(&p2.base)
                } else {
                    e1_signed.signed_shr(&e2)
                };

                let card = CLP::card_from_bounds(&base, &step, &new_e);

                Some(CLP::new_with(base, step, card))
            }
        }().unwrap_or_else(|| CLP::bottom(sz1))
    }

    pub fn signed_div(&self, rhs: &CLP) -> CLP {
        let sz = get_and_check_size(self, rhs);
        || -> Option<CLP> {
            let min_e1 = self.min_elem()?;
            let max_e1 = self.max_elem()?;
            let min_e2 = rhs.min_elem()?;
            let max_e2 = rhs.max_elem()?;

            if rhs.contains(&BitVec::zero(sz)) {
                None // or panic!?
            } else if self.cardinality().is_one() && rhs.cardinality().is_one() {
                Some(CLP::new(self.base.signed_div(&rhs.base)))
            } else {
                let min_max = min_e1.signed_div(&max_e2);
                let min_min = min_e1.signed_div(&min_e2);
                let max_max = max_e1.signed_div(&max_e2);
                let max_min = max_e1.signed_div(&min_e2);

                let base = (&min_max)
                    .min(&min_min)
                    .min(&max_max)
                    .min(&max_min)
                    .clone();

                let e = min_max
                    .max(min_min)
                    .max(max_max)
                    .max(max_min);

                let step = BitVec::one(sz);
                let card = CLP::card_from_bounds(&base, &step, &e);
                Some(CLP { base, step, card })
            }
        }().unwrap_or_else(|| CLP::bottom(sz))
    }

    pub fn signed_rem(&self, rhs: &CLP) -> CLP {
        self.sub(&(&self.signed_div(&rhs)).mul(rhs))
    }

    pub fn widen_join(&self, rhs: &CLP) -> CLP {
        assert!(self.subset(rhs));
        if self == rhs {
            self.clone()
        } else {
            CLP::infinite(rhs.base.clone(), rhs.step.clone())
        }
    }

    pub fn true_() -> CLP {
        CLP::new(BitVec::one(8))
    }

    pub fn false_() -> CLP {
        CLP::new(BitVec::zero(8))
    }

    pub fn bool_top() -> CLP {
        CLP::top(8)
    }

    pub fn bool_bottom() -> CLP {
        CLP::top(8)
    }

    pub fn equal(&self, rhs: &CLP) -> CLP {
        let sz1 = self.cardinality();
        let sz2 = rhs.cardinality();

        if sz1.is_zero() || sz2.is_zero() {
            CLP::bool_bottom()
        } else if sz1.is_one() && sz2.is_one() {
            if self == rhs { // canonised
                CLP::true_()
            } else {
                CLP::false_()
            }
        } else if self.overlap(&rhs) {
            CLP::bool_top()
        } else {
            CLP::false_()
        }
    }

    pub fn not_equal(&self, rhs: &CLP) -> CLP {
        let sz1 = self.cardinality();
        let sz2 = rhs.cardinality();

        if sz1.is_zero() || sz2.is_zero() {
            CLP::bool_bottom()
        } else if sz1.is_one() && sz2.is_one() {
            if self != rhs { // canonised
                CLP::true_()
            } else {
                CLP::false_()
            }
        } else if self.overlap(&rhs) {
            CLP::bool_top()
        } else {
            CLP::true_()
        }
    }

    pub fn less(&self, rhs: &CLP) -> CLP {
        || -> Option<CLP> {
            let min_e1 = self.min_elem()?;
            let max_e1 = self.max_elem()?;
            let min_e2 = rhs.min_elem()?;
            let max_e2 = rhs.max_elem()?;

            Some(if max_e1 < min_e2 {
                CLP::true_()
            } else if min_e1 >= max_e2 {
                CLP::false_()
            } else {
                CLP::bool_top()
            })
        }().unwrap_or_else(|| CLP::bool_bottom()) // bool is 8-bits
    }

    pub fn lesseq(&self, rhs: &CLP) -> CLP {
        || -> Option<CLP> {
            let min_e1 = self.min_elem()?;
            let max_e1 = self.max_elem()?;
            let min_e2 = rhs.min_elem()?;
            let max_e2 = rhs.max_elem()?;

            Some(if max_e1 <= min_e2 {
                CLP::true_()
            } else if min_e1 > max_e2 {
                CLP::false_()
            } else {
                CLP::bool_top()
            })
        }().unwrap_or_else(|| CLP::bool_bottom()) // bool is 8-bits
    }

    pub fn signed_less(&self, rhs: &CLP) -> CLP {
        || -> Option<CLP> {
            let min_e1 = self.min_elem_signed()?;
            let max_e1 = self.max_elem_signed()?;
            let min_e2 = rhs.min_elem_signed()?;
            let max_e2 = rhs.max_elem_signed()?;

            Some(if max_e1.signed() < min_e2.signed() {
                CLP::true_()
            } else if min_e1.signed() >= max_e2.signed() {
                CLP::false_()
            } else {
                CLP::bool_top()
            })
        }().unwrap_or_else(|| CLP::bool_bottom()) // bool is 8-bits
    }

    pub fn signed_lesseq(&self, rhs: &CLP) -> CLP {
        || -> Option<CLP> {
            let min_e1 = self.min_elem_signed()?;
            let max_e1 = self.max_elem_signed()?;
            let min_e2 = rhs.min_elem_signed()?;
            let max_e2 = rhs.max_elem_signed()?;

            Some(if max_e1.signed() <= min_e2.signed() {
                CLP::true_()
            } else if min_e1.signed() > max_e2.signed() {
                CLP::false_()
            } else {
                CLP::bool_top()
            })
        }().unwrap_or_else(|| CLP::bool_bottom()) // bool is 8-bits
    }

    pub fn from_iter_with<I>(bvs: I, width: usize, signed: bool) -> CLP
    where I: IntoIterator<Item=BitVec> {
        assert!(width > 0);
        || -> Option<CLP> {
            let mut sorted = bvs.into_iter()
                .map(|bv| if signed { bv.signed() } else { bv.unsigned() }.cast(width))
                .collect::<Vec<_>>();
            sorted.sort();

            let sorted_last = sorted.last();
            let sorted_rot = sorted_last.iter()
                .map(|v| *v)
                .chain(sorted.iter().skip(1));

            let diffs = sorted.iter()
                .zip(sorted_rot)
                .enumerate()
                .map(|(i, (l, r))| (i, l - r));

            let (idx, _, step) = diffs.fold((0, BitVec::zero(width), BitVec::zero(width)), |(idx, diff, step), (i, d)| {
                assert_eq!(diff.bits(), step.bits());
                assert_eq!(diff.bits(), d.bits());

                if d > diff {
                    (i, d.clone(), diff.gcd(&step))
                } else {
                    (idx, diff, d.gcd(&step))
                }
            });

            let (e, s) = sorted.split_at(idx);
            let it = e.iter().chain(s.iter());

            let base = it.clone().next()?;
            let e = it.last()?;
            let card = CLP::card_from_bounds(base, &step, e);

            assert_eq!(base.bits(), width);

            Some(CLP::new_with(base.clone(), step, card))
        }().unwrap_or_else(|| CLP::bottom(width))
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
            let step = &rhs.step * &self.base;
            CLP::new_with(base, step, rhs.card.clone())
        } else if rhs.step.is_zero() || rhs.card.is_one() {
            let base = &self.base * &rhs.base;
            let step = &self.step * &rhs.base;
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
                let e_no_wrap = shl_exact(&e1, max_p2i as usize);
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

impl Shr for CLP {
    type Output = CLP;

    fn shr(self, rhs: Self) -> Self::Output {
        (&self).shr(&rhs)
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

impl Div for CLP {
    type Output = CLP;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).div(&rhs)
    }
}

impl Div for &'_ CLP {
    type Output = CLP;

    fn div(self, rhs: Self) -> Self::Output {
        let sz = get_and_check_size(self, rhs);
        || -> Option<CLP> {
            let min_e1 = self.min_elem()?;
            let max_e1 = self.max_elem()?;
            let min_e2 = rhs.min_elem()?;
            let max_e2 = rhs.max_elem()?;

            if rhs.contains(&BitVec::zero(sz)) {
                // TODO: division by zero
                None // or panic!?
            } else {
                let base = min_e1 / max_e2;
                let e = max_e1 / min_e2;
                let step = if rhs.cardinality().is_one() && (&self.step % &rhs.base).is_zero() {
                    (&self.step / &rhs.base).gcd(&(&(&self.base / &rhs.base) - &base))
                } else {
                    BitVec::one(sz)
                };
                let card = CLP::card_from_bounds(&base, &step, &e);
                Some(CLP { base, step, card })
            }
        }().unwrap_or_else(|| CLP::bottom(sz))
    }
}

impl Rem for CLP {
    type Output = CLP;

    fn rem(self, rhs: Self) -> Self::Output {
        (&self).rem(&rhs)
    }
}

impl Rem for &'_ CLP {
    type Output = CLP;

    fn rem(self, rhs: Self) -> Self::Output {
        self.sub(&(&self.div(&rhs)).mul(rhs))
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simple() {
        // int arr[4] = { 0, 1, 2, 3 };
        // int i = top_32;
        // if (i >= -1 && i <= 3) {
        //    // ok, in-bounds
        //    ...
        // }

        let mut i = CLP::top(32);

        let chk1 = CLP::new(BitVec::from(0i32).signed());
        let chk2 = CLP::new(BitVec::from(3u32));

        // signed chk1
        let chk1_min = chk1.min_elem_signed().unwrap().signed();
        let chk1_max = BitVec::from(10u32).signed(); //BitVec::max_value_with(32, true).signed();
        let chk1_rng = CLP::range(chk1_min, chk1_max);

        // (1, i32::MAX)
        // (i32::MIN, 3)
        //
        // = (-1, +3)

        i = i.intersection(&chk1_rng);

        //assert_eq!(i, chk1_rng);

        // signed chk2
        let chk2_min = BitVec::from(-10i32).signed(); //BitVec::min_value_with(32, true).signed();
        let chk2_max = BitVec::from(3i32); //chk2.max_elem_signed().unwrap().signed();
        let chk2_rng = CLP::range(chk2_min, chk2_max);

        println!("base: {}, end: {} | ({}, {})", chk1_rng.base, chk1_rng.finite_end().unwrap(), chk1_rng.min_elem().unwrap(), chk1_rng.max_elem().unwrap());
        println!("base: {}, end: {} | ({}, {})", chk2_rng.base, chk2_rng.finite_end().unwrap(), chk2_rng.min_elem().unwrap(), chk2_rng.max_elem().unwrap());

        let (chk2l, chk2u) = chk2_rng.split_unsigned();
        println!("l: {:?}", chk2l);
        println!("u: {:?}", chk2u);

        println!("{:?}", chk2u.union(&chk2l));

        i = i.intersection(&chk2_rng);

        println!("({}, {})", chk2_rng.min_elem_signed().unwrap(), chk2_rng.max_elem_signed().unwrap());

        println!("{:?}", chk1_rng.intersection(&chk2_rng));

        println!("=({}, {})", i.min_elem().unwrap(), i.max_elem().unwrap());
        //for v in chk1_rng.intersection(&chk2_rng).iter().take(100) {
        for v in i.iter().take(100) {
            let v = v.signed();
            println!("{} {}", v, v.is_negative());
        }
    }
}
