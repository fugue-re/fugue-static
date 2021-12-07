use intervals::Interval;

use fugue::ir::il::ecode::Var;

pub trait AsInterval<T>
where
    T: Clone + Ord,
{
    fn as_interval(&self) -> Interval<T>;
}

impl AsInterval<u64> for Var {
    fn as_interval(&self) -> Interval<u64> {
        let off = self.offset();
        let sz = (self.bits() as u64 / 8) + (if self.bits() % 8 != 0 { 1 } else { 0 });

        Interval::from(off..=off + (sz - 1))
    }
}
