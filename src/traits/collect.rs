use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;

pub trait ValueRefCollector<'ecode, V>: Default {
    fn insert_ref(&mut self, value: &'ecode V);
    fn merge_ref(&mut self, other: &mut Self);
    fn retain_difference_ref(&mut self, other: &Self);
}

pub trait ValueMutCollector<'ecode, V>: Default {
    fn insert_mut(&mut self, value: &'ecode mut V);
    fn merge_mut(&mut self, other: &mut Self);
    fn retain_difference_mut(&mut self, other: &Self);
}

impl<'ecode, V> ValueRefCollector<'ecode, V> for Vec<&'ecode V> where V: Eq {
    #[inline(always)]
    fn insert_ref(&mut self, var: &'ecode V) {
        self.push(var);
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        self.append(other);
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        self.retain(|v| !other.contains(v))
    }
}

impl<'ecode, V> ValueMutCollector<'ecode, V> for Vec<&'ecode mut V> where V: Eq {
    #[inline(always)]
    fn insert_mut(&mut self, var: &'ecode mut V) {
        self.push(var);
    }

    #[inline(always)]
    fn merge_mut(&mut self, other: &mut Self) {
        self.append(other);
    }

    #[inline(always)]
    fn retain_difference_mut(&mut self, other: &Self) {
        self.retain(|v| !other.contains(v))
    }
}

impl<'ecode, V> ValueRefCollector<'ecode, V> for HashSet<&'ecode V> where V: Eq + Hash {
    #[inline(always)]
    fn insert_ref(&mut self, var: &'ecode V) {
        self.insert(var);
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        self.extend(other.drain());
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        use std::ops::Sub;
        *self = self.sub(other);
    }
}

impl<'ecode, V> ValueMutCollector<'ecode, V> for HashSet<&'ecode mut V> where V: Eq + Hash {
    #[inline(always)]
    fn insert_mut(&mut self, var: &'ecode mut V) {
        self.insert(var);
    }

    #[inline(always)]
    fn merge_mut(&mut self, other: &mut Self) {
        self.extend(other.drain());
    }

    #[inline(always)]
    fn retain_difference_mut(&mut self, other: &Self) {
        self.retain(|v| !other.contains(v))
    }
}

impl<'ecode, V> ValueRefCollector<'ecode, V> for BTreeSet<&'ecode V> where V: Ord {
    #[inline(always)]
    fn insert_ref(&mut self, var: &'ecode V) {
        self.insert(var);
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        self.append(other);
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        use std::ops::Sub;
        *self = self.sub(other)
    }
}

impl<'ecode, V> ValueMutCollector<'ecode, V> for BTreeSet<&'ecode mut V> where V: Ord {
    #[inline(always)]
    fn insert_mut(&mut self, var: &'ecode mut V) {
        self.insert(var);
    }

    #[inline(always)]
    fn merge_mut(&mut self, other: &mut Self) {
        self.append(other);
    }

    #[inline(always)]
    fn retain_difference_mut(&mut self, other: &Self) {
        self.retain(|v| !other.contains(v))
    }
}
