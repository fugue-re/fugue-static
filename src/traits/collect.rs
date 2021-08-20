use std::collections::{BTreeSet, HashSet};
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use fugue::ir::il::ecode::EntityId;

pub trait EntityValueCollector<V>: Default {
    fn get(&self, id: &EntityId) -> Option<&V>;
    fn insert(&mut self, id: EntityId, value: V);
    fn remove(&mut self, id: &EntityId) -> Option<V>;
}

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

impl<V> EntityValueCollector<V> for HashMap<EntityId, V> {
    #[inline(always)]
    fn get(&self, id: &EntityId) -> Option<&V> {
        self.get(id)
    }

    #[inline(always)]
    fn insert(&mut self, id: EntityId, value: V) {
        self.insert(id, value);
    }

    #[inline(always)]
    fn remove(&mut self, id: &EntityId) -> Option<V> {
        self.remove(id)
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

impl<V> EntityValueCollector<V> for BTreeMap<EntityId, V> {
    #[inline(always)]
    fn get(&self, id: &EntityId) -> Option<&V> {
        self.get(id)
    }

    #[inline(always)]
    fn insert(&mut self, id: EntityId, value: V) {
        self.insert(id, value);
    }

    #[inline(always)]
    fn remove(&mut self, id: &EntityId) -> Option<V> {
        self.remove(id)
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
