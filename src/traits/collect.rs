use std::collections::{BTreeSet, HashSet};
use std::collections::{BTreeMap, HashMap};

use intervals::collections::{IntervalMap, IntervalSet};

use std::hash::Hash;

use crate::traits::AsInterval;
use crate::types::{Id, Identifiable, EntityRef, IntoEntityRef};

pub trait EntityValueCollector<V, T>: Default {
    fn get(&self, id: &Id<V>) -> Option<&T>;
    fn insert(&mut self, id: Id<V>, value: T);
    fn remove(&mut self, id: &Id<V>) -> Option<T>;
}

pub trait EntityRefCollector<'a, V>: Default where V: Clone {
    fn get(&self, id: &Id<V>) -> Option<EntityRef<V>>;
    fn insert<E: IntoEntityRef<'a, T = V>>(&mut self, entity: E);
    fn remove(&mut self, id: &Id<V>);
}

pub trait ValueRefCollector<'ecode, V>: Default {
    fn insert_ref(&mut self, value: &'ecode V);
    fn merge_ref(&mut self, other: &mut Self);
    fn retain_difference_ref(&mut self, other: &Self);
}

pub trait EntityValueRefCollector<'ecode, V, T>: Default {
    fn insert_ref(&mut self, id: Id<V>, value: &'ecode T);
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

impl<'a, V> EntityRefCollector<'a, V> for Vec<EntityRef<'a, V>> where V: Clone {
    #[inline(always)]
    fn get(&self, id: &Id<V>) -> Option<EntityRef<V>> {
        self.iter().find(|r| r.id() == *id).map(|v| EntityRef::Borrowed(&*v))
    }

    #[inline(always)]
    fn insert<E: IntoEntityRef<'a, T = V>>(&mut self, value: E) {
        self.push(value.into_entity_ref());
    }

    #[inline(always)]
    fn remove(&mut self, id: &Id<V>) {
        self.retain(|v| v.id() != *id)
    }
}

impl<'a, V> EntityRefCollector<'a, V> for Option<EntityRef<'a, V>> where V: Clone {
    #[inline(always)]
    fn get(&self, id: &Id<V>) -> Option<EntityRef<V>> {
        if let Some(e) = self {
            if e.id() == *id {
                Some(e.into_entity_ref())
            } else {
                None
            }
        } else {
            None
        }
    }

    #[inline(always)]
    fn insert<E: IntoEntityRef<'a, T = V>>(&mut self, value: E) {
        if self.is_none() {
            *self = Some(value.into_entity_ref());
        }
    }

    #[inline(always)]
    fn remove(&mut self, id: &Id<V>) {
        if matches!(self, Some(e) if e.id() == *id) {
            *self = None;
        }
    }
}

impl<V, T> EntityValueCollector<V, T> for HashMap<Id<V>, T> {
    #[inline(always)]
    fn get(&self, id: &Id<V>) -> Option<&T> {
        self.get(id)
    }

    #[inline(always)]
    fn insert(&mut self, id: Id<V>, value: T) {
        self.insert(id, value);
    }

    #[inline(always)]
    fn remove(&mut self, id: &Id<V>) -> Option<T> {
        self.remove(id)
    }
}

impl<'a, V> EntityRefCollector<'a, V> for HashMap<Id<V>, EntityRef<'a, V>> where V: Clone {
    #[inline(always)]
    fn get(&self, id: &Id<V>) -> Option<EntityRef<V>> {
        self.get(id).map(|v| EntityRef::Borrowed(&*v))
    }

    #[inline(always)]
    fn insert<E: IntoEntityRef<'a, T = V>>(&mut self, value: E) {
        let er = value.into_entity_ref();
        self.insert(er.id(), er);
    }

    #[inline(always)]
    fn remove(&mut self, id: &Id<V>) {
        self.remove(id);
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

impl<'ecode, V, T> EntityValueRefCollector<'ecode, V, T> for HashSet<(Id<V>, &'ecode T)> where T: Eq + Hash {
    #[inline(always)]
    fn insert_ref(&mut self, id: Id<V>, var: &'ecode T) {
        self.insert((id, var));
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

impl<V, T> EntityValueCollector<V, T> for BTreeMap<Id<V>, T> {
    #[inline(always)]
    fn get(&self, id: &Id<V>) -> Option<&T> {
        self.get(id)
    }

    #[inline(always)]
    fn insert(&mut self, id: Id<V>, value: T) {
        self.insert(id, value);
    }

    #[inline(always)]
    fn remove(&mut self, id: &Id<V>) -> Option<T> {
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

impl<'ecode, V, T> EntityValueRefCollector<'ecode, V, T> for BTreeSet<(Id<V>, &'ecode T)> where T: Ord {
    #[inline(always)]
    fn insert_ref(&mut self, id: Id<V>, var: &'ecode T) {
        self.insert((id, var));
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

impl<'ecode, V, I> ValueRefCollector<'ecode, V> for IntervalSet<I> where V: AsInterval<I>, I: Clone + Ord {
    #[inline(always)]
    fn insert_ref(&mut self, var: &'ecode V) {
        let iv = var.as_interval();
        self.insert(iv);
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        let other = std::mem::take(other);
        self.extend(other.into_iter());
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        for iv in other.iter() {
            self.remove_exact(iv);
        }
    }
}

impl<'ecode, V, I> ValueRefCollector<'ecode, V> for IntervalMap<I, &'ecode V> where V: AsInterval<I>, I: Clone + Ord {
    #[inline(always)]
    fn insert_ref(&mut self, var: &'ecode V) {
        let iv = var.as_interval();
        self.insert(iv, var);
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        let other = std::mem::take(other);
        self.extend(other.into_iter());
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        for iv in other.iter() {
            self.remove_exact(iv.0)
        }
    }
}

impl<'ecode, V, I> ValueMutCollector<'ecode, V> for IntervalMap<I, &'ecode mut V> where V: AsInterval<I>, I: Clone + Ord {
    #[inline(always)]
    fn insert_mut(&mut self, var: &'ecode mut V) {
        let iv = var.as_interval();
        self.insert(iv, var);
    }

    #[inline(always)]
    fn merge_mut(&mut self, other: &mut Self) {
        let other = std::mem::take(other);
        self.extend(other.into_iter());
    }

    #[inline(always)]
    fn retain_difference_mut(&mut self, other: &Self) {
        for iv in other.iter() {
            self.remove_exact(iv.0)
        }
    }
}

impl<'ecode, V, I, T> EntityValueRefCollector<'ecode, V, T> for IntervalMap<I, (Id<V>, &'ecode T)> where T: AsInterval<I>, I: Clone + Ord {
    #[inline(always)]
    fn insert_ref(&mut self, id: Id<V>, var: &'ecode T) {
        self.insert(var.as_interval(), (id, var));
    }

    #[inline(always)]
    fn merge_ref(&mut self, other: &mut Self) {
        let other = std::mem::take(other);
        self.extend(other.into_iter());
    }

    #[inline(always)]
    fn retain_difference_ref(&mut self, other: &Self) {
        for iv in other.iter() {
            self.remove_exact(iv.0)
        }
    }
}
