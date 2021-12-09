use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

use fugue::ir::il::Location;
use fxhash::FxBuildHasher;

use crate::traits::collect::EntityRefCollector;
use crate::types::{Id, Identifiable, Located, Locatable, Relocatable};

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
pub struct Entity<V: Clone> {
    #[serde(bound(deserialize = "Id<V>: serde::Deserialize<'de>"))]
    id: Id<V>,
    value: V,
}

pub trait EntityIdMapping<V: Clone> {
    fn lookup_by_id(&self, id: Id<V>) -> Option<EntityRef<V>>;
}

pub trait EntityLocMapping<V: Clone> {
    fn lookup_by_location<'a, C: EntityRefCollector<'a, V>>(&'a self, loc: &Location) -> C {
        let mut collect = C::default();
        self.lookup_by_location_with(loc, &mut collect);
        collect
    }

    fn lookup_by_location_with<'a, C: EntityRefCollector<'a, V>>(&'a self, loc: &Location, collect: &mut C);
}

pub type EntityMap<T, R = FxBuildHasher> = HashMap<Id<T>, Entity<T>, R>;
pub type LocatableEntity<V> = Entity<Located<V>>;

impl<V> From<V> for Entity<V> where V: Clone + Identifiable<V> {
    fn from(value: V) -> Self {
        Self {
            id: value.id(),
            value,
        }
    }
}

impl<V> PartialEq for Entity<V> where V: Clone {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<V> Eq for Entity<V> where V: Clone {}

impl<V> Ord for Entity<V> where V: Clone {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
impl<V> PartialOrd for Entity<V> where V: Clone {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<V> Hash for Entity<V> where V: Clone {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<V> Deref for Entity<V> where V: Clone {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<V> DerefMut for Entity<V> where V: Clone {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<V> Entity<V> where V: Clone {
    pub fn new(tag: &'static str, value: V) -> Self {
        Self {
            id: Id::new(tag),
            value,
        }
    }

    pub fn map<U, F>(self, f: F) -> Entity<U>
    where
        U: Clone,
        F: Fn(V) -> U
    {
        Entity {
            id: self.id.retype(),
            value: f(self.value),
        }
    }

    pub fn value(&self) -> &V {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }

    pub fn into_value(self) -> V {
        self.value
    }

    pub fn from_parts(id: Id<V>, value: V) -> Self {
        Self {
            id,
            value,
        }
    }

    pub fn into_parts(self) -> (Id<V>, V) {
        (self.id, self.value)
    }
}

impl<V> Identifiable<V> for Entity<V> where V: Clone {
    fn id(&self) -> Id<V> {
        self.id
    }
}

impl<V> Locatable for Entity<V> where V: Clone + Locatable {
    fn location(&self) -> Location {
        self.value.location()
    }
}

impl<V> Relocatable for Entity<V> where V: Clone + Relocatable {
    fn location_mut(&mut self) -> &mut Location {
        self.value.location_mut()
    }
}

pub type EntityRef<'a, V> = Cow<'a, Entity<V>>;
pub type LocatableEntityRef<'a, V> = Cow<'a, Entity<Located<V>>>;

pub type EntityRefMap<'a, T, R = FxBuildHasher> = HashMap<Id<T>, EntityRef<'a, T>, R>;

pub trait IntoEntityRef<'a>: Clone {
    type T: Clone;

    fn into_entity_ref(self) -> EntityRef<'a, Self::T>;
}

impl<'a, T> IntoEntityRef<'a> for &'a Entity<T> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, Self::T> {
        Cow::Borrowed(self)
    }
}

impl<'a, T> IntoEntityRef<'a> for Entity<T> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, T> {
        Cow::Owned(self)
    }
}

impl<'a, T> IntoEntityRef<'a> for Cow<'a, Entity<T>> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, T> {
        self
    }
}

impl<'a, T> IntoEntityRef<'a> for &'_ Cow<'a, Entity<T>> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, T> {
        self.clone()
    }
}
