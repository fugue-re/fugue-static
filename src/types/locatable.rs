use fugue::ir::il::Location;
use fugue::ir::il::ecode::Expr;

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::fmt::{self, Debug, Display};

use crate::types::{Id, EntityIdMapping, EntityLocMapping, EntityRef};

pub trait Locatable {
    fn location(&self) -> Location;
}

impl Locatable for Location {
    fn location(&self) -> Location {
        self.clone()
    }
}

pub trait Relocatable: Locatable {
    fn relocate(&mut self, location: Location) {
        *self.location_mut() = location;
    }

    fn location_mut(&mut self) -> &mut Location;
}

impl Relocatable for Location {
    fn location_mut(&mut self) -> &mut Location {
        self
    }
}

#[derive(Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
pub struct Located<V> {
    location: Location,
    value: V,
}

impl<V> Borrow<V> for Located<V> {
    fn borrow(&self) -> &V {
        &self.value
    }
}

impl<V> BorrowMut<V> for Located<V> {
    fn borrow_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

impl<V> Deref for Located<V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<V> DerefMut for Located<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<V> Debug for Located<V> where V: Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <V as Debug>::fmt(&self.value, f)
    }
}

impl<V> Display for Located<V> where V: Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <V as Display>::fmt(&self.value, f)
    }
}

impl<V> Located<V> {
    pub fn new<L>(location: L, value: V) -> Self
    where L: Into<Location> {
        Located {
            location: location.into(),
            value,
        }
    }

    pub fn into_inner(self) -> V {
        self.value
    }
}

impl<V> Locatable for Located<V> {
    fn location(&self) -> Location {
        self.location.clone()
    }
}

impl<V> Relocatable for Located<V> {
    fn location_mut(&mut self) -> &mut Location {
        &mut self.location
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
pub enum LocationTarget<T> {
    #[serde(bound(deserialize = "Id<T>: serde::Deserialize<'de>"))]
    Resolved(Id<T>),       // id <-> location
    Fixed(Location),       // location -> address * position
    Computed(Expr, usize), // address * position
}

impl<T> Clone for LocationTarget<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Resolved(id) => Self::Resolved(id.clone()),
            Self::Fixed(loc) => Self::Fixed(loc.clone()),
            Self::Computed(expr, pos) => Self::Computed(expr.clone(), *pos)
        }
    }
}

impl<T> Debug for LocationTarget<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolved(id) => write!(f, "LocationTarget::Resolved({:?})", id),
            Self::Fixed(loc) => write!(f, "LocationTarget::Fixed({:?})", loc),
            Self::Computed(expr, pos) => write!(f, "LocationTarget::Computed({:?}, {:?})", expr, pos),
        }
    }
}

impl<T> PartialEq for LocationTarget<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Resolved(id1), Self::Resolved(id2)) => id1 == id2,
            (Self::Fixed(loc1), Self::Fixed(loc2)) => loc1 == loc2,
            (Self::Computed(exp1, pos1), Self::Computed(exp2, pos2)) => pos1 == pos2 && exp1 == exp2,
            _ => false,
        }
    }
}
impl<T> Eq for LocationTarget<T> { }

impl<T> Ord for LocationTarget<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            Self::Resolved(id1) => match other {
                Self::Resolved(id2) => id1.cmp(id2),
                _ => Ordering::Less,
            },
            Self::Fixed(loc1) => match other {
                Self::Resolved(_) => Ordering::Greater,
                Self::Fixed(loc2) => loc1.cmp(loc2),
                _ => Ordering::Less,
            },
            Self::Computed(exp1, pos1) => match other {
                Self::Computed(exp2, pos2) => (exp1, pos1).cmp(&(exp2, pos2)),
                _ => Ordering::Greater,
            }
        }
    }
}

impl<T> PartialOrd for LocationTarget<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Hash for LocationTarget<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        match self {
            Self::Resolved(id) => {
                hasher.write_u64(0);
                id.hash(hasher);
            },
            Self::Fixed(loc) => {
                hasher.write_u64(1);
                loc.hash(hasher);
            },
            Self::Computed(exp, pos) => {
                hasher.write_u64(2);
                exp.hash(hasher);
                pos.hash(hasher);
            }
        }
    }
}

impl<T> From<Expr> for LocationTarget<T> {
    fn from(expr: Expr) -> Self {
        Self::Computed(expr, 0)
    }
}

impl<T> From<Location> for LocationTarget<T> {
    fn from(location: Location) -> Self {
        Self::Fixed(location)
    }
}

impl<T> From<Id<T>> for LocationTarget<T> {
    fn from(id: Id<T>) -> Self {
        Self::Resolved(id)
    }
}

impl<T> LocationTarget<T> {
    pub fn new(target: impl Into<LocationTarget<T>>) -> Self {
        target.into()
    }

    pub fn is_determined(&self) -> bool {
        self.is_fixed() || self.is_resolved()
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Resolved(_))
    }

    pub fn is_computed(&self) -> bool {
        matches!(self, Self::Computed(_, _))
    }
}

impl<T> LocationTarget<T> where T: Clone {
    pub fn resolve_with<'a, M>(&self, mapping: &'a M) -> Option<EntityRef<'a, T>>
    where M: 'a + EntityIdMapping<T> + EntityLocMapping<T> {
        match self {
            Self::Resolved(id) => mapping.lookup_by_id(*id),
            Self::Fixed(loc) => mapping.lookup_by_location::<Option<_>>(loc),
            _ => None,
        }
    }
}
