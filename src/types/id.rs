use ron_uuid::UUID;
use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use fugue::ir::il::Location;
use crate::types::{Locatable, Relocatable};

pub type Erased = ();

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(remote = "UUID")]
enum UUIDRef {
    Name {
        name: u64,
        scope: u64,
    },
    Number {
        value1: u64,
        value2: u64,
    },
    Event {
        timestamp: u64,
        origin: u64,
    },
    Derived {
        timestamp: u64,
        origin: u64,
    },
}

#[derive(educe::Educe)]
#[educe(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(bound = "")]
pub struct Id<T> {
    tag: &'static str,
    #[serde(with = "UUIDRef")]
    uuid: UUID,
    #[educe(Debug(ignore), PartialEq(ignore), Eq(ignore), PartialOrd(ignore), Ord(ignore), Hash(ignore))]
    #[serde(skip_deserializing)]
    marker: PhantomData<T>,
}

impl<T> Display for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.tag, self.uuid)
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self {
            tag: self.tag,
            uuid: self.uuid,
            marker: PhantomData,
        }
    }
}
impl<T> Copy for Id<T> { }

impl<T> Id<T> {
    pub fn new(tag: &'static str) -> Self {
        Self {
            tag, 
            uuid: UUID::now(),
            marker: PhantomData,
        }
    }
    
    pub fn retype<U: Clone>(self) -> Id<U> {
        Id {
            tag: self.tag,
            uuid: self.uuid,
            marker: PhantomData,
        }
    }
    
    pub fn erase(self) -> Id<Erased> {
        self.retype()
    }
    
    pub fn invalid(tag: &'static str) -> Self {
        Self {
            tag,
            uuid: UUID::zero(),
            marker: PhantomData,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.is_invalid()
    }
    
    pub fn is_invalid(&self) -> bool {
        self.uuid.is_zero()
    }
    
    pub fn tag(&self) -> &'static str {
        self.tag
    }
    
    pub fn uuid(&self) -> UUID {
        self.uuid
    }
}

pub trait Identifiable<V> {
    fn id(&self) -> Id<V>;
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct LocatableId<T> {
    #[serde(bound(deserialize = "Id<T>: serde::Deserialize<'de>"))]
    id: Id<T>,
    location: Location,
}

impl<T> LocatableId<T> {
    pub fn new<L>(tag: &'static str, location: L) -> Self
    where L: Into<Location> {
        Self {
            id: Id::new(tag),
            location: location.into(),
        }
    }

    pub fn invalid<L>(tag: &'static str, location: L) -> Self
    where L: Into<Location> {
        Self {
            id: Id::invalid(tag),
            location: location.into(),
        }
    }
    
    pub fn erase(self) -> LocatableId<Erased> {
        LocatableId {
            id: self.id.erase(),
            location: self.location,
        }
    }
    
    pub fn is_valid(&self) -> bool {
        self.id.is_valid()
    }
    
    pub fn is_invalid(&self) -> bool {
        self.id.is_invalid()
    }
    
    pub fn from_parts(id: Id<T>, location: Location) -> Self {
        Self {
            id,
            location,
        }
    }

    pub fn into_parts(self) -> (Id<T>, Location) {
        (self.id, self.location)
    }
}

impl<T, U> From<&'_ U> for LocatableId<T> where U: Identifiable<T> + Locatable {
    fn from(value: &U) -> Self {
        Self::from_parts(value.id(), value.location())
    }
}

impl<T> Identifiable<T> for LocatableId<T> {
    fn id(&self) -> Id<T> {
        self.id
    }
}

impl<T> Locatable for LocatableId<T> {
    fn location(&self) -> Location {
        self.location.clone()
    }
}

impl<T> Relocatable for LocatableId<T> {
    fn location_mut(&mut self) -> &mut Location {
        &mut self.location
    }
}

impl<T> Clone for LocatableId<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            location: self.location.clone(),
        }
    }
}

impl<T> Debug for LocatableId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.id)
    }
}

impl<T> Display for LocatableId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{} @ {}", self.id.tag, self.id.uuid, self.location)
    }
}

impl<T> PartialEq for LocatableId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id) && self.location.eq(&other.location)
    }
}
impl<T> Eq for LocatableId<T> { }

impl<T> PartialOrd for LocatableId<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for LocatableId<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let ide = self.id.cmp(&other.id);
        if ide.is_eq() {
            self.location.cmp(&other.location)
        } else {
            ide
        }
    }
}

impl<T> Hash for LocatableId<T> {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        self.id.hash(state);
        self.location.hash(state);
    }
}