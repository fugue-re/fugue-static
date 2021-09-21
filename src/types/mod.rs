use std::borrow::Cow;
use std::collections::HashMap;

use fugue::ir::il::ecode::{Entity, EntityId};

pub type EntityRef<'a, T> = Cow<'a, Entity<T>>;
pub type EntityMap<T> = HashMap<EntityId, Entity<T>>;
pub type EntityRefMap<'a, T> = HashMap<EntityId, EntityRef<'a, T>>;
