use std::collections::HashMap;
use fugue::ir::il::ecode::{Entity, EntityId};

pub type EntityMap<T> = HashMap<EntityId, Entity<T>>;
