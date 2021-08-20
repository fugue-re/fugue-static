use std::borrow::Cow;
use std::collections::HashMap;

use petgraph::stable_graph::StableDiGraph as DiGraph;
use fugue::ir::il::ecode::{Entity, EntityId};

pub type EntityMap<T> = HashMap<EntityId, Entity<T>>;
pub type EntityRefMap<'a, T> = HashMap<EntityId, Cow<'a, Entity<T>>>;
pub type EntityGraph<E> = DiGraph<EntityId, E>;
