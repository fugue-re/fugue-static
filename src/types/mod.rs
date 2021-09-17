use std::borrow::Cow;
use std::collections::HashMap;

use fixedbitset::FixedBitSet;

use petgraph::data::DataMap;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph as DiGraph;
use petgraph::visit::{
    GraphRef, IntoNeighborsDirected, IntoNodeIdentifiers, IntoNodeReferences, NodeCount, Reversed,
    Visitable,
};

use fugue::ir::il::ecode::{Entity, EntityId};

pub type EntityMap<T> = HashMap<EntityId, Entity<T>>;
pub type EntityRefMap<'a, T> = HashMap<EntityId, Cow<'a, Entity<T>>>;
pub type EntityGraph<E> = DiGraph<EntityId, E>;

pub trait AsEntityGraph:
    GraphRef
    + DataMap<NodeId=NodeIndex, NodeWeight=EntityId>
    + IntoNeighborsDirected
    + IntoNodeIdentifiers
    + IntoNodeReferences
    + NodeCount
    + Visitable<NodeId = NodeIndex, Map = FixedBitSet>
{
}

impl<E> AsEntityGraph for &'_ EntityGraph<E> {}
impl<E> AsEntityGraph for Reversed<&'_ EntityGraph<E>> {}
