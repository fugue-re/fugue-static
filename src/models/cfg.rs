use std::fmt::{self, Debug, Display};
use std::ops::{Deref, DerefMut};

use crate::graphs::entity::{AsEntityGraph, AsEntityGraphMut, EntityGraph, VertexIndex};
use crate::types::IntoEntityRef;

#[derive(Debug, Copy, Clone)]
pub enum BranchKind {
    Call,
    Fall,
    Jump,
    Unresolved,
}

impl Display for BranchKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call => write!(f, "call"),
            Self::Fall => write!(f, "fall"),
            Self::Jump => write!(f, "jump"),
            Self::Unresolved => write!(f, "unresolved"),
        }
    }
}

impl BranchKind {
    pub fn is_call(&self) -> bool {
        matches!(self, Self::Call)
    }

    pub fn is_fall(&self) -> bool {
        matches!(self, Self::Fall)
    }

    pub fn is_jump(&self) -> bool {
        matches!(self, Self::Jump)
    }

    pub fn is_unresolved(&self) -> bool {
        matches!(self, Self::Unresolved)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct CFG<'a, V> where V: Clone {
    pub(crate) graph: EntityGraph<'a, V, BranchKind>,
}

impl<'a, V> Default for CFG<'a, V>
where V: Clone {
    fn default() -> Self {
        Self { graph: EntityGraph::new() }
    }
}

impl<'a, V> Deref for CFG<'a, V>
where V: Clone {
    type Target = EntityGraph<'a, V, BranchKind>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'a, V> DerefMut for CFG<'a, V> where V: Clone {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<'a, V> AsEntityGraph<'a, V, BranchKind> for CFG<'a, V>
where V: Clone {
    fn entity_graph(&self) -> &EntityGraph<'a, V, BranchKind> {
        &self.graph
    }
}

impl<'a, V> AsEntityGraphMut<'a, V, BranchKind> for CFG<'a, V>
where V: Clone {
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, V, BranchKind> {
        &mut self.graph
    }
}

impl<'a, V> CFG<'a, V>
where V: Clone {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cloned<'b>(&self) -> CFG<'b, V> {
        CFG {
            graph: self.graph.cloned()
        }
    }

    /// Corresponds to the first entry point added
    pub fn default_entry(&self) -> Option<VertexIndex<V>> {
        self.graph.root_entities().next().map(|(_, vx, _)| vx)
    }

    /*
    pub fn block_node(&self, block: &EntityId) -> Option<NodeIndex> {
        self.entity_mapping.get(block).copied()
    }

    pub fn block_at(&self, index: NodeIndex) -> &EntityRef<'e, Block> {
        let eid = &self.graph[index];
        &self.blocks[eid]
    }

    pub fn block_at_mut(&mut self, index: NodeIndex) -> &mut EntityRef<'e, Block> {
        let eid = &self.graph[index];
        self.blocks.get_mut(eid).unwrap()
    }

    pub fn block<L: Into<Location>>(&self, location: L) -> Option<(NodeIndex, &EntityRef<'e, Block>)> {
        let id = EntityId::new("blk", location.into());
        let eid = *self.entity_mapping.get(&id)?;
        let blk = self.blocks.get(&id)?;
        Some((eid, blk))
    }

    pub fn block_mut<L: Into<Location>>(&mut self, location: L) -> Option<(NodeIndex, &mut EntityRef<'e, Block>)> {
        let id = EntityId::new("blk", location.into());
        let eid = *self.entity_mapping.get(&id)?;
        let blk = self.blocks.get_mut(&id)?;
        Some((eid, blk))
    }

    pub fn blocks(&self) -> &EntityRefMap<'e, Block> {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut EntityRefMap<'e, Block> {
        &mut self.blocks
    }

    pub fn entity_graph(&self) -> &EntityGraph<BranchKind> {
        &self.graph
    }

    pub fn entity_graph_mut(&mut self) -> &mut EntityGraph<BranchKind> {
        &mut self.graph
    }

    pub fn add_entry<T: IntoEntityRef<'e, T=Block>>(&mut self, block: T) -> NodeIndex {
        let block = block.into_entity_ref();
        let id = block.id().clone();
        let idx = self.add_block(block);
        self.entry_points.insert((id, idx));
        idx
    }

    pub fn add_block<T: IntoEntityRef<'e, T=Block>>(&mut self, block: T) -> NodeIndex {
        let block = block.into_entity_ref();
        if let Some(idx) = self.entity_mapping.get(block.id()) {
            *idx
        } else {
            let id = block.id().clone();
            let idx = self.graph.add_node(id.clone());
            self.entity_mapping.insert(id.clone(), idx);
            self.blocks.insert(id, block);
            idx
        }
    }
    */

    pub fn add_call<S, T>(&mut self, s: S, t: T)
    where S: IntoEntityRef<'a, T=V>,
          T: IntoEntityRef<'a, T=V> {

        self.graph.add_relation(s, t, BranchKind::Call);
    }

    pub fn add_unresolved<S, T>(&mut self, s: S, t: T)
    where S: IntoEntityRef<'a, T=V>,
          T: IntoEntityRef<'a, T=V> {

        self.graph.add_relation(s, t, BranchKind::Unresolved);
    }

    pub fn add_cond<S, T, F>(&mut self, s: S, t: T, f: F)
    where S: IntoEntityRef<'a, T=V>,
          T: IntoEntityRef<'a, T=V>,
          F: IntoEntityRef<'a, T=V> {
        let s = s.into_entity_ref();
        //let fall_start = self.entity_mapping[blk.value().next_blocks().next().unwrap()];

        self.graph.add_relation(s.clone(), t, BranchKind::Jump);
        self.graph.add_relation(s, f, BranchKind::Fall);
    }

    pub fn add_jump<S, T>(&mut self, s: S, t: T)
    where S: IntoEntityRef<'a, T=V>,
          T: IntoEntityRef<'a, T=V> {
        self.graph.add_relation(s, t, BranchKind::Jump);
    }
}
