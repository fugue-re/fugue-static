use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug, Display};
use std::ops::{Deref, DerefMut};

use petgraph::dot::{Dot, Config as DotConfig};
use petgraph::stable_graph::NodeIndex;

use fugue::ir::il::ecode::{EntityId, Location};

use crate::models::Block;

use crate::traits::{EntityRef, IntoEntityRef};
use crate::traits::dominance::*;

use crate::types::{EntityRefMap, EntityGraph};

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

#[derive(Clone, Default)]
pub struct CFG<'e> {
    pub(crate) graph: EntityGraph<BranchKind>,

    pub(crate) entry_points: HashSet<(EntityId, NodeIndex)>,

    pub(crate) entity_mapping: HashMap<EntityId, NodeIndex>,
    pub(crate) blocks: EntityRefMap<'e, Block>,
}

// NOTE: this is a hack to get petgraph to render blocks nicely
pub struct DisplayAlways<T: Display>(T);

impl<T> Debug for DisplayAlways<T> where T: Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> Display for DisplayAlways<T> where T: Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct DotCFG<'a, 'e>(&'a CFG<'e>);

impl<'a, 'e> Display for DotCFG<'a, 'e> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.0
            .map(|_nx, n| {
                let block = &self.0.blocks()[n];
                DisplayAlways(format!("{}", block.value()))
            },
            |_ex, e| e);

        let dot = Dot::with_attr_getters(
            &g_str,
            &[DotConfig::EdgeNoLabel],
            &|_, _| "".to_owned(),
            &|_, _| "shape=box".to_owned());

        write!(f, "{}", dot)
    }
}

impl<'e> AsRef<EntityGraph<BranchKind>> for CFG<'e> {
    fn as_ref(&self) -> &EntityGraph<BranchKind> {
        &self.graph
    }
}

impl<'e> AsMut<EntityGraph<BranchKind>> for CFG<'e> {
    fn as_mut(&mut self) -> &mut EntityGraph<BranchKind> {
        &mut self.graph
    }
}

impl<'e> Deref for CFG<'e> {
    type Target = EntityGraph<BranchKind>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'e> DerefMut for CFG<'e> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<'e> CFG<'e> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Corresponds to the first entry point added
    pub fn default_entry(&self) -> Option<NodeIndex> {
        self.entry_points.iter().next().map(|(_, idx)| *idx)
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

    pub fn add_call<B, F>(&mut self, blk: B, fcn: F)
    where B: IntoEntityRef<'e, T=Block>,
          F: IntoEntityRef<'e, T=Block> {
        let blk = blk.into_entity_ref();
        let fcn = fcn.into_entity_ref();

        let blk_end = self.entity_mapping[blk.id()];
        let fcn_start = self.entity_mapping[fcn.id()];

        self.graph.add_edge(blk_end, fcn_start, BranchKind::Call);
    }

    pub fn add_cond<B, T>(&mut self, blk: B, tgt: T)
    where B: IntoEntityRef<'e, T=Block>,
          T: IntoEntityRef<'e, T=Block> {
        let blk = blk.into_entity_ref();
        let tgt = tgt.into_entity_ref();

        let blk_end = self.entity_mapping[blk.id()];
        let blk_start = self.entity_mapping[tgt.id()];

        let fall_start = self.entity_mapping[blk.value().next_blocks().next().unwrap()];

        self.graph.add_edge(blk_end, blk_start, BranchKind::Jump);
        self.graph.add_edge(blk_end, fall_start, BranchKind::Fall);
    }

    pub fn add_jump<B, T>(&mut self, blk: B, tgt: T)
    where B: IntoEntityRef<'e, T=Block>,
          T: IntoEntityRef<'e, T=Block> {
        let blk = blk.into_entity_ref();
        let tgt = tgt.into_entity_ref();

        let blk_end = self.entity_mapping[blk.id()];
        let blk_start = self.entity_mapping[tgt.id()];

        self.graph.add_edge(blk_end, blk_start, BranchKind::Jump);
    }

    pub fn dot<'a>(&'a self) -> DotCFG<'a, 'e> {
        DotCFG(self)
    }

    pub fn dominance_tree(&self) -> (NodeIndex, DominanceTree) {
        if self.entry_points.len() > 1 {
            panic!("dominance tree for multiple entry points")
        }

        self.graph.dominance_tree(self.default_entry().unwrap())
    }

    pub fn dominance_frontier(&self) -> DominanceFrontier {
        if self.entry_points.len() > 1 {
            panic!("dominance frontier for multiple entry points")
        }

        self.graph.dominance_frontier(self.default_entry().unwrap())
    }

    pub fn dominance(&self) -> Dominance {
        if self.entry_points.len() > 1 {
            panic!("dominance frontier for multiple entry points")
        }

        self.graph.dominance(self.default_entry().unwrap())
    }
}
