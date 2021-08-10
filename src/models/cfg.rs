use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use petgraph::EdgeDirection;
use petgraph::algo::dominators;
use petgraph::stable_graph::{NodeIndex, StableDiGraph as Graph};
use petgraph::visit::IntoNodeReferences;

use fugue::ir::il::ecode::{Entity, EntityId, Location, Stmt};

use crate::graphs::orders::{PostOrder, RevPostOrder};
use crate::models::Block;

#[derive(Debug, Clone)]
pub enum Edge<'a> {
    Call(Cow<'a, Entity<Stmt>>),
    Jump(Option<Cow<'a, Entity<Stmt>>>),
}

impl<'a> Edge<'a> {
    pub fn is_call(&self) -> bool {
        matches!(self, Edge::Call(_))
    }

    pub fn is_jump(&self) -> bool {
        matches!(self, Edge::Jump(_))
    }
}

pub type Node<'a> = Cow<'a, Entity<Block>>;

pub type CFGDominanceFrontier = HashMap<NodeIndex<u32>, HashSet<NodeIndex<u32>>>;
pub type CFGDominanceTree = Graph<NodeIndex<u32>, ()>;

#[derive(Clone, Default)]
pub struct CFG<'a> {
    graph: Graph<Node<'a>, Edge<'a>>,
    entry_points: HashSet<(EntityId, NodeIndex<u32>)>,
    entity_mapping: HashMap<EntityId, NodeIndex<u32>>,
}

impl<'a> CFG<'a> {
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
    pub fn default_entry(&self) -> Option<NodeIndex<u32>> {
        self.entry_points.iter().next().map(|(_, idx)| *idx)
    }

    pub fn block<L: Into<Location>>(&self, location: L) -> Option<(NodeIndex<u32>, &Node<'a>)> {
        let id = EntityId::new("blk", location.into());
        let eid = *self.entity_mapping.get(&id)?;
        Some((eid, self.graph.node_weight(eid)?))
    }

    pub fn block_mut<L: Into<Location>>(&mut self, location: L) -> Option<(NodeIndex<u32>, &mut Node<'a>)> {
        let id = EntityId::new("blk", location.into());
        let eid = *self.entity_mapping.get(&id)?;
        Some((eid, self.graph.node_weight_mut(eid)?))
    }

    pub fn blocks(&self) -> impl Iterator<Item=(NodeIndex<u32>, &Node<'a>)> {
        self.graph().node_references()
    }

    pub fn graph(&self) -> &Graph<Node<'a>, Edge<'a>> {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut Graph<Node<'a>, Edge<'a>> {
        &mut self.graph
    }

    pub fn map_blocks<F, G>(&self, nodef: F) -> Self
    where F: FnMut(NodeIndex<u32>, &Node<'a>) -> Node<'a> {
        Self {
            graph: self.graph.map(nodef, |_, v| v.clone()),
            entry_points: self.entry_points.clone(),
            entity_mapping: self.entity_mapping.clone(),
        }
    }

    pub fn with_preds<L, O, F>(&self, location: L, mut f: F) -> Option<O>
    where L: Into<Location>,
          O: Default,
          F: FnMut(O, &Cow<'a, Entity<Block>>, &Edge<'a>) -> O {

        let id = EntityId::new("blk", location.into());
        let blk_start = self.entity_mapping.get(&id)?;

        let mut walker = self.graph
            .neighbors_directed(*blk_start, EdgeDirection::Incoming)
            .detach();

        let mut out = O::default();
        while let Some((ex, nx)) = walker.next(&self.graph) {
            let node = self.graph.node_weight(nx).unwrap();
            let edge = self.graph.edge_weight(ex).unwrap();

            out = f(out, node, edge);
        }
        Some(out)
    }

    pub fn with_succs<L, O, F>(&self, location: L, mut f: F) -> Option<O>
    where L: Into<Location>,
          O: Default,
          F: FnMut(O, &Cow<'a, Entity<Block>>, &Edge<'a>) -> O {

        let id = EntityId::new("blk", location.into());
        let blk_end = self.entity_mapping.get(&id)?;

        let mut walker = self.graph
            .neighbors_directed(*blk_end, EdgeDirection::Outgoing)
            .detach();

        let mut out = O::default();
        while let Some((ex, nx)) = walker.next(&self.graph) {
            let node = self.graph.node_weight(nx).unwrap();
            let edge = self.graph.edge_weight(ex).unwrap();

            out = f(out, node, edge);
        }
        Some(out)
    }

    pub fn add_entry(&mut self, block: &'a Entity<Block>) -> NodeIndex<u32> {
        let idx = self.add_block(block);
        self.entry_points.insert((block.id().clone(), idx));
        idx
    }

    pub fn add_block(&mut self, block: &'a Entity<Block>) -> NodeIndex<u32> {
        if let Some(idx) = self.entity_mapping.get(block.id()) {
            *idx
        } else {
            let id = block.id().clone();
            let idx = self.graph.add_node(Cow::Borrowed(block));
            self.entity_mapping.insert(id, idx);
            idx
        }
    }

    pub fn add_call(&mut self, blk: &'a Entity<Block>, fcn: &'a Entity<Block>, via: &'a Entity<Stmt>) {
        let blk_end = self.entity_mapping[blk.id()];
        let fcn_start = self.entity_mapping[fcn.id()];

        self.graph.add_edge(blk_end, fcn_start, Edge::Call(Cow::Borrowed(via)));
    }

    pub fn add_cond(&mut self, blk: &'a Entity<Block>, tgt: &'a Entity<Block>) {
        let blk_end = self.entity_mapping[blk.id()];
        let blk_start = self.entity_mapping[tgt.id()];

        let fall_start = self.entity_mapping[blk.value().next_block()];

        self.graph.add_edge(blk_end, blk_start, Edge::Jump(Some(Cow::Borrowed(blk.value().last()))));
        self.graph.add_edge(blk_end, fall_start, Edge::Jump(None));
    }

    pub fn add_jump(&mut self, blk: &'a Entity<Block>, tgt: &'a Entity<Block>) {
        let blk_end = self.entity_mapping[blk.id()];
        let blk_start = self.entity_mapping[tgt.id()];

        self.graph.add_edge(blk_end, blk_start, Edge::Jump(Some(Cow::Borrowed(blk.value().last()))));
    }

    pub fn dominance_tree(&self) -> CFGDominanceTree {
        if self.entry_points.len() > 1 {
            panic!("dominance tree for multiple entry points")
        }

        let entry = self.default_entry().unwrap();

        let mut tree = Graph::default();
        let dominators = dominators::simple_fast(&self.graph, entry);

        for d in self.graph.node_indices() {
            tree.add_node(d);
        }

        for d in self.graph.node_indices().rev() {
            if let Some(idom) = dominators.immediate_dominator(d) {
                tree.add_edge(idom, d, ());
            }
        }

        tree
    }

    pub fn dominance_frontier(&self) -> CFGDominanceFrontier {
        if self.entry_points.len() > 1 {
            panic!("dominance frontier for multiple entry points")
        }

        let entry = self.default_entry().unwrap();

        let mut mapping = HashMap::default();
        let dominators = dominators::simple_fast(&self.graph, entry);
        for d in self.graph.node_indices() {
            let idom = if let Some(idom) = dominators.immediate_dominator(d) {
                idom
            } else {
                continue
            };

            for mut np in self.graph.neighbors_directed(d, EdgeDirection::Incoming) {
                while np != idom {
                    mapping.entry(np).or_insert_with(HashSet::default).insert(d);
                    np = if let Some(np_dom) = dominators.immediate_dominator(np) {
                        np_dom
                    } else {
                        break
                    }
                }
            }
        }

        mapping
    }
}

#[derive(Clone)]
pub struct PostOrderIterator<'a, 'b> {
    cfg: &'b CFG<'a>,
    visitor: PostOrder<NodeIndex<u32>>,
}

impl<'a, 'b> Iterator for PostOrderIterator<'a, 'b> {
    type Item = &'b Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.visitor.next(&self.cfg.graph)
            .and_then(|id| self.cfg.graph.node_weight(id))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.visitor.stack.len(),
            Some(self.cfg.node_count() - self.visitor.finished.len()),
        )
    }
}

#[derive(Clone)]
pub struct RevPostOrderIterator<'a, 'b> {
    cfg: &'b CFG<'a>,
    visitor: RevPostOrder<NodeIndex<u32>>,
}

impl<'a, 'b> Iterator for RevPostOrderIterator<'a, 'b> {
    type Item = &'b Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.visitor.next(&self.cfg.graph)
            .and_then(|id| self.cfg.graph.node_weight(id))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.visitor.stack.len(),
            Some(self.cfg.node_count() - self.visitor.finished.len()),
        )
    }
}
