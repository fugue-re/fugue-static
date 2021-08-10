use std::collections::{HashMap, HashSet};

use petgraph::stable_graph::{NodeIndex, StableDiGraph as Graph};
use petgraph::EdgeDirection;

use fugue::ir::il::ecode::{Entity, EntityId, Location, Stmt};

use crate::graphs::orders::{PostOrder, RevPostOrder};
use crate::models::Block;

#[derive(Debug, Clone)]
pub enum Edge<'a> {
    Call(&'a Entity<Stmt>),
    Jump(Option<&'a Entity<Stmt>>),
}

impl<'a> Edge<'a> {
    pub fn is_call(&self) -> bool {
        matches!(self, Edge::Call(_))
    }

    pub fn is_jump(&self) -> bool {
        matches!(self, Edge::Jump(_))
    }
}

#[derive(Debug, Clone)]
pub enum Node<'a> {
    BlockStart(&'a Entity<Block>),
    BlockEnd(&'a Entity<Block>),
}

impl<'a> Node<'a> {
    pub fn block(&self) -> &'a Entity<Block> {
        match self {
            Self::BlockStart(blk) | Self::BlockEnd(blk) => blk,
        }
    }
}

#[derive(Clone, Default)]
pub struct CFG<'a> {
    graph: Graph<Node<'a>, Edge<'a>>,
    entry_points: HashSet<EntityId>,
    entity_mapping: HashMap<EntityId, (NodeIndex<u32>, NodeIndex<u32>)>,
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

    pub fn get_block<L: Into<Location>>(&self, location: L) -> Option<&'a Entity<Block>> {
        let id = EntityId::new("blk", location.into());
        let (blk_start, _) = self.entity_mapping.get(&id)?;
        self.graph.node_weight(*blk_start)
            .map(Node::block)
    }

    pub fn with_preds<L, O, F>(&self, location: L, mut f: F) -> Option<O>
    where L: Into<Location>,
          O: Default,
          F: FnMut(O, &'a Entity<Block>, Edge<'a>) -> O {

        let id = EntityId::new("blk", location.into());
        let (blk_start, _) = self.entity_mapping.get(&id)?;

        let mut walker = self.graph
            .neighbors_directed(*blk_start, EdgeDirection::Incoming)
            .detach();

        let mut out = O::default();
        while let Some((ex, nx)) = walker.next(&self.graph) {
            let node = self.graph.node_weight(nx).unwrap().block();
            let edge = self.graph.edge_weight(ex).unwrap();

            out = f(out, node, edge.clone());
        }
        Some(out)
    }

    pub fn with_succs<L, O, F>(&self, location: L, mut f: F) -> Option<O>
    where L: Into<Location>,
          O: Default,
          F: FnMut(O, &'a Entity<Block>, Edge<'a>) -> O {

        let id = EntityId::new("blk", location.into());
        let (_, blk_end) = self.entity_mapping.get(&id)?;

        let mut walker = self.graph
            .neighbors_directed(*blk_end, EdgeDirection::Outgoing)
            .detach();

        let mut out = O::default();
        while let Some((ex, nx)) = walker.next(&self.graph) {
            let node = self.graph.node_weight(nx).unwrap().block();
            let edge = self.graph.edge_weight(ex).unwrap();

            out = f(out, node, edge.clone());
        }
        Some(out)
    }

    pub fn add_block(&mut self, block: &'a Entity<Block>) -> (NodeIndex<u32>, NodeIndex<u32>) {
        if let Some(idxs) = self.entity_mapping.get(block.id()) {
            *idxs
        } else {
            let sidx = self.graph.add_node(Node::BlockStart(block));
            let eidx = self.graph.add_node(Node::BlockEnd(block));

            let idxs = (sidx, eidx);

            self.entity_mapping.insert(block.id().clone(), idxs);

            idxs
        }
    }

    pub fn add_call(&mut self, blk: &'a Entity<Block>, fcn: &'a Entity<Block>, via: &'a Entity<Stmt>) {
        let (_, blk_end) = self.entity_mapping[blk.id()];
        let (fcn_start, _) = self.entity_mapping[fcn.id()];

        self.graph.add_edge(blk_end, fcn_start, Edge::Call(via));
    }

    pub fn add_cond(&mut self, blk: &'a Entity<Block>, tgt: &'a Entity<Block>) {
        let (_, blk_end) = self.entity_mapping[blk.id()];
        let (blk_start, _) = self.entity_mapping[tgt.id()];

        let (fall_start, _) = self.entity_mapping[blk.value().next_block()];

        self.graph.add_edge(blk_end, blk_start, Edge::Jump(Some(blk.value().last())));
        self.graph.add_edge(blk_end, fall_start, Edge::Jump(None));
    }

    pub fn add_jump(&mut self, blk: &'a Entity<Block>, tgt: &'a Entity<Block>) {
        let (_, blk_end) = self.entity_mapping[blk.id()];
        let (blk_start, _) = self.entity_mapping[tgt.id()];

        self.graph.add_edge(blk_end, blk_start, Edge::Jump(Some(blk.value().last())));
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
