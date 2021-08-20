use petgraph::EdgeDirection;
use petgraph::graph::NodeIndex;

use std::collections::HashMap;
use std::cmp::Ordering;

use crate::models::{Block, CFG};
use crate::graphs::traversals::{PostOrder, RevPostOrder, Traversal};

use fugue::ir::il::ecode::EntityId;

use thiserror::Error;

pub trait AnalysisCollector<O>: Default {
    fn get(&self, node: NodeIndex) -> Option<&O>;
    fn insert(&mut self, node: NodeIndex, value: O);
    fn remove(&mut self, node: NodeIndex) -> Option<O>;
}

impl<O> AnalysisCollector<O> for HashMap<NodeIndex, O> {
    #[inline(always)]
    fn get(&self, node: NodeIndex) -> Option<&O> {
        self.get(&node)
    }

    #[inline(always)]
    fn insert(&mut self, node: NodeIndex, value: O) {
        self.insert(node, value);
    }

    #[inline(always)]
    fn remove(&mut self, node: NodeIndex) -> Option<O> {
        self.remove(&node)
    }
}

#[derive(Debug, Error)]
pub enum AnalysisError<E: std::error::Error> {
    #[error(transparent)]
    Analysis(#[from] E),
    #[error("cannot order outputs for {0}")]
    CannotOrder(EntityId),
    #[error("new output for {0} is less than previous output")]
    LostInformation(EntityId),
}

pub trait FixedPointBackward<'a, O>
where O: Clone + Default + PartialOrd {
    type Err: std::error::Error;

    fn join(&mut self, current: O, next: &O) -> Result<O, Self::Err>;
    fn transfer(&mut self, block: &'a Block, current: Option<O>) -> Result<O, Self::Err>;

    #[inline(always)]
    fn analyse<C>(&mut self, graph: &'a CFG) -> Result<C, AnalysisError<Self::Err>>
    where C: AnalysisCollector<O> {
        self.analyse_with(graph, false)
    }

    fn analyse_with<C>(&mut self, graph: &'a CFG, always_merge: bool) -> Result<C, AnalysisError<Self::Err>>
    where C: AnalysisCollector<O> {
        let mut results = C::default();
        let mut queue = PostOrder::into_queue(graph);

        while let Some(node) = queue.pop_front() {
            let current_in = graph
                .entity_graph()
                .neighbors_directed(node, EdgeDirection::Outgoing)
                .try_fold(None, |acc, succ| if let Some(next) = results.get(succ) {
                    if let Some(acc) = acc {
                        self.join(acc, next).map(Option::from)
                    } else {
                        Ok(Some(next.clone()))
                    }
                } else {
                    Ok(acc)
                })?;

            let block = graph.block_at(node);
            let mut current = self.transfer(block.value(), current_in)?;

            if let Some(old_current) = results.get(node) {
                match current.partial_cmp(old_current) {
                    Some(Ordering::Greater) => (),
                    Some(Ordering::Equal) => continue, // no change
                    None | Some(Ordering::Less) if always_merge => {
                        current = self.join(current, old_current)?;
                    },
                    Some(Ordering::Less) => {
                        let eid = graph.entity_graph().node_weight(node).unwrap();
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },
                    None => {
                        let eid = graph.entity_graph().node_weight(node).unwrap();
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },

                }
            }

            results.insert(node, current);

            for pred in graph.entity_graph().neighbors_directed(node, EdgeDirection::Incoming) {
                if !queue.contains(&pred) {
                    queue.push_back(pred);
                }
            }
        }

        Ok(results)
    }
}

pub trait FixedPointForward<'a, O>
where O: Clone + Default + PartialOrd {
    type Err: std::error::Error;

    fn join(&mut self, current: O, next: &O) -> Result<O, Self::Err>;
    fn transfer(&mut self, block: &'a Block, current: Option<O>) -> Result<O, Self::Err>;

    #[inline(always)]
    fn analyse<C>(&mut self, graph: &'a CFG) -> Result<C, AnalysisError<Self::Err>>
    where C: AnalysisCollector<O> {
        self.analyse_with(graph, false)
    }

    fn analyse_with<C>(&mut self, graph: &'a CFG, always_merge: bool) -> Result<C, AnalysisError<Self::Err>>
    where C: AnalysisCollector<O> {
        let mut results = C::default();
        let mut queue = RevPostOrder::into_queue(graph);

        while let Some(node) = queue.pop_front() {
            let current_in = graph
                .entity_graph()
                .neighbors_directed(node, EdgeDirection::Incoming)
                .try_fold(None, |acc, pred| if let Some(next) = results.get(pred) {
                    if let Some(acc) = acc {
                        self.join(acc, next).map(Option::from)
                    } else {
                        Ok(Some(next.clone()))
                    }
                } else {
                    Ok(acc)
                })?;

            let block = graph.block_at(node);

            let mut current = self.transfer(block.value(), current_in)?;

            if let Some(old_current) = results.get(node) {
                match current.partial_cmp(old_current) {
                    Some(Ordering::Greater) => (),
                    Some(Ordering::Equal) => continue, // no change
                    None | Some(Ordering::Less) if always_merge => {
                        current = self.join(current, old_current)?;
                    },
                    Some(Ordering::Less) => {
                        let eid = graph.entity_graph().node_weight(node).unwrap();
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },
                    None => {
                        let eid = graph.entity_graph().node_weight(node).unwrap();
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },

                }
            }

            results.insert(node, current);

            for succ in graph.entity_graph().neighbors_directed(node, EdgeDirection::Outgoing) {
                if !queue.contains(&succ) {
                    queue.push_back(succ);
                }
            }
        }

        Ok(results)
    }
}
