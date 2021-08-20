use petgraph::EdgeDirection;
use std::cmp::Ordering;

use crate::models::{Block, CFG};
use crate::graphs::traversals::{PostOrder, RevPostOrder, Traversal};
use crate::traits::collect::EntityValueCollector;

use fugue::ir::il::ecode::EntityId;

use thiserror::Error;

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
    where C: EntityValueCollector<O> {
        self.analyse_with(graph, false)
    }

    fn analyse_with<C>(&mut self, graph: &'a CFG, always_merge: bool) -> Result<C, AnalysisError<Self::Err>>
    where C: EntityValueCollector<O> {
        let mut results = C::default();
        let mut queue = PostOrder::into_queue(graph);

        while let Some(node) = queue.pop_front() {
            let current_in = graph
                .entity_graph()
                .neighbors_directed(node, EdgeDirection::Outgoing)
                .try_fold(None, |acc, succ_nx| {
                    let succ = &graph[succ_nx];
                    if let Some(next) = results.get(succ) {
                        if let Some(acc) = acc {
                            self.join(acc, next).map(Option::from)
                        } else {
                            Ok(Some(next.clone()))
                        }
                    } else {
                        Ok(acc)
                    }
                })?;

            let eid = &graph[node];
            let block = graph.block_at(node);
            let mut current = self.transfer(block.value(), current_in)?;

            if let Some(old_current) = results.get(eid) {
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

            results.insert(eid.clone(), current);

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
    where C: EntityValueCollector<O> {
        self.analyse_with(graph, false)
    }

    fn analyse_with<C>(&mut self, graph: &'a CFG, always_merge: bool) -> Result<C, AnalysisError<Self::Err>>
    where C: EntityValueCollector<O> {
        let mut results = C::default();
        let mut queue = RevPostOrder::into_queue(graph);

        while let Some(node) = queue.pop_front() {
            let current_in = graph
                .entity_graph()
                .neighbors_directed(node, EdgeDirection::Incoming)
                .try_fold(None, |acc, pred_nx| {
                    let pred = &graph[pred_nx];
                    if let Some(next) = results.get(pred) {
                        if let Some(acc) = acc {
                            self.join(acc, next).map(Option::from)
                        } else {
                            Ok(Some(next.clone()))
                        }
                    } else {
                        Ok(acc)
                    }
                })?;

            let eid = &graph[node];
            let block = graph.block_at(node);

            let mut current = self.transfer(block.value(), current_in)?;

            if let Some(old_current) = results.get(eid) {
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

            results.insert(eid.clone(), current);

            for succ in graph.entity_graph().neighbors_directed(node, EdgeDirection::Outgoing) {
                if !queue.contains(&succ) {
                    queue.push_back(succ);
                }
            }
        }

        Ok(results)
    }
}
