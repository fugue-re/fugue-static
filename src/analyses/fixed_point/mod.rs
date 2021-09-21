use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::graphs::entity::AsEntityGraph;
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

pub trait FixedPointBackward<'a, V, E, G, O>
where V: 'a + Clone,
      E: 'a,
      G: AsEntityGraph<'a, V, E>,
      O: Clone + Default + PartialOrd {
    type Err: std::error::Error;

    fn join(&mut self, current: O, next: &O) -> Result<O, Self::Err>;
    fn transfer(&mut self, entity: &'a V, current: Option<O>) -> Result<O, Self::Err>;

    #[inline(always)]
    fn analyse<C>(&mut self, g: &'a G) -> Result<C, AnalysisError<Self::Err>>
    where C: EntityValueCollector<O> {
        self.analyse_with(g, false)
    }

    fn analyse_with<C>(&mut self, g: &'a G, always_merge: bool) -> Result<C, AnalysisError<Self::Err>>
    where C: EntityValueCollector<O> {
        let mut results = C::default();

        let graph = g.entity_graph();
        let mut queue = graph.post_order().map(|(_, v, _)| v).collect::<VecDeque<_>>();

        while let Some(node) = queue.pop_front() {
            let current_in = graph
                .successors(node)
                .try_fold(None, |acc, (succ_nx, _)| {
                    let succ = graph.entity(succ_nx);
                    if let Some(next) = results.get(succ.id()) {
                        if let Some(acc) = acc {
                            self.join(acc, next).map(Option::from)
                        } else {
                            Ok(Some(next.clone()))
                        }
                    } else {
                        Ok(acc)
                    }
                })?;

            let entity = graph.entity(node);
            let eid = entity.id();
            let mut current = self.transfer(entity.value(), current_in)?;

            if let Some(old_current) = results.get(eid) {
                match current.partial_cmp(old_current) {
                    Some(Ordering::Greater) => (),
                    Some(Ordering::Equal) => continue, // no change
                    None | Some(Ordering::Less) if always_merge => {
                        current = self.join(current, old_current)?;
                    },
                    Some(Ordering::Less) => {
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },
                    None => {
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },

                }
            }

            results.insert(eid.clone(), current);

            for (pred, _) in graph.predecessors(node) {
                if !queue.contains(&pred) {
                    queue.push_back(pred);
                }
            }
        }

        Ok(results)
    }
}

pub trait FixedPointForward<'a, V, E, G, O>
where V: 'a + Clone,
      E: 'a,
      G: AsEntityGraph<'a, V, E>,
      O: Clone + Default + PartialOrd {
    type Err: std::error::Error;

    fn join(&mut self, current: O, next: &O) -> Result<O, Self::Err>;
    fn transfer(&mut self, entity: &'a V, current: Option<O>) -> Result<O, Self::Err>;

    #[inline(always)]
    fn analyse<C>(&mut self, g: &'a G) -> Result<C, AnalysisError<Self::Err>>
    where C: EntityValueCollector<O> {
        self.analyse_with(g, false)
    }

    fn analyse_with<C>(&mut self, g: &'a G, always_merge: bool) -> Result<C, AnalysisError<Self::Err>>
    where C: EntityValueCollector<O> {
        let mut results = C::default();

        let graph = g.entity_graph();
        let mut queue = graph.reverse_post_order().map(|(_, v, _)| v).collect::<VecDeque<_>>();

        while let Some(node) = queue.pop_front() {
            let current_in = graph
                .predecessors(node)
                .try_fold(None, |acc, (pred_nx, _)| {
                    let pred = graph.entity(pred_nx);
                    if let Some(next) = results.get(pred.id()) {
                        if let Some(acc) = acc {
                            self.join(acc, next).map(Option::from)
                        } else {
                            Ok(Some(next.clone()))
                        }
                    } else {
                        Ok(acc)
                    }
                })?;

            let entity = graph.entity(node);
            let eid = entity.id();
            let mut current = self.transfer(entity.value(), current_in)?;

            if let Some(old_current) = results.get(eid) {
                match current.partial_cmp(old_current) {
                    Some(Ordering::Greater) => (),
                    Some(Ordering::Equal) => continue, // no change
                    None | Some(Ordering::Less) if always_merge => {
                        current = self.join(current, old_current)?;
                    },
                    Some(Ordering::Less) => {
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },
                    None => {
                        return Err(AnalysisError::CannotOrder(eid.clone()))
                    },

                }
            }

            results.insert(eid.clone(), current);

            for (succ, _) in graph.successors(node) {
                if !queue.contains(&succ) {
                    queue.push_back(succ);
                }
            }
        }

        Ok(results)
    }
}
