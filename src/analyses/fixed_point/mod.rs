use std::collections::VecDeque;

use crate::graphs::entity::AsEntityGraph;
use crate::traits::collect::EntityValueCollector;

pub trait FixedPointBackward<'a, V, E, G, O>
where V: 'a + Clone,
      E: 'a,
      G: AsEntityGraph<'a, V, E>,
      O: Clone + Default {
    type Err: std::error::Error;

    fn join(&mut self, current: O, next: &O) -> Result<O, Self::Err>;
    fn transfer(&mut self, entity: &'a V, current: Option<O>) -> Result<O, Self::Err>;

    fn analyse<C>(&mut self, g: &'a G) -> Result<C, Self::Err>
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
            let current = self.transfer(entity.value(), current_in)?;

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
      O: Clone + Default {
    type Err: std::error::Error;

    fn join(&mut self, current: O, next: &O) -> Result<O, Self::Err>;
    fn transfer(&mut self, entity: &'a V, current: Option<O>) -> Result<O, Self::Err>;

    fn analyse<C>(&mut self, g: &'a G) -> Result<C, Self::Err>
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
            let current = self.transfer(entity.value(), current_in)?;

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
