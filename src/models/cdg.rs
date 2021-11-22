use std::ops::{Deref, DerefMut};

use crate::graphs::algorithms::dominance::Dominance;
use crate::graphs::entity::{AsEntityGraph, AsEntityGraphMut, EntityGraph};

#[derive(Clone, Default)]
#[repr(transparent)]
pub struct CDG<'a, V, E> where V: Clone {
    pub(crate) graph: EntityGraph<'a, V, E>,
}

impl<'a, V, E> Deref for CDG<'a, V, E>
where V: Clone {
    type Target = EntityGraph<'a, V, E>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'a, V, E> DerefMut for CDG<'a, V, E>
where V: Clone {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<'a, V, E> AsEntityGraph<'a, V, E> for CDG<'a, V, E>
where V: Clone {
    fn entity_graph(&self) -> &EntityGraph<'a, V, E> {
        &self.graph
    }
}

impl<'a, V, E> AsEntityGraphMut<'a, V, E> for CDG<'a, V, E>
where V: Clone {
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, V, E> {
        &mut self.graph
    }
}

impl<'a, V, E> CDG<'a, V, E>
where V: Clone,
      E: Clone, {
    pub fn new<G>(cfg: G) -> CDG<'a, V, E>
    where G: AsEntityGraph<'a, V, E> {
        let mut graph = EntityGraph::new();

        let rd = cfg.entity_graph().post_dominators();
        let rdf = rd.dominance_frontier();

        for (vx, vys) in rdf.iter() {
            for vy in vys.iter() {
                if let Some(e) = cfg.entity_graph().edge(*vx, *vy) {
                    let ey = cfg.entity_graph().entity(*vy);
                    let ex = cfg.entity_graph().entity(*vx);
                    graph.add_relation(ey, ex, e.clone());
                }
            }
        }

        CDG {
            graph,
        }
    }
}
