use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use fugue::ir::il::ecode::EntityId;

use crate::models::CFG;
use crate::models::cfg::BranchKind;

use crate::traits::dominance::*;
use crate::types::EntityGraph;

#[derive(Clone, Default)]
pub struct CDG {
    pub(crate) graph: EntityGraph<BranchKind>,
    pub(crate) entity_mapping: HashMap<EntityId, NodeIndex>,
}

impl Borrow<EntityGraph<BranchKind>> for CDG {
    fn borrow(&self) -> &EntityGraph<BranchKind> {
        &self.graph
    }
}

impl Borrow<EntityGraph<BranchKind>> for &'_ CDG {
    fn borrow(&self) -> &EntityGraph<BranchKind> {
        &self.graph
    }
}

impl Borrow<EntityGraph<BranchKind>> for &'_ mut CDG {
    fn borrow(&self) -> &EntityGraph<BranchKind> {
        &self.graph
    }
}

impl BorrowMut<EntityGraph<BranchKind>> for CDG {
    fn borrow_mut(&mut self) -> &mut EntityGraph<BranchKind> {
        &mut self.graph
    }
}

impl BorrowMut<EntityGraph<BranchKind>> for &'_ mut CDG {
    fn borrow_mut(&mut self) -> &mut EntityGraph<BranchKind> {
        &mut self.graph
    }
}

impl AsRef<EntityGraph<BranchKind>> for CDG {
    fn as_ref(&self) -> &EntityGraph<BranchKind> {
        &self.graph
    }
}

impl AsMut<EntityGraph<BranchKind>> for CDG {
    fn as_mut(&mut self) -> &mut EntityGraph<BranchKind> {
        &mut self.graph
    }
}

impl Deref for CDG {
    type Target = EntityGraph<BranchKind>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for CDG {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl CDG {
    pub fn new(cfg: &CFG) -> CDG {
        let mut entity_mapping = HashMap::new();
        let mut rcfg = EntityGraph::new();

        for eid in cfg.node_weights() {
            entity_mapping.insert(eid.clone(), rcfg.add_node(eid.clone()));
        }

        let mut graph = rcfg.clone();

        for er in cfg.edge_references() {
            let se = &cfg[er.source()];
            let ee = &cfg[er.target()];

            rcfg.add_edge(entity_mapping[ee], entity_mapping[se], *er.weight());
        }

        let df = rcfg.dominance_frontier();

        for (nx, nys) in df.into_iter() {
            for ny in nys.into_iter() {
                if let Some(w) = rcfg.find_edge(ny, nx).and_then(|ex| rcfg.edge_weight(ex)) {
                    graph.add_edge(ny, nx, *w);
                }
            }
        }

        CDG {
            graph,
            entity_mapping,
        }
    }
}
