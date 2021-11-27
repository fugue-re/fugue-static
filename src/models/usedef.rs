use fugue::ir::il::ecode::{Entity, EntityId, Var};

use std::collections::BTreeSet;

use fxhash::FxHashMap as HashMap;
use petgraph::stable_graph::StableDiGraph;

use crate::models::Block;
use crate::graphs::entity::{VertexIndex, EdgeIndex};
use crate::traits::*;

pub struct UseDefs {
    graph: StableDiGraph<Var, EntityId>,
    variables: HashMap<Var, VertexIndex<Var>>,
}

impl Default for UseDefs {
    fn default() -> Self {
        Self {
            graph: StableDiGraph::new(),
            variables: HashMap::default(),
        }
    }
}

impl UseDefs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn var_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn add_var(&mut self, var: Var) -> VertexIndex<Var> {
        if let Some(vx) = self.variables.get(&var) {
            *vx
        } else {
            let vx = self.graph.add_node(var).into();
            self.variables.insert(var, vx);
            vx
        }
    }

    pub fn add_use_def(&mut self, u: Var, d: Var, e: EntityId) -> EdgeIndex<EntityId> {
        let ux = self.add_var(u);
        let dx = self.add_var(d);

        let du = self.graph.add_edge(*dx, *ux, e);

        du.into()
    }

    pub fn add_block(&mut self, block: Entity<Block>) {
        let pid = EntityId::new("phi", block.location().clone());
        for (d, us) in block.phis().iter() {
            for u in us {
                self.add_use_def(*u, *d, pid.clone());
            }
        }

        let mut ds = BTreeSet::new();
        let mut us = BTreeSet::new();

        for op in block.operations() {
            let eid = op.id().clone();

            op.defined_and_used_variables_with(&mut ds, &mut us);

            for d in ds.iter() {
                for u in us.iter() {
                    self.add_use_def(**u, **d, eid.clone());
                }
            }

            ds.clear();
            us.clear();
        }
    }
}
