use fugue::ir::il::ecode::{Entity, Stmt, Var};

use std::collections::BTreeSet;

use fxhash::FxHashMap as HashMap;

use petgraph::Direction;
use petgraph::stable_graph::StableDiGraph;

use crate::graphs::entity::{EdgeIndex, VertexIndex};
use crate::models::Block;
use crate::traits::*;
use crate::types::EntityRef;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalDependency<'a> {
    Phi(EntityRef<'a, Block>, usize),
    Stmt(EntityRef<'a, Stmt>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dependency<'a> {
    Bound(LocalDependency<'a>),
    Free,
}

impl<'a> Dependency<'a> {
    pub fn is_free(&self) -> bool {
        matches!(self, Self::Free)
    }

    pub fn is_bound(&self) -> bool {
        matches!(self, Self::Bound(_))
    }

    pub fn update(&mut self, access: Dependency<'a>) {
        if self.is_free() && !access.is_free() {
            *self = access
        }
    }
}

#[derive(Debug, Clone)]
pub struct UseDefs<'a> {
    graph: StableDiGraph<Var, LocalDependency<'a>>,
    variables: HashMap<Var, (VertexIndex<Var>, Dependency<'a>)>,
}

impl<'a> Default for UseDefs<'a> {
    fn default() -> Self {
        Self {
            graph: StableDiGraph::new(),
            variables: HashMap::default(),
        }
    }
}

impl<'a> UseDefs<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn var_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn add_var(&mut self, var: Var, access: Dependency<'a>) -> VertexIndex<Var> {
        if let Some((vx, caccess)) = self.variables.get_mut(&var) {
            caccess.update(access);
            *vx
        } else {
            let vx = self.graph.add_node(var).into();
            self.variables.insert(var, (vx, access));
            vx
        }
    }

    pub fn add_use_def(
        &mut self,
        u: Var,
        d: Var,
        dep: LocalDependency<'a>,
    ) -> EdgeIndex<LocalDependency<'a>> {
        let ux = self.add_var(u, Dependency::Free);
        let dx = self.add_var(d, Dependency::Bound(dep.clone()));

        let du = self.graph.add_edge(*ux, *dx, dep);

        du.into()
    }

    pub fn add_block(&mut self, block: &'a Entity<Block>) {
        for (i, (d, us)) in block.phis().iter().enumerate() {
            for u in us {
                let r = block.into_entity_ref();
                self.add_use_def(*u, *d, LocalDependency::Phi(r, i));
            }
        }

        let mut ds = BTreeSet::new();
        let mut us = BTreeSet::new();

        for op in block.operations() {
            op.defined_and_used_variables_with(&mut ds, &mut us);

            for d in ds.iter() {
                for u in us.iter() {
                    let r = op.into_entity_ref();
                    self.add_use_def(**u, **d, LocalDependency::Stmt(r));
                }
            }

            ds.clear();
            us.clear();
        }
    }

    pub fn add_stmt(&mut self, stmt: &'a Entity<Stmt>) {
        let (ds, us) = stmt.defined_and_used_variables::<BTreeSet<_>>();

        for d in ds.iter() {
            for u in us.iter() {
                let r = stmt.into_entity_ref();
                self.add_use_def(**u, **d, LocalDependency::Stmt(r));
            }
        }
    }

    pub fn dependents(&self, var: &Var) -> Vec<(&Var, &LocalDependency<'a>)> {
        let mut deps = Vec::new();
        if let Some((vx, _)) = self.variables.get(var) {
            let mut walker = self.graph.neighbors_directed(**vx, Direction::Outgoing).detach();

            while let Some((e, n)) = walker.next(&self.graph) {
                let v = &self.graph[n];
                let d = &self.graph[e];

                deps.push((v, d));
            }
        }
        deps
    }

    pub fn dependencies(&self, var: &Var) -> Vec<(&Var, &Dependency<'a>)> {
        let mut deps = Vec::new();
        if let Some((vx, _)) = self.variables.get(var) {
            for n in self.graph.neighbors_directed(**vx, Direction::Incoming) {
                let v = &self.graph[n];
                let d = &self.variables[v].1;

                deps.push((v, d));
            }
        }
        deps
    }
}