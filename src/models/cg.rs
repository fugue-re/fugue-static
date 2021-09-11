use fugue::ir::il::ecode::{EntityId, Location};

use petgraph::graph::NodeIndex;

use std::borrow::{Borrow, BorrowMut, Cow};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::models::Function;
use crate::types::{EntityRefMap, EntityGraph};
use crate::traits::dominance::*;
use crate::traits::{EntityRef, IntoEntityRef};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CallVia(Option<EntityId>);

impl Default for CallVia {
    fn default() -> Self {
        Self(None)
    }
}

impl From<Option<EntityId>> for CallVia {
    fn from(v: Option<EntityId>) -> Self {
        Self(v)
    }
}

impl From<CallVia> for Option<EntityId> {
    fn from(v: CallVia) -> Self {
        v.0
    }
}

#[derive(Clone, Default)]
pub struct CG<'e> {
    graph: EntityGraph<CallVia>,
    entity_mapping: HashMap<EntityId, NodeIndex>,
    functions: EntityRefMap<'e, Function>,
}

impl<'e> Borrow<EntityGraph<CallVia>> for CG<'e> {
    fn borrow(&self) -> &EntityGraph<CallVia> {
        &self.graph
    }
}

impl<'e> Borrow<EntityGraph<CallVia>> for &'_ CG<'e> {
    fn borrow(&self) -> &EntityGraph<CallVia> {
        &self.graph
    }
}

impl<'e> Borrow<EntityGraph<CallVia>> for &'_ mut CG<'e> {
    fn borrow(&self) -> &EntityGraph<CallVia> {
        &self.graph
    }
}

impl<'e> BorrowMut<EntityGraph<CallVia>> for CG<'e> {
    fn borrow_mut(&mut self) -> &mut EntityGraph<CallVia> {
        &mut self.graph
    }
}

impl<'e> BorrowMut<EntityGraph<CallVia>> for &'_ mut CG<'e> {
    fn borrow_mut(&mut self) -> &mut EntityGraph<CallVia> {
        &mut self.graph
    }
}

impl<'e> AsRef<EntityGraph<CallVia>> for CG<'e> {
    fn as_ref(&self) -> &EntityGraph<CallVia> {
        &self.graph
    }
}

impl<'e> AsMut<EntityGraph<CallVia>> for CG<'e> {
    fn as_mut(&mut self) -> &mut EntityGraph<CallVia> {
        &mut self.graph
    }
}

impl<'e> Deref for CG<'e> {
    type Target = EntityGraph<CallVia>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'e> DerefMut for CG<'e> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<'e> CG<'e> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cloned<'a>(&self) -> CG<'a> {
        CG {
            graph: self.graph.clone(),
            entity_mapping: self.entity_mapping.clone(),
            functions: self.functions.iter()
                .map(|(id, fcn)| (id.clone(), Cow::Owned(fcn.as_ref().clone())))
                .collect(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn add_function<T>(&mut self, fcn: T) -> NodeIndex
    where T: IntoEntityRef<'e, T=Function> {
        let fcn = fcn.into_entity_ref();
        if let Some(nx) = self.entity_mapping.get(fcn.id()) {
            *nx
        } else {
            let id = fcn.id();
            let nx = self.graph.add_node(id.clone());
            self.entity_mapping.insert(id.clone(), nx);
            self.functions.insert(id.clone(), fcn);
            nx
        }
    }

    fn add_call_aux<F1, F2, V>(&mut self, src: F1, tgt: F2, via: V)
    where F1: IntoEntityRef<'e, T=Function>,
          F2: IntoEntityRef<'e, T=Function>,
          V: Into<CallVia> {
        let sx = self.add_function(src);
        let tx = self.add_function(tgt);

        self.graph.add_edge(sx, tx, via.into());
    }

    pub fn function_node(&self, function: &EntityId) -> Option<NodeIndex> {
        self.entity_mapping.get(function).copied()
    }

    pub fn function_at(&self, index: NodeIndex) -> &EntityRef<'e, Function> {
        let eid = &self.graph[index];
        &self.functions[eid]
    }

    pub fn function_at_mut(&mut self, index: NodeIndex) -> &mut EntityRef<'e, Function> {
        let eid = &self.graph[index];
        self.functions.get_mut(eid).unwrap()
    }

    pub fn function<L: Into<Location>>(&self, location: L) -> Option<(NodeIndex, &EntityRef<'e, Function>)> {
        let id = EntityId::new("fcn", location.into());
        let eid = *self.entity_mapping.get(&id)?;
        let blk = self.functions.get(&id)?;
        Some((eid, blk))
    }

    pub fn function_mut<L: Into<Location>>(&mut self, location: L) -> Option<(NodeIndex, &mut EntityRef<'e, Function>)> {
        let id = EntityId::new("fcn", location.into());
        let eid = *self.entity_mapping.get(&id)?;
        let blk = self.functions.get_mut(&id)?;
        Some((eid, blk))
    }

    pub fn functions(&self) -> &EntityRefMap<'e, Function> {
        &self.functions
    }

    pub fn functions_mut(&mut self) -> &mut EntityRefMap<'e, Function> {
        &mut self.functions
    }

    pub fn entity_graph(&self) -> &EntityGraph<CallVia> {
        &self.graph
    }

    pub fn entity_graph_mut(&mut self) -> &mut EntityGraph<CallVia> {
        &mut self.graph
    }

    pub fn add_call<F1, F2>(&mut self, src: F1, tgt: F2)
    where F1: IntoEntityRef<'e, T=Function>,
          F2: IntoEntityRef<'e, T=Function> {
        self.add_call_aux(src, tgt, None)
    }

    pub fn add_call_via<F1, F2>(&mut self, src: F1, tgt: F2, via: EntityId)
    where F1: IntoEntityRef<'e, T=Function>,
          F2: IntoEntityRef<'e, T=Function> {
        self.add_call_aux(src, tgt, Some(via))
    }

    pub fn dominance_tree(&self) -> DominanceTree {
        self.graph.dominance_tree()
    }

    pub fn dominance_frontier(&self) -> DominanceFrontier {
        self.graph.dominance_frontier()
    }

    pub fn dominance(&self) -> Dominance {
        self.graph.dominance()
    }
}
