use fugue::ir::il::ecode::EntityId;

use std::collections::HashSet;
use std::fmt;
use std::ops::{Deref, DerefMut};

use crate::models::Function;
use crate::graphs::entity::{AsEntityGraph, AsEntityGraphMut, EntityGraph, VertexIndex};
use crate::traits::IntoEntityRef;
use crate::types::EntityRef;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CallVia(Option<EntityId>);

impl Default for CallVia {
    fn default() -> Self {
        Self(None)
    }
}

impl fmt::Display for CallVia {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref id) = &self.0 {
            write!(f, "calls via: {}", id)
        } else {
            write!(f, "calls")
        }
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
pub struct CG<'a> {
    graph: EntityGraph<'a, Function, CallVia>,
}

impl<'a> AsEntityGraph<'a, Function, CallVia> for CG<'a> {
    fn entity_graph(&self) -> &EntityGraph<'a, Function, CallVia> {
        &self.graph
    }
}

impl<'a> AsEntityGraphMut<'a, Function, CallVia> for CG<'a> {
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, Function, CallVia> {
        &mut self.graph
    }
}

impl<'a> Deref for CG<'a> {
    type Target = EntityGraph<'a, Function, CallVia>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'a> DerefMut for CG<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<'a> CG<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cloned<'b>(&self) -> CG<'b> {
        CG { graph: self.graph.cloned() }
    }

    pub fn add_function<T>(&mut self, fcn: T) -> VertexIndex<Function>
    where T: IntoEntityRef<'a, T=Function> {
        self.graph.add_entity(fcn)
    }

    fn add_call_aux<F1, F2, V>(&mut self, src: F1, tgt: F2, via: V)
    where F1: IntoEntityRef<'a, T=Function>,
          F2: IntoEntityRef<'a, T=Function>,
          V: Into<CallVia> {
        let sx = self.add_function(src);
        let tx = self.add_function(tgt);

        self.graph.add_vertex_relation(sx, tx, via.into());
    }

    fn called_from_aux(&self, function: &EntityId, bound: usize, called: &mut HashSet<EntityRef<'a, Function>>) {
        if bound == 0 {
            return
        }

        if let Some(nx) = self.entity_vertex(function) {
            for (nd, _) in self.graph.successors(nx) {
                let eid = self.graph.entity(nd);
                called.insert(eid.clone()); // cheap; Cow
                self.called_from_aux(eid.id(), bound - 1, called);
            }
        }
    }

    pub fn called_from(&self, function: &EntityId, bound: usize) -> HashSet<EntityRef<'a, Function>> {
        let mut called = HashSet::new();
        self.called_from_aux(function, bound, &mut called);
        called
    }

    /*
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
    */

    pub fn add_call<F1, F2>(&mut self, src: F1, tgt: F2)
    where F1: IntoEntityRef<'a, T=Function>,
          F2: IntoEntityRef<'a, T=Function> {
        self.add_call_aux(src, tgt, None)
    }

    pub fn add_call_via<F1, F2>(&mut self, src: F1, tgt: F2, via: EntityId)
    where F1: IntoEntityRef<'a, T=Function>,
          F2: IntoEntityRef<'a, T=Function> {
        self.add_call_aux(src, tgt, Some(via))
    }
}
