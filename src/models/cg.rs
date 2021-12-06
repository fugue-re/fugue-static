use std::collections::HashSet;
use std::fmt;
use std::ops::{Deref, DerefMut};

use crate::models::Function;
use crate::graphs::entity::{AsEntityGraph, AsEntityGraphMut, EntityGraph, VertexIndex};
use crate::types::{Id, Identifiable, EntityRef, IntoEntityRef};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CallVia<V>(Option<Id<V>>);

impl<V> Clone for CallVia<V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<V> Default for CallVia<V> {
    fn default() -> Self {
        Self(None)
    }
}

impl<V> fmt::Display for CallVia<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref id) = &self.0 {
            write!(f, "calls via: {}", id)
        } else {
            write!(f, "calls")
        }
    }
}

impl<V> From<Option<Id<V>>> for CallVia<V> {
    fn from(v: Option<Id<V>>) -> Self {
        Self(v)
    }
}

impl<V> From<CallVia<V>> for Option<Id<V>> {
    fn from(v: CallVia<V>) -> Self {
        v.0
    }
}

pub struct CG<'a, V> {
    graph: EntityGraph<'a, Function, CallVia<V>>,
}

impl<'a, V> Clone for CG<'a, V> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
    }
}

impl<'a, V> Default for CG<'a, V> {
    fn default() -> Self {
        Self {
            graph: EntityGraph::new()
        }
    }
}

impl<'a, V> AsEntityGraph<'a, Function, CallVia<V>> for CG<'a, V> {
    fn entity_graph(&self) -> &EntityGraph<'a, Function, CallVia<V>> {
        &self.graph
    }
}

impl<'a, V> AsEntityGraphMut<'a, Function, CallVia<V>> for CG<'a, V> {
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, Function, CallVia<V>> {
        &mut self.graph
    }
}

impl<'a, V> Deref for CG<'a, V> {
    type Target = EntityGraph<'a, Function, CallVia<V>>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<'a, V> DerefMut for CG<'a, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<'a, V> CG<'a, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cloned<'b>(&self) -> CG<'b, V> {
        CG { graph: self.graph.cloned() }
    }

    pub fn add_function<T>(&mut self, fcn: T) -> VertexIndex<Function>
    where T: IntoEntityRef<'a, T=Function> {
        self.graph.add_entity(fcn)
    }

    fn add_call_aux<F1, F2, T>(&mut self, src: F1, tgt: F2, via: T)
    where F1: IntoEntityRef<'a, T=Function>,
          F2: IntoEntityRef<'a, T=Function>,
          T: Into<CallVia<V>> {
        let sx = self.add_function(src);
        let tx = self.add_function(tgt);

        self.graph.add_vertex_relation(sx, tx, via.into());
    }

    fn called_from_aux(&self, function: Id<Function>, bound: usize, called: &mut HashSet<EntityRef<'a, Function>>) {
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

    pub fn called_from(&self, function: Id<Function>, bound: usize) -> HashSet<EntityRef<'a, Function>> {
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

    pub fn add_call_via<F1, F2>(&mut self, src: F1, tgt: F2, via: Id<V>)
    where F1: IntoEntityRef<'a, T=Function>,
          F2: IntoEntityRef<'a, T=Function> {
        self.add_call_aux(src, tgt, Some(via))
    }
}
