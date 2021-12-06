use fugue::ir::il::ecode::Location;
use fxhash::FxBuildHasher;

use petgraph::algo::{has_path_connecting, kosaraju_scc, DfsSpace};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::{Neighbors, StableDiGraph, WalkNeighbors};
use petgraph::Direction;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::marker::PhantomData;
use std::ops::Deref;

use indexmap::map::IterMut as EntityIterMutInner;

use crate::graphs::algorithms::dominance::{Dominance, Dominators};
use crate::graphs::algorithms::simple_cycles::SimpleCycles;
use crate::graphs::traversals::heuristic::{PostOrder, RevPostOrder};

use crate::types::{Id, Identifiable, Entity, EntityRef, IntoEntityRef};

type HashSet<K> = std::collections::HashSet<K, FxBuildHasher>;
type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
type IndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;
type IndexSet<K> = indexmap::IndexSet<K, FxBuildHasher>;

#[derive(educe::Educe)]
#[educe(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VertexIndex<T> {
    index: NodeIndex,
    #[educe(
        Debug(ignore),
        PartialEq(ignore),
        Eq(ignore),
        PartialOrd(ignore),
        Ord(ignore),
        Hash(ignore)
    )]
    marker: PhantomData<T>,
}

impl<T> AsRef<NodeIndex> for VertexIndex<T> {
    fn as_ref(&self) -> &NodeIndex {
        &self.index
    }
}

impl<T> Deref for VertexIndex<T> {
    type Target = NodeIndex;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl<T> From<VertexIndex<T>> for NodeIndex {
    fn from(slf: VertexIndex<T>) -> Self {
        slf.index
    }
}

impl<T> From<NodeIndex> for VertexIndex<T> {
    fn from(index: NodeIndex) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }
}

#[derive(educe::Educe)]
#[educe(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EdgeIndex<T> {
    index: petgraph::graph::EdgeIndex,
    #[educe(
        Debug(ignore),
        PartialEq(ignore),
        Eq(ignore),
        PartialOrd(ignore),
        Ord(ignore),
        Hash(ignore)
    )]
    marker: PhantomData<T>,
}

impl<T> AsRef<petgraph::graph::EdgeIndex> for EdgeIndex<T> {
    fn as_ref(&self) -> &petgraph::graph::EdgeIndex {
        &self.index
    }
}

impl<T> Deref for EdgeIndex<T> {
    type Target = petgraph::graph::EdgeIndex;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl<T> From<EdgeIndex<T>> for petgraph::graph::EdgeIndex {
    fn from(slf: EdgeIndex<T>) -> Self {
        slf.index
    }
}

impl<T> From<petgraph::graph::EdgeIndex> for EdgeIndex<T> {
    fn from(index: petgraph::graph::EdgeIndex) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }
}

impl<'a, V, E> AsRef<StableDiGraph<Id<V>, E>> for EntityGraph<'a, V, E>
where
    V: Clone,
{
    fn as_ref(&self) -> &StableDiGraph<Id<V>, E> {
        &self.entity_graph
    }
}

impl<'a, V, E> AsMut<StableDiGraph<Id<V>, E>> for EntityGraph<'a, V, E>
where
    V: Clone,
{
    fn as_mut(&mut self) -> &mut StableDiGraph<Id<V>, E> {
        &mut self.entity_graph
    }
}

#[derive(Debug, Clone)]
pub struct EntityGraph<'a, V, E>
where
    V: Clone,
{
    pub(crate) entity_graph: StableDiGraph<Id<V>, E>,

    pub(crate) entity_roots: IndexSet<(Id<V>, VertexIndex<V>)>,
    pub(crate) entity_leaves: IndexSet<(Id<V>, VertexIndex<V>)>,

    pub(crate) entities: IndexMap<Id<V>, (VertexIndex<V>, Cow<'a, Entity<V>>)>,
    pub(crate) entity_versions: HashMap<Location, usize>,
}

impl<'a, V, E> Default for EntityGraph<'a, V, E>
where
    V: Clone,
{
    fn default() -> Self {
        Self {
            entity_graph: Default::default(),
            entity_roots: Default::default(),
            entity_leaves: Default::default(),
            entities: Default::default(),
            entity_versions: Default::default(),
        }
    }
}

pub trait VisitEntityGraph<'a, V, E> {
    fn visit_entity_vertex(&mut self, vertex: VertexIndex<V>);
    fn visit_entity_edge(&mut self, edge: EdgeIndex<E>);
}

impl<'a, V, E> EntityGraph<'a, V, E>
where
    E: Clone,
    V: Clone,
{
    pub fn cloned<'b>(&self) -> EntityGraph<'b, V, E> {
        EntityGraph {
            entity_graph: self.entity_graph.clone(),
            entity_roots: self.entity_roots.clone(),
            entity_leaves: self.entity_leaves.clone(),
            entities: self
                .entities
                .iter()
                .map(|(id, (vx, v))| (id.clone(), (*vx, Cow::Owned(v.as_ref().clone()))))
                .collect(),
            entity_versions: self.entity_versions.clone(),
        }
    }

    pub fn subgraph<'g, I, E2, F>(&'g self, vertices: I) -> EntityGraph<'g, V, E>
    where
        I: IntoIterator<Item = VertexIndex<V>>,
    {
        self.subgraph_with(vertices, |_ex, e| e.clone())
    }
}

impl<'a, V, E> EntityGraph<'a, V, E>
where
    V: Clone,
{
    pub fn subgraph_with<'g, I, E2, F>(
        &'g self,
        vertices: I,
        mut edge_map: F,
    ) -> EntityGraph<'g, V, E2>
    where
        I: IntoIterator<Item = VertexIndex<V>>,
        F: FnMut(EdgeIndex<E>, &'g E) -> E2,
    {
        let retained = vertices.into_iter().collect::<BTreeSet<_>>();
        EntityGraph {
            entity_graph: self.entity_graph.filter_map(
                |nx, e| {
                    if retained.contains(&VertexIndex::from(nx)) {
                        Some(e.clone())
                    } else {
                        None
                    }
                },
                |ex, e| Some(edge_map(EdgeIndex::from(ex), e)),
            ),
            entity_roots: self
                .entity_roots
                .iter()
                .filter_map(|(e, v)| {
                    if retained.contains(v) {
                        Some((e.clone(), *v))
                    } else {
                        None
                    }
                })
                .collect(),
            entity_leaves: self
                .entity_leaves
                .iter()
                .filter_map(|(e, v)| {
                    if retained.contains(v) {
                        Some((e.clone(), *v))
                    } else {
                        None
                    }
                })
                .collect(),
            entities: self
                .entities
                .iter()
                .filter_map(|(id, (vx, e))| {
                    if retained.contains(vx) {
                        Some((id.clone(), (*vx, Cow::Borrowed(e.as_ref()))))
                    } else {
                        None
                    }
                })
                .collect(),
            entity_versions: self.entity_versions.clone(), // TODO: only keep relevant
        }
    }

    pub fn subgraph_view<'g, I>(&'g self, vertices: I) -> EntityGraph<'g, V, ()>
    where
        I: IntoIterator<Item = VertexIndex<V>>,
    {
        self.subgraph_with(vertices, |_ex, _e| ())
    }

    pub fn view<'g>(&'g self) -> EntityGraph<'g, V, ()> {
        EntityGraph {
            entity_graph: self.entity_graph.map(|_nx, e| e.clone(), |_, _| ()),
            entity_roots: self.entity_roots.clone(),
            entity_leaves: self.entity_leaves.clone(),
            entities: self
                .entities
                .iter()
                .map(|(id, (vx, v))| (id.clone(), (*vx, Cow::Borrowed(v.as_ref()))))
                .collect(),
            entity_versions: self.entity_versions.clone(),
        }
    }

    pub fn structure<'g>(&'g self) -> StableDiGraph<(), ()> {
        self.entity_graph.map(|_, _| (), |_, _| ())
    }
}

impl<'a, V, E> EntityGraph<'a, V, E>
where
    V: Clone,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn edge_count(&self) -> usize {
        self.entity_graph.edge_count()
    }

    #[inline(always)]
    pub fn vertex_count(&self) -> usize {
        self.entity_graph.node_count()
    }

    pub fn entity_vertex(&self, id: Id<V>) -> Option<VertexIndex<V>> {
        self.entities.get(&id).map(|v| v.0)
    }

    pub fn entity(&self, vertex: VertexIndex<V>) -> &EntityRef<'a, V> {
        let id = &self.entity_graph[*vertex];
        &self.entities[id].1
    }

    pub fn entity_mut(&mut self, vertex: VertexIndex<V>) -> &mut EntityRef<'a, V> {
        let id = &self.entity_graph[*vertex];
        &mut self.entities[id].1
    }

    pub fn contains_edge(&self, vs: VertexIndex<V>, vt: VertexIndex<V>) -> bool {
        self.entity_graph.contains_edge(*vs, *vt)
    }

    pub fn edge(&self, vs: VertexIndex<V>, vt: VertexIndex<V>) -> Option<&E> {
        self.entity_graph
            .find_edge(*vs, *vt)
            .and_then(|e| self.entity_graph.edge_weight(e))
    }

    pub fn edge_mut(&mut self, vs: VertexIndex<V>, vt: VertexIndex<V>) -> Option<&mut E> {
        self.entity_graph
            .find_edge(*vs, *vt)
            .and_then(move |e| self.entity_graph.edge_weight_mut(e))
    }

    pub fn edge_relation(&self, ex: EdgeIndex<E>) -> Option<(VertexIndex<V>, VertexIndex<V>, &E)> {
        let (s, e) = self.entity_graph.edge_endpoints(*ex)?;
        self.entity_graph.edge_weight(*ex)
            .map(|v| (s.into(), e.into(), v))
    }

    pub fn edge_relation_mut(&mut self, ex: EdgeIndex<E>) -> Option<(VertexIndex<V>, VertexIndex<V>, &mut E)> {
        let (s, e) = self.entity_graph.edge_endpoints(*ex)?;
        self.entity_graph.edge_weight_mut(*ex)
            .map(|v| (s.into(), e.into(), v))
    }
    
    pub fn add_entity<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let er = entity.into_entity_ref();
        if let Some(nx) = self.entities.get(&er.id()) {
            nx.0
        } else {
            let id = er.id();
            let nx = self.entity_graph.add_node(id).into();

            self.entities.insert(id, (nx, er));

            nx
        }
    }

    pub fn remove_vertex(&mut self, vertex: VertexIndex<V>) -> EntityRef<'a, V> {
        let id = self.entity_graph.remove_node(*vertex).unwrap();
        self.entities.remove(&id).unwrap().1
    }

    pub fn add_entity_alias<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let er = entity.into_entity_ref();
        let ner = Entity::new(er.id().tag(), er.into_owned().into_value());

        let id = ner.id();
        let nx = self.entity_graph.add_node(id).into();

        self.entities.insert(id, (nx, ner.into_entity_ref()));

        nx
    }

    pub fn add_root_entity<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let er = entity.into_entity_ref();
        let id = er.id().clone();
        let nx = self.add_entity(er);
        self.entity_roots.insert((id, nx));
        nx
    }

    pub fn add_leaf_entity<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let er = entity.into_entity_ref();
        let id = er.id().clone();
        let nx = self.add_entity(er);
        self.entity_leaves.insert((id, nx));
        nx
    }

    pub fn add_vertex_relation(&mut self, vs: VertexIndex<V>, vt: VertexIndex<V>, relation: E) {
        self.entity_graph.add_edge(*vs, *vt, relation);
    }

    pub fn add_relation<S, T>(
        &mut self,
        source: S,
        target: T,
        relation: E,
    ) -> (VertexIndex<V>, VertexIndex<V>)
    where
        S: IntoEntityRef<'a, T = V>,
        T: IntoEntityRef<'a, T = V>,
    {
        let sid = self.add_entity(source);
        let tid = self.add_entity(target);

        self.entity_graph.add_edge(*sid, *tid, relation);

        (sid, tid)
    }

    pub fn remove_relation<S, T>(&mut self, source: S, target: T) -> Option<E>
    where
        S: IntoEntityRef<'a, T = V>,
        T: IntoEntityRef<'a, T = V>,
    {
        let sid = self.entity_vertex(source.into_entity_ref().id())?;
        let tid = self.entity_vertex(target.into_entity_ref().id())?;

        let eid = self.entity_graph.find_edge(*sid, *tid)?;

        self.entity_graph.remove_edge(eid)
    }

    pub fn entities<'g>(&'g self) -> EntityRefIter<'g, 'a, V, E> {
        EntityRefIter::new(self)
    }

    pub fn entities_mut<'g>(&'g mut self) -> EntityRefIterMut<'g, 'a, V, E> {
        EntityRefIterMut::new(self)
    }

    pub fn root_entities<'g>(&'g self) -> EntityRootOrLeafIter<'g, 'a, V, E> {
        EntityRootOrLeafIter::new_roots(self)
    }

    pub fn leaf_entities<'g>(&'g self) -> EntityRootOrLeafIter<'g, 'a, V, E> {
        EntityRootOrLeafIter::new_leaves(self)
    }

    pub fn has_root_entities(&self) -> bool {
        !self.entity_roots.is_empty()
    }

    pub fn has_leaf_entities(&self) -> bool {
        !self.entity_roots.is_empty()
    }

    pub fn root_vertices<'g>(&'g self) -> Vec<VertexIndex<V>> {
        self.entity_roots.iter().map(|(_, v)| *v).collect()
    }

    pub fn leaf_vertices<'g>(&'g self) -> Vec<VertexIndex<V>> {
        self.entity_leaves.iter().map(|(_, v)| *v).collect()
    }

    pub fn estimate_leaf_entities<'g>(&'g self) -> EntityRefIterFiltered<'g, 'a, V, E> {
        EntityRefIterFiltered::new(self, |g, _, id, _| g.successors(id).next().is_none())
    }

    pub fn estimate_leaf_vertices<'g>(&'g self) -> Vec<VertexIndex<V>> {
        self.entity_graph
            .node_indices()
            .filter_map(|nx| {
                if self.successors(nx.into()).next().is_none() {
                    Some(nx.into())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn make_root_entity(&mut self, vertex: VertexIndex<V>) {
        let id = &self.entity_graph[*vertex];
        self.entity_roots.insert((id.clone(), vertex));
    }

    pub fn make_leaf_entity(&mut self, vertex: VertexIndex<V>) {
        let id = &self.entity_graph[*vertex];
        self.entity_leaves.insert((id.clone(), vertex));
    }

    pub fn find_root_entities(&self) -> HashSet<VertexIndex<V>> {
        let mut roots = self
            .entity_roots
            .iter()
            .map(|(_, r)| *r)
            .collect::<HashSet<VertexIndex<V>>>();

        let starts = self
            .entity_graph
            .externals(Direction::Incoming)
            .map(VertexIndex::from);

        roots.extend(starts);

        let mut dfs_space = DfsSpace::new(&self.entity_graph);
        for scc in kosaraju_scc(&self.entity_graph)
            .into_iter()
            .map(|mut v| v.pop().unwrap())
        {
            if !roots
                .iter()
                .any(|r| has_path_connecting(&self.entity_graph, **r, scc, Some(&mut dfs_space)))
            {
                roots.insert(scc.into());
            }
        }

        roots
    }

    pub fn update_root_entities(&mut self) {
        self.find_root_entities()
            .into_iter()
            .for_each(|vx| self.make_root_entity(vx))
    }

    pub fn predecessors<'g>(&'g self, vertex: VertexIndex<V>) -> VertexEdgeIter<'g, 'a, V, E> {
        VertexEdgeIter::new(
            self,
            self.entity_graph
                .neighbors_directed(vertex.into(), Direction::Incoming),
        )
    }

    pub fn successors<'g>(&'g self, vertex: VertexIndex<V>) -> VertexEdgeIter<'g, 'a, V, E> {
        VertexEdgeIter::new(
            self,
            self.entity_graph
                .neighbors_directed(vertex.into(), Direction::Outgoing),
        )
    }

    pub fn post_order<'g>(&'g self) -> VecDeque<VertexIndex<V>> {
        let mut t = super::traversals::PostOrder::new(self);
        let mut po = VecDeque::with_capacity(self.entities.len());
        while let Some(vx) = t.next(self) {
            po.push_back(vx);
        }
        po
    }

    pub fn reverse_post_order<'g>(&'g self) -> VecDeque<VertexIndex<V>> {
        let mut t = super::traversals::PostOrder::new(self);
        let mut rpo = VecDeque::with_capacity(self.entities.len());
        while let Some(vx) = t.next(self) {
            rpo.push_front(vx);
        }
        rpo
    }

    pub fn estimated_post_order<'g>(&'g self) -> PostOrderIter<'g, 'a, V, E> {
        PostOrderIter::new(self)
    }

    pub fn estimated_reverse_post_order<'g>(&'g self) -> RevPostOrderIter<'g, 'a, V, E> {
        RevPostOrderIter::new(self)
    }

    pub fn back_edges<'g>(&'g self) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let dominators = self.dominators();
        self.back_edges_with(&dominators)
    }

    pub fn back_edges_with<'g>(
        &'g self,
        dominators: &Dominators<V>,
    ) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let mut edges = Vec::new();
        for vx in self.reverse_post_order() {
            if let Some(doms) = dominators
                .dominators(vx)
                .map(|doms| doms.into_iter().collect::<HashSet<_>>())
            {
                for (succ, e) in self.successors(vx).filter(|(succ, _)| doms.contains(succ)) {
                    edges.push((vx, succ, e))
                }
            }
        }
        edges
    }

    pub fn remove_back_edges<'g>(&'g mut self) {
        let dominators = self.dominators();
        self.remove_back_edges_with(&dominators)
    }

    pub fn remove_back_edges_with<'g>(&'g mut self, dominators: &Dominators<V>) {
        for (_, _, e) in self.back_edges_with(dominators) {
            self.entity_graph.remove_edge(*e);
        }
    }

    pub fn natural_loops<'g>(&'g self) -> Vec<NaturalLoop<V, E>> {
        let dominators = self.dominators();
        self.natural_loops_with(&dominators)
    }

    pub fn natural_loops_with<'g>(&'g self, dominators: &Dominators<V>) -> Vec<NaturalLoop<V, E>> {
        let mut loops = Vec::new();

        for (n, s, e) in self.back_edges_with(dominators) {
            loops.push(NaturalLoop::new(self, s, n, e));
        }

        loops
    }

    pub fn strongly_connected_components<'g>(&'g self) -> StronglyConnectedComponents<V> {
        StronglyConnectedComponents(
            kosaraju_scc(&self.entity_graph)
                .into_iter()
                .map(|ccs| ccs.into_iter().map(VertexIndex::from).collect())
                .collect(),
        )
    }

    pub fn outer_loops<'g>(&'g self) -> Vec<Loop<V, E>> {
        let covered = self.entity_graph.node_count();
        let mut back_edges = BTreeMap::new();

        // if any cover the whole of G then just one
        for (s, t, e) in self.back_edges().into_iter() {
            let nl = NaturalLoop::new(self, t, s, e);
            if nl.body().len() == covered {
                return vec![Loop::from(nl)];
            }
            back_edges.insert((s, t), e);
        }

        // TODO: remove SCCs that are not loops
        let sccs = self.strongly_connected_components();

        sccs.into_inner()
            .into_iter()
            .map(|body| {
                let body = body.into_iter().collect::<BTreeSet<_>>();
                let heads =
                    body.iter()
                        .filter_map(|v| {
                            if self.predecessors(*v).any(|(p, _)| {
                                !body.contains(&p) || back_edges.contains_key(&(p, *v))
                            }) {
                                Some(*v)
                            } else {
                                None
                            }
                        })
                        .collect::<BTreeSet<_>>();

                let h = &heads;
                let back_edges = body
                    .iter()
                    .flat_map(|v| {
                        self.successors(*v).filter_map(move |(s, e)| {
                            if h.contains(&s) {
                                Some((*v, s, e))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                Loop::new(heads, back_edges, body)
            })
            .collect()
    }

    pub fn simple_loops<'g>(&'g self) -> Vec<Loop<V, E>> {
        let (_, cycles) = self.topological_simple_cycles(false);
        let back_edges = self
            .back_edges()
            .into_iter()
            .map(|(s, t, e)| ((s, t), e))
            .collect::<BTreeMap<_, _>>();

        cycles
            .into_iter()
            .map(|body| {
                let heads =
                    body.iter()
                        .filter_map(|v| {
                            if self.predecessors(*v).any(|(p, _)| {
                                !body.contains(&p) || back_edges.contains_key(&(p, *v))
                            }) {
                                Some(*v)
                            } else {
                                None
                            }
                        })
                        .collect::<BTreeSet<_>>();

                let h = &heads;
                let back_edges = body
                    .iter()
                    .flat_map(|v| {
                        self.successors(*v).filter_map(move |(s, e)| {
                            if h.contains(&s) {
                                Some((*v, s, e))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                Loop::new(heads, back_edges, body.into_body())
            })
            .collect()
    }

    /*
    pub fn loops<'g>(&'g self) -> Vec<Loop<V>> {
        let sccs = self.strongly_connected_components();
        let doms = self.dominators();
        self.loops_with(&sccs, &doms)
    }

    pub fn loops_with<'g>(&'g self, sccs: &StronglyConnectedComponents<V>, dominators: &Dominators<V>) -> Vec<Loop<V>> {
        let mut loops = Vec::new();

        let back_edges = self.back_edges_with(dominators);
        let mut natural_heads = BTreeMap::<_, BTreeSet<_>>::new();

        for (head, tail, _edge) in back_edges.into_iter() {
            natural_heads.entry(head)
                .or_default()
                .insert(tail);
        }

        for cg in sccs
            .iter()
            .filter(|cs| self.is_component_loop(*cs))
            .cloned()
        {
            let body = cg;
            let mut heads = body
                .iter()
                .filter_map(|v| {
                    if self.predecessors(*v).any(|(p, _)| !body.contains(&p)) {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            // handle the case of G = {A -> B, B -> A} where preds(G, A) == { B } and preds(G, B) == { A }
            if heads.is_empty() {
                heads = body.clone();
            }

            let tails = if heads.iter().all(|head| natural_heads.contains_key(head)) { // only heads!
                heads.iter().map(|head| natural_heads[head].iter().map(|v| *v)).flatten().collect()
            } else {
                // FIXME: empty when no exit from loop!
                body
                    .iter()
                    .filter_map(|v| {
                        if self.successors(*v).any(|(s, _)| !body.contains(&s)) {
                            Some(*v)
                        } else {
                            None
                        }
                    })
                    .collect()
            };

            let l = Loop::new(heads, tails, body);

            loops.push(l);
        }
        loops
    }

    fn is_component_loop<'g>(&'g self, components: &BTreeSet<VertexIndex<V>>) -> bool {
        components.len() > 1
            || matches!(components.iter().next(), Some(v) if self.predecessors(*v).any(|(p, _)| p == *v))
    }
    */
    
    pub fn visit_graph<T: VisitEntityGraph<'a, V, E>>(&self, visitor: &mut T) {
        for vx in self.entity_graph.node_indices().map(VertexIndex::from) {
            visitor.visit_entity_vertex(vx);
        }
        
        for ex in self.entity_graph.edge_indices().map(EdgeIndex::from) {
            visitor.visit_entity_edge(ex);
        }
    }
}

impl<'a, V, E> EntityGraph<'a, V, E>
where
    V: Clone + std::fmt::Display,
    E: Clone,
{
    pub fn unroll_natural_loop(&mut self, l: &NaturalLoop<V, E>, count: usize) {
        if count > 0 {
            let og = self.structure();

            let mut onodes = BTreeMap::new();
            let mut oedges = BTreeMap::new();

            // remove all (external) edges into the loop
            for (_pred, _node, ex) in l.external_predecessors(&self) {
                let edge = self.entity_graph.remove_edge(*ex).unwrap();
                oedges.insert(ex, edge);
            }

            // remove all (successors) from/inside the loop
            for n in l.body() {
                while let Some((_succ, ex)) = self.successors(*n).detach().next(&self) {
                    let edge = self.entity_graph.remove_edge(*ex).unwrap();
                    oedges.insert(ex, edge);
                }
            }

            for n in l.body() {
                onodes.insert(*n, self.remove_vertex(*n));
            }

            let mut nhead = None;
            let mut ptail = None::<VertexIndex<V>>;
            let mut mapping = BTreeMap::new();

            for i in 0..count {
                for node in l.body() {
                    let entity = onodes[node].clone();
                    let nnode = self.add_entity_alias(entity);

                    mapping.insert(*node, nnode);
                }

                if i == 0 {
                    nhead = Some((l.head(), mapping[&l.head()]));
                }

                for node in l.body() {
                    let nnode = mapping[node];
                    let mut succs = og.neighbors_directed(**node, Direction::Outgoing).detach();

                    while let Some((e, succ)) = succs.next(&og) {
                        // skip the back-edge
                        if e == *l.edge() {
                            continue;
                        }

                        let edge = oedges[&EdgeIndex::from(e)].clone();
                        if let Some(nsucc) = mapping.get(&VertexIndex::from(succ)) {
                            self.entity_graph.add_edge(*nnode, **nsucc, edge);
                        } else {
                            self.entity_graph.add_edge(*nnode, succ, edge);
                        }
                    }
                }

                // transform back-edge
                if let Some(ptail) = ptail {
                    let edge = oedges[&l.edge()].clone();
                    self.entity_graph
                        .add_edge(*ptail, *mapping[&l.head()], edge);
                }

                ptail = Some(mapping[&l.tail()]);

                mapping.clear();
            }

            // add external predecessors
            if let Some((ohead, nhead)) = nhead {
                let mut it = og.neighbors_directed(*ohead, Direction::Incoming).detach();
                while let Some((edge, pred)) = it.next(&og) {
                    if l.body().contains(&pred.into()) {
                        continue;
                    }

                    let edge = oedges[&edge.into()].clone();
                    self.entity_graph.add_edge(pred, *nhead, edge);
                }
            }
        } else {
            self.entity_graph.remove_edge(*l.edge());
        }
    }

    pub fn unroll_loop(&mut self, l: &Loop<V, E>, count: usize) {
        if count > 0 {
            // in this case we have to account for sharing of loop bodies!
            let og = self.structure();

            let mut onodes = BTreeMap::new();
            let mut oedges = BTreeMap::new();

            let nloops = l.natural_loops(self);

            // remove all (external) edges into the loop
            for (_pred, _node, ex) in l.external_predecessors(&self) {
                let edge = self.entity_graph.remove_edge(*ex).unwrap();
                oedges.insert(ex, edge);
            }

            // remove all (successors) from/inside the loop
            for n in l.body() {
                while let Some((_succ, ex)) = self.successors(*n).detach().next(&self) {
                    let edge = self.entity_graph.remove_edge(*ex).unwrap();
                    oedges.insert(ex, edge);
                }
            }

            for n in l.body() {
                onodes.insert(*n, self.remove_vertex(*n));
            }

            let mut nheads = Vec::new();

            for nl in nloops.into_iter() {
                let mut ptail = None::<VertexIndex<V>>;
                let mut mapping = BTreeMap::new();

                for i in 0..count {
                    for node in nl.body() {
                        let entity = onodes[node].clone();
                        let nnode = self.add_entity_alias(entity);

                        mapping.insert(*node, nnode);
                    }

                    if i == 0 {
                        nheads.push((nl.head(), mapping[&nl.head()]));
                    }

                    for node in nl.body() {
                        let nnode = mapping[node];
                        let mut succs = og.neighbors_directed(**node, Direction::Outgoing).detach();

                        while let Some((e, succ)) = succs.next(&og) {
                            // skip the back-edge
                            if e == *nl.edge() {
                                continue;
                            }

                            let edge = oedges[&EdgeIndex::from(e)].clone();
                            if let Some(nsucc) = mapping.get(&VertexIndex::from(succ)) {
                                self.entity_graph.add_edge(*nnode, **nsucc, edge);
                            } else {
                                self.entity_graph.add_edge(*nnode, succ, edge);
                            }
                        }
                    }

                    // transform back-edge
                    if let Some(ptail) = ptail {
                        let edge = oedges[&nl.edge()].clone();
                        self.entity_graph
                            .add_edge(*ptail, *mapping[&nl.head()], edge);
                    }

                    ptail = Some(mapping[&nl.tail()]);

                    mapping.clear();
                }
            }

            // add external predecessors
            for (ohead, nhead) in nheads.into_iter() {
                let mut it = og.neighbors_directed(*ohead, Direction::Incoming).detach();
                while let Some((edge, pred)) = it.next(&og) {
                    if l.body().contains(&pred.into()) {
                        continue;
                    }

                    let edge = oedges[&edge.into()].clone();
                    self.entity_graph.add_edge(pred, *nhead, edge);
                }
            }
        } else {
            for (_, _, e) in l.back_edges() {
                self.entity_graph.remove_edge(**e);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct StronglyConnectedComponents<V>(Vec<Vec<VertexIndex<V>>>);

impl<V> Deref for StronglyConnectedComponents<V> {
    type Target = Vec<Vec<VertexIndex<V>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> From<StronglyConnectedComponents<V>> for Vec<Vec<VertexIndex<V>>> {
    fn from(scc: StronglyConnectedComponents<V>) -> Self {
        scc.0
    }
}

impl<V> StronglyConnectedComponents<V> {
    pub fn into_inner(self) -> Vec<Vec<VertexIndex<V>>> {
        self.into()
    }

    pub fn into_iter(self) -> impl IntoIterator<Item = Vec<VertexIndex<V>>> {
        self.0.into_iter()
    }
}

impl<V, E> From<NaturalLoop<V, E>> for Loop<V, E>
where
    V: Clone,
{
    fn from(nl: NaturalLoop<V, E>) -> Self {
        let mut heads = BTreeSet::new();
        heads.insert(nl.head);
        Self {
            heads,
            back_edges: vec![nl.back_edge()],
            body: nl.body,
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Loop<V, E> {
    heads: BTreeSet<VertexIndex<V>>,
    back_edges: Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)>,
    body: BTreeSet<VertexIndex<V>>,
}

impl<V, E> Loop<V, E>
where
    V: Clone,
{
    fn new(
        heads: BTreeSet<VertexIndex<V>>,
        back_edges: Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)>,
        body: BTreeSet<VertexIndex<V>>,
    ) -> Self {
        Self {
            heads,
            back_edges,
            body,
        }
    }

    /// Do `self` and `other` share the same loop header?
    pub fn is_merged(&self, other: &Self) -> bool {
        self.heads.union(&other.heads).next().is_some()
    }

    /// Are `self` and `other` disjoint?
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.heads.is_disjoint(&other.heads)
    }

    /// Is `self` contained within `other`?
    pub fn is_nested(&self, other: &Self) -> bool {
        self.body.is_subset(&other.body)
    }

    /// Is `self` an outer loop of `other`?
    pub fn is_outer(&self, other: &Self) -> bool {
        other.is_nested(self)
    }

    /// Is `self` an inner loop of `other`?
    pub fn is_inner(&self, other: &Self) -> bool {
        self.is_nested(other)
    }

    pub fn external_predecessors<'g>(
        &self,
        g: &'g EntityGraph<'_, V, E>,
    ) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let mut preds = Vec::new();
        for head in self.heads.iter() {
            preds.extend(g.predecessors(*head).filter_map(|(p, e)| {
                if !self.body.contains(&p) {
                    Some((p, *head, e))
                } else {
                    None
                }
            }));
        }
        preds
    }

    pub fn external_successors<'g>(
        &self,
        g: &'g EntityGraph<'_, V, E>,
    ) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let mut preds = Vec::new();
        for head in self.heads.iter() {
            preds.extend(g.successors(*head).filter_map(|(s, e)| {
                if !self.body.contains(&s) {
                    Some((*head, s, e))
                } else {
                    None
                }
            }));
        }
        preds
    }

    pub fn is_reducible(&self) -> bool {
        self.heads.len() == 1
    }

    pub fn is_irreducible(&self) -> bool {
        !self.is_reducible()
    }

    pub fn head(&self) -> VertexIndex<V> {
        self.heads.iter().next().copied().unwrap()
    }

    pub fn heads(&self) -> &BTreeSet<VertexIndex<V>> {
        &self.heads
    }

    pub fn body(&self) -> &BTreeSet<VertexIndex<V>> {
        &self.body
    }

    pub fn back_edges(&self) -> &Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        &self.back_edges
    }

    pub fn natural_loops<'g>(&self, g: &'g EntityGraph<V, E>) -> Vec<NaturalLoop<V, E>> {
        if self.is_reducible() {
            self.back_edges
                .iter()
                .map(|(t, h, e)| NaturalLoop {
                    head: *h,
                    tail: *t,
                    edge: *e,
                    body: self.body().clone(),
                })
                .collect()
        } else {
            let mut loops = Vec::new();

            // assume each head is the only head
            for head in self.heads() {
                // find all edges originating from inside the loop body back to head; treat as back-edges
                for node in self.body().iter() {
                    if let Some(e) =
                        g.successors(*node)
                            .find_map(|(s, e)| if s == *head { Some(e) } else { None })
                    {
                        loops.push(NaturalLoop {
                            head: *head,
                            tail: *node,
                            edge: e,
                            body: self.body.clone(),
                        });
                    }
                }
            }

            loops
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct NaturalLoop<V, E> {
    head: VertexIndex<V>,
    tail: VertexIndex<V>,
    edge: EdgeIndex<E>,
    body: BTreeSet<VertexIndex<V>>,
}

impl<V, E> NaturalLoop<V, E>
where
    V: Clone,
{
    fn new<'g>(
        graph: &'g EntityGraph<'_, V, E>,
        head: VertexIndex<V>,
        tail: VertexIndex<V>,
        edge: EdgeIndex<E>,
    ) -> Self {
        let mut worklist = vec![tail];
        let mut body = BTreeSet::new();
        body.insert(head);

        while let Some(v) = worklist.pop() {
            if body.contains(&v) {
                continue;
            }

            body.insert(v);

            for (p, _) in graph.predecessors(v) {
                worklist.push(p);
            }
        }

        Self {
            head,
            tail,
            edge,
            body,
        }
    }

    pub fn head(&self) -> VertexIndex<V> {
        self.head
    }

    pub fn tail(&self) -> VertexIndex<V> {
        self.tail
    }

    pub fn edge(&self) -> EdgeIndex<E> {
        self.edge
    }

    pub fn back_edge(&self) -> (VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>) {
        (self.tail, self.head, self.edge)
    }

    pub fn body(&self) -> &BTreeSet<VertexIndex<V>> {
        &self.body
    }

    pub fn external_predecessors<'g>(
        &self,
        g: &'g EntityGraph<'_, V, E>,
    ) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        g.predecessors(self.head)
            .filter_map(|(p, e)| {
                if !self.body.contains(&p) {
                    Some((p, self.head, e))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn external_successors<'g>(
        &self,
        g: &'g EntityGraph<'_, V, E>,
    ) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        g.successors(self.head)
            .filter_map(|(s, e)| {
                if !self.body.contains(&s) {
                    Some((self.head, s, e))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
pub struct VertexEdgeIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    traverse: WalkNeighbors<petgraph::graph::DefaultIx>,
    size_hint: (usize, Option<usize>),
}

impl<'g, 'a, V, E> VertexEdgeIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>, traverse: Neighbors<'g, E>) -> Self {
        Self {
            graph,
            traverse: traverse.detach(),
            size_hint: traverse.size_hint(),
        }
    }

    pub fn detach(self) -> VertexEdgeDetachedIter<V, E> {
        VertexEdgeDetachedIter {
            traverse: self.traverse,
            marker: PhantomData,
        }
    }
}

impl<'g, 'a, V, E> Iterator for VertexEdgeIter<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (VertexIndex<V>, EdgeIndex<E>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let v = self
            .traverse
            .next(&self.graph.entity_graph)
            .map(|(e, v)| (VertexIndex::from(v), EdgeIndex::from(e)));
        if v.is_some() {
            let mut h = &mut self.size_hint;
            h.0 = if h.0 != 0 { h.0 - 1 } else { 0 };
            h.1 = h.1.map(|v| if v != 0 { v - 1 } else { 0 });
        }
        v
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
#[repr(transparent)]
pub struct VertexEdgeDetachedIter<V, E> {
    traverse: WalkNeighbors<petgraph::graph::DefaultIx>,
    marker: PhantomData<(V, E)>,
}

impl<V, E> VertexEdgeDetachedIter<V, E>
where
    V: Clone,
{
    #[inline(always)]
    pub fn next(&mut self, graph: &EntityGraph<V, E>) -> Option<(VertexIndex<V>, EdgeIndex<E>)> {
        self.traverse
            .next(&graph.entity_graph)
            .map(|(e, v)| (VertexIndex::from(v), EdgeIndex::from(e)))
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
pub struct PostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    traverse: PostOrder,
}

#[derive(educe::Educe)]
#[educe(Clone)]
#[repr(transparent)]
pub struct PostOrderDetachedIter<V, E> {
    traverse: PostOrder,
    marker: PhantomData<(V, E)>,
}

impl<'g, 'a, V, E> PostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        let mut traverse = PostOrder::new(&graph);
        traverse.start_neighbours.extend(
            graph
                .entity_roots
                .iter()
                .map(|(_, vx)| -> NodeIndex { **vx }),
        );

        Self { graph, traverse }
    }

    pub fn starting_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .visited_start_neighbours
            .into_iter()
            .map(VertexIndex::from)
            .collect()
    }

    pub fn detach(self) -> PostOrderDetachedIter<V, E> {
        PostOrderDetachedIter {
            traverse: self.traverse,
            marker: PhantomData,
        }
    }
}

impl<V, E> PostOrderDetachedIter<V, E>
where
    V: Clone,
{
    pub fn next(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>> {
        self.traverse.next(&graph).map(VertexIndex::from)
    }

    pub fn starting_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .visited_start_neighbours
            .into_iter()
            .map(VertexIndex::from)
            .collect()
    }

    pub fn len(&self, graph: &EntityGraph<V, E>) -> usize {
        let max = graph.vertex_count();
        let fin = self.traverse.finished.count_ones(..);
        max - fin
    }
}

impl<'g, 'a, V, E> Iterator for PostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g Id<V>, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.traverse.next(&self.graph).map(|vx| {
            let id = &self.graph.entity_graph[*vx];
            let (_, er) = &self.graph.entities[id];
            (id, vx, er)
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let max = self.graph.vertex_count();
        let fin = self.traverse.finished.count_ones(..);
        (max - fin, Some(max - fin))
    }
}

impl<'g, 'a, V, E> ExactSizeIterator for PostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn len(&self) -> usize {
        let max = self.graph.vertex_count();
        let fin = self.traverse.finished.count_ones(..);
        max - fin
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
pub struct RevPostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    traverse: RevPostOrder,
}

#[derive(educe::Educe)]
#[educe(Clone)]
#[repr(transparent)]
pub struct RevPostOrderDetachedIter<V, E> {
    traverse: RevPostOrder,
    marker: PhantomData<(V, E)>,
}

impl<'g, 'a, V, E> RevPostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            traverse: RevPostOrder::new(graph),
        }
    }

    pub fn terminal_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .visited_end_neighbours
            .into_iter()
            .map(VertexIndex::from)
            .collect()
    }

    pub fn detach(self) -> RevPostOrderDetachedIter<V, E> {
        RevPostOrderDetachedIter {
            traverse: self.traverse,
            marker: PhantomData,
        }
    }
}

impl<V, E> RevPostOrderDetachedIter<V, E>
where
    V: Clone,
{
    pub fn next(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>> {
        self.traverse.next(&graph)
    }

    pub fn terminal_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .visited_end_neighbours
            .into_iter()
            .map(VertexIndex::from)
            .collect()
    }

    pub fn len(&self, graph: &EntityGraph<V, E>) -> usize {
        let max = graph.vertex_count();
        let fin = self.traverse.finished.count_ones(..);
        max - fin
    }
}

impl<'g, 'a, V, E> Iterator for RevPostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g Id<V>, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.traverse.next(&self.graph).map(|vx| {
            let id = &self.graph.entity_graph[*vx];
            let (_, er) = &self.graph.entities[id];
            (id, vx, er)
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let max = self.graph.vertex_count();
        let fin = self.traverse.finished.count_ones(..);
        (max - fin, Some(max - fin))
    }
}

impl<'g, 'a, V, E> ExactSizeIterator for RevPostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn len(&self) -> usize {
        let max = self.graph.vertex_count();
        let fin = self.traverse.finished.count_ones(..);
        max - fin
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
pub struct EntityRefIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    eiter: indexmap::map::Iter<'g, Id<V>, (VertexIndex<V>, EntityRef<'a, V>)>,
}

impl<'g, 'a, V, E> EntityRefIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            eiter: graph.entities.iter(),
        }
    }
}

impl<'g, 'a, V, E> Iterator for EntityRefIter<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g Id<V>, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.eiter.next().map(|(eid, (v, e))| (eid, *v, e))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.eiter.size_hint()
    }
}

pub struct EntityRefIterFiltered<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    eiter: indexmap::map::Iter<'g, Id<V>, (VertexIndex<V>, EntityRef<'a, V>)>,
    fiter: Box<
        dyn Fn(
                &'g EntityGraph<'a, V, E>,
                &'g Id<V>,
                VertexIndex<V>,
                &'g EntityRef<'a, V>,
            ) -> bool
            + 'static,
    >,
}

impl<'g, 'a, V, E> EntityRefIterFiltered<'g, 'a, V, E>
where
    V: Clone,
{
    fn new<F>(graph: &'g EntityGraph<'a, V, E>, filter: F) -> Self
    where
        F: Fn(
                &'g EntityGraph<'a, V, E>,
                &'g Id<V>,
                VertexIndex<V>,
                &'g EntityRef<'a, V>,
            ) -> bool
            + 'static,
    {
        Self {
            graph,
            eiter: graph.entities.iter(),
            fiter: Box::new(filter),
        }
    }
}

impl<'g, 'a, V, E> Iterator for EntityRefIterFiltered<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g Id<V>, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((eid, (v, e))) = self.eiter.next() {
            if (self.fiter)(self.graph, eid, *v, e) {
                return Some((eid, *v, e));
            }
        }
        None
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.eiter.size_hint()
    }
}

pub struct EntityRefIterMut<'g, 'a, V, E>
where
    V: Clone,
{
    eiter: EntityIterMutInner<'g, Id<V>, (VertexIndex<V>, EntityRef<'a, V>)>,
    marker: PhantomData<&'g E>,
}

impl<'g, 'a, V, E> EntityRefIterMut<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g mut EntityGraph<'a, V, E>) -> Self {
        Self {
            eiter: graph.entities.iter_mut(),
            marker: PhantomData,
        }
    }
}

impl<'g, 'a, V, E> Iterator for EntityRefIterMut<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g Id<V>, VertexIndex<V>, &'g mut EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.eiter.next().map(|(eid, (v, e))| (eid, *v, e))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.eiter.size_hint()
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
pub struct EntityRootOrLeafIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    eiter: indexmap::set::Iter<'g, (Id<V>, VertexIndex<V>)>,
}

impl<'g, 'a, V, E> EntityRootOrLeafIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new_roots(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            eiter: graph.entity_roots.iter(),
        }
    }

    fn new_leaves(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            eiter: graph.entity_leaves.iter(),
        }
    }
}

impl<'g, 'a, V, E> Iterator for EntityRootOrLeafIter<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g Id<V>, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.eiter
            .next()
            .map(|(eid, v)| (eid, *v, &self.graph.entities[eid].1))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.eiter.size_hint()
    }
}

pub trait AsEntityGraph<'a, V, E>
where
    V: Clone,
{
    fn entity_graph(&self) -> &EntityGraph<'a, V, E>;
}

pub trait AsEntityGraphMut<'a, V, E>: AsEntityGraph<'a, V, E>
where
    V: Clone,
{
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, V, E>;
}

impl<'a, V, E> AsEntityGraph<'a, V, E> for &'_ EntityGraph<'a, V, E>
where
    V: Clone,
{
    #[inline(always)]
    fn entity_graph(&self) -> &EntityGraph<'a, V, E> {
        *self
    }
}

impl<'a, V, E> AsEntityGraph<'a, V, E> for &'_ mut EntityGraph<'a, V, E>
where
    V: Clone,
{
    #[inline(always)]
    fn entity_graph(&self) -> &EntityGraph<'a, V, E> {
        &**self
    }
}

impl<'a, V, E> AsEntityGraph<'a, V, E> for EntityGraph<'a, V, E>
where
    V: Clone,
{
    #[inline(always)]
    fn entity_graph(&self) -> &EntityGraph<'a, V, E> {
        self
    }
}

impl<'a, V, E> AsEntityGraphMut<'a, V, E> for EntityGraph<'a, V, E>
where
    V: Clone,
{
    #[inline(always)]
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, V, E> {
        self
    }
}

impl<'a, V, E> AsEntityGraphMut<'a, V, E> for &'_ mut EntityGraph<'a, V, E>
where
    V: Clone,
{
    #[inline(always)]
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, V, E> {
        *self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::visualise::AsDot;

    #[test]
    fn loop_sccs() {
        let mut graph = EntityGraph::<&'static str, ()>::new();

        let mk_ent = |name: &'static str| -> Entity<&'static str> {
            Entity::new("node", name)
        };

        let a = mk_ent("A");
        let b = mk_ent("B");
        let c = mk_ent("C");
        let d = mk_ent("D");

        graph.add_relation(&a, &b, ());
        graph.add_relation(&b, &a, ());

        graph.add_relation(&c, &d, ());
        graph.add_relation(&d, &c, ());

        graph.add_relation(&a, &c, ());

        for scc in graph.strongly_connected_components().iter() {
            println!("{:?}", scc);
        }
    }

    #[test]
    fn loop_test_nunroll() {
        let mut graph = EntityGraph::<&'static str, ()>::new();

        let mk_ent = |name: &'static str| -> Entity<&'static str> {
            Entity::new("node", name)
        };

        let _z = mk_ent("Z");
        let b = mk_ent("B");
        let c = mk_ent("C");
        let a = mk_ent("A");
        let d = mk_ent("D");
        let e = mk_ent("E");
        let f = mk_ent("F");
        let g = mk_ent("G");

        //graph.add_root_entity(&z);
        graph.add_root_entity(&a);

        //graph.add_relation(&z, &a, ());

        graph.add_relation(&a, &b, ());
        graph.add_relation(&a, &c, ());

        //graph.add_relation(&b, &a, ());
        graph.add_relation(&b, &c, ());

        graph.add_relation(&b, &d, ());

        graph.add_relation(&c, &b, ());

        graph.add_relation(&d, &e, ());
        graph.add_relation(&d, &a, ());

        //graph.add_relation(&e, &b, ());
        graph.add_relation(&e, &f, ());
        graph.add_relation(&f, &g, ());
        graph.add_relation(&g, &f, ());

        println!(
            "{}",
            graph.dot_with(|e, v| format!("id: {}, val: {}", e, v), |_| "".to_owned())
        );

        let nloops = graph.natural_loops();
        if let Some(l) = nloops.iter().max_by_key(|l| l.body().len()) {
            graph.unroll_natural_loop(l, 2);
        }

        // STEP 1: use SCC and natural loops to find the largest loop
        // STEP 2: proceed to reduce loops by combination of SCC + natural loops

        // outer loops is: SCC or G as a natural loop

        // deal with "complex" natural loop
        println!(
            "{}",
            graph.dot_with(|e, v| format!("id: {}, val: {}", e, v), |_| "".to_owned())
        );

        loop {
            println!("finding loops...");
            let mut loops = graph.simple_loops();
            println!("found {} loops: {:#?}!", loops.len(), loops);
            if let Some(l) = loops.pop() {
                println!("loop is reducible: {}", l.is_reducible());
                graph.unroll_loop(&l, 2);
                println!("unroll: {:?}", l);
                println!(
                    "{}",
                    graph.dot_with(|e, v| format!("id: {}, val: {}", e, v), |_| "".to_owned())
                );
            } else {
                break;
            }
        }

        println!(
            "{}",
            graph.dot_with(|e, v| format!("id: {}, val: {}", e, v), |_| "".to_owned())
        );
    }

    #[test]
    fn loop_test_unroll() {
        let mut graph = EntityGraph::<&'static str, ()>::new();

        let mk_ent = |name: &'static str| -> Entity<&'static str> {
            Entity::new("node", name)
        };

        let b = mk_ent("B");
        let c = mk_ent("C");
        let a = mk_ent("A");
        let d = mk_ent("D");
        let e = mk_ent("E");
        let f = mk_ent("F");
        let g = mk_ent("G");

        graph.add_root_entity(&a);

        graph.add_relation(&a, &b, ());
        graph.add_relation(&a, &c, ());

        graph.add_relation(&b, &c, ());

        graph.add_relation(&b, &d, ());

        graph.add_relation(&c, &b, ());

        graph.add_relation(&d, &e, ());
        graph.add_relation(&d, &a, ());

        graph.add_relation(&e, &b, ());
        graph.add_relation(&e, &f, ());
        graph.add_relation(&f, &g, ());
        graph.add_relation(&g, &f, ());

        println!(
            "{}",
            graph.dot_with(|_nx, e| e.to_string(), |_| "".to_owned())
        );

        /*
        let mut unroll = true;
        while unroll {
            unroll = false;
            for l in graph.natural_loops() {
                println!("{:?}", l);
                graph.unroll_loop(&l, 1);
                unroll = true;
            }
        }
        */

        for l in graph.simple_loops() {
            println!("loop (reducible: {}): {{", l.is_reducible());
            for v in l.body() {
                println!("\t{}", graph.entity(*v).value());
            }
            println!("}}");
        }
    }
}