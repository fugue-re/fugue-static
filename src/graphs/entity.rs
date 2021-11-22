use fugue::ir::il::ecode::{Entity, EntityId, Location};
use fxhash::FxBuildHasher;

use petgraph::algo::{has_path_connecting, kosaraju_scc, DfsSpace};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::{Neighbors, StableDiGraph, WalkNeighbors};
use petgraph::Direction;

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::ops::Deref;

use indexmap::map::IterMut as EntityIterMutInner;

use crate::traits::IntoEntityRef;
use crate::types::EntityRef;

use super::algorithms::dominance::{Dominance, Dominators};
use super::traversals::{PostOrder, RevPostOrder};

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

impl<'a, V, E> AsRef<StableDiGraph<EntityId, E>> for EntityGraph<'a, V, E>
where V: Clone {
    fn as_ref(&self) -> &StableDiGraph<EntityId, E> {
        &self.entity_graph
    }
}

#[derive(Debug, Clone)]
pub struct EntityGraph<'a, V, E> where V: Clone {
    pub(crate) entity_graph: StableDiGraph<EntityId, E>,

    //pub(crate) entity_mapping: IndexMap<EntityId, VertexIndex<V>>,
    pub(crate) entity_roots: IndexSet<(EntityId, VertexIndex<V>)>,

    pub(crate) entities: IndexMap<EntityId, (VertexIndex<V>, Cow<'a, Entity<V>>)>,
    pub(crate) entity_versions: HashMap<Location, usize>,
}

impl<'a, V, E> Default for EntityGraph<'a, V, E>
where
    V: Clone,
{
    fn default() -> Self {
        Self {
            entity_graph: Default::default(),
            //entity_mapping: Default::default(),
            entity_roots: Default::default(),
            entities: Default::default(),
            entity_versions: Default::default(),
        }
    }
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
            //entity_mapping: self.entity_mapping.clone(),
            entities: self
                .entities
                .iter()
                .map(|(id, (vx, v))| (id.clone(), (*vx, Cow::Owned(v.as_ref().clone()))))
                .collect(),
            entity_versions: self.entity_versions.clone(),
        }
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

    pub fn entity_vertex(&self, id: &EntityId) -> Option<VertexIndex<V>> {
        self.entities.get(id).map(|v| v.0)
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
        self.entity_graph.find_edge(*vs, *vt)
            .and_then(|e| self.entity_graph.edge_weight(e))
    }

    pub fn edge_mut(&mut self, vs: VertexIndex<V>, vt: VertexIndex<V>) -> Option<&mut E> {
        self.entity_graph.find_edge(*vs, *vt)
            .and_then(move |e| self.entity_graph.edge_weight_mut(e))
    }

    pub fn add_entity<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let er = entity.into_entity_ref();
        if let Some(nx) = self.entities.get(er.id()) {
            nx.0
        } else {
            let id = er.id();
            let loc = id.location().clone();
            let gen = id.generation();

            let nx = self.entity_graph.add_node(id.clone()).into();

            self.entities.insert(id.clone(), (nx, er));

            let ngen = self.entity_versions
                .entry(loc)
                .or_default();
            *ngen = gen.max(*ngen);

            nx
        }
    }

    pub fn add_entity_alias<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let mut er = entity.into_entity_ref();

        let loc = er.id().location().clone();
        let ngen = self.entity_versions
            .entry(loc)
            .or_default();
        *ngen += 1;

        *er.to_mut().id_mut().generation_mut() = *ngen;

        let id = er.id().clone();
        let nx = self.entity_graph.add_node(id.clone()).into();

        self.entities.insert(id, (nx, er));

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

    pub fn root_entities<'g>(&'g self) -> EntityRootIter<'g, 'a, V, E> {
        EntityRootIter::new(self)
    }

    pub fn has_root_entities(&self) -> bool {
        !self.entity_roots.is_empty()
    }

    pub fn leaf_entities<'g>(&'g self) -> EntityRefIterFiltered<'g, 'a, V, E> {
        EntityRefIterFiltered::new(self, |g, _, id, _| g.successors(id).next().is_none())
    }

    pub fn make_root_entity(&mut self, vertex: VertexIndex<V>) {
        let id = &self.entity_graph[*vertex];
        self.entity_roots.insert((id.clone(), vertex));
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

    pub fn post_order<'g>(&'g self) -> PostOrderIter<'g, 'a, V, E> {
        PostOrderIter::new(self)
    }

    pub fn reverse_post_order<'g>(&'g self) -> RevPostOrderIter<'g, 'a, V, E> {
        RevPostOrderIter::new(self)
    }

    pub fn back_edges<'g>(&'g self) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let dominators = self.dominators();
        self.back_edges_with(&dominators)
    }

    pub fn back_edges_with<'g>(&'g self, dominators: &Dominators<V>) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let mut edges = Vec::new();
        for (_, vx, _) in self.reverse_post_order() {
            if let Some(doms) = dominators.dominators(vx).map(|doms| doms.into_iter().collect::<HashSet<_>>()) {
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
        StronglyConnectedComponents(kosaraju_scc(&self.entity_graph)
            .into_iter()
            .map(|ccs| ccs.into_iter().map(VertexIndex::from).collect())
            .collect())
    }

    pub fn loops<'g>(&'g self) -> Vec<Loop<V>> {
        let sccs = self.strongly_connected_components();
        self.loops_with(&sccs)
    }

    pub fn loops_with<'g>(&'g self, sccs: &StronglyConnectedComponents<V>) -> Vec<Loop<V>> {
        let mut loops = Vec::new();
        for cg in sccs.iter().filter(|cg| cg.len() > 1).cloned() {
            let body = cg;
            let heads = body.iter()
                .filter_map(|v| if self.predecessors(*v).any(|(p, _)| !body.contains(&p)) {
                    Some(*v)
                } else {
                    None
                })
                .collect();
            let tails = body.iter()
                .filter_map(|v| if self.successors(*v).any(|(s, _)| !body.contains(&s)) {
                    Some(*v)
                } else {
                    None
                })
                .collect();
            loops.push(Loop::new(heads, tails, body));
        }
        loops
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct StronglyConnectedComponents<V>(Vec<BTreeSet<VertexIndex<V>>>);

impl<V> Deref for StronglyConnectedComponents<V> {
    type Target = Vec<BTreeSet<VertexIndex<V>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> From<StronglyConnectedComponents<V>> for Vec<BTreeSet<VertexIndex<V>>> {
    fn from(scc: StronglyConnectedComponents<V>) -> Self {
        scc.0
    }
}

impl<V> StronglyConnectedComponents<V> {
    pub fn into_inner(self) -> Vec<BTreeSet<VertexIndex<V>>> {
        self.into()
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Loop<V> {
    heads: BTreeSet<VertexIndex<V>>,
    tails: BTreeSet<VertexIndex<V>>,
    body: BTreeSet<VertexIndex<V>>,
}

impl<V> Loop<V> where V: Clone {
    fn new(heads: BTreeSet<VertexIndex<V>>, tails: BTreeSet<VertexIndex<V>>, body: BTreeSet<VertexIndex<V>>) -> Self {
        Self {
            heads,
            tails,
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

    pub fn external_predecessors<'g, E>(&self, g: &'g EntityGraph<'_, V, E>) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let mut preds = Vec::new();
        for head in self.heads.iter() {
            preds.extend(g.predecessors(*head).filter_map(|(p, e)| if !self.body.contains(&p) {
                Some((p, *head, e))
            } else {
                None
            }));
        }
        preds
    }

    pub fn external_successors<'g, E>(&self, g: &'g EntityGraph<'_, V, E>) -> Vec<(VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>)> {
        let mut preds = Vec::new();
        for head in self.heads.iter() {
            preds.extend(g.successors(*head).filter_map(|(s, e)| if !self.body.contains(&s) {
                Some((*head, s, e))
            } else {
                None
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

    pub fn tails(&self) -> &BTreeSet<VertexIndex<V>> {
        &self.tails
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct NaturalLoop<V, E> {
    head: VertexIndex<V>,
    tail: VertexIndex<V>,
    edge: EdgeIndex<E>,
    body: BTreeSet<VertexIndex<V>>,
}

impl<V, E> NaturalLoop<V, E> where V: Clone {
    fn new<'g>(graph: &'g EntityGraph<'_, V, E>, head: VertexIndex<V>, tail: VertexIndex<V>, edge: EdgeIndex<E>) -> Self {
        let mut worklist = vec![tail];
        let mut body = BTreeSet::new();
        body.insert(head);

        while let Some(v) = worklist.pop() {
            if body.contains(&v) {
                continue
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

    pub fn edge(&self) -> EdgeIndex<E> {
        self.edge
    }

    pub fn back_edge(&self) -> (VertexIndex<V>, VertexIndex<V>, EdgeIndex<E>) {
        (self.tail, self.head, self.edge)
    }

    pub fn body(&self) -> &BTreeSet<VertexIndex<V>> {
        &self.body
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
        let v = self.traverse
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
    traverse: PostOrder<NodeIndex>,
}

#[derive(educe::Educe)]
#[educe(Clone)]
#[repr(transparent)]
pub struct PostOrderDetachedIter<V, E> {
    traverse: PostOrder<NodeIndex>,
    marker: PhantomData<(V, E)>,
}

impl<'g, 'a, V, E> PostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        let mut traverse = PostOrder::new(&graph.entity_graph);
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
            .start_neighbours
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
        self.traverse
            .next(&graph.entity_graph)
            .map(VertexIndex::from)
    }

    pub fn starting_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .start_neighbours
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
    type Item = (&'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.traverse.next(&self.graph.entity_graph).map(|nx| {
            let vx = VertexIndex::from(nx);
            let id = &self.graph.entity_graph[nx];
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
    traverse: RevPostOrder<NodeIndex>,
}

#[derive(educe::Educe)]
#[educe(Clone)]
#[repr(transparent)]
pub struct RevPostOrderDetachedIter<V, E> {
    traverse: RevPostOrder<NodeIndex>,
    marker: PhantomData<(V, E)>,
}

impl<'g, 'a, V, E> RevPostOrderIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            traverse: RevPostOrder::new(&graph.entity_graph),
        }
    }

    pub fn terminal_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .end_neighbours
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
        self.traverse
            .next(&graph.entity_graph)
            .map(VertexIndex::from)
    }

    pub fn terminal_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse
            .end_neighbours
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
    type Item = (&'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.traverse.next(&self.graph.entity_graph).map(|nx| {
            let vx = VertexIndex::from(nx);
            let id = &self.graph.entity_graph[nx];
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
    eiter: indexmap::map::Iter<'g, EntityId, (VertexIndex<V>, EntityRef<'a, V>)>,
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
    type Item = (&'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.eiter
            .next()
            .map(|(eid, (v, e))| (eid, *v, e))
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
    eiter: indexmap::map::Iter<'g, EntityId, (VertexIndex<V>, EntityRef<'a, V>)>,
    fiter: Box<dyn Fn(&'g EntityGraph<'a, V, E>, &'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>) -> bool + 'static>,
}

impl<'g, 'a, V, E> EntityRefIterFiltered<'g, 'a, V, E>
where
    V: Clone,
{
    fn new<F>(graph: &'g EntityGraph<'a, V, E>, filter: F) -> Self
    where F: Fn(&'g EntityGraph<'a, V, E>, &'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>) -> bool + 'static {
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
    type Item = (&'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((eid, (v, e))) = self.eiter.next() {
            if (self.fiter)(self.graph, eid, *v, e) {
                return Some((eid, *v, e))
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
    eiter: EntityIterMutInner<'g, EntityId, (VertexIndex<V>, EntityRef<'a, V>)>,
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
    type Item = (&'g EntityId, VertexIndex<V>, &'g mut EntityRef<'a, V>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.eiter
            .next()
            .map(|(eid, (v, e))| (eid, *v, e))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.eiter.size_hint()
    }
}

#[derive(educe::Educe)]
#[educe(Clone)]
pub struct EntityRootIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    eiter: indexmap::set::Iter<'g, (EntityId, VertexIndex<V>)>,
}

impl<'g, 'a, V, E> EntityRootIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            eiter: graph.entity_roots.iter(),
        }
    }
}

impl<'g, 'a, V, E> Iterator for EntityRootIter<'g, 'a, V, E>
where
    V: Clone,
{
    type Item = (&'g EntityId, VertexIndex<V>, &'g EntityRef<'a, V>);

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
