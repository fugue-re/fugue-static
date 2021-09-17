use fugue::ir::il::ecode::{Entity, EntityId};
use fxhash::FxBuildHasher;

use petgraph::algo::{has_path_connecting, kosaraju_scc, DfsSpace};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use petgraph::Direction;

use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::traits::{EntityRef, IntoEntityRef};

use super::traversals::{PostOrder, RevPostOrder};

type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
type HashSet<K> = std::collections::HashSet<K, FxBuildHasher>;

type IndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;
type IndexSet<K> = indexmap::IndexSet<K, FxBuildHasher>;

#[derive(educe::Educe)]
#[educe(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Debug, Clone)]
pub struct EntityGraph<'a, V, E>
where
    V: Clone,
{
    pub(crate) entity_graph: StableDiGraph<EntityId, E>,

    pub(crate) entity_mapping: IndexMap<EntityId, VertexIndex<V>>,
    pub(crate) entity_roots: IndexSet<(EntityId, VertexIndex<V>)>,

    pub(crate) entities: HashMap<EntityId, Cow<'a, Entity<V>>>,
}

impl<'a, V, E> Default for EntityGraph<'a, V, E>
where
    V: Clone,
{
    fn default() -> Self {
        Self {
            entity_graph: Default::default(),
            entity_mapping: Default::default(),
            entity_roots: Default::default(),
            entities: Default::default(),
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
            entity_mapping: self.entity_mapping.clone(),
            entities: self
                .entities
                .iter()
                .map(|(id, v)| (id.clone(), Cow::Owned(v.as_ref().clone())))
                .collect(),
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
        self.entity_mapping.get(id).copied()
    }

    pub fn entity(&self, vertex: VertexIndex<V>) -> &EntityRef<'a, V> {
        let id = &self.entity_graph[*vertex];
        &self.entities[id]
    }

    pub fn entity_mut(&mut self, vertex: VertexIndex<V>) -> &mut EntityRef<'a, V> {
        let id = &self.entity_graph[*vertex];
        self.entities.get_mut(id).unwrap()
    }

    pub fn add_entity<T>(&mut self, entity: T) -> VertexIndex<V>
    where
        T: IntoEntityRef<'a, T = V>,
    {
        let er = entity.into_entity_ref();
        if let Some(nx) = self.entity_mapping.get(er.id()) {
            *nx
        } else {
            let id = er.id();
            let nx = self.entity_graph.add_node(id.clone()).into();
            self.entity_mapping.insert(id.clone(), nx);
            self.entities.insert(id.clone(), er);
            nx
        }
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

    pub fn entities<'g>(&'g self) -> EntityRefIter<'g, 'a, V, E> {
        EntityRefIter::new(self)
    }

    pub fn root_entities<'g>(&'g self) -> EntityRootIter<'g, 'a, V, E> {
        EntityRootIter::new(self)
    }

    pub fn has_root_entities(&self) -> bool {
        !self.entity_roots.is_empty()
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
            .node_indices()
            .filter_map(|nx| -> Option<VertexIndex<V>> {
                if self
                    .entity_graph
                    .neighbors_directed(nx, Direction::Incoming)
                    .next()
                    .is_none()
                {
                    Some(nx.into())
                } else {
                    None
                }
            });

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

    pub fn post_order<'g>(&'g self) -> PostOrderIter<'g, 'a, V, E> {
        PostOrderIter::new(self)
    }

    pub fn reverse_post_order<'g>(&'g self) -> RevPostOrderIter<'g, 'a, V, E> {
        RevPostOrderIter::new(self)
    }
}

pub struct PostOrderIter<'g, 'a, V, E> where V: Clone {
    graph: &'g EntityGraph<'a, V, E>,
    traverse: PostOrder<NodeIndex>,
}

pub struct PostOrderDetachedIter<V, E> {
    traverse: PostOrder<NodeIndex>,
    marker: PhantomData<(V, E)>,
}

impl<'g, 'a, V, E> PostOrderIter<'g, 'a, V, E> where V: Clone {
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        let mut traverse = PostOrder::new(&graph.entity_graph);
        traverse.start_neighbours
            .extend(graph.entity_roots.iter().map(|(_, vx)| -> NodeIndex { **vx }));

        Self {
            graph,
            traverse,
        }
    }

    pub fn starting_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse.start_neighbours
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

impl<V, E> PostOrderDetachedIter<V, E> where V: Clone {
    pub fn next(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>> {
        self.traverse
            .next(&graph.entity_graph)
            .map(VertexIndex::from)
    }

    pub fn starting_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse.start_neighbours
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
        self.traverse
            .next(&self.graph.entity_graph)
            .map(|nx| {
                let vx = VertexIndex::from(nx);
                let id = &self.graph.entity_graph[nx];
                let er = &self.graph.entities[id];
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

pub struct RevPostOrderIter<'g, 'a, V, E> where V: Clone {
    graph: &'g EntityGraph<'a, V, E>,
    traverse: RevPostOrder<NodeIndex>,
}

pub struct RevPostOrderDetachedIter<V, E> {
    traverse: RevPostOrder<NodeIndex>,
    marker: PhantomData<(V, E)>,
}

impl<'g, 'a, V, E> RevPostOrderIter<'g, 'a, V, E> where V: Clone {
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            traverse: RevPostOrder::new(&graph.entity_graph),
        }
    }

    pub fn terminal_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse.end_neighbours
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

impl<V, E> RevPostOrderDetachedIter<V, E> where V: Clone {
    pub fn next(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>> {
        self.traverse
            .next(&graph.entity_graph)
            .map(VertexIndex::from)
    }

    pub fn terminal_vertices(self) -> Vec<VertexIndex<V>> {
        self.traverse.end_neighbours
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
        self.traverse
            .next(&self.graph.entity_graph)
            .map(|nx| {
                let vx = VertexIndex::from(nx);
                let id = &self.graph.entity_graph[nx];
                let er = &self.graph.entities[id];
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

pub struct EntityRefIter<'g, 'a, V, E>
where
    V: Clone,
{
    graph: &'g EntityGraph<'a, V, E>,
    eiter: indexmap::map::Iter<'g, EntityId, VertexIndex<V>>,
}

impl<'g, 'a, V, E> EntityRefIter<'g, 'a, V, E>
where
    V: Clone,
{
    fn new(graph: &'g EntityGraph<'a, V, E>) -> Self {
        Self {
            graph,
            eiter: graph.entity_mapping.iter(),
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
            .map(|(eid, v)| (eid, *v, &self.graph.entities[eid]))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.eiter.size_hint()
    }
}

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
            .map(|(eid, v)| (eid, *v, &self.graph.entities[eid]))
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
