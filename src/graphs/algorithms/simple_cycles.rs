use petgraph::algo::toposort;
use petgraph::graph::DiGraph;

use std::collections::{BTreeMap, BTreeSet};
use std::mem::take;

use crate::graphs::entity::VertexIndex;
use crate::graphs::{AsEntityGraph, EntityGraph};

struct Blocked<V> {
    blocked: BTreeSet<VertexIndex<V>>,
    b: BTreeMap<VertexIndex<V>, BTreeSet<VertexIndex<V>>>,
}

impl<V> Default for Blocked<V> {
    fn default() -> Self {
        Self {
            blocked: BTreeSet::new(),
            b: BTreeMap::new(),
        }
    }
}

impl<V> Blocked<V>
where
    V: Clone,
{
    fn insert(&mut self, node: VertexIndex<V>) {
        self.blocked.insert(node);
    }

    fn contains(&self, node: &VertexIndex<V>) -> bool {
        self.blocked.contains(node)
    }

    fn block(&mut self, n1: VertexIndex<V>, n2: VertexIndex<V>) {
        self.b.entry(n1).or_default().insert(n2);
    }

    fn unblock(&mut self, node: VertexIndex<V>) {
        let mut worklist = vec![node];
        while let Some(node) = worklist.pop() {
            if self.blocked.remove(&node) {
                if let Some(nodes) = self.b.get_mut(&node) {
                    let nodes = take(nodes);
                    worklist.extend(nodes);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleCycle<V> {
    body: BTreeSet<VertexIndex<V>>,
}

impl<V> Default for SimpleCycle<V> {
    fn default() -> Self {
        Self {
            body: BTreeSet::new(),
        }
    }
}

impl<V> SimpleCycle<V> {
    fn new(body: BTreeSet<VertexIndex<V>>) -> Self {
        Self {
            body,
        }
    }

    fn singleton(node: VertexIndex<V>) -> Self {
        let mut s = BTreeSet::new();
        s.insert(node);
        Self::new(s)
    }

    pub fn body(&self) -> &BTreeSet<VertexIndex<V>> {
        &self.body
    }

    pub fn into_body(self) -> BTreeSet<VertexIndex<V>> {
        self.body
    }

    pub fn contains(&self, node: &VertexIndex<V>) -> bool {
        self.body.contains(node)
    }

    pub fn is_strictly_nested(&self, other: &Self) -> bool {
        self.len() < other.len() && self.is_nested(other)
    }

    pub fn is_nested(&self, other: &Self) -> bool {
        self.body.is_subset(&other.body)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item=&VertexIndex<V>> {
        self.body.iter()
    }

    pub fn into_iter(self) -> impl IntoIterator<Item=VertexIndex<V>> {
        self.body.into_iter()
    }

    pub fn len(&self) -> usize {
        self.body.len()
    }
}

impl<V> SimpleCycle<V>
where
    V: Clone,
{
    pub fn find_all<'g, E>(g: &'g EntityGraph<V, E>) -> Vec<Self> {
        // view over g's structure
        let mut view = g.view();
        let mut cycles = Vec::new();

        // strongly connected components with size > 1
        let mut sccs = view
            .strongly_connected_components()
            .into_inner()
            .into_iter()
            .filter_map(|scc| if scc.len() > 1 {
                Some(scc)
            } else {
                None
            })
            .collect::<Vec<_>>();

        // remove self-loops from graph view
        view.as_mut().retain_edges(|g, e| {
            if let Some((n1, n2)) = g.edge_endpoints(e) {
                if n1 == n2 {
                    cycles.push(SimpleCycle::singleton(n1.into()));
                    false
                } else {
                    true
                }
            } else {
                unreachable!()
            }
        });

        while let Some(mut scc) = sccs.pop() {
            let sview = view.subgraph_view(scc.iter().map(|v| *v));
            let start = scc.pop().unwrap();

            let mut blocked = Blocked::default();
            let mut closed = BTreeSet::new();

            blocked.insert(start);

            let mut path = vec![start];

            let mut worklist = vec![(
                start,
                sview.successors(start).into_iter().collect::<Vec<_>>(),
            )];

            while let Some((curr, succs)) = worklist.last_mut() {
                if let Some((next, _)) = succs.pop() {
                    if next == start {
                        // cycle closed
                        closed.extend(path.iter().map(|v| *v));

                        let cycle = SimpleCycle::new(path.iter().map(|v| *v).collect::<BTreeSet<_>>());

                        debug_assert_eq!(cycle.len(), path.len());

                        cycles.push(cycle);
                    } else if !blocked.contains(&next) {
                        path.push(next);

                        closed.remove(&next);
                        blocked.insert(next);

                        worklist.push((next, sview.successors(next).into_iter().collect()));

                        continue;
                    }
                }

                if succs.is_empty() {
                    if closed.contains(&*curr) {
                        blocked.unblock(*curr);
                    } else {
                        for (succ, _) in sview.successors(*curr) {
                            blocked.block(succ, *curr);
                        }
                    }
                    worklist.pop();
                    path.pop();
                }
            }

            // h is the subgraph of g with nodes from scc without the start node
            let h = view.subgraph_view(scc);
            let hsccs = h.strongly_connected_components();

            sccs.extend(hsccs.into_inner().into_iter().filter_map(|scc| if scc.len() > 1 {
                Some(scc)
            } else {
                None
            }));
        }

        cycles
    }
}
pub trait SimpleCycles<V, E> {
    fn simple_cycles(&self) -> Vec<SimpleCycle<V>>;
    fn topological_simple_cycles(&self, reverse: bool) -> (DiGraph<usize, ()>, Vec<SimpleCycle<V>>);
}

impl<'a, V, E, G> SimpleCycles<V, E> for G
where
    V: Clone + 'a,
    G: AsEntityGraph<'a, V, E>,
{
    fn simple_cycles(&self) -> Vec<SimpleCycle<V>> {
        SimpleCycle::find_all(self.entity_graph())
    }

    fn topological_simple_cycles(&self, reverse: bool) -> (DiGraph<usize, ()>, Vec<SimpleCycle<V>>) {
        let mut dag = DiGraph::new();
        println!("finding cycles...");
        let mut cycles = SimpleCycle::find_all(self.entity_graph());
        let ids = (0..cycles.len()).map(|i| dag.add_node(i)).collect::<Vec<_>>();

        println!("topological sorting...");
        // this is O(n^2)... can we avoid this at all?
        for (i, c1) in cycles.iter().enumerate() {
            let nx = ids[i];
            for (j, c2) in cycles.iter().enumerate().filter(|(j, _)| *j != i) {
                if c1.is_nested(&c2) {
                    let mx = ids[j];
                    if reverse {
                        dag.add_edge(mx, nx, ());
                    } else {
                        dag.add_edge(nx, mx, ());
                    }
                }
            }
        }

        let topo = toposort(&dag, None).unwrap();

        let mut ncycles = Vec::with_capacity(cycles.len());

        for (i, nx) in topo.into_iter().enumerate() {
            let j = &mut dag[nx];
            ncycles.push(take(&mut cycles[*j]));
            *j = i;
        }

        (dag, ncycles)
    }
}
