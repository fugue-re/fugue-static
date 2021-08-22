use fixedbitset::FixedBitSet;

use petgraph::graph::NodeIndex;

use petgraph::algo::kosaraju_scc;
use petgraph::graph::IndexType;
use petgraph::visit::{IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, GraphRef, NodeCount, Reversed, VisitMap, Visitable};

use std::borrow::Borrow;
use std::collections::VecDeque;

use crate::graphs::traversals::{PostOrder, Traversal};
use crate::types::EntityGraph;

/// Visit nodes in a depth-first-search (DFS) emitting nodes in reverse postorder
/// (each node after all its descendants have been emitted).
///
/// `RevPostOrder` is not recursive.
///
/// The traversal starts at a given node and only traverses nodes reachable
/// from it.
#[derive(Clone, Debug)]
pub struct RevPostOrder<N> {
    /// The stack of nodes to visit
    pub stack: Vec<Option<N>>,
    /// The map of discovered nodes
    pub discovered: FixedBitSet,
    /// The map of finished nodes
    pub finished: FixedBitSet,
    /// Virtual end neighbours
    pub end_neighbours: Vec<N>,
    /// If the virtual end has been discovered
    pub end_discovered: bool,
    /// Strongly connected components
    pub scc: Vec<Vec<N>>,
}

impl<N> Default for RevPostOrder<N> {
    fn default() -> Self {
        RevPostOrder {
            stack: Vec::new(),
            discovered: FixedBitSet::default(),
            finished: FixedBitSet::default(),
            end_neighbours: Vec::new(),
            end_discovered: false,
            scc: Vec::default(),
        }
    }
}

impl<N> RevPostOrder<N>
where
    N: IndexType,
{
    /// Create a new `RevPostOrder` using the graph's visitor map, and put
    /// `start` in the stack of nodes to visit.
    pub fn new<G>(graph: G) -> Self
    where
        G: GraphRef + IntoNeighborsDirected + IntoNodeIdentifiers + NodeCount + Visitable<NodeId = N, Map = FixedBitSet>,
    {
        let (start, mut dfs) = Self::empty(graph);
        dfs.move_to(start);
        dfs
    }

    fn empty<G>(graph: G) -> (Option<N>, Self)
    where
        G: GraphRef + IntoNeighborsDirected + IntoNodeIdentifiers + Visitable<NodeId = N, Map = FixedBitSet>,
    {
        let mut scc = kosaraju_scc(&graph);
        let end_neighbours = graph.node_identifiers()
            .filter(|nx| graph.neighbors(*nx).next().is_none())
            .collect::<Vec<_>>();

        let end = if end_neighbours.is_empty() {
            scc.pop().and_then(|mut cs| cs.pop())
        } else {
            None
        };

        let order = RevPostOrder {
            stack: Vec::new(),
            discovered: graph.visit_map(),
            finished: graph.visit_map(),
            end_neighbours,
            end_discovered: false,
            scc,
        };

        (end, order)
    }

    fn move_to(&mut self, start: Option<N>) {
        self.stack.clear();
        self.stack.push(start);
    }

    /// Return the next node in the traversal, or `None` if the traversal is done.
    pub fn next<G>(&mut self, graph: G) -> Option<N>
    where
        G: IntoNeighbors<NodeId = N> + NodeCount + IntoNeighborsDirected,
    {
        let graph = Reversed(graph);

        'outer: loop {
            while let Some(&nx) = self.stack.last() {
                if let Some(nx) = nx {
                    if self.discovered.visit(nx) {
                        // First time visiting `nx`: Push neighbors, don't pop `nx`
                        for succ in graph.neighbors(nx) {
                            if !self.discovered.is_visited(&succ) {
                                self.stack.push(Some(succ));
                            }
                        }
                    } else {
                        self.stack.pop();
                        if self.finished.visit(nx) {
                            // Second time: All reachable nodes must have been finished
                            return Some(nx);
                        }
                    }
                } else {
                    if !self.end_discovered {
                        self.end_discovered = true;
                        for succ in self.end_neighbours.iter().rev() {
                            if !self.discovered.is_visited(succ) {
                                self.stack.push(Some(*succ));
                            }
                        }
                    } else {
                        self.stack.pop();
                    }
                }
            }

            if self.finished.count_ones(..) != graph.node_count() {
                while let Some(cs) = self.scc.pop() {
                    if !self.discovered.is_visited(&cs[0]) {
                        self.end_neighbours.push(cs[0]);
                        self.stack.push(Some(cs[0]));
                        continue 'outer
                    }
                }
            }

            break
        }
        None
    }
}

impl<'a> Traversal<'a> for RevPostOrder<NodeIndex> {
    fn into_queue_with_roots<E, G>(graph: G) -> (Vec<NodeIndex>, VecDeque<NodeIndex>)
    where G: Borrow<EntityGraph<E>> + 'a {
        let g = graph.borrow();
        let mut traversal = PostOrder::new(g);
        let mut queue = VecDeque::new();

        while let Some(nx) = traversal.next(g) {
            queue.push_front(nx);
        }

        (traversal.start_neighbours, queue)
    }
}
