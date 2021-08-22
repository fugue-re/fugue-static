use fixedbitset::FixedBitSet;

use petgraph::graph::NodeIndex;

use petgraph::algo::kosaraju_scc;
use petgraph::Direction;
use petgraph::graph::IndexType;
use petgraph::visit::{IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, GraphRef, NodeCount, VisitMap, Visitable};

use std::borrow::Borrow;
use std::collections::VecDeque;

use crate::graphs::traversals::Traversal;
use crate::types::EntityGraph;

/// Visit nodes in a depth-first-search (DFS) emitting nodes in postorder.
///
/// The traversal starts at a "virtual" start node that is connected to the
/// roots of all SCC sub-graphs.
#[derive(Clone, Debug)]
pub struct PostOrder<N> {
    /// The stack of nodes to visit
    pub stack: Vec<Option<N>>,
    /// The map of discovered nodes
    pub discovered: FixedBitSet,
    /// The map of finished nodes
    pub finished: FixedBitSet,
    /// Virtual start neighbours
    pub start_neighbours: Vec<N>,
    /// If the virtual start has been discovered
    pub start_discovered: bool,
    /// Strongly connected components of graph
    pub scc: Vec<Vec<N>>,
}

impl<N> Default for PostOrder<N> {
    fn default() -> Self {
        Self {
            stack: Vec::default(),
            discovered: FixedBitSet::default(),
            finished: FixedBitSet::default(),
            start_neighbours: Vec::default(),
            start_discovered: false,
            scc: Vec::default(),
        }
    }
}

impl<N> PostOrder<N>
where
    N: IndexType
{
    /// Create a new `PostOrder` using the graph's visitor map, and put
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

        let start_neighbours = graph.node_identifiers()
            .filter(|nx| graph.neighbors_directed(*nx, Direction::Incoming).next().is_none())
            .collect::<Vec<_>>();

        let start = if start_neighbours.is_empty() {
            scc.pop().and_then(|mut cs| cs.pop())
        } else {
            None
        };

        let order = PostOrder {
            stack: Vec::new(),
            discovered: graph.visit_map(),
            finished: graph.visit_map(),
            start_neighbours,
            start_discovered: false,
            scc,
        };

        (start, order)
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
                    if !self.start_discovered {
                        self.start_discovered = true;
                        for succ in self.start_neighbours.iter() {
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
                        self.start_neighbours.push(cs[0]);
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

impl<'a> Traversal<'a> for PostOrder<NodeIndex> {
    fn into_queue_with_roots<E, G>(graph: G) -> (Vec<NodeIndex>, VecDeque<NodeIndex>)
    where G: Borrow<EntityGraph<E>> + 'a {
        let g = graph.borrow();
        let mut traversal = Self::new(g);
        let mut queue = VecDeque::new();

        while let Some(nx) = traversal.next(g) {
            queue.push_back(nx);
        }

        (traversal.start_neighbours, queue)
    }
}
