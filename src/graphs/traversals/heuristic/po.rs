use fixedbitset::FixedBitSet;

use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    IntoNodeIdentifiers, VisitMap, Visitable,
};
use petgraph::Direction;

use crate::graphs::entity::VertexIndex;
use crate::graphs::EntityGraph;

/// Visit nodes in a depth-first-search (DFS) emitting nodes in postorder.
///
/// The traversal starts at a "virtual" start node that is connected to the
/// roots of all SCC sub-graphs.
#[derive(Clone, Debug)]
pub struct PostOrder {
    /// The stack of nodes to visit
    pub stack: Vec<NodeIndex>,
    /// The map of discovered nodes
    pub discovered: FixedBitSet,
    /// The map of finished nodes
    pub finished: FixedBitSet,
    /// Unprocessed starts
    pub start_neighbours: Vec<NodeIndex>,
    /// Starts used to cover graph
    pub visited_start_neighbours: Vec<NodeIndex>,
}

impl Default for PostOrder {
    fn default() -> Self {
        Self {
            stack: Vec::default(),
            discovered: FixedBitSet::default(),
            finished: FixedBitSet::default(),
            start_neighbours: Vec::default(),
            visited_start_neighbours: Vec::default(),
        }
    }
}

impl PostOrder {
    /// Create a new `PostOrder` using the graph's visitor map, and put
    /// `start` in the stack of nodes to visit.
    pub fn new<V, E>(graph: &EntityGraph<V, E>) -> Self
    where
        V: Clone,
    {
        Self::empty(graph, Vec::new())
    }

    fn empty<V, E>(graph: &EntityGraph<V, E>, mut start_neighbours: Vec<NodeIndex>) -> Self
    where
        V: Clone,
    {
        start_neighbours.reverse(); // preserve initial order

        start_neighbours.extend(graph
            .as_ref()
            .node_identifiers()
            .filter(|nx| {
                graph.as_ref()
                    .neighbors_directed(*nx, Direction::Incoming)
                    .next()
                    .is_none()
            }));

        start_neighbours.extend(kosaraju_scc(graph.as_ref())
            .into_iter()
            .map(|mut scc| scc.pop().unwrap())
        );

        start_neighbours.reverse(); // prefer our specified

        let start = start_neighbours.pop();

        PostOrder {
            stack: if let Some(start) = start {
                vec![start]
            } else {
                Vec::new()
            },
            discovered: graph.as_ref().visit_map(),
            finished: graph.as_ref().visit_map(),
            start_neighbours,
            visited_start_neighbours: if let Some(start) = start {
                vec![start]
            } else {
                Vec::new()
            },
        }
    }

    /// Return the next node in the traversal, or `None` if the traversal is done.
    pub fn next<V, E>(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>>
    where
        V: Clone,
    {
        'outer: loop {
            while let Some(&nx) = self.stack.last() {
                if self.discovered.visit(nx) {
                    // First time visiting `nx`: Push neighbors, don't pop `nx`
                    for succ in graph.as_ref().neighbors(nx) {
                        if !self.discovered.is_visited(&succ) {
                            self.stack.push(succ);
                        }
                    }
                } else {
                    self.stack.pop();
                    if self.finished.visit(nx) {
                        // Second time: All reachable nodes must have been finished
                        return Some(nx.into());
                    }
                }
            }

            if self.finished.count_ones(..) != graph.as_ref().node_count() {
                while let Some(cs) = self.start_neighbours.pop() {
                    if !self.discovered.is_visited(&cs) {
                        self.visited_start_neighbours.push(cs);
                        self.stack.push(cs);
                        continue 'outer;
                    }
                }
            }

            break;
        }
        None
    }
}
