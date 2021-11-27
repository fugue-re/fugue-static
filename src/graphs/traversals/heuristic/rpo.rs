use fixedbitset::FixedBitSet;

use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    IntoNeighbors, IntoNodeIdentifiers, NodeCount, Reversed, VisitMap, Visitable,
};

use crate::graphs::EntityGraph;
use crate::graphs::entity::VertexIndex;

/// Visit nodes in a depth-first-search (DFS) emitting nodes in reverse postorder
/// (each node after all its descendants have been emitted).
///
/// `RevPostOrder` is not recursive.
///
/// The traversal starts at a given node and only traverses nodes reachable
/// from it.
#[derive(Clone, Debug)]
pub struct RevPostOrder {
    /// The stack of nodes to visit
    pub stack: Vec<NodeIndex>,
    /// The map of discovered nodes
    pub discovered: FixedBitSet,
    /// The map of finished nodes
    pub finished: FixedBitSet,
    /// Unprocessed ends
    pub end_neighbours: Vec<NodeIndex>,
    /// Ends used to cover graph
    pub visited_end_neighbours: Vec<NodeIndex>,
}

impl Default for RevPostOrder {
    fn default() -> Self {
        RevPostOrder {
            stack: Vec::new(),
            discovered: FixedBitSet::default(),
            finished: FixedBitSet::default(),
            end_neighbours: Vec::new(),
            visited_end_neighbours: Vec::new(),
        }
    }
}

impl RevPostOrder {
    /// Create a new `RevPostOrder` using the graph's visitor map, and put
    /// `start` in the stack of nodes to visit.
    pub fn new<V, E>(graph: &EntityGraph<V, E>) -> Self
    where
        V: Clone,
    {
        Self::empty(graph, Vec::new())
    }

    fn empty<V, E>(graph: &EntityGraph<V, E>, mut end_neighbours: Vec<NodeIndex>) -> Self
    where
        V: Clone,
    {
        end_neighbours.reverse(); // preserve initial order

        end_neighbours.extend(graph
            .as_ref()
            .node_identifiers()
            .filter(|nx| graph.as_ref().neighbors(*nx).next().is_none())
        );

        end_neighbours.extend(kosaraju_scc(graph.as_ref())
            .into_iter()
            .map(|mut scc| scc.pop().unwrap())
        );


        end_neighbours.reverse(); // prefer our specified

        let end = end_neighbours.pop();

        RevPostOrder {
            stack: if let Some(end) = end {
                vec![end]
            } else {
                Vec::new()
            },
            discovered: graph.as_ref().visit_map(),
            finished: graph.as_ref().visit_map(),
            end_neighbours,
            visited_end_neighbours: if let Some(end) = end {
                vec![end]
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
        let graph = Reversed(graph.as_ref());

        'outer: loop {
            while let Some(&nx) = self.stack.last() {
                if self.discovered.visit(nx) {
                    // First time visiting `nx`: Push neighbors, don't pop `nx`
                    for succ in graph.neighbors(nx) {
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

            if self.finished.count_ones(..) != graph.node_count() {
                while let Some(cs) = self.end_neighbours.pop() {
                    if !self.discovered.is_visited(&cs) {
                        self.visited_end_neighbours.push(cs);
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
