//! Compute dominators of a control-flow graph.
//!
//! # The Dominance Relation
//!
//! In a directed graph with a root node **R**, a node **A** is said to *dominate* a
//! node **B** iff every path from **R** to **B** contains **A**.
//!
//! The node **A** is said to *strictly dominate* the node **B** iff **A** dominates
//! **B** and **A ≠ B**.
//!
//! The node **A** is said to be the *immediate dominator* of a node **B** iff it
//! strictly dominates **B** and there does not exist any node **C** where **A**
//! dominates **C** and **C** dominates **B**.

use std::cmp::Ordering;
use std::collections::hash_map::Iter;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use petgraph::graph::NodeIndex;

use crate::graphs::traversals::{PostOrder, Traversal};
use crate::types::AsEntityGraph;

/// The dominance relation for some graph and root.
#[derive(Debug, Clone)]
pub struct Dominators<N>
where
    N: Copy + Eq + Hash,
{
    roots: HashSet<N>,
    dominators: HashMap<N, N>,
}

impl<N> Dominators<N>
where
    N: Copy + Eq + Hash,
{
    /// Get the root node(s) used to construct these dominance relations.
    pub fn roots(&self) -> &HashSet<N> {
        &self.roots
    }

    /// Get the immediate dominator of the given node.
    ///
    /// Returns `None` for any node that is not reachable from the root, and for
    /// the root itself.
    pub fn immediate_dominator(&self, node: N) -> Option<N> {
        if self.roots.contains(&node) {
            None
        } else {
            self.dominators.get(&node).cloned()
        }
    }

    /// Iterate over the given node's strict dominators.
    ///
    /// If the given node is not reachable from the root, then `None` is
    /// returned.
    pub fn strict_dominators(&self, node: N) -> Option<DominatorsIter<N>> {
        if self.dominators.contains_key(&node) {
            Some(DominatorsIter {
                dominators: self,
                node: self.immediate_dominator(node),
            })
        } else {
            None
        }
    }

    /// Iterate over all of the given node's dominators (including the given
    /// node itself).
    ///
    /// If the given node is not reachable from the root, then `None` is
    /// returned.
    pub fn dominators(&self, node: N) -> Option<DominatorsIter<N>> {
        if self.dominators.contains_key(&node) {
            Some(DominatorsIter {
                dominators: self,
                node: Some(node),
            })
        } else {
            None
        }
    }

    /// Iterate over all nodes immediately dominated by the given node (not
    /// including the given node itself).
    pub fn immediately_dominated_by(&self, node: N) -> DominatedByIter<N> {
        DominatedByIter {
            iter: self.dominators.iter(),
            node,
        }
    }
}

/// Iterator for a node's dominators.
#[derive(Debug, Clone)]
pub struct DominatorsIter<'a, N>
where
    N: 'a + Copy + Eq + Hash,
{
    dominators: &'a Dominators<N>,
    node: Option<N>,
}

impl<'a, N> Iterator for DominatorsIter<'a, N>
where
    N: 'a + Copy + Eq + Hash,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.node.take();
        if let Some(next) = next {
            self.node = self.dominators.immediate_dominator(next);
        }
        next
    }
}

/// Iterator for nodes dominated by a given node.
#[derive(Debug, Clone)]
pub struct DominatedByIter<'a, N>
where
    N: 'a + Copy + Eq + Hash,
{
    iter: Iter<'a, N, N>,
    node: N,
}

impl<'a, N> Iterator for DominatedByIter<'a, N>
where
    N: 'a + Copy + Eq + Hash,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(next) = self.iter.next() {
            if next.1 == &self.node {
                return Some(*next.0);
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

/// The undefined dominator sentinel, for when we have not yet discovered a
/// node's dominator.
const UNDEFINED: usize = ::std::usize::MAX;

/// This is an implementation of the engineered ["Simple, Fast Dominance
/// Algorithm"][0] discovered by Cooper et al.
///
/// This algorithm is **O(|V|²)**, and therefore has slower theoretical running time
/// than the Lengauer-Tarjan algorithm (which is **O(|E| log |V|)**. However,
/// Cooper et al found it to be faster in practice on control flow graphs of up
/// to ~30,000 vertices.
///
/// [0]: http://www.cs.rice.edu/~keith/EMBED/dom.pdf
pub fn simple_fast<G>(graph: G) -> Dominators<NodeIndex>
where
    G: AsEntityGraph,
{
    let (roots, po, predecessor_sets) = simple_fast_po(graph);
    let length = po.len();

    debug_assert!(length > 0);

    // From here on out we use indices into `po` instead of actual
    // `NodeId`s wherever possible. This greatly improves the performance of
    // this implementation, but we have to pay a little bit of upfront cost to
    // convert our data structures to play along first.

    // Maps a node to its index into `po`.
    let node_to_po_idx: HashMap<_, _> = po
        .iter()
        .enumerate()
        .map(|(idx, &node)| (node, idx))
        .collect();

    // Maps a node's `po` index to its set of predecessors's indices
    // into `rpo` (as a vec).
    let idx_to_predecessor_vec =
        predecessor_sets_to_idx_vecs(&po, &node_to_po_idx, predecessor_sets);

    let mut dominators = vec![UNDEFINED; length + 1];
    let mut root_idxs = HashSet::new();

    // Simulate a real root that connects to all other roots
    for (n, i) in node_to_po_idx.iter() {
        if roots.contains(n) {
            dominators[*i] = length;
            root_idxs.insert(*i);
        }
    }

    // Simulated root that connects to all other roots
    dominators[length] = length;

    let mut changed = true;
    while changed {
        changed = false;

        // Iterate in reverse post order, skipping the virtual root.

        'inner: for idx in (0..length).rev() {
            // Skip all known roots
            if root_idxs.contains(&idx) {
                continue 'inner;
            }

            // Take the intersection of every predecessor's dominator set; that
            // is the current best guess at the immediate dominator for this
            // node.

            let new_idom_idx = {
                let mut predecessors = idx_to_predecessor_vec[idx]
                    .iter()
                    .filter(|&&p| dominators[p] != UNDEFINED);
                let new_idom_idx = predecessors.next().expect(
                    "Because the root is initialized to dominate itself, and is the \
                     first node in every path, there must exist a predecessor to this \
                     node that also has a dominator",
                );
                predecessors.fold(*new_idom_idx, |new_idom_idx, &predecessor_idx| {
                    intersect(&dominators, new_idom_idx, predecessor_idx)
                })
            };

            debug_assert!(new_idom_idx <= length);

            if new_idom_idx != dominators[idx] {
                dominators[idx] = new_idom_idx;
                changed = true;
            }
        }
    }

    // All done! Translate the indices back into proper `G::NodeId`s.

    debug_assert!(!dominators.iter().any(|&dom| dom == UNDEFINED));

    Dominators {
        roots,
        dominators: dominators[..length]
            .into_iter()
            .enumerate()
            // here we remove the notion of a simulated root, and make any node dominated by it
            // dominate itself
            .map(|(idx, dom_idx)| {
                (
                    po[idx],
                    if *dom_idx == length {
                        po[idx]
                    } else {
                        po[*dom_idx]
                    },
                )
            })
            .collect(),
    }
}

fn intersect(dominators: &[usize], mut finger1: usize, mut finger2: usize) -> usize {
    loop {
        match finger1.cmp(&finger2) {
            Ordering::Equal => return finger1,
            Ordering::Less => finger1 = dominators[finger1],
            Ordering::Greater => finger2 = dominators[finger2],
        }
    }
}

fn predecessor_sets_to_idx_vecs<N>(
    post_order: &Vec<N>,
    node_to_post_order_idx: &HashMap<N, usize>,
    mut predecessor_sets: HashMap<N, HashSet<N>>,
) -> Vec<Vec<usize>>
where
    N: Copy + Eq + Hash,
{
    post_order
        .iter()
        .map(|node| {
            predecessor_sets
                .remove(node)
                .map(|predecessors| {
                    predecessors
                        .into_iter()
                        .map(|p| *node_to_post_order_idx.get(&p).unwrap())
                        .collect()
                })
                .unwrap_or_else(Vec::new)
        })
        .collect()
}

type PredecessorSets<NodeId> = HashMap<NodeId, HashSet<NodeId>>;

fn simple_fast_po<G>(
    graph: G,
) -> (
    HashSet<NodeIndex>,
    Vec<NodeIndex>,
    PredecessorSets<NodeIndex>,
)
where
    G: AsEntityGraph,
{
    let (roots, po) = PostOrder::into_queue_with_roots(graph);
    let mut predecessor_sets = HashMap::new();

    for node in po.iter() {
        for successor in graph.neighbors(*node) {
            predecessor_sets
                .entry(successor)
                .or_insert_with(HashSet::new)
                .insert(*node);
        }
    }

    let roots = roots.into_iter().collect::<HashSet<_>>();
    let po = po.into_iter().collect();

    (roots, po, predecessor_sets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntityGraph;
    use fugue::ir::address::AddressValue;
    use fugue::ir::il::ecode::{EntityId, Location};
    use fugue::ir::space::{AddressSpace, Space, SpaceKind};

    use std::sync::Arc;

    #[test]
    fn test_iter_dominators() {
        let mut roots = HashSet::new();
        roots.insert(0);

        let doms: Dominators<u32> = Dominators {
            roots,
            dominators: [(2, 1), (1, 0), (0, 0)].iter().cloned().collect(),
        };

        let all_doms: Vec<_> = doms.dominators(2).unwrap().collect();
        assert_eq!(vec![2, 1, 0], all_doms);

        assert_eq!(None::<()>, doms.dominators(99).map(|_| unreachable!()));

        let strict_doms: Vec<_> = doms.strict_dominators(2).unwrap().collect();
        assert_eq!(vec![1, 0], strict_doms);

        assert_eq!(
            None::<()>,
            doms.strict_dominators(99).map(|_| unreachable!())
        );

        let dom_by: Vec<_> = doms.immediately_dominated_by(1).collect();
        assert_eq!(vec![2], dom_by);
        assert_eq!(None, doms.immediately_dominated_by(99).next());
    }

    #[test]
    fn test_iter_dominators_multiple_roots() {
        let mut g = EntityGraph::<()>::new();

        let spc = Space::new(SpaceKind::Processor, "ram", 8, 1, 0, None, 0);
        let aspc = Arc::new(AddressSpace::Space(spc));

        let n1 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 0u64), 0),
        ));
        let n2 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 1u64), 0),
        ));
        let n3 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 2u64), 0),
        ));
        let n4 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 3u64), 0),
        ));
        let n5 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 4u64), 0),
        ));
        let n6 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 5u64), 0),
        ));
        let n7 = g.add_node(EntityId::new(
            "blk",
            Location::new(AddressValue::new(aspc.clone(), 6u64), 0),
        ));

        // g1 (root: n1)
        g.add_edge(n1, n2, ());
        g.add_edge(n1, n3, ());
        g.add_edge(n2, n3, ());
        g.add_edge(n3, n2, ());
        g.add_edge(n3, n4, ());

        // g2 (root: n7)
        g.add_edge(n7, n4, ());
        g.add_edge(n7, n5, ());
        g.add_edge(n5, n6, ());

        let doms = simple_fast(&g);

        println!("{:#?}", doms.dominators);
    }
}
