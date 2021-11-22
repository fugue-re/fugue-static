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
use std::ops::{Deref, DerefMut};

use fixedbitset::FixedBitSet;
use petgraph::stable_graph::StableDiGraph as DiGraph;
use petgraph::visit::{VisitMap, Visitable};

use crate::graphs::entity::{AsEntityGraph, VertexIndex};

pub type DominanceFrontier<V> = HashMap<VertexIndex<V>, HashSet<VertexIndex<V>>>;

#[derive(Debug, Clone)]
pub struct DominanceTree<V> {
    tree: DiGraph<VertexIndex<V>, ()>,
    roots: Vec<VertexIndex<V>>,
}

impl<V> Default for DominanceTree<V> {
    fn default() -> Self {
        Self {
            tree: DiGraph::new(),
            roots: Vec::default(),
        }
    }
}

/// The dominance relation for some graph and root.
#[derive(Debug, Clone)]
pub struct Dominators<V> {
    roots: HashSet<VertexIndex<V>>,
    dominators: HashMap<VertexIndex<V>, VertexIndex<V>>,
    dominance_tree: DominanceTree<V>,
    dominance_frontier: DominanceFrontier<V>,
}

impl<V> Deref for DominanceTree<V> {
    type Target = DiGraph<VertexIndex<V>, ()>;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

impl<V> DerefMut for DominanceTree<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tree
    }
}

impl<V> DominanceTree<V> {
    pub fn pre_order(&self) -> DominanceTreePreOrder<V> {
        DominanceTreePreOrder::new(self)
    }
}

#[derive(Debug, Clone)]
pub struct DominanceTreePreOrder<'tree, V> {
    stack: Vec<VertexIndex<V>>,
    discovered: FixedBitSet,
    tree: &'tree DominanceTree<V>,
}

impl<'tree, V> DominanceTreePreOrder<'tree, V> {
    fn new(tree: &'tree DominanceTree<V>) -> Self {
        Self {
            stack: tree.roots.clone(),
            discovered: tree.visit_map(),
            tree,
        }
    }
}

impl<'tree, V> Iterator for DominanceTreePreOrder<'tree, V> {
    type Item = VertexIndex<V>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            if self.discovered.visit(*node) {
                for succ in self.tree.neighbors(*node) {
                    if !self.discovered.is_visited(&succ) {
                        self.stack.push(succ.into());
                    }
                }
                return Some(node);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tree.node_count() - self.discovered.count_ones(..);
        (remaining, Some(remaining))
    }
}

impl<V> Dominators<V> {
    /// Get the root node(s) used to construct these dominance relations.
    pub fn roots(&self) -> &HashSet<VertexIndex<V>> {
        &self.roots
    }

    /// Get the immediate dominator of the given node.
    ///
    /// Returns `None` for any node that is not reachable from the root, and for
    /// the root itself.
    pub fn immediate_dominator(&self, node: VertexIndex<V>) -> Option<VertexIndex<V>> {
        if self.roots.contains(&node) {
            None
        } else {
            self.dominators.get(&node).copied()
        }
    }

    /// Iterate over the given node's strict dominators.
    ///
    /// If the given node is not reachable from the root, then `None` is
    /// returned.
    pub fn strict_dominators(&self, node: VertexIndex<V>) -> Option<DominatorsIter<V>> {
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
    pub fn dominators(&self, node: VertexIndex<V>) -> Option<DominatorsIter<V>> {
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
    pub fn immediately_dominated_by(&self, node: VertexIndex<V>) -> DominatedByIter<V> {
        DominatedByIter {
            iter: self.dominators.iter(),
            node,
        }
    }

    pub fn dominance_tree(&self) -> &DominanceTree<V> {
        &self.dominance_tree
    }

    pub fn dominance_frontier(&self) -> &DominanceFrontier<V> {
        &self.dominance_frontier
    }
}

fn compute_tree_and_frontier<'a, V, E, G>(graph: G, dominators: &mut Dominators<V>, reversed: bool)
where
    V: Clone + 'a,
    G: AsEntityGraph<'a, V, E>,
{
    let roots = dominators.roots();
    let mut tree = DominanceTree::default();//&mut dominators.dominance_tree.tree;
    let mut mapping = DominanceFrontier::default();//&mut dominators.dominance_frontier;


    let mut tree_mapping = HashMap::new();
    for root in roots.iter() {
        let idx = tree.add_node(*root);
        tree_mapping.insert(*root, idx);
        tree.roots.push(idx.into());
    }

    if reversed {
        for (_, d, _) in graph.entity_graph().post_order() {
            if !roots.contains(&d) {
                tree_mapping.insert(d, tree.add_node(d));
            }

            let idom = if let Some(idom) = dominators.immediate_dominator(d) {
                idom
            } else {
                continue;
            };

            for (ni, _) in graph.entity_graph().successors(d) {
                let mut np = ni;
                while np != idom {
                    mapping.entry(np).or_insert_with(HashSet::new).insert(d);
                    np = if let Some(np_dom) = dominators.immediate_dominator(np) {
                        np_dom
                    } else {
                        break;
                    };
                }
            }
        }

        for (_, d, _) in graph.entity_graph().post_order() {
            if let Some(idom) = dominators.immediate_dominator(d) {
                tree.add_edge(tree_mapping[&idom], tree_mapping[&d], ());
            }
        }
    } else {
        for (_, d, _) in graph.entity_graph().reverse_post_order() {
            if !roots.contains(&d) {
                tree_mapping.insert(d, tree.add_node(d));
            }

            let idom = if let Some(idom) = dominators.immediate_dominator(d) {
                idom
            } else {
                continue;
            };

            for (ni, _) in graph.entity_graph().predecessors(d) {
                let mut np = ni;
                while np != idom {
                    mapping.entry(np).or_insert_with(HashSet::new).insert(d);
                    np = if let Some(np_dom) = dominators.immediate_dominator(np) {
                        np_dom
                    } else {
                        break;
                    };
                }
            }
        }

        for (_, d, _) in graph.entity_graph().reverse_post_order() {
            if let Some(idom) = dominators.immediate_dominator(d) {
                tree.add_edge(tree_mapping[&idom], tree_mapping[&d], ());
            }
        }
    }

    dominators.dominance_tree = tree;
    dominators.dominance_frontier = mapping;
}

/// Iterator for a node's dominators.
#[derive(Debug, Clone)]
pub struct DominatorsIter<'a, V> {
    dominators: &'a Dominators<V>,
    node: Option<VertexIndex<V>>,
}

impl<'a, V> Iterator for DominatorsIter<'a, V> {
    type Item = VertexIndex<V>;

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
pub struct DominatedByIter<'a, V> {
    iter: Iter<'a, VertexIndex<V>, VertexIndex<V>>,
    node: VertexIndex<V>,
}

impl<'a, V> Iterator for DominatedByIter<'a, V> {
    type Item = VertexIndex<V>;

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
pub fn simple_fast<'a, V, E, G>(graph: G, reversed: bool) -> Dominators<V>
where
    G: AsEntityGraph<'a, V, E>,
    V: Clone + 'a,
{
    let (roots, o, predecessor_sets) = if reversed {
        simple_fast_rpo(graph.entity_graph())
    } else {
        simple_fast_po(graph.entity_graph())
    };
    let length = o.len();

    debug_assert!(length > 0);

    // From here on out we use indices into `o` instead of actual
    // `NodeId`s wherever possible. This greatly improves the performance of
    // this implementation, but we have to pay a little bit of upfront cost to
    // convert our data structures to play along first.

    // Maps a node to its index into `o`.
    let node_to_oidx: HashMap<_, _> = o
        .iter()
        .enumerate()
        .map(|(idx, &node)| (node, idx))
        .collect();

    // Maps a node's `po` index to its set of predecessors's indices
    // into `rpo` (as a vec).
    let idx_to_predecessor_vec =
        predecessor_sets_to_idx_vecs(&o, &node_to_oidx, predecessor_sets);

    let mut dominators = vec![UNDEFINED; length + 1];
    let mut root_idxs = HashSet::new();

    // Simulate a real root that connects to all other roots
    for (n, i) in node_to_oidx.iter() {
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

    let mut doms = Dominators {
        roots,
        dominators: dominators[..length]
            .into_iter()
            .enumerate()
            // here we remove the notion of a simulated root, and make any node dominated by it
            // dominate itself
            .map(|(idx, dom_idx)| {
                (
                    o[idx],
                    if *dom_idx == length {
                        o[idx]
                    } else {
                        o[*dom_idx]
                    },
                )
            })
            .collect(),
        dominance_frontier: DominanceFrontier::default(),
        dominance_tree: DominanceTree::default(),
    };

    compute_tree_and_frontier(graph, &mut doms, reversed);
    doms
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
    order: &Vec<N>,
    node_to_order_idx: &HashMap<N, usize>,
    mut predecessor_sets: HashMap<N, HashSet<N>>,
) -> Vec<Vec<usize>>
where
    N: Copy + Eq + Hash,
{
    order
        .iter()
        .map(|node| {
            predecessor_sets
                .remove(node)
                .map(|predecessors| {
                    predecessors
                        .into_iter()
                        .map(|p| *node_to_order_idx.get(&p).unwrap())
                        .collect()
                })
                .unwrap_or_else(Vec::new)
        })
        .collect()
}

type PredecessorSets<V> = HashMap<VertexIndex<V>, HashSet<VertexIndex<V>>>;

fn simple_fast_po<'a, V, E, G>(
    graph: G,
) -> (
    HashSet<VertexIndex<V>>,
    Vec<VertexIndex<V>>,
    PredecessorSets<V>,
)
where
    V: Clone + 'a,
    G: AsEntityGraph<'a, V, E>,
{
    let mut po = graph.entity_graph().post_order();
    let mut predecessor_sets = HashMap::new();

    let mut nodes = Vec::with_capacity(po.size_hint().0);

    for (_, node, _) in &mut po {
        for (successor, _) in graph.entity_graph().successors(node) {
            predecessor_sets
                .entry(successor)
                .or_insert_with(HashSet::new)
                .insert(node);
        }
        nodes.push(node);
    }

    let roots = po.starting_vertices().into_iter().collect::<HashSet<_>>();

    (roots, nodes, predecessor_sets)
}

fn simple_fast_rpo<'a, V, E, G>(
    graph: G,
) -> (
    HashSet<VertexIndex<V>>,
    Vec<VertexIndex<V>>,
    PredecessorSets<V>,
)
where
    V: Clone + 'a,
    G: AsEntityGraph<'a, V, E>,
{
    let mut rpo = graph.entity_graph().reverse_post_order();
    let mut predecessor_sets = HashMap::new();

    let mut nodes = Vec::with_capacity(rpo.size_hint().0);

    for (_, node, _) in &mut rpo {
        for (successor, _) in graph.entity_graph().predecessors(node) {
            predecessor_sets
                .entry(successor)
                .or_insert_with(HashSet::new)
                .insert(node);
        }
        nodes.push(node);
    }

    let roots = rpo.terminal_vertices().into_iter().collect::<HashSet<_>>();

    (roots, nodes, predecessor_sets)
}

pub trait Dominance<V, E> {
    fn dominators(&self) -> Dominators<V>;
    fn post_dominators(&self) -> Dominators<V>;
}

impl<'a, V, E, G> Dominance<V, E> for G
where
    V: Clone + 'a,
    G: AsEntityGraph<'a, V, E>,
{
    fn dominators(&self) -> Dominators<V> {
        simple_fast(self.entity_graph(), false)
    }

    fn post_dominators(&self) -> Dominators<V> {
        simple_fast(self.entity_graph(), true)
    }
}

/*
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
*/
