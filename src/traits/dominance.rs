use petgraph::Direction;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph as DiGraph;
use petgraph::visit::{VisitMap, Visitable};

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use fixedbitset::FixedBitSet;

use crate::graphs::algorithms::dominance;
use crate::types::EntityGraph;

pub type DominanceFrontier = HashMap<NodeIndex, HashSet<NodeIndex>>;

#[derive(Debug, Clone)]
pub struct DominanceTree {
    tree: DiGraph<NodeIndex, ()>,
    roots: Vec<NodeIndex>,
}

impl Deref for DominanceTree {
    type Target = DiGraph<NodeIndex, ()>;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

impl DerefMut for DominanceTree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tree
    }
}

impl DominanceTree {
    pub fn visit_pre_order(&self) -> DominanceTreePreOrder {
        DominanceTreePreOrder::new(self)
    }
}

#[derive(Debug, Clone)]
pub struct DominanceTreePreOrder<'tree> {
    stack: Vec<NodeIndex>,
    discovered: FixedBitSet,
    tree: &'tree DominanceTree,
}

impl<'tree> DominanceTreePreOrder<'tree> {
    fn new(tree: &'tree DominanceTree) -> Self {
        Self {
            stack: tree.roots.clone(),
            discovered: tree.visit_map(),
            tree,
        }
    }
}

impl<'tree> Iterator for DominanceTreePreOrder<'tree> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            if self.discovered.visit(node) {
                for succ in self.tree.neighbors(node) {
                    if !self.discovered.is_visited(&succ) {
                        self.stack.push(succ);
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

#[derive(Debug, Clone)]
pub struct Dominance {
    pub tree: DominanceTree,
    pub frontier: DominanceFrontier,
}

pub trait DominanceExt {
    fn dominance_tree(&self) -> DominanceTree;
    fn dominance_frontier(&self) -> DominanceFrontier;
    fn dominance(&self) -> Dominance;
}

impl<E> DominanceExt for EntityGraph<E> {
    fn dominance_tree(&self) -> DominanceTree {
        let graph = self;
        let mut tree = DiGraph::new();

        let dominators = dominance::simple_fast(graph);
        let roots = dominators.roots();

        let tree_roots = roots.iter().map(|root| tree.add_node(*root)).collect();

        for d in graph.node_indices().filter(|nx| !roots.contains(nx)) {
            tree.add_node(d);
        }

        for d in graph.node_indices() {
            if let Some(idom) = dominators.immediate_dominator(d) {
                tree.add_edge(idom, d, ());
            }
        }

        DominanceTree {
            roots: tree_roots,
            tree,
        }
    }

    fn dominance_frontier(&self) -> DominanceFrontier {
        let graph = self;
        let mut mapping = DominanceFrontier::new();

        let dominators = dominance::simple_fast(graph);

        for d in graph.node_indices() {
            let idom = if let Some(idom) = dominators.immediate_dominator(d) {
                idom
            } else {
                continue
            };

            for ni in graph.neighbors_directed(d, Direction::Incoming) {
                let mut np = ni;
                while np != idom {
                    mapping.entry(np).or_insert_with(HashSet::new).insert(d);
                    np = if let Some(np_dom) = dominators.immediate_dominator(np) {
                        np_dom
                    } else {
                        break
                    };
                }
            }
        }

        mapping
    }

    fn dominance(&self) -> Dominance {
        let graph = self;

        let mut tree = DiGraph::new();
        let mut mapping = DominanceFrontier::new();

        let dominators = dominance::simple_fast(graph);
        let roots = dominators.roots();

        let tree_roots = roots.iter().map(|root| tree.add_node(*root)).collect();

        for d in graph.node_indices() {
            if !roots.contains(&d) {
                tree.add_node(d);
            }

            let idom = if let Some(idom) = dominators.immediate_dominator(d) {
                idom
            } else {
                continue
            };

            for ni in graph.neighbors_directed(d, Direction::Incoming) {
                let mut np = ni;
                while np != idom {
                    mapping.entry(np).or_insert_with(HashSet::new).insert(d);
                    np = if let Some(np_dom) = dominators.immediate_dominator(np) {
                        np_dom
                    } else {
                        break
                    };
                }
            }
        }

        for d in graph.node_indices() {
            if let Some(idom) = dominators.immediate_dominator(d) {
                tree.add_edge(idom, d, ());
            }
        }

        Dominance {
            frontier: mapping,
            tree: DominanceTree {
                tree,
                roots: tree_roots,
            },
        }
    }
}
