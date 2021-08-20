use petgraph::Direction;
use petgraph::algo::dominators;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph as DiGraph;

use std::collections::{HashMap, HashSet};

use crate::types::EntityGraph;

pub type DominanceFrontier = HashMap<NodeIndex, HashSet<NodeIndex>>;
pub type DominanceTree = DiGraph<NodeIndex, ()>;

#[derive(Debug, Clone)]
pub struct Dominance {
    pub tree: DominanceTree,
    pub tree_root: NodeIndex,
    pub frontier: DominanceFrontier,
}

pub trait DominanceExt {
    fn dominance_tree(&self, root: NodeIndex) -> (NodeIndex, DominanceTree);
    fn dominance_frontier(&self, root: NodeIndex) -> DominanceFrontier;
    fn dominance(&self, root: NodeIndex) -> Dominance;
}

impl<E> DominanceExt for EntityGraph<E> {
    fn dominance_tree(&self, root: NodeIndex) -> (NodeIndex, DominanceTree) {
        let graph = self;
        let mut tree = DominanceTree::new();

        let dominators = dominators::simple_fast(graph, root);

        let tree_root = tree.add_node(root);

        for d in graph.node_indices().filter(|nx| *nx != root) {
            tree.add_node(d);
        }

        for d in graph.node_indices() {
            if let Some(idom) = dominators.immediate_dominator(d) {
                tree.add_edge(idom, d, ());
            }
        }

        (tree_root, tree)
    }

    fn dominance_frontier(&self, root: NodeIndex) -> DominanceFrontier {
        let graph = self;
        let mut mapping = DominanceFrontier::new();

        let dominators = dominators::simple_fast(graph, root);

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

    fn dominance(&self, root: NodeIndex) -> Dominance {
        let graph = self;

        let mut tree = DominanceTree::new();
        let mut mapping = DominanceFrontier::new();

        let dominators = dominators::simple_fast(graph, root);
        let tree_root = tree.add_node(root);

        for d in graph.node_indices() {
            if d != root {
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
            tree,
            tree_root,
        }
    }
}
