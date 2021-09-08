pub mod po;
pub use po::PostOrder;

pub mod rpo;
pub use rpo::RevPostOrder;

use petgraph::graph::NodeIndex;

use std::collections::VecDeque;

use crate::types::AsEntityGraph;

pub trait Traversal<'a> {
    fn into_queue_with_roots<G>(graph: G) -> (Vec<NodeIndex>, VecDeque<NodeIndex>)
        where G: AsEntityGraph + 'a;

    fn into_queue<G>(graph: G) -> VecDeque<NodeIndex>
        where G: AsEntityGraph + 'a {
        Self::into_queue_with_roots(graph).1
    }

}
