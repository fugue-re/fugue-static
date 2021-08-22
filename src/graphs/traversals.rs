pub mod po;
pub use po::PostOrder;

pub mod rpo;
pub use rpo::RevPostOrder;

use petgraph::graph::NodeIndex;

use std::borrow::Borrow;
use std::collections::VecDeque;

use crate::types::EntityGraph;

pub trait Traversal<'a> {
    fn into_queue_with_roots<E, G>(graph: G) -> (Vec<NodeIndex>, VecDeque<NodeIndex>)
        where G: Borrow<EntityGraph<E>> + 'a;

    fn into_queue<E, G>(graph: G) -> VecDeque<NodeIndex>
        where G: Borrow<EntityGraph<E>> + 'a {
        Self::into_queue_with_roots(graph).1
    }

}
