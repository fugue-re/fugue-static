pub mod po;
pub use po::PostOrder;

pub mod rpo;
pub use rpo::RevPostOrder;

use petgraph::graph::NodeIndex;
use std::collections::VecDeque;

use crate::types::EntityGraph;

pub trait Traversal<'a> {
    fn into_queue<E, G>(graph: G) -> VecDeque<NodeIndex>
        where G: AsRef<EntityGraph<E>> + 'a;
}
