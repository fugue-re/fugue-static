// uses specified entry-points to perform traversal
pub mod simple;

// computes entry/exits using heuristics while performing traversal
pub mod heuristic;

pub use simple::{PostOrder, RevPostOrder};
