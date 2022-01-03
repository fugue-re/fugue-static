pub mod constant;
pub use constant::{ConstExpr, ConstEvaluator};

pub mod symbolic;

pub mod var_classes;
pub use var_classes::{VClassMap, VLattice};
