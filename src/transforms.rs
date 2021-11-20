pub mod egraph;
pub mod normalise;
pub mod ssa;

pub use normalise::{
    NormaliseVariables, NormaliseAliases,
    VariableNormaliser, VariableAliasNormaliser,
};
pub use ssa::SSATransform as SSA;
