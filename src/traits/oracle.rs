use fugue::ir::il::Location;
use std::borrow::Cow;

use crate::models::{Block, Function};
use crate::types::Id;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NullOracle;

impl BlockOracle for NullOracle { }
impl FunctionOracle for NullOracle { }

pub trait BlockOracle {
    #[allow(unused)]
    fn block_size(&self, loc: &Location) -> Option<usize> {
        None
    }

    #[allow(unused)]
    fn block_succs(&self, loc: &Location) -> Option<Vec<Location>> {
        None
    }
    
    #[allow(unused)]
    fn block_identity(&mut self, loc: &Location, id: Id<Block>) { }
}

pub trait FunctionOracle {
    fn function_starts(&self) -> Vec<Location> {
        Vec::default()
    }

    #[allow(unused)]
    fn function_blocks(&self, loc: &Location) -> Option<Vec<Location>> {
        None
    }

    #[allow(unused)]
    fn function_symbol(&self, loc: &Location) -> Option<Cow<'static, str>> {
        None
    }

    #[allow(unused)]
    fn function_identity(&mut self, loc: &Location, id: Id<Function>) { }
}