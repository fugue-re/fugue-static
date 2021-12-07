use fugue::ir::il::Location;
use std::borrow::Cow;

pub trait BlockOracle {
    fn block_size(&self, loc: &Location) -> Option<usize>;
    fn block_succs(&self, loc: &Location) -> Option<Vec<Location>>;
}

pub trait FunctionOracle {
    fn function_starts(&self) -> Vec<Location>;
    fn function_blocks(&self, loc: &Location) -> Option<Vec<Location>>;
    fn function_symbol(&self, loc: &Location) -> Option<Cow<'static, str>>;
}