use fugue::db::Database;
use fugue::ir::Translator;
use fugue::ir::il::Location;

use std::borrow::Cow;
use std::collections::BTreeMap;

use intervals::Interval;
use intervals::collections::{DisjointIntervalSet, DisjointIntervalMap};

use crate::models::{Block, Function};
use crate::types::Id;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NullOracle;

impl BlockOracle for NullOracle { }
impl FunctionOracle for NullOracle { }

#[derive(Debug, Default, Clone, Hash)]
pub struct OracleBlockMapping(DisjointIntervalSet<u64>);

impl OracleBlockMapping {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn map_block(&mut self, addr: u64, size: usize) {
        let iv = Interval::from(addr..=addr + (size as u64 - 1));
        if self.0.find(&iv).is_some() {
            panic!("block mapping is not disjoint");
        }
        self.0.insert(iv);
    }
    
    pub fn location_bounds(&self, loc: &Location) -> Option<Interval<u64>> {
        self.0.find_point(loc.address().offset())
           .map(|e| Interval::from(*e.interval().start()..=*e.interval().end()))
    }
}

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
    #[allow(unused)]
    fn function_starts(&self, translator: &Translator) -> Vec<Location> {
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

#[derive(Debug, Default, Clone, Hash)]
pub struct DatabaseBlockOracle {
    cache: OracleBlockMapping,
    blocks: DisjointIntervalMap<u64, (usize, Vec<u64>)>,
}

impl BlockOracle for DatabaseBlockOracle {
    fn block_size(&self, loc: &Location) -> Option<usize> {
        let iv = self.cache.location_bounds(loc)?;
        self.blocks.find_exact(iv).map(|e| e.value().0)
    }

    fn block_succs(&self, loc: &Location) -> Option<Vec<Location>> {
        let iv = self.cache.location_bounds(loc)?;
        self.blocks.find_exact(iv).map(|e| {
            e.value().1.iter().map(|addr| loc.address().wrap(*addr).into()).collect()
        })
    }
}

#[derive(Debug, Default, Clone, Hash)]
pub struct DatabaseFunctionOracle {
    functions: BTreeMap<u64, (String, Vec<u64>)>,
}

impl FunctionOracle for DatabaseFunctionOracle {
    fn function_starts(&self, translator: &Translator) -> Vec<Location> {
        self.functions.keys()
            .map(|addr| translator.address(*addr).into())
            .collect()
    }
    
    fn function_symbol(&self, loc: &Location) -> Option<Cow<'static, str>> {
        self.functions.get(&loc.address().offset())
            .map(|(s, _)| s.to_owned().into())
    }

    fn function_blocks(&self, loc: &Location) -> Option<Vec<Location>> {
        self.functions.get(&loc.address().offset())
            .map(|(_, blks)| blks.iter().map(|addr| loc.address().wrap(*addr).into()).collect())
    }
}

pub fn database_oracles(db: &Database) -> (DatabaseBlockOracle, DatabaseFunctionOracle) {
    let mut dbo = DatabaseBlockOracle::default();
    let mut dbf = DatabaseFunctionOracle::default();
    
    for f in db.functions() {
        let sym = f.name();
        let addr = f.address();
        
        let blks = f.blocks();
        let mut blk_starts = Vec::with_capacity(blks.len());

        for blk in blks.iter() {
            dbo.cache.map_block(blk.address(), blk.len());

            let interval = Interval::from(blk.address()..=blk.address() + (blk.len() as u64 - 1));
            let succs = blk.successors().iter().map(move |succ| {
                let sblk = &blks[succ.target_id().index()];
                sblk.address()
            })
            .collect::<Vec<_>>();

            dbo.blocks.insert(interval, (blk.len(), succs));

            blk_starts.push(blk.address());
        }
        
        dbf.functions.insert(addr, (sym.to_owned(), blk_starts));
    }

    (dbo, dbf)
}