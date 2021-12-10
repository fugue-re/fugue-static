use crate::models::{Block, Function};
use crate::models::memory::{Memory, Region, RegionIOError};
use crate::models::lifter::{Lifter, LifterBuilder, LifterBuilderError};
use crate::models::function::{FunctionBuilder, Error as FunctionBuilderError};
use crate::traits::EntityRefCollector;
use crate::transforms::{NormaliseVariables, VariableNormaliser};
use crate::types::{Entity, EntityIdMapping, EntityLocMapping, EntityRef, Id, Identifiable, Locatable};
use crate::traits::oracle::*;

use fugue::bytes::Endian;
use fugue::ir::{Address, IntoAddress, Translator};
use fugue::ir::il::Location;
use fugue::ir::disassembly::ContextDatabase;

use std::borrow::{Borrow, Cow};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use thiserror::Error;

pub struct ProjectBuilder {
    lifter_builder: LifterBuilder,
}

#[derive(Debug, Error)]
pub enum ProjectBuilderError {
    #[error(transparent)]
    LifterBuilder(#[from] LifterBuilderError),
}

impl ProjectBuilder {
    pub fn new_with(
        path: impl AsRef<Path>,
        ignore_errors: bool,
    ) -> Result<Self, ProjectBuilderError> {
        Ok(Self {
            lifter_builder: LifterBuilder::new_with(path, ignore_errors)?,
        })
    }

    pub fn new(path: impl AsRef<Path>) -> Result<Self, ProjectBuilderError> {
        Ok(Self {
            lifter_builder: LifterBuilder::new(path)?,
        })
    }

    pub fn project(
        &self,
        name: impl Into<Cow<'static, str>>,
        arch: impl Into<Cow<'static, str>>,
        convention: impl AsRef<str>,
    ) -> Result<Entity<Project>, ProjectBuilderError> {
        Ok(Project::new(
            name,
            self.lifter_builder.build(arch, convention)?,
        ))
    }

    pub fn project_with(
        &self,
        name: impl Into<Cow<'static, str>>,
        processor: impl AsRef<str>,
        endian: Endian,
        bits: u32,
        variant: impl AsRef<str>,
        convention: impl AsRef<str>,
    ) -> Result<Entity<Project>, ProjectBuilderError> {
        Ok(Project::new(
            name,
            self.lifter_builder.build_with(processor, endian, bits, variant, convention)?,
        ))
    }
}

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("block oracle inconsistent with block reported by function oracle for {0}")]
    BlockOracleUnsized(Location),
    #[error("block oracle bounds containing {0} inconsistent with known regions")]
    BlockOracleUnmappedBounds(Location),
    #[error("function oracle inconsistent with user-defined function at {0}")]
    FunctionOracleInconsistent(Location),
    #[error(transparent)]
    FunctionBuilder(#[from] FunctionBuilderError),
    #[error(transparent)]
    RegionAccess(#[from] RegionIOError),
}

#[derive(Clone)]
pub struct Project {
    name: Cow<'static, str>,

    lifter: Lifter,
    disassembly_context: ContextDatabase,

    memory: Memory,

    blk_oracle: Option<Arc<RwLock<dyn BlockOracle>>>,
    fcn_oracle: Option<Arc<RwLock<dyn FunctionOracle>>>,
    fcn_oracle_starts: BTreeSet<Location>,

    blks: BTreeMap<Id<Block>, Entity<Block>>,
    blks_to_locs: BTreeMap<Id<Block>, Location>,
    locs_to_blks: BTreeMap<Location, BTreeSet<Id<Block>>>,

    fcns: BTreeMap<Id<Function>, Entity<Function>>,
    fcns_to_locs: BTreeMap<Id<Function>, Location>,
    locs_to_fcns: BTreeMap<Location, Id<Function>>,
    syms_to_fcns: BTreeMap<Cow<'static, str>, Id<Function>>,
}

impl Project {
    pub fn new(name: impl Into<Cow<'static, str>>, lifter: Lifter) -> Entity<Self> {
        Entity::new("project", Self {
            name: name.into(),

            disassembly_context: lifter.context(),
            lifter,

            memory: Memory::new("M"),

            blk_oracle: None,
            fcn_oracle: None,
            fcn_oracle_starts: Default::default(),

            blks: Default::default(),
            blks_to_locs: Default::default(),
            locs_to_blks: Default::default(),

            fcns: Default::default(),
            fcns_to_locs: Default::default(),
            locs_to_fcns: Default::default(),
            syms_to_fcns: Default::default(),
        })
    }

    pub fn set_block_oracle<O: BlockOracle + 'static>(&mut self, oracle: O) {
        self.blk_oracle = Some(Arc::new(RwLock::new(oracle)))
    }

    pub fn set_function_oracle<O: FunctionOracle + 'static>(&mut self, oracle: O) {
        let oracle = Arc::new(RwLock::new(oracle));
        self.fcn_oracle_starts.extend(oracle.read().function_starts(self.lifter.translator()).into_iter());
        self.fcn_oracle = Some(oracle);
    }

    pub fn add_region_mapping(&mut self, region: Entity<Region>) {
        self.memory.add_region(region);
    }

    pub fn add_region_mapping_with(
        &mut self,
        name: impl Into<Arc<str>>,
        addr: impl Into<Address>,
        endian: Endian,
        bytes: impl Into<Vec<u8>>,
    ) {
        self.memory.add_region(Region::new(name, addr, endian, bytes));
    }

    pub fn add_function(&mut self, location: impl IntoAddress) -> Result<Id<Function>, ProjectError> {
        let location = location.into_address_value(self.lifter.translator().manager().default_space_ref());
        let location = location.into();
        if !self.fcn_oracle_starts.contains(&location) {
            return Err(ProjectError::FunctionOracleInconsistent(location))
        }

        let sym = self.fcn_oracle.as_ref().and_then(|o| o.read().function_symbol(&location))
            .unwrap_or_else(|| Cow::from(format!("sub_{}", location.address())));

        let mut norm = VariableNormaliser::new_with(self.lifter.translator(), self.lifter().temporary_space_id());

        let mut fcn_builder = FunctionBuilder::new(
            self.lifter.translator(),
            &mut self.disassembly_context,
            location.address().offset(),
            &*sym,
        );

        let blks = self.fcn_oracle.as_ref().and_then(|o| o.read().function_blocks(&location))
            .unwrap_or_default();


        for blk in blks.into_iter() {
            let addr = blk.address();
            let basic_addr = Address::from(&*addr);
            let region = self.memory.find_region(&basic_addr)
                .ok_or_else(|| ProjectError::BlockOracleUnmappedBounds(blk.clone()))?;
            let size_hint = self.blk_oracle.as_ref().and_then(|o| o.read().block_size(&blk))
                .ok_or_else(|| ProjectError::BlockOracleUnsized(blk.clone()))?;

            let bytes = region.view_bytes(&basic_addr, size_hint)?;
            fcn_builder.add_block_with(blk.address().offset(), bytes, |ecode| ecode.normalise_variables(&mut norm))?;
        }

        let (fcn, blks) = fcn_builder.build();

        for blk in blks {
            let id = blk.id();
            let loc = blk.location();

            if let Some(ref o) = self.blk_oracle {
                o.write().block_identity(&loc, id);
            }

            self.blks.insert(id, blk);
            self.blks_to_locs.insert(id, loc.clone());
            self.locs_to_blks.entry(loc).or_default().insert(id);
        }

        let id = fcn.id();
        let loc = fcn.location();

        if let Some(ref o) = self.fcn_oracle {
            o.write().function_identity(&loc, id);
        }

        self.fcns.insert(id, fcn);
        self.fcns_to_locs.insert(id, loc.clone());
        self.locs_to_fcns.insert(loc, id);
        self.syms_to_fcns.insert(sym, id);

        Ok(id)
    }

    pub fn lifter(&self) -> &Lifter {
        &self.lifter
    }
}

impl EntityIdMapping<Block> for Project {
    fn lookup_by_id(&self, id: Id<Block>) -> Option<EntityRef<Block>> {
        self.blks.get(&id).map(Cow::Borrowed)
    }
}

impl EntityLocMapping<Block> for Project {
    fn lookup_by_location_with<'a, C: EntityRefCollector<'a, Block>>(&'a self, loc: &Location, collect: &mut C) {
        if let Some(ids) = self.locs_to_blks.get(loc) {
            for id in ids.iter() {
                if let Some(e) = self.lookup_by_id(*id) {
                    collect.insert(e);
                }
            }
        }
    }
}

impl EntityIdMapping<Function> for Project {
    fn lookup_by_id(&self, id: Id<Function>) -> Option<EntityRef<Function>> {
        self.fcns.get(&id).map(Cow::Borrowed)
    }
}

impl EntityLocMapping<Function> for Project {
    fn lookup_by_location_with<'a, C: EntityRefCollector<'a, Function>>(&'a self, loc: &Location, collect: &mut C) {
        if let Some(id) = self.locs_to_fcns.get(loc) {
            if let Some(e) = self.lookup_by_id(*id) {
                collect.insert(e);
            }
        }
    }
}

impl BlockOracle for Project {
    fn block_size(&self, loc: &Location) -> Option<usize> {
        self.blk_oracle.as_ref()?.read().block_size(loc)
    }

    fn block_succs(&self, loc: &Location) -> Option<Vec<Location>> {
        self.blk_oracle.as_ref()?.read().block_succs(loc)
    }

    fn block_identity(&mut self, loc: &Location, id: Id<Block>) {
        if let Some(oracle) = self.blk_oracle.as_ref() {
            oracle.write().block_identity(loc, id)
        }
    }
}

impl FunctionOracle for Project {
    fn function_starts(&self, _translator: &Translator) -> Vec<Location> {
        self.fcn_oracle_starts.iter().cloned().collect()
    }

    fn function_symbol(&self, loc: &Location) -> Option<Cow<'static, str>> {
        self.fcn_oracle.as_ref()?.read().function_symbol(loc)
    }

    fn function_blocks(&self, loc: &Location) -> Option<Vec<Location>> {
        self.fcn_oracle.as_ref()?.read().function_blocks(loc)
    }

    fn function_identity(&mut self, loc: &Location, id: Id<Function>) {
        if let Some(oracle) = self.fcn_oracle.as_ref() {
            oracle.write().function_identity(loc, id)
        }
    }
}

impl Borrow<Translator> for Project {
    fn borrow(&self) -> &Translator {
        self.lifter.translator()
    }
}

impl Borrow<Translator> for &'_ Project {
    fn borrow(&self) -> &Translator {
        self.lifter.translator()
    }
}
