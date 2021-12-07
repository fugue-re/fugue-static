use std::collections::{BTreeSet, HashMap};

use fugue::db;

use fugue::ir::Translator;
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::{BranchTarget, ECode, Location, Stmt};

use crate::models::cfg::{BranchKind, CFG};
use crate::models::{Block, BlockLifter};
use crate::traits::{BlockOracle, StmtExt};
use crate::traits::oracle::NullOracle;
use crate::types::{Id, Identifiable, LocationTarget, Locatable, LocatableId, Entity, EntityIdMapping, EntityLocMapping};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Lifting(#[from] crate::models::block::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    id: LocatableId<Function>,
    symbol: String,
    block_ids: HashMap<Id<Block>, Location>,
    callers: HashMap<LocatableId<Function>, LocatableId<Stmt>>,
}

impl Identifiable<Function> for Function {
    fn id(&self) -> Id<Function> {
        self.id.id()
    }
}

impl Locatable for Function {
    fn location(&self) -> Location {
        self.id.location()
    }
}

impl Function {
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn blocks(&self) -> &HashMap<Id<Block>, Location> {
        &self.block_ids
    }

    pub fn callers(&self) -> &HashMap<LocatableId<Function>, LocatableId<Stmt>> {
        &self.callers
    }

    pub fn callers_mut(&mut self) -> &mut HashMap<LocatableId<Function>, LocatableId<Stmt>> {
        &mut self.callers
    }

    pub fn cfg<'db, M>(&self, mapping: &'db M) -> CFG<'db, Block>
    where M: 'db + EntityIdMapping<Block> + EntityLocMapping<Block> {
        self.cfg_with(mapping, &NullOracle)
    }

    pub fn cfg_with<'db, M, O>(&self, mapping: &'db M, oracle: &O) -> CFG<'db, Block>
    where M: 'db + EntityIdMapping<Block> + EntityLocMapping<Block>,
          O: 'db + BlockOracle {
        let mut cfg = CFG::new();
        let mut succs = Vec::new();

        for blkid in self.block_ids.keys() {
            let blk = &mapping.lookup_by_id(*blkid).expect("block exists");
            if blk.location() == self.location() {
                cfg.add_root_entity(blk);
            } else {
                cfg.add_entity(blk);
            }
        }

        for blkid in self.block_ids.keys() {
            let blk = &mapping.lookup_by_id(*blkid).expect("block exists");
            let mut oracle_resolved = oracle.block_succs(&blk.location())
                .unwrap_or_default()
                .into_iter()
                .collect::<BTreeSet<_>>();
            
            for known in blk.next_blocks() {
                if let LocationTarget::Fixed(ref loc) = known {
                    oracle_resolved.remove(loc);
                }
            }

            match &**blk.value().last().value() {
                Stmt::CBranch(_, t) => match t {
                    BranchTarget::Location(ref location) => {
                        let tgt = &mapping.lookup_by_location::<Option<_>>(location).expect("block exists");
                        if self.block_ids.contains_key(&tgt.id()) {
                            let fall = blk.next_block_entities::<_, Option<_>>(mapping).unwrap();
                            cfg.add_cond(blk, tgt, fall);
                        }
                    },
                    BranchTarget::Computed(_) => {
                        for location in oracle_resolved.into_iter() {
                            let tgt = &mapping.lookup_by_location::<Option<_>>(&location).expect("block exists");
                            if self.block_ids.contains_key(&tgt.id()) {
                                cfg.add_jump(blk, tgt);
                            }
                        }
                        let fall = blk.next_block_entities::<_, Option<_>>(mapping).unwrap();
                        cfg.add_fall(blk, fall);
                    },
                }
                Stmt::Branch(t) => match t {
                    BranchTarget::Location(ref location) => {
                        let tgt = &mapping.lookup_by_location::<Option<_>>(location).expect("block exists");
                        if self.block_ids.contains_key(&tgt.id()) {
                            cfg.add_jump(blk, tgt);
                        }
                    },
                    BranchTarget::Computed(_) => {
                        for location in oracle_resolved.into_iter() {
                            let tgt = &mapping.lookup_by_location::<Option<_>>(&location).expect("block exists");
                            if self.block_ids.contains_key(&tgt.id()) {
                                cfg.add_jump(blk, tgt);
                            }
                        }
                    },
                }
                _ => (),
            }

            let blkx = cfg.entity_vertex(*blkid).unwrap();
            blk.next_block_entities_with(mapping, &mut succs);
            for tgt in succs.drain(..) {
                if let Some(vx) = cfg.entity_vertex(tgt.id()) {
                    if !cfg.contains_edge(blkx, vx)
                        && !blk
                            .operations()
                            .last()
                            .map(|op| op.is_return())
                            .unwrap_or(false)
                    {
                        cfg.add_vertex_relation(blkx, vx, BranchKind::Fall);
                    }
                }
            }
        }

        cfg
    }
}

pub struct FunctionBuilder<'trans> {
    translator: &'trans Translator,
    context_db: &'trans mut ContextDatabase,
    symbol: String,
    address: u64,
    block_indices: Vec<usize>,
    blocks: Vec<Entity<Block>>,
}

impl<'trans> FunctionBuilder<'trans> {
    pub fn new(translator: &'trans Translator, context_db: &'trans mut ContextDatabase, address: u64, symbol: impl Into<String>) -> Self {
        Self {
            translator,
            context_db,
            symbol: symbol.into(),
            address,
            block_indices: Vec::new(),
            blocks: Vec::new(),
        }
    }

    pub fn add_block(&mut self, address: u64, bytes: &[u8]) -> Result<usize, Error> {
        self.add_block_with(address, bytes, |_| ())
    }

    pub fn add_block_with<F>(&mut self, address: u64, bytes: &[u8], transform: F) -> Result<usize, Error>
    where F: FnMut(&mut ECode) {
        let id = self.block_indices.len();
        let rid = self.blocks.len();
        self.blocks.extend(Block::new_with(&self.translator, &mut self.context_db, address, bytes, transform)?);
        self.block_indices.push(rid);
        Ok(id)
    }

    pub fn block(&self, id: usize) -> Option<&[Entity<Block>]> {
        let sid = self.block_indices.get(id)?;
        if let Some(eid) = self.block_indices.get(id + 1) {
            Some(&self.blocks[*sid..*eid])
        } else {
            Some(&self.blocks[*sid..])
        }
    }

    pub fn block_mut(&mut self, id: usize) -> Option<&mut [Entity<Block>]> {
        let sid = self.block_indices.get(id)?;
        if let Some(eid) = self.block_indices.get(id + 1) {
            Some(&mut self.blocks[*sid..*eid])
        } else {
            Some(&mut self.blocks[*sid..])
        }
    }

    pub fn build(self) -> (Entity<Function>, Vec<Entity<Block>>) {
        let id = LocatableId::new("fcn", Location::new(self.translator.address(self.address), 0));
        let mut block_ids = HashMap::default();
        
        for blk in self.blocks.iter() {
            let (id, loc) = LocatableId::from(blk).into_parts();
            block_ids.insert(id, loc);
        }
        
        let f = Function {
            id,
            symbol: self.symbol,
            block_ids,
            callers: HashMap::default(),
        };
        let blocks = self.blocks;
        
        (f.into(), blocks)
    }
}

pub struct FunctionLifter<'trans> {
    translator: &'trans Translator,
    database: &'trans db::Database,
    block_lifter: BlockLifter<'trans>,
}

impl<'trans> FunctionLifter<'trans> {
    pub fn new(translator: &'trans Translator, database: &'trans db::Database) -> Self {
        Self {
            translator,
            database,
            block_lifter: BlockLifter::new(translator),
        }
    }

    pub fn from_function(
        &mut self,
        f: &db::Function,
    ) -> Result<(Entity<Function>, Vec<Entity<Block>>), Error> {
        self.from_function_with(f, |_| ())
    }

    pub fn from_function_with<F>(
        &mut self,
        f: &db::Function,
        mut transform: F,
    ) -> Result<(Entity<Function>, Vec<Entity<Block>>), Error>
    where F: FnMut(&mut ECode) {
        let id = LocatableId::new("fcn", Location::new(self.translator.address(f.address()), 0));
        let mut blocks = Vec::new();
        let mut function = Function {
            id,
            symbol: f.name().to_string(),
            block_ids: HashMap::new(),
            callers: HashMap::new(),
        };

        for b in f.blocks() {
            let blks = self.block_lifter.from_block_with(b, &mut transform)?;

            blocks.reserve(blks.len());
            function.block_ids.reserve(blks.len());

            for sb in blks.iter() {
                let (id, loc) = LocatableId::from(sb).into_parts();
                function.block_ids.insert(id, loc);
            }

            blocks.extend(blks.into_iter());
        }

        for r in f
            .references()
            .iter()
            .filter(|r| r.is_call() && !r.source_id().is_invalid())
        {
            if let Some(sfcn) = self.database.functions().get(r.source_id().index()) {
                let fid = LocatableId::invalid(
                    "fcn",
                    Location::new(self.translator.address(sfcn.address()), 0),
                );
                let sid = LocatableId::invalid(
                    "stmt",
                    Location::new(self.translator.address(r.address()), 0),
                );
                function.callers.insert(fid, sid);
            }
        }

        Ok((function.into(), blocks))
    }
}