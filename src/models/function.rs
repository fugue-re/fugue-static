use std::collections::{HashMap, HashSet};

use fugue::db;

use fugue::ir::Translator;
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::{BranchTarget, ECode, Entity, EntityId, Location, Stmt};

use crate::models::cfg::{BranchKind, CFG};
use crate::models::{Block, BlockLifter, BlockMapping};
use crate::traits::StmtExt;
use crate::types::EntityMap;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Lifting(#[from] crate::models::block::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    symbol: String,
    location: Location,
    blocks: HashSet<EntityId>,
    callers: HashMap<EntityId, EntityId>,
}

pub trait FunctionMapping {
    fn functions(&self) -> &EntityMap<Function>;
    fn functions_mut(&mut self) -> &mut EntityMap<Function>;
}

impl FunctionMapping for EntityMap<Function> {
    fn functions(&self) -> &EntityMap<Function> {
        self
    }

    fn functions_mut(&mut self) -> &mut EntityMap<Function> {
        self
    }
}

impl Function {
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn location(&self) -> &Location {
        &self.location
    }

    pub fn blocks(&self) -> &HashSet<EntityId> {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut HashSet<EntityId> {
        &mut self.blocks
    }

    pub fn callers(&self) -> &HashMap<EntityId, EntityId> {
        &self.callers
    }

    pub fn callers_mut(&mut self) -> &mut HashMap<EntityId, EntityId> {
        &mut self.callers
    }

    pub fn cfg<'db, M: BlockMapping>(&self, mapping: &'db M) -> CFG<'db, Block> {
        let mut cfg = CFG::new();

        let blks = mapping.blocks();
        for blkid in self.blocks.iter() {
            let blk = &blks[blkid];
            if blk.location() == self.location() {
                cfg.add_root_entity(blk);
            } else {
                cfg.add_entity(blk);
            }
        }

        for blkid in self.blocks.iter() {
            let blk = &blks[blkid];
            match blk.value().last().value() {
                Stmt::CBranch(_, BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    if self.blocks.contains(&tgt_id) {
                        let tgt = &blks[&tgt_id];
                        let fall_id = blk.value().next_blocks().next().unwrap();
                        let fall = &blks[&fall_id];
                        cfg.add_cond(blk, tgt, fall);
                    }
                }
                Stmt::Branch(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    if self.blocks.contains(&tgt_id) {
                        let tgt = &blks[&tgt_id];
                        cfg.add_jump(blk, tgt);
                    }
                }
                _ => (),
            }

            let blkx = cfg.entity_vertex(blkid).unwrap();
            for tgt in blk.next_blocks() {
                if let Some(vx) = cfg.entity_vertex(tgt) {
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
    context_db: ContextDatabase,
    symbol: String,
    address: u64,
    block_indices: Vec<usize>,
    blocks: Vec<Entity<Block>>,
}

impl<'trans> FunctionBuilder<'trans> {
    pub fn new(translator: &'trans Translator, address: u64, symbol: impl Into<String>) -> Self {
        Self {
            translator,
            context_db: translator.context_database(),
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
        let f = Function {
            symbol: self.symbol,
            location: Location::new(self.translator.address(self.address), 0),
            blocks: self.blocks.iter().map(|blk| blk.id().clone()).collect(),
            callers: HashMap::default(),
        };
        let blocks = self.blocks;

        (Entity::new("fcn", f.location.clone(), f), blocks)
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
        let mut blocks = Vec::new();
        let mut function = Function {
            symbol: f.name().to_string(),
            location: Location::new(self.translator.address(f.address()), 0),
            blocks: HashSet::new(),
            callers: HashMap::new(),
        };

        for b in f.blocks() {
            let blks = self.block_lifter.from_block_with(b, &mut transform)?;

            blocks.reserve(blks.len());
            function.blocks.reserve(blks.len());

            for sb in blks.iter() {
                function.blocks.insert(sb.id().clone());
            }

            blocks.extend(blks.into_iter());
        }

        for r in f
            .references()
            .iter()
            .filter(|r| r.is_call() && !r.source_id().is_invalid())
        {
            if let Some(sfcn) = self.database.functions().get(r.source_id().index()) {
                let fid = EntityId::new(
                    "fcn",
                    Location::new(self.translator.address(sfcn.address()), 0),
                );
                let sid = EntityId::new(
                    "stmt",
                    Location::new(self.translator.address(r.address()), 0),
                );
                function.callers.insert(fid, sid);
            }
        }

        let entity = Entity::new("fcn", function.location.clone(), function);
        Ok((entity, blocks))
    }
}
