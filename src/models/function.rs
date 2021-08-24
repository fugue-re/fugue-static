use std::collections::HashSet;

use fugue::db;

use fugue::ir::Translator;
use fugue::ir::il::ecode::{BranchTarget, Entity, EntityId, Location, Stmt};

use crate::models::{Block, BlockLifter, Program};
use crate::models::cfg::{BranchKind, CFG};
use crate::traits::StmtExt;

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

    pub fn cfg<'db>(&self, program: &'db Program) -> CFG<'db> {
        let mut cfg = CFG::new();

        let blks = program.blocks();
        for blkid in self.blocks.iter() {
            let blk = &blks[blkid];
            if blk.location() == self.location() {
                cfg.add_entry(blk);
            } else {
                cfg.add_block(blk);
            }
        }

        for blkid in self.blocks.iter() {
            let blk = &blks[blkid];
            match blk.value().last().value() {
                Stmt::CBranch(_, BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    if self.blocks.contains(&tgt_id) {
                        let tgt = &blks[&tgt_id];
                        cfg.add_cond(blk, tgt);
                    }
                },
                Stmt::Branch(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    if self.blocks.contains(&tgt_id) {
                        let tgt = &blks[&tgt_id];
                        cfg.add_jump(blk, tgt);
                    }
                },
                _ => (),
            }

            let blkx = cfg.block_node(blkid).unwrap();
            for tgt in blk.next_blocks() {
                if let Some(nx) = cfg.block_node(tgt) {
                    if !cfg.contains_edge(blkx, nx) && !blk.operations().last().map(|op| op.is_return()).unwrap_or(false) {
                        cfg.add_edge(blkx, nx, BranchKind::Fall);
                    }
                }
            }
        }
        cfg
    }
}

pub struct FunctionLifter<'trans> {
    translator: &'trans Translator,
    block_lifter: BlockLifter<'trans>,
}

impl<'trans> FunctionLifter<'trans> {
    pub fn new(translator: &'trans Translator) -> Self {
        Self {
            translator,
            block_lifter: BlockLifter::new(translator),
        }
    }

    pub fn from_function(&mut self, f: &db::Function) -> Result<(Entity<Function>, Vec<Entity<Block>>), Error> {
        let mut blocks = Vec::new();
        let mut function = Function {
            symbol: f.name().to_string(),
            location: Location::new(self.translator.address(f.address()), 0),
            blocks: HashSet::new(),
        };

        for b in f.blocks() {
            let blks = self.block_lifter.from_block(b)?;

            blocks.reserve(blks.len());
            function.blocks.reserve(blks.len());

            for sb in blks.iter() {
                function.blocks.insert(sb.id().clone());
            }

            blocks.extend(blks.into_iter());
        }

        let entity = Entity::new("fcn", function.location.clone(), function);
        Ok((entity, blocks))
    }
}
