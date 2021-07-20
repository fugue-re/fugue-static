use fugue::db::Database;

use fugue::ir::Translator;
use fugue::ir::il::ecode::{BranchTarget, EntityId, Entity, Stmt};

use std::collections::HashMap;

use crate::models::Block;
use crate::models::{Function, FunctionLifter};
use crate::models::ICFG;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Lifting(#[from] crate::models::function::Error),
}

#[derive(Clone)]
pub struct Program<'db> {
    database: &'db Database,
    translator: Translator,
    functions: HashMap<EntityId, Entity<Function>>,
    blocks: HashMap<EntityId, Entity<Block>>,
}

impl<'db> Program<'db> {
    pub fn new(database: &'db Database) -> Result<Self, Error> {
        let trans = database.default_translator();
        let mut function_lifter = FunctionLifter::new(&trans);

        let mut functions = HashMap::new();
        let mut blocks = HashMap::new();

        for f in database.functions() {
            let (fcn, blks) = function_lifter.from_function(f)?;

            functions.insert(fcn.id().clone(), fcn);
            blocks.reserve(blks.len());
            blocks.extend(blks.into_iter().map(|blk| (blk.id().clone(), blk)));
        }

        Ok(Program {
            database,
            translator: trans,
            functions,
            blocks,
        })
    }

    pub fn icfg(&self) -> ICFG {
        let mut icfg = ICFG::new();

        // add all blocks
        for blk in self.blocks.values() {
            icfg.add_block(blk);
        }

        // add all resolvable jumps/calls
        for blk in self.blocks.values() {
            match blk.value().last().value() {
                Stmt::Call(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];
                    icfg.add_call(blk, tgt, blk.value().last());
                },
                Stmt::CBranch(_, BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];
                    icfg.add_cond(blk, tgt);
                },
                Stmt::Branch(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];
                    icfg.add_jump(blk, tgt);
                },
                _ => (),
            }
        }
        icfg
    }
}
