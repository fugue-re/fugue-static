use std::collections::HashSet;

use fugue::db;

use fugue::ir::Translator;
use fugue::ir::il::ecode::{Entity, EntityId, Location};

use crate::models::{Block, BlockLifter};

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

    pub(crate) fn from_function(&mut self, f: &db::Function) -> Result<(Entity<Function>, Vec<Entity<Block>>), Error> {
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
