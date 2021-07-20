use std::collections::HashSet;

use fugue::db::BasicBlock;

use fugue::ir::Translator;
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::{BranchTarget, Entity, EntityId, Location, Stmt};


use thiserror::Error;

use crate::traits::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Lifting(#[from] fugue::ir::error::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Block {
    operations: Vec<Entity<Stmt>>,
    next_block: EntityId,
}

impl Block {
    pub fn location(&self) -> &Location {
        self.first().location()
    }

    pub fn next_block(&self) -> &EntityId {
        &self.next_block
    }

    pub fn next_location(&self) -> &Location {
        self.next_block.location()
    }

    pub fn first(&self) -> &Entity<Stmt> {
        &self.operations[0]
    }

    pub fn first_mut(&mut self) -> &mut Entity<Stmt> {
        &mut self.operations[0]
    }

    pub fn last(&self) -> &Entity<Stmt> {
        &self.operations[self.operations.len()-1]
    }

    pub fn last_mut(&mut self) -> &mut Entity<Stmt> {
        let offset = self.operations.len() - 1;
        &mut self.operations[offset]
    }

    pub fn operations(&self) -> &[Entity<Stmt>] {
        &self.operations
    }

    pub fn operations_mut(&mut self) -> &mut [Entity<Stmt>] {
        &mut self.operations
    }
}

pub struct BlockLifter<'translator> {
    translator: &'translator Translator,
    context: ContextDatabase,
}

impl<'trans> BlockLifter<'trans> {
    pub fn new(translator: &'trans Translator) -> Self {
        Self {
            translator,
            context: translator.context_database(),
        }
    }

    pub fn from_block(&mut self, block: &BasicBlock) -> Result<Vec<Entity<Block>>, Error> {
        let mut offset = 0;
        let mut blocks = Vec::with_capacity(4);

        let block_addr = block.address();
        let bytes = block.bytes();

        let mut targets = HashSet::new();

        while offset < bytes.len() {
            let address = block_addr + offset as u64;
            let mut ecode = self.translator.lift_ecode(
                &mut self.context,
                self.translator.address(address),
                &bytes[offset..])?;

            // Each `ecode` block represents a single architectural
            // instruction, which may have local control-flow.
            //
            // Local control-flow is facilitated by `Stmt::Branch`
            // and `Stmt::CBranch` instructions that have a definite
            // target within the block.
            //
            // For all other control-flow, we split after the operation.
            //
            // We first locate all of the split points, which correspond
            // to "instruction local" branch targets and global branches.
            // Then, we process them in reverse order to partition the
            // vector of operations into blocks.

            let mut local_targets = ecode.operations().iter()
                .enumerate()
                .filter_map(|(offset, operation)| match operation {
                    Stmt::Branch(BranchTarget::Location(location)) |
                    Stmt::CBranch(_, BranchTarget::Location(location)) => {
                        targets.insert(location.clone());
                        if ecode.address() == *location.address() {
                            Some(location.position())
                        } else if offset + 1 < ecode.operations().len() {
                            Some(offset + 1)
                        } else {
                            None
                        }
                    },
                    Stmt::Call(BranchTarget::Location(location)) |
                    Stmt::Return(BranchTarget::Location(location)) => {
                        targets.insert(location.clone());
                        if offset + 1 < ecode.operations().len() {
                            Some(offset + 1)
                        } else {
                            None
                        }
                    },
                    stmt if stmt.is_branch() && offset + 1 < ecode.operations().len() => {
                        Some(offset + 1)
                    }
                    _ => None,
                })
                .filter(|offset| *offset != 0)
                .collect::<Vec<_>>();

            // Sort in descending order
            local_targets.sort_by(|u, v| u.cmp(&v).reverse());

            let address = ecode.address();
            let mut operations = ecode
                .operations
                .drain(..)
                .enumerate()
                .map(|(offset, operation)| {
                    Entity::new("stmt", Location::new(address.clone(), offset), operation)
                })
                .collect::<Vec<_>>();

            let mut local_blocks = Vec::with_capacity(local_targets.len() + 1);

            let mut last_location = Location::new(address.clone() + ecode.length, 0);

            for start in local_targets.into_iter() {
                let block = Block {
                    operations: operations.split_off(start),
                    next_block: EntityId::new("blk", last_location),
                };
                last_location = block.location().clone();
                local_blocks.push(Entity::new("blk", last_location.clone(), block));
            }

            local_blocks.push(Entity::new("blk", Location::new(address.clone(), 0), Block {
                operations: if operations.is_empty() {
                    vec![Entity::new("stmt", Location::new(address, 0), Stmt::skip())]
                } else {
                    operations
                },
                next_block: EntityId::new("blk", last_location),
            }));

            blocks.extend(local_blocks.into_iter().rev());

            offset += ecode.length;
        }

        // Merge blocks that are not targets of other blocks
        for index in (1..blocks.len()).rev() {
            if !blocks[index - 1].value().last().value().is_branch() && !targets.contains(blocks[index].location()) {
                let block = blocks.remove(index).into_value();
                blocks[index - 1].value_mut().operations
                    .extend(block.operations.into_iter());
                blocks[index - 1].value_mut().next_block = block.next_block;
            }
        }

        Ok(blocks)
    }
}
