use fugue::db::Database;

use fugue::ir::Translator;
use fugue::ir::address::IntoAddress;
use fugue::ir::space::{AddressSpace, SpaceKind};
use fugue::ir::il::ecode::{BranchTarget, EntityId, Entity, Location, Stmt};

use std::collections::HashMap;
use std::sync::Arc;

use crate::models::Block;
use crate::models::CFG;
use crate::models::{Function, FunctionLifter};
use crate::types::EntityMap;

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
    stack_space: Arc<AddressSpace>,
    functions: EntityMap<Function>,
    functions_by_location: HashMap<Location, EntityId>,
    functions_by_symbol: HashMap<String, EntityId>,
    blocks: EntityMap<Block>,
}

impl<'db> Program<'db> {
    pub fn new(database: &'db Database) -> Result<Self, Error> {
        let mut trans = database.default_translator();

        let stack_space = if let Some(stack) = trans.manager().space_by_name("stack") {
            stack
        } else {
            let address_size = trans.manager().address_size();
            trans.manager_mut().add_space(
                SpaceKind::Processor,
                "stack",
                address_size,
                1, // word size
                None, // properties
                0, // delay
            )
        };

        let mut function_lifter = FunctionLifter::new(&trans);

        let mut functions = EntityMap::default();
        let mut blocks = EntityMap::default();

        let mut functions_by_symbol = HashMap::default();
        let mut functions_by_location = HashMap::default();

        for f in database.functions() {
            let (fcn, blks) = function_lifter.from_function(f)?;

            let id = fcn.id().clone();
            let location = fcn.value().location().to_owned();
            let symbol = fcn.value().symbol().to_owned();

            functions.insert(id.clone(), fcn);
            functions_by_symbol.insert(symbol, id.clone());
            functions_by_location.insert(location, id);

            blocks.reserve(blks.len());
            blocks.extend(blks.into_iter().map(|blk| (blk.id().clone(), blk)));
        }

        Ok(Program {
            database,
            translator: trans,
            stack_space,
            functions,
            functions_by_symbol,
            functions_by_location,
            blocks,
        })
    }

    pub fn blocks(&self) -> &EntityMap<Block> {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut EntityMap<Block> {
        &mut self.blocks
    }

    pub fn function_by_address<A: IntoAddress>(&self, address: A) -> Option<&Entity<Function>> {
        let address = address.into_address_value(self.translator.manager().default_space());
        let location = Location::new(address, 0);

        self.functions_by_location.get(&location)
            .and_then(|id| self.functions.get(id))
    }

    pub fn function_by_symbol<S: AsRef<str>>(&self, symbol: S) -> Option<&Entity<Function>> {
        self.functions_by_symbol.get(symbol.as_ref())
            .and_then(|id| self.functions.get(id))
    }

    pub fn functions(&self) -> &EntityMap<Function> {
        &self.functions
    }

    pub fn functions_mut(&mut self) -> &mut EntityMap<Function> {
        &mut self.functions
    }

    pub fn icfg(&self) -> CFG {
        let mut icfg = CFG::new();

        // add all blocks
        for blk in self.blocks.values() {
            if self.functions_by_location.contains_key(blk.location()) {
                icfg.add_entry(blk);
            } else {
                icfg.add_block(blk);
            }
        }

        // add all resolvable jumps/calls
        for blk in self.blocks.values() {
            match blk.value().last().value() {
                Stmt::Call(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];
                    icfg.add_call(blk, tgt);
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
