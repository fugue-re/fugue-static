use fugue::db::Database;

use fugue::ir::address::IntoAddress;
use fugue::ir::il::ecode::{BranchTarget, Entity, EntityId, Location, Stmt};
use fugue::ir::space::{AddressSpace, SpaceKind};
use fugue::ir::Translator;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::Block;
use crate::models::{Function, FunctionLifter};
use crate::models::{CFG, CG};
use crate::transforms::normalise::{NormaliseVariables, VariableNormaliser};
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
                1,    // word size
                None, // properties
                0,    // delay
            )
        };

        let mut function_lifter = FunctionLifter::new(&trans, database);

        let mut functions = EntityMap::default();
        let mut blocks = EntityMap::default();

        let mut functions_by_symbol = HashMap::default();
        let mut functions_by_location = HashMap::default();

        let mut normaliser = VariableNormaliser::new(&trans);

        for f in database.functions() {
            let (fcn, blks) = function_lifter.from_function_with(f, |ecode| {
                ecode.normalise_variables(&mut normaliser)
            })?;

            let id = fcn.id().clone();
            let location = fcn.value().location().to_owned();
            let symbol = fcn.value().symbol().to_owned();

            functions.insert(id.clone(), fcn);
            functions_by_symbol.insert(symbol, id.clone());
            functions_by_location.insert(location, id);

            blocks.reserve(blks.len());
            blocks.extend(blks.into_iter().map(|blk| (blk.id().clone(), blk)));

            normaliser.reset();
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

    pub fn translator(&self) -> &Translator {
        &self.translator
    }

    pub fn blocks(&self) -> &EntityMap<Block> {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut EntityMap<Block> {
        &mut self.blocks
    }

    pub fn function_by_address<A: IntoAddress>(&self, address: A) -> Option<&Entity<Function>> {
        let address = address.into_address_value(self.translator.manager().default_space_ref());
        let location = Location::new(address, 0);

        self.functions_by_location
            .get(&location)
            .and_then(|id| self.functions.get(id))
    }

    pub fn function_by_symbol<S: AsRef<str>>(&self, symbol: S) -> Option<&Entity<Function>> {
        self.functions_by_symbol
            .get(symbol.as_ref())
            .and_then(|id| self.functions.get(id))
    }

    pub fn functions(&self) -> &EntityMap<Function> {
        &self.functions
    }

    pub fn functions_mut(&mut self) -> &mut EntityMap<Function> {
        &mut self.functions
    }

    pub fn cg(&self) -> CG {
        let mut cg = CG::new();

        // add all functions
        for fcn in self.functions.values() {
            cg.add_function(fcn);
        }

        // add all calls
        for fcn in self.functions.values() {
            for (caller, stmt) in fcn.callers().iter().filter_map(|(id, sid)| self.functions.get(id).map(|f| (f, sid))) {
                cg.add_call_via(caller, fcn, stmt.clone());
            }
        }

        cg
    }

    pub fn icfg(&self) -> CFG<Block> {
        let mut icfg = CFG::new();

        // add all blocks
        for blk in self.blocks.values() {
            if self.functions_by_location.contains_key(blk.location()) {
                icfg.add_root_entity(blk);
            } else {
                icfg.add_entity(blk);
            }
        }

        // add all resolvable jumps/calls
        for blk in self.blocks.values() {
            match blk.value().last().value() {
                Stmt::Call(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];
                    icfg.add_call(blk, tgt);
                }
                Stmt::CBranch(_, BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];

                    let fall_id = blk.next_blocks().next().unwrap();
                    let fall = &self.blocks[&fall_id];

                    icfg.add_cond(blk, tgt, fall);
                }
                Stmt::Branch(BranchTarget::Location(location)) => {
                    let tgt_id = EntityId::new("blk", location.clone());
                    let tgt = &self.blocks[&tgt_id];
                    icfg.add_jump(blk, tgt);
                }
                _ => (),
            }
        }
        icfg
    }
}

impl<'db> Borrow<Translator> for Program<'db> {
    fn borrow(&self) -> &Translator {
        self.translator()
    }
}

impl<'db> Borrow<Translator> for &'_ Program<'db> {
    fn borrow(&self) -> &Translator {
        self.translator()
    }
}
