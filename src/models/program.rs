use fugue::db::Database;

use fugue::ir::address::IntoAddress;
use fugue::ir::il::ecode::{BranchTarget, Location, Stmt};
use fugue::ir::space::{AddressSpace, SpaceKind};
use fugue::ir::Translator;

use std::borrow::{Borrow, Cow};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::models::Block;
use crate::models::{Function, FunctionLifter};
use crate::models::{CFG, CG};
use crate::transforms::normalise::{NormaliseVariables, VariableNormaliser};
use crate::traits::collect::EntityRefCollector;
use crate::types::{Id, Identifiable, Locatable, EntityIdMapping, EntityLocMapping, EntityMap, EntityRef, IntoEntityRef};

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
    functions_by_location: HashMap<Location, BTreeSet<Id<Function>>>,
    functions_by_symbol: HashMap<String, Id<Function>>,
    blocks: EntityMap<Block>,
    blocks_by_location: HashMap<Location, BTreeSet<Id<Block>>>,
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

        let mut blocks = EntityMap::default();
        let mut blocks_by_location = HashMap::<Location, BTreeSet<_>>::default();

        let mut functions = EntityMap::default();
        let mut functions_by_symbol = HashMap::default();
        let mut functions_by_location = HashMap::<Location, BTreeSet<_>>::default();

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
            functions_by_location.entry(location)
                .or_default()
                .insert(id);

            blocks.reserve(blks.len());
            blocks_by_location.reserve(blks.len());

            for blk in blks.into_iter() {
                let id = blk.id();
                let location = blk.location();

                blocks.insert(id, blk);
                blocks_by_location.entry(location)
                    .or_default()
                    .insert(id);
            }

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
            blocks_by_location,
        })
    }

    pub fn translator(&self) -> &Translator {
        &self.translator
    }

    pub fn function_by_address<'a, C: EntityRefCollector<'a, Function>>(&'a self, address: impl IntoAddress) -> C {
        let address = address.into_address_value(self.translator.manager().default_space_ref());
        let location = Location::new(address, 0);

        self.lookup_by_location::<C>(&location)
    }

    pub fn function_by_symbol<S: AsRef<str>>(&self, symbol: S) -> Option<EntityRef<Function>> {
        self.functions_by_symbol
            .get(symbol.as_ref())
            .and_then(|id| self.functions.get(id))
            .map(|e| e.into_entity_ref())
    }

    pub fn cg(&self) -> CG<Stmt> {
        let mut cg = CG::new();

        // add all functions
        for fcn in self.functions.values() {
            cg.add_function(fcn);
        }

        // add all calls
        for fcn in self.functions.values() {
            for (caller, stmt) in fcn.callers().iter().filter_map(|(lid, sid)| self.functions.get(&lid.id()).map(|f| (f, sid))) {
                cg.add_call_via(caller, fcn, stmt.id());
            }
        }

        cg
    }

    pub fn icfg(&self) -> CFG<Block> {
        let mut icfg = CFG::new();

        // add all blocks
        for blk in self.blocks.values() {
            if self.functions_by_location.contains_key(&blk.location()) {
                icfg.add_root_entity(blk);
            } else {
                icfg.add_entity(blk);
            }
        }

        // add all resolvable jumps/calls
        for blk in self.blocks.values() {
            match &**blk.value().last().value() {
                Stmt::Call(BranchTarget::Location(location), _) => {
                    if let Some(tgt) = self.lookup_by_location::<Option<_>>(location) {
                        icfg.add_call(blk, tgt);
                    }
                }
                Stmt::CBranch(_, BranchTarget::Location(location)) => {
                    let tgt = self.lookup_by_location::<Option<_>>(location);
                    if tgt.is_none() {
                        continue;
                    }

                    let tgt = tgt.unwrap();
                    let fall = blk.next_block_entities::<_, Option<_>>(self).unwrap();

                    icfg.add_cond(blk, tgt, fall);
                }
                Stmt::Branch(BranchTarget::Location(location)) => {
                    if let Some(tgt) = self.lookup_by_location::<Option<_>>(location) {
                        icfg.add_jump(blk, tgt);
                    }
                }
                _ => (),
            }
        }
        icfg
    }
}

impl<'db> EntityIdMapping<Block> for Program<'db> {
    fn lookup_by_id(&self, id: Id<Block>) -> Option<EntityRef<Block>> {
        self.blocks.get(&id).map(Cow::Borrowed)
    }
}

impl<'db> EntityLocMapping<Block> for Program<'db> {
    fn lookup_by_location_with<'a, C: EntityRefCollector<'a, Block>>(&'a self, loc: &Location, collect: &mut C) {
        if let Some(ids) = self.blocks_by_location.get(loc) {
            for id in ids.iter() {
                if let Some(e) = self.lookup_by_id(*id) {
                    collect.insert(e);
                }
            }
        }
    }
}

impl<'db> EntityIdMapping<Function> for Program<'db> {
    fn lookup_by_id(&self, id: Id<Function>) -> Option<EntityRef<Function>> {
        self.functions.get(&id).map(Cow::Borrowed)
    }
}

impl<'db> EntityLocMapping<Function> for Program<'db> {
    fn lookup_by_location_with<'a, C: EntityRefCollector<'a, Function>>(&'a self, loc: &Location, collect: &mut C) {
        if let Some(ids) = self.functions_by_location.get(loc) {
            for id in ids.iter() {
                if let Some(e) = self.lookup_by_id(*id) {
                    collect.insert(e);
                }
            }
        }
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
