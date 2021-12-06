use crate::analyses::fixed_point::FixedPointForward;
use crate::models::{Block, Phi};
use crate::traits::*;
use crate::types::{Identifiable, Locatable, LocatableId, SimpleVar};
use crate::graphs::entity::AsEntityGraph;

use fugue::ir::il::ecode::Stmt;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhiOrStmt {
    Phi(LocatableId<Phi>),
    Stmt(LocatableId<Stmt>),
}

impl From<LocatableId<Phi>> for PhiOrStmt {
    fn from(lid: LocatableId<Phi>) -> Self {
        Self::Phi(lid)
    }
}

impl From<LocatableId<Stmt>> for PhiOrStmt {
    fn from(lid: LocatableId<Stmt>) -> Self {
        Self::Stmt(lid)
    }
}

pub type DefinitionMap<'ecode> = BTreeMap<SimpleVar<'ecode>, BTreeSet<PhiOrStmt>>;

#[derive(Default)]
pub struct ReachingDefinitions;

// We could use a fixedbitset using a mapping from entity ids -> indices
// to avoid costly set comparions/merges.
//
// We can compute this by walking over G in a single forward pass and
// inserting each EntityId into an IndexSet:
//
// indexmap::set::IndexSet
//
// We can then build a FixedBitSet of size IndexSet::len and use the
// insertion ordering of entity ids (to get an index) in the IndexSet
// to determine an entity's membership in the FixedBitSet.

impl<'ecode, E, G> FixedPointForward<'ecode, Block, E, G, DefinitionMap<'ecode>> for ReachingDefinitions
where E: 'ecode,
      G: AsEntityGraph<'ecode, Block, E> {
    type Err = Infallible;

    fn join(&mut self, mut current: DefinitionMap<'ecode>, prev: &DefinitionMap<'ecode>) -> Result<DefinitionMap<'ecode>, Self::Err> {
        for (v, defs) in prev.iter() {
            current.entry(v.clone())
                .and_modify(|cdefs| cdefs.extend(defs.iter().cloned()))
                .or_insert_with(|| defs.clone());
        }
        Ok(current)
    }

    fn transfer(&mut self, block: &'ecode Block, current: Option<DefinitionMap<'ecode>>) -> Result<DefinitionMap<'ecode>, Self::Err> {
        let mut gen = BTreeMap::new();
        let mut lgen = BTreeSet::new();

        for phi in block.phis() {
            let mut eids = BTreeSet::new();
            eids.insert(LocatableId::from_parts(phi.id().retype::<Phi>(), phi.location()).into());
            gen.insert(SimpleVar::from(phi.var()), eids);
        }

        for op in block.operations() {
            // def -> gen
            op.defined_variables_with(&mut lgen);

            for d in lgen.iter() {
                let mut eids = BTreeSet::new();
                eids.insert(LocatableId::from_parts(op.id().retype::<Stmt>(), op.location()).into());
                gen.insert(SimpleVar::from(*d), eids);
            }
        }

        Ok(if let Some(mut out) = current {
            for (v, defs) in gen.into_iter() {
                out.insert(v, defs);
            }
            out
        } else {
            gen
        })
    }
}
