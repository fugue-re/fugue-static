use crate::analyses::fixed_point::FixedPointForward;
use crate::models::block::Block;
use crate::traits::*;
use crate::transforms::ssa::simple_var::SimpleVar;
use crate::graphs::entity::AsEntityGraph;

use fugue::ir::il::ecode::EntityId;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;

pub type DefinitionMap<'ecode> = BTreeMap<SimpleVar<'ecode>, BTreeSet<Cow<'ecode, EntityId>>>;

#[derive(Default)]
pub struct ReachingDefinitions;

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

        for (lvar, _rvars) in block.phis() {
            let mut eids = BTreeSet::new();
            eids.insert(Cow::Owned(EntityId::new("phi", block.location().clone())));
            gen.insert(SimpleVar::from(lvar), eids);
        }

        for op in block.operations() {
            // def -> gen
            op.defined_variables_with(&mut lgen);

            for d in lgen.iter() {
                let mut eids = BTreeSet::new();
                eids.insert(Cow::Borrowed(op.id()));
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
