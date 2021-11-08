/// This analysis normalises the ECode IR to handle implicitly
/// aliased variables which occur as artefacts from the PCode
/// lifting process.
///
/// NOTE: we assume that the graph is not in SSA form.
///
/// We perform the analysis in four stages:
///
/// 1) We extract all variables defined and used within the
///    entity graph; we use them to build an interval set for
///    each respective address space (e.g., registers or
///    temporaries). We can then check if two variables belong to
///    the same equivalence class by checking if they share the
///    same "parent" variable. This allows use to determine AH
///    and AL are in the same class by noting that their common
///    parent is AX, EAX, or RAX.
///
/// 2) We perform reaching definitions analysis over the CFG,
///    where we assume that a definition is killed by a
///    definition of any variable sharing the same equivalence
///    class. When a variable is killed in this way (i.e., it is
///    killed by a partial overlap), we construct a pseudo definition
///    of that variable with respect to its assumed old value and the
///    aliased variable.
///
/// 3) We perform a live variables analysis using our pseudo
///    definitions from step 2, and use it to prune our pseudo
///    definitions.
///
/// 4) We convert live pseudo definitions into real definitions.
///

use std::borrow::Cow;
use std::convert::Infallible;
use std::collections::{BTreeMap, BTreeSet};
use std::iter::FromIterator;

use fugue::ir::il::ecode::EntityId;

use crate::analyses::fixed_point::FixedPointForward;
use crate::models::{Block, CFG};
use crate::models::cfg::BranchKind;
use crate::types::{SimpleVar, VarViews};
use crate::traits::*;

trait VarClass<'a> {
    fn class_equivalent<V: Into<SimpleVar<'a>>>(&'a self, other: V, classes: &VarViews) -> bool;
}

impl<'a, T> VarClass<'a> for T where T: 'a, &'a T: Into<SimpleVar<'a>> {
    fn class_equivalent<V: Into<SimpleVar<'a>>>(&'a self, other: V, classes: &VarViews) -> bool {
        let sc = classes.enclosing(self);
        let oc = classes.enclosing(other);
        sc == oc
    }
}

/// The goal of this analysis is to minimise the number of inserted
/// definitions. Therefore, this function only considers aliases with
/// respect to the CFG being transformed.
fn extract_variable_aliases(graph: &CFG<Block>) -> VarViews {
    let mut defs = BTreeSet::new();
    let mut uses = BTreeSet::new();

    for (_, _, block) in graph.entities() {
        block.defined_and_used_variables_with(&mut defs, &mut uses);
    }

    VarViews::from_iter(defs.union(&uses).map(|v| *v))
}

pub type DefinitionMap<'ecode> = BTreeMap<SimpleVar<'ecode>, BTreeSet<Cow<'ecode, EntityId>>>;

#[derive(Default)]
pub struct ReachingDefinitions;

impl<'ecode> FixedPointForward<'ecode, Block, BranchKind, CFG<'ecode, Block>, DefinitionMap<'ecode>> for ReachingDefinitions {
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

        // Sanity: CFG is not in SSA form!
        debug_assert!(block.phis().is_empty());

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
