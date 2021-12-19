use crate::analyses::fixed_point::FixedPointForward;
use crate::models::{Block, Phi};
use crate::traits::*;
use crate::types::{Id, Identifiable, Located, Locatable, LocatableId, SimpleVar};
use crate::graphs::entity::AsEntityGraph;

use fugue::ir::il::ecode::{Stmt, Var};
use fugue::ir::il::traits::Variable;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::convert::Infallible;

use fixedbitset::{FixedBitSet, Ones};
type IndexMap<K, V> = indexmap::IndexMap<K, V, fxhash::FxBuildHasher>;

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

pub type DefinitionMap<'ecode> = BTreeMap<SimpleVar<'ecode>, FixedBitSet>;

pub struct ReachingDefinitions {
    stmt_defs: BTreeMap<Id<Located<Stmt>>, (FixedBitSet, FixedBitSet)>,
    variables: IndexMap<Var, Option<PhiOrStmt>>,
    free_variables: FixedBitSet,
}

impl ReachingDefinitions {
    pub fn new<'a, E: 'a, G>(g: &'a G) -> Self
    where G: AsEntityGraph<'a, Block, E> {
        let mut variables = IndexMap::<Var, Option<PhiOrStmt>>::default();
        let mut ldefs = HashSet::default();
        let mut luses = HashSet::default();

        for (_, _, blk) in g.entity_graph().entities() {
            for phi in blk.phis() {
                let lid = LocatableId::from_parts(phi.id().retype::<Phi>(), phi.location());
                variables.insert(*phi.var(), Some(lid.into()));

                for var in phi.assign() {
                    if var.generation() == 0 {
                        variables.insert(*var, None);
                    }
                }
            }

            for op in blk.operations() {
                let lid = PhiOrStmt::from(LocatableId::from_parts(op.id().retype::<Stmt>(), op.location()));
                op.defined_and_used_variables_with(&mut ldefs, &mut luses);

                for d in ldefs.drain() {
                    variables.insert(*d, Some(lid.clone()));
                }

                for u in luses.drain() {
                    if u.generation() == 0 {
                        variables.insert(*u, None);
                    }
                }
            }
        }

        let mut free_variables = FixedBitSet::with_capacity(variables.len());
        for (idx, (_, v)) in variables.iter().enumerate() {
            if v.is_none() {
                free_variables.put(idx);
            }
        }

        let mut slf = Self {
            stmt_defs: BTreeMap::new(),
            variables,
            free_variables,
        };

        slf.analyse::<BTreeMap<_, _>>(g).unwrap();

        slf
    }

    pub fn get(&self, var: &Var) -> Option<(usize, &Option<PhiOrStmt>)> {
        self.variables.get_full(var).map(|(i, _, v)| (i, v))
    }

    pub fn get_index(&self, var: usize) -> Option<(&Var, &Option<PhiOrStmt>)> {
        self.variables.get_index(var)
    }

    pub fn get_index_of(&self, var: &Var) -> Option<usize> {
        self.variables.get_index_of(var)
    }

    pub fn reaches<S>(&self, stmt: &S, var: &Var) -> bool
    where S: Identifiable<Located<Stmt>> {
        let eid = stmt.id();
        self.stmt_defs.get(&eid)
            .and_then(|(iset, _)| {
                self.variables.get_index_of(var)
                    .map(|vid| iset.contains(vid))
            })
            .unwrap_or(false)
    }

    pub fn all_reaching<S>(&self, stmt: &S) -> Option<&FixedBitSet>
    where S: Identifiable<Located<Stmt>> {
        let eid = stmt.id();
        self.stmt_defs.get(&eid).map(|(iset, _)| iset)
    }

    pub fn all_reaching_vars<'a, S>(&'a self, stmt: &S) -> Option<ReachingDefinitionIter<'a>>
    where S: Identifiable<Located<Stmt>> {
        let eid = stmt.id();
        let bits = &self.stmt_defs.get(&eid)?.0;
        Some(ReachingDefinitionIter {
            iter: bits.ones(),
            defs: self,
        })
    }

    fn empty(&self) -> FixedBitSet {
        FixedBitSet::with_capacity(self.variables.len())
    }

    fn singleton(&self, var: &Var) -> FixedBitSet {
        let mut s = self.empty();
        s.put(self.var_id(var));
        s
    }

    fn var_id(&self, var: &Var) -> usize {
        self.variables.get_index_of(var).unwrap()
    }
}

pub struct ReachingDefinitionIter<'a> {
    iter: Ones<'a>,
    defs: &'a ReachingDefinitions,
}

impl<'a> Iterator for ReachingDefinitionIter<'a> {
    type Item = (&'a Var, &'a Option<PhiOrStmt>);

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
            .and_then(|i| self.defs.variables.get_index(i))
    }
}

impl<'ecode, E, G> FixedPointForward<'ecode, Block, E, G, DefinitionMap<'ecode>> for ReachingDefinitions
where E: 'ecode,
      G: AsEntityGraph<'ecode, Block, E> {
    type Err = Infallible;

    fn join(&mut self, mut current: DefinitionMap<'ecode>, prev: &DefinitionMap<'ecode>) -> Result<DefinitionMap<'ecode>, Self::Err> {
        for (v, defs) in prev.iter() {
            current.entry(v.clone())
                .and_modify(|cdefs| cdefs.union_with(defs))
                .or_insert_with(|| defs.clone());
        }
        Ok(current)
    }

    fn transfer(&mut self, block: &'ecode Block, current: Option<DefinitionMap<'ecode>>) -> Result<DefinitionMap<'ecode>, Self::Err> {
        let mut gen = current.unwrap_or_default();
        let mut lgen = BTreeSet::new();

        for phi in block.phis() {
            let var = SimpleVar::from(phi.var());
            gen.insert(var, self.singleton(phi.var()));
        }

        let mut bits = if gen.is_empty() {
            for fv in self.free_variables.ones() {
                let v = *self.variables.get_index(fv).unwrap().0;
                let mut fv_bits = self.empty();
                fv_bits.put(fv);
                gen.insert(SimpleVar::from(v), fv_bits);
            }
            self.free_variables.clone()
        } else {
            let mut bits = self.empty();
            for set in gen.values() {
                bits.union_with(set);
            }
            bits
        };

        for op in block.operations() {
            op.defined_variables_with(&mut lgen);

            let last = bits.clone();

            for d in lgen.iter() {
                let var = SimpleVar::from(*d);
                match gen.entry(var) {
                    Entry::Vacant(e) => {
                        // if gen and no reaching, then assume that if FV exists, then it reaches, so kill
                        if let Some(idx) = self.variables.get_index_of(&d.with_generation(0)) {
                            bits.set(idx, false);
                        }
                        e.insert(self.singleton(d));
                    },
                    Entry::Occupied(mut e) => {
                        let v = e.get_mut();
                        bits.difference_with(v);
                        *v = self.singleton(d);
                    }
                }
                bits.put(self.var_id(d));
            }

            self.stmt_defs.insert(op.id(), (last, bits.clone()));
        }

        Ok(gen)
    }
}
