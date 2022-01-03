use std::fmt::Debug;
use std::marker::PhantomData;

use ena::unify::{InPlaceUnificationTable, NoError};
use ena::unify::{UnifyKey, UnifyValue};

use fugue::ir::il::ecode::Stmt;
use fugue::ir::il::ecode::Var;

use crate::models::{Block, CFG};

type IndexSet<K> = indexmap::IndexSet<K, fxhash::FxBuildHasher>;

#[derive(educe::Educe)]
#[educe(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct VarT<V>(u32, PhantomData<V>);

impl<V> UnifyKey for VarT<V> where V: UnifyValue {
    type Value = V;

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(u: u32) -> Self {
        Self(u, PhantomData)
    }

    fn tag() -> &'static str {
        "var"
    }
}

#[derive(educe::Educe)]
#[educe(Clone, Debug, Default)]
pub struct VClassMap<V> where V: UnifyValue {
    mapping: IndexSet<Var>,
    classes: InPlaceUnificationTable<VarT<V>>,
}

impl<V> From<&'_ CFG<'_, Block>> for VClassMap<V> where V: UnifyValue {
    fn from(g: &'_ CFG<Block>) -> Self {
        let mut mapping = IndexSet::default();

        for (_, _, blk) in g.entities() {
            for phi in blk.phis() {
                mapping.insert(*phi.var());
            }

            for op in blk.operations() {
                if let Stmt::Assign(var, _) = &***op {
                    mapping.insert(*var);
                }
            }
        }

        Self {
            mapping,
            classes: Default::default(),
        }
    }
}

/*
       Top
  ... .   ...
 /   /       \
v0 v1  ...   vN
 \  \        /
  '' '     ''
       Bot
*/
#[derive(Clone, Debug)]
pub enum VLattice<V> {
    Top,
    Val(V),
    Bot,
}

impl<V> UnifyValue for VLattice<V> where V: Clone + Debug + Eq {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        Ok(match (value1, value2) {
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Bot, v) | (v, Self::Bot) => v.clone(),
            (Self::Val(u), Self::Val(v)) => if u == v {
                Self::Val(u.clone())
            } else {
                Self::Top
            },
        })
    }
}

impl<V> VClassMap<V> where V: UnifyValue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn unify_vars(&mut self, var1: &Var, var2: &Var) -> Result<(), <V as UnifyValue>::Error> {
        let v1_id = self.mapping.get_index_of(var1);
        let v2_id = self.mapping.get_index_of(var2);

        if let Some((v1, v2)) = v1_id.and_then(|v1| v2_id.map(|v2| (v1, v2))) {
            self.classes.unify_var_var(
                VarT::<V>::from_index(v1 as u32),
                VarT::<V>::from_index(v2 as u32),
            )
        } else {
            Ok(())
        }
    }

    pub fn unify_val(&mut self, var: &Var, val: V) -> Result<(), <V as UnifyValue>::Error> {
        if let Some(v_id) = self.mapping.get_index_of(var) {
            self.classes.unify_var_value(VarT::<V>::from_index(v_id as u32), val)
        } else {
            Ok(())
        }
    }
}
