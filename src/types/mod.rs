use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::Deref;

use interval_tree::{Interval, IntervalSet};

use fugue::ir::il::ecode::{Entity, EntityId, Var};
use fugue::ir::space::AddressSpaceId;

use crate::models::Program;

pub type EntityRef<'a, T> = Cow<'a, Entity<T>>;
pub type EntityMap<T> = HashMap<EntityId, Entity<T>>;
pub type EntityRefMap<'a, T> = HashMap<EntityId, EntityRef<'a, T>>;

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct SimpleVar<'a>(Cow<'a, Var>);

impl<'a> SimpleVar<'a> {
    pub(crate) fn owned(var: &Var) -> Self {
        Self(Cow::Owned(var.clone()))
    }

    pub(crate) fn into_owned<'b>(self) -> SimpleVar<'b>
    where
        'a: 'b,
    {
        Self(Cow::Owned(self.0.into_owned()))
    }
}

impl<'a> From<Var> for SimpleVar<'a> {
    fn from(var: Var) -> Self {
        Self(Cow::Owned(var))
    }
}

impl<'a> From<&'a SimpleVar<'a>> for SimpleVar<'a> {
    fn from(var: &'a SimpleVar<'a>) -> Self {
        Self(Cow::Borrowed(&***var))
    }
}

impl<'a> From<&'a Var> for SimpleVar<'a> {
    fn from(var: &'a Var) -> Self {
        Self(Cow::Borrowed(var))
    }
}

impl<'a> From<&'a mut Var> for SimpleVar<'a> {
    fn from(var: &'a mut Var) -> Self {
        Self(Cow::Borrowed(var))
    }
}

impl<'a> Deref for SimpleVar<'a> {
    type Target = Cow<'a, Var>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> PartialEq for SimpleVar<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.space().index() == other.space().index()
            && self.offset() == other.offset()
            && self.bits() == other.bits()
    }
}
impl<'a> Eq for SimpleVar<'a> {}

impl<'a> PartialOrd for SimpleVar<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.space()
            .index()
            .partial_cmp(&other.space().index())
            .and_then(|o| {
                if o.is_ne() {
                    Some(o)
                } else {
                    self.offset().partial_cmp(&other.offset())
                }
            })
            .and_then(|o| {
                if o.is_ne() {
                    Some(o)
                } else {
                    self.bits().partial_cmp(&other.bits())
                }
            })
    }
}
impl<'a> Ord for SimpleVar<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        let sc = self.space().index().cmp(&other.space().index());
        if sc.is_ne() {
            return sc;
        }

        let oc = self.offset().cmp(&other.offset());
        if oc.is_ne() {
            return oc;
        }

        self.bits().cmp(&other.bits())
    }
}

impl<'a> Hash for SimpleVar<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.space().hash(state);
        self.offset().hash(state);
        self.bits().hash(state);
    }
}

#[derive(Debug, Default, Clone)]
pub struct VarViews(BTreeMap<AddressSpaceId, IntervalSet<u64>>);

impl<'a> FromIterator<&'a Var> for VarViews {
    fn from_iter<T: IntoIterator<Item = &'a Var>>(iter: T) -> Self {
        let mut views = Self::default();
        for var in iter.into_iter() {
            views.insert(var)
        }
        views
    }
}

impl VarViews {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn registers(program: &Program) -> Self {
        let t = program.translator();
        let space_id = t.manager().register_space_id();

        let mut m = BTreeMap::new();

        m.insert(
            space_id,
            IntervalSet::from_iter(
                t.registers()
                    .iter()
                    .map(|((off, sz), _)| (Interval::from(off + (off + (*sz as u64) - 1)), ())),
            ),
        );

        Self(m)
    }

    pub fn merge(&mut self, other: VarViews) {
        for (spc, ivss) in other.0.into_iter() {
            let ivsd = self.0.entry(spc).or_default();
            ivsd.extend(ivss.into_iter());
        }
    }

    pub fn insert<'a, V: Into<SimpleVar<'a>>>(&mut self, var: V) {
        let (space, iv) = Self::interval(var);
        self.0.entry(space).or_default().insert(iv, ());
    }

    pub fn replace_overlaps<'a, 'b, V: Into<SimpleVar<'a>>>(
        &mut self,
        var: V,
    ) -> Vec<SimpleVar<'b>> {
        let var = var.into();
        let overlaps = self.take_overlaps(&var);
        self.insert(var);
        overlaps
    }

    pub fn take_overlaps<'a, 'b, V: Into<SimpleVar<'a>>>(&mut self, var: V) -> Vec<SimpleVar<'b>> {
        let var = var.into();
        let (space, iv) = Self::interval(&var);
        if let Some(ivs) = self.0.get_mut(&space) {
            ivs.take_overlaps(iv)
                .into_iter()
                .map(|(iv, _)| {
                    SimpleVar::from(Var::new(
                        space,
                        *iv.start(),
                        (1 + *iv.end() - *iv.start()) as usize,
                        var.generation(),
                    ))
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn remove_overlaps<'a, V: Into<SimpleVar<'a>>>(&mut self, var: V) {
        let (space, iv) = Self::interval(var);
        if let Some(ref mut ivs) = self.0.get_mut(&space) {
            ivs.remove_overlaps(iv)
        }
    }

    pub fn overlaps<'a, 'b, V: Into<SimpleVar<'a>>>(&self, var: V) -> Vec<SimpleVar<'b>> {
        let var = var.into();
        let (space, iv) = Self::interval(&var);
        if let Some(ref ivs) = self.0.get(&space) {
            ivs.find_all(iv)
                .into_iter()
                .map(|e| {
                    let iv = e.interval();
                    SimpleVar::from(Var::new(
                        space,
                        *iv.start(),
                        (1 + *iv.end() - *iv.start()) as usize,
                        var.generation(),
                    ))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    }

    pub fn enclosing<'a, V: Into<SimpleVar<'a>>>(&self, var: V) -> SimpleVar<'a> {
        let var = var.into();
        let (space, iv) = Self::interval(&var);
        if let Some(ivs) = self.0.get(&space) {
            let iv = ivs.find_all(&iv).into_iter().fold(&iv, |iv, ent| {
                let eiv = ent.interval();
                if (eiv.start() < iv.start() && eiv.end() >= iv.end())
                    || (eiv.start() <= iv.start() && eiv.end() > iv.end())
                {
                    eiv
                } else {
                    iv
                }
            });
            SimpleVar::from(Var::new(
                space,
                *iv.start(),
                (1 + *iv.end() - *iv.start()) as usize,
                var.generation(),
            ))
        } else {
            var
        }
    }

    fn interval<'a, V: Into<SimpleVar<'a>>>(var: V) -> (AddressSpaceId, Interval<u64>) {
        let var = var.into();
        let iv = Interval::from(var.offset()..=(var.offset() + (var.bits() as u64) / 8 - 1));
        (var.space(), iv)
    }

    pub fn contains<'a, V: Into<SimpleVar<'a>>>(&self, var: V) -> bool {
        let (space, iv) = Self::interval(var);
        self.0
            .get(&space)
            .map(|ivs| ivs.overlaps(iv))
            .unwrap_or(false)
    }

    // similar to contains, except only true if overlaps and not equal
    pub fn contains_partial<'a, V: Into<SimpleVar<'a>>>(&self, var: V) -> bool {
        let (space, iv) = Self::interval(var);
        self.0
            .get(&space)
            .map(|ivs| ivs.overlaps(&iv) && ivs.find_exact(&iv).is_none())
            .unwrap_or(false)
    }
}
