use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use fugue::ir::il::ecode::Var;

#[derive(Debug, Clone)]
#[repr(transparent)]
pub(crate) struct SimpleVar<'a>(Cow<'a, Var>);

impl<'a> SimpleVar<'a> {
    pub(crate) fn owned(var: &Var) -> Self {
        Self(Cow::Owned(var.clone()))
    }

    pub(crate) fn into_owned<'b>(self) -> SimpleVar<'b> where 'a: 'b {
        Self(Cow::Owned(self.0.into_owned()))
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
        self.space().index() == other.space().index() &&
            self.offset() == other.offset() &&
            self.bits() == other.bits()
    }
}
impl<'a> Eq for SimpleVar<'a> { }

impl<'a> Hash for SimpleVar<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.space().hash(state);
        self.offset().hash(state);
        self.bits().hash(state);
    }
}
