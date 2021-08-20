use std::borrow::Cow;
use fugue::ir::il::ecode::Entity;

pub type EntityRef<'a, T> = Cow<'a, Entity<T>>;

pub trait IntoEntityRef<'a>: Clone {
    type T: Clone;

    fn into_entity_ref(self) -> EntityRef<'a, Self::T>;
}

impl<'a, T> IntoEntityRef<'a> for &'a Entity<T> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, Self::T> {
        Cow::Borrowed(self)
    }
}

impl<'a, T> IntoEntityRef<'a> for Entity<T> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, T> {
        Cow::Owned(self)
    }
}

impl<'a, T> IntoEntityRef<'a> for Cow<'a, Entity<T>> where T: Clone {
    type T = T;

    fn into_entity_ref(self) -> EntityRef<'a, T> {
        self
    }
}
