use std::borrow::Cow;
use fugue::ir::il::ecode::Entity;

pub trait IntoEntityCow<'a>: Clone {
    type T: Clone;

    fn into_entity_cow(self) -> Cow<'a, Entity<Self::T>>;
}

impl<'a, T> IntoEntityCow<'a> for &'a Entity<T> where T: Clone {
    type T = T;

    fn into_entity_cow(self) -> Cow<'a, Entity<T>> {
        Cow::Borrowed(self)
    }
}

impl<'a, T> IntoEntityCow<'a> for Entity<T> where T: Clone {
    type T = T;

    fn into_entity_cow(self) -> Cow<'a, Entity<T>> {
        Cow::Owned(self)
    }
}

impl<'a, T> IntoEntityCow<'a> for Cow<'a, Entity<T>> where T: Clone {
    type T = T;

    fn into_entity_cow(self) -> Cow<'a, Entity<T>> {
        self
    }
}
