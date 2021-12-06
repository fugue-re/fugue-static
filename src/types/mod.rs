pub mod id;
pub mod entity;
pub mod locatable;
pub mod variable;

pub use id::{Erased, Id, Identifiable, LocatableId};
pub use entity::{Entity, EntityIdMapping, EntityLocMapping, EntityMap, EntityRef, EntityRefMap, IntoEntityRef, LocatableEntity, LocatableEntityRef};
pub use locatable::{Located, Locatable, LocationTarget, Relocatable};
pub use variable::{SimpleVar, VarView, VarViews};