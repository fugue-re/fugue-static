use std::borrow::Cow;
use std::fmt;

use fugue::ir::{AddressSpaceId, AddressValue, SpaceManager, Translator, VarnodeData};
use fugue::ir::disassembly::IRBuilderArena;
use fugue::ir::space_manager::{FromSpace, IntoSpace};
use fugue::ir::il::pcode::Operand;
use fugue::ir::il::traits::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Location {
    pub(crate) address: AddressValue,
    pub(crate) position: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.address, self.position)
    }
}

impl<'loc, 'trans> TranslatorDisplay<'loc, 'trans> for Location {
    type Target = &'loc Self;

    fn display_with(&'loc self, _translator: Option<&'trans Translator>) -> Self::Target {
        self
    }
}

impl Location {
    pub fn new<A>(address: A, position: usize) -> Location
    where
        A: Into<AddressValue>,
    {
        Self {
            address: address.into(),
            position,
        }
    }

    pub fn address(&self) -> Cow<AddressValue> {
        Cow::Borrowed(&self.address)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn space(&self) -> AddressSpaceId {
        self.address.space()
    }

    pub fn is_relative(&self) -> bool {
        self.space().is_constant() && self.position == 0
    }

    pub fn is_absolute(&self) -> bool {
        !self.is_relative()
    }

    pub fn absolute_from<A>(&mut self, address: A, position: usize)
    where
        A: Into<AddressValue>,
    {
        if self.is_absolute() {
            return;
        }

        let offset = self.address().offset() as i64;
        let position = if offset.is_negative() {
            position
                .checked_sub(offset.abs() as usize)
                .expect("negative offset from position in valid range")
        } else {
            position
                .checked_add(offset as usize)
                .expect("positive offset from position in valid range")
        };

        self.address = address.into();
        self.position = position;
    }
}

impl<'z> FromSpace<'z, VarnodeData> for Location {
    fn from_space_with(t: VarnodeData, _arena: &'z IRBuilderArena, manager: &SpaceManager) -> Self {
        Location::from_space(t, manager)
    }

    fn from_space(vnd: VarnodeData, manager: &SpaceManager) -> Self {
        Self {
            address: AddressValue::new(manager.unchecked_space_by_id(vnd.space()), vnd.offset()),
            position: 0,
        }
    }
}

impl<'z> FromSpace<'z, Operand> for Location {
    fn from_space_with(t: Operand, _arena: &'z IRBuilderArena, manager: &SpaceManager) -> Self {
        Location::from_space(t, manager)
    }

    fn from_space(operand: Operand, manager: &SpaceManager) -> Self {
        match operand {
            Operand::Address { value, .. } => Location {
                address: value.into_space(manager),
                position: 0,
            },
            Operand::Constant { value, .. } => Location {
                address: AddressValue::new(manager.constant_space_ref(), value),
                position: 0,
            },
            Operand::Register { offset, .. } => Location {
                address: AddressValue::new(manager.register_space_ref(), offset),
                position: 0,
            },
            Operand::Variable { offset, space, .. } => Location {
                address: AddressValue::new(manager.unchecked_space_by_id(space), offset),
                position: 0,
            },
        }
    }
}


impl From<AddressValue> for Location {
    fn from(address: AddressValue) -> Self {
        Self {
            address,
            position: 0,
        }
    }
}
