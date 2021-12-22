use std::fmt;

use fugue::ir::{AddressSpaceId, Translator, VarnodeData};
use fugue::ir::disassembly::IRBuilderArena;
use fugue::ir::il::pcode::Operand;
use fugue::ir::il::traits::*;
use fugue::ir::space_manager::{FromSpace, SpaceManager};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    pub(crate) space: AddressSpaceId,
    pub(crate) offset: u64,
    pub(crate) bits: usize,
    pub(crate) generation: usize,
}

impl Var {
    pub fn new<S: Into<AddressSpaceId>>(
        space: S,
        offset: u64,
        bits: usize,
        generation: usize,
    ) -> Self {
        Self {
            space: space.into(),
            offset,
            bits,
            generation,
        }
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl BitSize for Var {
    fn bits(&self) -> usize {
        self.bits
    }
}

impl Variable for Var {
    fn generation(&self) -> usize {
        self.generation
    }

    fn generation_mut(&mut self) -> &mut usize {
        &mut self.generation
    }

    fn with_generation(&self, generation: usize) -> Self {
        Self {
            space: self.space.clone(),
            generation,
            ..*self
        }
    }

    fn space(&self) -> AddressSpaceId {
        self.space
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "space[{}][{:#x}].{}:{}",
            self.space().index(),
            self.offset(),
            self.generation(),
            self.bits()
        )
    }
}

impl<'var, 'trans> fmt::Display for VarFormatter<'var, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(trans) = self.translator {
            let space = trans.manager().unchecked_space_by_id(self.var.space());
            if space.is_register() {
                let name = trans
                    .registers()
                    .get(self.var.offset(), self.var.bits() / 8)
                    .unwrap();
                write!(f, "{}.{}:{}", name, self.var.generation(), self.var.bits())
            } else {
                let off = self.var.offset();
                let sig = (off as i64).signum() as i128;
                write!(
                    f,
                    "{}[{}{:#x}].{}:{}",
                    space.name(),
                    if sig == 0 {
                        ""
                    } else if sig > 0 {
                        "+"
                    } else {
                        "-"
                    },
                    self.var.offset() as i64 as i128 * sig,
                    self.var.generation(),
                    self.var.bits()
                )
            }
        } else {
            self.fmt(f)
        }
    }
}

pub struct VarFormatter<'var, 'trans> {
    var: &'var Var,
    translator: Option<&'trans Translator>,
}

impl<'var, 'trans> TranslatorDisplay<'var, 'trans> for Var {
    type Target = VarFormatter<'var, 'trans>;

    fn display_with(
        &'var self,
        translator: Option<&'trans Translator>,
    ) -> VarFormatter<'var, 'trans> {
        VarFormatter {
            var: self,
            translator,
        }
    }
}

impl From<VarnodeData> for Var {
    fn from(vnd: VarnodeData) -> Self {
        Self {
            space: vnd.space(),
            offset: vnd.offset(),
            bits: vnd.size() * 8,
            generation: 0,
        }
    }
}

impl<'z> FromSpace<'z, Operand> for Var {
    fn from_space_with(t: Operand, _arena: &'z IRBuilderArena, manager: &SpaceManager) -> Self {
        Var::from_space(t, manager)
    }

    fn from_space(operand: Operand, manager: &SpaceManager) -> Self {
        match operand {
            Operand::Address { value, size } => Var {
                offset: value.offset(),
                space: manager.default_space_id(),
                bits: size * 8,
                generation: 0,
            },
            Operand::Register { offset, size, .. } => Var {
                offset,
                space: manager.register_space_id(),
                bits: size * 8,
                generation: 0,
            },
            Operand::Variable {
                offset,
                space,
                size,
            } => Var {
                offset,
                space,
                bits: size * 8,
                generation: 0,
            },
            _ => panic!("cannot create Var from Operand::Constant"),
        }
    }
}
