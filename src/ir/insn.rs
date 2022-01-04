use std::borrow::Cow;
use std::fmt;

use fugue::ir::address::AddressValue;
use fugue::ir::disassembly::{ContextDatabase, IRBuilderArena};
use fugue::ir::error::Error;
use fugue::ir::il::pcode::PCode;
use fugue::ir::il::traits::*;
use fugue::ir::Translator;

use fnv::FnvHashMap as Map;
use hashcons::Term;

use smallvec::{smallvec, SmallVec};

use crate::ir::{FloatKind, Stmt};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Insn {
    pub address: AddressValue,
    pub operations: SmallVec<[Term<Stmt>; 8]>,
    pub delay_slots: usize,
    pub length: usize,
}

impl Insn {
    pub fn lift(
        t: &Translator,
        ffs: &Map<usize, FloatKind>,
        irb: &mut IRBuilderArena,
        ctx: &mut ContextDatabase,
        addr: u64,
        bytes: &[u8],
    ) -> Result<Self, Error> {
        let addr = t.address(addr);
        let raw = t.lift_pcode_raw(ctx, irb, addr, bytes)?;

        let manager = t.manager();
        let user_ops = t.user_ops();
        let address = raw.address;

        Ok(Insn {
            operations: raw
                .operations
                .into_iter()
                .enumerate()
                .map(|(i, op)| {
                    Stmt::from_parts(
                        manager,
                        ffs,
                        user_ops,
                        &address,
                        i,
                        op.opcode,
                        op.inputs,
                        op.output,
                    )
                })
                .collect(),
            address,
            delay_slots: raw.delay_slots as usize,
            length: raw.length as usize,
        })
    }

    pub fn nop(address: AddressValue, length: usize) -> Self {
        Self {
            address,
            operations: smallvec![Stmt::skip()],
            delay_slots: 0,
            length,
        }
    }

    pub fn address(&self) -> AddressValue {
        self.address.clone()
    }

    pub fn operations(&self) -> &[Term<Stmt>] {
        self.operations.as_ref()
    }

    pub fn operations_mut(&mut self) -> &mut SmallVec<[Term<Stmt>; 8]> {
        &mut self.operations
    }

    pub fn delay_slots(&self) -> usize {
        self.delay_slots
    }

    pub fn length(&self) -> usize {
        self.length
    }
}

impl Insn {
    pub fn from_pcode(translator: &Translator, pcode: PCode) -> Self {
        let address = pcode.address;
        let mut operations = SmallVec::with_capacity(pcode.operations.len());

        for (i, op) in pcode.operations.into_iter().enumerate() {
            operations.push(Stmt::from_pcode(translator, op, &address, i));
        }

        Self {
            operations,
            address,
            delay_slots: pcode.delay_slots,
            length: pcode.length,
        }
    }
}

impl fmt::Display for Insn {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let len = self.operations.len();
        if len > 0 {
            for (i, op) in self.operations.iter().enumerate() {
                write!(
                    f,
                    "{}.{:02}: {}{}",
                    self.address,
                    i,
                    op,
                    if i == len - 1 { "" } else { "\n" }
                )?;
            }
            Ok(())
        } else {
            write!(f, "{}.00: skip", self.address)
        }
    }
}

pub struct InsnFormatter<'ecode, 'trans> {
    ecode: &'ecode Insn,
    fmt: Cow<'trans, TranslatorFormatter<'trans>>,
}

impl<'ecode, 'trans> fmt::Display for InsnFormatter<'ecode, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let len = self.ecode.operations.len();
        if len > 0 {
            for (i, op) in self.ecode.operations.iter().enumerate() {
                write!(
                    f,
                    "{}.{:02}: {}{}",
                    self.ecode.address,
                    i,
                    op.display_full(Cow::Borrowed(&*self.fmt)),
                    if i == len - 1 { "" } else { "\n" }
                )?;
            }
            Ok(())
        } else {
            write!(f, "{}.00: skip", self.ecode.address)
        }
    }
}

impl<'ecode, 'trans> TranslatorDisplay<'ecode, 'trans> for Insn {
    type Target = InsnFormatter<'ecode, 'trans>;

    fn display_full(&'ecode self, fmt: Cow<'trans, TranslatorFormatter<'trans>>) -> Self::Target {
        InsnFormatter { ecode: self, fmt }
    }
}
