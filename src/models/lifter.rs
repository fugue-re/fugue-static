use fugue::bytes::Endian;
use fugue::ir::convention::Convention;
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::{BranchTarget, ECode, Stmt, Var};
use fugue::ir::il::Location;
use fugue::ir::{AddressSpace, AddressSpaceId, LanguageDB, Translator};

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::Path;

use thiserror::Error;

use crate::models::Block;
use crate::traits::ecode::*;
use crate::traits::stmt::*;
use crate::types::{
    Entity, Identifiable, Locatable, LocatableId, Located, LocationTarget, VarView,
};

#[derive(Clone)]
pub struct LifterBuilder {
    language_db: LanguageDB,
}

#[derive(Debug, Error)]
pub enum LifterBuilderError {
    #[error(transparent)]
    ArchDef(#[from] fugue::arch::ArchDefParseError),
    #[error(transparent)]
    Backend(#[from] fugue::ir::error::Error),
    #[error("unsupported architecture")]
    UnsupportedArch,
    #[error("unsupported architecture calling convention")]
    UnsupportedConv,
}

impl LifterBuilder {
    pub fn new_with(
        path: impl AsRef<Path>,
        ignore_errors: bool,
    ) -> Result<Self, LifterBuilderError> {
        let language_db = LanguageDB::from_directory_with(path, ignore_errors)?;
        Ok(Self { language_db })
    }
    pub fn new(path: impl AsRef<Path>) -> Result<Self, LifterBuilderError> {
        Self::new_with(path, true)
    }

    pub fn build(
        &self,
        tag: impl Into<Cow<'static, str>>,
        convention: impl AsRef<str>,
    ) -> Result<Lifter, LifterBuilderError> {
        let tag = tag.into();
        let convention = convention.as_ref();

        let builder = self
            .language_db
            .lookup_str(&*tag)?
            .ok_or_else(|| LifterBuilderError::UnsupportedArch)?;
        let translator = builder.build()?;

        if let Some(convention) = translator.compiler_conventions().get(&*convention).cloned() {
            Ok(Lifter::new(translator, convention))
        } else {
            Err(LifterBuilderError::UnsupportedConv)
        }
    }

    pub fn build_with(
        &self,
        processor: impl AsRef<str>,
        endian: Endian,
        bits: u32,
        variant: impl AsRef<str>,
        convention: impl AsRef<str>,
    ) -> Result<Lifter, LifterBuilderError> {
        let convention = convention.as_ref();

        let processor = processor.as_ref();
        let variant = variant.as_ref();

        let builder = self
            .language_db
            .lookup(processor, endian, bits as usize, variant)
            .ok_or_else(|| LifterBuilderError::UnsupportedArch)?;
        let translator = builder.build()?;

        if let Some(convention) = translator.compiler_conventions().get(&*convention).cloned() {
            Ok(Lifter::new(translator, convention))
        } else {
            Err(LifterBuilderError::UnsupportedConv)
        }
    }
}

#[derive(Clone)]
pub struct Lifter {
    translator: Translator,
    convention: Convention,
    register_space: AddressSpaceId,
    register_index: VarView,

    global_space: AddressSpaceId,
    stack_space: AddressSpaceId,
    temporary_space: AddressSpaceId,

    program_counter: Var,
    stack_pointer: Var,
}

#[derive(Debug, Error)]
pub enum LifterError {
    #[error(transparent)]
    Disassembly(#[from] fugue::ir::error::Error),
}

impl Lifter {
    pub fn new(mut translator: Translator, convention: Convention) -> Self {
        let (register_space, register_index) = VarView::registers(&translator);
        let default_space = translator.manager().default_space();

        let global_space = translator
            .manager_mut()
            .add_space_like("global", &*default_space)
            .id();
        let stack_space = translator
            .manager_mut()
            .add_space_like("stack", &*default_space)
            .id();
        let temporary_space = translator
            .manager_mut()
            .add_space_like("tmp", &*default_space)
            .id();

        let program_counter = Var::new(
            translator.manager().register_space_ref(),
            translator.program_counter().offset(),
            translator.program_counter().size() * 8,
            0,
        );

        let conv_sp = convention.stack_pointer();
        let stack_pointer = Var::new(
            translator.manager().register_space_ref(),
            conv_sp.varnode().offset(),
            conv_sp.varnode().size() * 8,
            0,
        );

        Self {
            translator,
            convention,
            register_space,
            register_index,
            global_space,
            stack_space,
            temporary_space,
            program_counter,
            stack_pointer,
        }
    }

    pub fn program_counter(&self) -> Var {
        self.program_counter
    }

    pub fn stack_pointer(&self) -> Var {
        self.stack_pointer
    }

    pub fn convention(&self) -> &Convention {
        &self.convention
    }

    pub fn context(&self) -> ContextDatabase {
        self.translator.context_database()
    }

    pub fn global_space(&self) -> &AddressSpace {
        self.translator().manager().space_by_id(self.global_space)
    }

    pub fn global_space_id(&self) -> AddressSpaceId {
        self.global_space
    }

    pub fn register_space(&self) -> &AddressSpace {
        self.translator().manager().space_by_id(self.register_space)
    }

    pub fn register_space_id(&self) -> AddressSpaceId {
        self.register_space
    }

    pub fn stack_space(&self) -> &AddressSpace {
        self.translator().manager().space_by_id(self.stack_space)
    }

    pub fn stack_space_id(&self) -> AddressSpaceId {
        self.stack_space
    }

    pub fn temporary_space(&self) -> &AddressSpace {
        self.translator()
            .manager()
            .space_by_id(self.temporary_space)
    }

    pub fn temporary_space_id(&self) -> AddressSpaceId {
        self.temporary_space
    }

    pub fn translator(&self) -> &Translator {
        &self.translator
    }

    // We lift blocks based on IDA's model of basic blocks (i.e., only
    // terminate a block on local control-flow/a return).
    //
    // We note that each lifted instruction may have internal control-flow
    // and so for a given block under IDA's model, we may produce many strict
    // basic blocks (i.e., Blks). For each instruction lifted, we determine
    // the kind of control-flow that leaves the chunk of ECode statements. We
    // classify each flow as one of six kinds:
    //
    //  1. Unresolved (call, return, branch, cbranch with computed target)
    //  2. IntraIns   (cbranch, branch with inter-chunk flow)
    //  3. IntraBlk   (fall-through)
    //  4. InterBlk   (cbranch, branch with non-inter-chunk flow)
    //  5. InterSub   (call, return)
    //  6. Intrinsic  (intrinsic in statement position)
    //
    // Each architectural instruction initially becomes one or more blocks; we
    // can later apply a merge strategy to clean blocks up if needed. However,
    // this representation enables us to avoid splitting blocks at a later
    // stage and allows us to build a mapping between each instruction and its
    // blocks.
    pub fn lift_block<F>(
        &self,
        ctxt: &mut ContextDatabase,
        addr: u64,
        bytes: &[u8],
        size_hint: Option<usize>,
        merge: bool,
        mut transform: F,
    ) -> (
        Vec<Entity<Block>>,
        BTreeSet<Location>,
        BTreeSet<Location>,
        usize,
    )
    where
        F: FnMut(&mut ECode),
    {
        let actual_size = bytes.len();
        let attempt_size = size_hint
            .map(|hint| actual_size.min(hint))
            .unwrap_or(actual_size);

        let bytes = &bytes[..attempt_size];

        log::debug!(
            "lifting statements at {:x} with size boundary of {} (actual: {})",
            addr,
            attempt_size,
            actual_size,
        );

        let mut all_targets = BTreeSet::default();
        let mut fcn_targets = BTreeSet::default();

        let mut blks = Vec::new();
        let mut offset = 0;

        while offset < attempt_size {
            let taddr = self.translator.address(addr + offset as u64);
            let view = &bytes[offset..];

            log::trace!("lifting instruction at {}", taddr);

            if let Ok(mut ecode) = self.translator.lift_ecode(ctxt, taddr, view) {
                log::trace!(
                    "lifted instruction sequence consists of {} operations over {} bytes",
                    ecode.operations().len(),
                    ecode.length()
                );

                if ecode.operations.is_empty() {
                    log::trace!("lifted instruction is a no-op");
                    ecode.operations_mut().push(Stmt::skip());
                }

                let targets = ecode.branch_targets();
                let length = ecode.length();

                log::trace!(
                    "lifted instruction sequence consists of {} branch targets",
                    targets.len(),
                );

                let mut should_stop = false;
                for (i, tgt) in targets.iter() {
                    log::trace!("- from {}.{}: {}", addr + offset as u64, i, tgt);
                    should_stop |= tgt.ends_block();
                }

                log::trace!("lifted instruction should terminate block: {}", should_stop);

                let mut local_targets = Vec::default();
                for (_, target) in targets.into_iter() {
                    match target {
                        ECodeTarget::IntraIns(loc, _) => {
                            local_targets.push(loc.position());
                            all_targets.insert(loc);
                        }
                        ECodeTarget::InterBlk(BranchTarget::Location(loc))
                        | ECodeTarget::IntraBlk(loc, false) => {
                            // only inter-blk that isn't fall
                            fcn_targets.insert(loc.clone());
                            all_targets.insert(loc);
                        }
                        ECodeTarget::InterSub(BranchTarget::Location(loc))
                        | ECodeTarget::InterRet(BranchTarget::Location(loc), _) => {
                            all_targets.insert(loc);
                        }
                        _ => (),
                    }
                }

                // Transformation pass on ECode
                transform(&mut ecode);

                // Sort in descending order
                local_targets.sort_by(|u, v| u.cmp(&v).reverse());

                let address = ecode.address();
                let mut operations = ecode
                    .operations
                    .drain(..)
                    .enumerate()
                    .map(|(offset, operation)| {
                        Entity::new(
                            "stmt",
                            Located::new(Location::new(address.clone(), offset), operation),
                        )
                    })
                    .collect::<Vec<_>>();

                let mut local_blocks = Vec::with_capacity(local_targets.len() + 1);
                let mut last_location =
                    LocationTarget::from(Location::new(address.clone() + ecode.length, 0));

                for start in local_targets.into_iter() {
                    let lid = LocatableId::new("blk", Location::new(address.clone(), start));
                    let mut block = Block {
                        id: lid,
                        operations: operations.split_off(start),
                        phis: Default::default(),
                        next_blocks: Vec::default(),
                    };
                    if block
                        .operations()
                        .last()
                        .map(|o| o.has_fall())
                        .unwrap_or(true)
                    {
                        block.next_blocks.push(last_location);
                    }
                    last_location = block.id().into();
                    local_blocks.push(Entity::from_parts(block.id(), block));
                }

                if local_blocks.is_empty() || !operations.is_empty() {
                    let lid = LocatableId::new("blk", Location::new(address.clone(), 0));
                    local_blocks.push(Entity::from_parts(lid.id(), {
                        let mut block = Block {
                            id: lid,
                            operations: if operations.is_empty() {
                                vec![Entity::new(
                                    "stmt",
                                    Located::new(Location::new(address, 0), Stmt::skip()),
                                )]
                            } else {
                                operations
                            },
                            phis: Default::default(),
                            next_blocks: Vec::default(),
                        };
                        if block
                            .operations()
                            .last()
                            .map(|o| o.has_fall())
                            .unwrap_or(true)
                        {
                            block.next_blocks.push(last_location);
                        }
                        block
                    }));
                }

                blks.extend(local_blocks.into_iter().rev());

                offset += length;

                if should_stop {
                    break;
                }
            } else {
                log::trace!("instruction could not be lifted");
                break;
            }
        }

        if merge {
            // Merge blocks that are not targets of other blocks
            for index in (1..blks.len()).rev() {
                if !blks[index - 1].value().last().value().is_branch()
                    && !all_targets.contains(&blks[index].location())
                {
                    let blk = blks.remove(index).into_value();

                    let ops = &mut blks[index - 1].value_mut().operations;

                    if ops.len() == 1 && ops[0].is_skip() {
                        *ops = blk.operations;
                    } else {
                        ops.extend(blk.operations.into_iter());
                    }

                    blks[index - 1].value_mut().next_blocks = blk.next_blocks;
                }
            }
        }

        if cfg!(test) {
            for blk in blks.iter() {
                log::trace!("block positioned at {}", blk.location());
                for next in blk.next_blocks() {
                    log::trace!("block fall is {:?}", next);
                }
            }

            for target in all_targets.iter() {
                log::trace!("block outgoing (fixed) target: {}", target);
            }
        }

        (blks, fcn_targets, all_targets, offset)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_blk_disasm() -> Result<(), Box<dyn std::error::Error>> {
        //env_logger::init();

        let path = Path::new("./processors");

        let builder = LifterBuilder::new(&path)?;
        let lifter = builder.build("x86:LE:64:default", "gcc")?;

        let mut ctxt = lifter.context();

        let mut lift =
            |addr: u64, bytes: &[u8]| lifter.lift_block(&mut ctxt, addr, bytes, None, true, |_| ());

        let (blks1, _, _, len1) = lift(0x1000, &[0x90u8]);
        assert_eq!(blks1.len(), 1);
        assert_eq!(len1, 1);

        let (blks3, _, _, len3) = lift(
            0x1004,
            &[
                0x50, 0xF3, 0xAA, 0x53, 0xFF, 0x13, 0x0F, 0x85, 0xFC, 0x00, 0x00, 0x00,
            ],
        );
        assert_eq!(blks3.len(), 5);
        assert_eq!(len3, 12);

        let (blks4, _, _, len4) = lift(0x1010, &[0xE9, 0xFC, 0x00, 0x00, 0x00]);
        assert_eq!(blks4.len(), 1);
        assert_eq!(len4, 5);

        let (blks5, _, _, len5) = lift(0x1015, &[0x5B, 0xC2, 0x04, 0x00]);
        assert_eq!(blks5.len(), 1);
        assert_eq!(len5, 4);

        let mut lift2 =
            |addr: u64, bytes: &[u8]| lifter.lift_block(&mut ctxt, addr, bytes, Some(2), true, |_| ());

        let (blks2, _, _, len2) = lift2(0x1001, &[0x8b, 0xc7, 0x5f, 0x5e]);
        assert_eq!(blks2.len(), 1); // stosb .. gives two blocks
        assert_eq!(len2, 2);

        Ok(())
    }
}
