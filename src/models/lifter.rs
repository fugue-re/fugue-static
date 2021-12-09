use fugue::bytes::Endian;
use fugue::ir::convention::Convention;
use fugue::ir::{AddressSpace, AddressSpaceId, LanguageDB, Translator};
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::Var;

use std::borrow::Cow;
use std::path::Path;

use thiserror::Error;

use crate::types::VarView;

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
    fn new(mut translator: Translator, convention: Convention) -> Self {
        let (register_space, register_index) = VarView::registers(&translator);
        let default_space = translator.manager().default_space();

        let global_space = translator.manager_mut().add_space_like("global", &*default_space).id();
        let stack_space = translator.manager_mut().add_space_like("stack", &*default_space).id();
        let temporary_space = translator.manager_mut().add_space_like("tmp", &*default_space).id();

        let program_counter = Var::new(
            translator.manager().register_space_ref(),
            translator.program_counter().offset(),
            translator.program_counter().size(),
            0,
        );

        let conv_sp = convention.stack_pointer();
        let stack_pointer = Var::new(
            translator.manager().register_space_ref(),
            conv_sp.varnode().offset(),
            conv_sp.varnode().size(),
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
        self.translator().manager().space_by_id(self.temporary_space)
    }

    pub fn temporary_space_id(&self) -> AddressSpaceId {
        self.temporary_space
    }

    pub fn translator(&self) -> &Translator {
        &self.translator
    }
}
