use fugue::bytes::Endian;
use fugue::ir::convention::Convention;
use fugue::ir::{AddressSpaceId, LanguageDB, Translator};
use fugue::ir::disassembly::ContextDatabase;

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
}

#[derive(Debug, Error)]
pub enum LifterError {
    #[error(transparent)]
    Disassembly(#[from] fugue::ir::error::Error),
}

impl Lifter {
    fn new(translator: Translator, convention: Convention) -> Self {
        let (register_space, register_index) = VarView::registers(&translator);
        Self {
            translator,
            convention,
            register_space,
            register_index,
        }
    }
    
    pub fn context(&self) -> ContextDatabase {
        self.translator.context_database()
    }
    
    pub fn translator(&self) -> &Translator {
        &self.translator
    }
}