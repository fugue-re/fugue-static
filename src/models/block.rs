use std::collections::HashSet;
use std::fmt::{self, Debug, Display};

use fugue::bv::BitVec;
use fugue::db::BasicBlock;

use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::{BranchTarget, ECode, Location, Stmt, StmtT, Var};
use fugue::ir::il::traits::*;
use fugue::ir::Translator;

use thiserror::Error;

use crate::models::PhiT;
use crate::traits::*;
use crate::types::{
    Entity, EntityIdMapping, EntityLocMapping, EntityRef, Id, Identifiable, Locatable, LocatableId,
    Located, LocationTarget,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Lifting(#[from] fugue::ir::error::Error),
}

pub type Block = BlockT<Location, BitVec, Var>;

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct BlockT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    #[serde(bound(deserialize = "LocatableId<BlockT<Loc, Val, Var>>: serde::Deserialize<'de>"))]
    pub(crate) id: LocatableId<BlockT<Loc, Val, Var>>,
    #[serde(bound(deserialize = "Entity<Located<PhiT<Var>>>: serde::Deserialize<'de>"))]
    pub(crate) phis: Vec<Entity<Located<PhiT<Var>>>>,
    #[serde(bound(
        deserialize = "Entity<Located<StmtT<Loc, Val, Var>>>: serde::Deserialize<'de>"
    ))]
    pub(crate) operations: Vec<Entity<Located<StmtT<Loc, Val, Var>>>>,
    #[serde(bound(
        deserialize = "LocationTarget<BlockT<Loc, Val, Var>>: serde::Deserialize<'de>"
    ))]
    pub(crate) next_blocks: Vec<LocationTarget<BlockT<Loc, Val, Var>>>,
}

impl<Loc, Val, Var> Identifiable<BlockT<Loc, Val, Var>> for BlockT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    fn id(&self) -> Id<Self> {
        self.id.id()
    }
}

impl<Loc, Val, Var> Locatable for BlockT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    fn location(&self) -> Location {
        self.id.location()
    }
}

pub trait BlockMapping<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    fn block_by_id(
        &self,
        id: Id<BlockT<Loc, Val, Var>>,
    ) -> Option<EntityRef<BlockT<Loc, Val, Var>>>;
    fn block_by_location(&self, location: &Loc) -> Option<EntityRef<BlockT<Loc, Val, Var>>>;
}

impl<Loc, Val, Var> Display for BlockT<Loc, Val, Var>
where
    Loc: Clone + Display,
    Val: Clone + Display,
    Var: Clone + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for phi in self.phis.iter() {
            writeln!(f, "{} {}", phi.location(), phi.value())?;
        }

        for stmt in self.operations.iter() {
            writeln!(f, "{} {}", stmt.location(), stmt.value())?;
        }

        Ok(())
    }
}

pub struct BlockDisplay<'blk, 'trans, Loc, Val, Var>
where
    Loc: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
    Val: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
    Var: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
{
    blk: &'blk BlockT<Loc, Val, Var>,
    trans: Option<&'trans Translator>,
}

impl<'blk, 'trans, Loc, Val, Var> Display for BlockDisplay<'blk, 'trans, Loc, Val, Var>
where
    Loc: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
    Val: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
    Var: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for phi in self.blk.phis.iter() {
            writeln!(
                f,
                "{} {}",
                phi.location(),
                (**phi.value()).display_with(self.trans)
            )?;
        }

        for stmt in self.blk.operations.iter() {
            writeln!(f, "{} {}", stmt.location(), stmt.display_with(self.trans))?;
        }

        Ok(())
    }
}

impl Block {
    pub fn new(
        translator: &Translator,
        context: &mut ContextDatabase,
        block_addr: u64,
        bytes: &[u8],
    ) -> Result<Vec<Entity<Self>>, Error> {
        Self::new_with(translator, context, block_addr, bytes, |_| ())
    }

    pub fn new_with<F>(
        translator: &Translator,
        context: &mut ContextDatabase,
        block_addr: u64,
        bytes: &[u8],
        mut transform: F,
    ) -> Result<Vec<Entity<Self>>, Error>
    where
        F: FnMut(&mut ECode),
    {
        let mut offset = 0;
        let mut blocks = Vec::with_capacity(4);

        let mut targets = HashSet::new();

        while offset < bytes.len() {
            let address = block_addr + offset as u64;
            let mut ecode =
                translator.lift_ecode(context, translator.address(address), &bytes[offset..])?;

            transform(&mut ecode);

            // Each `ecode` block represents a single architectural
            // instruction, which may have local control-flow.
            //
            // Local control-flow is facilitated by `Stmt::Branch`
            // and `Stmt::CBranch` instructions that have a definite
            // target within the block.
            //
            // For all other control-flow, we split after the operation.
            //
            // We first locate all of the split points, which correspond
            // to "instruction local" branch targets and global branches.
            // Then, we process them in reverse order to partition the
            // vector of operations into blocks.
            //
            // NOTE: Expr::Call will not appear due to lifting, and so we
            // do not need to consider it here.

            let mut local_targets = ecode
                .operations()
                .iter()
                .enumerate()
                .filter_map(|(offset, operation)| match operation {
                    Stmt::Branch(BranchTarget::Location(location))
                    | Stmt::CBranch(_, BranchTarget::Location(location)) => {
                        targets.insert(location.clone());
                        if ecode.address() == *location.address() {
                            Some(location.position())
                        } else if offset + 1 < ecode.operations().len() {
                            Some(offset + 1)
                        } else {
                            None
                        }
                    }
                    Stmt::Call(BranchTarget::Location(location), _)
                    | Stmt::Return(BranchTarget::Location(location)) => {
                        targets.insert(location.clone());
                        if offset + 1 < ecode.operations().len() {
                            Some(offset + 1)
                        } else {
                            None
                        }
                    }
                    stmt if stmt.is_branch() && offset + 1 < ecode.operations().len() => {
                        Some(offset + 1)
                    }
                    _ => None,
                })
                .filter(|offset| *offset != 0)
                .collect::<Vec<_>>();

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

            blocks.extend(local_blocks.into_iter().rev());

            offset += ecode.length;
        }

        // Merge blocks that are not targets of other blocks
        for index in (1..blocks.len()).rev() {
            if !blocks[index - 1].value().last().value().is_branch()
                && !targets.contains(&blocks[index].location())
            {
                let block = blocks.remove(index).into_value();
                blocks[index - 1]
                    .value_mut()
                    .operations
                    .extend(block.operations.into_iter());
                blocks[index - 1].value_mut().next_blocks = block.next_blocks;
            }
        }

        Ok(blocks)
    }
}

impl<Loc, Val, Var> BlockT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    // next blocks are implicit flows due to fall-through behaviour
    pub fn next_blocks(&self) -> impl Iterator<Item = &LocationTarget<Self>> {
        self.next_blocks.iter()
    }

    // next blocks are implicit flows due to fall-through behaviour
    pub fn next_blocks_mut(&mut self) -> &mut Vec<LocationTarget<Self>> {
        &mut self.next_blocks
    }

    pub fn next_block_entities<'a, M, C>(&self, mapping: &'a M) -> C
    where
        M: 'a + EntityIdMapping<Self> + EntityLocMapping<Self>,
        C: EntityRefCollector<'a, Self>,
        Loc: 'a,
        Val: 'a,
        Var: 'a,
    {
        let mut c = C::default();
        self.next_block_entities_with(mapping, &mut c);
        c
    }

    pub fn next_block_entities_with<'a, M, C>(&self, mapping: &'a M, collect: &mut C)
    where
        M: 'a + EntityIdMapping<Self> + EntityLocMapping<Self>,
        C: EntityRefCollector<'a, Self>,
        Loc: 'a,
        Val: 'a,
        Var: 'a,
    {
        for e in self
            .next_blocks
            .iter()
            .filter_map(|tgt| tgt.resolve_with(mapping))
        {
            collect.insert(e);
        }
    }

    pub fn first(&self) -> &Entity<Located<StmtT<Loc, Val, Var>>> {
        &self.operations[0]
    }

    pub fn first_mut(&mut self) -> &mut Entity<Located<StmtT<Loc, Val, Var>>> {
        &mut self.operations[0]
    }

    pub fn last(&self) -> &Entity<Located<StmtT<Loc, Val, Var>>> {
        &self.operations[self.operations.len() - 1]
    }

    pub fn last_mut(&mut self) -> &mut Entity<Located<StmtT<Loc, Val, Var>>> {
        let offset = self.operations.len() - 1;
        &mut self.operations[offset]
    }

    pub fn phis(&self) -> &[Entity<Located<PhiT<Var>>>] {
        &self.phis
    }

    pub fn phis_mut(&mut self) -> &mut Vec<Entity<Located<PhiT<Var>>>> {
        &mut self.phis
    }

    pub fn operations(&self) -> &[Entity<Located<StmtT<Loc, Val, Var>>>] {
        &self.operations
    }

    pub fn operations_mut(&mut self) -> &mut [Entity<Located<StmtT<Loc, Val, Var>>>] {
        &mut self.operations
    }

    pub fn translate<T: TranslateIR<Loc, Val, Var>>(self, t: &T) -> BlockT<T::TLoc, T::TVal, T::TVar>
    where
        T::TLoc: Clone,
        T::TVal: Clone,
        T::TVar: Clone,
    {
        BlockT {
            id: self.id.retype(),
            phis: self.phis.into_iter().map(|e| e.map(|lphi| lphi.map(|phi| phi.translate(t)))).collect(),
            operations: self.operations.into_iter().map(|e| e.map(|lop| lop.map(|op| op.translate(t)))).collect(),
            next_blocks: self.next_blocks.into_iter().map(|nb| nb.retype()).collect(),
        }
    }
}

impl<'blk, 'trans: 'blk, Loc, Val, Var> TranslatorDisplay<'blk, 'trans> for BlockT<Loc, Val, Var>
where
    Loc: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
    Val: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
    Var: Clone + TranslatorDisplay<'blk, 'trans> + 'blk,
{
    type Target = BlockDisplay<'blk, 'trans, Loc, Val, Var>;

    fn display_with(
        &'blk self,
        t: Option<&'trans Translator>,
    ) -> BlockDisplay<'blk, 'trans, Loc, Val, Var> {
        BlockDisplay {
            blk: self,
            trans: t,
        }
    }
}

impl<'ecode, Loc, Val, Var> Variables<'ecode, Var> for BlockT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    fn all_variables_with<C>(&'ecode self, vars: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        for phi in self.phis.iter() {
            vars.insert_ref(&phi.var());
            for pvar in phi.assign().iter() {
                vars.insert_ref(pvar);
            }
        }

        for stmt in self.operations.iter().map(|v| v.value()) {
            stmt.all_variables_with(vars);
        }
    }

    fn all_variables_mut_with<C>(&'ecode mut self, vars: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        for phi in self.phis.iter_mut() {
            let (v, vs) = phi.parts_mut();
            vars.insert_mut(v);
            for pvar in vs {
                vars.insert_mut(pvar);
            }
        }

        for stmt in self.operations.iter_mut().map(|v| v.value_mut()) {
            stmt.all_variables_mut_with(vars);
        }
    }

    // i.e. all vars that are a target of an assignment
    fn defined_variables_with<C>(&'ecode self, defs: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        for phi in self.phis.iter() {
            defs.insert_ref(phi.var());
        }

        for stmt in self.operations.iter().map(|v| v.value()) {
            stmt.defined_variables_with(defs);
        }
    }

    // i.e. all vars used without first being defined aka free vars
    fn used_variables_with<C>(&'ecode self, uses: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        let mut defs = C::default();
        self.defined_and_used_variables_with(&mut defs, uses)
    }

    fn defined_and_used_variables_with<C>(&'ecode self, defs: &mut C, uses: &mut C)
    where
        C: ValueRefCollector<'ecode, Var>,
    {
        let mut ldefs = C::default();
        let mut luses = C::default();

        for phi in self.phis.iter() {
            ldefs.insert_ref(phi.var());
            for rvar in phi.assign() {
                luses.insert_ref(rvar);
            }
        }

        for stmt in self.operations.iter().map(|v| v.value()) {
            stmt.defined_and_used_variables_with(&mut ldefs, &mut luses);

            luses.retain_difference_ref(&defs);

            uses.merge_ref(&mut luses);
            defs.merge_ref(&mut ldefs);
        }
    }

    // i.e. all vars that are a target of an assignment
    fn defined_variables_mut_with<C>(&'ecode mut self, defs: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        for phi in self.phis.iter_mut() {
            defs.insert_mut(phi.var_mut());
        }

        for stmt in self.operations.iter_mut().map(|v| v.value_mut()) {
            stmt.defined_variables_mut_with(defs);
        }
    }

    // i.e. all vars used without first being defined aka free vars
    fn used_variables_mut_with<C>(&'ecode mut self, uses: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        let mut defs = C::default();
        self.defined_and_used_variables_mut_with(&mut defs, uses)
    }

    fn defined_and_used_variables_mut_with<C>(&'ecode mut self, defs: &mut C, uses: &mut C)
    where
        C: ValueMutCollector<'ecode, Var>,
    {
        let mut ldefs = C::default();
        let mut luses = C::default();

        for phi in self.phis.iter_mut() {
            let (v, vs) = phi.parts_mut();
            ldefs.insert_mut(v);
            for rvar in vs {
                luses.insert_mut(rvar);
            }
        }

        for stmt in self.operations.iter_mut().map(|v| v.value_mut()) {
            stmt.defined_and_used_variables_mut_with(&mut ldefs, &mut luses);

            luses.retain_difference_mut(&defs);

            uses.merge_mut(&mut luses);
            defs.merge_mut(&mut ldefs);
        }
    }
}

pub struct BlockLifter<'translator> {
    translator: &'translator Translator,
    context: ContextDatabase,
}

impl<'trans> BlockLifter<'trans> {
    pub fn new(translator: &'trans Translator) -> Self {
        Self {
            translator,
            context: translator.context_database(),
        }
    }

    pub fn from_block(&mut self, block: &BasicBlock) -> Result<Vec<Entity<Block>>, Error> {
        self.from_block_with(block, |_| ())
    }

    pub fn from_block_with<F>(
        &mut self,
        block: &BasicBlock,
        transform: F,
    ) -> Result<Vec<Entity<Block>>, Error>
    where
        F: FnMut(&mut ECode),
    {
        Block::new_with(
            &self.translator,
            &mut self.context,
            block.address(),
            block.bytes(),
            transform,
        )
    }
}
