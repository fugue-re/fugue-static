use std::collections::{BTreeSet, HashMap};

use fugue::bv::BitVec;
use fugue::db;

use fugue::ir::Translator;
use fugue::ir::disassembly::{ContextDatabase, IRBuilderArena};
use fugue::ir::il::ecode::{BranchTargetT, ECode, Location, StmtT, Var};
use fugue::ir::il::traits::*;

use crate::models::cfg::{BranchKind, CFG};
use crate::models::{Block, BlockT, BlockLifter, Lifter};
use crate::traits::{BlockOracle, StmtExt};
use crate::traits::oracle::NullOracle;
use crate::types::{Id, Identifiable, LocationTarget, Locatable, LocatableId, Entity, EntityIdMapping, EntityLocMapping};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Lifting(#[from] crate::models::block::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone
{
    id: LocatableId<FunctionT<Loc, Val, Var>>,
    symbol: String,
    block_ids: HashMap<Id<BlockT<Loc, Val, Var>>, Location>,
    callers: HashMap<LocatableId<FunctionT<Loc, Val, Var>>, LocatableId<StmtT<Loc, Val, Var>>>,
}

pub type Function = FunctionT<Location, BitVec, Var>;

impl<Loc, Val, Var> Identifiable<FunctionT<Loc, Val, Var>> for FunctionT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone
{
    fn id(&self) -> Id<FunctionT<Loc, Val, Var>> {
        self.id.id()
    }
}

impl<Loc, Val, Var> Locatable for FunctionT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone
{
    fn location(&self) -> Location {
        self.id.location()
    }
}

impl<Loc, Val, Var> FunctionT<Loc, Val, Var>
where
    Loc: Clone,
    Val: Clone,
    Var: Clone,
{
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn blocks(&self) -> &HashMap<Id<BlockT<Loc, Val, Var>>, Location> {
        &self.block_ids
    }

    pub fn callers(&self) -> &HashMap<LocatableId<FunctionT<Loc, Val, Var>>, LocatableId<StmtT<Loc, Val, Var>>> {
        &self.callers
    }

    pub fn callers_mut(&mut self) -> &mut HashMap<LocatableId<FunctionT<Loc, Val, Var>>, LocatableId<StmtT<Loc, Val, Var>>> {
        &mut self.callers
    }

    pub fn cfg<'db, M>(&self, mapping: &'db M) -> CFG<'db, BlockT<Loc, Val, Var>>
    where M: 'db + EntityIdMapping<BlockT<Loc, Val, Var>> + EntityLocMapping<BlockT<Loc, Val, Var>>,
          Loc: Locatable,
          Val: 'db,
          Var: 'db {
        self.cfg_with(mapping, &NullOracle)
    }

    pub fn cfg_with<'db, M, O>(&self, mapping: &'db M, oracle: &O) -> CFG<'db, BlockT<Loc, Val, Var>>
    where M: 'db + EntityIdMapping<BlockT<Loc, Val, Var>> + EntityLocMapping<BlockT<Loc, Val, Var>>,
          O: 'db + BlockOracle,
          Loc: Locatable,
          Val: 'db,
          Var: 'db {
        let mut cfg = CFG::new();
        let mut succs = Vec::new();

        for blkid in self.block_ids.keys() {
            let blk = &mapping.lookup_by_id(*blkid).expect("block exists");
            if blk.location() == self.location() {
                cfg.add_root_entity(blk);
            } else {
                cfg.add_entity(blk);
            }
        }

        for blkid in self.block_ids.keys() {
            let blk = &mapping.lookup_by_id(*blkid).expect("block exists");
            let mut oracle_resolved = oracle.block_succs(&blk.location())
                .unwrap_or_default()
                .into_iter()
                .collect::<BTreeSet<_>>();

            for known in blk.next_blocks() {
                if let LocationTarget::Fixed(ref loc) = known {
                    oracle_resolved.remove(loc);
                }
            }

            match &**blk.value().last().value() {
                StmtT::CBranch(_, t) => match t {
                    BranchTargetT::Location(ref location) => {
                        let tgt = &mapping.lookup_by_location::<Option<_>>(&location.location()).expect(&format!("block exists at {}", location.location()));
                        if self.block_ids.contains_key(&tgt.id()) {
                            let fall = blk.next_block_entities::<_, Option<_>>(mapping).unwrap();
                            cfg.add_cond(blk, tgt, fall);
                        }
                    },
                    BranchTargetT::Computed(_) => {
                        for location in oracle_resolved.into_iter() {
                            let tgt = &mapping.lookup_by_location::<Option<_>>(&location).expect("block exists");
                            if self.block_ids.contains_key(&tgt.id()) {
                                cfg.add_jump(blk, tgt);
                            }
                        }
                        let fall = blk.next_block_entities::<_, Option<_>>(mapping).unwrap();
                        cfg.add_fall(blk, fall);
                    },
                }
                StmtT::Branch(t) => match t {
                    BranchTargetT::Location(ref location) => {
                        let tgt = &mapping.lookup_by_location::<Option<_>>(&location.location()).expect(&format!("block exists at {}", location.location()));
                        if self.block_ids.contains_key(&tgt.id()) {
                            cfg.add_jump(blk, tgt);
                        }
                    },
                    BranchTargetT::Computed(_) => {
                        for location in oracle_resolved.into_iter() {
                            let tgt = &mapping.lookup_by_location::<Option<_>>(&location).expect("block exists");
                            if self.block_ids.contains_key(&tgt.id()) {
                                cfg.add_jump(blk, tgt);
                            }
                        }
                    },
                }
                _ => (),
            }

            let blkx = cfg.entity_vertex(*blkid).unwrap();
            blk.next_block_entities_with(mapping, &mut succs);
            for tgt in succs.drain(..) {
                if let Some(vx) = cfg.entity_vertex(tgt.id()) {
                    if !cfg.contains_edge(blkx, vx)
                        && !blk
                            .operations()
                            .last()
                            .map(|op| op.is_return())
                            .unwrap_or(false)
                    {
                        cfg.add_vertex_relation(blkx, vx, BranchKind::Fall);
                    }
                }
            }
        }

        cfg
    }

    pub fn translate<T: TranslateIR<Loc, Val, Var>>(self, _: &T) -> FunctionT<T::TLoc, T::TVal, T::TVar>
    where
        T::TLoc: Clone,
        T::TVal: Clone,
        T::TVar: Clone,
    {
        FunctionT {
            id: self.id.retype(),
            symbol: self.symbol,
            block_ids: self.block_ids.into_iter().map(|(k, v)| (k.retype(), v)).collect(),
            callers: self.callers.into_iter().map(|(k, v)| (k.retype(), v.retype())).collect(),
        }
    }
}

pub struct FunctionBuilder<'trans> {
    lifter: &'trans Lifter,
    irb: &'trans mut IRBuilderArena,
    context_db: &'trans mut ContextDatabase,
    symbol: String,
    address: u64,
    block_indices: Vec<usize>,
    blocks: Vec<Entity<Block>>,
}

impl<'trans> FunctionBuilder<'trans> {
    pub fn new(lifter: &'trans Lifter, irb: &'trans mut IRBuilderArena, context_db: &'trans mut ContextDatabase, address: u64, symbol: impl Into<String>) -> Self {
        Self {
            lifter,
            irb,
            context_db,
            symbol: symbol.into(),
            address,
            block_indices: Vec::new(),
            blocks: Vec::new(),
        }
    }

    pub fn add_block(&mut self, address: u64, bytes: &[u8]) -> Result<usize, Error> {
        self.add_block_with(address, bytes, |_| ())
    }

    pub fn add_block_with<F>(&mut self, address: u64, bytes: &[u8], transform: F) -> Result<usize, Error>
    where F: FnMut(&mut ECode) {
        let id = self.block_indices.len();
        let rid = self.blocks.len();
        self.blocks.extend(Block::new_with(&self.lifter.translator(), &mut self.context_db, address, bytes, transform)?);
        self.block_indices.push(rid);
        Ok(id)
    }

    pub fn add_block_and_explore<F>(&mut self, address: u64, bytes: &[u8], hint: Option<usize>, merge: bool, transform: F) -> Option<(usize, BTreeSet<Location>)>
    where F: FnMut(&mut ECode) {
        let id = self.block_indices.len();
        let rid = self.blocks.len();
        let (blocks, mut targets, _, _) = self.lifter.lift_block(&mut self.irb, &mut self.context_db, address, bytes, hint, merge, transform);

        if blocks.is_empty() {
            None
        } else {
            self.blocks.extend(blocks);

            // remove already explored to produce unexplored targets
            for block in self.blocks.iter() {
                targets.remove(&block.location());
            }

            self.block_indices.push(rid);
            Some((id, targets))
        }
    }

    pub fn block(&self, id: usize) -> Option<&[Entity<Block>]> {
        let sid = self.block_indices.get(id)?;
        if let Some(eid) = self.block_indices.get(id + 1) {
            Some(&self.blocks[*sid..*eid])
        } else {
            Some(&self.blocks[*sid..])
        }
    }

    pub fn block_mut(&mut self, id: usize) -> Option<&mut [Entity<Block>]> {
        let sid = self.block_indices.get(id)?;
        if let Some(eid) = self.block_indices.get(id + 1) {
            Some(&mut self.blocks[*sid..*eid])
        } else {
            Some(&mut self.blocks[*sid..])
        }
    }

    pub fn build(self) -> (Entity<Function>, Vec<Entity<Block>>) {
        let id = LocatableId::new("fcn", Location::new(self.lifter.translator().address(self.address), 0));
        let mut block_ids = HashMap::default();

        for blk in self.blocks.iter() {
            let (id, loc) = LocatableId::from(blk).into_parts();
            block_ids.insert(id, loc);
        }

        let f = Function {
            id,
            symbol: self.symbol,
            block_ids,
            callers: HashMap::default(),
        };
        let blocks = self.blocks;

        (f.into(), blocks)
    }
}

pub struct FunctionLifter<'trans> {
    translator: &'trans Translator,
    database: &'trans db::Database,
    block_lifter: BlockLifter<'trans>,
}

impl<'trans> FunctionLifter<'trans> {
    pub fn new(translator: &'trans Translator, database: &'trans db::Database) -> Self {
        Self {
            translator,
            database,
            block_lifter: BlockLifter::new(translator),
        }
    }

    pub fn from_function(
        &mut self,
        f: &db::Function,
    ) -> Result<(Entity<Function>, Vec<Entity<Block>>), Error> {
        self.from_function_with(f, |_| ())
    }

    pub fn from_function_with<F>(
        &mut self,
        f: &db::Function,
        mut transform: F,
    ) -> Result<(Entity<Function>, Vec<Entity<Block>>), Error>
    where F: FnMut(&mut ECode) {
        let id = LocatableId::new("fcn", Location::new(self.translator.address(f.address()), 0));
        let mut blocks = Vec::new();
        let mut function = Function {
            id,
            symbol: f.name().to_string(),
            block_ids: HashMap::new(),
            callers: HashMap::new(),
        };

        for b in f.blocks() {
            let blks = self.block_lifter.from_block_with(b, &mut transform)?;

            blocks.reserve(blks.len());
            function.block_ids.reserve(blks.len());

            for sb in blks.iter() {
                let (id, loc) = LocatableId::from(sb).into_parts();
                function.block_ids.insert(id, loc);
            }

            blocks.extend(blks.into_iter());
        }

        for r in f
            .references()
            .iter()
            .filter(|r| r.is_call() && !r.source_id().is_invalid())
        {
            if let Some(sfcn) = self.database.functions().get(r.source_id().index()) {
                let fid = LocatableId::invalid(
                    "fcn",
                    Location::new(self.translator.address(sfcn.address()), 0),
                );
                let sid = LocatableId::invalid(
                    "stmt",
                    Location::new(self.translator.address(r.address()), 0),
                );
                function.callers.insert(fid, sid);
            }
        }

        Ok((function.into(), blocks))
    }
}
