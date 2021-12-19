/*
use std::collections::VecDeque;
use std::collections::BTreeMap;

use indexmap::IndexSet;
use petgraph::EdgeDirection;
use fugue::ir::il::ecode::Var;
use fugue::ir::il::traits::*;

use fixedbitset::FixedBitSet;

use crate::graphs::algorithms::dominance::{Dominance, DominanceTree};
use crate::graphs::entity::{AsEntityGraphMut, VertexIndex};
use crate::models::{Block, Phi};
use crate::traits::{StmtExt, Variables};
use crate::types::{Locatable, SimpleVar};

type SSAMapping<'a> = BTreeMap<SimpleVar<'a>, usize>;

#[derive(Clone)]
struct SSAScope<'a>(SSAMapping<'a>);

impl<'a> SSAScope<'a> {
    pub fn new() -> Self {
        Self(SSAMapping::new())
    }

    pub fn current(&self, operand: &SimpleVar<'a>) -> Option<usize> {
        self.0.get(operand).copied()
    }

    pub fn define(&mut self, mapping: &mut SSAMapping<'a>, operand: SimpleVar<'a>) -> usize {
        let operand = operand.into_owned();
        let generation = *mapping
            .entry(operand.clone())
            .and_modify(|v| *v += 1)
            .or_insert(1);

        self.0.insert(operand, generation);
        generation
    }
}

struct SSAScopeStack<'a>(Vec<SSAScope<'a>>);

impl<'a> SSAScopeStack<'a> {
    fn new() -> (SSAMapping<'a>, Self) {
        (BTreeMap::new(), Self(vec![SSAScope::new()]))
    }

    fn push(&mut self, scope: &SSAScope<'a>) {
        self.0.push(scope.clone());
    }

    fn pop(&mut self) -> SSAScope<'a> {
        self.0.pop().unwrap()
    }
}

fn transform<'ecode, E, G>(g: &mut G)
where G: AsEntityGraphMut<'ecode, Block, E> {
    let dominance = g.entity_graph().dominators();
    let dt = dominance.dominance_tree();
    let df = dominance.dominance_frontier();

    // block mapping (sparse)
    // variable mapping (not sparse)

    let blk_count = g.entity_graph().vertex_count();
    let mut blk_map = IndexSet::with_capacity(blk_count);
    let mut var_map = IndexSet::new();

    // VecMap<FixedBitSet> (var-id -> blocks)
    let mut defined = BTreeMap::new(); // SimpleVar -> Set<BlockIds>
    // FixedBitSet (var-id)
    let mut free = FixedBitSet::new(); //HashSet::new(); // Set<SimpleVar> (non-local definitions)

    // we keep these to avoid allocating many hash sets
    // that get transformed
    //let mut defs_tmp = HashSet::new();
    //let mut uses_tmp = HashSet::new();

    // 1. Clear old phis if any and alloc new (owned) blocks.
    // 2. Build a block mapping.
    //
    // NOTE: the order of traversal is stable.
    //
    for (_eid, vx, blk) in g.entity_graph_mut().entities_mut() {
        blk.to_mut().phis_mut().clear();
        blk_map.insert(vx);
    }

    // 3. def-use information for each block and g.
    //
    // NOTE: since variable ids are ordered, we can build fixedbitsets
    //       that contain |var_map| at the time of visiting, since the
    //       ids observed at this point will always be less than |var_map|.
    //
    let mut defs = FixedBitSet::with_capacity(var_map.len());
    let mut uses = FixedBitSet::with_capacity(var_map.len());
    let mut diff = FixedBitSet::with_capacity(var_map.len());

    for (i, (_eid, _vx, blk)) in g.entity_graph().entities().enumerate() {
        for op in blk.operations() {
            op.for_each_variable_use(|u| {
                let (idx, is_new) = var_map.insert_full(SimpleVar::from(*u));
                if is_new {
                    defs.grow(idx + 1);
                    uses.grow(idx + 1);
                }
                uses.put(idx);
            });


            op.for_each_variable_def(|d| {
                let (idx, is_new) = var_map.insert_full(SimpleVar::from(*d));
                if is_new {
                    defs.grow(idx + 1);
                    uses.grow(idx + 1);
                }
                defs.put(idx);
            });

            /*
            diff.clone_from(&uses);
            diff.difference_with(&defs);

            free.union_with(&diff);
            */
            uses.difference_with(&defs);
        }

        free.union_with(&uses);

        for def in defs.ones() {
            defined.entry(def)
                .or_insert_with(|| {
                    let dvar = var_map.get_index(def).unwrap().clone();
                    (dvar, FixedBitSet::with_capacity(blk_count))
                })
                .1
                .put(i);
        }

        defs.clear();
        uses.clear();
        diff.clear();
    }

    // phi placement
    let mut phi_locs = BTreeMap::new(); // bid -> (map<vid, preds>)

    for (_did, (def, blks)) in defined.into_iter().filter(|(did, _)| free.contains(*did)) {
        let mut phis = FixedBitSet::with_capacity(blk_count);
        let mut workq = blks.ones().map(|bid| *blk_map.get_index(bid).unwrap()).collect::<VecDeque<_>>();

        while let Some(node) = workq.pop_front() {
            if let Some(df_nodes) = df.get(&node) {
                for df_node in df_nodes.iter() {
                    let df_idx = blk_map.get_index_of(df_node).unwrap();
                    if phis.contains(df_idx) {
                        continue
                    }

                    let preds = g.entity_graph().predecessors(*df_node)
                        .into_iter()
                        .map(|(vx, _)| vx)
                        .collect::<Vec<_>>();

                    phi_locs.entry(*df_node)
                        .or_insert_with(BTreeMap::new)
                        .entry(def.clone())
                        .or_insert((preds, Vec::new()));

                    phis.put(df_idx);

                    if !blks.contains(df_idx) {
                        workq.push_back(*df_node);
                    }
                }
            }
        }
    }

    // rename + construct phi assignment vectors
    transform_rename(
        &mut phi_locs,
        &dt,
        g,
    );

    // populate phi assignments for each block
    for (vx, mut phim) in phi_locs.into_iter() {
        let blk = g.entity_graph_mut().entity_mut(vx);
        for phi in blk.to_mut().value_mut().phis_mut() {
            let (_, mut ns) = phim.remove(&SimpleVar::from(*phi.var())).unwrap();
            ns.dedup(); // TODO: back to set?
            *phi.assign_mut() = ns;
        }
    }
}

fn transform_rename<'a, E, G>(
    phi_locs: &mut BTreeMap<VertexIndex<Block>, BTreeMap<SimpleVar, (Vec<VertexIndex<Block>>, Vec<Var>)>>,
    dt: &DominanceTree<Block>,
    g: &mut G,
) where G: AsEntityGraphMut<'a, Block, E> {
    let mut pre_order = dt.pre_order();
    let (mut gmapping, mut ssa_stack) = SSAScopeStack::new();

    while let Some(dt_node) = pre_order.next() {
        let mut renamer = ssa_stack.pop();

        let node = dt[*dt_node];
        let block = g.entity_graph_mut().entity_mut(node).to_mut();

        if let Some(phi) = phi_locs.get(&node) {
            for (var, _) in phi.iter() {
                let generation = renamer.define(&mut gmapping, var.clone());
                let nvar = var.with_generation(generation);
                let loc = block.location();
                block.value_mut().phis_mut().push(Phi::new(loc, nvar, Vec::new()));
            }
        }

        for op in block.value_mut().operations_mut() {
            let oper = op.value_mut();
            oper.for_each_variable_use_mut(|var| {
                let simple = SimpleVar::from(&*var);
                if let Some(generation) = renamer.current(&simple) {
                    *var = var.with_generation(generation);
                }
            });

            oper.for_each_variable_def_mut(|var| {
                let simple = SimpleVar::owned(&*var);
                *var = var.with_generation(renamer.define(&mut gmapping, simple));
            });
        }

        let mut walker = g.entity_graph_mut()
            .successors(node)
            .detach();

        while let Some((succ, _)) = walker.next(g.entity_graph()) {
            if let Some(phi) = phi_locs.get_mut(&succ) {
                for (var, (_, ns)) in phi.iter_mut().filter(|(_, (preds, _))| preds.contains(&node)) {
                    let generation = renamer.current(var).unwrap_or(0);
                    ns.push(var.with_generation(generation));
                }
            }
        }

        for _succ in dt.neighbors_directed(*dt_node, EdgeDirection::Outgoing) {
            ssa_stack.push(&renamer);
        }
    }
}

pub trait SSATransform<E> {
    fn ssa(&mut self);
}

impl<'ecode, E, G> SSATransform<E> for G where G: AsEntityGraphMut<'ecode, Block, E> {
    #[inline(always)]
    fn ssa(&mut self) {
        transform(self)
    }
}
*/
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};

use petgraph::EdgeDirection;
use fugue::ir::il::ecode::Var;
use fugue::ir::il::traits::*;

use crate::graphs::algorithms::dominance::{Dominance, DominanceTree};
use crate::graphs::entity::{AsEntityGraphMut, VertexIndex};
use crate::models::{Block, Phi};
use crate::traits::{StmtExt, Variables};
use crate::types::{Locatable, SimpleVar};

type SSAMapping<'a> = HashMap<SimpleVar<'a>, usize>;

#[derive(Clone)]
struct SSAScope<'a>(SSAMapping<'a>);

impl<'a> SSAScope<'a> {
    pub fn new() -> Self {
        Self(SSAMapping::new())
    }

    pub fn current(&self, operand: &SimpleVar<'a>) -> Option<usize> {
        self.0.get(operand).copied()
    }

    pub fn define(&mut self, mapping: &mut SSAMapping<'a>, operand: SimpleVar<'a>) -> usize {
        let operand = operand.into_owned();
        let generation = *mapping
            .entry(operand.clone())
            .and_modify(|v| *v += 1)
            .or_insert(1);

        self.0.insert(operand, generation);
        generation
    }
}

struct SSAScopeStack<'a>(Vec<SSAScope<'a>>);

impl<'a> SSAScopeStack<'a> {
    fn new() -> (SSAMapping<'a>, Self) {
        (HashMap::new(), Self(vec![SSAScope::new()]))
    }

    fn push(&mut self, scope: &SSAScope<'a>) {
        self.0.push(scope.clone());
    }

    fn pop(&mut self) -> SSAScope<'a> {
        self.0.pop().unwrap()
    }
}

fn transform<'ecode, E, G>(g: &mut G)
where G: AsEntityGraphMut<'ecode, Block, E> {
    let dominance = g.entity_graph().dominators();
    let dt = dominance.dominance_tree();
    let df = dominance.dominance_frontier();

    let mut defined = HashMap::new(); // SimpleVar -> Set<BlockIds>
    let mut free = HashSet::new(); // Set<SimpleVar> (non-local definitions)

    // we keep these to avoid allocating many hash sets
    // that get transformed
    let mut defs_tmp = HashSet::new();
    let mut uses_tmp = HashSet::new();

    // clear old phis if any and alloc new (owned) blocks
    for (_eid, _vx, blk) in g.entity_graph_mut().entities_mut() {
        blk.to_mut().phis_mut().clear();
    }

    // def. use information for each block and g
    for (_eid, vx, blk) in g.entity_graph().entities() {
        blk.value().defined_and_used_variables_with(&mut defs_tmp, &mut uses_tmp);
        free.extend(uses_tmp.drain().map(SimpleVar::from));

        for def in defs_tmp.drain().map(SimpleVar::owned) {
            defined.entry(def)
                .or_insert_with(HashSet::new)
                .insert(vx);
        }
    }

    // phi placement
    let mut phi_locs = HashMap::new(); // block -> (map<var, preds>)

    for (def, blks) in defined.into_iter().filter(|(def, _)| free.contains(&def)) {
        let mut phis = HashSet::new();
        let mut workq = blks.iter().cloned().collect::<VecDeque<_>>();

        while let Some(node) = workq.pop_front() {
            if let Some(df_nodes) = df.get(&node) {
                for df_node in df_nodes.iter() {
                    if phis.contains(df_node) {
                        continue
                    }

                    let preds = g.entity_graph().predecessors(*df_node)
                        .into_iter()
                        .map(|(vx, _)| vx)
                        .collect::<HashSet<_>>();

                    phi_locs.entry(*df_node)
                        .or_insert_with(HashMap::new)
                        .entry(def.clone())
                        .or_insert((preds, Vec::new()));

                    phis.insert(df_node);

                    if !blks.contains(df_node) {
                        workq.push_back(*df_node);
                    }
                }
            }
        }
    }

    // rename + construct phi assignment vectors
    transform_rename(
        &mut phi_locs,
        &dt,
        g,
    );

    // populate phi assignments for each block
    for (vx, mut phim) in phi_locs.into_iter() {
        let blk = g.entity_graph_mut().entity_mut(vx);
        for phi in blk.to_mut().value_mut().phis_mut() {
            let (_, mut ns) = phim.remove(&SimpleVar::from(*phi.var())).unwrap();
            ns.dedup(); // TODO: back to set?
            *phi.assign_mut() = ns;
        }
    }
}

fn transform_rename<'a, E, G>(
    phi_locs: &mut HashMap<VertexIndex<Block>, HashMap<SimpleVar<'a>, (HashSet<VertexIndex<Block>>, Vec<Var>)>>,
    dt: &DominanceTree<Block>,
    g: &mut G,
) where G: AsEntityGraphMut<'a, Block, E> {
    let mut pre_order = dt.pre_order();
    let (mut gmapping, mut ssa_stack) = SSAScopeStack::new();

    while let Some(dt_node) = pre_order.next() {
        let mut renamer = ssa_stack.pop();

        let node = dt[*dt_node];
        let block = g.entity_graph_mut().entity_mut(node).to_mut();

        if let Some(phi) = phi_locs.get(&node) {
            for (var, _) in phi.iter() {
                let generation = renamer.define(&mut gmapping, var.clone());
                let nvar = var.with_generation(generation);
                let loc = block.location();
                block.value_mut().phis_mut().push(Phi::new(loc, nvar, Vec::new()));
            }
        }

        for op in block.value_mut().operations_mut() {
            let oper = op.value_mut();
            //for var in oper.used_variables_mut::<Vec<_>>().into_iter() {
            oper.for_each_variable_use_mut(|var| {
                let simple = SimpleVar::from(&*var);
                if let Some(generation) = renamer.current(&simple) {
                    *var = var.with_generation(generation);
                }
            });

            //for var in oper.defined_variables_mut::<Vec<_>>().into_iter() {
            oper.for_each_variable_def_mut(|var| {
                let simple = SimpleVar::owned(&*var);
                *var = var.with_generation(renamer.define(&mut gmapping, simple));
            });
        }

        let mut walker = g.entity_graph_mut()
            .successors(node)
            .detach();

        while let Some((succ, _)) = walker.next(g.entity_graph()) {
            if let Some(phi) = phi_locs.get_mut(&succ) {
                for (var, (_, ns)) in phi.iter_mut().filter(|(_, (preds, _))| preds.contains(&node)) {
                    let generation = renamer.current(var).unwrap_or(0);
                    ns.push(var.with_generation(generation));
                }
            }
        }

        for _succ in dt.neighbors_directed(*dt_node, EdgeDirection::Outgoing) {
            ssa_stack.push(&renamer);
        }
    }
}

pub trait SSATransform<E> {
    fn ssa(&mut self);
}

impl<'ecode, E, G> SSATransform<E> for G where G: AsEntityGraphMut<'ecode, Block, E> {
    #[inline(always)]
    fn ssa(&mut self) {
        transform(self)
    }
}
