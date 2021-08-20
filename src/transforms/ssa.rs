use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use petgraph::EdgeDirection;
use petgraph::graph::NodeIndex;
use petgraph::visit::Dfs;

use fugue::ir::il::ecode::Var;

use crate::models::CFG;
use crate::traits::Variables;
use crate::traits::dominance::DominanceTree;

#[derive(Debug, Clone)]
#[repr(transparent)]
struct SimpleVar(Var);

impl From<&Var> for SimpleVar {
    fn from(var: &Var) -> Self {
        Self(var.clone())
    }
}

impl From<&mut Var> for SimpleVar {
    fn from(var: &mut Var) -> Self {
        Self(var.clone())
    }
}

impl Deref for SimpleVar {
    type Target = Var;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for SimpleVar {
    fn eq(&self, other: &Self) -> bool {
        self.space().index() == other.space().index() &&
            self.offset() == other.offset() &&
            self.bits() == other.bits()
    }
}
impl Eq for SimpleVar { }

impl Hash for SimpleVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.space().hash(state);
        self.offset().hash(state);
        self.bits().hash(state);
    }
}

type SSAMapping = HashMap<SimpleVar, usize>;

#[derive(Clone)]
struct SSAScope(SSAMapping);

impl SSAScope {
    pub fn new() -> Self {
        Self(SSAMapping::new())
    }

    pub fn current(&self, operand: &SimpleVar) -> Option<usize> {
        self.0.get(operand).copied()
    }

    pub fn define(&mut self, mapping: &mut SSAMapping, operand: &SimpleVar) -> usize {
        let generation = *mapping
            .entry(operand.clone())
            .and_modify(|v| *v += 1)
            .or_insert(1);

        self.0.insert(operand.clone(), generation);
        generation
    }
}

struct SSAScopeStack(Vec<SSAScope>);

impl SSAScopeStack {
    fn new() -> (SSAMapping, Self) {
        (HashMap::new(), Self(vec![SSAScope::new()]))
    }

    fn push(&mut self, scope: &SSAScope) {
        self.0.push(scope.clone());
    }

    fn pop(&mut self) -> SSAScope {
        self.0.pop().unwrap()
    }
}

fn transform<'ecode>(g: &mut CFG<'ecode>) {
    let df = g.dominance_frontier();
    let (dt_root, dt) = g.dominance_tree();

    let mut defined = HashMap::new(); // SimpleVar -> Set<BlockIds>
    let mut free = HashSet::new(); // Set<SimpleVar>

    // we keep these to avoid allocating many hash sets
    // that get transformed
    let mut defs_tmp = HashSet::new();
    let mut uses_tmp = HashSet::new();

    // def. use information for each block and g
    for (eid, blk) in g.blocks.iter() {
        blk.value().defined_and_used_variables_with(&mut defs_tmp, &mut uses_tmp);
        free.extend(uses_tmp.drain().map(SimpleVar::from));

        let nx = g.entity_mapping[eid];

        for def in defs_tmp.drain().map(SimpleVar::from) {
            defined.entry(def)
                .or_insert_with(HashSet::new)
                .insert(nx);
        }
    }

    // phi placement
    let mut phi_locs = HashMap::new();

    for (def, blks) in defined.into_iter().filter(|(def, _)| free.contains(&def)) {
        let mut phis = HashSet::new();
        let mut workq = blks.iter().cloned().collect::<VecDeque<_>>();

        while let Some(node) = workq.pop_front() {
            if let Some(df_nodes) = df.get(&node) {
                for df_node in df_nodes.iter() {
                    if phis.contains(df_node) {
                        continue
                    }

                    phi_locs.entry(*df_node)
                        .or_insert_with(HashMap::new)
                        .entry(def.clone())
                        .or_insert_with(|| (HashSet::new(), Vec::new()))
                        .0
                        .insert(node);

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
        dt_root,
        &mut phi_locs,
        &dt,
        g,
    );

    // populate phi assignments for each block
    for (nx, mut phim) in phi_locs.into_iter() {
        let eid = g[nx].clone();
        let block = g.blocks.get_mut(&eid).unwrap();

        for (var, phin) in block.to_mut().value_mut().phis_mut() {
            // TODO: remove allocation
            let (_, ns) = phim.remove(&SimpleVar::from(var)).unwrap();
            *phin = ns;
        }
    }
}

fn transform_rename(
    root: NodeIndex,
    phi_locs: &mut HashMap<NodeIndex, HashMap<SimpleVar, (HashSet<NodeIndex>, Vec<Var>)>>,
    dt: &DominanceTree,
    g: &mut CFG,
) {
    let mut pre_order = Dfs::new(dt, root);
    let (mut gmapping, mut ssa_stack) = SSAScopeStack::new();

    while let Some(node) = pre_order.next(dt) {
        let mut renamer = ssa_stack.pop();

        let eid = g[node].clone();
        let block = g.blocks.get_mut(&eid).unwrap();

        if let Some(phi) = phi_locs.get(&node) {
            for (var, _) in phi.iter() {
                let generation = renamer.define(&mut gmapping, var);
                let nvar = var.with_generation(generation);
                block.to_mut().value_mut().phis_mut().insert(nvar, Vec::new());
            }
        }

        for op in block.to_mut().value_mut().operations_mut() {
            for var in op.value_mut().used_variables_mut::<Vec<_>>().into_iter() {
                // TODO: avoid this allocation
                let simple = SimpleVar::from(&*var);
                if let Some(generation) = renamer.current(&simple) {
                    *var = var.with_generation(generation);
                }
            }

            for var in op.value_mut().defined_variables_mut::<Vec<_>>().into_iter() {
                // TODO: avoid this allocation
                let simple = SimpleVar::from(&*var);
                *var = var.with_generation(renamer.define(&mut gmapping, &simple));
            }
        }

        let mut walker = g.entity_graph()
            .neighbors_directed(node, EdgeDirection::Outgoing)
            .detach();

        while let Some((_, succ)) = walker.next(g.entity_graph()) {
            if let Some(phi) = phi_locs.get_mut(&succ) {
                for (var, (_, ns)) in phi.iter_mut() {
                    let generation = renamer.current(var).unwrap_or(0);
                    ns.push(var.with_generation(generation));
                }
            }
        }

        for _succ in dt.neighbors_directed(node, EdgeDirection::Outgoing) {
            // TODO: rewrite to remove recursion
            ssa_stack.push(&renamer);
        }
    }
}

pub trait SSATransform {
    fn ssa(&mut self);
}

impl<'ecode> SSATransform for CFG<'ecode> {
    #[inline(always)]
    fn ssa(&mut self) {
        transform(self)
    }
}
