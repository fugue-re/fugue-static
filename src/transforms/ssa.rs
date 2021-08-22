use std::borrow::Cow;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use petgraph::EdgeDirection;
use petgraph::graph::NodeIndex;

use fugue::ir::il::ecode::Var;

use crate::models::CFG;
use crate::traits::Variables;
use crate::traits::dominance::DominanceTree;

#[derive(Debug, Clone)]
#[repr(transparent)]
struct SimpleVar<'a>(Cow<'a, Var>);

impl<'a> SimpleVar<'a> {
    fn owned(var: &Var) -> Self {
        Self(Cow::Owned(var.clone()))
    }

    fn into_owned<'b>(self) -> SimpleVar<'b> where 'a: 'b {
        Self(Cow::Owned(self.0.into_owned()))
    }
}

impl<'a> From<&'a Var> for SimpleVar<'a> {
    fn from(var: &'a Var) -> Self {
        Self(Cow::Borrowed(var))
    }
}

impl<'a> From<&'a mut Var> for SimpleVar<'a> {
    fn from(var: &'a mut Var) -> Self {
        Self(Cow::Borrowed(var))
    }
}

impl<'a> Deref for SimpleVar<'a> {
    type Target = Cow<'a, Var>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> PartialEq for SimpleVar<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.space().index() == other.space().index() &&
            self.offset() == other.offset() &&
            self.bits() == other.bits()
    }
}
impl<'a> Eq for SimpleVar<'a> { }

impl<'a> Hash for SimpleVar<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.space().hash(state);
        self.offset().hash(state);
        self.bits().hash(state);
    }
}

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

fn transform<'ecode>(g: &mut CFG<'ecode>) {
    let df = g.dominance_frontier();
    let dt = g.dominance_tree();

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

        for def in defs_tmp.drain().map(SimpleVar::owned) {
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
        &mut phi_locs,
        &dt,
        g,
    );

    // populate phi assignments for each block
    for (nx, mut phim) in phi_locs.into_iter() {
        let eid = g[nx].clone();
        let block = g.blocks.get_mut(&eid).unwrap();

        for (var, phin) in block.to_mut().value_mut().phis_mut() {
            let (_, ns) = phim.remove(&SimpleVar::from(var)).unwrap();
            *phin = ns;
        }
    }
}

fn transform_rename<'a>(
    phi_locs: &mut HashMap<NodeIndex, HashMap<SimpleVar<'a>, (HashSet<NodeIndex>, Vec<Var>)>>,
    dt: &DominanceTree,
    g: &mut CFG,
) {
    let mut pre_order = dt.visit_pre_order();
    let (mut gmapping, mut ssa_stack) = SSAScopeStack::new();

    while let Some(node) = pre_order.next() {
        let mut renamer = ssa_stack.pop();

        let eid = g[node].clone();
        let block = g.blocks.get_mut(&eid).unwrap();

        if let Some(phi) = phi_locs.get(&node) {
            for (var, _) in phi.iter() {
                let generation = renamer.define(&mut gmapping, var.clone());
                let nvar = var.with_generation(generation);
                block.to_mut().value_mut().phis_mut().push((nvar, Vec::new()));
            }
        }

        for op in block.to_mut().value_mut().operations_mut() {
            for var in op.value_mut().used_variables_mut::<Vec<_>>().into_iter() {
                let simple = SimpleVar::from(&*var);
                if let Some(generation) = renamer.current(&simple) {
                    *var = var.with_generation(generation);
                }
            }

            for var in op.value_mut().defined_variables_mut::<Vec<_>>().into_iter() {
                let simple = SimpleVar::owned(&*var);
                *var = var.with_generation(renamer.define(&mut gmapping, simple));
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
