use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use petgraph::EdgeDirection;
use petgraph::graph::NodeIndex;

use fugue::ir::il::ecode::Var;

use crate::models::CFG;
use crate::models::cfg::CFGDominanceTree;

use crate::traits::{Variables, VariablesMut};

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

#[derive(Default)]
pub struct SSA {
    mapping: HashMap<SimpleVar, usize>,
    scoping: Vec<HashMap<SimpleVar, usize>>,
}

impl SSA {
    pub fn transform<'ecode>(g: &mut CFG<'ecode>) {
        let mut t = Self::default();

        let df = g.dominance_frontier();
        let dt = g.dominance_tree();

        let mut defined = HashMap::new(); // SimpleVar -> Set<BlockIds>
        let mut free = HashSet::new(); // Set<SimpleVar>

        // we keep these to avoid allocating many hash sets
        // that get transformed
        let mut defs_tmp = HashSet::new();
        let mut uses_tmp = HashSet::new();

        // def. use information for each block and g
        for (id, blk) in g.blocks() {
            blk.value().defined_and_used_variables_with(&mut defs_tmp, &mut uses_tmp);

            free.extend(uses_tmp.drain().map(SimpleVar::from));

            for def in defs_tmp.drain().map(SimpleVar::from) {
                defined.entry(def)
                    .or_insert_with(HashSet::new)
                    .insert(id);
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
        t.transform_rename(
            g.default_entry().unwrap(),
            &mut phi_locs,
            &dt,
            g,
        );

        // populate phi assignments for each block
        for (nx, mut phim) in phi_locs.into_iter() {
            let block = g.graph_mut().node_weight_mut(nx).unwrap();
            for (var, phin) in block.to_mut().value_mut().phis_mut() {
                // TODO: remove allocation
                let (_, ns) = phim.remove(&SimpleVar::from(var)).unwrap();
                *phin = ns;
            }
        }
    }

    fn transform_rename(
        &mut self,
        node: NodeIndex<u32>,
        phi_locs: &mut HashMap<NodeIndex, HashMap<SimpleVar, (HashSet<NodeIndex>, Vec<Var>)>>,
        dt: &CFGDominanceTree,
        g: &mut CFG,
    ) {
        self.enter_scope();
        {
            let block = g.graph_mut().node_weight_mut(node).unwrap();

            if let Some(phi) = phi_locs.get(&node) {
                for (var, _) in phi.iter() {
                    let generation = self.next_generation(var);
                    let nvar = var.with_generation(generation);
                    block.to_mut().value_mut().phis_mut().insert(nvar, Vec::new());
                }
            }

            for op in block.to_mut().value_mut().operations_mut() {
                op.value_mut().used_variables_mut(|var| {
                    // TODO: avoid this allocation
                    let simple = SimpleVar::from(&*var);
                    if let Some(generation) = self.generation(&simple) {
                        var.with_generation(generation);
                    }
                });

                op.value_mut().defined_variables_mut(|var| {
                    // TODO: avoid this allocation
                    let simple = SimpleVar::from(var);
                    self.next_generation(&simple);
                });
            }
        }

        let mut walker = g.graph()
            .neighbors_directed(node, EdgeDirection::Outgoing)
            .detach();

        while let Some((_, succ)) = walker.next(g.graph()) {
            if let Some(phi) = phi_locs.get_mut(&succ) {
                for (var, (_, ns)) in phi.iter_mut() {
                    let generation = self.generation(var).unwrap_or(0);
                    ns.push(var.with_generation(generation));
                }
            }
        }

        for succ in dt.neighbors_directed(node, EdgeDirection::Outgoing) {
            // TODO: rewrite to remove recursion
            self.transform_rename(succ, phi_locs, dt, g)
        }

        self.leave_scope();
    }

    fn enter_scope(&mut self) {
        let scope = self.scoping.last()
            .cloned()
            .unwrap_or_else(HashMap::default);
        self.scoping.push(scope);
    }

    fn leave_scope(&mut self) {
        self.scoping.pop();
    }

    fn generation(&self, var: &SimpleVar) -> Option<usize> {
        self.scoping.last()
            .and_then(|scope| scope.get(var))
            .copied()
    }

    fn next_generation(&mut self, var: &SimpleVar) -> usize {
        let generation = *self.mapping.entry(var.clone())
            .and_modify(|v| *v += 1)
            .or_insert(1);
        self.scoping.last_mut().unwrap().insert(var.clone(), generation);
        generation
    }
}
