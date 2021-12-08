// “D. J. Kuck, R. H. Kuhn, D. A. Padua, B. Leasure, and M. Wolfe (1981). DEPENDENCE GRAPHS AND COMPILER OPTIMIZATIONS.”
//
// The current implementation of DDG differs slightly from the dependence graph described in [1] in the following ways:
//
// The graph nodes in the paper represent three main program components, namely assignment statements, for loop headers and while loop headers. In this implementation, DDG nodes naturally represent LLVM IR instructions. An assignment statement in this implementation typically involves a node representing the store instruction along with a number of individual nodes computing the right-hand-side of the assignment that connect to the store node via a def-use edge. The loop header instructions are not represented as special nodes in this implementation because they have limited uses and can be easily identified, for example, through LoopAnalysis.
// The paper describes five types of dependency edges between nodes namely loop dependency, flow-, anti-, output-, and input- dependencies. In this implementation memory edges represent the flow-, anti-, output-, and input- dependencies. However, loop dependencies are not made explicit, because they mainly represent association between a loop structure and the program elements inside the loop and this association is fairly obvious in LLVM IR itself.
// The paper describes two types of pi-blocks; recurrences whose bodies are SCCs and IN nodes whose bodies are not part of any SCC. In this implementation, pi-blocks are only created for recurrences. IN nodes remain as simple DDG nodes in the graph.

// https://llvm.org/docs/DependenceGraphs/index.html#id6

use std::collections::HashSet;

use fugue::ir::il::ecode::Var;

use crate::models::Block;
use crate::graphs::entity::{AsEntityGraph, AsEntityGraphMut, EntityGraph};
use crate::types::Id;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DependenceKind {
    Flow,   // E1 writes var E2 later reads (reaching definitions)
    Anti,   // E1 reads var E2 later writes (use-def when E1's use in E2's kill -> killed liveness)
    Input,  // E1 reads var E2 later reads (use-use)
    Output, // E1 writes var E2 later writes (def-def when E1's def in E2's kill)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dependence {
    kind: DependenceKind,
    variable: Var,
}

#[derive(Debug, Clone)]
pub struct PIBlock(HashSet<Id<Block>>); // SSC blocks -> do we need them?

#[derive(Clone)]
pub struct DDG<'a, V> where V: Clone {
    graph: EntityGraph<'a, V, DependenceKind>,
    pi_blocks: Vec<PIBlock>,
}

impl<'a, V> AsEntityGraph<'a, V, DependenceKind> for DDG<'a, V>
where V: Clone {
    fn entity_graph(&self) -> &EntityGraph<'a, V, DependenceKind> {
        &self.graph
    }
}

impl<'a, V> AsEntityGraphMut<'a, V, DependenceKind> for DDG<'a, V>
where V: Clone {
    fn entity_graph_mut(&mut self) -> &mut EntityGraph<'a, V, DependenceKind> {
        &mut self.graph
    }
}

impl<'a, V> DDG<'a, V>
where V: Clone {
    pub fn new<E, G>(_cfg: G) -> Self
    where G: AsEntityGraph<'a, V, E> {
        todo!()

        /*
        let mut entity_mapping = HashMap::new();
        let mut ddg = EntityGraph::new();

        for blk in cfg.blocks().values() {
            // NOTE:
            // phi's are implicit and have a stable ordering
            // we can treat them as a single block of defs/uses since
            // we will never have a situation where one phi def kills
            // another one
            //
            if !blk.phis().is_empty() {
                let pid = EntityId::new("phi", blk.location().clone());
                entity_mapping.insert(pid.clone(), ddg.add_node(pid));
            }

            for stmt in blk.operations().iter() {
                let sid = stmt.id();
                entity_mapping.insert(sid.clone(), ddg.add_node(sid.clone()));
            }
        }

        let mut defs = HashMap::new(); // Var -> HashSet<Loc>
        let mut uses = HashMap::new(); // Var -> HashSet<Loc>

        // generate edges; we do a RPO traversal of the CFG; since
        // we already have the CFG in SSA form, this should enable
        // use to get any Input/Output/Anti
        for nx in RevPostOrder::into_queue(&**cfg).into_iter() {
            let blk = cfg.block_at(nx);
            if !blk.phis().is_empty() {
                let pid = EntityId::new("phi", blk.location().clone());
            }

            for stmt in blk.operations().iter() {
                let ldefs = stmt.defined_variables::<HashSet<_>>();
                let luses = stmt.used_variables::<HashSet<_>>();
            }
        }

        // we can do a separate pass for flow to get the remaining RDs;
        // the SSA form implicitly encodes this information anyway?
        todo!()
        */
    }
}
