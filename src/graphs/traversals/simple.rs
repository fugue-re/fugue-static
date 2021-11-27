use fixedbitset::FixedBitSet;

use petgraph::graph::NodeIndex;
use petgraph::visit::{VisitMap, Visitable};
use petgraph::Direction;

use crate::graphs::entity::VertexIndex;
use crate::graphs::EntityGraph;

#[derive(Clone, Debug)]
struct Traversal {
    direction: Direction,
    stack: Vec<NodeIndex>,
    starts: Vec<NodeIndex>,
    discovered: FixedBitSet,
    finished: FixedBitSet,
}

impl Traversal {
    fn new_with<V, E, F>(graph: &EntityGraph<V, E>, direction: Direction, starts: F) -> Self
    where
        V: Clone,
        F: FnOnce(&EntityGraph<V, E>) -> Vec<VertexIndex<V>>,
    {
        let mut starts = starts(graph).into_iter().map(|vx| *vx).collect::<Vec<_>>();
        let start = starts.pop().unwrap();

        Self {
            direction,
            stack: vec![start],
            starts,
            discovered: graph.as_ref().visit_map(),
            finished: graph.as_ref().visit_map(),
        }
    }

    fn next<V, E>(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>>
    where
        V: Clone,
    {
        'outer: loop {
            while let Some(&nx) = self.stack.last() {
                if self.discovered.visit(nx) {
                    // First time visiting `nx`: Push neighbors, don't pop `nx`
                    for succ in graph.as_ref().neighbors_directed(nx, self.direction) {
                        if !self.discovered.is_visited(&succ) {
                            self.stack.push(succ);
                        }
                    }
                } else {
                    self.stack.pop();
                    if self.finished.visit(nx) {
                        // Second time: All reachable nodes must have been finished
                        return Some(nx.into());
                    }
                }
            }

            if self.finished.count_ones(..) != graph.as_ref().node_count() {
                while let Some(cs) = self.starts.pop() {
                    if !self.discovered.is_visited(&cs) {
                        self.stack.push(cs);
                        continue 'outer;
                    }
                }
            }

            break;
        }
        None
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct PostOrder(Traversal);

impl PostOrder {
    pub fn new<V, E>(graph: &EntityGraph<V, E>) -> Self
    where
        V: Clone,
    {
        Self(Traversal::new_with(graph, Direction::Outgoing, |g| {
            g.root_vertices()
        }))
    }

    pub fn new_with<V, E, I>(graph: &EntityGraph<V, E>, roots: I) -> Self
    where
        V: Clone,
        I: IntoIterator<Item = VertexIndex<V>>,
    {
        Self(Traversal::new_with(graph, Direction::Outgoing, |_| {
            roots.into_iter().collect()
        }))
    }

    pub fn next<V, E>(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>>
    where
        V: Clone,
    {
        self.0.next(graph)
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct RevPostOrder(Traversal);

impl RevPostOrder {
    pub fn new<V, E>(graph: &EntityGraph<V, E>) -> Self
    where
        V: Clone,
    {
        Self(Traversal::new_with(graph, Direction::Incoming, |g| {
            g.leaf_vertices()
        }))
    }

    pub fn new_with<V, E, I>(graph: &EntityGraph<V, E>, leaves: I) -> Self
    where
        V: Clone,
        I: IntoIterator<Item = VertexIndex<V>>,
    {
        Self(Traversal::new_with(graph, Direction::Incoming, |_| {
            leaves.into_iter().collect()
        }))
    }

    pub fn next<V, E>(&mut self, graph: &EntityGraph<V, E>) -> Option<VertexIndex<V>>
    where
        V: Clone,
    {
        self.0.next(graph)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use fugue::ir::il::ecode::{Entity, Location};
    use fugue::ir::{AddressSpace, AddressValue};

    #[test]
    fn post_order_traversal() {
        let mut graph = EntityGraph::<&'static str, ()>::new();

        let spc = AddressSpace::constant("nodes", 0);
        let mk_ent = |name: &'static str, loc: usize| -> Entity<&'static str> {
            Entity::new("node", Location::new(AddressValue::new(&spc, 0), loc), name)
        };

        let a = mk_ent("A", 0);
        let b = mk_ent("B", 1);
        let c = mk_ent("C", 2);
        let d = mk_ent("D", 3);
        let e = mk_ent("E", 4);
        let f = mk_ent("F", 5);
        let g = mk_ent("G", 6);

        graph.add_root_entity(&a);

        graph.add_relation(&a, &b, ());
        graph.add_relation(&a, &c, ());

        graph.add_relation(&b, &c, ());

        graph.add_relation(&b, &d, ());

        graph.add_relation(&c, &b, ());

        graph.add_relation(&d, &e, ());
        graph.add_relation(&d, &a, ());

        graph.add_relation(&e, &b, ());
        graph.add_relation(&e, &f, ());
        graph.add_relation(&f, &g, ());
        graph.add_relation(&g, &f, ());

        let po = graph
            .post_order()
            .into_iter()
            .map(|vx| *graph.entity(vx).value())
            .collect::<Vec<_>>();

        assert_eq!(po, ["C", "G", "F", "E", "D", "B", "A"]);
    }

    #[test]
    fn reverse_post_order_traversal() {
        let mut graph = EntityGraph::<&'static str, ()>::new();
        let mut graph2 = EntityGraph::<&'static str, ()>::new();

        let spc = AddressSpace::constant("nodes", 0);
        let mk_ent = |name: &'static str, loc: usize| -> Entity<&'static str> {
            Entity::new("node", Location::new(AddressValue::new(&spc, 0), loc), name)
        };

        let a = mk_ent("A", 0);
        let b = mk_ent("B", 1);
        let c = mk_ent("C", 2);
        let d = mk_ent("D", 3);
        let e = mk_ent("E", 4);
        let f = mk_ent("F", 5);
        let g = mk_ent("G", 6);

        graph.add_root_entity(&a);
        graph.add_leaf_entity(&g);

        graph.add_relation(&a, &b, ());
        graph.add_relation(&a, &c, ());

        graph.add_relation(&b, &c, ());

        graph.add_relation(&b, &d, ());

        graph.add_relation(&c, &b, ());

        graph.add_relation(&d, &e, ());
        graph.add_relation(&d, &a, ());

        graph.add_relation(&e, &b, ());
        graph.add_relation(&e, &f, ());
        graph.add_relation(&f, &g, ());
        graph.add_relation(&g, &f, ());

        graph2.add_root_entity(&g);
        graph2.add_leaf_entity(&a);

        graph2.add_relation(&b, &a, ());
        graph2.add_relation(&c, &a, ());

        graph2.add_relation(&c, &b, ());

        graph2.add_relation(&d, &b, ());

        graph2.add_relation(&b, &c, ());

        graph2.add_relation(&e, &d, ());
        graph2.add_relation(&a, &d, ());

        graph2.add_relation(&b, &e, ());
        graph2.add_relation(&f, &e, ());
        graph2.add_relation(&g, &f, ());
        graph2.add_relation(&f, &g, ());

        let mut rpo = RevPostOrder::new(&graph);
        let mut vxs = Vec::new();
        while let Some(vx) = rpo.next(&graph) {
            vxs.push(*graph.entity(vx).value());
        }

        let po = graph2.post_order().into_iter().map(|vx| *graph2.entity(vx).value()).collect::<Vec<_>>();

        // where C is chosen before B or B is chosen before C
        let valid1 = ["A", "C", "B", "D", "E", "F", "G"];
        let valid2 = ["A", "B", "D", "E", "F", "G", "C"];

        assert!(vxs == valid1 || vxs == valid2);
        assert!(po == valid1 || po == valid2);
    }
}
