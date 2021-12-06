use std::fmt::{self, Debug, Display};
use petgraph::dot::{Dot as DotRender, Config as DotConfig};

use std::cell::RefCell;
use std::marker::PhantomData;

use crate::graphs::entity::{AsEntityGraph, EntityGraph};
use crate::models::block::Block;
use crate::models::cfg::{BranchKind, CFG};
use crate::types::{Id, Identifiable};

// NOTE: this is a hack to get petgraph to render blocks nicely
struct DisplayAlways<T: Display>(T);

impl<T> Debug for DisplayAlways<T> where T: Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> Display for DisplayAlways<T> where T: Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct DotDisplay<'a, T>(&'a T);
pub struct DotDisplayWith<'a, V, VR: 'a, E, ER: 'a, F, G, T>{
    g: &'a T,
    nf: RefCell<F>,
    ef: RefCell<G>,
    marker: PhantomData<(V, VR, E, ER)>,
}

pub trait AsDot<'a>: Sized {
    type V: 'a;
    type E: 'a;

    fn dot(&'a self) -> DotDisplay<'a, Self>;
    fn dot_with<VR, ER, F, G>(&'a self, nf: F, ef: G) -> DotDisplayWith<'a, Self::V, VR, Self::E, ER, F, G, Self>
        where F: FnMut(Id<Self::V>, &'a Self::V) -> VR,
              G: FnMut(&'a Self::E) -> ER,
              VR: Display + 'a,
              ER: Display + 'a;
}

impl<'a, 'e> AsDot<'a> for CFG<'e, Block> {
    type V = Block;
    type E = BranchKind;

    fn dot(&'a self) -> DotDisplay<'a, Self> {
        DotDisplay(self)
    }

    fn dot_with<VR, ER, F, G>(&'a self, nf: F, ef: G) -> DotDisplayWith<'a, Self::V, VR, Self::E, ER, F, G, Self>
        where F: FnMut(Id<Self::V>, &'a Self::V) -> VR,
              G: FnMut(&'a Self::E) -> ER,
              VR: Display + 'a,
              ER: Display + 'a {
        DotDisplayWith {
            g: self,
            nf: RefCell::new(nf),
            ef: RefCell::new(ef),
            marker: PhantomData,
        }
    }
}

impl<'a, 'e> Display for DotDisplay<'a, CFG<'e, Block>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.0
            .entity_graph()
            .as_ref()
            .map(|nx, _n| {
                let block = self.0.entity(nx.into());
                DisplayAlways(format!("{}", block.value()))
            },
            |_ex, e| e);

        let dot = DotRender::with_attr_getters(
            &g_str,
            &[DotConfig::EdgeNoLabel],
            &|_, _| "".to_owned(),
            &|_, _| "shape=box".to_owned());

        write!(f, "{}", dot)
    }
}

impl<'a, 'e, VR, ER, F, G> Display for DotDisplayWith<'a, Block, VR, BranchKind, ER, F, G, CFG<'e, Block>>
where F: FnMut(Id<Block>, &'a Block) -> VR,
      G: FnMut(&'a BranchKind) -> ER,
      VR: Display + 'a,
      ER: Display + 'a {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.g
            .entity_graph()
            .as_ref()
            .map(|nx, _n| {
                let block = &self.g.entity(nx.into());
                DisplayAlways(format!("{}", (self.nf.borrow_mut())(block.id(), block.value())))
            },
            |_ex, e| {
                DisplayAlways(format!("{}", (self.ef.borrow_mut())(e)))
            });

        let dot = DotRender::with_attr_getters(
            &g_str,
            &[],
            &|_, _| "".to_owned(),
            &|_, _| "shape=box".to_owned());

        write!(f, "{}", dot)
    }
}

impl<'a, V, E> AsDot<'a> for EntityGraph<'a, V, E>
where V: Clone + 'a,
      E: 'a {
    type V = V;
    type E = E;

    fn dot(&'a self) -> DotDisplay<'a, Self> {
        DotDisplay(self)
    }

    fn dot_with<VR, ER, F, G>(&'a self, nf: F, ef: G) -> DotDisplayWith<'a, Self::V, VR, Self::E, ER, F, G, Self>
        where F: FnMut(Id<Self::V>, &'a V) -> VR,
              G: FnMut(&'a E) -> ER,
              VR: Display + 'a,
              ER: Display + 'a {
        DotDisplayWith {
            g: self,
            nf: RefCell::new(nf),
            ef: RefCell::new(ef),
            marker: PhantomData,
        }
    }
}

impl<'a, V, E> Display for DotDisplay<'a, EntityGraph<'a, V, E>>
where V: Clone + Display + 'a,
      E: Display + 'a {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.0
            .as_ref()
            .map(
                |nx, _n| {
                    let v = self.0.entity(nx.into());
                    DisplayAlways(format!("{}", v.value()))
                },
                |_ex, e| DisplayAlways(format!("{}", e))
            );

        let dot = DotRender::with_attr_getters(
            &g_str,
            &[],
            &|_, _| "".to_owned(),
            &|_, _| "shape=box".to_owned());

        write!(f, "{}", dot)
    }
}

impl<'a, V, VR, E, ER, F, G> Display for DotDisplayWith<'a, V, VR, E, ER, F, G, EntityGraph<'a, V, E>>
where F: FnMut(Id<V>, &'a V) -> VR,
      G: FnMut(&'a E) -> ER,
      VR: Display + 'a,
      ER: Display + 'a,
      V: Clone + 'a,
      E: 'a {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.g
            .as_ref()
            .map(
                |nx, _n| {
                    let v = self.g.entity(nx.into());
                    DisplayAlways(format!("{}", (self.nf.borrow_mut())(v.id(), v.value())))
                },
                |_ex, e| DisplayAlways(format!("{}", (self.ef.borrow_mut())(e))),
            );

        let dot = DotRender::with_attr_getters(
            &g_str,
            &[],
            &|_, _| "".to_owned(),
            &|_, _| "shape=box".to_owned());

        write!(f, "{}", dot)
    }
}
