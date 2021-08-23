use std::fmt::{self, Debug, Display};
use petgraph::dot::{Dot as DotRender, Config as DotConfig};

use std::cell::RefCell;
use std::marker::PhantomData;

use fugue::ir::il::ecode::EntityId;

use crate::models::block::Block;
use crate::models::cfg::{BranchKind, CFG};
use crate::types::EntityGraph;

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
pub struct DotDisplayWith<'a, N, NR: 'a, E, ER: 'a, F, G, T>{
    g: &'a T,
    nf: RefCell<F>,
    ef: RefCell<G>,
    marker: PhantomData<(N, NR, E, ER)>,
}

pub trait AsDot<'a>: Sized {
    type N: 'a;
    type E: 'a;

    fn dot(&'a self) -> DotDisplay<'a, Self>;
    fn dot_with<NR, ER, F, G>(&'a self, nf: F, ef: G) -> DotDisplayWith<'a, Self::N, NR, Self::E, ER, F, G, Self>
        where F: FnMut(&'a Self::N) -> NR,
              G: FnMut(&'a Self::E) -> ER,
              NR: Display + 'a,
              ER: Display + 'a;
}

impl<'a, 'e> AsDot<'a> for CFG<'e> {
    type N = Block;
    type E = BranchKind;

    fn dot(&'a self) -> DotDisplay<'a, Self> {
        DotDisplay(self)
    }

    fn dot_with<NR, ER, F, G>(&'a self, nf: F, ef: G) -> DotDisplayWith<'a, Self::N, NR, Self::E, ER, F, G, Self>
        where F: FnMut(&'a Self::N) -> NR,
              G: FnMut(&'a Self::E) -> ER,
              NR: Display + 'a,
              ER: Display + 'a {
        DotDisplayWith {
            g: self,
            nf: RefCell::new(nf),
            ef: RefCell::new(ef),
            marker: PhantomData,
        }
    }
}

impl<'a, 'e> Display for DotDisplay<'a, CFG<'e>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.0
            .map(|_nx, n| {
                let block = &self.0.blocks()[n];
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

impl<'a, 'e, NR, ER, F, G> Display for DotDisplayWith<'a, Block, NR, BranchKind, ER, F, G, CFG<'e>>
where F: FnMut(&'a Block) -> NR,
      G: FnMut(&'a BranchKind) -> ER,
      NR: Display + 'a,
      ER: Display + 'a {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.g
            .map(|_nx, n| {
                let block = &self.g.blocks()[n];
                DisplayAlways(format!("{}", (self.nf.borrow_mut())(block.value())))
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

impl<'a, E: 'a> AsDot<'a> for EntityGraph<E> {
    type N = EntityId;
    type E = E;

    fn dot(&'a self) -> DotDisplay<'a, Self> {
        DotDisplay(self)
    }

    fn dot_with<NR, ER, F, G>(&'a self, nf: F, ef: G) -> DotDisplayWith<'a, Self::N, NR, Self::E, ER, F, G, Self>
        where F: FnMut(&'a EntityId) -> NR,
              G: FnMut(&'a E) -> ER,
              NR: Display + 'a,
              ER: Display + 'a {
        DotDisplayWith {
            g: self,
            nf: RefCell::new(nf),
            ef: RefCell::new(ef),
            marker: PhantomData,
        }
    }
}

impl<'a, E: Display> Display for DotDisplay<'a, EntityGraph<E>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.0
            .map(
                |_nx, n| DisplayAlways(format!("{}", n)),
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

impl<'a, NR, E, ER, F, G> Display for DotDisplayWith<'a, EntityId, NR, E, ER, F, G, EntityGraph<E>>
where F: FnMut(&'a EntityId) -> NR,
      G: FnMut(&'a E) -> ER,
      NR: Display + 'a,
      ER: Display + 'a,
      E: 'a {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g_str = self.g
            .map(
                |_nx, n| DisplayAlways(format!("{}", (self.nf.borrow_mut())(n))),
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
