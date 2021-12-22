use std::collections::BTreeMap;

use fugue::bv::BitVec;

use crate::models::Block;
use crate::graphs::entity::{AsEntityGraph, VertexIndex};
use crate::traits::stmt::*;

const BITS: usize = 600_000;

pub fn path_counting<'g, G, E>(g: &'g G) -> BTreeMap<VertexIndex<Block>, BitVec>
where G: AsEntityGraph<'g, Block, E> {
    let g = g.entity_graph();
    let order = g.post_order();

    let mut w = BTreeMap::new();

    for vx in order {
        let blk = g.entity(vx);
        if blk.last().is_return() {
            w.insert(vx, BitVec::one(BITS));
        } else if blk.last().is_call() {
            todo!()
        } else {
            let mut vw = BitVec::zero(BITS);
            for (sx, _) in g.successors(vx) {
                if let Some(sw) = w.get(&sx) {
                    vw = &vw + sw;
                }
            }
            w.insert(vx, vw);
        }
    }

    w
}
