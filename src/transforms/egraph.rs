use crate::models::Block;
use crate::traits::Visit;

use std::borrow::Cow;

use fugue::bv::BitVec;
use fugue::ir::{AddressValue, Translator};
use fugue::ir::space::AddressSpaceId;
use fugue::ir::il::ecode::{BinOp, BinRel, BranchTarget, Cast, Expr, Location, Stmt, UnOp, UnRel, Var};
use fugue::ir::il::traits::*;

use egg::{define_language, EGraph, Id, Symbol, Subst, Applier, PatternAst};
use egg::{rewrite, Analysis, AstSize, Extractor, RecExpr, Rewrite, Runner};

define_language! {
    pub enum ECodeLanguage {
        "not" = Not(Id),
        "neg" = Neg(Id),
        "abs" = Abs(Id),
        "sqrt" = Sqrt(Id),
        "ceiling" = Ceiling(Id),
        "floor" = Floor(Id),
        "round" = Round(Id),
        "pop-count" = PopCount(Id),

        "nan" = NaN(Id),

        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "xor" = Xor([Id; 2]),
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "div" = Div([Id; 2]),
        "sdiv" = SDiv([Id; 2]),
        "mul" = Mul([Id; 2]),
        "rem" = Rem([Id; 2]),
        "srem" = SRem([Id; 2]),
        "shl" = Shl([Id; 2]),
        "sar" = Sar([Id; 2]),
        "shr" = Shr([Id; 2]),

        "eq" = Eq([Id; 2]),
        "neq" = NotEq([Id; 2]),
        "lt" = Less([Id; 2]),
        "le" = LessEq([Id; 2]),
        "slt" = SLess([Id; 2]),
        "sle" = SLessEq([Id; 2]),

        "sborrow" = SBorrow([Id; 2]),
        "carry" = Carry([Id; 2]),
        "scarry" = SCarry([Id; 2]),

        "cast" = Cast([Id; 2]),

        "void" = CastVoid,
        "bool" = CastBool,
        "float" = CastFloat(Id),
        "signed" = CastSigned(Id),
        "unsigned" = CastUnsigned(Id),
        "pointer" = CastPtr([Id; 2]), // typ * sz
        "function" = CastFn(Vec<Id>), // [rtyp, ptyp1, ..., ptypN]
        "named" = CastNamed([Id; 2]), // name * sz

        "extract-high" = ExtractHigh([Id; 2]),
        "extract-low" = ExtractLow([Id; 2]),
        "extract" = Extract([Id; 3]), // expr * lsb * msb

        "concat" = Concat([Id; 2]),

        "if" = IfElse([Id; 3]),

        "intrinsic" = Intrinsic(Vec<Id>),

        // "assign" = Assign([Id; 2]),
        "store" = Store([Id; 4]), // to * from * bits * space
        "load" = Load([Id; 3]), // from * bits * space

        "skip" = Skip,

        "branch" = Branch(Id),
        "cbranch" = CBranch([Id; 2]),
        "call" = Call(Vec<Id>),
        "return" = Return(Id),

        "location" = Location([Id; 3]), // address * position * space
        "variable" = Variable([Id; 4]), // value

        BitVec(BitVec),
        Value(u64),
        Name(Symbol),
    }
}

impl ECodeLanguage {
    pub fn bitvec(&self) -> Option<&BitVec> {
        if let ECodeLanguage::BitVec(ref val) = self {
            Some(val)
        } else {
            None
        }
    }

    pub fn value(&self) -> Option<u64> {
        if let ECodeLanguage::Value(val) = self {
            Some(*val)
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct ConstantFolding;

impl Analysis<ECodeLanguage> for ConstantFolding {
    type Data = Option<BitVec>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        if from != *to {
            *to = from;
            egg::DidMerge(true, false)
        } else {
            egg::DidMerge(false, false)
        }
    }

    fn make(egraph: &EGraph<ECodeLanguage, Self>, enode: &ECodeLanguage) -> Self::Data {
        use ECodeLanguage as L;
        use std::ops::{Neg, Not};
        use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};

        let bv = |eid: &Id| egraph[*eid].data.as_ref();
        match enode {
            // const -> bitvec
            L::BitVec(bv) => Some(bv.clone()),
            L::Cast([val, cast]) => match egraph[*cast].leaves().next() {
                None => None,
                Some(kind) => match kind {
                    L::CastBool => Some(if bv(val)?.is_zero() {
                        BitVec::zero(8)
                    } else {
                        BitVec::one(8)
                    }),
                    L::CastSigned(bits) => {
                        let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                        Some(bv(val)?.clone().signed().cast(bits))
                    },
                    L::CastUnsigned(bits) => {
                        let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                        Some(bv(val)?.clone().unsigned().cast(bits))
                    },
                    L::CastPtr([_, bits]) => {
                        let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                        Some(bv(val)?.clone().unsigned().cast(bits))
                    },
                    _ => None,
                }
            }

            // bit-ops
            L::Concat([lhs, rhs]) => {
                let lhs = bv(lhs)?;
                let rhs = bv(rhs)?;

                let sz = lhs.bits() + rhs.bits();

                let lv = lhs.clone().unsigned().cast(sz).shl(rhs.bits() as u32);
                let rv = rhs.clone().unsigned().cast(sz);

                Some(lv.bitor(rv))
            },
            L::Extract([val, lsb, msb]) => {
                let val = bv(val)?;
                let lsb = egraph[*lsb].leaves().find_map(|v| v.value())? as u32;
                let msb = egraph[*msb].leaves().find_map(|v| v.value())? as u32;

                if (msb - lsb) as usize == val.bits() {
                    Some(val.clone())
                } else {
                    Some(if lsb > 0 {
                        (val >> lsb).unsigned_cast((msb - lsb) as usize)
                    } else {
                        val.unsigned_cast((msb - lsb) as usize)
                    })
                }
            },
            L::ExtractHigh([val, bits]) => {
                let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                let bvv = bv(val)?;

                if bvv.bits() > bits {
                    Some(bvv.shr(bvv.bits() as u32 - bits as u32).cast(bits))
                } else {
                    Some(bvv.clone().unsigned().cast(bits))
                }
            },
            L::ExtractLow([val, bits]) => {
                let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                Some(bv(val)?.clone().unsigned().cast(bits))
            },

            // unary ops
            L::Neg(val) => Some(bv(val)?.neg()),
            L::Not(val) => Some(bv(val)?.not()),
            L::PopCount(val) => Some(bv(val)?.count_ones().into()),

            // binary ops
            L::And([lhs, rhs]) => Some(bv(lhs)?.bitand(bv(rhs)?)),
            L::Add([lhs, rhs]) => Some(bv(lhs)?.add(bv(rhs)?)),
            L::Sub([lhs, rhs]) => Some(bv(lhs)?.sub(bv(rhs)?)),
            L::Div([lhs, rhs]) => {
                let rhs = bv(rhs)?;
                if rhs.is_zero() {
                    None
                } else {
                    Some(bv(lhs)?.div(rhs))
                }
            },
            L::Mul([lhs, rhs]) => Some(bv(lhs)?.mul(bv(rhs)?)),
            L::Or([lhs, rhs]) => Some(bv(lhs)?.bitor(bv(rhs)?)),
            L::Rem([lhs, rhs]) => {
                let rhs = bv(rhs)?;
                if rhs.is_zero() {
                    None
                } else {
                    Some(bv(lhs)?.rem(rhs))
                }
            },
            L::Sar([lhs, rhs]) => Some(bv(lhs)?.signed_shr(bv(rhs)?)),
            L::SDiv([lhs, rhs]) => {
                let rhs = bv(rhs)?;
                if rhs.is_zero() {
                    None
                } else {
                    Some(bv(lhs)?.signed_div(rhs))
                }
            },
            L::SRem([lhs, rhs]) => {
                let rhs = bv(rhs)?;
                if rhs.is_zero() {
                    None
                } else {
                    Some(bv(lhs)?.signed_rem(rhs))
                }
            },
            L::Shl([lhs, rhs]) => Some(bv(lhs)?.shl(bv(rhs)?)),
            L::Shr([lhs, rhs]) => Some(bv(lhs)?.shr(bv(rhs)?)),
            L::Xor([lhs, rhs]) => Some(bv(lhs)?.bitxor(bv(rhs)?)),

            L::Carry([lhs, rhs]) => Some((bv(lhs)?.signed_borrow(bv(rhs)?) as u8).into()),
            L::SBorrow([lhs, rhs]) => Some((bv(lhs)?.carry(bv(rhs)?) as u8).into()),
            L::SCarry([lhs, rhs]) => Some((bv(lhs)?.signed_carry(bv(rhs)?) as u8).into()),

            // bin-op equality
            L::Eq([lhs, rhs]) => Some((bv(lhs)?.cmp(bv(rhs)?).is_eq() as u8).into()),
            L::NotEq([lhs, rhs]) => Some((bv(lhs)?.cmp(bv(rhs)?).is_ne() as u8).into()),
            L::Less([lhs, rhs]) => Some((bv(lhs)?.cmp(bv(rhs)?).is_lt() as u8).into()),
            L::LessEq([lhs, rhs]) => Some((bv(lhs)?.cmp(bv(rhs)?).is_le() as u8).into()),
            L::SLess([lhs, rhs]) => Some((bv(lhs)?.signed_cmp(bv(rhs)?).is_lt() as u8).into()),
            L::SLessEq([lhs, rhs]) => Some((bv(lhs)?.signed_cmp(bv(rhs)?).is_le() as u8).into()),
            _ => None,
        }
    }

    fn modify(egraph: &mut EGraph<ECodeLanguage, Self>, id: Id) {
        let v = egraph[id].data.clone();
        if let Some(bv) = v {
            let val = egraph.add(ECodeLanguage::BitVec(bv));
            egraph.union(id, val);
        }
    }
}

pub struct Rewriter<'ecode> {
    graph: EGraph<ECodeLanguage, ConstantFolding>,
    stmts: Vec<(Option<&'ecode Var>, Id, RecExpr<ECodeLanguage>)>,
    rules: Vec<Rewrite<ECodeLanguage, ConstantFolding>>,
    last_id: Option<Id>, // stack ids for child nodes
}

impl<'ecode> Default for Rewriter<'ecode> {
    fn default() -> Self {
        let rules: Vec<Rewrite<ECodeLanguage, ConstantFolding>> = vec![
            rewrite!("and-self"; "(and ?a ?a)" => "?a"),
            rewrite!("or-self"; "(or ?a ?a)" => "?a"),
            rewrite!("xor-self"; "(xor ?a ?a)" => { CancelIdentity {
                a: "?a".parse().unwrap(),
                v: 0,
            }}),

            rewrite!("double-neg"; "(neg (neg ?a))" => "?a"),
            rewrite!("double-not"; "(not (not ?a))" => "?a"),

            rewrite!("eq-t"; "(eq ?a ?a)" => "1:8"),
            rewrite!("neq-t"; "(neq ?a ?a)" => "0:8"),

            rewrite!("le-t"; "(le ?a ?a)" => "1:8"),
            rewrite!("sle-t"; "(sle ?a ?a)" => "1:8"),

            rewrite!("lt-f"; "(lt ?a ?a)" => "0:8"),
            rewrite!("slt-f"; "(slt ?a ?a)" => "0:8"),

            rewrite!("sub-add"; "(sub ?a ?b)" => "(add ?a (neg ?b))"),
            rewrite!("add-sub"; "(add ?a (neg ?b))" => "(sub ?a ?b)"),

            rewrite!("add-assoc"; "(add ?a (add ?b ?c))" => "(add (add ?a ?b) ?c)"),
            rewrite!("mul-assoc"; "(mul ?a (mul ?b ?c))" => "(mul (mul ?a ?b) ?c)"),

            rewrite!("and-assoc"; "(and ?a (and ?b ?c))" => "(and (and ?a ?b) ?c)"),
            rewrite!("or-assoc"; "(or ?a (or ?b ?c))" => "(or (or ?a ?b) ?c)"),
            rewrite!("xor-assoc"; "(xor ?a (xor ?b ?c))" => "(xor (xor ?a ?b) ?c)"),

            rewrite!("add-id"; "(add ?a ?b)" => "?a" if is_const("?b", |bv| bv.is_zero())),

            rewrite!("mul-id"; "(mul ?a ?b)" => "?a" if is_const("?b", |bv| bv.is_one())),
            rewrite!("mul-zero"; "(mul ?a ?b)" => "?b" if is_const("?b", |bv| bv.is_zero())),

            rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
            rewrite!("and-comm"; "(and ?a ?b)" => "(and ?b ?a)"),
            rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
            rewrite!("xor-comm"; "(xor ?a ?b)" => "(xor ?b ?a)"),

            rewrite!("cast-unsigned-nop"; "(cast (cast ?a (unsigned ?sz1)) (unsigned ?sz2))" => "(cast ?a (unsigned ?sz2))"
                     if bits_cmp("?a", "?sz1", |l, r| l <= r)
                     if bits_cmp("?sz1", "?sz2", |l, r| l <= r)),

            rewrite!("extract-low0"; "(extract-low ?a ?sz)" => "?a" if bits_eq("?a", "?sz")),
            rewrite!("extract-high0"; "(extract-high ?a ?sz)" => "?a" if bits_eq("?a", "?sz")),

            rewrite!("cast-extract-nop"; "(extract-low (cast ?a (unsigned ?sz1)) ?sz2)" => "(cast ?a (unsigned ?sz2))"
                     if bits_cmp("?a", "?sz2", |l, r| l <= r)
                     if bits_cmp("?sz1", "?sz2", |l, r| l >= r)),

            rewrite!("extract-low1"; "(extract-low (concat ?a ?b) ?sz)" => "?b" if bits_eq("?b", "?sz")),
            rewrite!("extract-high1"; "(extract-high (concat ?a ?b) ?sz)" => "?a" if bits_eq("?a", "?sz")),
        ];

        Self {
            graph: Default::default(),
            stmts: Vec::new(),
            rules,
            last_id: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CancelIdentity {
    a: egg::Var,
    v: u64,
}

impl Applier<ECodeLanguage, ConstantFolding> for CancelIdentity {

    fn apply_one(&self, egraph: &mut EGraph<ECodeLanguage, ConstantFolding>, matched_id: Id, subst: &Subst, _: Option<&PatternAst<ECodeLanguage>>, _: Symbol) -> Vec<Id> {
        use ECodeLanguage as L;

        let a = subst[self.a];
        if let Some(bits) = lexpr_bits(egraph, a) {
            let val = egraph.add(L::BitVec(BitVec::from_u64(self.v, bits as usize)));
            if egraph.union(matched_id, val) {
                vec![val]
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }
}

fn lexpr_bits(nodes: &EGraph<ECodeLanguage, ConstantFolding>, id: Id) -> Option<u64> {
    use ECodeLanguage as L;

    let node = &nodes[id].nodes[0];
    let value = |v| nodes[v].nodes[0].value();

    match node {
        L::Value(sz) => Some(*sz),
        L::BitVec(bv) => Some(bv.bits() as u64),
        //L::Constant([_, sz]) |
        L::Variable([_, _, sz, _]) |
        L::Load([_, sz, _]) |
        L::ExtractHigh([_, sz]) |
        L::ExtractLow([_, sz]) => value(*sz),
        L::Extract([_, ms, ls]) => value(*ms).and_then(|mv| value(*ls).map(|lv| mv - lv)),
        L::Cast([_, c]) => match &nodes[*c].nodes[0] {
            L::CastNamed([_, sz]) |
            L::CastSigned(sz) |
            L::CastUnsigned(sz) |
            L::CastPtr([_, sz]) |
            L::CastFloat(sz) => value(*sz),
            L::CastBool => Some(8),
            _ => None,
        },
        L::And([v, _]) |
        L::Or([v, _]) |
        L::Xor([v, _]) |
        L::Add([v, _]) |
        L::Sub([v, _]) |
        L::Div([v, _]) |
        L::SDiv([v, _]) |
        L::Mul([v, _]) |
        L::Rem([v, _]) |
        L::SRem([v, _]) |
        L::Shl([v, _]) |
        L::Sar([v, _]) |
        L::Shr([v, _]) |
        L::IfElse([_, v, _]) => lexpr_bits(nodes, *v),
        L::Not(v) |
        L::Neg(v) |
        L::Abs(v) |
        L::Sqrt(v) |
        L::Ceiling(v) |
        L::Floor(v) |
        L::Round(v) => lexpr_bits(nodes, *v),
        L::Eq(_) |
        L::NotEq(_) |
        L::Less(_) |
        L::LessEq(_) |
        L::SLess(_) |
        L::SLessEq(_) |
        L::SBorrow(_) |
        L::Carry(_) |
        L::SCarry(_) => Some(8),
        L::Concat([l, r]) => lexpr_bits(nodes, *l).and_then(|ls| lexpr_bits(nodes, *r).map(|rs| ls + rs)),
        _ => None,
    }
}

fn is_const<F>(var1: &'static str, f: F) -> impl Fn(&mut EGraph<ECodeLanguage, ConstantFolding>, Id, &Subst) -> bool
where F: Fn(&BitVec) -> bool {
    let var1 = var1.parse().unwrap();

    move |egraph, _, subst| {
        egraph[subst[var1]].nodes.iter().any(|v| if let ECodeLanguage::BitVec(ref bv) = v {
            f(bv)
        } else {
            false
        })
    }
}

fn bits_eq(var1: &'static str, var2: &'static str) -> impl Fn(&mut EGraph<ECodeLanguage, ConstantFolding>, Id, &Subst) -> bool {
    bits_cmp(var1, var2, |l, r| l == r)
}

fn bits_cmp<F>(var1: &'static str, var2: &'static str, cmp: F) -> impl Fn(&mut EGraph<ECodeLanguage, ConstantFolding>, Id, &Subst) -> bool
where F: Fn(u64, u64) -> bool {
    let var1 = var1.parse().unwrap();
    let var2 = var2.parse().unwrap();

    move |egraph, _, subst| {
        let lsz = lexpr_bits(egraph, subst[var1]);
        let rsz = lexpr_bits(egraph, subst[var2]);
        matches!((lsz, rsz), (Some(ls), Some(rs)) if cmp(ls, rs))
    }
}

impl<'ecode> Rewriter<'ecode> {
    #[inline(always)]
    fn add(&mut self, e: ECodeLanguage) {
        let id = self.graph.add(e);
        self.insert_id(id);
    }

    #[inline(always)]
    fn add_eff(&mut self, e: ECodeLanguage) {
        let oid = self.graph.add(e);
        self.simplify();

        let ex = Extractor::new(&self.graph, AstSize);
        let (_, rec) = ex.find_best(oid.clone());

        self.stmts.push((None, oid.clone(), rec));

        self.insert_id(oid)
    }

    #[inline(always)]
    fn take_id(&mut self) -> Id {
        self.last_id.take().unwrap()
    }

    #[inline(always)]
    fn insert_id(&mut self, id: Id) {
        self.last_id = Some(id);
    }
}

impl<'ecode> Visit<'ecode, Location, BitVec, Var> for Rewriter<'ecode> {
    fn visit_expr_var(&mut self, var: &'ecode Var) {
        use ECodeLanguage as L;

        let space = self.graph.add(L::Value(var.space().index() as u64));
        let offset = self.graph.add(L::Value(var.offset()));
        let size = self.graph.add(L::Value(var.bits() as u64));
        let generation = self.graph.add(L::Value(var.generation() as u64));

        self.add(L::Variable([space, offset, size, generation]));
    }

    fn visit_expr_val(&mut self, bv: &'ecode BitVec) {
        use ECodeLanguage as L;

        /*let val =*/ self.add(L::BitVec(bv.clone())); // TODO: this could be a Cow<'_, BitVec>
        //let siz = self.graph.add(L::Value(bv.bits() as u64));

        //self.add(L::Constant([val, siz]))
    }

    fn visit_cast(&mut self, cast: &'ecode Cast) {
        use ECodeLanguage as L;

        match cast {
            Cast::Void => {
                self.add(L::CastVoid)
            },
            Cast::Bool => {
                self.add(L::CastBool)
            },
            Cast::Signed(bits) => {
                let size = self.graph.add(L::Value(*bits as u64));
                self.add(L::CastSigned(size))
            },
            Cast::Unsigned(bits) => {
                let size = self.graph.add(L::Value(*bits as u64));
                self.add(L::CastUnsigned(size))
            },
            Cast::Float(ref fmt) => {
                let size = self.graph.add(L::Value(fmt.bits() as u64));
                self.add(L::CastFloat(size))
            },
            Cast::Pointer(typ, bits) => {
                self.visit_cast(typ);
                let tid = self.take_id();
                let bitsid = self.graph.add(L::Value(*bits as u64));
                self.add(L::CastPtr([tid, bitsid]))
            },
            Cast::Function(rtyp, ptyps) => {
                self.visit_cast(rtyp);
                let mut args = vec![self.take_id()];
                for ptyp in ptyps {
                    self.visit_cast(ptyp);
                    args.push(self.take_id());
                }
                self.add(L::CastFn(args))
            },
            Cast::Named(name, bits) => {
                let symid = self.graph.add(L::Name(name.into()));
                let bitsid = self.graph.add(L::Value(*bits as u64));
                self.add(L::CastNamed([symid, bitsid]))
            },
        }
    }

    fn visit_expr_cast(&mut self, expr: &'ecode Expr, cast: &'ecode Cast) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();

        self.visit_cast(cast);
        let tid = self.take_id();

        self.add(L::Cast([exid, tid]))
    }

    fn visit_expr_unrel(&mut self, op: UnRel, expr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        match op {
            UnRel::NAN => self.add(L::NaN(exid)),
        }
    }

    fn visit_expr_unop(&mut self, op: UnOp, expr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        match op {
            UnOp::NOT => self.add(L::Not(exid)),
            UnOp::NEG => self.add(L::Neg(exid)),
            UnOp::ABS => self.add(L::Abs(exid)),
            UnOp::SQRT => self.add(L::Sqrt(exid)),
            UnOp::FLOOR => self.add(L::Floor(exid)),
            UnOp::ROUND => self.add(L::Round(exid)),
            UnOp::CEILING => self.add(L::Ceiling(exid)),
            UnOp::POPCOUNT => self.add(L::PopCount(exid)),
        }
    }

    fn visit_expr_binrel(&mut self, op: BinRel, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        let args = [lexid, rexid];

        match op {
            BinRel::EQ => self.add(L::Eq(args)),
            BinRel::NEQ => self.add(L::NotEq(args)),
            BinRel::LT => self.add(L::Less(args)),
            BinRel::LE => self.add(L::LessEq(args)),
            BinRel::SLT => self.add(L::SLess(args)),
            BinRel::SLE => self.add(L::SLessEq(args)),
            BinRel::CARRY => self.add(L::Carry(args)),
            BinRel::SCARRY => self.add(L::SCarry(args)),
            BinRel::SBORROW => self.add(L::SBorrow(args)),
        }
    }

    fn visit_expr_binop(&mut self, op: BinOp, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        let args = [lexid, rexid];

        match op {
            BinOp::AND => self.add(L::And(args)),
            BinOp::OR => self.add(L::Or(args)),
            BinOp::XOR => self.add(L::Xor(args)),
            BinOp::ADD => self.add(L::Add(args)),
            BinOp::SUB => self.add(L::Sub(args)),
            BinOp::MUL => self.add(L::Mul(args)),
            BinOp::DIV => self.add(L::Div(args)),
            BinOp::SDIV => self.add(L::SDiv(args)),
            BinOp::REM => self.add(L::Rem(args)),
            BinOp::SREM => self.add(L::SRem(args)),
            BinOp::SHL => self.add(L::Shl(args)),
            BinOp::SHR => self.add(L::Shr(args)),
            BinOp::SAR => self.add(L::Sar(args)),
        }
    }

    fn visit_expr_load(&mut self, expr: &'ecode Expr, size: usize, space: AddressSpaceId) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        let szid = self.graph.add(L::Value(size as u64));
        let spid = self.graph.add(L::Value(space.index() as u64));

        self.add(ECodeLanguage::Load([exid, szid, spid]));
    }

    fn visit_expr_extract_low(&mut self, expr: &'ecode Expr, bits: usize) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();

        let bitsid = self.graph.add(L::Value(bits as u64));

        self.add(L::ExtractLow([exid, bitsid]))
    }

    fn visit_expr_extract_high(&mut self, expr: &'ecode Expr, bits: usize) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();

        let bitsid = self.graph.add(L::Value(bits as u64));

        self.add(L::ExtractHigh([exid, bitsid]))
    }

    fn visit_expr_extract(&mut self, expr: &'ecode Expr, lsb: usize, msb: usize) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        let lsbid = self.graph.add(L::Value(lsb as u64));
        let msbid = self.graph.add(L::Value(msb as u64));

        self.add(ECodeLanguage::Extract([exid, lsbid, msbid]));
    }

    fn visit_expr_concat(&mut self, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        self.add(L::Concat([lexid, rexid]))
    }

    fn visit_expr_ite(&mut self, cond: &'ecode Expr, texpr: &'ecode Expr, fexpr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(cond);
        let cid = self.take_id();

        self.visit_expr(texpr);
        let lexid = self.take_id();

        self.visit_expr(fexpr);
        let rexid = self.take_id();

        self.add(L::IfElse([cid, lexid, rexid]))
    }

    fn visit_expr_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Box<Expr>], bits: usize) {
        use ECodeLanguage as L;

        let mut ids = vec![
            self.graph.add(L::Name(name.into())),
            self.graph.add(L::Value(bits as u64)),
        ];

        for arg in args {
            self.visit_expr(arg);
            ids.push(self.take_id());
        }

        self.add(L::Intrinsic(ids));
    }

    fn visit_expr_call(&mut self, branch_target: &'ecode BranchTarget, args: &'ecode [Box<Expr>], bits: usize) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();

        let mut ids = vec![
            tid,
            self.graph.add(L::Value(bits as u64)),
        ];

        for arg in args {
            self.visit_expr(arg);
            ids.push(self.take_id());
        }

        self.add(L::Intrinsic(ids));
    }

    /*
    fn visit_stmt_phi(&mut self, var: &'ecode Var, vars: &'ecode [Var]) {
        use ECodeLanguage as L;

        let mut varids = Vec::new();
        for v in vars {
            self.visit_expr_var(v);
            varids.push(self.take_id());
        }

        let exid = self.graph.add(L::Phi(varids));

        self.simplify();

        let mut ex = Extractor::new(&self.graph, AstSize);
        let (_, rec) = ex.find_best(exid.clone());

        self.stmts.push((Some(var), exid.clone(), rec));

        let space = self.graph.add(L::Value(var.space().index() as u64));
        let offset = self.graph.add(L::Value(var.offset()));
        let size = self.graph.add(L::Value(var.bits() as u64));
        let generation = self.graph.add(L::Value(var.generation() as u64));

        let var = self.graph.add(L::Variable([space, offset, size, generation]));

        self.graph.union(var.clone(), exid);
        self.graph.rebuild();

        self.insert_id(var)
    }
    */

    fn visit_stmt_assign(&mut self, var: &'ecode Var, expr: &'ecode Expr) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();

        self.simplify();

        let ex = Extractor::new(&self.graph, AstSize);
        let (_, rec) = ex.find_best(exid.clone());

        self.stmts.push((Some(var), exid.clone(), rec));

        let space = self.graph.add(L::Value(var.space().index() as u64));
        let offset = self.graph.add(L::Value(var.offset()));
        let size = self.graph.add(L::Value(var.bits() as u64));
        let generation = self.graph.add(L::Value(var.generation() as u64));

        let var = self.graph.add(L::Variable([space, offset, size, generation]));

        self.graph.union(var.clone(), exid);
        self.graph.rebuild();

        self.insert_id(var)
    }

    fn visit_stmt_store(&mut self, loc: &'ecode Expr, val: &'ecode Expr, size: usize, space: AddressSpaceId) {
        use ECodeLanguage as L;

        self.visit_expr(loc);
        let locid = self.take_id();

        self.visit_expr(val);
        let valid = self.take_id();

        let szid = self.graph.add(L::Value(size as u64));
        let spid = self.graph.add(L::Value(space.index() as u64));

        self.add_eff(L::Store([locid, valid, szid, spid]));
    }

    fn visit_stmt_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Expr]) {
        use ECodeLanguage as L;

        let mut ids = vec![
            self.graph.add(L::Name(name.into())),
            self.graph.add(L::Value(0)),
        ];

        for arg in args {
            self.visit_expr(arg);
            ids.push(self.take_id());
        }

        self.add_eff(ECodeLanguage::Intrinsic(ids));
    }

    fn visit_branch_target_location(&mut self, location: &'ecode Location) {
        use ECodeLanguage as L;

        let addr = self.graph.add(L::Value(location.address().offset()));
        let pos = self.graph.add(L::Value(location.position() as u64));
        let spid = self.graph.add(L::Value(location.address().space().index() as u64));

        self.add(L::Location([addr, pos, spid]))
    }

    fn visit_stmt_branch(&mut self, branch_target: &'ecode BranchTarget) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add_eff(L::Branch(tid))
    }

    fn visit_stmt_cbranch(&mut self, cond: &'ecode Expr, branch_target: &'ecode BranchTarget) {
        use ECodeLanguage as L;

        self.visit_expr(cond);
        let exid = self.take_id();

        self.visit_branch_target(branch_target);
        let tid = self.take_id();

        self.add_eff(L::CBranch([exid, tid]))
    }

    fn visit_stmt_call(&mut self, branch_target: &'ecode BranchTarget, args: &'ecode [Expr]) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();

        let mut ids = vec![
            tid,
            self.graph.add(L::Value(0)),
        ];

        for arg in args {
            self.visit_expr(arg);
            ids.push(self.take_id());
        }

        self.add_eff(L::Call(ids))
    }

    fn visit_stmt_return(&mut self, branch_target: &'ecode BranchTarget) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add_eff(L::Return(tid))
    }

    fn visit_stmt_skip(&mut self) {
        use ECodeLanguage as L;
        self.add_eff(L::Skip);
    }
}

impl<'ecode> Rewriter<'ecode> {
    fn into_name(nodes: &RecExpr<ECodeLanguage>, i: usize) -> Cow<str> {
        use ECodeLanguage as L;

        if let L::Name(v) = &nodes.as_ref()[i] {
            Cow::from(v.as_str())
        } else {
            panic!("language term at index {} is not a name", i);
        }
    }

    fn into_value(nodes: &RecExpr<ECodeLanguage>, i: usize) -> u64 {
        use ECodeLanguage as L;

        if let L::Value(v) = &nodes.as_ref()[i] {
            *v
        } else {
            panic!("language term at index {} is not a value", i);
        }
    }

    fn into_location(translator: &'ecode Translator, nodes: &RecExpr<ECodeLanguage>, i: usize) -> Location {
        use ECodeLanguage as L;

        if let L::Location([off, pos, spc]) = &nodes.as_ref()[i] {
            let off = Self::into_value(nodes, (*off).into());
            let pos = Self::into_value(nodes, (*pos).into()) as usize;
            let spc = Self::into_value(nodes, (*spc).into()) as usize;

            let space = translator.manager().spaces()[spc].clone();
            let address = AddressValue::new(space, off);

            Location::new(address, pos)
        } else {
            panic!("language term at index {} is not a location", i);
        }
    }

    fn into_branch_target(translator: &'ecode Translator, nodes: &RecExpr<ECodeLanguage>, i: usize) -> BranchTarget {
        use ECodeLanguage as L;

        let node = &nodes.as_ref()[i];
        match node {
            L::Location(_) => Self::into_location(translator, nodes, i).into(),
            _ => BranchTarget::computed(Self::into_expr_aux(translator, nodes, i))
        }
    }

    fn into_variable(translator: &'ecode Translator, nodes: &RecExpr<ECodeLanguage>, i: usize) -> Var {
        use ECodeLanguage as L;

        if let L::Variable([spc, off, sz, gen]) = &nodes.as_ref()[i] {
            let space = &translator.manager()
                .spaces()[Self::into_value(nodes, (*spc).into()) as usize];

            Var::new(
                &**space,
                Self::into_value(nodes, (*off).into()),
                Self::into_value(nodes, (*sz).into()) as usize,
                Self::into_value(nodes, (*gen).into()) as usize,
            )
        } else {
            panic!("language term at index {} is not a variable", i);
        }
    }

    fn into_cast(translator: &'ecode Translator, nodes: &RecExpr<ECodeLanguage>, i: usize) -> Cast {
        use ECodeLanguage as L;

        let node = &nodes.as_ref()[i];
        match node {
            // casts
            L::CastVoid => Cast::Void,
            L::CastBool => Cast::Bool,
            L::CastFloat(bits) => {
                let fmt = translator.float_format(Self::into_value(nodes, (*bits).into()) as usize)
                    .expect("valid float-format");
                Cast::Float(fmt)
            },
            L::CastSigned(bits) => Cast::Signed(Self::into_value(nodes, (*bits).into()) as usize),
            L::CastUnsigned(bits) => Cast::Unsigned(Self::into_value(nodes, (*bits).into()) as usize),
            L::CastPtr([cast, bits]) => {
                let bits = Self::into_value(nodes, (*bits).into()) as usize;
                let cast = Self::into_cast(translator, nodes, (*cast).into());
                Cast::Pointer(Box::new(cast), bits)
            },
            L::CastFn(typs) => {
                let rtyp = Self::into_cast(translator, nodes, typs[0].into());
                let atyps = typs[1..].iter().map(|typ| Box::new(Self::into_cast(translator, nodes, (*typ).into()))).collect();
                Cast::Function(Box::new(rtyp), atyps)
            }
            L::CastNamed([name, bits]) => {
                let bits = Self::into_value(nodes, (*bits).into()) as usize;
                let name = Self::into_name(nodes, (*name).into());
                Cast::Named(name.as_ref().into(), bits)
            }
            _ => panic!("language term {} at index {} is not a cast", node, i)
        }
    }

    fn into_expr_aux(translator: &'ecode Translator, nodes: &RecExpr<ECodeLanguage>, i: usize) -> Expr {
        use ECodeLanguage as L;

        let node = &nodes.as_ref()[i];
        match node {
            // unary
            L::Not(expr) => Expr::UnOp(
                UnOp::NOT,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::Neg(expr) => Expr::UnOp(
                UnOp::NEG,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::Abs(expr) => Expr::UnOp(
                UnOp::ABS,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::Sqrt(expr) => Expr::UnOp(
                UnOp::SQRT,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::Ceiling(expr) => Expr::UnOp(
                UnOp::CEILING,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::Floor(expr) => Expr::UnOp(
                UnOp::FLOOR,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::Round(expr) => Expr::UnOp(
                UnOp::ROUND,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::PopCount(expr) => Expr::UnOp(
                UnOp::POPCOUNT,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),
            L::NaN(expr) => Expr::UnRel(
                UnRel::NAN,
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
            ),

            // binary
            L::And([lhs, rhs]) => Expr::BinOp(
                BinOp::AND,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Or([lhs, rhs]) => Expr::BinOp(
                BinOp::OR,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Xor([lhs, rhs]) => Expr::BinOp(
                BinOp::XOR,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Add([lhs, rhs]) => Expr::BinOp(
                BinOp::ADD,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Sub([lhs, rhs]) => Expr::BinOp(
                BinOp::SUB,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Div([lhs, rhs]) => Expr::BinOp(
                BinOp::DIV,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::SDiv([lhs, rhs]) => Expr::BinOp(
                BinOp::SDIV,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Mul([lhs, rhs]) => Expr::BinOp(
                BinOp::MUL,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Rem([lhs, rhs]) => Expr::BinOp(
                BinOp::REM,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::SRem([lhs, rhs]) => Expr::BinOp(
                BinOp::SREM,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Shl([lhs, rhs]) => Expr::BinOp(
                BinOp::SHL,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Sar([lhs, rhs]) => Expr::BinOp(
                BinOp::SAR,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Shr([lhs, rhs]) => Expr::BinOp(
                BinOp::SHR,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),

            L::Eq([lhs, rhs]) => Expr::BinRel(
                BinRel::EQ,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::NotEq([lhs, rhs]) => Expr::BinRel(
                BinRel::NEQ,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Less([lhs, rhs]) => Expr::BinRel(
                BinRel::LT,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::LessEq([lhs, rhs]) => Expr::BinRel(
                BinRel::LE,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::SLess([lhs, rhs]) => Expr::BinRel(
                BinRel::SLT,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::SLessEq([lhs, rhs]) => Expr::BinRel(
                BinRel::SLE,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),

            L::SBorrow([lhs, rhs]) => Expr::BinRel(
                BinRel::SBORROW,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::Carry([lhs, rhs]) => Expr::BinRel(
                BinRel::CARRY,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::SCarry([lhs, rhs]) => Expr::BinRel(
                BinRel::SCARRY,
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),

            L::Cast([expr, typ]) => Expr::Cast(
                Box::new(Self::into_expr_aux(translator, nodes, (*expr).into())),
                Self::into_cast(translator, nodes, (*typ).into())
            ),

            // misc
            L::Load([src, sz, spc]) => {
                let space = translator.manager()
                    .spaces()[Self::into_value(nodes, (*spc).into()) as usize]
                    .id();

                Expr::Load(
                    Box::new(Self::into_expr_aux(translator, nodes, (*src).into())),
                    Self::into_value(nodes, (*sz).into()) as usize,
                    space,
                )
            },
            L::IfElse([cond, texpr, fexpr]) => Expr::IfElse(
                Box::new(Self::into_expr_aux(translator, nodes, (*cond).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*texpr).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*fexpr).into())),
            ),
            L::Concat([lhs, rhs]) => Expr::Concat(
                Box::new(Self::into_expr_aux(translator, nodes, (*lhs).into())),
                Box::new(Self::into_expr_aux(translator, nodes, (*rhs).into())),
            ),
            L::ExtractHigh([expr, bits]) => Expr::extract_high(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*bits).into()) as usize,
            ),
            L::ExtractLow([expr, bits]) => Expr::extract_low(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*bits).into()) as usize,
            ),
            L::Extract([expr, lsb, msb]) => Expr::extract(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*lsb).into()) as usize,
                Self::into_value(nodes, (*msb).into()) as usize,
            ),
            L::Call(parts) => {
                let branch_target = Self::into_branch_target(translator, nodes, parts[0].into());
                let size = Self::into_value(nodes, parts[1].into()) as usize;

                Expr::call_with(
                    branch_target,
                    parts[2..].iter().map(|arg| Self::into_expr_aux(translator, nodes, (*arg).into())),
                    size,
                )
            },
            L::Intrinsic(parts) => {
                let name = Self::into_name(nodes, parts[0].into());
                let size = Self::into_value(nodes, parts[1].into()) as usize;

                Expr::intrinsic(
                    name.as_ref(),
                    parts[2..].iter().map(|arg| Self::into_expr_aux(translator, nodes, (*arg).into())),
                    size,
                )
            },

            // primitives
            L::BitVec(bv) => bv.clone().into(),
            /*
            L::Constant([val, sz]) => {
                let val = Self::into_bitvec(nodes, (*val).into());
                let sz = Self::into_value(nodes, (*sz).into()) as usize;
                val.cast(sz).into()
            },
            */
            L::Variable(_) => Self::into_variable(translator, nodes, i).into(),

            _ => panic!("language term {} at index {} is not an expression", node, i)
        }
    }

    fn into_stmt(translator: &'ecode Translator, output: Option<&'ecode Var>, nodes: &RecExpr<ECodeLanguage>) -> Stmt {
        use ECodeLanguage as L;

        let i = &nodes.as_ref().len() - 1;
        let node = &nodes.as_ref()[i];
        match node {
            L::Store([tgt, src, sz, spc]) => {
                let space = translator.manager()
                    .spaces()[Self::into_value(nodes, (*spc).into()) as usize]
                    .id();

                Stmt::Store(
                    Self::into_expr_aux(translator, nodes, (*tgt).into()),
                    Self::into_expr_aux(translator, nodes, (*src).into()),
                    Self::into_value(nodes, (*sz).into()) as usize,
                    space,
                )
            },
            L::Intrinsic(parts) => {
                assert!(output.is_none());

                let name = Self::into_name(nodes, parts[0].into());

                Stmt::intrinsic(
                    &*name,
                    parts[2..].iter().map(|arg| Self::into_expr_aux(translator, nodes, (*arg).into())),
                )
            },
            L::Branch(bt) => Stmt::Branch(
                Self::into_branch_target(translator, nodes, (*bt).into()),
            ),
            L::CBranch([c, bt]) => Stmt::CBranch(
                Self::into_expr_aux(translator, nodes, (*c).into()),
                Self::into_branch_target(translator, nodes, (*bt).into()),
            ),
            L::Call(parts) => {
                let target = Self::into_branch_target(translator, nodes, parts[0].into());
                Stmt::call_with(
                    target,
                    parts[2..].iter().map(|arg| Self::into_expr_aux(translator, nodes, (*arg).into())),
                )
            },
            L::Return(bt) => Stmt::Return(
                Self::into_branch_target(translator, nodes, (*bt).into()),
            ),
            L::Skip => Stmt::Skip,
            _ => if let Some(var) = output {
                let expr = Self::into_expr_aux(translator, nodes, i);
                Stmt::Assign(var.clone(), expr)
            } else {
                panic!("language term at index {} is not a statement", i)
            }
        }
    }
}

impl<'ecode> Rewriter<'ecode> {
    pub fn new(rules: Vec<Rewrite<ECodeLanguage, ConstantFolding>>) -> Self {
        Self {
            graph: Default::default(),
            stmts: Vec::new(),
            rules,
            last_id: None,
        }
    }

    pub fn push(&mut self, stmt: &'ecode Stmt) {
        self.last_id = None;
        self.visit_stmt(stmt);
    }

    pub fn simplify_expr(&mut self, translator: &'ecode Translator, expr: &'ecode Expr) -> Expr {
        self.last_id = None;
        self.visit_expr(expr);

        let exid = self.take_id();
        self.simplify();

        let ex = Extractor::new(&self.graph, AstSize);
        let (_, rec) = ex.find_best(exid.clone());

        let nexpr = Self::into_expr_aux(translator, &rec, rec.as_ref().len() - 1);

        nexpr
    }

    pub fn operations(&self) -> impl Iterator<Item=(Option<&'ecode Var>, &RecExpr<ECodeLanguage>)> {
        self.stmts.iter().map(|(v, _, e)| (*v, e))
    }

    pub fn extract(&self, translator: &'ecode Translator) -> Vec<Stmt> {
        self.stmts.iter()
            .map(|(v, _, e)| Self::into_stmt(translator, *v, e))
            .collect()
    }

    pub fn extract_into(&self, translator: &'ecode Translator, block: &'ecode mut Block) {
        let it = block.operations_mut()
            .iter_mut()
            .zip(self.stmts.iter().map(|(v, _, e)| Self::into_stmt(translator, *v, e)));

        for (old, new) in it {
            **old.value_mut() = new;
        }
    }

    fn simplify(&mut self) {
        let egraph = std::mem::take(&mut self.graph);
        let mut runner = Runner::default()
            .with_egraph(egraph);

        runner.roots.extend(self.stmts.iter().map(|(_, id, _)| id.clone()));

        let runner = runner.run(self.rules.iter());
        self.graph = runner.egraph;
    }
}

#[cfg(test)]
mod test {
    use crate::analyses::expressions::symbolic::{SymExprs, SymPropFold};
    use crate::models::{Lifter, Project};
    use crate::traits::Substitutor;
    use crate::traits::oracle::database_oracles;
    use crate::types::EntityIdMapping;
    use fugue::db::Database;
    use fugue::ir::LanguageDB;

    #[test]
    fn test_failure() -> Result<(), Box<dyn std::error::Error>> {

        let ldb = LanguageDB::from_directory_with("./processors", true)?;
        let db = Database::from_file("./tests/lojax.fdb", &ldb)?;

        let translator = db.default_translator();
        let convention = translator.compiler_conventions()["windows"].clone();
        let lifter = Lifter::new(translator, convention);

        let mut project = Project::new("lojax", lifter);
        let (bo, fo) = database_oracles(&db);

        project.set_block_oracle(bo);
        project.set_function_oracle(fo);

        for seg in db.segments().values() {
            if seg.is_code() && !seg.is_external() {
                project.add_region_mapping_with(
                    seg.name(),
                    seg.address(),
                    seg.endian(),
                    seg.bytes(),
                );
            }
        }

        let sample2 = db.function("_ModuleEntryPoint").unwrap();
        let fid = project.add_function(sample2.address()).unwrap();

        let sample2f = project.lookup_by_id(fid).unwrap();

        let mut cfg = sample2f.cfg_with(&*project, &*project);
        let mut prop = SymExprs::new(project.lifter().translator());

        cfg.propagate_expressions(&mut prop);

        let mut subst = Substitutor::new(prop.propagator());
        subst.apply_graph(&mut cfg);

        Ok(())
    }
}
