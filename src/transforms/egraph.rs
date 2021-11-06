use crate::models::Block;
use crate::traits::Visit;

use std::borrow::Cow;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::{AddressSpace, AddressValue, Translator};
use fugue::ir::il::ecode::{BinOp, BinRel, BranchTarget, Cast, Expr, Location, Stmt, UnOp, UnRel, Var};

use egg::{define_language, EGraph, Id, Symbol};
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

        "cast-bool" = CastBool(Id),
        "cast-float" = CastFloat([Id; 2]),
        "cast-signed" = CastSigned([Id; 2]),
        "cast-unsigned" = CastUnsigned([Id; 2]),

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
        "call" = Call(Id),
        "return" = Return(Id),

        "location" = Location([Id; 3]), // address * position * space

        "constant" = Constant([Id; 2]), // value * size
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

    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        egg::merge_if_different(to, to.as_ref().cloned().or(from))
    }

    fn make(egraph: &EGraph<ECodeLanguage, Self>, enode: &ECodeLanguage) -> Self::Data {
        use ECodeLanguage as L;
        use std::ops::{Neg, Not};
        use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr};

        let bv = |eid: &Id| egraph[*eid].data.as_ref();
        match enode {
            // const -> bitvec
            L::Constant([val, bits]) => {
                let val = egraph[*val].leaves().find_map(|v| v.bitvec().cloned())?;
                let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                Some(val.cast(bits))
            },

            // casts
            L::CastBool(val) => Some(if bv(val)?.is_zero() {
                BitVec::zero(8)
            } else {
                BitVec::one(8)
            }),
            L::CastSigned([val, bits]) => {
                let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                Some(bv(val)?.clone().signed().cast(bits))
            },
            L::CastUnsigned([val, bits]) => {
                let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                Some(bv(val)?.clone().unsigned().cast(bits))
            },

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
        let v = egraph[id].data
            .as_ref()
            .map(|bv| (bv.clone(), bv.bits()));

        if let Some((bvv, bits)) = v {
            let val = egraph.add(ECodeLanguage::BitVec(bvv));
            let siz = egraph.add(ECodeLanguage::Value(bits as u64));
            let cst = egraph.add(ECodeLanguage::Constant([val, siz]));
            egraph.union(id, cst);
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
            rewrite!("and-0"; "(and ?a (constant 0:1 ?sz))" => "(constant 0:1 ?sz)"),

            rewrite!("or-self"; "(or ?a ?a)" => "?a"),
            rewrite!("xor-self"; "(xor (variable ?sp ?off ?sz ?gen) (variable ?sp ?off ?sz ?gen))" => "(constant 0 ?sz)"),

            rewrite!("double-not"; "(not (not ?a))" => "?a"),

            rewrite!("add-0"; "(add ?a (constant ?z ?sz))" => "?a"),
            rewrite!("sub-0"; "(sub ?a (constant ?z ?sz))" => "?a"),
            rewrite!("mul-0"; "(mul ?a (constant ?z ?sz))" => "(constant 0:1 ?sz)"),

            rewrite!("eq-t0"; "(eq ?a ?a)" => "(constant 1:8 8)"),

            rewrite!("slt-f0"; "(slt ?a ?a)" => "(constant 0:8 8)"),

            rewrite!("lt-f1"; "(lt ?a ?a)" => "(constant 0:8 8)"),

            rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
            rewrite!("and-comm"; "(and ?a ?b)" => "(and ?b ?a)"),
            rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        ];

        Self {
            graph: Default::default(),
            stmts: Vec::new(),
            rules,
            last_id: None,
        }
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

        let mut ex = Extractor::new(&self.graph, AstSize);
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

impl<'ecode> Visit<'ecode> for Rewriter<'ecode> {
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

        let val = self.graph.add(L::BitVec(bv.clone())); // TODO: this could be a Cow<'_, BitVec>
        let siz = self.graph.add(L::Value(bv.bits() as u64));

        self.add(L::Constant([val, siz]))
    }

    fn visit_expr_cast(&mut self, expr: &'ecode Expr, cast: &'ecode Cast) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();

        match cast {
            Cast::Bool => {
                self.add(L::CastBool(exid))
            },
            Cast::Signed(bits) => {
                let size = self.graph.add(L::Value(*bits as u64));
                self.add(L::CastSigned([exid, size]))
            },
            Cast::Unsigned(bits) => {
                let size = self.graph.add(L::Value(*bits as u64));
                self.add(L::CastUnsigned([exid, size]))
            },
            Cast::Float(ref fmt) => {
                let size = self.graph.add(L::Value(fmt.bits() as u64));
                self.add(L::CastFloat([exid, size]))
            },
            Cast::High(bits) => {
                let bitsid = self.graph.add(L::Value(*bits as u64));
                self.add(L::ExtractHigh([exid, bitsid]))
            },
            Cast::Low(bits) => {
                let bitsid = self.graph.add(L::Value(*bits as u64));
                self.add(L::ExtractLow([exid, bitsid]))
            },
        };
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

    fn visit_expr_load(&mut self, expr: &'ecode Expr, size: usize, space: &'ecode Arc<AddressSpace>) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        let szid = self.graph.add(L::Value(size as u64));
        let spid = self.graph.add(L::Value(space.index() as u64));

        self.add(ECodeLanguage::Load([exid, szid, spid]));
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

    fn visit_stmt_store(&mut self, loc: &'ecode Expr, val: &'ecode Expr, size: usize, space: &'ecode Arc<AddressSpace>) {
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

    fn visit_stmt_call(&mut self, branch_target: &'ecode BranchTarget) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add_eff(L::Call(tid))
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

    fn into_bitvec(nodes: &RecExpr<ECodeLanguage>, i: usize) -> BitVec {
        use ECodeLanguage as L;

        if let L::BitVec(ref v) = &nodes.as_ref()[i] {
            v.clone()
        } else {
            panic!("language term at index {} is not a bitvec", i);
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
            let space = translator.manager()
                .spaces()[Self::into_value(nodes, (*spc).into()) as usize]
                .clone();

            Var::new(
                space,
                Self::into_value(nodes, (*off).into()),
                Self::into_value(nodes, (*sz).into()) as usize,
                Self::into_value(nodes, (*gen).into()) as usize,
            )
        } else {
            panic!("language term at index {} is not a variable", i);
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

            // casts
            L::CastBool(expr) => Expr::cast_bool(
                Self::into_expr_aux(translator, nodes, (*expr).into())
            ),
            L::CastFloat([expr, bits]) => Expr::cast_float(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                translator.float_format(Self::into_value(nodes, (*bits).into()) as usize)
                    .expect("valid float-format")
            ),
            L::CastSigned([expr, bits]) => Expr::cast_signed(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*bits).into()) as usize,
            ),
            L::CastUnsigned([expr, bits]) => Expr::cast_unsigned(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*bits).into()) as usize,
            ),

            // misc
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
            L::ExtractLow([expr, bits]) => Expr::extract_high(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*bits).into()) as usize,
            ),
            L::Extract([expr, lsb, msb]) => Expr::extract(
                Self::into_expr_aux(translator, nodes, (*expr).into()),
                Self::into_value(nodes, (*lsb).into()) as usize,
                Self::into_value(nodes, (*msb).into()) as usize,
            ),
            L::Intrinsic(parts) => {
                let name = Self::into_name(nodes, parts[0].into());
                let size = Self::into_value(nodes, parts[1].into()) as usize;

                Expr::intrinsic(
                    name.into(),
                    parts[2..].iter().map(|arg| Self::into_expr_aux(translator, nodes, (*arg).into())),
                    size,
                )
            },

            // primitives
            L::Constant([val, sz]) => {
                let val = Self::into_bitvec(nodes, (*val).into());
                let sz = Self::into_value(nodes, (*sz).into()) as usize;
                val.cast(sz).into()
            },
            L::Variable(_) => Self::into_variable(translator, nodes, i).into(),

            _ => panic!("language term at index {} is not an expression", i)
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
                    .clone();

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
                    name.into(),
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
            L::Call(bt) => Stmt::Call(
                Self::into_branch_target(translator, nodes, (*bt).into()),
            ),
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

        let mut ex = Extractor::new(&self.graph, AstSize);
        let (_, rec) = ex.find_best(exid.clone());

        Self::into_expr_aux(translator, &rec, rec.as_ref().len() - 1)
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
            *old.value_mut() = new;
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
