use crate::traits::Visit;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::AddressSpace;
use fugue::ir::il::ecode::{self as il, Cast, Location, Var};

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
        "extract" = Extract([Id; 3]),

        "concat" = Concat([Id; 2]),
        "intrinsic" = Intrinsic(Vec<Id>),

        // "assign" = Assign([Id; 2]),
        "store" = Store([Id; 4]), // to * from * bits * space
        "load" = Load([Id; 3]), // from * bits * space
        "phi" = Phi(Vec<Id>),

        "skip" = Skip,

        "branch" = Branch(Id),
        "cbranch" = CBranch([Id; 2]),
        "call" = Call(Id),
        "return" = Return(Id),

        "location" = Location([Id; 3]), // address * position * space

        "constant" = Constant([Id; 2]), // value * size
        "variable" = Variable([Id; 4]), // value

        Value(u64), // for now we restrict BVs to 64-bit
        Name(Symbol),
    }
}

impl ECodeLanguage {
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
                let val = egraph[*val].leaves().find_map(|v| v.value())?;
                let bits = egraph[*bits].leaves().find_map(|v| v.value())? as usize;
                Some(BitVec::from_u64(val, bits))
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
            }
            // TODO: L::Extract([val, msb, lsb]) => { }

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
            .and_then(|bv| bv.to_u64().map(|bvv| (bvv, bv.bits())));

        if let Some((bvv, bits)) = v {
            let val = egraph.add(ECodeLanguage::Value(bvv));
            let siz = egraph.add(ECodeLanguage::Value(bits as u64));
            let cst = egraph.add(ECodeLanguage::Constant([val, siz]));
            egraph.union(id, cst);
        }
    }
}

pub struct ECodeRewriter<'ecode> {
    graph: EGraph<ECodeLanguage, ConstantFolding>,
    stmts: Vec<(Option<&'ecode Var>, Id, RecExpr<ECodeLanguage>)>,
    rules: Vec<Rewrite<ECodeLanguage, ConstantFolding>>,
    last_id: Option<Id>, // stack ids for child nodes
}

impl<'ecode> Default for ECodeRewriter<'ecode> {
    fn default() -> Self {
        let rules: Vec<Rewrite<ECodeLanguage, ConstantFolding>> = vec![
            rewrite!("and-self"; "(and ?a ?a)" => "?a"),
            rewrite!("and-0"; "(and ?a (constant 0 ?sz))" => "(constant 0 ?sz)"),

            rewrite!("or-self"; "(or ?a ?a)" => "?a"),
            rewrite!("xor-self"; "(xor (variable ?sp ?off ?sz ?gen) (variable ?sp ?off ?sz ?gen))" => "(constant 0 ?sz)"),

            rewrite!("double-not"; "(not (not ?a))" => "?a"),

            rewrite!("add-0"; "(add ?a (constant 0 ?sz))" => "?a"),
            rewrite!("sub-0"; "(sub ?a (constant 0 ?sz))" => "?a"),
            rewrite!("mul-0"; "(mul ?a (constant 0 ?sz))" => "(constant 0 ?sz)"),
            rewrite!("mul-1"; "(mul ?a 1)" => "?a"),

            rewrite!("eq-t0"; "(eq ?a ?a)" => "(constant 1 8)"),

            rewrite!("slt-f0"; "(slt ?a ?a)" => "(constant 0 8)"),

            rewrite!("lt-f0"; "(lt ?a 0)" => "(constant 0 8)"),
            rewrite!("lt-f1"; "(lt ?a ?a)" => "(constant 0 8)"),

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

impl<'ecode> ECodeRewriter<'ecode> {
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
        self.last_id.insert(id);
    }
}

impl<'ecode> Visit<'ecode> for ECodeRewriter<'ecode> {
    fn visit_expr_var(&mut self, var: &'ecode il::Var) {
        use ECodeLanguage as L;

        let space = self.graph.add(L::Value(var.space().index() as u64));
        let offset = self.graph.add(L::Value(var.offset()));
        let size = self.graph.add(L::Value(var.bits() as u64));
        let generation = self.graph.add(L::Value(var.generation() as u64));

        self.add(L::Variable([space, offset, size, generation]));
    }

    fn visit_expr_val(&mut self, bv: &'ecode BitVec) {
        use ECodeLanguage as L;

        let val = self.graph.add(L::Value(bv.to_u64().unwrap_or(0)));
        let siz = self.graph.add(L::Value(bv.bits() as u64));

        self.add(L::Constant([val, siz]))
    }

    fn visit_expr_cast(&mut self, expr: &'ecode il::Expr, cast: &'ecode Cast) {
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

    fn visit_expr_unrel(&mut self, op: il::UnRel, expr: &'ecode il::Expr) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        match op {
            il::UnRel::NAN => self.add(L::NaN(exid)),
        }
    }

    fn visit_expr_unop(&mut self, op: il::UnOp, expr: &'ecode il::Expr) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        match op {
            il::UnOp::NOT => self.add(L::Not(exid)),
            il::UnOp::NEG => self.add(L::Neg(exid)),
            il::UnOp::ABS => self.add(L::Abs(exid)),
            il::UnOp::SQRT => self.add(L::Sqrt(exid)),
            il::UnOp::FLOOR => self.add(L::Floor(exid)),
            il::UnOp::ROUND => self.add(L::Round(exid)),
            il::UnOp::CEILING => self.add(L::Ceiling(exid)),
            il::UnOp::POPCOUNT => self.add(L::PopCount(exid)),
        }
    }

    fn visit_expr_binrel(&mut self, op: il::BinRel, lexpr: &'ecode il::Expr, rexpr: &'ecode il::Expr) {
        use ECodeLanguage as L;

        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        let args = [lexid, rexid];

        match op {
            il::BinRel::EQ => self.add(L::Eq(args)),
            il::BinRel::NEQ => self.add(L::NotEq(args)),
            il::BinRel::LT => self.add(L::Less(args)),
            il::BinRel::LE => self.add(L::LessEq(args)),
            il::BinRel::SLT => self.add(L::SLess(args)),
            il::BinRel::SLE => self.add(L::SLessEq(args)),
            il::BinRel::CARRY => self.add(L::Carry(args)),
            il::BinRel::SCARRY => self.add(L::SCarry(args)),
            il::BinRel::SBORROW => self.add(L::SBorrow(args)),
        }
    }

    fn visit_expr_binop(&mut self, op: il::BinOp, lexpr: &'ecode il::Expr, rexpr: &'ecode il::Expr) {
        use ECodeLanguage as L;

        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        let args = [lexid, rexid];

        match op {
            il::BinOp::AND => self.add(L::And(args)),
            il::BinOp::OR => self.add(L::Or(args)),
            il::BinOp::XOR => self.add(L::Xor(args)),
            il::BinOp::ADD => self.add(L::Add(args)),
            il::BinOp::SUB => self.add(L::Sub(args)),
            il::BinOp::MUL => self.add(L::Mul(args)),
            il::BinOp::DIV => self.add(L::Div(args)),
            il::BinOp::SDIV => self.add(L::SDiv(args)),
            il::BinOp::REM => self.add(L::Rem(args)),
            il::BinOp::SREM => self.add(L::SRem(args)),
            il::BinOp::SHL => self.add(L::Shl(args)),
            il::BinOp::SHR => self.add(L::Shr(args)),
            il::BinOp::SAR => self.add(L::Sar(args)),
        }
    }

    fn visit_expr_load(&mut self, expr: &'ecode il::Expr, size: usize, space: &'ecode Arc<AddressSpace>) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        let szid = self.graph.add(L::Value(size as u64));
        let spid = self.graph.add(L::Value(space.index() as u64));

        self.add(ECodeLanguage::Load([exid, szid, spid]));
    }

    fn visit_expr_extract(&mut self, expr: &'ecode il::Expr, lsb: usize, msb: usize) {
        use ECodeLanguage as L;

        self.visit_expr(expr);
        let exid = self.take_id();
        let lsbid = self.graph.add(L::Value(lsb as u64));
        let msbid = self.graph.add(L::Value(msb as u64));

        self.add(ECodeLanguage::Extract([exid, lsbid, msbid]));
    }

    fn visit_expr_concat(&mut self, lexpr: &'ecode il::Expr, rexpr: &'ecode il::Expr) {
        use ECodeLanguage as L;

        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        self.add(L::Concat([lexid, rexid]))
    }

    fn visit_expr_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Box<il::Expr>], bits: usize) {
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

    fn visit_stmt_phi(&mut self, var: &'ecode il::Var, vars: &'ecode [il::Var]) {
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

    fn visit_stmt_assign(&mut self, var: &'ecode il::Var, expr: &'ecode il::Expr) {
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

    fn visit_stmt_store(&mut self, loc: &'ecode il::Expr, val: &'ecode il::Expr, size: usize, space: &'ecode Arc<AddressSpace>) {
        use ECodeLanguage as L;

        self.visit_expr(loc);
        let locid = self.take_id();

        self.visit_expr(val);
        let valid = self.take_id();

        let szid = self.graph.add(L::Value(size as u64));
        let spid = self.graph.add(L::Value(space.index() as u64));

        self.add_eff(L::Store([locid, valid, szid, spid]));
    }

    fn visit_stmt_intrinsic(&mut self, name: &'ecode str, args: &'ecode [il::Expr]) {
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

    fn visit_stmt_branch(&mut self, branch_target: &'ecode il::BranchTarget) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add_eff(L::Branch(tid))
    }

    fn visit_stmt_cbranch(&mut self, cond: &'ecode il::Expr, branch_target: &'ecode il::BranchTarget) {
        use ECodeLanguage as L;

        self.visit_expr(cond);
        let exid = self.take_id();

        self.visit_branch_target(branch_target);
        let tid = self.take_id();

        self.add_eff(L::CBranch([exid, tid]))
    }

    fn visit_stmt_call(&mut self, branch_target: &'ecode il::BranchTarget) {
        use ECodeLanguage as L;

        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add_eff(L::Call(tid))
    }

    fn visit_stmt_return(&mut self, branch_target: &'ecode il::BranchTarget) {
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

impl<'ecode> ECodeRewriter<'ecode> {
    pub fn new(rules: Vec<Rewrite<ECodeLanguage, ConstantFolding>>) -> Self {
        Self {
            graph: Default::default(),
            stmts: Vec::new(),
            rules,
            last_id: None,
        }
    }

    pub fn push_statement(&mut self, stmt: &'ecode il::Stmt) {
        self.last_id = None;
        self.visit_stmt(stmt);
    }

    pub fn statements(&self) -> impl Iterator<Item=(Option<&'ecode Var>, &RecExpr<ECodeLanguage>)> {
        self.stmts.iter().map(|(v, _, e)| (*v, e))
    }

    pub fn simplify(&mut self) {
        let egraph = std::mem::take(&mut self.graph);
        let mut runner = Runner::default()
            .with_egraph(egraph);

        runner.roots.extend(self.stmts.iter().map(|(_, id, _)| id.clone()));

        let runner = runner.run(self.rules.iter());
        self.graph = runner.egraph;
    }
}
