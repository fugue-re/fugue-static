use crate::traits::Visit;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::AddressSpace;
use fugue::ir::il::ecode::{self as il, Cast, Location};

use std::num::ParseIntError;
use egg::{define_language, EGraph, Id, Symbol};
use egg::{rewrite, AstSize, Extractor, RecExpr, Rewrite, Runner};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    Bool,
    Signed(usize),
    Unsigned(usize),
    Float(usize),
}


impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => write!(f, "bool"),
            Self::Signed(bits) => write!(f, "signed:{}", bits),
            Self::Unsigned(bits) => write!(f, "unsigned:{}", bits),
            Self::Float(bits) => write!(f, "float:{})", bits),
        }
    }
}

#[derive(Debug, Error)]
pub enum ParseTypeError {
    #[error(transparent)]
    Size(#[from] ParseIntError),
    #[error("invalid type")]
    InvalidType,
}

impl FromStr for Type {
    type Err = ParseTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(if s == "bool" {
            Self::Bool
        } else if let Some(n) = s.strip_prefix("signed:") {
            Self::Signed(n.parse::<usize>()?)
        } else if let Some(n) = s.strip_prefix("unsigned:") {
            Self::Unsigned(n.parse::<usize>()?)
        } else if let Some(n) = s.strip_prefix("float:") {
            Self::Float(n.parse::<usize>()?)
        } else {
            return Err(ParseTypeError::InvalidType)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    space: usize, // index
    offset: u64,
    bits: usize,
    generation: usize,
}

impl From<&il::Var> for Var {
    fn from(v: &il::Var) -> Self {
        Self {
            space: v.space().index(),
            offset: v.offset(),
            bits: v.bits(),
            generation: v.generation(),
        }
    }
}

impl Var {
    fn into_var_tuple(&self) -> (usize, u64, usize) {
        (self.space, self.offset, self.generation)
    }
}

#[derive(Debug, Error)]
pub enum ParseVarError {
    #[error("invalid variable generation")]
    InvalidGeneration(#[source] ParseIntError),
    #[error("invalid variable offset")]
    InvalidOffset(#[source] ParseIntError),
    #[error("invalid variable bits")]
    InvalidBits(#[source] ParseIntError),
    #[error("invalid variable space")]
    InvalidSpace(#[source] ParseIntError),
    #[error("invalid variable")]
    Invalid,
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "var:{}:{}:{}:{}", self.space, self.offset, self.bits, self.generation)
    }
}

impl FromStr for Var {
    type Err = ParseVarError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.strip_prefix("var:").ok_or(ParseVarError::Invalid)?;

        let mut parts = s.split(":");

        let space = parts.next()
            .ok_or(ParseVarError::Invalid)?
            .parse::<usize>()
            .map_err(ParseVarError::InvalidSpace)?;
        let offset = parts.next()
            .ok_or(ParseVarError::Invalid)?
            .parse::<u64>()
            .map_err(ParseVarError::InvalidOffset)?;
        let bits = parts.next()
            .ok_or(ParseVarError::Invalid)?
            .parse::<usize>()
            .map_err(ParseVarError::InvalidBits)?;
        let generation = parts.next()
            .ok_or(ParseVarError::Invalid)?
            .parse::<usize>()
            .map_err(ParseVarError::InvalidGeneration)?;

        Ok(Var {
            space,
            offset,
            bits,
            generation,
        })
    }
}

define_language! {
    pub enum ELanguage {
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

        "type" = TypeCast([Id; 2]),

        "extract-high" = ExtractHigh([Id; 2]),
        "extract-low" = ExtractLow([Id; 2]),
        "extract" = Extract([Id; 3]),

        "concat" = Concat([Id; 2]),
        "intrinsic" = Intrinsic(Vec<Id>),

        "assign" = Assign([Id; 2]),
        "store" = Store([Id; 4]), // to * from * bits * space
        "load" = Load([Id; 3]), // from * bits * space

        "branch" = Branch(Id),
        "cbranch" = CBranch([Id; 2]),
        "call" = Call(Id),
        "return" = Return(Id),

        "location" = Location([Id; 3]), // address * position * space

        Var(Var),

        Bool(bool),
        BV(u64), // for now we restrict BVs to 64-bit

        Address(u64),
        Position(usize),

        Bits(usize),
        Name(Symbol),
        Space(usize),
        Type(Type),
    }
}

#[derive(Default)]
struct ELanguageVisitor {
    graph: EGraph<ELanguage, ()>,
    last_id: Option<Id>, // stack ids for child nodes
    vars: HashMap<(usize, u64, usize), (usize, Id)>,
    simple_ssa: bool,
}

impl ELanguageVisitor {
    #[inline(always)]
    fn add(&mut self, e: ELanguage) {
        let id = self.graph.add(e);
        self.insert_id(id);
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

impl<'ecode> Visit<'ecode> for ELanguageVisitor {
    fn visit_expr_var(&mut self, var: &'ecode il::Var) {
        let var = Var::from(var);
        if self.simple_ssa {
            if let Some((_, id)) = self.vars.get(&var.into_var_tuple()) {
                let id = id.clone();
                self.insert_id(id);
                return
            }
        }
        self.add(ELanguage::Var(var));
    }

    fn visit_expr_val(&mut self, bv: &'ecode BitVec) {
        self.add(ELanguage::BV(bv.to_u64().unwrap_or(0)));
    }

    fn visit_expr_cast(&mut self, expr: &'ecode il::Expr, cast: &'ecode Cast) {
        self.visit_expr(expr);
        let exid = self.take_id();

        match cast {
            Cast::Bool => {
                let typid = self.graph.add(ELanguage::Type(Type::Bool));
                self.add(ELanguage::TypeCast([exid, typid]))
            },
            Cast::Signed(bits) => {
                let typid = self.graph.add(ELanguage::Type(Type::Signed(*bits)));
                self.add(ELanguage::TypeCast([exid, typid]))
            },
            Cast::Unsigned(bits) => {
                let typid = self.graph.add(ELanguage::Type(Type::Unsigned(*bits)));
                self.add(ELanguage::TypeCast([exid, typid]))
            },
            Cast::Float(ref fmt) => {
                let typid = self.graph.add(ELanguage::Type(Type::Float(fmt.bits())));
                self.add(ELanguage::TypeCast([exid, typid]))
            },
            Cast::High(bits) => {
                let bitsid = self.graph.add(ELanguage::Bits(*bits));
                self.add(ELanguage::ExtractHigh([exid, bitsid]))
            },
            Cast::Low(bits) => {
                let bitsid = self.graph.add(ELanguage::Bits(*bits));
                self.add(ELanguage::ExtractLow([exid, bitsid]))
            },
        };
    }

    fn visit_expr_unrel(&mut self, op: il::UnRel, expr: &'ecode il::Expr) {
        self.visit_expr(expr);
        let exid = self.take_id();
        match op {
            il::UnRel::NAN => self.add(ELanguage::NaN(exid)),
        }
    }

    fn visit_expr_unop(&mut self, op: il::UnOp, expr: &'ecode il::Expr) {
        self.visit_expr(expr);
        let exid = self.take_id();
        match op {
            il::UnOp::NOT => self.add(ELanguage::Not(exid)),
            il::UnOp::NEG => self.add(ELanguage::Neg(exid)),
            il::UnOp::ABS => self.add(ELanguage::Abs(exid)),
            il::UnOp::SQRT => self.add(ELanguage::Sqrt(exid)),
            il::UnOp::FLOOR => self.add(ELanguage::Floor(exid)),
            il::UnOp::ROUND => self.add(ELanguage::Round(exid)),
            il::UnOp::CEILING => self.add(ELanguage::Ceiling(exid)),
            il::UnOp::POPCOUNT => self.add(ELanguage::PopCount(exid)),
        }
    }

    fn visit_expr_binrel(&mut self, op: il::BinRel, lexpr: &'ecode il::Expr, rexpr: &'ecode il::Expr) {
        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        let args = [lexid, rexid];

        match op {
            il::BinRel::EQ => self.add(ELanguage::Eq(args)),
            il::BinRel::NEQ => self.add(ELanguage::NotEq(args)),
            il::BinRel::LT => self.add(ELanguage::Less(args)),
            il::BinRel::LE => self.add(ELanguage::LessEq(args)),
            il::BinRel::SLT => self.add(ELanguage::SLess(args)),
            il::BinRel::SLE => self.add(ELanguage::SLessEq(args)),
            il::BinRel::CARRY => self.add(ELanguage::Carry(args)),
            il::BinRel::SCARRY => self.add(ELanguage::SCarry(args)),
            il::BinRel::SBORROW => self.add(ELanguage::SBorrow(args)),
        }
    }

    fn visit_expr_binop(&mut self, op: il::BinOp, lexpr: &'ecode il::Expr, rexpr: &'ecode il::Expr) {
        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        let args = [lexid, rexid];

        match op {
            il::BinOp::AND => self.add(ELanguage::And(args)),
            il::BinOp::OR => self.add(ELanguage::Or(args)),
            il::BinOp::XOR => self.add(ELanguage::Xor(args)),
            il::BinOp::ADD => self.add(ELanguage::Add(args)),
            il::BinOp::SUB => self.add(ELanguage::Sub(args)),
            il::BinOp::MUL => self.add(ELanguage::Mul(args)),
            il::BinOp::DIV => self.add(ELanguage::Div(args)),
            il::BinOp::SDIV => self.add(ELanguage::SDiv(args)),
            il::BinOp::REM => self.add(ELanguage::Rem(args)),
            il::BinOp::SREM => self.add(ELanguage::SRem(args)),
            il::BinOp::SHL => self.add(ELanguage::Shl(args)),
            il::BinOp::SHR => self.add(ELanguage::Shr(args)),
            il::BinOp::SAR => self.add(ELanguage::Sar(args)),
        }
    }

    fn visit_expr_load(&mut self, expr: &'ecode il::Expr, size: usize, space: &'ecode Arc<AddressSpace>) {
        self.visit_expr(expr);
        let exid = self.take_id();
        let szid = self.graph.add(ELanguage::Bits(size));
        let spid = self.graph.add(ELanguage::Space(space.index()));

        self.add(ELanguage::Load([exid, szid, spid]));
    }

    fn visit_expr_extract(&mut self, expr: &'ecode il::Expr, lsb: usize, msb: usize) {
        self.visit_expr(expr);
        let exid = self.take_id();
        let lsbid = self.graph.add(ELanguage::Bits(lsb));
        let msbid = self.graph.add(ELanguage::Bits(msb));

        self.add(ELanguage::Extract([exid, lsbid, msbid]));
    }

    fn visit_expr_concat(&mut self, lexpr: &'ecode il::Expr, rexpr: &'ecode il::Expr) {
        self.visit_expr(lexpr);
        let lexid = self.take_id();

        self.visit_expr(rexpr);
        let rexid = self.take_id();

        self.add(ELanguage::Concat([lexid, rexid]))
    }

    fn visit_expr_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Box<il::Expr>], bits: usize) {
        let mut ids = vec![
            self.graph.add(ELanguage::Name(name.into())),
            self.graph.add(ELanguage::Bits(bits)),
        ];

        for arg in args {
            self.visit_expr(arg);
            ids.push(self.take_id());
        }

        self.add(ELanguage::Intrinsic(ids));
    }

    fn visit_stmt_phi(&mut self, _var: &'ecode il::Var, _vars: &'ecode [il::Var]) {
        /* ignore */
    }

    fn visit_stmt_assign(&mut self, var: &'ecode il::Var, expr: &'ecode il::Expr) {
        self.visit_expr(expr);
        let exid = self.take_id();

        let mut var = Var::from(var);

        if self.simple_ssa {
            let generation = self.vars.entry(var.into_var_tuple()).or_insert((var.generation, exid.clone()));
            generation.0 += 1;
            generation.1 = exid.clone();
            var.generation = generation.0;
        }

        let varid = self.graph.add(ELanguage::Var(var));

        self.add(ELanguage::Assign([varid, exid]))
    }

    fn visit_stmt_store(&mut self, loc: &'ecode il::Expr, val: &'ecode il::Expr, size: usize, space: &'ecode Arc<AddressSpace>) {
        self.visit_expr(loc);
        let locid = self.take_id();

        self.visit_expr(val);
        let valid = self.take_id();

        let szid = self.graph.add(ELanguage::Bits(size));
        let spid = self.graph.add(ELanguage::Space(space.index()));

        self.add(ELanguage::Store([locid, valid, szid, spid]))
    }

    fn visit_stmt_intrinsic(&mut self, name: &'ecode str, args: &'ecode [il::Expr]) {
        let mut ids = vec![
            self.graph.add(ELanguage::Name(name.into())),
            self.graph.add(ELanguage::Bits(0)),
        ];

        for arg in args {
            self.visit_expr(arg);
            ids.push(self.take_id());
        }

        self.add(ELanguage::Intrinsic(ids));
    }

    fn visit_branch_target_location(&mut self, location: &'ecode Location) {
        let addr = self.graph.add(ELanguage::Address(location.address().offset()));
        let pos = self.graph.add(ELanguage::Position(location.position()));
        let spid = self.graph.add(ELanguage::Space(location.address().space().index()));

        self.add(ELanguage::Location([addr, pos, spid]))
    }

    fn visit_stmt_branch(&mut self, branch_target: &'ecode il::BranchTarget) {
        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add(ELanguage::Branch(tid))
    }

    fn visit_stmt_cbranch(&mut self, cond: &'ecode il::Expr, branch_target: &'ecode il::BranchTarget) {
        self.visit_expr(cond);
        let exid = self.take_id();

        self.visit_branch_target(branch_target);
        let tid = self.take_id();

        self.add(ELanguage::CBranch([exid, tid]))
    }

    fn visit_stmt_call(&mut self, branch_target: &'ecode il::BranchTarget) {
        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add(ELanguage::Call(tid))
    }

    fn visit_stmt_return(&mut self, branch_target: &'ecode il::BranchTarget) {
        self.visit_branch_target(branch_target);
        let tid = self.take_id();
        self.add(ELanguage::Return(tid))
    }
}

impl ELanguage {
    pub fn from_stmt(stmt: &il::Stmt) -> (Id, EGraph<ELanguage, ()>) {
        Self::from_stmt_with(stmt, false)
    }

    pub fn from_stmt_with(stmt: &il::Stmt, ssa: bool) -> (Id, EGraph<ELanguage, ()>) {
        let mut visit = ELanguageVisitor::default();
        visit.simple_ssa = ssa;

        visit.visit_stmt(stmt);
        (visit.take_id(), visit.graph)
    }

    pub fn from_stmts<'a, I>(stmts: I) -> (Vec<Id>, EGraph<ELanguage, ()>)
    where I: Iterator<Item=&'a il::Stmt> {
        Self::from_stmts_with(stmts, false)
    }

    pub fn from_stmts_with<'a, I>(stmts: I, ssa: bool) -> (Vec<Id>, EGraph<ELanguage, ()>)
    where I: Iterator<Item=&'a il::Stmt> {
        let mut visit = ELanguageVisitor::default();
        visit.simple_ssa = ssa;

        let mut roots = Vec::default();

        for stmt in stmts {
            visit.visit_stmt(stmt);
            if let Some(id) = visit.last_id.take() {
                roots.push(id);
            }
        }

        (roots, visit.graph)
    }

    pub fn simplify(graph: EGraph<ELanguage, ()>, roots: Vec<Id>) -> Vec<RecExpr<ELanguage>> {
        let rules: Vec<Rewrite<Self, ()>> = vec![
            rewrite!("and-self"; "(and ?a ?a)" => "?a"),
            rewrite!("and-0"; "(and ?a 0)" => "0"),

            rewrite!("or-self"; "(or ?a ?a)" => "?a"),
            rewrite!("xor-self"; "(xor ?a ?a)" => "0"),

            rewrite!("double-not"; "(not (not ?a))" => "?a"),

            rewrite!("add-0"; "(add ?a 0)" => "?a"),
            rewrite!("sub-0"; "(sub ?a 0)" => "?a"),
            rewrite!("mul-0"; "(mul ?a 0)" => "0"),
            rewrite!("mul-1"; "(mul ?a 1)" => "?a"),

            rewrite!("eq-t0"; "(eq ?a ?a)" => "1"),

            rewrite!("slt-f0"; "(slt ?a ?a)" => "0"),

            rewrite!("lt-f0"; "(lt ?a 0)" => "0"),
            rewrite!("lt-f1"; "(lt ?a ?a)" => "0"),

            rewrite!("type-0"; "(type 0 ?t)" => "0"),
            rewrite!("pop-count-0"; "(pop-count 0)" => "0"),

            rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
            rewrite!("and-comm"; "(and ?a ?b)" => "(and ?b ?a)"),
            rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        ];

        let mut runner = Runner::default()
            .with_egraph(graph);

        runner.roots = roots.clone();

        let runner = runner.run(rules.iter());
        let egraph = runner.egraph;

        let mut extractor = Extractor::new(&egraph, AstSize);
        runner.roots.into_iter().map(|root| {
            let (_best_cost, best) = extractor.find_best(root);
            best
        })
        .collect()
    }
}

#[cfg(test)]
mod test {
    use egg::RecExpr;
    use super::*;

    #[test]
    fn parse_var() -> Result<(), Box<dyn std::error::Error>> {
        let var: RecExpr<ELanguage> = "var:0:1000:32:0".parse()?;
        assert_eq!(var.to_string(), "var:0:1000:32:0");
        Ok(())
    }
}
