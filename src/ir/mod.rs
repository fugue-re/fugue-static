use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::address::AddressValue;
use fugue::ir::disassembly::{IRBuilderArena, Opcode, VarnodeData};
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::pcode::{Operand, PCode, PCodeOp};
use fugue::ir::il::traits::*;
use fugue::ir::space::{AddressSpace, AddressSpaceId};
use fugue::ir::space_manager::{FromSpace, IntoSpace, SpaceManager};
use fugue::ir::Translator;

use hashcons::hashconsing::consign;
use hashcons::Term;

use fnv::FnvHashMap as Map;
use smallvec::{smallvec, SmallVec};
use ustr::Ustr;

use crate::models::{Block, Function};
use crate::types::Id;

consign! { let TRGT = consign(1024) for BranchTarget; }
consign! { let CAST = consign(1024) for Cast; }
consign! { let EXPR = consign(1024) for Expr; }
consign! { let STMT = consign(1024) for Stmt; }

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    space: AddressSpaceId,
    offset: u64,
    bits: usize,
    generation: usize,
}

impl Var {
    pub fn new<S: Into<AddressSpaceId>>(
        space: S,
        offset: u64,
        bits: usize,
        generation: usize,
    ) -> Self {
        Self {
            space: space.into(),
            offset,
            bits,
            generation,
        }
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl BitSize for Var {
    fn bits(&self) -> usize {
        self.bits
    }
}

impl Variable for Var {
    fn generation(&self) -> usize {
        self.generation
    }

    fn generation_mut(&mut self) -> &mut usize {
        &mut self.generation
    }

    fn with_generation(&self, generation: usize) -> Self {
        Self {
            space: self.space.clone(),
            generation,
            ..*self
        }
    }

    fn space(&self) -> AddressSpaceId {
        self.space
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "space[{}][{:#x}].{}:{}",
            self.space().index(),
            self.offset(),
            self.generation(),
            self.bits()
        )
    }
}

impl<'var, 'trans> fmt::Display for VarFormatter<'var, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(trans) = self.translator {
            let space = trans.manager().unchecked_space_by_id(self.var.space());
            if space.is_register() {
                let name = trans
                    .registers()
                    .get(self.var.offset(), self.var.bits() / 8)
                    .unwrap();
                write!(f, "{}.{}:{}", name, self.var.generation(), self.var.bits())
            } else {
                let off = self.var.offset();
                let sig = (off as i64).signum() as i128;
                write!(
                    f,
                    "{}[{}{:#x}].{}:{}",
                    space.name(),
                    if sig == 0 {
                        ""
                    } else if sig > 0 {
                        "+"
                    } else {
                        "-"
                    },
                    self.var.offset() as i64 as i128 * sig,
                    self.var.generation(),
                    self.var.bits()
                )
            }
        } else {
            self.fmt(f)
        }
    }
}

pub struct VarFormatter<'var, 'trans> {
    var: &'var Var,
    translator: Option<&'trans Translator>,
}

impl<'var, 'trans> TranslatorDisplay<'var, 'trans> for Var {
    type Target = VarFormatter<'var, 'trans>;

    fn display_with(
        &'var self,
        translator: Option<&'trans Translator>,
    ) -> VarFormatter<'var, 'trans> {
        VarFormatter {
            var: self,
            translator,
        }
    }
}

impl From<VarnodeData> for Var {
    fn from(vnd: VarnodeData) -> Self {
        Self {
            space: vnd.space(),
            offset: vnd.offset(),
            bits: vnd.size() * 8,
            generation: 0,
        }
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct Location {
    address: AddressValue,
    position: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.address, self.position)
    }
}

impl<'loc, 'trans> TranslatorDisplay<'loc, 'trans> for Location {
    type Target = &'loc Self;

    fn display_with(&'loc self, _translator: Option<&'trans Translator>) -> Self::Target {
        self
    }
}

impl Location {
    pub fn new<A>(address: A, position: usize) -> Location
    where
        A: Into<AddressValue>,
    {
        Self {
            address: address.into(),
            position,
        }
    }

    pub fn address(&self) -> Cow<AddressValue> {
        Cow::Borrowed(&self.address)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn space(&self) -> AddressSpaceId {
        self.address.space()
    }

    pub fn is_relative(&self) -> bool {
        self.space().is_constant() && self.position == 0
    }

    pub fn is_absolute(&self) -> bool {
        !self.is_relative()
    }

    pub fn absolute_from<A>(&mut self, address: A, position: usize)
    where
        A: Into<AddressValue>,
    {
        if self.is_absolute() {
            return;
        }

        let offset = self.address().offset() as i64;
        let position = if offset.is_negative() {
            position
                .checked_sub(offset.abs() as usize)
                .expect("negative offset from position in valid range")
        } else {
            position
                .checked_add(offset as usize)
                .expect("positive offset from position in valid range")
        };

        self.address = address.into();
        self.position = position;
    }
}

impl<'z> FromSpace<'z, VarnodeData> for Location {
    fn from_space_with(t: VarnodeData, _arena: &'z IRBuilderArena, manager: &SpaceManager) -> Self {
        Location::from_space(t, manager)
    }

    fn from_space(vnd: VarnodeData, manager: &SpaceManager) -> Self {
        Self {
            address: AddressValue::new(manager.unchecked_space_by_id(vnd.space()), vnd.offset()),
            position: 0,
        }
    }
}

impl From<AddressValue> for Location {
    fn from(address: AddressValue) -> Self {
        Self {
            address,
            position: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BranchTarget {
    Block(Id<Block>),
    Function(Id<Function>),
    Location(Location),
    Computed(Term<Expr>),
}

impl<'target, 'trans> fmt::Display for BranchTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BranchTarget::Block(id) => write!(f, "{}", id),
            BranchTarget::Function(id) => write!(f, "{}", id),
            BranchTarget::Location(loc) => write!(f, "{}", loc),
            BranchTarget::Computed(expr) => write!(f, "{}", expr),
        }
    }
}

pub struct BranchTargetFormatter<'target, 'trans> {
    target: &'target BranchTarget,
    translator: Option<&'trans Translator>,
}

impl<'target, 'trans> fmt::Display for BranchTargetFormatter<'target, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target {
            BranchTarget::Block(id) => {
                write!(f, "{}", id)
            }
            BranchTarget::Function(id) => {
                write!(f, "{}", id)
            }
            BranchTarget::Location(loc) => {
                write!(f, "{}", loc.display_with(self.translator.clone()))
            }
            BranchTarget::Computed(expr) => {
                write!(f, "{}", expr.display_with(self.translator.clone()))
            }
        }
    }
}

impl<'target, 'trans> TranslatorDisplay<'target, 'trans> for BranchTarget {
    type Target = BranchTargetFormatter<'target, 'trans>;

    fn display_with(
        &'target self,
        translator: Option<&'trans Translator>,
    ) -> BranchTargetFormatter<'target, 'trans> {
        BranchTargetFormatter {
            target: self,
            translator,
        }
    }
}

impl BranchTarget {
    pub fn computed<E: Into<Term<Expr>>>(expr: E) -> Term<Self> {
        Self::Computed(expr.into()).into()
    }

    pub fn is_fixed(&self) -> bool {
        !self.is_computed()
    }

    pub fn is_computed(&self) -> bool {
        matches!(self, Self::Computed(_))
    }

    pub fn location<L: Into<Location>>(location: L) -> Term<Self> {
        Self::Location(location.into()).into()
    }
}

impl From<BranchTarget> for Term<BranchTarget> {
    fn from(tgt: BranchTarget) -> Self {
        Term::new(&TRGT, tgt)
    }
}

impl From<Location> for Term<BranchTarget> {
    fn from(t: Location) -> Self {
        Term::new(&TRGT, BranchTarget::Location(t))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cast {
    Void,
    Bool, // T -> Bool

    Signed(usize),   // sign-extension
    Unsigned(usize), // zero-extension

    Float(Arc<FloatFormat>), // T -> FloatFormat::T

    Pointer(Term<Cast>, usize),
    Function(Term<Cast>, SmallVec<[Term<Cast>; 4]>),
    Named(Ustr, usize),
}

impl From<Cast> for Term<Cast> {
    fn from(c: Cast) -> Self {
        Term::new(&CAST, c)
    }
}

impl Cast {
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Void)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, Self::Signed(_))
    }

    pub fn is_signed_with(&self, bits: usize) -> bool {
        matches!(self, Self::Signed(b) if *b == bits)
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(self, Self::Unsigned(_))
    }

    pub fn is_unsigned_with(&self, bits: usize) -> bool {
        matches!(self, Self::Unsigned(b) if *b == bits)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float(_))
    }

    pub fn is_float_with(&self, bits: usize) -> bool {
        matches!(self, Self::Float(f) if f.bits() == bits)
    }

    pub fn is_float_format(&self, fmt: &FloatFormat) -> bool {
        matches!(self, Self::Float(f) if &**f == fmt)
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_, _))
    }

    pub fn is_pointer_with(&self, bits: usize) -> bool {
        matches!(self, Self::Pointer(_, b) if *b == bits)
    }

    pub fn is_pointer_kind<F>(&self, f: F) -> bool
    where F: Fn(&Self) -> bool {
        matches!(self, Self::Pointer(t, _) if f(t))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_, _))
    }

    pub fn is_function_kind<F>(&self, f: F) -> bool
    where F: Fn(&Self, &[Term<Self>]) -> bool {
        matches!(self, Self::Function(rt, ats) if f(rt, ats))
    }

    pub fn is_named(&self) -> bool {
        matches!(self, Self::Named(_, _))
    }

    pub fn is_named_with<F>(&self, f: F) -> bool
    where F: Fn(&str) -> bool {
        matches!(self, Self::Named(nm, _) if f(nm))
    }

    pub fn bool() -> Term<Self> {
        Self::Bool.into()
    }

    pub fn void() -> Term<Self> {
        Self::Void.into()
    }

    pub fn signed(bits: usize) -> Term<Self> {
        Self::Signed(bits).into()
    }

    pub fn unsigned(bits: usize) -> Term<Self> {
        Self::Unsigned(bits).into()
    }

    pub fn float(fmt: Arc<FloatFormat>) -> Term<Self> {
        Self::Float(fmt).into()
    }

    pub fn pointer<T>(t: T, bits: usize) -> Term<Self>
    where T: Into<Term<Self>> {
        Self::Pointer(t.into(), bits).into()
    }

    pub fn function<R, I, A>(ret: R, args: I) -> Term<Self>
    where R: Into<Term<Self>>,
          I: ExactSizeIterator<Item=A>,
          A: Into<Term<Self>> {
        Self::Function(ret.into(), args.map(|arg| arg.into()).collect()).into()
    }

    pub fn named<N>(name: N, bits: usize) -> Term<Self>
    where N: Into<Ustr> {
        Self::Named(name.into(), bits).into()
    }
}

impl fmt::Display for Cast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::Bool => write!(f, "bool"),
            Self::Float(format) => write!(f, "f{}", format.bits()),
            Self::Signed(bits) => write!(f, "i{}", bits),
            Self::Unsigned(bits) => write!(f, "u{}", bits),
            Self::Pointer(typ, _) => write!(f, "ptr<{}>", typ),
            Self::Function(typ, typs) => {
                write!(f, "fn(")?;
                if !typs.is_empty() {
                    write!(f, "{}", typs[0])?;
                    for typ in &typs[1..] {
                        write!(f, "{}", typ)?;
                    }
                }
                write!(f, ") -> {}", typ)
            }
            Self::Named(name, _) => write!(f, "{}", name),
        }
    }
}

impl BitSize for Cast {
    fn bits(&self) -> usize {
        match self {
            Self::Void | Self::Function(_, _) => 0, // do not have a size
            Self::Bool => 1,
            Self::Float(format) => format.bits(),
            Self::Signed(bits)
            | Self::Unsigned(bits)
            | Self::Pointer(_, bits)
            | Self::Named(_, bits) => *bits,
        }
    }
}

#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub enum UnOp {
    NOT,
    NEG,

    ABS,
    SQRT,
    CEILING,
    FLOOR,
    ROUND,

    POPCOUNT,
}

#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub enum UnRel {
    NAN,
}

#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub enum BinOp {
    AND,
    OR,
    XOR,
    ADD,
    SUB,
    DIV,
    SDIV,
    MUL,
    REM,
    SREM,
    SHL,
    SAR,
    SHR,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinRel {
    EQ,
    NEQ,
    LT,
    LE,
    SLT,
    SLE,

    SBORROW,
    CARRY,
    SCARRY,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    UnRel(UnRel, Term<Expr>),               // T -> bool
    BinRel(BinRel, Term<Expr>, Term<Expr>), // T * T -> bool

    UnOp(UnOp, Term<Expr>),               // T -> T
    BinOp(BinOp, Term<Expr>, Term<Expr>), // T * T -> T

    Cast(Term<Expr>, Term<Cast>),         // T -> Cast::T
    Load(Term<Expr>, usize, AddressSpaceId), // SPACE[T]:SIZE -> T

    IfElse(Term<Expr>, Term<Expr>, Term<Expr>), // if T then T else T

    Extract(Term<Expr>, usize, usize), // T T[LSB..MSB) -> T
    ExtractHigh(Term<Expr>, usize),
    ExtractLow(Term<Expr>, usize),

    Concat(Term<Expr>, Term<Expr>), // T * T -> T

    Call(Term<BranchTarget>, SmallVec<[Term<Expr>; 4]>, usize),
    Intrinsic(Ustr, SmallVec<[Term<Expr>; 4]>, usize),

    Val(BitVec), // BitVec -> T
    Var(Var),    // String * usize -> T
}

impl From<Expr> for Term<Expr> {
    fn from(e: Expr) -> Self {
        Term::new(&EXPR, e)
    }
}

impl Expr {
    fn fmt_l1(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Val(v) => write!(f, "{}", v),
            Expr::Var(v) => write!(f, "{}", v),

            Expr::Intrinsic(name, args, _) => {
                write!(f, "{}(", name)?;
                if !args.is_empty() {
                    write!(f, "{}", args[0])?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg)?;
                    }
                }
                write!(f, ")")
            }

            Expr::ExtractHigh(expr, bits) => {
                write!(f, "extract-high({}, bits={})", expr, bits)
            }
            Expr::ExtractLow(expr, bits) => write!(f, "extract-low({}, bits={})", expr, bits),

            Expr::Cast(expr, t) => {
                expr.fmt_l1(f)?;
                write!(f, " as {}", t)
            }

            Expr::Load(expr, bits, space) => {
                write!(f, "space[{}][{}]:{}", space.index(), expr, bits)
            }

            Expr::Extract(expr, lsb, msb) => {
                write!(f, "extract({}, from={}, to={})", expr, lsb, msb)
            }

            Expr::UnOp(UnOp::ABS, expr) => write!(f, "abs({})", expr),
            Expr::UnOp(UnOp::SQRT, expr) => {
                write!(f, "sqrt({})", expr)
            }
            Expr::UnOp(UnOp::ROUND, expr) => {
                write!(f, "round({})", expr)
            }
            Expr::UnOp(UnOp::CEILING, expr) => {
                write!(f, "ceiling({})", expr)
            }
            Expr::UnOp(UnOp::FLOOR, expr) => {
                write!(f, "floor({})", expr)
            }
            Expr::UnOp(UnOp::POPCOUNT, expr) => {
                write!(f, "popcount({})", expr)
            }

            Expr::UnRel(UnRel::NAN, expr) => {
                write!(f, "is-nan({})", expr)
            }

            Expr::BinRel(BinRel::CARRY, e1, e2) => write!(f, "carry({}, {})", e1, e2),
            Expr::BinRel(BinRel::SCARRY, e1, e2) => write!(f, "scarry({}, {})", e1, e2),
            Expr::BinRel(BinRel::SBORROW, e1, e2) => write!(f, "sborrow({}, {})", e1, e2),

            expr => write!(f, "({})", expr),
        }
    }

    fn fmt_l2(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::UnOp(UnOp::NEG, expr) => {
                write!(f, "-")?;
                expr.fmt_l1(f)
            }
            Expr::UnOp(UnOp::NOT, expr) => {
                write!(f, "!")?;
                expr.fmt_l1(f)
            }
            expr => expr.fmt_l1(f),
        }
    }

    fn fmt_l3(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::MUL, e1, e2) => {
                e1.fmt_l3(f)?;
                write!(f, " * ")?;
                e2.fmt_l2(f)
            }
            Expr::BinOp(BinOp::DIV, e1, e2) => {
                e1.fmt_l3(f)?;
                write!(f, " / ")?;
                e2.fmt_l2(f)
            }
            Expr::BinOp(BinOp::SDIV, e1, e2) => {
                e1.fmt_l3(f)?;
                write!(f, " s/ ")?;
                e2.fmt_l2(f)
            }
            Expr::BinOp(BinOp::REM, e1, e2) => {
                e1.fmt_l3(f)?;
                write!(f, " % ")?;
                e2.fmt_l2(f)
            }
            Expr::BinOp(BinOp::SREM, e1, e2) => {
                e1.fmt_l3(f)?;
                write!(f, " s% ")?;
                e2.fmt_l2(f)
            }
            expr => expr.fmt_l2(f),
        }
    }

    fn fmt_l4(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::ADD, e1, e2) => {
                e1.fmt_l4(f)?;
                write!(f, " + ")?;
                e2.fmt_l3(f)
            }
            Expr::BinOp(BinOp::SUB, e1, e2) => {
                e1.fmt_l4(f)?;
                write!(f, " - ")?;
                e2.fmt_l3(f)
            }
            expr => expr.fmt_l3(f),
        }
    }

    fn fmt_l5(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::SHL, e1, e2) => {
                e1.fmt_l5(f)?;
                write!(f, " << ")?;
                e2.fmt_l4(f)
            }
            Expr::BinOp(BinOp::SHR, e1, e2) => {
                e1.fmt_l5(f)?;
                write!(f, " >> ")?;
                e2.fmt_l4(f)
            }
            Expr::BinOp(BinOp::SAR, e1, e2) => {
                e1.fmt_l5(f)?;
                write!(f, " s>> ")?;
                e2.fmt_l4(f)
            }
            expr => expr.fmt_l4(f),
        }
    }

    fn fmt_l6(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::LT, e1, e2) => {
                e1.fmt_l6(f)?;
                write!(f, " < ")?;
                e2.fmt_l5(f)
            }
            Expr::BinRel(BinRel::LE, e1, e2) => {
                e1.fmt_l6(f)?;
                write!(f, " <= ")?;
                e2.fmt_l5(f)
            }
            Expr::BinRel(BinRel::SLT, e1, e2) => {
                e1.fmt_l6(f)?;
                write!(f, " s< ")?;
                e2.fmt_l5(f)
            }
            Expr::BinRel(BinRel::SLE, e1, e2) => {
                e1.fmt_l6(f)?;
                write!(f, " s<= ")?;
                e2.fmt_l5(f)
            }
            expr => expr.fmt_l5(f),
        }
    }

    fn fmt_l7(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::EQ, e1, e2) => {
                e1.fmt_l7(f)?;
                write!(f, " == ")?;
                e2.fmt_l6(f)
            }
            Expr::BinRel(BinRel::NEQ, e1, e2) => {
                e1.fmt_l7(f)?;
                write!(f, " != ")?;
                e2.fmt_l6(f)
            }
            expr => expr.fmt_l6(f),
        }
    }

    fn fmt_l8(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::BinOp(BinOp::AND, e1, e2) = self {
            e1.fmt_l8(f)?;
            write!(f, " & ")?;
            e2.fmt_l7(f)
        } else {
            self.fmt_l7(f)
        }
    }

    fn fmt_l9(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::BinOp(BinOp::XOR, e1, e2) = self {
            e1.fmt_l9(f)?;
            write!(f, " ^ ")?;
            e2.fmt_l8(f)
        } else {
            self.fmt_l8(f)
        }
    }

    fn fmt_l10(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::BinOp(BinOp::OR, e1, e2) = self {
            e1.fmt_l10(f)?;
            write!(f, " | ")?;
            e2.fmt_l9(f)
        } else {
            self.fmt_l9(f)
        }
    }

    fn fmt_l11(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::Concat(e1, e2) = self {
            e1.fmt_l11(f)?;
            write!(f, " ++ ")?;
            e2.fmt_l10(f)
        } else {
            self.fmt_l10(f)
        }
    }

    fn fmt_l12(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::IfElse(c, et, ef) = self {
            write!(f, "if ")?;
            c.fmt_l12(f)?;
            write!(f, " then ")?;
            et.fmt_l12(f)?;
            write!(f, " else ")?;
            ef.fmt_l12(f)
        } else {
            self.fmt_l11(f)
        }
    }
}

impl<'v, 't> Expr {
    fn fmt_l1_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::Val(v) => write!(f, "{}", v.display_with(translator.clone())),
            Expr::Var(v) => write!(f, "{}", v.display_with(translator.clone())),

            Expr::Intrinsic(name, args, _) => {
                write!(f, "{}(", name)?;
                if !args.is_empty() {
                    write!(f, "{}", args[0].display_with(translator.clone()))?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg.display_with(translator.clone()))?;
                    }
                }
                write!(f, ")")
            }

            Expr::ExtractHigh(expr, bits) => write!(
                f,
                "extract-high({}, bits={})",
                expr.display_with(translator.clone()),
                bits
            ),
            Expr::ExtractLow(expr, bits) => write!(
                f,
                "extract-low({}, bits={})",
                expr.display_with(translator.clone()),
                bits
            ),

            Expr::Cast(expr, t) => {
                expr.fmt_l1_with(f, translator)?;
                write!(f, " as {}", t)
            }

            Expr::Load(expr, bits, space) => {
                if let Some(trans) = translator {
                    let space = trans.manager().unchecked_space_by_id(*space);
                    write!(
                        f,
                        "{}[{}]:{}",
                        space.name(),
                        expr.display_with(translator.clone()),
                        bits
                    )
                } else {
                    write!(
                        f,
                        "space[{}][{}]:{}",
                        space.index(),
                        expr.display_with(translator.clone()),
                        bits
                    )
                }
            }

            Expr::Extract(expr, lsb, msb) => write!(
                f,
                "extract({}, from={}, to={})",
                expr.display_with(translator.clone()),
                lsb,
                msb
            ),

            Expr::UnOp(UnOp::ABS, expr) => {
                write!(f, "abs({})", expr.display_with(translator.clone()))
            }
            Expr::UnOp(UnOp::SQRT, expr) => {
                write!(f, "sqrt({})", expr.display_with(translator.clone()))
            }
            Expr::UnOp(UnOp::ROUND, expr) => {
                write!(f, "round({})", expr.display_with(translator.clone()))
            }
            Expr::UnOp(UnOp::CEILING, expr) => {
                write!(f, "ceiling({})", expr.display_with(translator.clone()))
            }
            Expr::UnOp(UnOp::FLOOR, expr) => {
                write!(f, "floor({})", expr.display_with(translator.clone()))
            }
            Expr::UnOp(UnOp::POPCOUNT, expr) => {
                write!(f, "popcount({})", expr.display_with(translator.clone()))
            }

            Expr::UnRel(UnRel::NAN, expr) => {
                write!(f, "is-nan({})", expr.display_with(translator.clone()))
            }

            Expr::BinRel(BinRel::CARRY, e1, e2) => write!(
                f,
                "carry({}, {})",
                e1.display_with(translator.clone()),
                e2.display_with(translator.clone())
            ),
            Expr::BinRel(BinRel::SCARRY, e1, e2) => write!(
                f,
                "scarry({}, {})",
                e1.display_with(translator.clone()),
                e2.display_with(translator.clone())
            ),
            Expr::BinRel(BinRel::SBORROW, e1, e2) => write!(
                f,
                "sborrow({}, {})",
                e1.display_with(translator.clone()),
                e2.display_with(translator.clone())
            ),

            expr => write!(f, "({})", expr.display_with(translator)),
        }
    }

    fn fmt_l2_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::UnOp(UnOp::NEG, expr) => {
                write!(f, "-")?;
                expr.fmt_l1_with(f, translator)
            }
            Expr::UnOp(UnOp::NOT, expr) => {
                write!(f, "!")?;
                expr.fmt_l1_with(f, translator)
            }
            expr => expr.fmt_l1_with(f, translator),
        }
    }

    fn fmt_l3_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::MUL, e1, e2) => {
                e1.fmt_l3_with(f, translator.clone())?;
                write!(f, " * ")?;
                e2.fmt_l2_with(f, translator)
            }
            Expr::BinOp(BinOp::DIV, e1, e2) => {
                e1.fmt_l3_with(f, translator.clone())?;
                write!(f, " / ")?;
                e2.fmt_l2_with(f, translator)
            }
            Expr::BinOp(BinOp::SDIV, e1, e2) => {
                e1.fmt_l3_with(f, translator.clone())?;
                write!(f, " s/ ")?;
                e2.fmt_l2_with(f, translator)
            }
            Expr::BinOp(BinOp::REM, e1, e2) => {
                e1.fmt_l3_with(f, translator.clone())?;
                write!(f, " % ")?;
                e2.fmt_l2_with(f, translator)
            }
            Expr::BinOp(BinOp::SREM, e1, e2) => {
                e1.fmt_l3_with(f, translator.clone())?;
                write!(f, " s% ")?;
                e2.fmt_l2_with(f, translator)
            }
            expr => expr.fmt_l2_with(f, translator),
        }
    }

    fn fmt_l4_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::ADD, e1, e2) => {
                e1.fmt_l4_with(f, translator.clone())?;
                write!(f, " + ")?;
                e2.fmt_l3_with(f, translator)
            }
            Expr::BinOp(BinOp::SUB, e1, e2) => {
                e1.fmt_l4_with(f, translator.clone())?;
                write!(f, " - ")?;
                e2.fmt_l3_with(f, translator)
            }
            expr => expr.fmt_l3_with(f, translator),
        }
    }

    fn fmt_l5_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::SHL, e1, e2) => {
                e1.fmt_l5_with(f, translator.clone())?;
                write!(f, " << ")?;
                e2.fmt_l4_with(f, translator)
            }
            Expr::BinOp(BinOp::SHR, e1, e2) => {
                e1.fmt_l5_with(f, translator.clone())?;
                write!(f, " >> ")?;
                e2.fmt_l4_with(f, translator)
            }
            Expr::BinOp(BinOp::SAR, e1, e2) => {
                e1.fmt_l5_with(f, translator.clone())?;
                write!(f, " s>> ")?;
                e2.fmt_l4_with(f, translator)
            }
            expr => expr.fmt_l4_with(f, translator),
        }
    }

    fn fmt_l6_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::LT, e1, e2) => {
                e1.fmt_l6_with(f, translator.clone())?;
                write!(f, " < ")?;
                e2.fmt_l5_with(f, translator)
            }
            Expr::BinRel(BinRel::LE, e1, e2) => {
                e1.fmt_l6_with(f, translator.clone())?;
                write!(f, " <= ")?;
                e2.fmt_l5_with(f, translator)
            }
            Expr::BinRel(BinRel::SLT, e1, e2) => {
                e1.fmt_l6_with(f, translator.clone())?;
                write!(f, " s< ")?;
                e2.fmt_l5_with(f, translator)
            }
            Expr::BinRel(BinRel::SLE, e1, e2) => {
                e1.fmt_l6_with(f, translator.clone())?;
                write!(f, " s<= ")?;
                e2.fmt_l5_with(f, translator)
            }
            expr => expr.fmt_l5_with(f, translator),
        }
    }

    fn fmt_l7_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::EQ, e1, e2) => {
                e1.fmt_l7_with(f, translator.clone())?;
                write!(f, " == ")?;
                e2.fmt_l6_with(f, translator)
            }
            Expr::BinRel(BinRel::NEQ, e1, e2) => {
                e1.fmt_l7_with(f, translator.clone())?;
                write!(f, " != ")?;
                e2.fmt_l6_with(f, translator)
            }
            expr => expr.fmt_l6_with(f, translator),
        }
    }

    fn fmt_l8_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        if let Expr::BinOp(BinOp::AND, e1, e2) = self {
            e1.fmt_l8_with(f, translator.clone())?;
            write!(f, " & ")?;
            e2.fmt_l7_with(f, translator)
        } else {
            self.fmt_l7_with(f, translator)
        }
    }

    fn fmt_l9_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        if let Expr::BinOp(BinOp::XOR, e1, e2) = self {
            e1.fmt_l9_with(f, translator.clone())?;
            write!(f, " ^ ")?;
            e2.fmt_l8_with(f, translator)
        } else {
            self.fmt_l8_with(f, translator)
        }
    }

    fn fmt_l10_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        if let Expr::BinOp(BinOp::OR, e1, e2) = self {
            e1.fmt_l10_with(f, translator.clone())?;
            write!(f, " | ")?;
            e2.fmt_l9_with(f, translator)
        } else {
            self.fmt_l9_with(f, translator)
        }
    }

    fn fmt_l11_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        if let Expr::Concat(e1, e2) = self {
            e1.fmt_l11_with(f, translator.clone())?;
            write!(f, " ++ ")?;
            e2.fmt_l10_with(f, translator.clone())
        } else {
            self.fmt_l10_with(f, translator)
        }
    }

    fn fmt_l12_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        translator: Option<&'t Translator>,
    ) -> fmt::Result {
        if let Expr::IfElse(c, et, ef) = self {
            write!(f, "if ")?;
            c.fmt_l12_with(f, translator.clone())?;
            write!(f, " then ")?;
            et.fmt_l12_with(f, translator.clone())?;
            write!(f, " else ")?;
            ef.fmt_l12_with(f, translator)
        } else {
            self.fmt_l11_with(f, translator)
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_l12(f)
    }
}

pub struct ExprFormatter<'expr, 'trans> {
    expr: &'expr Expr,
    translator: Option<&'trans Translator>,
}

impl<'expr, 'trans> fmt::Display for ExprFormatter<'expr, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expr.fmt_l12_with(f, self.translator.clone())
    }
}

impl<'expr, 'trans> TranslatorDisplay<'expr, 'trans> for Expr {
    type Target = ExprFormatter<'expr, 'trans>;

    fn display_with(
        &'expr self,
        translator: Option<&'trans Translator>,
    ) -> ExprFormatter<'expr, 'trans> {
        ExprFormatter {
            expr: self,
            translator,
        }
    }
}

impl From<BitVec> for Expr {
    fn from(val: BitVec) -> Self {
        Self::Val(val)
    }
}

impl From<Var> for Expr {
    fn from(var: Var) -> Self {
        Self::Var(var)
    }
}

impl<'z> FromSpace<'z, VarnodeData> for Expr {
    fn from_space_with(t: VarnodeData, _arena: &'_ IRBuilderArena, manager: &SpaceManager) -> Self {
        Expr::from_space(t, manager)
    }

    fn from_space(vnd: VarnodeData, manager: &SpaceManager) -> Expr {
        let space = manager.unchecked_space_by_id(vnd.space());
        if space.is_constant() {
            Expr::from(BitVec::from_u64(vnd.offset(), vnd.size() * 8))
        } else {
            // if space.is_unique() || space.is_register() {
            Expr::from(Var::from(vnd))
        } /* else {
              // address-like: the vnd size is what it points to
              let asz = space.address_size() * 8;
              let val = BitVec::from_u64(vnd.offset(), asz);
              let src = if space.word_size() > 1 {
                  let s = Expr::from(val);
                  let bits = s.bits();

                  let w = Expr::from(BitVec::from_usize(space.word_size(), bits));

                  Expr::int_mul(s, w)
              } else {
                  Expr::from(val)
              };
              // TODO: we should preserve this information!!!
              //Expr::Cast(Box::new(src), Cast::Pointer(Box::new(Cast::Void), asz))
              Expr::load(
                  src,
                  vnd.space().address_size() * 8,
                  vnd.space(),
              )
          }
          */
    }
}

impl BitSize for Expr {
    fn bits(&self) -> usize {
        match self {
            Self::UnRel(_, _) | Self::BinRel(_, _, _) => 1,
            Self::UnOp(_, e) | Self::BinOp(_, e, _) => e.bits(),
            Self::Cast(_, cast) => cast.bits(),
            Self::Load(_, bits, _) => *bits,
            Self::Extract(_, lsb, msb) => *msb - *lsb,
            Self::ExtractHigh(_, bits) | Self::ExtractLow(_, bits) => *bits,
            Self::Concat(l, r) => l.bits() + r.bits(),
            Self::IfElse(_, e, _) => e.bits(),
            Self::Call(_, _, bits) => *bits,
            Self::Intrinsic(_, _, bits) => *bits,
            Self::Val(bv) => bv.bits(),
            Self::Var(var) => var.bits(),
        }
    }
}

impl Expr {
    pub fn is_cast_kind<F>(&self, f: F) -> bool
    where F: Fn(&Cast) -> bool {
        matches!(self, Self::Cast(_, cst) if f(cst))
    }

    pub fn is_bool(&self) -> bool {
        self.is_cast_kind(Cast::is_bool)
    }

    pub fn is_float(&self) -> bool {
        self.is_cast_kind(Cast::is_float)
    }

    pub fn is_float_format(&self, format: &FloatFormat) -> bool {
        self.is_cast_kind(|f| f.is_float_format(format))
    }

    pub fn is_signed(&self) -> bool {
        self.is_cast_kind(Cast::is_signed)
    }

    pub fn is_signed_bits(&self, bits: usize) -> bool {
        self.is_cast_kind(|s| s.is_signed_with(bits))
    }

    pub fn is_unsigned(&self) -> bool {
        self.is_cast_kind(|cst| matches!(cst, Cast::Unsigned(_) | Cast::Pointer(_, _))) ||
            !matches!(self, Self::Cast(_, _))
    }

    pub fn is_unsigned_bits(&self, bits: usize) -> bool {
        self.is_cast_kind(|cst| matches!(cst, Cast::Unsigned(sz) | Cast::Pointer(_, sz) if *sz == bits)) ||
            (!matches!(self, Self::Cast(_, _)) && self.bits() == bits)
    }

    pub fn value(&self) -> Option<&BitVec> {
        if let Self::Val(ref v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn cast_bool<E>(expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        if expr.is_bool() {
            expr
        } else {
            Self::Cast(expr.into(), Cast::bool()).into()
        }
    }

    pub fn cast_signed<E>(expr: E, bits: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        if expr.is_signed_bits(bits) {
            expr
        } else {
            Self::Cast(expr.into(), Cast::signed(bits)).into()
        }
    }

    pub fn cast_unsigned<E>(expr: E, bits: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        if expr.is_unsigned_bits(bits) {
            expr
        } else {
            Self::Cast(expr.into(), Cast::unsigned(bits)).into()
        }
    }

    pub fn cast_float<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        if expr.is_float_format(&*format) {
            expr
        } else {
            Self::Cast(expr.into(), Cast::float(format)).into()
        }
    }

    pub fn cast<E>(expr: E, bits: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::Cast(expr.into(), Cast::unsigned(bits)).into()
    }

    pub fn extract_high<E>(expr: E, bits: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        if expr.is_unsigned_bits(bits) {
            expr
        } else {
            Self::ExtractHigh(expr.into(), bits).into()
        }
    }

    pub fn extract_low<E>(expr: E, bits: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        if expr.is_unsigned_bits(bits) {
            expr
        } else {
            Self::ExtractLow(expr.into(), bits).into()
        }
    }

    pub fn concat<E1, E2>(lhs: E1, rhs: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        Self::Concat(lhs, rhs).into()
    }

    pub(crate) fn unary_op<E>(op: UnOp, expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::UnOp(op, expr.into()).into()
    }

    pub(crate) fn unary_rel<E>(rel: UnRel, expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::cast_bool(Self::UnRel(rel, expr.into()))
    }

    pub(crate) fn binary_op<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::BinOp(op, expr1.into(), expr2.into()).into()
    }

    pub(crate) fn binary_op_promote_as<E1, E2, F>(op: BinOp, expr1: E1, expr2: E2, cast: F) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
        F: Fn(Term<Self>, usize) -> Term<Self>,
    {
        let e1 = expr1.into();
        let e2 = expr2.into();
        let bits = e1.bits().max(e2.bits());

        Self::binary_op(op, cast(e1, bits), cast(e2, bits))
    }

    pub(crate) fn binary_op_promote<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(op, expr1, expr2, |e, sz| Self::cast_unsigned(e, sz))
    }

    pub(crate) fn binary_op_promote_bool<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(op, expr1, expr2, |e, _sz| Self::cast_bool(e))
    }

    pub(crate) fn binary_op_promote_signed<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(op, expr1, expr2, |e, sz| Self::cast_signed(e, sz))
    }

    pub(crate) fn binary_op_promote_float<E1, E2>(
        op: BinOp,
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(op, expr1, expr2, |e, sz| {
            Self::cast_float(Self::cast_signed(e, sz), formats[&sz].clone())
        })
    }

    pub(crate) fn binary_rel<E1, E2>(rel: BinRel, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::cast_bool(Self::BinRel(
            rel,
            expr1.into(),
            expr2.into(),
        ))
    }

    pub(crate) fn binary_rel_promote_as<E1, E2, F>(
        op: BinRel,
        expr1: E1,
        expr2: E2,
        cast: F,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
        F: Fn(Term<Self>, usize) -> Term<Self>,
    {
        let e1 = expr1.into();
        let e2 = expr2.into();
        let bits = e1.bits().max(e2.bits());

        Self::binary_rel(op, cast(e1, bits), cast(e2, bits))
    }

    pub(crate) fn binary_rel_promote<E1, E2>(op: BinRel, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_as(op, expr1, expr2, |e, sz| Self::cast_unsigned(e, sz))
    }

    pub(crate) fn binary_rel_promote_float<E1, E2>(
        op: BinRel,
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_as(op, expr1, expr2, |e, sz| {
            Self::cast_float(Self::cast_signed(e, sz), formats[&sz].clone())
        })
    }

    pub(crate) fn binary_rel_promote_signed<E1, E2>(op: BinRel, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_as(op, expr1, expr2, |e, sz| Self::cast_signed(e, sz))
    }

    pub(crate) fn binary_rel_promote_bool<E1, E2>(op: BinRel, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_as(op, expr1, expr2, |e, _sz| Self::cast_bool(e))
    }

    pub fn load<E>(expr: E, size: usize, space: &AddressSpace) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::Load(
            Self::cast_unsigned(expr, space.address_size() * 8),
            size,
            space.id(),
        ).into()
    }

    pub fn call<T>(target: T, bits: usize) -> Term<Self>
    where
        T: Into<Term<BranchTarget>>,
    {
        Self::Call(target.into(), Default::default(), bits).into()
    }

    pub fn call_with<T, I, E>(target: T, arguments: I, bits: usize) -> Term<Self>
    where
        T: Into<Term<BranchTarget>>,
        I: ExactSizeIterator<Item = E>,
        E: Into<Term<Self>>,
    {
        let mut args = SmallVec::with_capacity(arguments.len());
        for arg in arguments {
            args.push(arg.into());
        }

        Self::Call(target.into(), args, bits).into()
    }

    pub fn intrinsic<N, I, E>(name: N, arguments: I, bits: usize) -> Term<Self>
    where
        N: Into<Ustr>,
        I: ExactSizeIterator<Item = E>,
        E: Into<Term<Self>>,
    {
        let mut args = SmallVec::with_capacity(arguments.len());
        for arg in arguments {
            args.push(arg.into());
        }

        Self::Intrinsic(name.into(), args, bits).into()
    }

    pub fn extract<E>(expr: E, loff: usize, moff: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::Extract(expr.into(), loff, moff).into()
    }

    pub fn ite<C, E1, E2>(cond: C, texpr: E1, fexpr: E2) -> Term<Self>
    where
        C: Into<Term<Self>>,
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        let e1 = texpr.into();
        let e2 = fexpr.into();
        let bits = e1.bits().max(e2.bits());

        Self::IfElse(
            Self::cast_bool(cond),
            Self::cast_unsigned(e1, bits),
            Self::cast_unsigned(e2, bits),
        ).into()
    }

    pub fn bool_not<E>(expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::unary_op(UnOp::NOT, Self::cast_bool(expr))
    }

    pub fn bool_eq<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_bool(BinRel::EQ, expr1, expr2)
    }

    pub fn bool_neq<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_bool(BinRel::NEQ, expr1, expr2)
    }

    pub fn bool_and<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_bool(BinOp::AND, expr1, expr2)
    }

    pub fn bool_or<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_bool(BinOp::OR, expr1, expr2)
    }

    pub fn bool_xor<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_bool(BinOp::XOR, expr1, expr2)
    }

    pub fn float_nan<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();

        let format = formats[&bits].clone();

        Self::unary_rel(
            UnRel::NAN,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_neg<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();
        let format = formats[&bits].clone();

        Self::unary_op(
            UnOp::NEG,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_abs<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();
        let format = formats[&bits].clone();

        Self::unary_op(
            UnOp::ABS,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_sqrt<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();
        let format = formats[&bits].clone();

        Self::unary_op(
            UnOp::SQRT,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_ceiling<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();
        let format = formats[&bits].clone();

        Self::unary_op(
            UnOp::CEILING,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_round<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();
        let format = formats[&bits].clone();

        Self::unary_op(
            UnOp::ROUND,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_floor<E>(expr: E, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let bits = expr.bits();
        let format = formats[&bits].clone();

        Self::unary_op(
            UnOp::FLOOR,
            Expr::cast_float(Expr::cast_signed(expr, bits), format),
        )
    }

    pub fn float_eq<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::EQ, expr1, expr2, formats)
    }

    pub fn float_neq<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::NEQ, expr1, expr2, formats)
    }

    pub fn float_lt<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::LT, expr1, expr2, formats)
    }

    pub fn float_le<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::LE, expr1, expr2, formats)
    }

    pub fn float_add<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::ADD, expr1, expr2, formats)
    }

    pub fn float_sub<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::SUB, expr1, expr2, formats)
    }

    pub fn float_div<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::DIV, expr1, expr2, formats)
    }

    pub fn float_mul<E1, E2>(expr1: E1, expr2: E2, formats: &Map<usize, Arc<FloatFormat>>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::MUL, expr1, expr2, formats)
    }

    pub fn count_ones<E>(expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::unary_op(UnOp::POPCOUNT, expr.into())
    }

    pub fn int_neg<E>(expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let size = expr.bits();
        Self::unary_op(UnOp::NEG, Self::cast_signed(expr, size))
    }

    pub fn int_not<E>(expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let size = expr.bits();
        Self::unary_op(UnOp::NOT, Self::cast_unsigned(expr, size))
    }

    pub fn int_eq<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote(BinRel::EQ, expr1, expr2)
    }

    pub fn int_neq<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote(BinRel::NEQ, expr1, expr2)
    }

    pub fn int_lt<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote(BinRel::LT, expr1, expr2)
    }

    pub fn int_le<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote(BinRel::LE, expr1, expr2)
    }

    pub fn int_slt<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_signed(BinRel::SLT, expr1, expr2)
    }

    pub fn int_sle<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_signed(BinRel::SLE, expr1, expr2)
    }

    pub fn int_carry<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote(BinRel::CARRY, expr1, expr2)
    }

    pub fn int_scarry<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_signed(BinRel::SCARRY, expr1, expr2)
    }

    pub fn int_sborrow<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_signed(BinRel::SBORROW, expr1, expr2)
    }

    pub fn int_add<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::ADD, expr1, expr2)
    }

    pub fn int_sub<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::SUB, expr1, expr2)
    }

    pub fn int_mul<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::MUL, expr1, expr2)
    }

    pub fn int_div<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::DIV, expr1, expr2)
    }

    pub fn int_sdiv<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_signed(BinOp::SDIV, expr1, expr2)
    }

    pub fn int_rem<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::REM, expr1, expr2)
    }

    pub fn int_srem<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_signed(BinOp::SREM, expr1, expr2)
    }

    pub fn int_shl<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::SHL, expr1, expr2)
    }

    pub fn int_shr<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::SHR, expr1, expr2)
    }

    pub fn int_sar<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_signed(BinOp::SAR, expr1, expr2)
    }

    pub fn int_and<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::AND, expr1, expr2)
    }

    pub fn int_or<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::OR, expr1, expr2)
    }

    pub fn int_xor<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote(BinOp::XOR, expr1, expr2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stmt {
    Assign(Var, Term<Expr>),

    Store(Term<Expr>, Term<Expr>, usize, AddressSpaceId), // SPACE[T]:SIZE <- T

    Branch(Term<BranchTarget>),
    CBranch(Term<Expr>, Term<BranchTarget>),

    Call(Term<BranchTarget>, SmallVec<[Term<Expr>; 4]>),
    Return(Term<BranchTarget>),

    Skip, // NO-OP

    Intrinsic(Ustr, SmallVec<[Term<Expr>; 4]>), // no output intrinsic
}

impl From<Stmt> for Term<Stmt> {
    fn from(e: Stmt) -> Self {
        Term::new(&STMT, e)
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Assign(dest, src) => write!(f, "{} ← {}", dest, src),
            Self::Store(dest, src, size, spc) => {
                write!(f, "space[{}][{}]:{} ← {}", spc.index(), dest, size, src)
            }
            Self::Branch(target) => write!(f, "goto {}", target),
            Self::CBranch(cond, target) => write!(f, "goto {} if {}", target, cond),
            Self::Call(target, args) => {
                if !args.is_empty() {
                    write!(f, "call {}(", target)?;
                    write!(f, "{}", args[0])?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg)?;
                    }
                    write!(f, ")")
                } else {
                    write!(f, "call {}", target)
                }
            }
            Self::Return(target) => write!(f, "return {}", target),
            Self::Skip => write!(f, "skip"),
            Self::Intrinsic(name, args) => {
                write!(f, "{}(", name)?;
                if !args.is_empty() {
                    write!(f, "{}", args[0])?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg)?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

impl<'stmt, 'trans> fmt::Display for StmtFormatter<'stmt, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.stmt {
            Stmt::Assign(dest, src) => write!(
                f,
                "{} ← {}",
                dest.display_with(self.translator.clone()),
                src.display_with(self.translator.clone())
            ),
            Stmt::Store(dest, src, size, spc) => {
                if let Some(trans) = self.translator {
                    let space = trans.manager().unchecked_space_by_id(*spc);
                    write!(
                        f,
                        "{}[{}]:{} ← {}",
                        space.name(),
                        dest.display_with(self.translator.clone()),
                        size,
                        src.display_with(self.translator.clone())
                    )
                } else {
                    write!(
                        f,
                        "space[{}][{}]:{} ← {}",
                        spc.index(),
                        dest.display_with(self.translator.clone()),
                        size,
                        src.display_with(self.translator.clone())
                    )
                }
            }
            Stmt::Branch(target) => {
                write!(f, "goto {}", target.display_with(self.translator.clone()))
            }
            Stmt::CBranch(cond, target) => write!(
                f,
                "goto {} if {}",
                target.display_with(self.translator.clone()),
                cond.display_with(self.translator.clone())
            ),
            Stmt::Call(target, args) => {
                if !args.is_empty() {
                    write!(f, "call {}(", target.display_with(self.translator.clone()))?;
                    write!(f, "{}", args[0].display_with(self.translator.clone()))?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg.display_with(self.translator.clone()))?;
                    }
                    write!(f, ")")
                } else {
                    write!(f, "call {}", target.display_with(self.translator.clone()))
                }
            }
            Stmt::Return(target) => {
                write!(f, "return {}", target.display_with(self.translator.clone()))
            }
            Stmt::Skip => write!(f, "skip"),
            Stmt::Intrinsic(name, args) => {
                write!(f, "{}(", name)?;
                if !args.is_empty() {
                    write!(f, "{}", args[0].display_with(self.translator.clone()))?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg.display_with(self.translator.clone()))?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

pub struct StmtFormatter<'stmt, 'trans> {
    stmt: &'stmt Stmt,
    translator: Option<&'trans Translator>,
}

impl<'stmt, 'trans> TranslatorDisplay<'stmt, 'trans> for Stmt {
    type Target = StmtFormatter<'stmt, 'trans>;

    fn display_with(
        &'stmt self,
        translator: Option<&'trans Translator>,
    ) -> StmtFormatter<'stmt, 'trans> {
        StmtFormatter {
            stmt: self,
            translator,
        }
    }
}

impl Stmt {
    pub fn from_parts<I: ExactSizeIterator<Item = VarnodeData>>(
        manager: &SpaceManager,
        float_formats: &Map<usize, Arc<FloatFormat>>,
        user_ops: &[Arc<str>],
        address: &AddressValue,
        position: usize,
        opcode: Opcode,
        inputs: I,
        output: Option<VarnodeData>,
    ) -> Term<Self> {
        let mut inputs = inputs.into_iter();
        let spaces = manager.spaces();
        match opcode {
            Opcode::Copy => Self::assign(
                output.unwrap(),
                Expr::from_space(inputs.next().unwrap(), manager),
            ),
            Opcode::Load => {
                let space = &spaces[inputs.next().unwrap().offset() as usize];
                let destination = output.unwrap();
                let source: Expr = inputs.next().unwrap().into_space(manager);
                let s: Term<Expr> = source.into();
                let size = destination.size() * 8;

                let src = if space.word_size() > 1 {
                    let bits = s.bits();

                    let w = Expr::from(BitVec::from_usize(space.word_size(), bits));

                    Expr::int_mul(s, w)
                } else {
                    s
                };

                Self::assign(destination, Expr::load(src, size, space))
            }
            Opcode::Store => {
                let space = &spaces[inputs.next().unwrap().offset() as usize];
                let destination: Expr = inputs.next().unwrap().into_space(manager);
                let d: Term<Expr> = destination.into();
                let source = inputs.next().unwrap();
                let size = source.size() * 8;

                let dest = if space.word_size() > 1 {
                    let bits = d.bits();

                    let w = Expr::from(BitVec::from_usize(space.word_size(), bits));

                    Expr::int_mul(d, w)
                } else {
                    d
                };

                Self::store(dest, Expr::from_space(source, manager), size, space)
            }
            Opcode::Branch => {
                let mut target = Location::from_space(inputs.next().unwrap(), manager);
                target.absolute_from(address.to_owned(), position);

                Self::branch(target)
            }
            Opcode::CBranch => {
                let mut target = Location::from_space(inputs.next().unwrap(), manager);
                target.absolute_from(address.to_owned(), position);

                let condition = Expr::from_space(inputs.next().unwrap(), manager);

                Self::branch_conditional(condition, target)
            }
            Opcode::IBranch => {
                let target = Expr::from_space(inputs.next().unwrap(), manager);
                let space = manager.unchecked_space_by_id(address.space());

                Self::branch_indirect(target, space)
            }
            Opcode::Call => {
                let mut target = Location::from_space(inputs.next().unwrap(), manager);
                target.absolute_from(address.to_owned(), position);

                Self::call(target)
            }
            Opcode::ICall => {
                let target = Expr::from_space(inputs.next().unwrap(), manager);
                let space = manager.unchecked_space_by_id(address.space());

                Self::call_indirect(target, space)
            }
            Opcode::CallOther => {
                // TODO: eliminate this allocation
                let name = user_ops[inputs.next().unwrap().offset() as usize].clone();
                if let Some(output) = output {
                    let output = Var::from(output);
                    let bits = output.bits();
                    Self::assign(
                        output,
                        Expr::intrinsic(
                            &*name,
                            inputs.into_iter().map(|v| Expr::from_space(v, manager)),
                            bits,
                        ),
                    )
                } else {
                    Self::intrinsic(
                        &*name,
                        inputs.into_iter().map(|v| Expr::from_space(v, manager)),
                    )
                }
            }
            Opcode::Return => {
                let target = Expr::from_space(inputs.next().unwrap(), manager);
                let space = manager.unchecked_space_by_id(address.space());

                Self::return_(target, space)
            }
            Opcode::Subpiece => {
                let source = Expr::from_space(inputs.next().unwrap(), manager);
                let src_size = source.bits();

                let output = output.unwrap();
                let out_size = output.size() * 8;

                let loff = inputs.next().unwrap().offset() as usize * 8;
                let trun_size = src_size.checked_sub(loff).unwrap_or(0);

                let trun = if out_size > trun_size {
                    // extract high + expand
                    let source_htrun = Expr::extract_high(source, trun_size);
                    Expr::cast_unsigned(source_htrun, out_size)
                } else {
                    // extract
                    let hoff = loff + out_size;
                    Expr::extract(source, loff, hoff)
                };

                Self::assign(output, trun)
            }
            Opcode::PopCount => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = Var::from(output.unwrap());

                let size = output.bits();
                let popcount = Expr::count_ones(input);

                Self::assign(output, Expr::cast_unsigned(popcount, size))
            }
            Opcode::BoolNot => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::bool_not(input))
            }
            Opcode::BoolAnd => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::bool_and(input1, input2))
            }
            Opcode::BoolOr => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::bool_or(input1, input2))
            }
            Opcode::BoolXor => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::bool_xor(input1, input2))
            }
            Opcode::IntNeg => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_neg(input))
            }
            Opcode::IntNot => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_not(input))
            }
            Opcode::IntSExt => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();
                let size = output.size() * 8;

                Self::assign(output, Expr::cast_signed(input, size))
            }
            Opcode::IntZExt => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();
                let size = output.size() * 8;

                Self::assign(output, Expr::cast_unsigned(input, size))
            }
            Opcode::IntEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_eq(input1, input2))
            }
            Opcode::IntNotEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_neq(input1, input2))
            }
            Opcode::IntLess => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_lt(input1, input2))
            }
            Opcode::IntLessEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_le(input1, input2))
            }
            Opcode::IntSLess => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_slt(input1, input2))
            }
            Opcode::IntSLessEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_sle(input1, input2))
            }
            Opcode::IntCarry => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_carry(input1, input2))
            }
            Opcode::IntSCarry => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_scarry(input1, input2))
            }
            Opcode::IntSBorrow => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_sborrow(input1, input2))
            }
            Opcode::IntAdd => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_add(input1, input2))
            }
            Opcode::IntSub => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_sub(input1, input2))
            }
            Opcode::IntDiv => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_div(input1, input2))
            }
            Opcode::IntSDiv => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_sdiv(input1, input2))
            }
            Opcode::IntMul => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_mul(input1, input2))
            }
            Opcode::IntRem => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_rem(input1, input2))
            }
            Opcode::IntSRem => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_srem(input1, input2))
            }
            Opcode::IntLShift => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_shl(input1, input2))
            }
            Opcode::IntRShift => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_shr(input1, input2))
            }
            Opcode::IntSRShift => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_sar(input1, input2))
            }
            Opcode::IntAnd => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_and(input1, input2))
            }
            Opcode::IntOr => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_or(input1, input2))
            }
            Opcode::IntXor => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::int_xor(input1, input2))
            }
            Opcode::FloatIsNaN => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_nan(input, float_formats))
            }
            Opcode::FloatAbs => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_abs(input, float_formats))
            }
            Opcode::FloatNeg => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_neg(input, float_formats))
            }
            Opcode::FloatSqrt => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_sqrt(input, float_formats))
            }
            Opcode::FloatFloor => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_floor(input, float_formats))
            }
            Opcode::FloatCeiling => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_ceiling(input, float_formats))
            }
            Opcode::FloatRound => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_round(input, float_formats))
            }
            Opcode::FloatEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_eq(input1, input2, float_formats))
            }
            Opcode::FloatNotEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_neq(input1, input2, float_formats))
            }
            Opcode::FloatLess => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_lt(input1, input2, float_formats))
            }
            Opcode::FloatLessEq => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_le(input1, input2, float_formats))
            }
            Opcode::FloatAdd => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_add(input1, input2, float_formats))
            }
            Opcode::FloatSub => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_sub(input1, input2, float_formats))
            }
            Opcode::FloatDiv => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_div(input1, input2, float_formats))
            }
            Opcode::FloatMul => {
                let input1 = Expr::from_space(inputs.next().unwrap(), manager);
                let input2 = Expr::from_space(inputs.next().unwrap(), manager);
                let output = output.unwrap();

                Self::assign(output, Expr::float_mul(input1, input2, float_formats))
            }
            Opcode::FloatOfFloat => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let input_size = input.bits();

                let output = Var::from(output.unwrap());
                let output_size = output.bits();

                let input_format = float_formats[&input_size].clone();
                let output_format = float_formats[&output_size].clone();

                Self::assign(
                    output,
                    Expr::cast_float(Expr::cast_float(input, input_format), output_format),
                )
            }
            Opcode::FloatOfInt => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let input_size = input.bits();

                let output = Var::from(output.unwrap());
                let output_size = output.bits();

                let format = float_formats[&output_size].clone();
                Self::assign(
                    output,
                    Expr::cast_float(Expr::cast_signed(input, input_size), format),
                )
            }
            Opcode::FloatTruncate => {
                let input = Expr::from_space(inputs.next().unwrap(), manager);
                let input_size = input.bits();

                let output = Var::from(output.unwrap());
                let output_size = output.bits();

                let format = float_formats[&input_size].clone();
                Self::assign(
                    output,
                    Expr::cast_signed(Expr::cast_float(input, format), output_size),
                )
            }
            Opcode::Label => Self::skip(),
            Opcode::Build
            | Opcode::CrossBuild
            | Opcode::CPoolRef
            | Opcode::Piece
            | Opcode::Extract
            | Opcode::DelaySlot
            | Opcode::New
            | Opcode::Insert
            | Opcode::Cast
            | Opcode::SegmentOp => {
                panic!("unimplemented due to spec.")
            }
        }
    }
}

impl Stmt {
    pub fn assign<D, S>(destination: D, source: S) -> Term<Self>
    where
        D: Into<Var>,
        S: Into<Term<Expr>>,
    {
        let dest = destination.into();
        let bits = dest.bits();
        Self::Assign(dest, Expr::cast_unsigned(source, bits)).into()
    }

    pub fn store<D, S>(destination: D, source: S, size: usize, space: &AddressSpace) -> Term<Self>
    where
        D: Into<Term<Expr>>,
        S: Into<Term<Expr>>,
    {
        Self::Store(
            Expr::cast_unsigned(destination.into(), space.address_size() * 8),
            source.into(),
            size,
            space.id(),
        ).into()
    }

    pub fn branch<T>(target: T) -> Term<Self>
    where
        T: Into<Term<BranchTarget>>,
    {
        Self::Branch(target.into()).into()
    }

    pub fn branch_conditional<C, T>(condition: C, target: T) -> Term<Self>
    where
        C: Into<Term<Expr>>,
        T: Into<Term<BranchTarget>>,
    {
        Self::CBranch(Expr::cast_bool(condition), target.into()).into()
    }

    pub fn branch_indirect<T>(target: T, space: &AddressSpace) -> Term<Self>
    where
        T: Into<Term<Expr>>,
    {
        let vptr = Cast::pointer(Cast::void(), space.address_size() * 8);

        Self::Branch(BranchTarget::computed(Expr::Cast(
            target.into(),
            vptr,
        ))).into()
        /*
        Self::Branch(BranchTarget::computed(Expr::load(
            target,
            space.address_size() * 8,
            space,
        )))
        */
    }

    pub fn call<T>(target: T) -> Term<Self>
    where
        T: Into<Term<BranchTarget>>,
    {
        Self::Call(target.into(), Default::default()).into()
    }

    pub fn call_indirect<T>(target: T, space: &AddressSpace) -> Term<Self>
    where
        T: Into<Term<Expr>>,
    {
        let fptr = Cast::pointer(
            Cast::function(Cast::void(), std::iter::empty::<Term<Cast>>()),
            space.address_size() * 8,
        );

        Self::Call(
            BranchTarget::computed(Expr::Cast(target.into(), fptr)),
            Default::default(),
        ).into()
    }

    pub fn call_with<T, I, E>(target: T, arguments: I) -> Term<Self>
    where
        T: Into<Term<BranchTarget>>,
        I: ExactSizeIterator<Item = E>,
        E: Into<Term<Expr>>,
    {
        let mut args = SmallVec::with_capacity(arguments.len());
        for arg in arguments.map(|e| e.into()) {
            args.push(arg);
        }

        Self::Call(target.into(), args).into()
    }

    pub fn call_indirect_with<T, I, E>(target: T, space: &AddressSpace, arguments: I) -> Term<Self>
    where
        T: Into<Term<Expr>>,
        I: ExactSizeIterator<Item = E>,
        E: Into<Term<Expr>>,
    {
        let mut args = SmallVec::with_capacity(arguments.len());
        for arg in arguments.map(|e| e.into()) {
            args.push(arg);
        }

        let fptr = Cast::pointer(
            Cast::function(Cast::void(), std::iter::empty::<Term<Cast>>()),
            space.address_size() * 8,
        );

        Self::Call(
            BranchTarget::computed(Expr::Cast(target.into(), fptr)),
            args,
        ).into()
    }

    pub fn return_<T>(target: T, space: &AddressSpace) -> Term<Self>
    where
        T: Into<Term<Expr>>,
    {
        let vptr = Cast::pointer(Cast::void(), space.address_size() * 8);

        Self::Return(BranchTarget::computed(Expr::Cast(
            target.into(),
            vptr,
        ))).into()
        /*
            target,
            space.address_size() * 8,
            space,
        )))
        */
    }

    pub fn skip() -> Term<Self> {
        Self::Skip.into()
    }

    pub fn intrinsic<N, I, E>(name: N, arguments: I) -> Term<Self>
    where
        N: Into<Ustr>,
        I: ExactSizeIterator<Item = E>,
        E: Into<Term<Expr>>,
    {
        let mut args = SmallVec::with_capacity(arguments.len());
        for arg in arguments.map(|e| e.into()) {
            args.push(arg);
        }

        Self::Intrinsic(name.into(), args).into()
    }
}

impl<'z> FromSpace<'z, Operand> for Var {
    fn from_space_with(t: Operand, _arena: &'z IRBuilderArena, manager: &SpaceManager) -> Self {
        Var::from_space(t, manager)
    }

    fn from_space(operand: Operand, manager: &SpaceManager) -> Self {
        match operand {
            Operand::Address { value, size } => Var {
                offset: value.offset(),
                space: manager.default_space_id(),
                bits: size * 8,
                generation: 0,
            },
            Operand::Register { offset, size, .. } => Var {
                offset,
                space: manager.register_space_id(),
                bits: size * 8,
                generation: 0,
            },
            Operand::Variable {
                offset,
                space,
                size,
            } => Var {
                offset,
                space,
                bits: size * 8,
                generation: 0,
            },
            _ => panic!("cannot create Var from Operand::Constant"),
        }
    }
}

impl<'z> FromSpace<'z, Operand> for Expr {
    fn from_space_with(
        operand: Operand,
        _arena: &'z IRBuilderArena,
        manager: &SpaceManager,
    ) -> Self {
        Expr::from_space(operand, manager)
    }

    fn from_space(operand: Operand, manager: &SpaceManager) -> Self {
        if let Operand::Constant { value, size, .. } = operand {
            Expr::Val(BitVec::from_u64(value, size * 8))
        } else {
            Var::from_space(operand, manager).into()
        }
    }
}

impl<'z> FromSpace<'z, Operand> for Location {
    fn from_space_with(t: Operand, _arena: &'z IRBuilderArena, manager: &SpaceManager) -> Self {
        Location::from_space(t, manager)
    }

    fn from_space(operand: Operand, manager: &SpaceManager) -> Self {
        match operand {
            Operand::Address { value, .. } => Location {
                address: value.into_space(manager),
                position: 0,
            },
            Operand::Constant { value, .. } => Location {
                address: AddressValue::new(manager.constant_space_ref(), value),
                position: 0,
            },
            Operand::Register { offset, .. } => Location {
                address: AddressValue::new(manager.register_space_ref(), offset),
                position: 0,
            },
            Operand::Variable { offset, space, .. } => Location {
                address: AddressValue::new(manager.unchecked_space_by_id(space), offset),
                position: 0,
            },
        }
    }
}

impl Stmt {
    pub fn from_pcode(
        translator: &Translator,
        pcode: PCodeOp,
        address: &AddressValue,
        position: usize,
    ) -> Term<Self> {
        let manager = translator.manager();
        let formats = translator.float_formats();

        match pcode {
            PCodeOp::Copy {
                destination,
                source,
            } => Self::assign(
                Var::from_space(destination, manager),
                Expr::from_space(source, manager),
            ),
            PCodeOp::Load {
                destination,
                source,
                space,
            } => {
                let space = manager.unchecked_space_by_id(space);
                let size = destination.size() * 8;
                let src = if space.word_size() > 1 {
                    let s = Expr::from_space(source, manager);
                    let bits = s.bits();

                    let w = Expr::from(BitVec::from_usize(space.word_size(), bits));

                    Expr::int_mul(s, w)
                } else {
                    Expr::from_space(source, manager).into()
                };

                Self::assign(
                    Var::from_space(destination, manager),
                    Expr::load(src, size, space),
                )
            }
            PCodeOp::Store {
                destination,
                source,
                space,
            } => {
                let space = manager.unchecked_space_by_id(space);
                let size = source.size() * 8;

                let dest = if space.word_size() > 1 {
                    let d = Expr::from_space(destination, manager);
                    let bits = d.bits();

                    let w = Expr::from(BitVec::from_usize(space.word_size(), bits));

                    Expr::int_mul(d, w)
                } else {
                    Expr::from_space(destination, manager).into()
                };

                Self::store(dest, Expr::from_space(source, manager), size, space)
            }
            PCodeOp::Branch { destination } => {
                let mut target = Location::from_space(destination, manager);
                target.absolute_from(address.to_owned(), position);

                Self::branch(target)
            }
            PCodeOp::CBranch {
                condition,
                destination,
            } => {
                let mut target = Location::from_space(destination, manager);
                target.absolute_from(address.to_owned(), position);

                Self::branch_conditional(Expr::from_space(condition, manager), target)
            }
            PCodeOp::IBranch { destination } => {
                let space = manager.unchecked_space_by_id(address.space());

                Self::branch_indirect(Expr::from_space(destination, manager), space)
            }
            PCodeOp::Call { destination } => {
                let mut target = Location::from_space(destination, manager);
                target.absolute_from(address.to_owned(), position);

                Self::call(target)
            }
            PCodeOp::ICall { destination } => {
                let space = manager.unchecked_space_by_id(address.space());

                Self::call_indirect(Expr::from_space(destination, manager), space)
            }
            PCodeOp::Intrinsic {
                name,
                operands,
                result,
            } => {
                if let Some(result) = result {
                    let output = Var::from_space(result, manager);
                    let bits = output.bits();
                    Self::assign(
                        output,
                        Expr::intrinsic(
                            &*name,
                            operands.into_iter().map(|v| Expr::from_space(v, manager)),
                            bits,
                        ),
                    )
                } else {
                    Self::intrinsic(
                        &*name,
                        operands.into_iter().map(|v| Expr::from_space(v, manager)),
                    )
                }
            }
            PCodeOp::Return { destination } => {
                let space = manager.unchecked_space_by_id(address.space());

                Self::return_(Expr::from_space(destination, manager), space)
            }
            PCodeOp::Subpiece {
                operand,
                amount,
                result,
            } => {
                let source = Expr::from_space(operand, manager);
                let src_size = source.bits();
                let out_size = result.size() * 8;

                let loff = amount.offset() as usize * 8;
                let trun_size = src_size.checked_sub(loff).unwrap_or(0);

                let trun = if out_size > trun_size {
                    // extract high + expand
                    let source_htrun = Expr::extract_high(source, trun_size);
                    Expr::cast_unsigned(source_htrun, out_size)
                } else {
                    // extract
                    let hoff = loff + out_size;
                    Expr::extract(source, loff, hoff)
                };

                Self::assign(Var::from_space(result, manager), trun)
            }
            PCodeOp::PopCount { result, operand } => {
                let output = Var::from_space(result, manager);

                let size = output.bits();
                let popcount = Expr::unary_op(UnOp::POPCOUNT, Expr::from_space(operand, manager));

                Self::assign(output, Expr::cast_unsigned(popcount, size))
            }
            PCodeOp::BoolNot { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::bool_not(Expr::from_space(operand, manager)),
            ),
            PCodeOp::BoolAnd {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::bool_and(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::BoolOr {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::bool_or(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::BoolXor {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::bool_xor(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntNeg { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_neg(Expr::from_space(operand, manager)),
            ),
            PCodeOp::IntNot { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_not(Expr::from_space(operand, manager)),
            ),
            PCodeOp::IntSExt { result, operand } => {
                let size = result.size() * 8;
                Self::assign(
                    Var::from_space(result, manager),
                    Expr::cast_signed(Expr::from_space(operand, manager), size),
                )
            }
            PCodeOp::IntZExt { result, operand } => {
                let size = result.size() * 8;
                Self::assign(
                    Var::from_space(result, manager),
                    Expr::cast_unsigned(Expr::from_space(operand, manager), size),
                )
            }
            PCodeOp::IntEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_eq(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntNotEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_neq(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntLess {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_lt(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntLessEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_le(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSLess {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_slt(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSLessEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_sle(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntCarry {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_carry(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSCarry {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_scarry(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSBorrow {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_sborrow(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntAdd {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_add(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSub {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_sub(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntDiv {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_div(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSDiv {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_sdiv(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntMul {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_mul(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntRem {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_rem(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSRem {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_srem(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntLeftShift {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_shl(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntRightShift {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_shr(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntSRightShift {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_sar(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntAnd {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_and(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntOr {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_or(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::IntXor {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::int_xor(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                ),
            ),
            PCodeOp::FloatIsNaN { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_nan(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatAbs { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_abs(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatNeg { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_neg(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatSqrt { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_sqrt(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatFloor { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_floor(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatCeiling { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_ceiling(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatRound { result, operand } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_round(Expr::from_space(operand, manager), &formats),
            ),
            PCodeOp::FloatEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_eq(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatNotEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_neq(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatLess {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_lt(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatLessEq {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_le(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatAdd {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_add(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatSub {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_sub(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatDiv {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_div(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatMul {
                result,
                operands: [operand1, operand2],
            } => Self::assign(
                Var::from_space(result, manager),
                Expr::float_mul(
                    Expr::from_space(operand1, manager),
                    Expr::from_space(operand2, manager),
                    &formats,
                ),
            ),
            PCodeOp::FloatOfFloat { result, operand } => {
                let input = Expr::from_space(operand, manager);
                let input_size = input.bits();

                let output = Var::from_space(result, manager);
                let output_size = output.bits();

                let input_format = formats[&input_size].clone();
                let output_format = formats[&output_size].clone();

                Self::assign(
                    output,
                    Expr::cast_float(Expr::cast_float(input, input_format), output_format),
                )
            }
            PCodeOp::FloatOfInt { result, operand } => {
                let input = Expr::from_space(operand, manager);
                let input_size = input.bits();

                let output = Var::from_space(result, manager);
                let output_size = output.bits();

                let format = formats[&output_size].clone();
                Self::assign(
                    output,
                    Expr::cast_float(Expr::cast_signed(input, input_size), format),
                )
            }
            PCodeOp::FloatTruncate { result, operand } => {
                let input = Expr::from_space(operand, manager);
                let input_size = input.bits();

                let output = Var::from_space(result, manager);
                let output_size = output.bits();

                let format = formats[&input_size].clone();
                Self::assign(
                    output,
                    Expr::cast_signed(Expr::cast_float(input, format), output_size),
                )
            }
            PCodeOp::Skip => Self::skip(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ECode {
    pub address: AddressValue,
    pub operations: SmallVec<[Term<Stmt>; 8]>,
    pub delay_slots: usize,
    pub length: usize,
}

impl ECode {
    pub fn nop(address: AddressValue, length: usize) -> Self {
        Self {
            address,
            operations: smallvec![Stmt::skip()],
            delay_slots: 0,
            length,
        }
    }

    pub fn address(&self) -> AddressValue {
        self.address.clone()
    }

    pub fn operations(&self) -> &[Term<Stmt>] {
        self.operations.as_ref()
    }

    pub fn operations_mut(&mut self) -> &mut SmallVec<[Term<Stmt>; 8]> {
        &mut self.operations
    }

    pub fn delay_slots(&self) -> usize {
        self.delay_slots
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn display<'ecode, 'trans>(
        &'ecode self,
        translator: &'trans Translator,
    ) -> ECodeFormatter<'ecode, 'trans> {
        ECodeFormatter {
            ecode: self,
            translator,
        }
    }
}

impl ECode {
    pub fn from_pcode(translator: &Translator, pcode: PCode) -> Self {
        let address = pcode.address;
        let mut operations = SmallVec::with_capacity(pcode.operations.len());

        for (i, op) in pcode.operations.into_iter().enumerate() {
            operations.push(Stmt::from_pcode(translator, op, &address, i));
        }

        Self {
            operations,
            address,
            delay_slots: pcode.delay_slots,
            length: pcode.length,
        }
    }
}

impl fmt::Display for ECode {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let len = self.operations.len();
        if len > 0 {
            for (i, op) in self.operations.iter().enumerate() {
                write!(
                    f,
                    "{}.{:02}: {}{}",
                    self.address,
                    i,
                    op,
                    if i == len - 1 { "" } else { "\n" }
                )?;
            }
            Ok(())
        } else {
            write!(f, "{}.00: skip", self.address)
        }
    }
}

pub struct ECodeFormatter<'ecode, 'trans> {
    ecode: &'ecode ECode,
    translator: &'trans Translator,
}

impl<'ecode, 'trans> fmt::Display for ECodeFormatter<'ecode, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let len = self.ecode.operations.len();
        if len > 0 {
            for (i, op) in self.ecode.operations.iter().enumerate() {
                write!(
                    f,
                    "{}.{:02}: {}{}",
                    self.ecode.address,
                    i,
                    op.display_with(Some(self.translator)),
                    if i == len - 1 { "" } else { "\n" }
                )?;
            }
            Ok(())
        } else {
            write!(f, "{}.00: skip", self.ecode.address)
        }
    }
}
