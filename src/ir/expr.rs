use std::fmt;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::disassembly::{IRBuilderArena, VarnodeData};
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::pcode::Operand;
use fugue::ir::il::traits::*;
use fugue::ir::space::{AddressSpace, AddressSpaceId};
use fugue::ir::space_manager::{FromSpace, SpaceManager};
use fugue::ir::Translator;

use hashcons::hashconsing::consign;
use hashcons::Term;

use fnv::FnvHashMap as Map;
use smallvec::SmallVec;
use ustr::Ustr;

use crate::ir::{BranchTarget, Type, Var};

consign! { let EXPR = consign(1024) for Expr; }

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnRel {
    NAN,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    Cast(Term<Expr>, Term<Type>),            // T -> Type::T
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
              //Expr::Cast(Box::new(src), Type::Pointer(Box::new(Type::Void), asz))
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
    where
        F: Fn(&Type) -> bool,
    {
        matches!(self, Self::Cast(_, cst) if f(cst))
    }

    pub fn is_bool(&self) -> bool {
        self.is_cast_kind(Type::is_bool)
    }

    pub fn is_float(&self) -> bool {
        self.is_cast_kind(Type::is_float)
    }

    pub fn is_float_format(&self, format: &FloatFormat) -> bool {
        self.is_cast_kind(|f| f.is_float_format(format))
    }

    pub fn is_signed(&self) -> bool {
        self.is_cast_kind(Type::is_signed)
    }

    pub fn is_signed_bits(&self, bits: usize) -> bool {
        self.is_cast_kind(|s| s.is_signed_with(bits))
    }

    pub fn is_unsigned(&self) -> bool {
        self.is_cast_kind(|cst| matches!(cst, Type::Unsigned(_) | Type::Pointer(_, _)))
            || !matches!(self, Self::Cast(_, _))
    }

    pub fn is_unsigned_bits(&self, bits: usize) -> bool {
        self.is_cast_kind(
            |cst| matches!(cst, Type::Unsigned(sz) | Type::Pointer(_, sz) if *sz == bits),
        ) || (!matches!(self, Self::Cast(_, _)) && self.bits() == bits)
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
            Self::Cast(expr.into(), Type::bool()).into()
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
            Self::Cast(expr.into(), Type::signed(bits)).into()
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
            Self::Cast(expr.into(), Type::unsigned(bits)).into()
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
            Self::Cast(expr.into(), Type::float(format)).into()
        }
    }

    pub fn cast<E>(expr: E, bits: usize) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::Cast(expr.into(), Type::unsigned(bits)).into()
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

    pub(crate) fn binary_op_promote_as<E1, E2, F>(
        op: BinOp,
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
        Self::cast_bool(Self::BinRel(rel, expr1.into(), expr2.into()))
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
        )
        .into()
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
        )
        .into()
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

    pub fn float_eq<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::EQ, expr1, expr2, formats)
    }

    pub fn float_neq<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::NEQ, expr1, expr2, formats)
    }

    pub fn float_lt<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::LT, expr1, expr2, formats)
    }

    pub fn float_le<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_rel_promote_float(BinRel::LE, expr1, expr2, formats)
    }

    pub fn float_add<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::ADD, expr1, expr2, formats)
    }

    pub fn float_sub<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::SUB, expr1, expr2, formats)
    }

    pub fn float_div<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float(BinOp::DIV, expr1, expr2, formats)
    }

    pub fn float_mul<E1, E2>(
        expr1: E1,
        expr2: E2,
        formats: &Map<usize, Arc<FloatFormat>>,
    ) -> Term<Self>
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
