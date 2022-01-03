use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::disassembly::{IRBuilderArena, VarnodeData};
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::pcode::Operand;
use fugue::ir::il::traits::*;
use fugue::ir::space::{AddressSpace, AddressSpaceId};
use fugue::ir::space_manager::{FromSpace, SpaceManager};

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

impl UnOp {
    pub fn apply<E>(&self, e: E) -> Term<Expr>
    where
        E: Into<Term<Expr>>,
    {
        let e = e.into();

        match self {
            Self::NOT => {
                if e.is_bool() {
                    Expr::bool_not(e)
                } else {
                    Expr::int_not(e)
                }
            }
            Self::NEG => {
                if let Some(fmt) = e.float_kind() {
                    Expr::float_neg_with(e, fmt)
                } else {
                    Expr::int_neg(e)
                }
            }
            _ => Expr::unary_op(*self, e),
        }
    }
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

impl BinOp {
    pub fn is_associative(&self) -> bool {
        matches!(
            self,
            Self::AND | Self::OR | Self::XOR | Self::ADD | Self::MUL
        )
    }

    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            Self::AND | Self::OR | Self::XOR | Self::ADD | Self::MUL
        )
    }

    pub fn apply<E1, E2>(&self, l: E1, r: E2) -> Term<Expr>
    where
        E1: Into<Term<Expr>>,
        E2: Into<Term<Expr>>,
    {
        let l = l.into();
        let r = r.into();

        match self {
            Self::AND => {
                if l.is_bool() {
                    Expr::bool_and(l, r)
                } else {
                    Expr::int_and(l, r)
                }
            }
            Self::OR => {
                if l.is_bool() {
                    Expr::bool_or(l, r)
                } else {
                    Expr::int_or(l, r)
                }
            }
            Self::XOR => {
                if l.is_bool() {
                    Expr::bool_xor(l, r)
                } else {
                    Expr::int_xor(l, r)
                }
            }
            Self::ADD => {
                if let Some(fmt) = l.float_kind() {
                    Expr::float_add_with(l, r, fmt)
                } else {
                    Expr::int_add(l, r)
                }
            }
            Self::DIV => {
                if let Some(fmt) = l.float_kind() {
                    Expr::float_div_with(l, r, fmt)
                } else if l.is_signed() {
                    Expr::int_sdiv(l, r)
                } else {
                    Expr::int_div(l, r)
                }
            }
            Self::MUL => {
                if let Some(fmt) = l.float_kind() {
                    Expr::float_mul_with(l, r, fmt)
                } else {
                    Expr::int_mul(l, r)
                }
            }
            Self::REM => {
                if l.is_signed() {
                    Expr::int_srem(l, r)
                } else {
                    Expr::int_rem(l, r)
                }
            }
            Self::SUB => {
                if let Some(fmt) = l.float_kind() {
                    Expr::float_sub_with(l, r, fmt)
                } else {
                    Expr::int_sub(l, r)
                }
            }
            Self::SHL => Expr::int_shl(l, r),
            Self::SHR => {
                if l.is_signed() {
                    Expr::int_sar(l, r)
                } else {
                    Expr::int_shr(l, r)
                }
            }
            _ => Expr::BinOp(*self, l, r).into(),
        }
    }
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

impl BinRel {
    pub fn is_commutative(&self) -> bool {
        matches!(self, Self::EQ | Self::NEQ)
    }
}

// Total ordering over Expr terms gives: Val(.) < Var(.) < ...
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    Val(BitVec), // BitVec -> T
    Var(Var),    // String * usize -> T

    Load(Term<Expr>, usize, AddressSpaceId), // SPACE[T]:SIZE -> T
    Cast(Term<Expr>, Term<Type>),            // T -> Type::T

    UnOp(UnOp, Term<Expr>),   // T -> T
    UnRel(UnRel, Term<Expr>), // T -> bool

    BinOp(BinOp, Term<Expr>, Term<Expr>),   // T * T -> T
    BinRel(BinRel, Term<Expr>, Term<Expr>), // T * T -> bool

    IfElse(Term<Expr>, Term<Expr>, Term<Expr>), // if T then T else T

    Extract(Term<Expr>, usize, usize), // T T[LSB..MSB) -> T
    ExtractHigh(Term<Expr>, usize),
    ExtractLow(Term<Expr>, usize),

    Concat(Term<Expr>, Term<Expr>), // T * T -> T

    Call(Term<BranchTarget>, SmallVec<[Term<Expr>; 4]>, usize),
    Intrinsic(Ustr, SmallVec<[Term<Expr>; 4]>, usize),
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
    fn fmt_l1_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::Val(v) => write!(f, "{}", v.display_full(Cow::Borrowed(&*d.fmt)),),
            Expr::Var(v) => write!(f, "{}", v.display_full(Cow::Borrowed(&*d.fmt))),

            Expr::Intrinsic(name, args, _) => {
                write!(f, "{}{}{}(", d.fmt.keyword_start, name, d.fmt.keyword_end)?;
                if !args.is_empty() {
                    write!(f, "{}", args[0].display_full(Cow::Borrowed(&*d.fmt)))?;
                    for arg in &args[1..] {
                        write!(f, ", {}", arg.display_full(Cow::Borrowed(&*d.fmt)))?;
                    }
                }
                write!(f, ")")
            }

            Expr::ExtractHigh(expr, bits) => write!(
                f,
                "{}extract-high{}({}, {}bits{}={}{}{})",
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                expr.display_full(Cow::Borrowed(&*d.fmt)),
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                d.fmt.value_start,
                bits,
                d.fmt.value_end,
            ),
            Expr::ExtractLow(expr, bits) => write!(
                f,
                "{}extract-low{}({}, {}bits{}={}{}{})",
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                expr.display_full(Cow::Borrowed(&*d.fmt)),
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                d.fmt.value_start,
                bits,
                d.fmt.value_end,
            ),

            Expr::Cast(expr, t) => {
                expr.fmt_l1_with(f, d)?;
                write!(
                    f,
                    " {}as{} {}{}{}",
                    d.fmt.keyword_start, d.fmt.keyword_end, d.fmt.type_start, t, d.fmt.type_end
                )
            }

            Expr::Load(expr, bits, space) => {
                if let Some(trans) = d.fmt.translator {
                    let space = trans.manager().unchecked_space_by_id(*space);
                    write!(
                        f,
                        "{}{}{}[{}]:{}{}{}",
                        d.fmt.variable_start,
                        space.name(),
                        d.fmt.variable_end,
                        expr.display_full(Cow::Borrowed(&*d.fmt)),
                        d.fmt.value_start,
                        bits,
                        d.fmt.value_end,
                    )
                } else {
                    write!(
                        f,
                        "{}space{}[{}{}{}][{}]:{}{}{}",
                        d.fmt.variable_start,
                        d.fmt.variable_end,
                        d.fmt.value_start,
                        space.index(),
                        d.fmt.value_end,
                        expr.display_full(Cow::Borrowed(&*d.fmt)),
                        d.fmt.value_start,
                        bits,
                        d.fmt.value_end,
                    )
                }
            }

            Expr::Extract(expr, lsb, msb) => write!(
                f,
                "{}extract{}({}, {}from{}={}{}{}, {}to{}={}{}{})",
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                expr.display_full(Cow::Borrowed(&*d.fmt)),
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                d.fmt.value_start,
                lsb,
                d.fmt.value_end,
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                d.fmt.value_start,
                msb,
                d.fmt.value_end,
            ),

            Expr::UnOp(UnOp::ABS, expr) => {
                write!(
                    f,
                    "{}abs{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }
            Expr::UnOp(UnOp::SQRT, expr) => {
                write!(
                    f,
                    "{}sqrt{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }
            Expr::UnOp(UnOp::ROUND, expr) => {
                write!(
                    f,
                    "{}round{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }
            Expr::UnOp(UnOp::CEILING, expr) => {
                write!(
                    f,
                    "{}ceiling{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }
            Expr::UnOp(UnOp::FLOOR, expr) => {
                write!(
                    f,
                    "{}floor{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }
            Expr::UnOp(UnOp::POPCOUNT, expr) => {
                write!(
                    f,
                    "{}popcount{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }

            Expr::UnRel(UnRel::NAN, expr) => {
                write!(
                    f,
                    "{}is-nan{}({})",
                    d.fmt.keyword_start,
                    d.fmt.keyword_end,
                    expr.display_full(Cow::Borrowed(&*d.fmt))
                )
            }

            Expr::BinRel(BinRel::CARRY, e1, e2) => write!(
                f,
                "{}carry{}({}, {})",
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                e1.display_full(Cow::Borrowed(&*d.fmt)),
                e2.display_full(Cow::Borrowed(&*d.fmt))
            ),
            Expr::BinRel(BinRel::SCARRY, e1, e2) => write!(
                f,
                "{}scarry{}({}, {})",
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                e1.display_full(Cow::Borrowed(&*d.fmt)),
                e2.display_full(Cow::Borrowed(&*d.fmt))
            ),
            Expr::BinRel(BinRel::SBORROW, e1, e2) => write!(
                f,
                "{}sborrow{}({}, {})",
                d.fmt.keyword_start,
                d.fmt.keyword_end,
                e1.display_full(Cow::Borrowed(&*d.fmt)),
                e2.display_full(Cow::Borrowed(&*d.fmt))
            ),

            expr => write!(f, "({})", expr.display_full(Cow::Borrowed(&*d.fmt))),
        }
    }

    fn fmt_l2_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::UnOp(UnOp::NEG, expr) => {
                write!(f, "{}-{}", d.fmt.keyword_start, d.fmt.keyword_end)?;
                expr.fmt_l1_with(f, d)
            }
            Expr::UnOp(UnOp::NOT, expr) => {
                write!(f, "{}!{}", d.fmt.keyword_start, d.fmt.keyword_end)?;
                expr.fmt_l1_with(f, d)
            }
            expr => expr.fmt_l1_with(f, d),
        }
    }

    fn fmt_l3_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::MUL, e1, e2) => {
                e1.fmt_l3_with(f, d)?;
                write!(f, " {}*{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l2_with(f, d)
            }
            Expr::BinOp(BinOp::DIV, e1, e2) => {
                e1.fmt_l3_with(f, d)?;
                write!(f, " {}/{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l2_with(f, d)
            }
            Expr::BinOp(BinOp::SDIV, e1, e2) => {
                e1.fmt_l3_with(f, d)?;
                write!(f, " {}s/{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l2_with(f, d)
            }
            Expr::BinOp(BinOp::REM, e1, e2) => {
                e1.fmt_l3_with(f, d)?;
                write!(f, " {}%{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l2_with(f, d)
            }
            Expr::BinOp(BinOp::SREM, e1, e2) => {
                e1.fmt_l3_with(f, d)?;
                write!(f, " {}s%{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l2_with(f, d)
            }
            expr => expr.fmt_l2_with(f, d),
        }
    }

    fn fmt_l4_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::ADD, e1, e2) => {
                e1.fmt_l4_with(f, d)?;
                write!(f, " {}+{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l3_with(f, d)
            }
            Expr::BinOp(BinOp::SUB, e1, e2) => {
                e1.fmt_l4_with(f, d)?;
                write!(f, " {}-{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l3_with(f, d)
            }
            expr => expr.fmt_l3_with(f, d),
        }
    }

    fn fmt_l5_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::SHL, e1, e2) => {
                e1.fmt_l5_with(f, d)?;
                write!(f, " {}<<{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l4_with(f, d)
            }
            Expr::BinOp(BinOp::SHR, e1, e2) => {
                e1.fmt_l5_with(f, d)?;
                write!(f, " {}>>{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l4_with(f, d)
            }
            Expr::BinOp(BinOp::SAR, e1, e2) => {
                e1.fmt_l5_with(f, d)?;
                write!(f, " {}s>>{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l4_with(f, d)
            }
            expr => expr.fmt_l4_with(f, d),
        }
    }

    fn fmt_l6_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::LT, e1, e2) => {
                e1.fmt_l6_with(f, d)?;
                write!(f, " {}<{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l5_with(f, d)
            }
            Expr::BinRel(BinRel::LE, e1, e2) => {
                e1.fmt_l6_with(f, d)?;
                write!(f, " {}<={} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l5_with(f, d)
            }
            Expr::BinRel(BinRel::SLT, e1, e2) => {
                e1.fmt_l6_with(f, d)?;
                write!(f, " {}s<{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l5_with(f, d)
            }
            Expr::BinRel(BinRel::SLE, e1, e2) => {
                e1.fmt_l6_with(f, d)?;
                write!(f, " {}s<={} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l5_with(f, d)
            }
            expr => expr.fmt_l5_with(f, d),
        }
    }

    fn fmt_l7_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::EQ, e1, e2) => {
                e1.fmt_l7_with(f, d)?;
                write!(f, " {}=={} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l6_with(f, d)
            }
            Expr::BinRel(BinRel::NEQ, e1, e2) => {
                e1.fmt_l7_with(f, d)?;
                write!(f, " {}!={} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
                e2.fmt_l6_with(f, d)
            }
            expr => expr.fmt_l6_with(f, d),
        }
    }

    fn fmt_l8_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        if let Expr::BinOp(BinOp::AND, e1, e2) = self {
            e1.fmt_l8_with(f, d)?;
            write!(f, " {}&{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            e2.fmt_l7_with(f, d)
        } else {
            self.fmt_l7_with(f, d)
        }
    }

    fn fmt_l9_with(&'v self, f: &mut fmt::Formatter<'_>, d: &ExprFormatter<'v, 't>) -> fmt::Result {
        if let Expr::BinOp(BinOp::XOR, e1, e2) = self {
            e1.fmt_l9_with(f, d)?;
            write!(f, " {}^{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            e2.fmt_l8_with(f, d)
        } else {
            self.fmt_l8_with(f, d)
        }
    }

    fn fmt_l10_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        d: &ExprFormatter<'v, 't>,
    ) -> fmt::Result {
        if let Expr::BinOp(BinOp::OR, e1, e2) = self {
            e1.fmt_l10_with(f, d)?;
            write!(f, " {}|{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            e2.fmt_l9_with(f, d)
        } else {
            self.fmt_l9_with(f, d)
        }
    }

    fn fmt_l11_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        d: &ExprFormatter<'v, 't>,
    ) -> fmt::Result {
        if let Expr::Concat(e1, e2) = self {
            e1.fmt_l11_with(f, d)?;
            write!(f, " {}++{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            e2.fmt_l10_with(f, d)
        } else {
            self.fmt_l10_with(f, d)
        }
    }

    fn fmt_l12_with(
        &'v self,
        f: &mut fmt::Formatter<'_>,
        d: &ExprFormatter<'v, 't>,
    ) -> fmt::Result {
        if let Expr::IfElse(c, et, ef) = self {
            write!(f, "{}if{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            c.fmt_l12_with(f, d)?;
            write!(f, " {}then{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            et.fmt_l12_with(f, d)?;
            write!(f, " {}else{} ", d.fmt.keyword_start, d.fmt.keyword_end)?;
            ef.fmt_l12_with(f, d)
        } else {
            self.fmt_l11_with(f, d)
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
    fmt: Cow<'trans, TranslatorFormatter<'trans>>,
}

impl<'expr, 'trans> fmt::Display for ExprFormatter<'expr, 'trans> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expr.fmt_l12_with(f, self)
    }
}

impl<'expr, 'trans> TranslatorDisplay<'expr, 'trans> for Expr {
    type Target = ExprFormatter<'expr, 'trans>;

    fn display_full(
        &'expr self,
        fmt: Cow<'trans, TranslatorFormatter<'trans>>,
    ) -> ExprFormatter<'expr, 'trans> {
        ExprFormatter { expr: self, fmt }
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
        match self {
            Expr::BinRel(_, _, _) | Expr::UnRel(_, _) => true,
            Expr::BinOp(_, e, _) | Expr::UnOp(_, e) | Expr::IfElse(_, e, _) => e.is_bool(),
            Expr::Cast(_, t) => t.is_bool(),
            _ => false,
        }
    }

    fn is_cast_kind_aux_op<F>(&self, f: F) -> bool
    where
        F: Fn(&Type) -> bool,
    {
        match self {
            Expr::BinOp(_, e, _) | Expr::UnOp(_, e) | Expr::IfElse(_, e, _) => {
                e.is_cast_kind_aux_op(f)
            }
            Expr::Cast(_, t) => f(t),
            _ => false,
        }
    }

    fn float_kind(&self) -> Option<Arc<FloatFormat>> {
        match self {
            Expr::BinOp(_, e, _) | Expr::UnOp(_, e) | Expr::IfElse(_, e, _) => e.float_kind(),
            Expr::Cast(_, t) => {
                if let Type::Float(fmt) = &**t {
                    Some(fmt.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn is_signed(&self) -> bool {
        self.is_cast_kind_aux_op(Type::is_signed)
    }

    pub fn is_signed_bits(&self, bits: usize) -> bool {
        self.is_cast_kind_aux_op(|s| s.is_signed_with(bits))
    }

    pub fn is_float(&self) -> bool {
        self.is_cast_kind_aux_op(Type::is_float)
    }

    pub fn is_float_format(&self, format: &FloatFormat) -> bool {
        self.is_cast_kind_aux_op(|f| f.is_float_format(format))
    }

    pub fn is_unsigned(&self) -> bool {
        match self {
            Expr::BinRel(_, _, _) | Expr::UnRel(_, _) => false,
            Expr::Cast(_, t) => matches!(**t, Type::Unsigned(_) | Type::Pointer(_, _)),
            Expr::UnOp(_, e) | Expr::BinOp(_, e, _) | Expr::IfElse(_, e, _) => e.is_unsigned(),
            _ => true,
        }
    }

    pub fn is_unsigned_bits(&self, bits: usize) -> bool {
        match self {
            Expr::BinRel(_, _, _) | Expr::UnRel(_, _) => false,
            Expr::Cast(_, t) => matches!(**t, Type::Unsigned(n) | Type::Pointer(_, n) if n == bits),
            Expr::UnOp(_, e) | Expr::BinOp(_, e, _) | Expr::IfElse(_, e, _) => {
                e.is_unsigned() && e.bits() == bits
            }
            e => e.bits() == bits,
        }
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
        if let Self::Val(e) = &*expr {
            Self::val(if bits >= e.bits() {
                e.unsigned_cast(bits)
            } else {
                (&*e >> (e.bits() as u32 - bits as u32))
                    .unsigned()
                    .cast(bits)
            })
        } else if expr.is_unsigned_bits(bits) {
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
        if let Self::Val(e) = &*expr {
            Self::val(e.unsigned_cast(bits))
        } else if expr.is_unsigned_bits(bits) {
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

        match (&*lhs, &*rhs) {
            (Self::Val(h), Self::Val(l)) => Self::val({
                let bits = h.bits() + l.bits();
                (h.unsigned_cast(bits) << l.bits() as u32) | l.unsigned_cast(bits)
            }),
            _ => Self::Concat(lhs, rhs).into(),
        }
    }

    pub(crate) fn unary_op<E>(op: UnOp, expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        Self::UnOp(op, expr.into()).into()
    }

    pub fn unary_op_with<E, F>(op: UnOp, expr: E, eval: F) -> Term<Self>
    where
        E: Into<Term<Self>>,
        F: Fn(&Term<Self>) -> Option<Term<Self>>,
    {
        let expr = expr.into();

        eval(&expr).unwrap_or_else(|| Self::unary_op(op, expr))
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

    pub fn binary_op_promote_as<E1, E2, F, G>(
        op: BinOp,
        expr1: E1,
        expr2: E2,
        cast: F,
        eval: G,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
        F: Fn(Term<Self>, usize) -> Term<Self>,
        G: Fn(&Term<Self>, &Term<Self>) -> Option<Term<Self>>,
    {
        let e1 = expr1.into();
        let e2 = expr2.into();

        let bits = e1.bits().max(e2.bits());

        let v1 = cast(e1, bits);
        let v2 = cast(e2, bits);

        eval(&v1, &v2).unwrap_or_else(|| Self::binary_op(op, v1, v2))
    }

    pub fn binary_op_promote<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(
            op,
            expr1,
            expr2,
            |e, sz| Self::cast_unsigned(e, sz),
            |_, _| None,
        )
    }

    pub fn binary_op_promote_with<E1, E2, F>(op: BinOp, expr1: E1, expr2: E2, eval: F) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
        F: Fn(&Term<Self>, &Term<Self>) -> Option<Term<Self>>,
    {
        Self::binary_op_promote_as(op, expr1, expr2, |e, sz| Self::cast_unsigned(e, sz), eval)
    }

    pub(crate) fn binary_op_promote_bool<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(op, expr1, expr2, |e, _sz| Self::cast_bool(e), |_, _| None)
    }

    pub(crate) fn binary_op_promote_signed<E1, E2>(op: BinOp, expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(
            op,
            expr1,
            expr2,
            |e, sz| Self::cast_signed(e, sz),
            |_, _| None,
        )
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
        Self::binary_op_promote_as(
            op,
            expr1,
            expr2,
            |e, sz| Self::cast_float(Self::cast_signed(e, sz), formats[&sz].clone()),
            |_, _| None,
        )
    }

    pub(crate) fn binary_op_promote_float_with<E1, E2>(
        op: BinOp,
        expr1: E1,
        expr2: E2,
        format: Arc<FloatFormat>,
    ) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_as(
            op,
            expr1,
            expr2,
            |e, _sz| Self::cast_float(Self::cast_signed(e, format.bits()), format.clone()),
            |_, _| None,
        )
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
        let expr = expr.into();
        if let Self::Val(e) = &*expr {
            Self::val(if loff > 0 {
                (e >> loff as u32).unsigned_cast(moff - loff)
            } else {
                e.unsigned_cast(moff - loff)
            })
        } else {
            Self::Extract(expr, loff, moff).into()
        }
    }

    pub fn ite<C, E1, E2>(cond: C, texpr: E1, fexpr: E2) -> Term<Self>
    where
        C: Into<Term<Self>>,
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        let cond = Self::cast_bool(cond);
        let e1 = texpr.into();
        let e2 = fexpr.into();

        assert_eq!(e1.bits(), e2.bits());

        if let Self::Val(v) = &*cond {
            if v.is_zero() {
                e2
            } else {
                e1
            }
        } else if e1 == e2 {
            e1
        } else {
            Self::IfElse(cond, e1, e2).into()
        }
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

    pub fn float_neg_with<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::unary_op(
            UnOp::NEG,
            if expr.is_float_format(&*format) {
                expr
            } else {
                Expr::cast_float(Expr::cast_signed(expr, format.bits()), format)
            },
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

    pub fn float_abs_with<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::unary_op(
            UnOp::ABS,
            if expr.is_float_format(&*format) {
                expr
            } else {
                Expr::cast_float(Expr::cast_signed(expr, format.bits()), format)
            },
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

    pub fn float_sqrt_with<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::unary_op(
            UnOp::SQRT,
            if expr.is_float_format(&*format) {
                expr
            } else {
                Expr::cast_float(Expr::cast_signed(expr, format.bits()), format)
            },
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

    pub fn float_ceiling_with<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::unary_op(
            UnOp::CEILING,
            if expr.is_float_format(&*format) {
                expr
            } else {
                Expr::cast_float(Expr::cast_signed(expr, format.bits()), format)
            },
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

    pub fn float_round_with<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::unary_op(
            UnOp::ROUND,
            if expr.is_float_format(&*format) {
                expr
            } else {
                Expr::cast_float(Expr::cast_signed(expr, format.bits()), format)
            },
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

    pub fn float_floor_with<E>(expr: E, format: Arc<FloatFormat>) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        Self::unary_op(
            UnOp::FLOOR,
            if expr.is_float_format(&*format) {
                expr
            } else {
                Expr::cast_float(Expr::cast_signed(expr, format.bits()), format)
            },
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

    pub fn float_add_with<E1, E2>(expr1: E1, expr2: E2, format: Arc<FloatFormat>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float_with(BinOp::ADD, expr1, expr2, format)
    }

    pub fn float_sub_with<E1, E2>(expr1: E1, expr2: E2, format: Arc<FloatFormat>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float_with(BinOp::SUB, expr1, expr2, format)
    }

    pub fn float_div_with<E1, E2>(expr1: E1, expr2: E2, format: Arc<FloatFormat>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float_with(BinOp::DIV, expr1, expr2, format)
    }

    pub fn float_mul_with<E1, E2>(expr1: E1, expr2: E2, format: Arc<FloatFormat>) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_float_with(BinOp::MUL, expr1, expr2, format)
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

        Self::unary_op_with(UnOp::NEG, Self::cast_unsigned(expr, size), |v| match &**v {
            Expr::Val(v) => Some(Expr::val(-v)),
            _ => None,
        })
    }

    pub fn int_not<E>(expr: E) -> Term<Self>
    where
        E: Into<Term<Self>>,
    {
        let expr = expr.into();
        let size = expr.bits();

        Self::unary_op_with(UnOp::NOT, Self::cast_unsigned(expr, size), |v| match &**v {
            Expr::Val(v) => Some(Expr::val(!v)),
            _ => None,
        })
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
        Self::binary_op_promote_with(BinOp::ADD, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv + rv)),
            (Expr::Val(v), _) if v.is_zero() => Some(r.clone()),
            (_, Expr::Val(v)) if v.is_zero() => Some(l.clone()),
            _ => None,
        })
    }

    pub fn int_sub<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_with(BinOp::SUB, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv - rv)),
            _ => None,
        })
    }

    pub fn int_mul<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_with(BinOp::MUL, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv * rv)),
            (Expr::Val(v), _) if v.is_zero() => Some(l.clone()),
            (_, Expr::Val(v)) if v.is_zero() => Some(r.clone()),
            (Expr::Val(v), _) if v.is_one() => Some(r.clone()),
            (_, Expr::Val(v)) if v.is_one() => Some(l.clone()),
            _ => None,
        })
    }

    pub fn int_div<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_with(BinOp::DIV, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => {
                if rv.is_zero() {
                    None
                } else {
                    Some(Expr::val(lv / rv))
                }
            }
            _ => None,
        })
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
        Self::binary_op_promote_with(BinOp::REM, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => {
                if rv.is_zero() {
                    None
                } else {
                    Some(Expr::val(lv % rv))
                }
            }
            _ => None,
        })
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
        Self::binary_op_promote_with(BinOp::SHL, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv << rv)),
            (_, Expr::Val(v)) if v.is_zero() => Some(l.clone()),
            _ => None,
        })
    }

    pub fn int_shr<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_with(BinOp::SHR, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv >> rv)),
            (_, Expr::Val(v)) if v.is_zero() => Some(l.clone()),
            _ => None,
        })
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
        Self::binary_op_promote_with(BinOp::AND, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv & rv)),
            (Expr::Val(v), _) if v.is_zero() => Some(l.clone()),
            (_, Expr::Val(v)) if v.is_zero() => Some(r.clone()),
            _ if l == r => Some(l.clone()),
            _ => None,
        })
    }

    pub fn int_or<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_with(BinOp::OR, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv | rv)),
            (Expr::Val(v), _) if v.is_zero() => Some(r.clone()),
            (_, Expr::Val(v)) if v.is_zero() => Some(l.clone()),
            _ if l == r => Some(l.clone()),
            _ => None,
        })
    }

    pub fn int_xor<E1, E2>(expr1: E1, expr2: E2) -> Term<Self>
    where
        E1: Into<Term<Self>>,
        E2: Into<Term<Self>>,
    {
        Self::binary_op_promote_with(BinOp::XOR, expr1, expr2, |l, r| match (&**l, &**r) {
            (Expr::Val(lv), Expr::Val(rv)) => Some(Expr::val(lv ^ rv)),
            (Expr::Val(v), _) if v.is_zero() => Some(r.clone()),
            (_, Expr::Val(v)) if v.is_zero() => Some(l.clone()),
            _ if l == r => Some(Expr::val(BitVec::zero(l.bits()))),
            _ => None,
        })
    }

    pub fn val<V>(v: V) -> Term<Self>
    where
        V: Into<BitVec>,
    {
        Self::Val(v.into()).into()
    }

    pub fn var<V>(v: V) -> Term<Self>
    where
        V: Into<Var>,
    {
        Self::Var(v.into()).into()
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

impl Expr {
    pub fn is_val(&self) -> bool {
        matches!(self, Expr::Val(_))
    }

    pub fn is_val_with<F>(&self, f: F) -> bool
    where
        F: FnOnce(&BitVec) -> bool,
    {
        matches!(self, Expr::Val(v) if f(v))
    }

    pub fn is_var(&self) -> bool {
        matches!(self, Expr::Var(_))
    }

    pub fn is_var_with<F>(&self, f: F) -> bool
    where
        F: FnOnce(&Var) -> bool,
    {
        matches!(self, Expr::Var(v) if f(v))
    }

    pub fn is_binop(&self, op: BinOp) -> bool {
        matches!(self, Expr::BinOp(op1, _, _) if op == *op1)
    }

    pub fn is_binop_with<F>(&self, op: BinOp, f: F) -> bool
    where
        F: FnOnce(&Expr, &Expr) -> bool,
    {
        matches!(self, Expr::BinOp(op1, l, r) if op == *op1 && f(&*l, &*r))
    }

    pub fn is_binop_llr_with<F>(&self, op: BinOp, f: F) -> bool
    where
        F: FnOnce(&Expr, &Expr, &Expr) -> bool,
    {
        matches!(self, Expr::BinOp(op1, l, r) if op == *op1 &&
                 matches!(&**l, Expr::BinOp(op2, ll, lr) if op == *op2 && f(ll, lr, r)))
    }

    pub fn is_binop_lrr_with<F>(&self, op: BinOp, f: F) -> bool
    where
        F: FnOnce(&Expr, &Expr, &Expr) -> bool,
    {
        matches!(self, Expr::BinOp(op1, l, r) if op == *op1 &&
                 matches!(&**r, Expr::BinOp(op2, rl, rr) if op == *op2 && f(l, rl, rr)))
    }

    fn comm3_reduce<'a>(
        op: BinOp,
        a: &'a Term<Expr>,
        b: &'a Term<Expr>,
        c: &'a Term<Expr>,
    ) -> (bool, Term<Expr>) {
        let mut t = [a, b, c];

        let sorted = **a >= **b && **b >= **c;

        if **t[0] < **t[1] {
            t.swap(0, 1);
        }
        if **t[1] < **t[2] {
            t.swap(1, 2);
        }
        if **t[0] < **t[1] {
            t.swap(0, 1);
        }

        let v = if b.is_val() && c.is_val() {
            op.apply(a.clone(), op.apply(b.clone(), c.clone()))
        } else {
            op.apply(op.apply(a.clone(), b.clone()), c.clone())
        };

        (sorted, v)
    }

    pub fn group_left(&self) -> Term<Self> {
        match self {
            Self::BinOp(op1, l, r) => match &**r {
                Self::BinOp(op2, rl, rr) if *op1 == *op2 => if op1.is_commutative() {
                    Self::comm3_reduce(*op1, l, rl, rr).1
                } else {
                    op1.apply(op1.apply(l.clone(), rl.clone()), rr.clone())
                }
                .canonical(),
                _ if op1.is_commutative() => match &**l {
                    Self::BinOp(op2, ll, rl) if *op1 == *op2 => {
                        let (sorted, t) = Self::comm3_reduce(*op1, ll, rl, r);
                        if !sorted {
                            t.canonical()
                        } else {
                            self.clone().into()
                        }
                    }
                    _ => self.clone().into(),
                },
                _ => self.clone().into(),
            },
            _ => self.clone().into(),
        }
    }

    // NOTES: rewrites expressions such that:
    // - a op b becomes b op a if a < b and op is comm.
    // - a op (b op c) becomes (a op b) op c if op is assoc.
    // ...
    //
    // The goal of canonicalisation is to produce expressions that are in the form
    // X + a, where a is some constant and X is some expression. Since we have a total
    // order over expressions, we can enforce the form Var + Val.
    pub fn canonical(&self) -> Term<Self> {
        match self {
            Self::BinOp(op, l, r) => {
                let lx = l.canonical();
                let rx = r.canonical();

                let nx = if op.is_commutative() {
                    let (nlx, nrx) = if *lx < *rx { (rx, lx) } else { (lx, rx) };
                    op.apply(nlx, nrx)
                } else {
                    op.apply(lx, rx)
                };

                if op.is_associative() {
                    nx.group_left()
                } else {
                    nx.into()
                }
            }
            t => t.clone().into(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_canon() {
        let spc = AddressSpace::unique("uniq", 0, None);

        let x = Expr::int_add(
            Expr::int_add(
                Expr::val(10u32),
                Expr::int_mul(
                    Expr::int_add(Expr::val(99u32), Expr::val(1u32)),
                    Expr::var(Var::new(&spc, 0, 32, 0)),
                ),
            ),
            Expr::int_add(
                Expr::int_add(
                    Expr::val(44u32),
                    Expr::int_mul(Expr::var(Var::new(&spc, 4, 32, 0)), Expr::val(0u32)),
                ),
                Expr::val(32u32),
            ),
        )
        .canonical();

        let y = Expr::int_add(
            Expr::val(10u32),
            Expr::int_add(
                Expr::val(99u32),
                Expr::int_add(Expr::val(44u32), Expr::val(32u32)),
            ),
        )
        .canonical();

        assert_eq!(
            x,
            Expr::int_add(
                Expr::int_mul(Expr::var(Var::new(&spc, 0, 32, 0)), Expr::val(100u32)),
                Expr::val(86u32),
            ),
        );
        assert_eq!(y, Expr::val(185u32),);
    }
}
