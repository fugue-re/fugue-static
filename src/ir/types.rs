use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::traits::*;

use hashcons::hashconsing::consign;
use hashcons::Term;

use smallvec::SmallVec;
use ustr::Ustr;

use crate::ir::Expr;

consign! { let TYPE = consign(1024) for Type; }
consign! { let FFMT = consign(1024) for FloatFormat; }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct FloatKind(Term<FloatFormat>);

impl Borrow<FloatFormat> for FloatKind {
    fn borrow(&self) -> &FloatFormat {
        &*self.0
    }
}

impl Borrow<FloatFormat> for &'_ FloatKind {
    fn borrow(&self) -> &FloatFormat {
        &*self.0
    }
}

impl Deref for FloatKind {
    type Target = FloatFormat;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    Void,
    Bool, // T -> Bool

    Signed(usize),   // sign-extension
    Unsigned(usize), // zero-extension

    Float(FloatKind), // T -> FloatFormat::T

    Pointer(Term<Type>, usize),
    Function(Term<Type>, SmallVec<[Term<Type>; 4]>),
    Struct(Ustr, SmallVec<[(usize, Term<Type>); 4]>, usize), // name * [(offset * 't)] * bits
    Named(Ustr, usize),
}

impl From<Type> for Term<Type> {
    fn from(c: Type) -> Self {
        Term::new(&TYPE, c)
    }
}

impl From<FloatFormat> for FloatKind {
    fn from(c: FloatFormat) -> Self {
        Self(Term::new(&FFMT, c))
    }
}

impl Type {
    pub fn apply<E>(&self, e: E) -> Term<Expr>
    where
        E: Into<Term<Expr>>,
    {
        let e = e.into();

        match self {
            Self::Bool => Expr::cast_bool(e),
            Self::Signed(bits) => Expr::cast_signed(e, *bits),
            Self::Unsigned(bits) => Expr::cast_unsigned(e, *bits),
            Self::Pointer(_, bits) => {
                Expr::Cast(Expr::cast_unsigned(e, *bits), self.clone().into()).into()
            }
            _ => Expr::Cast(e, self.clone().into()).into(),
        }
    }

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

    pub fn is_float_kind<F>(&self, fmt: F) -> bool
    where
        F: Borrow<FloatFormat>,
    {
        matches!(self, Self::Float(f) if &**f == fmt.borrow())
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_, _))
    }

    pub fn is_pointer_with(&self, bits: usize) -> bool {
        matches!(self, Self::Pointer(_, b) if *b == bits)
    }

    pub fn is_pointer_kind<F>(&self, f: F) -> bool
    where
        F: Fn(&Self) -> bool,
    {
        matches!(self, Self::Pointer(t, _) if f(t))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_, _))
    }

    pub fn is_function_kind<F>(&self, f: F) -> bool
    where
        F: Fn(&Self, &[Term<Self>]) -> bool,
    {
        matches!(self, Self::Function(rt, ats) if f(rt, ats))
    }

    pub fn is_named(&self) -> bool {
        matches!(self, Self::Named(_, _))
    }

    pub fn is_named_with<F>(&self, f: F) -> bool
    where
        F: Fn(&str) -> bool,
    {
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

    pub fn float<K>(fmt: K) -> Term<Self>
    where
        K: Into<FloatKind>,
    {
        Self::Float(fmt.into()).into()
    }

    pub fn pointer<T>(t: T, bits: usize) -> Term<Self>
    where
        T: Into<Term<Self>>,
    {
        Self::Pointer(t.into(), bits).into()
    }

    pub fn function<R, I, A>(ret: R, args: I) -> Term<Self>
    where
        R: Into<Term<Self>>,
        I: ExactSizeIterator<Item = A>,
        A: Into<Term<Self>>,
    {
        Self::Function(ret.into(), args.map(|arg| arg.into()).collect()).into()
    }

    pub fn struct_<N, I, T>(name: N, fields: I, bytes: usize) -> Term<Self>
    where
        N: Into<Ustr>,
        I: ExactSizeIterator<Item = (usize, T)>,
        T: Into<Term<Self>>,
    {
        Self::Struct(
            name.into(),
            fields.into_iter().map(|(off, t)| (off, t.into())).collect(),
            bytes * 8,
        )
        .into()
    }

    pub fn named<N>(name: N, bytes: usize) -> Term<Self>
    where
        N: Into<Ustr>,
    {
        Self::Named(name.into(), bytes * 8).into()
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::Bool => write!(f, "bool"),
            Self::Float(format) => write!(f, "float{}", format.bits()),
            Self::Signed(bits) => write!(f, "int{}", bits),
            Self::Unsigned(bits) => write!(f, "uint{}", bits),
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
            Self::Struct(name, _, _) => write!(f, "{}", name),
            Self::Named(name, _) => write!(f, "{}", name),
        }
    }
}

impl BitSize for Type {
    fn bits(&self) -> usize {
        match self {
            Self::Void | Self::Function(_, _) => 0, // do not have a size
            Self::Bool => 1,
            Self::Float(format) => format.bits(),
            Self::Signed(bits)
            | Self::Unsigned(bits)
            | Self::Pointer(_, bits)
            | Self::Struct(_, _, bits) => *bits,
            Self::Named(_, bits) => *bits,
        }
    }
}
