use std::fmt;
use std::sync::Arc;

use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::traits::*;

use hashcons::hashconsing::consign;
use hashcons::Term;

use smallvec::SmallVec;
use ustr::Ustr;

consign! { let TYPE = consign(1024) for Type; }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    Void,
    Bool, // T -> Bool

    Signed(usize),   // sign-extension
    Unsigned(usize), // zero-extension

    Float(Arc<FloatFormat>), // T -> FloatFormat::T

    Pointer(Term<Type>, usize),
    Function(Term<Type>, SmallVec<[Term<Type>; 4]>),
    Named(Ustr, usize),
}

impl From<Type> for Term<Type> {
    fn from(c: Type) -> Self {
        Term::new(&TYPE, c)
    }
}

impl Type {
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

    pub fn float(fmt: Arc<FloatFormat>) -> Term<Self> {
        Self::Float(fmt).into()
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

    pub fn named<N>(name: N, bits: usize) -> Term<Self>
    where
        N: Into<Ustr>,
    {
        Self::Named(name.into(), bits).into()
    }
}

impl fmt::Display for Type {
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

impl BitSize for Type {
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
