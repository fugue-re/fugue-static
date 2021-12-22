use std::fmt;
use std::sync::Arc;

use fugue::bv::BitVec;
use fugue::ir::address::AddressValue;
use fugue::ir::disassembly::{ArenaVec, Opcode, VarnodeData};
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::traits::*;
use fugue::ir::il::pcode::PCodeOp;
use fugue::ir::space::{AddressSpace, AddressSpaceId};
use fugue::ir::space_manager::{FromSpace, IntoSpace, SpaceManager};
use fugue::ir::Translator;

use hashcons::hashconsing::consign;
use hashcons::Term;

use fnv::FnvHashMap as Map;
use smallvec::SmallVec;
use ustr::Ustr;

use crate::ir::{BranchTarget, Expr, Location, Type, UnOp, Var};

consign! { let STMT = consign(1024) for Stmt; }

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
    pub fn from_parts<'a>(
        manager: &SpaceManager,
        float_formats: &Map<usize, Arc<FloatFormat>>,
        user_ops: &[Arc<str>],
        address: &AddressValue,
        position: usize,
        opcode: Opcode,
        inputs: ArenaVec<'a, VarnodeData>,
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
        )
        .into()
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
        let vptr = Type::pointer(Type::void(), space.address_size() * 8);

        Self::Branch(BranchTarget::computed(Expr::Cast(target.into(), vptr))).into()
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
        let fptr = Type::pointer(
            Type::function(Type::void(), std::iter::empty::<Term<Type>>()),
            space.address_size() * 8,
        );

        Self::Call(
            BranchTarget::computed(Expr::Cast(target.into(), fptr)),
            Default::default(),
        )
        .into()
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

        let fptr = Type::pointer(
            Type::function(Type::void(), std::iter::empty::<Term<Type>>()),
            space.address_size() * 8,
        );

        Self::Call(
            BranchTarget::computed(Expr::Cast(target.into(), fptr)),
            args,
        )
        .into()
    }

    pub fn return_<T>(target: T, space: &AddressSpace) -> Term<Self>
    where
        T: Into<Term<Expr>>,
    {
        let vptr = Type::pointer(Type::void(), space.address_size() * 8);

        Self::Return(BranchTarget::computed(Expr::Cast(target.into(), vptr))).into()
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

    pub fn is_branch(&self) -> bool {
        matches!(self,
                 Stmt::Branch(_) |
                 Stmt::CBranch(_, _) |
                 Stmt::Call(_, _) |
                 Stmt::Intrinsic(_, _) |
                 Stmt::Return(_))
    }

    pub fn is_jump(&self) -> bool {
        matches!(self, Stmt::Branch(_) | Stmt::CBranch(_, _))
    }

    pub fn is_cond(&self) -> bool {
        matches!(self, Stmt::CBranch(_, _))
    }

    pub fn is_call(&self) -> bool {
        matches!(self, Stmt::Call(_, _))
    }

    pub fn is_intrinsic(&self) -> bool {
        matches!(self, Stmt::Intrinsic(_, _))
    }

    pub fn has_fall(&self) -> bool {
        !matches!(self, Stmt::Branch(_) | Stmt::Return(_))
    }

    pub fn is_return(&self) -> bool {
        matches!(self, Stmt::Return(_))
    }

    pub fn is_skip(&self) -> bool {
        matches!(self, Stmt::Skip)
    }

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
