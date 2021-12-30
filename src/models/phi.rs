use fugue::ir::il::ecode::{Location, Var};
use fugue::ir::il::traits::*;

use std::borrow::Cow;
use std::fmt::{self, Display};

use crate::types::{Entity, Located};

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct PhiT<Var> {
    var: Var,
    vars: Vec<Var>,
}

pub type Phi = PhiT<Var>;

impl<Var> Display for PhiT<Var>
where
    Var: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.vars.is_empty() {
            // NOTE: should never happen
            write!(f, "{} ← ϕ(<empty>)", self.var)?;
        } else {
            write!(f, "{} ← ϕ({}", self.var, self.vars[0])?;
            for aop in &self.vars[1..] {
                write!(f, ", {}", aop)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

pub struct PhiDisplay<'a, 'trans, Var> {
    phi: &'a PhiT<Var>,
    fmt: Cow<'trans, TranslatorFormatter<'trans>>,
}

impl<'a, 'trans, Var> Display for PhiDisplay<'a, 'trans, Var>
where
    Var: for<'t> TranslatorDisplay<'a, 't>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.phi.vars.is_empty() {
            // NOTE: should never happen
            write!(
                f,
                "{} {}←{} {}ϕ{}(<empty>)",
                self.phi.var.display_full(Cow::Borrowed(&*self.fmt)),
                self.fmt.keyword_start,
                self.fmt.keyword_end,
                self.fmt.keyword_start,
                self.fmt.keyword_end,
            )?;
        } else {
            write!(
                f,
                "{} {}←{} {}ϕ{}({}",
                self.phi.var.display_full(Cow::Borrowed(&*self.fmt)),
                self.fmt.keyword_start,
                self.fmt.keyword_end,
                self.fmt.keyword_start,
                self.fmt.keyword_end,
                self.phi.vars[0].display_full(Cow::Borrowed(&*self.fmt)),
            )?;
            for aop in &self.phi.vars[1..] {
                write!(
                    f,
                    ", {}",
                    aop.display_full(Cow::Borrowed(&*self.fmt)),
                )?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl<Var> PhiT<Var>
where
    Var: Clone,
{
    pub fn new(loc: impl Into<Location>, var: Var, vars: Vec<Var>) -> Entity<Located<Self>> {
        Entity::new("phi", Located::new(loc.into(), PhiT { var, vars }))
    }

    pub fn var(&self) -> &Var {
        &self.var
    }

    pub fn var_mut(&mut self) -> &mut Var {
        &mut self.var
    }

    pub fn assign(&self) -> &[Var] {
        &self.vars
    }

    pub fn assign_mut(&mut self) -> &mut Vec<Var> {
        &mut self.vars
    }

    pub fn parts_mut(&mut self) -> (&mut Var, &mut Vec<Var>) {
        (&mut self.var, &mut self.vars)
    }

    pub fn translate<Loc, Val, T: TranslateIR<Loc, Val, Var>>(self, t: &T) -> PhiT<T::TVar> {
        PhiT {
            var: t.translate_var(self.var),
            vars: self.vars.into_iter().map(|v| t.translate_var(v)).collect(),
        }
    }
}

impl<'phi, 'trans, Var> TranslatorDisplay<'phi, 'trans> for PhiT<Var>
where
    Var: for<'a> TranslatorDisplay<'phi, 'a> + 'phi,
{
    type Target = PhiDisplay<'phi, 'trans, Var>;

    fn display_full(
        &'phi self,
        fmt: Cow<'trans, TranslatorFormatter<'trans>>,
    ) -> Self::Target {
        PhiDisplay {
            phi: self,
            fmt,
        }
    }
}
