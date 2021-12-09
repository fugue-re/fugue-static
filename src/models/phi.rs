use fugue::ir::il::ecode::{Location, Var};
use fugue::ir::il::traits::*;
use fugue::ir::Translator;

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
    trans: Option<&'trans Translator>,
}

impl<'a, 'trans, Var> Display for PhiDisplay<'a, 'trans, Var>
where
    Var: TranslatorDisplay<'a, 'trans>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.phi.vars.is_empty() {
            // NOTE: should never happen
            write!(f, "{} ← ϕ(<empty>)", self.phi.var.display_with(self.trans))?;
        } else {
            write!(
                f,
                "{} ← ϕ({}",
                self.phi.var.display_with(self.trans),
                self.phi.vars[0].display_with(self.trans)
            )?;
            for aop in &self.phi.vars[1..] {
                write!(f, ", {}", aop.display_with(self.trans))?;
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
}

impl<'phi, 'trans, Var> TranslatorDisplay<'phi, 'trans> for PhiT<Var>
where
    Var: TranslatorDisplay<'phi, 'trans> + 'phi,
{
    type Target = PhiDisplay<'phi, 'trans, Var>;

    fn display_with(&'phi self, t: Option<&'trans Translator>) -> PhiDisplay<'phi, 'trans, Var> {
        PhiDisplay {
            phi: self,
            trans: t,
        }
    }
}
