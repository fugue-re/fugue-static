use fugue::ir::Translator;
use fugue::ir::il::ecode::{Location, Var};
use fugue::ir::il::traits::*;

use std::fmt::{self, Display};

use crate::types::{Entity, Located};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(serde::Deserialize, serde::Serialize)]
pub struct Phi {
    var: Var,
    vars: Vec<Var>
}

impl Display for Phi {
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

pub struct PhiDisplay<'a> {
    phi: &'a Phi,
    trans: Option<&'a Translator>,
}

impl<'a> Display for PhiDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.phi.vars.is_empty() {
            // NOTE: should never happen
            write!(f, "{} ← ϕ(<empty>)", self.phi.var.display_with(self.trans))?;
        } else {
            write!(f, "{} ← ϕ({}", self.phi.var.display_with(self.trans), self.phi.vars[0].display_with(self.trans))?;
            for aop in &self.phi.vars[1..] {
                write!(f, ", {}", aop.display_with(self.trans))?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl Phi {
    pub fn new(loc: impl Into<Location>, var: Var, vars: Vec<Var>) -> Entity<Located<Self>> {
        Entity::new("phi", Located::new(loc.into(), Phi { var, vars }))
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

impl<'phi, 'trans: 'phi> TranslatorDisplay<'phi, 'trans> for Phi {
    type Target = PhiDisplay<'phi>;

    fn display_with(&'phi self, t: Option<&'trans Translator>) -> PhiDisplay<'phi> {
        PhiDisplay { phi: self, trans: t }
    }
}
