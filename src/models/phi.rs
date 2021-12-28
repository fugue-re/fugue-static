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
    keyword_start: &'trans str,
    keyword_end: &'trans str,
    location_start: &'trans str,
    location_end: &'trans str,
    value_start: &'trans str,
    value_end: &'trans str,
    variable_start: &'trans str,
    variable_end: &'trans str,
}

impl<'a, 'trans, Var> Display for PhiDisplay<'a, 'trans, Var>
where
    Var: TranslatorDisplay<'a, 'trans>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.phi.vars.is_empty() {
            // NOTE: should never happen
            write!(
                f,
                "{} ← {}ϕ{}(<empty>)",
                self.phi.var.display_full(
                    self.trans,
                    self.keyword_start,
                    self.keyword_end,
                    self.location_start,
                    self.location_end,
                    self.value_start,
                    self.value_end,
                    self.variable_start,
                    self.variable_end,
                ),
                self.keyword_start,
                self.keyword_end,
            )?;
        } else {
            write!(
                f,
                "{} ← {}ϕ{}({}",
                self.phi.var.display_full(
                    self.trans,
                    self.keyword_start,
                    self.keyword_end,
                    self.location_start,
                    self.location_end,
                    self.value_start,
                    self.value_end,
                    self.variable_start,
                    self.variable_end,
                ),
                self.keyword_start,
                self.keyword_end,
                self.phi.vars[0].display_full(
                    self.trans,
                    self.keyword_start,
                    self.keyword_end,
                    self.location_start,
                    self.location_end,
                    self.value_start,
                    self.value_end,
                    self.variable_start,
                    self.variable_end,
                ),
            )?;
            for aop in &self.phi.vars[1..] {
                write!(
                    f,
                    ", {}",
                    aop.display_full(
                        self.trans,
                        self.keyword_start,
                        self.keyword_end,
                        self.location_start,
                        self.location_end,
                        self.value_start,
                        self.value_end,
                        self.variable_start,
                        self.variable_end,
                    )
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
    Var: TranslatorDisplay<'phi, 'trans> + 'phi,
{
    type Target = PhiDisplay<'phi, 'trans, Var>;

    fn display_full(
        &'phi self,
        trans: Option<&'trans Translator>,
        keyword_start: &'trans str,
        keyword_end: &'trans str,
        location_start: &'trans str,
        location_end: &'trans str,
        value_start: &'trans str,
        value_end: &'trans str,
        variable_start: &'trans str,
        variable_end: &'trans str,
    ) -> Self::Target {
        PhiDisplay {
            phi: self,
            trans,
            keyword_start,
            keyword_end,
            location_start,
            location_end,
            value_start,
            value_end,
            variable_start,
            variable_end,
        }
    }
}
