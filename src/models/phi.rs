use fugue::ir::il::ecode::{Location, Var};

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
            writeln!(f, "{} ← ϕ(<empty>)", self.var)?;
        } else {
            write!(f, "{} ← ϕ({}", self.var, self.vars[0])?;
            for aop in &self.vars[1..] {
                write!(f, ", {}", aop)?;
            }
            writeln!(f, ")")?;
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