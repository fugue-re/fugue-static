use fugue::ir::il::ecode::Stmt;
use crate::traits::Variables;

pub trait StmtExt: Variables {
    fn is_branch(&self) -> bool;
    fn is_jump(&self) -> bool;
    fn is_cond(&self) -> bool;
    fn is_call(&self) -> bool;
    fn is_return(&self) -> bool;
}

impl StmtExt for Stmt {
    fn is_branch(&self) -> bool {
        matches!(self,
                 Stmt::Branch(_) |
                 Stmt::CBranch(_, _) |
                 Stmt::Call(_) |
                 Stmt::Return(_))
    }

    fn is_jump(&self) -> bool {
        matches!(self, Stmt::Branch(_) | Stmt::CBranch(_, _))
    }

    fn is_cond(&self) -> bool {
        matches!(self, Stmt::CBranch(_, _))
    }

    fn is_call(&self) -> bool {
        matches!(self, Stmt::Call(_))
    }

    fn is_return(&self) -> bool {
        matches!(self, Stmt::Return(_))
    }
}
