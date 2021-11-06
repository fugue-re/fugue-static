use fugue::bv::BitVec;
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::ecode::{BinOp, BinRel, UnOp, UnRel};
use fugue::ir::il::ecode::{BranchTarget, Cast, Expr, Location, Stmt, Var};
use fugue::ir::AddressSpace;

use smallvec::SmallVec;
use std::sync::Arc;

pub trait VisitMap<'ecode> {
    fn visit_val(&mut self, bv: BitVec) -> BitVec {
        bv
    }
    fn visit_var(&mut self, var: Var) -> Var {
        var
    }
    fn visit_location(&mut self, location: Location) -> Location {
        location
    }

    fn visit_branch_target_location(&mut self, location: Location) -> BranchTarget {
        BranchTarget::Location(self.visit_location(location))
    }

    fn visit_branch_target_computed(&mut self, expr: Expr) -> BranchTarget {
        BranchTarget::Computed(self.visit_expr(expr))
    }

    fn visit_branch_target(&mut self, branch_target: BranchTarget) -> BranchTarget {
        match branch_target {
            BranchTarget::Location(location) => self.visit_branch_target_location(location),
            BranchTarget::Computed(expr) => self.visit_branch_target_computed(expr),
        }
    }

    fn visit_cast_bool(&mut self) -> Cast {
        Cast::Bool
    }

    fn visit_cast_float(&mut self, format: Arc<FloatFormat>) -> Cast {
        Cast::Float(format)
    }

    fn visit_cast_extend_signed(&mut self, bits: usize) -> Cast {
        Cast::Signed(bits)
    }

    fn visit_cast_extend_unsigned(&mut self, bits: usize) -> Cast {
        Cast::Unsigned(bits)
    }

    fn visit_cast_truncate_msb(&mut self, bits: usize) -> Cast {
        Cast::High(bits)
    }

    fn visit_cast_truncate_lsb(&mut self, bits: usize) -> Cast {
        Cast::Low(bits)
    }

    fn visit_cast(&mut self, cast: Cast) -> Cast {
        match cast {
            Cast::Bool => self.visit_cast_bool(),
            Cast::Float(format) => self.visit_cast_float(format),
            Cast::Signed(bits) => self.visit_cast_extend_signed(bits),
            Cast::Unsigned(bits) => self.visit_cast_extend_unsigned(bits),
            Cast::High(bits) => self.visit_cast_truncate_msb(bits),
            Cast::Low(bits) => self.visit_cast_truncate_lsb(bits),
        }
    }

    fn visit_expr_unop(&mut self, op: UnOp, expr: Expr) -> Expr {
        Expr::UnOp(op, Box::new(self.visit_expr(expr)))
    }

    fn visit_expr_unrel(&mut self, op: UnRel, expr: Expr) -> Expr {
        Expr::UnRel(op, Box::new(self.visit_expr(expr)))
    }

    fn visit_expr_binop(&mut self, op: BinOp, lexpr: Expr, rexpr: Expr) -> Expr {
        Expr::BinOp(
            op,
            Box::new(self.visit_expr(lexpr)),
            Box::new(self.visit_expr(rexpr)),
        )
    }

    fn visit_expr_binrel(&mut self, op: BinRel, lexpr: Expr, rexpr: Expr) -> Expr {
        Expr::BinRel(
            op,
            Box::new(self.visit_expr(lexpr)),
            Box::new(self.visit_expr(rexpr)),
        )
    }

    fn visit_expr_cast(&mut self, expr: Expr, cast: Cast) -> Expr {
        Expr::Cast(Box::new(self.visit_expr(expr)), self.visit_cast(cast))
    }

    fn visit_expr_load(&mut self, expr: Expr, size: usize, space: Arc<AddressSpace>) -> Expr {
        Expr::load(self.visit_expr(expr), size, space)
    }

    fn visit_expr_extract(&mut self, expr: Expr, lsb: usize, msb: usize) -> Expr {
        Expr::extract(self.visit_expr(expr), lsb, msb)
    }

    fn visit_expr_concat(&mut self, lexpr: Expr, rexpr: Expr) -> Expr {
        Expr::concat(self.visit_expr(lexpr), self.visit_expr(rexpr))
    }

    fn visit_expr_ite(&mut self, cond: Expr, texpr: Expr, fexpr: Expr) -> Expr {
        Expr::ite(
            self.visit_expr(cond),
            self.visit_expr(texpr),
            self.visit_expr(fexpr),
        )
    }

    #[allow(unused_variables)]
    fn visit_expr_intrinsic(
        &mut self,
        name: Arc<str>,
        args: SmallVec<[Box<Expr>; 4]>,
        bits: usize,
    ) -> Expr {
        Expr::Intrinsic(
            name,
            args.into_iter()
                .map(|arg| Box::new(self.visit_expr(*arg)))
                .collect(),
            bits,
        )
    }

    fn visit_expr_val(&mut self, bv: BitVec) -> Expr {
        self.visit_val(bv).into()
    }

    fn visit_expr_var(&mut self, var: Var) -> Expr {
        self.visit_var(var).into()
    }

    fn visit_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::UnRel(op, expr) => self.visit_expr_unrel(op, *expr),
            Expr::UnOp(op, expr) => self.visit_expr_unop(op, *expr),
            Expr::BinRel(op, lexpr, rexpr) => self.visit_expr_binrel(op, *lexpr, *rexpr),
            Expr::BinOp(op, lexpr, rexpr) => self.visit_expr_binop(op, *lexpr, *rexpr),
            Expr::Cast(expr, cast) => self.visit_expr_cast(*expr, cast),
            Expr::Load(expr, size, space) => self.visit_expr_load(*expr, size, space),
            Expr::Extract(expr, lsb, msb) => self.visit_expr_extract(*expr, lsb, msb),
            Expr::Concat(lexpr, rexpr) => self.visit_expr_concat(*lexpr, *rexpr),
            Expr::IfElse(cond, texpr, fexpr) => self.visit_expr_ite(*cond, *texpr, *fexpr),
            Expr::Intrinsic(name, args, bits) => self.visit_expr_intrinsic(name, args, bits),
            Expr::Val(bv) => self.visit_expr_val(bv),
            Expr::Var(var) => self.visit_expr_var(var),
        }
    }

    fn visit_stmt_assign(&mut self, var: Var, expr: Expr) -> Stmt {
        Stmt::Assign(self.visit_var(var), self.visit_expr(expr))
    }

    fn visit_stmt_store(
        &mut self,
        loc: Expr,
        val: Expr,
        size: usize,
        space: Arc<AddressSpace>,
    ) -> Stmt {
        Stmt::Store(self.visit_expr(loc), self.visit_expr(val), size, space)
    }

    fn visit_stmt_branch(&mut self, branch_target: BranchTarget) -> Stmt {
        Stmt::Branch(self.visit_branch_target(branch_target))
    }

    fn visit_stmt_cbranch(&mut self, cond: Expr, branch_target: BranchTarget) -> Stmt {
        Stmt::CBranch(
            self.visit_expr(cond),
            self.visit_branch_target(branch_target),
        )
    }

    fn visit_stmt_call(&mut self, branch_target: BranchTarget) -> Stmt {
        Stmt::Call(self.visit_branch_target(branch_target))
    }

    fn visit_stmt_return(&mut self, branch_target: BranchTarget) -> Stmt {
        Stmt::Return(self.visit_branch_target(branch_target))
    }

    fn visit_stmt_skip(&mut self) -> Stmt {
        Stmt::Skip
    }

    fn visit_stmt_intrinsic(&mut self, name: Arc<str>, args: SmallVec<[Expr; 4]>) -> Stmt {
        Stmt::Intrinsic(
            name,
            args.into_iter().map(|arg| self.visit_expr(arg)).collect(),
        )
    }

    fn visit_stmt(&mut self, stmt: Stmt) -> Stmt {
        match stmt {
            Stmt::Assign(var, expr) => self.visit_stmt_assign(var, expr),
            Stmt::Store(loc, val, size, space) => self.visit_stmt_store(loc, val, size, space),
            Stmt::Branch(branch_target) => self.visit_stmt_branch(branch_target),
            Stmt::CBranch(cond, branch_target) => self.visit_stmt_cbranch(cond, branch_target),
            Stmt::Call(branch_target) => self.visit_stmt_call(branch_target),
            Stmt::Return(branch_target) => self.visit_stmt_return(branch_target),
            Stmt::Skip => self.visit_stmt_skip(),
            Stmt::Intrinsic(name, args) => self.visit_stmt_intrinsic(name, args),
        }
    }
}
