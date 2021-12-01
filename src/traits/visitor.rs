use fugue::bv::BitVec;
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::ecode::{BinOp, BinRel, UnOp, UnRel};
use fugue::ir::il::ecode::{BranchTarget, Cast, Expr, Location, Stmt, Var};
use fugue::ir::space::AddressSpaceId;

use std::sync::Arc;

pub trait Visit<'ecode> {
    #[allow(unused)]
    fn visit_val(&mut self, bv: &'ecode BitVec) {}
    #[allow(unused)]
    fn visit_var(&mut self, var: &'ecode Var) {}
    #[allow(unused)]
    fn visit_location(&mut self, location: &'ecode Location) {}

    fn visit_branch_target_location(&mut self, location: &'ecode Location) {
        self.visit_location(location)
    }

    fn visit_branch_target_computed(&mut self, expr: &'ecode Expr) {
        self.visit_expr(expr)
    }

    fn visit_branch_target(&mut self, branch_target: &'ecode BranchTarget) {
        match branch_target {
            BranchTarget::Location(ref location) => self.visit_branch_target_location(location),
            BranchTarget::Computed(ref expr) => self.visit_branch_target_computed(expr),
        }
    }

    #[allow(unused)]
    fn visit_cast_bool(&mut self) {}
    #[allow(unused)]
    fn visit_cast_float(&mut self, format: &'ecode Arc<FloatFormat>) {}
    #[allow(unused)]
    fn visit_cast_extend_signed(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_extend_unsigned(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_truncate_msb(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_truncate_lsb(&mut self, bits: usize) {}
    fn visit_cast(&mut self, cast: &'ecode Cast) {
        match cast {
            Cast::Bool => self.visit_cast_bool(),
            Cast::Float(ref format) => self.visit_cast_float(format),
            Cast::Signed(bits) => self.visit_cast_extend_signed(*bits),
            Cast::Unsigned(bits) => self.visit_cast_extend_unsigned(*bits),
            Cast::High(bits) => self.visit_cast_truncate_msb(*bits),
            Cast::Low(bits) => self.visit_cast_truncate_lsb(*bits),
        }
    }

    #[allow(unused)]
    fn visit_expr_unop_op(&mut self, op: UnOp) {}

    fn visit_expr_unop(&mut self, op: UnOp, expr: &'ecode Expr) {
        self.visit_expr_unop_op(op);
        self.visit_expr(expr)
    }

    #[allow(unused)]
    fn visit_expr_unrel_op(&mut self, op: UnRel) {}

    fn visit_expr_unrel(&mut self, op: UnRel, expr: &'ecode Expr) {
        self.visit_expr_unrel_op(op);
        self.visit_expr(expr)
    }

    #[allow(unused)]
    fn visit_expr_binop_op(&mut self, op: BinOp) {}

    fn visit_expr_binop(&mut self, op: BinOp, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        self.visit_expr(lexpr);
        self.visit_expr_binop_op(op);
        self.visit_expr(rexpr)
    }

    #[allow(unused)]
    fn visit_expr_binrel_op(&mut self, op: BinRel) {}

    fn visit_expr_binrel(&mut self, op: BinRel, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        self.visit_expr(lexpr);
        self.visit_expr_binrel_op(op);
        self.visit_expr(rexpr)
    }

    fn visit_expr_cast(&mut self, expr: &'ecode Expr, cast: &'ecode Cast) {
        self.visit_expr(expr);
        self.visit_cast(cast);
    }

    #[allow(unused)]
    fn visit_expr_load_size(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_expr_load_space(&mut self, space: AddressSpaceId) {}

    fn visit_expr_load(
        &mut self,
        expr: &'ecode Expr,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_expr(expr);
        self.visit_expr_load_size(size);
        self.visit_expr_load_space(space);
    }

    #[allow(unused)]
    fn visit_expr_extract_lsb(&mut self, lsb: usize) {}
    #[allow(unused)]
    fn visit_expr_extract_msb(&mut self, msb: usize) {}

    fn visit_expr_extract(&mut self, expr: &'ecode Expr, lsb: usize, msb: usize) {
        self.visit_expr(expr);
        self.visit_expr_extract_lsb(lsb);
        self.visit_expr_extract_msb(msb);
    }

    fn visit_expr_concat(&mut self, lexpr: &'ecode Expr, rexpr: &'ecode Expr) {
        self.visit_expr(lexpr);
        self.visit_expr(rexpr)
    }

    fn visit_expr_ite(&mut self, cond: &'ecode Expr, texpr: &'ecode Expr, fexpr: &'ecode Expr) {
        self.visit_expr(cond);
        self.visit_expr(texpr);
        self.visit_expr(fexpr)
    }

    #[allow(unused_variables)]
    fn visit_expr_call(&mut self, target: &'ecode BranchTarget, args: &'ecode [Box<Expr>], bits: usize) {
        self.visit_branch_target(target);
        for arg in args.iter() {
            self.visit_expr(arg);
        }
    }

    #[allow(unused_variables)]
    fn visit_expr_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Box<Expr>], bits: usize) {
        for arg in args.iter() {
            self.visit_expr(arg);
        }
    }

    fn visit_expr_val(&mut self, bv: &'ecode BitVec) {
        self.visit_val(bv)
    }

    fn visit_expr_var(&mut self, var: &'ecode Var) {
        self.visit_var(var)
    }

    fn visit_expr(&mut self, expr: &'ecode Expr) {
        match expr {
            Expr::UnRel(op, ref expr) => self.visit_expr_unrel(*op, expr),
            Expr::UnOp(op, ref expr) => self.visit_expr_unop(*op, expr),
            Expr::BinRel(op, ref lexpr, ref rexpr) => self.visit_expr_binrel(*op, lexpr, rexpr),
            Expr::BinOp(op, ref lexpr, ref rexpr) => self.visit_expr_binop(*op, lexpr, rexpr),
            Expr::Cast(ref expr, ref cast) => self.visit_expr_cast(expr, cast),
            Expr::Load(ref expr, size, space) => self.visit_expr_load(expr, *size, *space),
            Expr::Extract(ref expr, lsb, msb) => self.visit_expr_extract(expr, *lsb, *msb),
            Expr::Concat(ref lexpr, ref rexpr) => self.visit_expr_concat(lexpr, rexpr),
            Expr::IfElse(ref cond, ref texpr, ref fexpr) => self.visit_expr_ite(cond, texpr, fexpr),
            Expr::Call(ref target, ref args, bits) => {
                self.visit_expr_call(target, args, *bits)
            }
            Expr::Intrinsic(ref name, ref args, bits) => {
                self.visit_expr_intrinsic(name, args, *bits)
            }
            Expr::Val(ref bv) => self.visit_expr_val(bv),
            Expr::Var(ref var) => self.visit_expr_var(var),
        }
    }

    fn visit_stmt_assign(&mut self, var: &'ecode Var, expr: &'ecode Expr) {
        self.visit_var(var);
        self.visit_expr(expr);
    }

    #[allow(unused)]
    fn visit_stmt_store_size(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_stmt_store_space(&mut self, space: AddressSpaceId) {}

    fn visit_stmt_store_location(&mut self, expr: &'ecode Expr, size: usize, space: AddressSpaceId) {
        self.visit_stmt_store_size(size);
        self.visit_stmt_store_space(space);
        self.visit_expr(expr);
    }

    fn visit_stmt_store_value(&mut self, expr: &'ecode Expr) {
        self.visit_expr(expr);
    }

    fn visit_stmt_store(
        &mut self,
        loc: &'ecode Expr,
        val: &'ecode Expr,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_stmt_store_location(loc, size, space);
        self.visit_stmt_store_value(val)
    }

    fn visit_stmt_branch(&mut self, branch_target: &'ecode BranchTarget) {
        self.visit_branch_target(branch_target)
    }

    fn visit_stmt_cbranch(&mut self, cond: &'ecode Expr, branch_target: &'ecode BranchTarget) {
        self.visit_expr(cond);
        self.visit_branch_target(branch_target)
    }

    fn visit_stmt_call(&mut self, branch_target: &'ecode BranchTarget, args: &'ecode [Expr]) {
        self.visit_branch_target(branch_target);
        for arg in args {
            self.visit_expr(arg);
        }
    }

    fn visit_stmt_return(&mut self, branch_target: &'ecode BranchTarget) {
        self.visit_branch_target(branch_target)
    }

    #[allow(unused)]
    fn visit_stmt_skip(&mut self) {}

    #[allow(unused)]
    fn visit_stmt_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Expr]) {
        for arg in args {
            self.visit_expr(arg);
        }
    }

    fn visit_stmt(&mut self, stmt: &'ecode Stmt) {
        match stmt {
            Stmt::Assign(ref var, ref expr) => self.visit_stmt_assign(var, expr),
            Stmt::Store(ref loc, ref val, size, space) => {
                self.visit_stmt_store(loc, val, *size, *space)
            }
            Stmt::Branch(ref branch_target) => self.visit_stmt_branch(branch_target),
            Stmt::CBranch(ref cond, ref branch_target) => {
                self.visit_stmt_cbranch(cond, branch_target)
            }
            Stmt::Call(ref branch_target, ref args) => self.visit_stmt_call(branch_target, args),
            Stmt::Return(ref branch_target) => self.visit_stmt_return(branch_target),
            Stmt::Skip => self.visit_stmt_skip(),
            Stmt::Intrinsic(ref name, ref args) => self.visit_stmt_intrinsic(name, args),
        }
    }
}
