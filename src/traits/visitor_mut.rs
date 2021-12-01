use fugue::bv::BitVec;
use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::ecode::{BinOp, BinRel, UnOp, UnRel};
use fugue::ir::il::ecode::{BranchTarget, Cast, Expr, Location, Stmt, Var};
use fugue::ir::space::AddressSpaceId;

use smallvec::SmallVec;
use std::sync::Arc;

pub trait VisitMut<'ecode> {
    #[allow(unused)]
    fn visit_val_mut(&mut self, bv: &'ecode mut BitVec) {}
    #[allow(unused)]
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {}
    #[allow(unused)]
    fn visit_location_mut(&mut self, location: &mut Location) {}

    fn visit_branch_target_location_mut(&mut self, location: &mut Location) {
        self.visit_location_mut(location)
    }

    fn visit_branch_target_computed_mut(&mut self, expr: &'ecode mut Expr) {
        self.visit_expr_mut(expr)
    }

    fn visit_branch_target_mut(&mut self, branch_target: &'ecode mut BranchTarget) {
        match branch_target {
            BranchTarget::Location(ref mut location) => {
                self.visit_branch_target_location_mut(location)
            }
            BranchTarget::Computed(ref mut expr) => self.visit_branch_target_computed_mut(expr),
        }
    }

    #[allow(unused)]
    fn visit_cast_bool_mut(&mut self) {}
    #[allow(unused)]
    fn visit_cast_float_mut(&mut self, format: &'ecode mut Arc<FloatFormat>) {}
    #[allow(unused)]
    fn visit_cast_extend_signed_mut(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_extend_unsigned_mut(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_truncate_msb_mut(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_truncate_lsb_mut(&mut self, bits: usize) {}
    fn visit_cast_mut(&mut self, cast: &'ecode mut Cast) {
        match cast {
            Cast::Bool => self.visit_cast_bool_mut(),
            Cast::Float(ref mut format) => self.visit_cast_float_mut(format),
            Cast::Signed(bits) => self.visit_cast_extend_signed_mut(*bits),
            Cast::Unsigned(bits) => self.visit_cast_extend_unsigned_mut(*bits),
            Cast::High(bits) => self.visit_cast_truncate_msb_mut(*bits),
            Cast::Low(bits) => self.visit_cast_truncate_lsb_mut(*bits),
        }
    }

    #[allow(unused)]
    fn visit_expr_unop_op_mut(&mut self, op: UnOp) {}

    fn visit_expr_unop_mut(&mut self, op: UnOp, expr: &'ecode mut Expr) {
        self.visit_expr_unop_op_mut(op);
        self.visit_expr_mut(expr)
    }

    #[allow(unused)]
    fn visit_expr_unrel_op_mut(&mut self, op: UnRel) {}

    fn visit_expr_unrel_mut(&mut self, op: UnRel, expr: &'ecode mut Expr) {
        self.visit_expr_unrel_op_mut(op);
        self.visit_expr_mut(expr)
    }

    #[allow(unused)]
    fn visit_expr_binop_op_mut(&mut self, op: BinOp) {}

    fn visit_expr_binop_mut(&mut self, op: BinOp, lexpr: &'ecode mut Expr, rexpr: &'ecode mut Expr) {
        self.visit_expr_mut(lexpr);
        self.visit_expr_binop_op_mut(op);
        self.visit_expr_mut(rexpr)
    }

    #[allow(unused)]
    fn visit_expr_binrel_op_mut(&mut self, op: BinRel) {}

    fn visit_expr_binrel_mut(&mut self, op: BinRel, lexpr: &'ecode mut Expr, rexpr: &'ecode mut Expr) {
        self.visit_expr_mut(lexpr);
        self.visit_expr_binrel_op_mut(op);
        self.visit_expr_mut(rexpr)
    }

    fn visit_expr_cast_mut(&mut self, expr: &'ecode mut Expr, cast: &'ecode mut Cast) {
        self.visit_expr_mut(expr);
        self.visit_cast_mut(cast);
    }

    #[allow(unused)]
    fn visit_expr_load_size_mut(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_expr_load_space_mut(&mut self, space: AddressSpaceId) {}

    fn visit_expr_load_mut(&mut self, expr: &'ecode mut Expr, size: usize, space: AddressSpaceId) {
        self.visit_expr_mut(expr);
        self.visit_expr_load_size_mut(size);
        self.visit_expr_load_space_mut(space);
    }

    #[allow(unused)]
    fn visit_expr_extract_lsb_mut(&mut self, lsb: usize) {}
    #[allow(unused)]
    fn visit_expr_extract_msb_mut(&mut self, msb: usize) {}

    fn visit_expr_extract_mut(&mut self, expr: &'ecode mut Expr, lsb: usize, msb: usize) {
        self.visit_expr_mut(expr);
        self.visit_expr_extract_lsb_mut(lsb);
        self.visit_expr_extract_msb_mut(msb);
    }

    fn visit_expr_concat_mut(&mut self, lexpr: &'ecode mut Expr, rexpr: &'ecode mut Expr) {
        self.visit_expr_mut(lexpr);
        self.visit_expr_mut(rexpr)
    }

    fn visit_expr_ite_mut(&mut self, cond: &'ecode mut Expr, texpr: &'ecode mut Expr, fexpr: &'ecode mut Expr) {
        self.visit_expr_mut(cond);
        self.visit_expr_mut(texpr);
        self.visit_expr_mut(fexpr)
    }

    #[allow(unused)]
    fn visit_expr_call_mut(
        &mut self,
        branch_target: &'ecode mut BranchTarget,
        args: &'ecode mut SmallVec<[Box<Expr>; 4]>,
        bits: usize,
    ) {
        self.visit_branch_target_mut(branch_target);
        for arg in args.iter_mut() {
            self.visit_expr_mut(arg);
        }
    }

    #[allow(unused_variables)]
    fn visit_expr_intrinsic_mut(
        &mut self,
        name: &str,
        args: &'ecode mut SmallVec<[Box<Expr>; 4]>,
        bits: usize,
    ) {
        for arg in args.iter_mut() {
            self.visit_expr_mut(arg);
        }
    }

    fn visit_expr_val_mut(&mut self, bv: &'ecode mut BitVec) {
        self.visit_val_mut(bv)
    }

    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        self.visit_var_mut(var)
    }

    fn visit_expr_mut(&mut self, expr: &'ecode mut Expr) {
        match expr {
            Expr::UnRel(op, ref mut expr) => self.visit_expr_unrel_mut(*op, expr),
            Expr::UnOp(op, ref mut expr) => self.visit_expr_unop_mut(*op, expr),
            Expr::BinRel(op, ref mut lexpr, ref mut rexpr) => {
                self.visit_expr_binrel_mut(*op, lexpr, rexpr)
            }
            Expr::BinOp(op, ref mut lexpr, ref mut rexpr) => {
                self.visit_expr_binop_mut(*op, lexpr, rexpr)
            }
            Expr::Cast(ref mut expr, ref mut cast) => self.visit_expr_cast_mut(expr, cast),
            Expr::Load(ref mut expr, size, space) => {
                self.visit_expr_load_mut(expr, *size, *space)
            }
            Expr::Extract(ref mut expr, lsb, msb) => self.visit_expr_extract_mut(expr, *lsb, *msb),
            Expr::Concat(ref mut lexpr, ref mut rexpr) => self.visit_expr_concat_mut(lexpr, rexpr),
            Expr::IfElse(ref mut cond, ref mut texpr, ref mut fexpr) => self.visit_expr_ite_mut(cond, texpr, fexpr),
            Expr::Call(ref mut branch_target, ref mut args, bits) => {
                self.visit_expr_call_mut(branch_target, args, *bits)
            }
            Expr::Intrinsic(ref name, ref mut args, bits) => {
                self.visit_expr_intrinsic_mut(name, args, *bits)
            }
            Expr::Val(ref mut bv) => self.visit_expr_val_mut(bv),
            Expr::Var(ref mut var) => self.visit_expr_var_mut(var),
        }
    }

    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, expr: &'ecode mut Expr) {
        self.visit_var_mut(var);
        self.visit_expr_mut(expr);
    }

    #[allow(unused)]
    fn visit_stmt_store_size_mut(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_stmt_store_space_mut(&mut self, space: AddressSpaceId) {}

    fn visit_stmt_store_location_mut(&mut self, expr: &'ecode mut Expr, size: usize, space: AddressSpaceId) {
        self.visit_stmt_store_size_mut(size);
        self.visit_stmt_store_space_mut(space);
        self.visit_expr_mut(expr);
    }

    fn visit_stmt_store_value_mut(&mut self, expr: &'ecode mut Expr) {
        self.visit_expr_mut(expr);
    }

    fn visit_stmt_store_mut(
        &mut self,
        loc: &'ecode mut Expr,
        val: &'ecode mut Expr,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_stmt_store_location_mut(loc, size, space);
        self.visit_stmt_store_value_mut(val)
    }

    fn visit_stmt_branch_mut(&mut self, branch_target: &'ecode mut BranchTarget) {
        self.visit_branch_target_mut(branch_target)
    }

    fn visit_stmt_cbranch_mut(&mut self, cond: &'ecode mut Expr, branch_target: &'ecode mut BranchTarget) {
        self.visit_expr_mut(cond);
        self.visit_branch_target_mut(branch_target)
    }

    fn visit_stmt_call_mut(&mut self, branch_target: &'ecode mut BranchTarget, args: &'ecode mut SmallVec<[Expr; 4]>) {
        self.visit_branch_target_mut(branch_target);
        for arg in args {
            self.visit_expr_mut(arg);
        }
    }

    fn visit_stmt_return_mut(&mut self, branch_target: &'ecode mut BranchTarget) {
        self.visit_branch_target_mut(branch_target)
    }

    #[allow(unused)]
    fn visit_stmt_skip_mut(&mut self) {}

    #[allow(unused)]
    fn visit_stmt_intrinsic_mut(&mut self, name: &str, args: &'ecode mut SmallVec<[Expr; 4]>) {
        for arg in args {
            self.visit_expr_mut(arg);
        }
    }

    fn visit_stmt_mut(&mut self, stmt: &'ecode mut Stmt) {
        match stmt {
            Stmt::Assign(ref mut var, ref mut expr) => self.visit_stmt_assign_mut(var, expr),
            Stmt::Store(ref mut loc, ref mut val, size, space) => {
                self.visit_stmt_store_mut(loc, val, *size, *space)
            }
            Stmt::Branch(ref mut branch_target) => self.visit_stmt_branch_mut(branch_target),
            Stmt::CBranch(ref mut cond, ref mut branch_target) => {
                self.visit_stmt_cbranch_mut(cond, branch_target)
            }
            Stmt::Call(ref mut branch_target, ref mut args) => self.visit_stmt_call_mut(branch_target, args),
            Stmt::Return(ref mut branch_target) => self.visit_stmt_return_mut(branch_target),
            Stmt::Skip => self.visit_stmt_skip_mut(),
            Stmt::Intrinsic(ref name, ref mut args) => self.visit_stmt_intrinsic_mut(name, args),
        }
    }
}
