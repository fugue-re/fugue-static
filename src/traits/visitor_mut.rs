use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::ecode::{BinOp, BinRel, UnOp, UnRel};
use fugue::ir::il::ecode::{BranchTargetT, Cast, ExprT, StmtT};
use fugue::ir::space::AddressSpaceId;

use smallvec::SmallVec;
use std::sync::Arc;

pub trait VisitMut<'ecode, Loc, Val, Var> {
    #[allow(unused)]
    fn visit_val_mut(&mut self, bv: &'ecode mut Val) {}
    #[allow(unused)]
    fn visit_var_mut(&mut self, var: &'ecode mut Var) {}
    #[allow(unused)]
    fn visit_location_mut(&mut self, location: &'ecode mut Loc) {}

    fn visit_branch_target_location_mut(&mut self, location: &'ecode mut Loc) {
        self.visit_location_mut(location)
    }

    fn visit_branch_target_computed_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(expr)
    }

    fn visit_branch_target_mut(&mut self, branch_target: &'ecode mut BranchTargetT<Loc, Val, Var>) {
        match branch_target {
            BranchTargetT::Location(ref mut location) => {
                self.visit_branch_target_location_mut(location)
            }
            BranchTargetT::Computed(ref mut expr) => self.visit_branch_target_computed_mut(expr),
        }
    }

    #[allow(unused)]
    fn visit_cast_bool_mut(&mut self) {}
    #[allow(unused)]
    fn visit_cast_void_mut(&mut self) {}
    #[allow(unused)]
    fn visit_cast_float_mut(&mut self, format: &'ecode mut Arc<FloatFormat>) {}
    #[allow(unused)]
    fn visit_cast_extend_signed_mut(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_extend_unsigned_mut(&mut self, bits: usize) {}

    #[allow(unused)]
    fn visit_cast_pointer_mut(&mut self, cast: &'ecode mut Box<Cast>, bits: usize) {
        self.visit_cast_mut(cast)
    }

    #[allow(unused)]
    fn visit_cast_function_mut(&mut self, rtyp: &'ecode mut Box<Cast>, ptyps: &'ecode mut SmallVec<[Box<Cast>; 4]>) {
        for ptyp in ptyps {
            self.visit_cast_mut(ptyp);
        }
        self.visit_cast_mut(rtyp);
    }

    #[allow(unused)]
    fn visit_cast_named_mut(&mut self, cast: &'ecode str, bits: usize) {}

    fn visit_cast_mut(&mut self, cast: &'ecode mut Cast) {
        match cast {
            Cast::Void => self.visit_cast_void_mut(),
            Cast::Bool => self.visit_cast_bool_mut(),
            Cast::Float(ref mut format) => self.visit_cast_float_mut(format),
            Cast::Signed(bits) => self.visit_cast_extend_signed_mut(*bits),
            Cast::Unsigned(bits) => self.visit_cast_extend_unsigned_mut(*bits),
            Cast::Pointer(ref mut cast, bits) => self.visit_cast_pointer_mut(cast, *bits),
            Cast::Function(ref mut rtyp, ref mut ptyps) => self.visit_cast_function_mut(rtyp, ptyps),
            Cast::Named(ref mut name, bits) => self.visit_cast_named_mut(name, *bits),
        }
    }

    #[allow(unused)]
    fn visit_expr_unop_op_mut(&mut self, op: UnOp) {}

    fn visit_expr_unop_mut(&mut self, op: UnOp, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_unop_op_mut(op);
        self.visit_expr_mut(expr)
    }

    #[allow(unused)]
    fn visit_expr_unrel_op_mut(&mut self, op: UnRel) {}

    fn visit_expr_unrel_mut(&mut self, op: UnRel, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_unrel_op_mut(op);
        self.visit_expr_mut(expr)
    }

    #[allow(unused)]
    fn visit_expr_binop_op_mut(&mut self, op: BinOp) {}

    fn visit_expr_binop_mut(&mut self, op: BinOp, lexpr: &'ecode mut ExprT<Loc, Val, Var>, rexpr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(lexpr);
        self.visit_expr_binop_op_mut(op);
        self.visit_expr_mut(rexpr)
    }

    #[allow(unused)]
    fn visit_expr_binrel_op_mut(&mut self, op: BinRel) {}

    fn visit_expr_binrel_mut(&mut self, op: BinRel, lexpr: &'ecode mut ExprT<Loc, Val, Var>, rexpr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(lexpr);
        self.visit_expr_binrel_op_mut(op);
        self.visit_expr_mut(rexpr)
    }

    fn visit_expr_cast_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>, cast: &'ecode mut Cast) {
        self.visit_expr_mut(expr);
        self.visit_cast_mut(cast);
    }

    #[allow(unused)]
    fn visit_expr_load_size_mut(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_expr_load_space_mut(&mut self, space: AddressSpaceId) {}

    fn visit_expr_load_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>, size: usize, space: AddressSpaceId) {
        self.visit_expr_mut(expr);
        self.visit_expr_load_size_mut(size);
        self.visit_expr_load_space_mut(space);
    }

    #[allow(unused)]
    fn visit_expr_extract_low_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>, bits: usize) {
        self.visit_expr_mut(expr);
    }

    #[allow(unused)]
    fn visit_expr_extract_high_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>, bits: usize) {
        self.visit_expr_mut(expr);
    }

    #[allow(unused)]
    fn visit_expr_extract_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>, lsb: usize, msb: usize) {
        self.visit_expr_mut(expr);
    }

    fn visit_expr_concat_mut(&mut self, lexpr: &'ecode mut ExprT<Loc, Val, Var>, rexpr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(lexpr);
        self.visit_expr_mut(rexpr)
    }

    fn visit_expr_ite_mut(&mut self, cond: &'ecode mut ExprT<Loc, Val, Var>, texpr: &'ecode mut ExprT<Loc, Val, Var>, fexpr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(cond);
        self.visit_expr_mut(texpr);
        self.visit_expr_mut(fexpr)
    }

    #[allow(unused)]
    fn visit_expr_call_mut(
        &mut self,
        branch_target: &'ecode mut BranchTargetT<Loc, Val, Var>,
        args: &'ecode mut SmallVec<[Box<ExprT<Loc, Val, Var>>; 4]>,
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
        args: &'ecode mut SmallVec<[Box<ExprT<Loc, Val, Var>>; 4]>,
        bits: usize,
    ) {
        for arg in args.iter_mut() {
            self.visit_expr_mut(arg);
        }
    }

    fn visit_expr_val_mut(&mut self, bv: &'ecode mut Val) {
        self.visit_val_mut(bv)
    }

    fn visit_expr_var_mut(&mut self, var: &'ecode mut Var) {
        self.visit_var_mut(var)
    }

    fn visit_expr_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        match expr {
            ExprT::UnRel(op, ref mut expr) => self.visit_expr_unrel_mut(*op, expr),
            ExprT::UnOp(op, ref mut expr) => self.visit_expr_unop_mut(*op, expr),
            ExprT::BinRel(op, ref mut lexpr, ref mut rexpr) => {
                self.visit_expr_binrel_mut(*op, lexpr, rexpr)
            }
            ExprT::BinOp(op, ref mut lexpr, ref mut rexpr) => {
                self.visit_expr_binop_mut(*op, lexpr, rexpr)
            }
            ExprT::Cast(ref mut expr, ref mut cast) => self.visit_expr_cast_mut(expr, cast),
            ExprT::Load(ref mut expr, size, space) => {
                self.visit_expr_load_mut(expr, *size, *space)
            }
            ExprT::ExtractHigh(ref mut expr, bits) => self.visit_expr_extract_high_mut(expr, *bits),
            ExprT::ExtractLow(ref mut expr, bits) => self.visit_expr_extract_low_mut(expr, *bits),
            ExprT::Extract(ref mut expr, lsb, msb) => self.visit_expr_extract_mut(expr, *lsb, *msb),
            ExprT::Concat(ref mut lexpr, ref mut rexpr) => self.visit_expr_concat_mut(lexpr, rexpr),
            ExprT::IfElse(ref mut cond, ref mut texpr, ref mut fexpr) => self.visit_expr_ite_mut(cond, texpr, fexpr),
            ExprT::Call(ref mut branch_target, ref mut args, bits) => {
                self.visit_expr_call_mut(branch_target, args, *bits)
            }
            ExprT::Intrinsic(ref name, ref mut args, bits) => {
                self.visit_expr_intrinsic_mut(name, args, *bits)
            }
            ExprT::Val(ref mut bv) => self.visit_expr_val_mut(bv),
            ExprT::Var(ref mut var) => self.visit_expr_var_mut(var),
        }
    }

    fn visit_stmt_assign_mut(&mut self, var: &'ecode mut Var, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_var_mut(var);
        self.visit_expr_mut(expr);
    }

    #[allow(unused)]
    fn visit_stmt_store_size_mut(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_stmt_store_space_mut(&mut self, space: AddressSpaceId) {}

    fn visit_stmt_store_location_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>, size: usize, space: AddressSpaceId) {
        self.visit_stmt_store_size_mut(size);
        self.visit_stmt_store_space_mut(space);
        self.visit_expr_mut(expr);
    }

    fn visit_stmt_store_value_mut(&mut self, expr: &'ecode mut ExprT<Loc, Val, Var>) {
        self.visit_expr_mut(expr);
    }

    fn visit_stmt_store_mut(
        &mut self,
        loc: &'ecode mut ExprT<Loc, Val, Var>,
        val: &'ecode mut ExprT<Loc, Val, Var>,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_stmt_store_location_mut(loc, size, space);
        self.visit_stmt_store_value_mut(val)
    }

    fn visit_stmt_branch_mut(&mut self, branch_target: &'ecode mut BranchTargetT<Loc, Val, Var>) {
        self.visit_branch_target_mut(branch_target)
    }

    fn visit_stmt_cbranch_mut(&mut self, cond: &'ecode mut ExprT<Loc, Val, Var>, branch_target: &'ecode mut BranchTargetT<Loc, Val, Var>) {
        self.visit_expr_mut(cond);
        self.visit_branch_target_mut(branch_target)
    }

    fn visit_stmt_call_mut(&mut self, branch_target: &'ecode mut BranchTargetT<Loc, Val, Var>, args: &'ecode mut SmallVec<[ExprT<Loc, Val, Var>; 4]>) {
        self.visit_branch_target_mut(branch_target);
        for arg in args {
            self.visit_expr_mut(arg);
        }
    }

    fn visit_stmt_return_mut(&mut self, branch_target: &'ecode mut BranchTargetT<Loc, Val, Var>) {
        self.visit_branch_target_mut(branch_target)
    }

    #[allow(unused)]
    fn visit_stmt_skip_mut(&mut self) {}

    #[allow(unused)]
    fn visit_stmt_intrinsic_mut(&mut self, name: &str, args: &'ecode mut SmallVec<[ExprT<Loc, Val, Var>; 4]>) {
        for arg in args {
            self.visit_expr_mut(arg);
        }
    }

    fn visit_stmt_mut(&mut self, stmt: &'ecode mut StmtT<Loc, Val, Var>) {
        match stmt {
            StmtT::Assign(ref mut var, ref mut expr) => self.visit_stmt_assign_mut(var, expr),
            StmtT::Store(ref mut loc, ref mut val, size, space) => {
                self.visit_stmt_store_mut(loc, val, *size, *space)
            }
            StmtT::Branch(ref mut branch_target) => self.visit_stmt_branch_mut(branch_target),
            StmtT::CBranch(ref mut cond, ref mut branch_target) => {
                self.visit_stmt_cbranch_mut(cond, branch_target)
            }
            StmtT::Call(ref mut branch_target, ref mut args) => self.visit_stmt_call_mut(branch_target, args),
            StmtT::Return(ref mut branch_target) => self.visit_stmt_return_mut(branch_target),
            StmtT::Skip => self.visit_stmt_skip_mut(),
            StmtT::Intrinsic(ref name, ref mut args) => self.visit_stmt_intrinsic_mut(name, args),
        }
    }
}
