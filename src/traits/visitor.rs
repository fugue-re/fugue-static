use fugue::ir::float_format::FloatFormat;
use fugue::ir::il::ecode::{BinOp, BinRel, UnOp, UnRel};
use fugue::ir::il::ecode::{BranchTargetT, Cast, ExprT, StmtT};
use fugue::ir::space::AddressSpaceId;

use std::sync::Arc;

pub trait Visit<'ecode, Loc, Val, Var> {
    #[allow(unused)]
    fn visit_val(&mut self, bv: &'ecode Val) {}
    #[allow(unused)]
    fn visit_var(&mut self, var: &'ecode Var) {}
    #[allow(unused)]
    fn visit_location(&mut self, location: &'ecode Loc) {}

    fn visit_branch_target_location(&mut self, location: &'ecode Loc) {
        self.visit_location(location)
    }

    fn visit_branch_target_computed(&mut self, expr: &'ecode ExprT<Loc, Val, Var>) {
        self.visit_expr(expr)
    }

    fn visit_branch_target(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>) {
        match branch_target {
            BranchTargetT::Location(ref location) => self.visit_branch_target_location(location),
            BranchTargetT::Computed(ref expr) => self.visit_branch_target_computed(expr),
        }
    }

    #[allow(unused)]
    fn visit_cast_bool(&mut self) {}
    #[allow(unused)]
    fn visit_cast_void(&mut self) {}

    #[allow(unused)]
    fn visit_cast_float(&mut self, format: &'ecode Arc<FloatFormat>) {}

    #[allow(unused)]
    fn visit_cast_extend_signed(&mut self, bits: usize) {}
    #[allow(unused)]
    fn visit_cast_extend_unsigned(&mut self, bits: usize) {}

    #[allow(unused)]
    fn visit_cast_pointer(&mut self, cast: &'ecode Cast, bits: usize) {
        self.visit_cast(cast)
    }

    #[allow(unused)]
    fn visit_cast_function(&mut self, rtyp: &'ecode Cast, ptyps: &'ecode [Box<Cast>]) {
        for ptyp in ptyps {
            self.visit_cast(ptyp);
        }
        self.visit_cast(rtyp);
    }

    #[allow(unused)]
    fn visit_cast_named(&mut self, cast: &'ecode str, bits: usize) {}

    fn visit_cast(&mut self, cast: &'ecode Cast) {
        match cast {
            Cast::Void => self.visit_cast_void(),
            Cast::Bool => self.visit_cast_bool(),
            Cast::Float(ref format) => self.visit_cast_float(format),
            Cast::Signed(bits) => self.visit_cast_extend_signed(*bits),
            Cast::Unsigned(bits) => self.visit_cast_extend_unsigned(*bits),
            Cast::Pointer(ref cast, bits) => self.visit_cast_pointer(cast, *bits),
            Cast::Function(ref rtyp, ref ptyps) => self.visit_cast_function(rtyp, ptyps),
            Cast::Named(ref name, ref bits) => self.visit_cast_named(name, *bits),
        }
    }

    #[allow(unused)]
    fn visit_expr_unop_op(&mut self, op: UnOp) {}

    fn visit_expr_unop(&mut self, op: UnOp, expr: &'ecode ExprT<Loc, Val, Var>) {
        self.visit_expr_unop_op(op);
        self.visit_expr(expr)
    }

    #[allow(unused)]
    fn visit_expr_unrel_op(&mut self, op: UnRel) {}

    fn visit_expr_unrel(&mut self, op: UnRel, expr: &'ecode ExprT<Loc, Val, Var>) {
        self.visit_expr_unrel_op(op);
        self.visit_expr(expr)
    }

    #[allow(unused)]
    fn visit_expr_binop_op(&mut self, op: BinOp) {}

    fn visit_expr_binop(
        &mut self,
        op: BinOp,
        lexpr: &'ecode ExprT<Loc, Val, Var>,
        rexpr: &'ecode ExprT<Loc, Val, Var>,
    ) {
        self.visit_expr(lexpr);
        self.visit_expr_binop_op(op);
        self.visit_expr(rexpr)
    }

    #[allow(unused)]
    fn visit_expr_binrel_op(&mut self, op: BinRel) {}

    fn visit_expr_binrel(
        &mut self,
        op: BinRel,
        lexpr: &'ecode ExprT<Loc, Val, Var>,
        rexpr: &'ecode ExprT<Loc, Val, Var>,
    ) {
        self.visit_expr(lexpr);
        self.visit_expr_binrel_op(op);
        self.visit_expr(rexpr)
    }

    fn visit_expr_cast(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, cast: &'ecode Cast) {
        self.visit_expr(expr);
        self.visit_cast(cast);
    }

    #[allow(unused)]
    fn visit_expr_load_size(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_expr_load_space(&mut self, space: AddressSpaceId) {}

    fn visit_expr_load(
        &mut self,
        expr: &'ecode ExprT<Loc, Val, Var>,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_expr(expr);
        self.visit_expr_load_size(size);
        self.visit_expr_load_space(space);
    }

    #[allow(unused)]
    fn visit_expr_extract_high(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, bits: usize) {
        self.visit_expr(expr);
    }

    #[allow(unused)]
    fn visit_expr_extract_low(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, bits: usize) {
        self.visit_expr(expr);
    }

    #[allow(unused)]
    fn visit_expr_extract(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, lsb: usize, msb: usize) {
        self.visit_expr(expr);
    }

    fn visit_expr_concat(
        &mut self,
        lexpr: &'ecode ExprT<Loc, Val, Var>,
        rexpr: &'ecode ExprT<Loc, Val, Var>,
    ) {
        self.visit_expr(lexpr);
        self.visit_expr(rexpr)
    }

    fn visit_expr_ite(
        &mut self,
        cond: &'ecode ExprT<Loc, Val, Var>,
        texpr: &'ecode ExprT<Loc, Val, Var>,
        fexpr: &'ecode ExprT<Loc, Val, Var>,
    ) {
        self.visit_expr(cond);
        self.visit_expr(texpr);
        self.visit_expr(fexpr)
    }

    #[allow(unused_variables)]
    fn visit_expr_call(
        &mut self,
        target: &'ecode BranchTargetT<Loc, Val, Var>,
        args: &'ecode [Box<ExprT<Loc, Val, Var>>],
        bits: usize,
    ) {
        self.visit_branch_target(target);
        for arg in args.iter() {
            self.visit_expr(arg);
        }
    }

    #[allow(unused_variables)]
    fn visit_expr_intrinsic(
        &mut self,
        name: &'ecode str,
        args: &'ecode [Box<ExprT<Loc, Val, Var>>],
        bits: usize,
    ) {
        for arg in args.iter() {
            self.visit_expr(arg);
        }
    }

    fn visit_expr_val(&mut self, bv: &'ecode Val) {
        self.visit_val(bv)
    }

    fn visit_expr_var(&mut self, var: &'ecode Var) {
        self.visit_var(var)
    }

    fn visit_expr(&mut self, expr: &'ecode ExprT<Loc, Val, Var>) {
        match expr {
            ExprT::UnRel(op, ref expr) => self.visit_expr_unrel(*op, expr),
            ExprT::UnOp(op, ref expr) => self.visit_expr_unop(*op, expr),
            ExprT::BinRel(op, ref lexpr, ref rexpr) => self.visit_expr_binrel(*op, lexpr, rexpr),
            ExprT::BinOp(op, ref lexpr, ref rexpr) => self.visit_expr_binop(*op, lexpr, rexpr),
            ExprT::Cast(ref expr, ref cast) => self.visit_expr_cast(expr, cast),
            ExprT::Load(ref expr, size, space) => self.visit_expr_load(expr, *size, *space),
            ExprT::ExtractHigh(ref expr, bits) => self.visit_expr_extract_high(expr, *bits),
            ExprT::ExtractLow(ref expr, bits) => self.visit_expr_extract_low(expr, *bits),
            ExprT::Extract(ref expr, lsb, msb) => self.visit_expr_extract(expr, *lsb, *msb),
            ExprT::Concat(ref lexpr, ref rexpr) => self.visit_expr_concat(lexpr, rexpr),
            ExprT::IfElse(ref cond, ref texpr, ref fexpr) => {
                self.visit_expr_ite(cond, texpr, fexpr)
            }
            ExprT::Call(ref target, ref args, bits) => self.visit_expr_call(target, args, *bits),
            ExprT::Intrinsic(ref name, ref args, bits) => {
                self.visit_expr_intrinsic(name, args, *bits)
            }
            ExprT::Val(ref bv) => self.visit_expr_val(bv),
            ExprT::Var(ref var) => self.visit_expr_var(var),
        }
    }

    fn visit_stmt_assign(&mut self, var: &'ecode Var, expr: &'ecode ExprT<Loc, Val, Var>) {
        self.visit_var(var);
        self.visit_expr(expr);
    }

    #[allow(unused)]
    fn visit_stmt_store_size(&mut self, size: usize) {}
    #[allow(unused)]
    fn visit_stmt_store_space(&mut self, space: AddressSpaceId) {}

    fn visit_stmt_store_location(
        &mut self,
        expr: &'ecode ExprT<Loc, Val, Var>,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_stmt_store_size(size);
        self.visit_stmt_store_space(space);
        self.visit_expr(expr);
    }

    fn visit_stmt_store_value(&mut self, expr: &'ecode ExprT<Loc, Val, Var>) {
        self.visit_expr(expr);
    }

    fn visit_stmt_store(
        &mut self,
        loc: &'ecode ExprT<Loc, Val, Var>,
        val: &'ecode ExprT<Loc, Val, Var>,
        size: usize,
        space: AddressSpaceId,
    ) {
        self.visit_stmt_store_location(loc, size, space);
        self.visit_stmt_store_value(val)
    }

    fn visit_stmt_branch(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>) {
        self.visit_branch_target(branch_target)
    }

    fn visit_stmt_cbranch(
        &mut self,
        cond: &'ecode ExprT<Loc, Val, Var>,
        branch_target: &'ecode BranchTargetT<Loc, Val, Var>,
    ) {
        self.visit_expr(cond);
        self.visit_branch_target(branch_target)
    }

    fn visit_stmt_call(
        &mut self,
        branch_target: &'ecode BranchTargetT<Loc, Val, Var>,
        args: &'ecode [ExprT<Loc, Val, Var>],
    ) {
        self.visit_branch_target(branch_target);
        for arg in args {
            self.visit_expr(arg);
        }
    }

    fn visit_stmt_return(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>) {
        self.visit_branch_target(branch_target)
    }

    #[allow(unused)]
    fn visit_stmt_skip(&mut self) {}

    #[allow(unused)]
    fn visit_stmt_intrinsic(&mut self, name: &'ecode str, args: &'ecode [ExprT<Loc, Val, Var>]) {
        for arg in args {
            self.visit_expr(arg);
        }
    }

    fn visit_stmt(&mut self, stmt: &'ecode StmtT<Loc, Val, Var>) {
        match stmt {
            StmtT::Assign(ref var, ref expr) => self.visit_stmt_assign(var, expr),
            StmtT::Store(ref loc, ref val, size, space) => {
                self.visit_stmt_store(loc, val, *size, *space)
            }
            StmtT::Branch(ref branch_target) => self.visit_stmt_branch(branch_target),
            StmtT::CBranch(ref cond, ref branch_target) => {
                self.visit_stmt_cbranch(cond, branch_target)
            }
            StmtT::Call(ref branch_target, ref args) => self.visit_stmt_call(branch_target, args),
            StmtT::Return(ref branch_target) => self.visit_stmt_return(branch_target),
            StmtT::Skip => self.visit_stmt_skip(),
            StmtT::Intrinsic(ref name, ref args) => self.visit_stmt_intrinsic(name, args),
        }
    }
}
