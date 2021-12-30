use fugue::ir::il::ecode::{BinOp, BinRel, UnOp, UnRel};
use fugue::ir::il::ecode::{BranchTargetT, Cast, ExprT, StmtT};
use fugue::ir::space::AddressSpaceId;

pub trait Interpret<'ecode, Loc, Val, Var> {
    type ILoc;
    type IVal;

    fn eval_branch_target_location(&mut self, location: &'ecode Loc) -> Self::ILoc;
    fn eval_branch_target_computed(&mut self, expr: &'ecode ExprT<Loc, Val, Var>) -> Self::ILoc;

    fn eval_branch_target(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>) -> Self::ILoc {
        match branch_target {
            BranchTargetT::Location(ref location) => self.eval_branch_target_location(location),
            BranchTargetT::Computed(ref expr) => self.eval_branch_target_computed(expr),
        }
    }

    fn eval_expr_unop(&mut self, op: UnOp, expr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal;
    fn eval_expr_unrel(&mut self, op: UnRel, expr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal;

    fn eval_expr_binop(&mut self, op: BinOp, lexpr: &'ecode ExprT<Loc, Val, Var>, rexpr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal;
    fn eval_expr_binrel(&mut self, op: BinRel, lexpr: &'ecode ExprT<Loc, Val, Var>, rexpr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal;

    fn eval_expr_cast(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, cast: &'ecode Cast) -> Self::IVal;

    fn eval_expr_load(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, size: usize, space: AddressSpaceId) -> Self::IVal;

    fn eval_expr_extract_high(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, bits: usize) -> Self::IVal;
    fn eval_expr_extract_low(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, bits: usize) -> Self::IVal;
    fn eval_expr_extract(&mut self, expr: &'ecode ExprT<Loc, Val, Var>, lsb: usize, msb: usize) -> Self::IVal;

    fn eval_expr_concat(&mut self, lexpr: &'ecode ExprT<Loc, Val, Var>, rexpr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal;

    fn eval_expr_ite(&mut self, cond: &'ecode ExprT<Loc, Val, Var>, texpr: &'ecode ExprT<Loc, Val, Var>, fexpr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal;
    fn eval_expr_call(&mut self, target: &'ecode BranchTargetT<Loc, Val, Var>, args: &'ecode [Box<ExprT<Loc, Val, Var>>], bits: usize) -> Self::IVal;

    fn eval_expr_intrinsic(&mut self, name: &'ecode str, args: &'ecode [Box<ExprT<Loc, Val, Var>>], bits: usize) -> Self::IVal;

    fn eval_expr_val(&mut self, bv: &'ecode Val) -> Self::IVal;
    fn eval_expr_var(&mut self, var: &'ecode Var) -> Self::IVal;

    fn eval_expr(&mut self, expr: &'ecode ExprT<Loc, Val, Var>) -> Self::IVal {
        match expr {
            ExprT::UnRel(op, ref expr) => self.eval_expr_unrel(*op, expr),
            ExprT::UnOp(op, ref expr) => self.eval_expr_unop(*op, expr),
            ExprT::BinRel(op, ref lexpr, ref rexpr) => self.eval_expr_binrel(*op, lexpr, rexpr),
            ExprT::BinOp(op, ref lexpr, ref rexpr) => self.eval_expr_binop(*op, lexpr, rexpr),
            ExprT::Cast(ref expr, ref cast) => self.eval_expr_cast(expr, cast),
            ExprT::Load(ref expr, size, space) => self.eval_expr_load(expr, *size, *space),
            ExprT::ExtractHigh(ref expr, bits) => self.eval_expr_extract_high(expr, *bits),
            ExprT::ExtractLow(ref expr, bits) => self.eval_expr_extract_low(expr, *bits),
            ExprT::Extract(ref expr, lsb, msb) => self.eval_expr_extract(expr, *lsb, *msb),
            ExprT::Concat(ref lexpr, ref rexpr) => self.eval_expr_concat(lexpr, rexpr),
            ExprT::IfElse(ref cond, ref texpr, ref fexpr) => {
                self.eval_expr_ite(cond, texpr, fexpr)
            }
            ExprT::Call(ref target, ref args, bits) => self.eval_expr_call(target, args, *bits),
            ExprT::Intrinsic(ref name, ref args, bits) => {
                self.eval_expr_intrinsic(name, args, *bits)
            }
            ExprT::Val(ref bv) => self.eval_expr_val(bv),
            ExprT::Var(ref var) => self.eval_expr_var(var),
        }
    }

    fn eval_stmt_assign(&mut self, var: &'ecode Var, expr: &'ecode ExprT<Loc, Val, Var>);

    fn eval_stmt_store(&mut self, loc: &'ecode ExprT<Loc, Val, Var>, val: &'ecode ExprT<Loc, Val, Var>, size: usize, space: AddressSpaceId);

    fn eval_stmt_branch(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>);
    fn eval_stmt_cbranch(&mut self, cond: &'ecode ExprT<Loc, Val, Var>, branch_target: &'ecode BranchTargetT<Loc, Val, Var>);

    fn eval_stmt_call(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>, args: &'ecode [ExprT<Loc, Val, Var>]);
    fn eval_stmt_return(&mut self, branch_target: &'ecode BranchTargetT<Loc, Val, Var>);

    fn eval_stmt_skip(&mut self);
    fn eval_stmt_intrinsic(&mut self, name: &'ecode str, args: &'ecode [ExprT<Loc, Val, Var>]);

    fn eval_stmt(&mut self, stmt: &'ecode StmtT<Loc, Val, Var>) {
        match stmt {
            StmtT::Assign(ref var, ref expr) => self.eval_stmt_assign(var, expr),
            StmtT::Store(ref loc, ref val, size, space) => {
                self.eval_stmt_store(loc, val, *size, *space)
            }
            StmtT::Branch(ref branch_target) => self.eval_stmt_branch(branch_target),
            StmtT::CBranch(ref cond, ref branch_target) => {
                self.eval_stmt_cbranch(cond, branch_target)
            }
            StmtT::Call(ref branch_target, ref args) => self.eval_stmt_call(branch_target, args),
            StmtT::Return(ref branch_target) => self.eval_stmt_return(branch_target),
            StmtT::Skip => self.eval_stmt_skip(),
            StmtT::Intrinsic(ref name, ref args) => self.eval_stmt_intrinsic(name, args),
        }
    }
}
