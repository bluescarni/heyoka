// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to add a polynomial root finder to an llvm_state.
//
// The root finder is added to s as a function called name. It accepts in input a polynomial q of degree n whose
// coefficients are intervals. It is assumed that the coefficients of q enclose the coefficients of a polynomial q_exact
// which has been proven to have exactly one simple root in the (lo, hi) open range, and whose sign at lo is strictly
// positive or negative (never 0). The function will return an approximate value for the root of q_exact within (lo,
// hi). How close the approximate result is to the exact root depends on the tightness of the coefficients of q and on
// the magnitude of the derivative of q_exact around the root (a "flatter" polynomial will result in a less precise
// approximation of the root of q_exact).
//
// The coefficients of q are represented as interval enclosures via values of the scalar floating-point type fp_t. The
// interval coefficients are stored in ascending degree order in the monomial basis via the usual flattened lo-negated
// format:
//
// [-c0_lo, c0_hi, -c1_lo, c1_hi, -c2_lo, c2_hi, ...]
//
// The C++ prototype of the root finder is
//
// void name(const ext_fp_t *cfs, const ext_fp_t *bounds, const std::int32_t *sign, ext_fp_t *out)
//
// where ext_fp_t is the C++ type corresponding to fp_t and:
//
// - cfs points to the 2 * (n + 1) values of the interval coefficients,
// - bounds points to the values (lo, hi),
// - sign points to the sign of q_exact at lo,
// - out points to the output value x.
//
// The following preconditions must hold:
//
// - the coefficients of q enclose the coefficients of q_exact,
// - lo and hi are finite and 0 <= lo < hi,
// - lo/2, hi/2 and lo/2 + hi/2 are exactly representable in fp_t,
// - the interval coefficients are valid intervals (in the sense of llvm_ival_guard),
// - q_exact has exactly one root in (lo, hi), and it is simple,
// - *sign is the sign of q_exact at lo, which is either -1 or 1.
//
// The root is searched via a safeguarded Newton iteration based on rtsafe() from Numerical Recipes, within a bracket
// which is initially (lo, hi), and which is shrunk according to the sign of the interval evaluation of q. The value x
// written to out lies in (lo, hi). The search stops when the interval evaluation of q at x contains zero, or when a
// bisection step is required and the computed midpoint of the current bracket does not lie strictly inside it (i.e.,
// the bracket cannot be bisected further in floating-point arithmetic).
//
// If a global value with the same name exists already in the module, an error will be raised.
//
// NOTE: the search terminates after a finite number of steps, and the bracket contains the root of q_exact at every
// step. The returned x is however not guaranteed to be close to the root: the search stops at the first x at which the
// interval evaluation of q contains zero, i.e., at which the value of q_exact is within the evaluation error of zero
// (which is small if the bounds of the coefficients of q are tight). Usually such an x is close to the root, but in
// principle it could also be a "false" root away from it, e.g., a stationary point of q_exact at which its value is
// very close to zero.
llvm::Function *irr_add_poly_root_finder(const std::string &name, llvm_state &s, llvm::Type *const fp_t,
                                         const std::uint32_t n)
{
    assert(fp_t != nullptr);
    assert(!fp_t->isVectorTy());

    // Overflow check: we need to be able to index into cfs via std::uint32_t.
    try {
        using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;
        static_cast<void>(static_cast<std::uint32_t>((safe_uint32_t(n) + 1) * 2));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in irr_add_poly_root_finder()");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Remember and restore the current insertion point.
    const llvm::IRBuilderBase::InsertPointGuard ipg(bld);

    // The function arguments:
    //
    // - pointer to the array of poly coefficients (read-only),
    // - pointer to the (lo, hi) bounds (read-only),
    // - pointer to the sign (read-only),
    // - pointer to the output.
    //
    // No overlap is allowed. All floating-point pointers are to external types. The function does not return anything.
    std::array<llvm::Type *, 4> fargs{};
    std::ranges::fill(fargs, bld.getPtrTy());
    auto *const ft = llvm::FunctionType::get(bld.getVoidTy(), fargs, false);
    // Now create the function.
    auto *const f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &md);

    // Set the names/attributes of the function arguments.
    auto *const cfs_ptr = f->args().begin();
    cfs_ptr->setName("cfs_ptr");
    llvm_add_no_capture_argattr(s, cfs_ptr);
    cfs_ptr->addAttr(llvm::Attribute::NoAlias);
    cfs_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *const bounds_ptr = f->args().begin() + 1;
    bounds_ptr->setName("bounds_ptr");
    llvm_add_no_capture_argattr(s, bounds_ptr);
    bounds_ptr->addAttr(llvm::Attribute::NoAlias);
    bounds_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *const sign_ptr = f->args().begin() + 2;
    sign_ptr->setName("sign_ptr");
    llvm_add_no_capture_argattr(s, sign_ptr);
    sign_ptr->addAttr(llvm::Attribute::NoAlias);
    sign_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *const out_ptr = f->args().begin() + 3;
    out_ptr->setName("out_ptr");
    llvm_add_no_capture_argattr(s, out_ptr);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    // Fetch the external type.
    auto *const ext_fp_t = make_external_llvm_type(fp_t);

    // Create a new basic block to start insertion into.
    auto *const bb = llvm::BasicBlock::Create(ctx, "entry", f);
    bld.SetInsertPoint(bb);

    // Setup the interval arithmetic guard.
    //
    // NOTE: in addition to interval arithmetic, this function also performs regular (non-interval) floating-point
    // arithmetic, which in the portable codepath is rounded up instead of to nearest. This is ok because regular
    // arithmetic is used only to compute the initial value of x, which is exact by the preconditions, and to produce
    // candidate values for x (Newton steps, including the evaluation of the derivative and the convergence speed check,
    // and bisection midpoints). A candidate is accepted only if it lies strictly inside the current bracket, which is
    // checked via comparisons, and comparisons are not affected by the rounding mode. Thus, the rounding mode may
    // affect the speed of convergence and the final value of x, but not the guarantees stated in the function
    // documentation.
    //
    // NOTE: the documentation of llvm_ival_guard recommends performing all floating-point operations via its member
    // functions. Here we deviate from it for regular arithmetic, for which the guard provides no member functions:
    // llvm_fadd()/llvm_fsub()/llvm_fmul() use the builder as configured by the guard (constrained intrinsics with
    // dynamic rounding in the portable codepath, calls to MPFR wrappers for mppp::real, which the guard allows), and
    // thus they generate valid IR.
    const auto ivg = llvm_ival_guard(s, f, fp_t);

    // Generate the 0.5 constant.
    auto *const half = llvm_constantfp(s, fp_t, 0.5);

    // Get the boolean type.
    auto *const int1_t = bld.getInt1Ty();

    // Load the coefficients for the polynomial q and its derivative q'. The coefficients for q are intervals, the
    // coefficients for q' are scalars.
    std::vector<llvm::Value *> cfs, cfs_prime;
    cfs.reserve(n + 1u);
    cfs_prime.reserve(n);
    for (std::uint32_t i = 0; i <= n; ++i) {
        // Construct the current coefficient for q.
        auto *ptr = llvm_unsigned_gep(bld, ext_fp_t, cfs_ptr, bld.getInt32(i * 2u));
        auto *const cur_cf_mlo = ext_load_vector_from_memory(s, fp_t, ptr, 1);

        ptr = llvm_unsigned_gep(bld, ext_fp_t, cfs_ptr, bld.getInt32((i * 2u) + 1u));
        auto *const cur_cf_hi = ext_load_vector_from_memory(s, fp_t, ptr, 1);

        cfs.push_back(ivg.ival_construct(cur_cf_mlo, cur_cf_hi));

        // Construct the current coefficient for q'.
        if (i > 0u) {
            // Derive a scalar from the current interval coefficient: cur_cf = cur_cf_hi/2 - cur_cf_mlo/2. The q'
            // coefficient is then obtained by multiplying cur_cf by i.
            auto *const cur_cf_mlo_half = llvm_fmul(s, cur_cf_mlo, half);
            auto *const cur_cf_hi_half = llvm_fmul(s, cur_cf_hi, half);
            auto *cur_cf = llvm_fsub(s, cur_cf_hi_half, cur_cf_mlo_half);
            if (i != 1u) {
                auto *const mult = llvm_constantfp(s, fp_t, static_cast<double>(i));
                cur_cf = llvm_fmul(s, cur_cf, mult);
            }
            cfs_prime.push_back(cur_cf);
        }
    }

    // Load the (lo, hi) bounds.
    auto *const orig_lo = ext_load_vector_from_memory(s, fp_t, bounds_ptr, 1);
    auto *const orig_hi
        = ext_load_vector_from_memory(s, fp_t, llvm_unsigned_gep(bld, ext_fp_t, bounds_ptr, bld.getInt32(1)), 1);

    // Load the sign of the polynomial at lo (1 or -1 following the preconditions).
    auto *const sign_lo = bld.CreateLoad(bld.getInt32Ty(), sign_ptr);

    // s0_pos: true if q is positive at orig_lo, false if negative.
    auto *const s0_pos = bld.CreateICmpEQ(sign_lo, bld.getInt32(1));

    // The zero constant, used in the sign test.
    auto *const zero = llvm_constantfp(s, fp_t, 0.);

    // Compute the initial value of x (lo/2+hi/2) and dx/dxold (hi - lo). By the preconditions, x is computed exactly
    // (in any rounding mode).
    auto *const lo_half = llvm_fmul(s, orig_lo, half);
    auto *const hi_half = llvm_fmul(s, orig_hi, half);
    auto *const orig_x = llvm_fadd(s, lo_half, hi_half);
    auto *const orig_dx = llvm_fsub(s, orig_hi, orig_lo);

    // Create the allocas.
    auto *const lo_alloca = llvm_entry_alloca(s, fp_t);
    auto *const hi_alloca = llvm_entry_alloca(s, fp_t);
    auto *const x_alloca = llvm_entry_alloca(s, fp_t);
    auto *const dx_alloca = llvm_entry_alloca(s, fp_t);
    auto *const dx_old_alloca = llvm_entry_alloca(s, fp_t);
    auto *const flag_alloca = llvm_entry_alloca(s, int1_t);

    // Fill them in.
    bld.CreateStore(orig_lo, lo_alloca);
    bld.CreateStore(orig_hi, hi_alloca);
    bld.CreateStore(orig_x, x_alloca);
    bld.CreateStore(orig_dx, dx_alloca);
    bld.CreateStore(orig_dx, dx_old_alloca);
    // NOTE: the initial value of the flag is (lo < x && x < hi) which is guaranteed to be true under the preconditions.
    bld.CreateStore(bld.getInt1(true), flag_alloca);

    // Evaluate q (in interval arithmetic) and q' (in regular arithmetic) at x via Horner's scheme.
    const auto eval_q_qp = [&](llvm::Value *const x) {
        auto *q = cfs[n];
        auto *qp = (n == 0u) ? zero : cfs_prime[n - 1u];
        for (auto i = n; i > 0u; --i) {
            // NOTE: we are iterating backwards from the high-order coefficients to the low-order ones.
            const auto cur_degree = i - 1u;

            // NOTE: here we can use ival_mul_nn_val() because we are guaranteeing that x is non-negative (x is strictly
            // inside a bracket with lo>=0).
            q = ivg.ival_add(ivg.ival_mul_nn_val(q, x), cfs[cur_degree]);
            if (cur_degree > 0u) {
                qp = llvm_fadd(s, llvm_fmul(s, qp, x), cfs_prime[cur_degree - 1u]);
            }
        }

        // NOTE: the return value is the pair (enclosure of q(x), approximation of q'(x)).
        return std::make_pair(q, qp);
    };

    // The main loop.
    //
    // NOTE: at each iteration, the loop begins with a bracket (lo, hi) guaranteed to contain the root and a point x,
    // with x representing the current candidate value for the root.
    //
    // q and q' are then evaluated at x. If the sign of q at x is certain, then the bracket is shrunk accordingly, with
    // x replacing lo or hi (depending on the sign). If the sign of q at x is undecided, then the bracket remains (lo,
    // hi), the loop won't iterate further and the values computed in the rest of the loop body will be ignored (i.e., x
    // will be returned after the loop exit).
    //
    // Next, two new candidates for x are determined.
    //
    // The first candidate is determined with the Newton rule, the second one with a simple bisection as the midpoint of
    // the new bracket. Both candidates must fall strictly within the new bracket, and the Newton candidate is also
    // subject to a convergence speed check (which, if failed, eliminates the Newton candidate from consideration).
    //
    // If either the sign of q at x was determined to be undecided, or if both new candidates are eliminated because
    // they do not satisfy the stated constraints, then the loop will exit, and the current value of x won't be updated.
    // Otherwise, the new candidate takes the place of the current x (with precedence given to the Newton candidate) for
    // the next iteration.
    llvm_while_loop(
        s, [&] -> llvm::Value * { return bld.CreateLoad(int1_t, flag_alloca); },
        [&] {
            // Load lo, hi, x, dx and dxold.
            auto *const cur_lo = bld.CreateLoad(fp_t, lo_alloca);
            auto *const cur_hi = bld.CreateLoad(fp_t, hi_alloca);
            auto *const cur_x = bld.CreateLoad(fp_t, x_alloca);
            auto *const cur_dx = bld.CreateLoad(fp_t, dx_alloca);
            auto *const cur_dx_old = bld.CreateLoad(fp_t, dx_old_alloca);

            // Evaluate q and q' at x.
            const auto [q, qp] = eval_q_qp(cur_x);

            // Extract q into the interval components.
            const auto [q_mlo, q_hi] = ivg.ival_extract(q);

            // Determine if q(x) is certainly positive (lower bound -q_mlo > 0) or certainly negative (upper bound q_hi
            // < 0).
            auto *const q_pos = ivg.val_lt(q_mlo, zero);
            auto *const q_neg = ivg.val_lt(q_hi, zero);

            // Is the sign of q at x determined?
            auto *const decided = bld.CreateLogicalOr(q_pos, q_neg);

            // x is left of the root if q(x) certainly has the same sign as q(orig_lo), right of the root if it
            // certainly has the opposite sign. If the sign of q(x) is undecided, x_left and x_right are both false.
            auto *const x_left = bld.CreateSelect(s0_pos, q_pos, q_neg);
            auto *const x_right = bld.CreateSelect(s0_pos, q_neg, q_pos);

            // Determine the new bracket bounds. These are the same as the current bounds if the sign of q at x is
            // undecided.
            auto *const new_lo = bld.CreateSelect(x_left, cur_x, cur_lo);
            auto *const new_hi = bld.CreateSelect(x_right, cur_x, cur_hi);

            // The Newton candidate x - q(x)/q'(x), using the midpoint of the enclosure of q(x) as the value of q(x).
            auto *const q_mid = llvm_fsub(s, llvm_fmul(s, q_hi, half), llvm_fmul(s, q_mlo, half));
            auto *const newton_step = llvm_fdiv(s, q_mid, qp);
            auto *const newton_x = llvm_fsub(s, cur_x, newton_step);

            // The Newton candidate must lie strictly inside the new bracket. This is false also if newton_x is NaN or
            // infinite (e.g., if q'(x) is zero).
            auto *const newton_inside
                = bld.CreateLogicalAnd(ivg.val_lt(new_lo, newton_x), ivg.val_lt(newton_x, new_hi));

            // rtsafe's convergence speed check: the Newton step is rejected if it is larger than half the step taken
            // two iterations ago (dxold, initially, the width of the original bracket). As in rtsafe, the check is
            // written as |2*q(x)| > |dxold*q'(x)|, without a division.
            auto *const two_q_mid = llvm_fadd(s, q_mid, q_mid);
            auto *const too_slow = ivg.val_lt(ivg.val_abs(llvm_fmul(s, cur_dx_old, qp)), ivg.val_abs(two_q_mid));

            // Use the Newton candidate if it is inside the bracket and fast enough.
            auto *const use_newton = bld.CreateLogicalAnd(newton_inside, bld.CreateNot(too_slow));

            // The bisection candidate: the midpoint of the new bracket.
            auto *const bisect_x = llvm_fadd(s, llvm_fmul(s, new_lo, half), llvm_fmul(s, new_hi, half));

            // The bisection candidate can be used only if it lies strictly inside the new bracket. If it does not, the
            // bracket cannot be bisected further in floating-point arithmetic.
            auto *const bisect_inside
                = bld.CreateLogicalAnd(ivg.val_lt(new_lo, bisect_x), ivg.val_lt(bisect_x, new_hi));

            // The next point: the Newton candidate if accepted, the bisection candidate otherwise.
            auto *const next_x = bld.CreateSelect(use_newton, newton_x, bisect_x);

            // Continue the search only if the sign of q(x) was decided and the next point lies strictly inside the new
            // bracket.
            auto *const next_flag = bld.CreateLogicalAnd(decided, bld.CreateLogicalOr(use_newton, bisect_inside));

            // The step taken in this iteration: the Newton step, or half the width of the new bracket for a bisection.
            // Only its magnitude matters (it is used via |dxold| in the convergence speed check).
            auto *const bisect_step = llvm_fmul(s, llvm_fsub(s, new_hi, new_lo), half);
            auto *const step_taken = bld.CreateSelect(use_newton, newton_step, bisect_step);

            // If the search continues, x moves to the next point. Otherwise x stays the current point, which is the
            // result.
            auto *const new_x = bld.CreateSelect(next_flag, next_x, cur_x);

            // Store the new state. The step of the previous iteration (cur_dx) becomes dxold, and the step taken in
            // this iteration becomes dx.
            bld.CreateStore(new_lo, lo_alloca);
            bld.CreateStore(new_hi, hi_alloca);
            bld.CreateStore(new_x, x_alloca);
            bld.CreateStore(cur_dx, dx_old_alloca);
            bld.CreateStore(step_taken, dx_alloca);
            bld.CreateStore(next_flag, flag_alloca);
        });

    // Load the result from x_alloca.
    auto *const ret = bld.CreateLoad(fp_t, x_alloca);

    // Store it in the output pointer.
    ext_store_vector_to_memory(s, out_ptr, ret);

    // Finalise the guard before emitting the return statement.
    ivg.finalise();

    // Exit.
    bld.CreateRetVoid();

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE
