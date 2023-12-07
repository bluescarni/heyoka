// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

#if !defined(NDEBUG)

// Small helper to compute pi at the precision of the floating-point type tp.
// Used only for debugging purposes.
// NOTE: this is a repetition of number_like(), perhaps we can abstract this?
number inv_kep_E_pi_like(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);

    auto &context = s.context();

    if (tp == to_llvm_type<float>(context, false)) {
        return number{boost::math::constants::pi<float>()};
    } else if (tp == to_llvm_type<double>(context, false)) {
        return number{boost::math::constants::pi<double>()};
    } else if (tp == to_llvm_type<long double>(context, false)) {
        return number{boost::math::constants::pi<long double>()};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (tp == to_llvm_type<mppp::real128>(context, false)) {
        return number{mppp::pi_128};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto prec = llvm_is_real(tp)) {
        return number{mppp::real_pi(prec)};
#endif
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Unable to create a number of type '{}' from the constant pi", llvm_type_name(tp)));
    // LCOV_EXCL_STOP
}

#endif

// clang-format off
//
// Helper function to construct a dictionary of double-length approximations of 2*pi
// for floating-point types.
//
// The hi/lo parts of the 2*pi approximation can be computed via the following Python
// code (using heyoka.py's multiprecision class hy.real):
//
// >>> import heyoka as hy
// >>> # Set the desired precision in bits (e.g., 24 for single-precision).
// >>> prec = 24
// >>> # Decimal 2*pi approximation with many more digits than necessary (here
// >>> # corresponding to 4 times quadruple-precision, i.e., 113 * 4 bits).
// >>> twopi_str = '6.28318530717958647692528676655900576839433879875021164194988918461563281257241799725606965068423413596429617302656461329418768921910116447'
// >>> dl_twopi = hy.real(twopi_str, prec*2)
// >>> twopi_hi = hy.real(dl_twopi, prec)
// >>> twopi_lo = hy.real(dl_twopi - twopi_hi, prec)
// >>> assert(twopi_hi + twopi_lo == twopi_hi)
// >>> assert(hy.real(twopi_hi, prec*2) + hy.real(twopi_lo, prec*2) == dl_twopi)
// >>> print((twopi_hi, twopi_lo))
//
// clang-format on
auto make_dl_twopi_dict()
{
    std::unordered_map<std::type_index, std::pair<number, number>> retval;

    auto impl = [&retval](auto twopi_hi, auto twopi_lo) {
        using type = decltype(twopi_hi);
        static_assert(std::is_same_v<type, decltype(twopi_lo)>);

        assert(retval.find(typeid(type)) == retval.end());

        retval[typeid(type)] = std::make_pair(number{twopi_hi}, number{twopi_lo});
    };

    // Handle float.
    if (is_ieee754_binary32<float>) {
        impl(6.28318548f, -1.74845553e-7f);
    }

    // Handle double.
    if (is_ieee754_binary64<double>) {
        impl(6.2831853071795862, 2.4492935982947059e-16);
    }

    // Handle long double.
    if (is_ieee754_binary64<long double>) {
        impl(6.2831853071795862l, 2.4492935982947059e-16l);
    } else if (is_x86_fp80<long double>) {
        impl(6.28318530717958647703l, -1.00331152253366640475e-19l);
    } else if (is_ieee754_binary128<long double>) {
        impl(6.28318530717958647692528676655900559l, 1.73436202602475620495940880520867045e-34l);
    }

#if defined(HEYOKA_HAVE_REAL128)

    // Handle real128.
    impl(mppp::real128{"6.28318530717958647692528676655900559"},
         mppp::real128{"1.73436202602475620495940880520867045e-34"});

#endif

    return retval;
};

// NOLINTNEXTLINE(cert-err58-cpp)
const auto dl_twopi_dict = make_dl_twopi_dict();

std::pair<number, number> inv_kep_E_dl_twopi_like(llvm_state &s, llvm::Type *fp_t)
{
    if (fp_t->isFloatingPointTy() && fp_t->isIEEE()) {
#if !defined(NDEBUG)
        // Sanity check.
        const auto &sem = fp_t->getFltSemantics();
        const auto prec = llvm::APFloatBase::semanticsPrecision(sem);

        assert(prec <= 113u);
#endif

        auto impl = [](auto val) {
            using type = decltype(val);
            const auto it = dl_twopi_dict.find(typeid(type));

            // LCOV_EXCL_START
            if (it == dl_twopi_dict.end()) {
                throw std::invalid_argument(
                    fmt::format("Cannot generate a double-length 2*pi approximation for the C++ type '{}'",
                                boost::core::demangle(typeid(type).name())));
            }
            // LCOV_EXCL_STOP

            return it->second;
        };

        auto &context = s.context();

        if (fp_t == to_llvm_type<float>(context, false)) {
            return impl(0.f);
        } else if (fp_t == to_llvm_type<double>(context, false)) {
            return impl(0.);
        } else if (fp_t == to_llvm_type<long double>(context, false)) {
            return impl(0.l);
#if defined(HEYOKA_HAVE_REAL128)
        } else if (fp_t == to_llvm_type<mppp::real128>(context, false)) {
            return impl(mppp::real128(0));
#endif
        }

        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format(
            "Cannot generate a double-length 2*pi approximation for the LLVM type '{}'", detail::llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP

#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto prec = llvm_is_real(fp_t)) {
        // Generate the 2*pi constant with prec * 4 precision.
        auto twopi = mppp::real_pi(boost::safe_numerics::safe<::mpfr_prec_t>(prec) * 4);
        mppp::mul_2ui(twopi, twopi, 1ul);

        // Fetch the hi/lo components in precision prec.
        auto twopi_hi = mppp::real{twopi, prec};
        auto twopi_lo = mppp::real{std::move(twopi) - twopi_hi, prec};

        assert(twopi_hi + twopi_lo == twopi_hi); // LCOV_EXCL_LINE

        return std::make_pair(number(std::move(twopi_hi)), number(std::move(twopi_lo)));
#endif
        // LCOV_EXCL_START
    } else {
        throw std::invalid_argument(fmt::format("Cannot generate a double-length 2*pi approximation for the type '{}'",
                                                detail::llvm_type_name(fp_t)));
    }
    // LCOV_EXCL_STOP
}

// Small helper to return the epsilon of the floating-point type tp as a number.
// NOTE: this is a repetition of number_like(), perhaps we can abstract this?
number inv_kep_E_eps_like(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);

    auto &context = s.context();

    if (tp == to_llvm_type<float>(context, false)) {
        return number{std::numeric_limits<float>::epsilon()};
    } else if (tp == to_llvm_type<double>(context, false)) {
        return number{std::numeric_limits<double>::epsilon()};
    } else if (tp == to_llvm_type<long double>(context, false)) {
        return number{std::numeric_limits<long double>::epsilon()};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (tp == to_llvm_type<mppp::real128>(context, false)) {
        return number{std::numeric_limits<mppp::real128>::epsilon()};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto prec = llvm_is_real(tp)) {
        return number{eps_from_prec(prec)};
#endif
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Unable to create a number version of the epsilon for the type '{}'", llvm_type_name(tp)));
    // LCOV_EXCL_STOP
}

// Helper to clamp the floating-point value x to the range [lb, ub].
// This is guaranteed to return NaN if x is NaN.
// NOTE: perhaps a dedicated clamp() primitive could give better
// performance for real? In such a case, we would need to take
// care that NaN is handled correctly.
llvm::Value *llvm_clamp(llvm_state &s, llvm::Value *x, llvm::Value *lb, llvm::Value *ub)
{
    // NOTE: if x is NaN, then ret will remain NaN after these comparisons.
    auto *ret = llvm_max(s, x, lb);
    ret = llvm_min(s, ret, ub);

    return ret;
}

// Helper to accurately reduce the input floating-point value x to the standard trigonometric
// range [0, 2pi). This will return NaN if x is NaN.
llvm::Value *llvm_trig_arg_reduce(llvm_state &s, llvm::Value *x)
{
    // The type of x.
    auto *tp = x->getType();

    // The scalar type of x.
    auto *fp_t = tp->getScalarType();

    // NOTE: the current implementation employs double-length arithmetic
    // to ameliorate catastrophic cancellation. A more rigorous approach
    // is described in the classic paper:
    //
    // https://redirect.cs.umbc.edu/~phatak/645/supl/Ng-ArgReduction.pdf
    //
    // Another pragmatic approach is to compute sin(x)/cos(x) (on the assumption
    // of properly-implemented sin/cos primitives) and then use atan2(). Performance
    // vs accuracy tradeoff wrt the current approach is to be assessed.

    // Fetch 2pi in double-length precision.
    const auto [dl_twopi_hi, dl_twopi_lo] = inv_kep_E_dl_twopi_like(s, fp_t);

#if !defined(NDEBUG)
    assert(dl_twopi_hi == number_like(s, fp_t, 2.) * inv_kep_E_pi_like(s, fp_t)); // LCOV_EXCL_LINE
#endif

    // Reduce x modulo 2*pi in extended precision.
    auto *retval = llvm_dl_modulus(s, x, llvm_constantfp(s, tp, 0.), llvm_codegen(s, tp, dl_twopi_hi),
                                   llvm_codegen(s, tp, dl_twopi_lo))
                       .first;

    // NOTE: I am not 100% sure that double-length arithmetic is guaranteed to return
    // a hi part in the [0, 2pi) range for the reduction. Just to be sure, let us further clamp.
    return llvm_clamp(s, retval, llvm_constantfp(s, tp, 0.),
                      // NOTE: half-open interval implemented as closed interval with
                      // upper bound set to the floating-point number immediately preceding 2pi.
                      llvm_codegen(s, tp, nextafter(dl_twopi_hi, number_like(s, fp_t, 0.))));
}

} // namespace

// Implementation of the inverse Kepler equation for the eccentric anomaly E.
llvm::Function *llvm_add_inv_kep_E(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch vector floating-point type.
    auto *tp = make_vector_type(fp_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka.inv_kep_E.{}", llvm_mangle_type(tp));

    // The function arguments:
    // - eccentricity,
    // - mean anomaly.
    const std::vector<llvm::Type *> fargs{tp, tp};

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f != nullptr) {
        // The function was already created, return it.
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The return type is tp.
    auto *ft = llvm::FunctionType::get(tp, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *ecc_arg = f->args().begin();
    auto *M_arg = f->args().begin() + 1;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Is the eccentricity a quiet NaN or less than 0?
    auto *ecc_is_nan_or_neg = llvm_fcmp_ult(s, ecc_arg, llvm_constantfp(s, tp, 0.));
    // Is the eccentricity >= 1?
    auto *ecc_is_gte1 = llvm_fcmp_oge(s, ecc_arg, llvm_constantfp(s, tp, 1.));

    // Is the eccentricity NaN or out of range?
    // NOTE: this is a logical OR.
    auto *ecc_invalid = builder.CreateSelect(
        ecc_is_nan_or_neg, llvm::ConstantInt::getAllOnesValue(ecc_is_nan_or_neg->getType()), ecc_is_gte1);

    // Replace invalid eccentricity values with quiet NaNs.
    auto *ecc
        = builder.CreateSelect(ecc_invalid, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), ecc_arg);

    // Reduce M to the [0, 2pi) range.
    auto *M = llvm_trig_arg_reduce(s, M_arg);

    // Compute the initial guess from the usual elliptic expansion
    // to the third order in eccentricities:
    // E = M + e*sin(M) + e**2*sin(M)*cos(M) + e**3*sin(M)*(3/2*cos(M)**2 - 1/2) + ...
    auto [sin_M, cos_M] = llvm_sincos(s, M);
    // e*sin(M).
    auto *e_sin_M = llvm_fmul(s, ecc, sin_M);
    // e*cos(M).
    auto *e_cos_M = llvm_fmul(s, ecc, cos_M);
    // e**2.
    auto *e2 = llvm_fmul(s, ecc, ecc);
    // cos(M)**2.
    auto *cos_M_2 = llvm_fmul(s, cos_M, cos_M);

    // 3/2 and 1/2 constants.
    auto *c_3_2 = llvm_codegen(s, tp, number_like(s, fp_t, 3. / 2));
    auto *c_1_2 = llvm_codegen(s, tp, number_like(s, fp_t, 1. / 2));

    // M + e*sin(M).
    auto *tmp1 = llvm_fadd(s, M, e_sin_M);
    // e**2*sin(M)*cos(M).
    auto *tmp2 = llvm_fmul(s, e_sin_M, e_cos_M);
    // e**3*sin(M).
    auto *tmp3 = llvm_fmul(s, e2, e_sin_M);
    // 3/2*cos(M)**2 - 1/2.
    auto *tmp4 = llvm_fsub(s, llvm_fmul(s, c_3_2, cos_M_2), c_1_2);

    // Put it together.
    auto *ig1 = llvm_fadd(s, tmp1, tmp2);
    auto *ig2 = llvm_fmul(s, tmp3, tmp4);
    auto *ig = llvm_fadd(s, ig1, ig2);

    // Compute the initial bounding range - that is, the bounds for E which
    // are guaranteed to contain the root. These are [0, 2pi).
    const auto twopi_num = inv_kep_E_dl_twopi_like(s, fp_t).first;
    auto *lb_init = llvm_constantfp(s, tp, 0.);
    auto *ub_init = llvm_codegen(s, tp, nextafter(twopi_num, number_like(s, fp_t, 0.)));

    // Store them.
    auto *lb_storage = builder.CreateAlloca(tp);
    auto *ub_storage = builder.CreateAlloca(tp);
    builder.CreateStore(lb_init, lb_storage);
    builder.CreateStore(ub_init, ub_storage);

    // Clamp the initial guess to the initial bounding range.
    // NOTE: in case ig ends up being NaN (because the arguments are NaN or for whatever
    // other reason), then ig will remain NaN.
    ig = llvm_clamp(s, ig, lb_init, ub_init);

    // Store the initial guess in the storage for the return value. This will hold the
    // iteratively-determined value for E.
    auto *retval = builder.CreateAlloca(tp);
    builder.CreateStore(ig, retval);

    // Create the counter.
    auto *counter = builder.CreateAlloca(builder.getInt32Ty());
    builder.CreateStore(builder.getInt32(0), counter);

    // Variables to store sin(E) and cos(E).
    auto *sin_E = builder.CreateAlloca(tp);
    auto *cos_E = builder.CreateAlloca(tp);

    // Write the initial values for sin_E and cos_E.
    auto sin_cos_E = llvm_sincos(s, builder.CreateLoad(tp, retval));
    builder.CreateStore(sin_cos_E.first, sin_E);
    builder.CreateStore(sin_cos_E.second, cos_E);

    // Helper to compute f(E).
    auto fE_compute = [&]() {
        // e*sin(E).
        auto *e_sinE = llvm_fmul(s, ecc, builder.CreateLoad(tp, sin_E));
        // E - M.
        auto *e_m_M = llvm_fsub(s, builder.CreateLoad(tp, retval), M);
        // E - M - e*sin(E).
        return llvm_fsub(s, e_m_M, e_sinE);
    };
    // Variable to hold the value of f(E) = E - e*sin(E) - M.
    auto *fE = builder.CreateAlloca(tp);
    // Compute and store the initial value of f(E).
    builder.CreateStore(fE_compute(), fE);

    // Create a variable to hold the result of the tolerance check
    // computed at the beginning of each iteration of the NR loop.
    // This is "true" if the loop needs to continue, or "false" if the
    // loop can stop because we achieved the desired tolerance.
    // NOTE: this is only allocated, it will be immediately written to
    // the first time loop_cond is evaluated.
    // NOTE: this needs to persist outside loop_cond because we will use it
    // after the loop in case we exceeded the max iter number.
    auto *vec_bool_t = make_vector_type(builder.getInt1Ty(), batch_size);
    auto *tol_check_ptr = builder.CreateAlloca(vec_bool_t);

    // Define the continuing condition functor. This will return "true"
    // if the NR loop needs to continue, "false" if it can be stopped
    // (either because a solution has been found, or because the max
    // number of iterations has been exceeded).
    // NOTE: hard-code max_iter for the time being. It would probably
    // make sense to make it dependent on the epsilon of fp_t though?
    auto *max_iter = builder.getInt32(50);
    auto loop_cond = [&]() -> llvm::Value * {
        // NOTE: we use an *absolute* tolerance of 4*eps for both the check
        // on the magnitude of f(E) and on the magnitude of the bounding
        // range. This of course means that the final result is not guaranteed
        // to be the best possible approximation of E, for at least the
        // following reasons:
        // - abs(f(E)) < tol is a poor criterion when the derivative of
        //   f(E) is small, because large variations in E result in small
        //   variations of f(E);
        // - the check on the bounding range should not use an absolute
        //   tolerance but a relative one (i.e., rescaled wrt the extrema
        //   of the bounds). This is what Boost's bisection implementation does,
        //   but I am not sure I understand if/how this works in case of a root
        //   equal to or very close to zero.
        // In the future, we should consider:
        // - understanding and enabling bounding range checking with relative tolerance,
        // - take a further step if we stop because abs(f(E)) < tol and the magnitude
        //   of the derivative of f(E) is less than 1: use the value of
        //   the derivative of f(E) (computed during the NR step) to find a small bounding
        //   range and take a few bisection steps to refine the solution. Or maybe directly
        //   compute the position of the root using the linear approximation of f(E)?
        auto *tol = llvm_codegen(s, tp, inv_kep_E_eps_like(s, fp_t) * number_like(s, fp_t, 4.));

        // Load the current values of f(E) and E.
        auto *cur_fE = builder.CreateLoad(tp, fE);
        auto *cur_E = builder.CreateLoad(tp, retval);

        // Compute the sign of f(E).
        // NOTE: this will be zero if f(E) is NaN.
        auto *fE_sgn = llvm_sgn(s, cur_fE);

        // Update the bounds.
        // NOTE: if f(E) is NaN, then new_ub/new_lb will also be set to NaN.
        auto *cur_ub = builder.CreateLoad(tp, ub_storage);
        auto *cur_lb = builder.CreateLoad(tp, lb_storage);
        auto *zero_c = vector_splat(builder, builder.getInt32(0), batch_size);
        auto *new_ub = builder.CreateSelect(builder.CreateICmpSGE(fE_sgn, zero_c), cur_E, cur_ub);
        auto *new_lb = builder.CreateSelect(builder.CreateICmpSLE(fE_sgn, zero_c), cur_E, cur_lb);
        builder.CreateStore(new_ub, ub_storage);
        builder.CreateStore(new_lb, lb_storage);

        // Compute the size of the new bounding range.
        auto *bsize = llvm_fsub(s, new_ub, new_lb);

        // First tolerance check: abs(f(E)) > tol.
        // NOTE: if E is NaN, then f(E) is NaN and tol1_check is false.
        auto *tol1_check = llvm_fcmp_ogt(s, llvm_abs(s, cur_fE), tol);

        // Second tolerance check: (ub - lb) > tol.
        // NOTE: if E is NaN, then tol2_check is false.
        auto *tol2_check = llvm_fcmp_ogt(s, bsize, tol);

        // Put them together with a logical AND.
        auto *tol_check
            = builder.CreateSelect(tol1_check, tol2_check, llvm::ConstantInt::get(tol2_check->getType(), 0u));
        // NOTE: we need OR reduction in batch mode: continue if *any* element of the batch
        // needs more iterations.
        auto *tol_cond = (batch_size == 1u) ? tol_check : builder.CreateOrReduce(tol_check);

        // Store the result of the tolerance check.
        builder.CreateStore(tol_check, tol_check_ptr);

        // Check the number of iterations.
        auto *c_cond = builder.CreateICmpULT(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter);

        // Combine tolerance check and number of iterations check with a logical AND.
        return builder.CreateSelect(c_cond, tol_cond, llvm::ConstantInt::get(tol_cond->getType(), 0u));
    };

    // Run the loop.
    llvm_while_loop(s, loop_cond, [&]() {
        // Compute the new value via the Newton-Raphson formula.
        auto *old_val = builder.CreateLoad(tp, retval);
        auto *one_c = llvm_constantfp(s, tp, 1.);
        auto *fdiv = llvm_fdiv(s, builder.CreateLoad(tp, fE),
                               llvm_fsub(s, one_c, llvm_fmul(s, ecc, builder.CreateLoad(tp, cos_E))));
        auto *new_val = llvm_fsub(s, old_val, fdiv);

        // Bisect if new_val > cur_ub.
        auto *half_c = llvm_codegen(s, tp, number_like(s, fp_t, 1. / 2));
        auto *cur_ub = builder.CreateLoad(tp, ub_storage);
        auto *bcheck = llvm_fcmp_ogt(s, new_val, cur_ub);
        new_val = builder.CreateSelect(bcheck, llvm_fmul(s, half_c, llvm_fadd(s, old_val, cur_ub)), new_val);

        // Bisect if new_val < cur_lb.
        auto *cur_lb = builder.CreateLoad(tp, lb_storage);
        bcheck = llvm_fcmp_olt(s, new_val, cur_lb);
        new_val = builder.CreateSelect(bcheck, llvm_fmul(s, half_c, llvm_fadd(s, old_val, cur_lb)), new_val);

        // Store the new value.
        builder.CreateStore(new_val, retval);

        // Update sin_E/cos_E.
        sin_cos_E = llvm_sincos(s, new_val);
        builder.CreateStore(sin_cos_E.first, sin_E);
        builder.CreateStore(sin_cos_E.second, cos_E);

        // Update f(E).
        builder.CreateStore(fE_compute(), fE);

        // Update the counter.
        builder.CreateStore(builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), counter), builder.getInt32(1)),
                            counter);
    });

    // After exiting the NR loop, check the counter.
    llvm_if_then_else(
        s, builder.CreateICmpEQ(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter),
        [&]() {
            // Load the tol_check variable.
            auto *tol_check = builder.CreateLoad(vec_bool_t, tol_check_ptr);

            // Set to quiet NaN in the return value all the lanes for which tol_check is 1.
            auto *old_val = builder.CreateLoad(tp, retval);
            auto *new_val = builder.CreateSelect(
                tol_check, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), old_val);
            builder.CreateStore(new_val, retval);

            llvm_invoke_external(s, "heyoka_inv_kep_E_max_iter", builder.getVoidTy(), {},
                                 llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                                          {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn}));
        },
        []() {});

    // Return the result.
    builder.CreateRet(builder.CreateLoad(tp, retval));

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

// Helper to create a wrapper for kepE() usable from C++ code.
// Input/output is done through pointers.
void llvm_add_inv_kep_E_wrapper(llvm_state &s, llvm::Type *scal_t, std::uint32_t batch_size, const std::string &name)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Make sure the function does not exist already.
    assert(md.getFunction(name) == nullptr); // LCOV_EXCL_LINE

    // Add the implementation function.
    auto *impl_f = llvm_add_inv_kep_E(s, scal_t, batch_size);

    // Fetch the external type.
    auto *ext_fp_t = llvm_ext_type(scal_t);

    // The function arguments:
    // - output pointer (write only),
    // - input ecc and mean anomaly pointers (read only).
    // No overlap allowed.
    const std::vector<llvm::Type *> fargs(3u, llvm::PointerType::getUnqual(ext_fp_t));
    // The return type is void.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    // Create the function
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &md);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // Setup the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *ecc_ptr = f->args().begin() + 1;
    ecc_ptr->setName("ecc_ptr");
    ecc_ptr->addAttr(llvm::Attribute::NoCapture);
    ecc_ptr->addAttr(llvm::Attribute::NoAlias);
    ecc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *M_ptr = f->args().begin() + 2;
    M_ptr->setName("M_ptr");
    M_ptr->addAttr(llvm::Attribute::NoCapture);
    M_ptr->addAttr(llvm::Attribute::NoAlias);
    M_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Load the data from the pointers.
    auto *ecc = ext_load_vector_from_memory(s, scal_t, ecc_ptr, batch_size);
    auto *M = ext_load_vector_from_memory(s, scal_t, M_ptr, batch_size);

    // Invoke the implementation function.
    auto *ret = builder.CreateCall(impl_f, {ecc, M});

    // Store the result.
    ext_store_vector_to_memory(s, out_ptr, ret);

    // Return.
    builder.CreateRetVoid();

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);
}

// Implementation of the inverse Kepler equation for the eccentric longitude F.
// https://articles.adsabs.harvard.edu//full/1972CeMec...5..303B/0000309.000.html
llvm::Function *llvm_add_inv_kep_F(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch vector floating-point type.
    auto *tp = make_vector_type(fp_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka.inv_kep_F.{}", llvm_mangle_type(tp));

    // The function arguments:
    // - h and k,
    // - mean longitude.
    const std::vector<llvm::Type *> fargs{tp, tp, tp};

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f != nullptr) {
        // The function was already created, return it.
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The return type is tp.
    auto *ft = llvm::FunctionType::get(tp, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *h_arg = f->args().begin();
    auto *k_arg = f->args().begin() + 1;
    auto *lam_arg = f->args().begin() + 2;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // h**2 + k**2.
    auto *h2 = llvm_square(s, h_arg);
    auto *k2 = llvm_square(s, k_arg);
    auto *h2k2 = llvm_fadd(s, h2, k2);

    // Is h2k2 a quiet NaN or >=1?
    auto *h2k2_invalid = llvm_fcmp_uge(s, h2k2, llvm_constantfp(s, tp, 1.));

    // Replace invalid values of h and k with quiet NaNs.
    auto *h
        = builder.CreateSelect(h2k2_invalid, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), h_arg);
    auto *k
        = builder.CreateSelect(h2k2_invalid, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), k_arg);

    // Reduce lam to the [0, 2pi) range.
    auto *lam = llvm_trig_arg_reduce(s, lam_arg);

    // Compute the initial guess according to Dario Izzo (tranquilo bepi):
    // L + k sL - h cL + (k^2-h^2) cL sL + hk (sL^2-cL^2) + 1/2 (ksL-hcL)(2(kcL+hsL)^2 - (ksL-hcL)^2),
    // where:
    // - L = lam,
    // - sL = sin(L)
    // - cL = cos(L).
    auto [sL, cL] = llvm_sincos(s, lam);
    // k*sL.
    auto *k_sL = llvm_fmul(s, k, sL);
    // h*cL.
    auto *h_cL = llvm_fmul(s, h, cL);
    // k*cL.
    auto *k_cL = llvm_fmul(s, k, cL);
    // h*sL.
    auto *h_sL = llvm_fmul(s, h, sL);
    // hk.
    auto *hk = llvm_fmul(s, h, k);
    // k^2-h^2.
    auto *k2_m_h2 = llvm_fsub(s, k2, h2);
    // sL^2-cL^2.
    auto *sL2_m_cl2 = llvm_fsub(s, llvm_square(s, sL), llvm_square(s, cL));
    // cL*sL.
    auto *cLsL = llvm_fmul(s, cL, sL);
    // k*sL-h*cL.
    auto *ksL_m_hcl = llvm_fsub(s, k_sL, h_cL);
    // k*cL+h*sL.
    auto *kcL_p_hsL = llvm_fadd(s, k_cL, h_sL);
    // 1/2 constant.
    auto *c_1_2 = llvm_codegen(s, tp, number_like(s, fp_t, 1. / 2));
    // (k*cL+h*sL)**2.
    auto *kcL_p_hsL2 = llvm_square(s, kcL_p_hsL);
    // (k*sL-h*cL)**2.
    auto *ksL_m_hcl2 = llvm_square(s, ksL_m_hcl);

    // Put it together.
    auto *ig1 = llvm_fadd(s, lam, ksL_m_hcl);
    auto *ig2 = llvm_fmul(s, k2_m_h2, cLsL);
    auto *ig3 = llvm_fmul(s, hk, sL2_m_cl2);

    auto *tmp1 = llvm_fmul(s, c_1_2, ksL_m_hcl);
    auto *tmp2 = llvm_fsub(s, llvm_fadd(s, kcL_p_hsL2, kcL_p_hsL2), ksL_m_hcl2);
    auto *ig4 = llvm_fmul(s, tmp1, tmp2);

    auto *ig = llvm_fadd(s, llvm_fadd(s, ig1, ig2), llvm_fadd(s, ig3, ig4));

    // Compute the initial bounding range - that is, the bounds for F which
    // are guaranteed to contain the root. These are [-1, 2pi + 1).
    const auto twopi_num = inv_kep_E_dl_twopi_like(s, fp_t).first;
    auto *lb_init = llvm_constantfp(s, tp, -1.);
    auto *ub_init = llvm_codegen(s, tp, nextafter(twopi_num + number_like(s, fp_t, 1.), number_like(s, fp_t, 0.)));

    // Store them.
    auto *lb_storage = builder.CreateAlloca(tp);
    auto *ub_storage = builder.CreateAlloca(tp);
    builder.CreateStore(lb_init, lb_storage);
    builder.CreateStore(ub_init, ub_storage);

    // Clamp the initial guess to the initial bounding range.
    // NOTE: in case ig ends up being NaN (because the arguments are NaN or for whatever
    // other reason), then ig will remain NaN.
    ig = llvm_clamp(s, ig, lb_init, ub_init);

    // Store the initial guess in the storage for the return value. This will hold the
    // iteratively-determined value for F.
    auto *retval = builder.CreateAlloca(tp);
    builder.CreateStore(ig, retval);

    // Create the counter.
    auto *counter = builder.CreateAlloca(builder.getInt32Ty());
    builder.CreateStore(builder.getInt32(0), counter);

    // Variables to store sin(F) and cos(F).
    auto *sin_F = builder.CreateAlloca(tp);
    auto *cos_F = builder.CreateAlloca(tp);

    // Write the initial values for sin_F and cos_F.
    auto sin_cos_F = llvm_sincos(s, builder.CreateLoad(tp, retval));
    builder.CreateStore(sin_cos_F.first, sin_F);
    builder.CreateStore(sin_cos_F.second, cos_F);

    // Helper to compute f(F).
    auto fF_compute = [&]() {
        // h*cos(F).
        auto *h_cosF = llvm_fmul(s, h, builder.CreateLoad(tp, cos_F));
        // k*sin(F).
        auto *k_sinF = llvm_fmul(s, k, builder.CreateLoad(tp, sin_F));
        // F - lam.
        auto *f_m_lam = llvm_fsub(s, builder.CreateLoad(tp, retval), lam);
        // F - lam + h*cos(F) - k*sin(F).
        return llvm_fsub(s, llvm_fadd(s, f_m_lam, h_cosF), k_sinF);
    };
    // Variable to hold the value of f(F).
    auto *fF = builder.CreateAlloca(tp);
    // Compute and store the initial value of f(F).
    builder.CreateStore(fF_compute(), fF);

    // Create a variable to hold the result of the tolerance check
    // computed at the beginning of each iteration of the NR loop.
    // This is "true" if the loop needs to continue, or "false" if the
    // loop can stop because we achieved the desired tolerance.
    // NOTE: this is only allocated, it will be immediately written to
    // the first time loop_cond is evaluated.
    // NOTE: this needs to persist outside loop_cond because we will use it
    // after the loop in case we exceeded the max iter number.
    auto *vec_bool_t = make_vector_type(builder.getInt1Ty(), batch_size);
    auto *tol_check_ptr = builder.CreateAlloca(vec_bool_t);

    // Define the continuing condition functor. This will return "true"
    // if the NR loop needs to continue, "false" if it can be stopped
    // (either because a solution has been found, or because the max
    // number of iterations has been exceeded).
    // NOTE: hard-code max_iter for the time being. It would probably
    // make sense to make it dependent on the epsilon of fp_t though?
    auto *max_iter = builder.getInt32(50);
    auto loop_cond = [&]() -> llvm::Value * {
        // NOTE: we use an *absolute* tolerance of 4*eps for both the check
        // on the magnitude of f(F) and on the magnitude of the bounding
        // range. This of course means that the final result is not guaranteed
        // to be the best possible approximation of F, for at least the
        // following reasons:
        // - abs(f(F)) < tol is a poor criterion when the derivative of
        //   f(F) is small, because large variations in F result in small
        //   variations of f(F);
        // - the check on the bounding range should not use an absolute
        //   tolerance but a relative one (i.e., rescaled wrt the extrema
        //   of the bounds). This is what Boost's bisection implementation does,
        //   but I am not sure I understand if/how this works in case of a root
        //   equal to or very close to zero.
        auto *tol = llvm_codegen(s, tp, inv_kep_E_eps_like(s, fp_t) * number_like(s, fp_t, 4.));

        // Load the current values of f(F) and F.
        auto *cur_fF = builder.CreateLoad(tp, fF);
        auto *cur_F = builder.CreateLoad(tp, retval);

        // Compute the sign of f(F).
        // NOTE: this will be zero if f(F) is NaN.
        auto *fF_sgn = llvm_sgn(s, cur_fF);

        // Update the bounds.
        // NOTE: if f(F) is NaN, then new_ub/new_lb will also be set to NaN.
        auto *cur_ub = builder.CreateLoad(tp, ub_storage);
        auto *cur_lb = builder.CreateLoad(tp, lb_storage);
        auto *zero_c = vector_splat(builder, builder.getInt32(0), batch_size);
        auto *new_ub = builder.CreateSelect(builder.CreateICmpSGE(fF_sgn, zero_c), cur_F, cur_ub);
        auto *new_lb = builder.CreateSelect(builder.CreateICmpSLE(fF_sgn, zero_c), cur_F, cur_lb);
        builder.CreateStore(new_ub, ub_storage);
        builder.CreateStore(new_lb, lb_storage);

        // Compute the size of the new bounding range.
        auto *bsize = llvm_fsub(s, new_ub, new_lb);

        // First tolerance check: abs(f(F)) > tol.
        // NOTE: if F is NaN, then f(F) is NaN and tol1_check is false.
        auto *tol1_check = llvm_fcmp_ogt(s, llvm_abs(s, cur_fF), tol);

        // Second tolerance check: (ub - lb) > tol.
        // NOTE: if F is NaN, then tol2_check is false.
        auto *tol2_check = llvm_fcmp_ogt(s, bsize, tol);

        // Put them together with a logical AND.
        auto *tol_check
            = builder.CreateSelect(tol1_check, tol2_check, llvm::ConstantInt::get(tol2_check->getType(), 0u));
        // NOTE: we need OR reduction in batch mode: continue if *any* element of the batch
        // needs more iterations.
        auto *tol_cond = (batch_size == 1u) ? tol_check : builder.CreateOrReduce(tol_check);

        // Store the result of the tolerance check.
        builder.CreateStore(tol_check, tol_check_ptr);

        // Check the number of iterations.
        auto *c_cond = builder.CreateICmpULT(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter);

        // Combine tolerance check and number of iterations check with a logical AND.
        return builder.CreateSelect(c_cond, tol_cond, llvm::ConstantInt::get(tol_cond->getType(), 0u));
    };

    // Run the loop.
    llvm_while_loop(s, loop_cond, [&]() {
        // Compute the new value via the Newton-Raphson formula.
        auto *old_val = builder.CreateLoad(tp, retval);
        auto *one_c = llvm_constantfp(s, tp, 1.);
        auto *diff = llvm_fsub(s, one_c, llvm_fmul(s, h, builder.CreateLoad(tp, sin_F)));
        diff = llvm_fsub(s, diff, llvm_fmul(s, k, builder.CreateLoad(tp, cos_F)));
        auto *fdiv = llvm_fdiv(s, builder.CreateLoad(tp, fF), diff);
        auto *new_val = llvm_fsub(s, old_val, fdiv);

        // Bisect if new_val > cur_ub.
        auto *half_c = llvm_codegen(s, tp, number_like(s, fp_t, 1. / 2));
        auto *cur_ub = builder.CreateLoad(tp, ub_storage);
        auto *bcheck = llvm_fcmp_ogt(s, new_val, cur_ub);
        new_val = builder.CreateSelect(bcheck, llvm_fmul(s, half_c, llvm_fadd(s, old_val, cur_ub)), new_val);

        // Bisect if new_val < cur_lb.
        auto *cur_lb = builder.CreateLoad(tp, lb_storage);
        bcheck = llvm_fcmp_olt(s, new_val, cur_lb);
        new_val = builder.CreateSelect(bcheck, llvm_fmul(s, half_c, llvm_fadd(s, old_val, cur_lb)), new_val);

        // Store the new value.
        builder.CreateStore(new_val, retval);

        // Update sin_F/cos_F.
        sin_cos_F = llvm_sincos(s, new_val);
        builder.CreateStore(sin_cos_F.first, sin_F);
        builder.CreateStore(sin_cos_F.second, cos_F);

        // Update f(F).
        builder.CreateStore(fF_compute(), fF);

        // Update the counter.
        builder.CreateStore(builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), counter), builder.getInt32(1)),
                            counter);
    });

    // After exiting the NR loop, check the counter.
    llvm_if_then_else(
        s, builder.CreateICmpEQ(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter),
        [&]() {
            // Load the tol_check variable.
            auto *tol_check = builder.CreateLoad(vec_bool_t, tol_check_ptr);

            // Set to quiet NaN in the return value all the lanes for which tol_check is 1.
            auto *old_val = builder.CreateLoad(tp, retval);
            auto *new_val = builder.CreateSelect(
                tol_check, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), old_val);
            builder.CreateStore(new_val, retval);

            llvm_invoke_external(s, "heyoka_inv_kep_F_max_iter", builder.getVoidTy(), {},
                                 llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                                          {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn}));
        },
        []() {});

    // Load the result.
    llvm::Value *ret = builder.CreateLoad(tp, retval);

    // Codegen 2pi, used below.
    auto *twopi_const = llvm_codegen(s, tp, twopi_num);

    // Reduce the result to the standard trigonometric range [0, 2pi).
    // NOTE: this reduction will not change ret if it is NaN.
    // Is ret < 0?
    auto *ret_lt_0 = llvm_fcmp_olt(s, ret, llvm_constantfp(s, tp, 0.));
    ret = builder.CreateSelect(ret_lt_0, llvm_fadd(s, twopi_const, ret), ret);

    // Is ret >= 2pi?
    auto *ret_ge_2pi = llvm_fcmp_oge(s, ret, twopi_const);
    ret = builder.CreateSelect(ret_ge_2pi, llvm_fsub(s, ret, twopi_const), ret);

    // Return the result.
    builder.CreateRet(ret);

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

// Implementation of the inverse Kepler equation for the delta eccentric anomaly.
// See Battin, section 4.3.
llvm::Function *llvm_add_inv_kep_DE(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch vector floating-point type.
    auto *tp = make_vector_type(fp_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka.inv_kep_DE.{}", llvm_mangle_type(tp));

    // The function arguments:
    // - s0 and c0,
    // - delta mean anomaly.
    const std::vector<llvm::Type *> fargs{tp, tp, tp};

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f != nullptr) {
        // The function was already created, return it.
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The return type is tp.
    auto *ft = llvm::FunctionType::get(tp, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *s0_arg = f->args().begin();
    auto *c0_arg = f->args().begin() + 1;
    auto *DM_arg = f->args().begin() + 2;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // s0**2 + c0**2.
    auto *s02 = llvm_square(s, s0_arg);
    auto *c02 = llvm_square(s, c0_arg);
    auto *s02c02 = llvm_fadd(s, s02, c02);

    // Is s02c02 a quiet NaN or >=1?
    auto *s02c02_invalid = llvm_fcmp_uge(s, s02c02, llvm_constantfp(s, tp, 1.));

    // Replace invalid values of s0 and c0 with quiet NaNs.
    auto *s0 = builder.CreateSelect(s02c02_invalid, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()),
                                    s0_arg);
    auto *c0 = builder.CreateSelect(s02c02_invalid, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()),
                                    c0_arg);

    // Reduce DM to the [0, 2pi) range.
    auto *DM = llvm_trig_arg_reduce(s, DM_arg);

    // Compute the initial guess following kep3:
    // https://github.com/esa/kep3/blob/e171a4a61431f3cd120898821c0d64c0e25a814e/src/core_astro/propagate_lagrangian.cpp#L60
    auto [sDM, cDM] = llvm_sincos(s, DM);
    // c0*cos(DM).
    auto *c0_cDM = llvm_fmul(s, c0, cDM);
    // s0*cos(DM).
    auto *s0_cDM = llvm_fmul(s, s0, cDM);
    // c0*sin(DM).
    auto *c0_sDM = llvm_fmul(s, c0, sDM);
    // s0*sin(DM).
    auto *s0_sDM = llvm_fmul(s, s0, sDM);
    // 1/2 constant.
    auto *c_1_2 = llvm_codegen(s, tp, number_like(s, fp_t, 1. / 2));
    // c0 * cosDM - s0 * sinDM.
    auto *c0cDM_s0sDM = llvm_fsub(s, c0_cDM, s0_sDM);
    // c0 * sinDM + s0 * cosDM.
    auto *c0sDM_s0cDM = llvm_fadd(s, c0_sDM, s0_cDM);
    // c0 * sinDM + s0 * cosDM - s0.
    auto *c0sDM_s0cDM_s0 = llvm_fsub(s, c0sDM_s0cDM, s0);

    // Put it together.
    auto *ig1 = llvm_fadd(s, DM, c0sDM_s0cDM_s0);
    auto *ig2 = llvm_fmul(s, c0cDM_s0sDM, c0sDM_s0cDM_s0);

    auto *tmp1 = llvm_fmul(s, c_1_2, c0sDM_s0cDM_s0);
    auto *tmp2 = llvm_square(s, c0cDM_s0sDM);
    auto *tmp3 = llvm_fadd(s, tmp2, tmp2);
    auto *tmp4 = llvm_fmul(s, c0sDM_s0cDM_s0, c0sDM_s0cDM);
    auto *tmp5 = llvm_fsub(s, tmp3, tmp4);

    auto *ig3 = llvm_fmul(s, tmp1, tmp5);

    auto *ig = llvm_fadd(s, llvm_fadd(s, ig1, ig2), ig3);

    // Compute the initial bounding range - that is, the bounds for DE which
    // are guaranteed to contain the root. These are [-1, 2pi + 1).
    const auto twopi_num = inv_kep_E_dl_twopi_like(s, fp_t).first;
    auto *lb_init = llvm_constantfp(s, tp, -1.);
    auto *ub_init = llvm_codegen(s, tp, nextafter(twopi_num + number_like(s, fp_t, 1.), number_like(s, fp_t, 0.)));

    // Store them.
    auto *lb_storage = builder.CreateAlloca(tp);
    auto *ub_storage = builder.CreateAlloca(tp);
    builder.CreateStore(lb_init, lb_storage);
    builder.CreateStore(ub_init, ub_storage);

    // Clamp the initial guess to the initial bounding range.
    // NOTE: in case ig ends up being NaN (because the arguments are NaN or for whatever
    // other reason), then ig will remain NaN.
    ig = llvm_clamp(s, ig, lb_init, ub_init);

    // Store the initial guess in the storage for the return value. This will hold the
    // iteratively-determined value for DE.
    auto *retval = builder.CreateAlloca(tp);
    builder.CreateStore(ig, retval);

    // Create the counter.
    auto *counter = builder.CreateAlloca(builder.getInt32Ty());
    builder.CreateStore(builder.getInt32(0), counter);

    // Variables to store sin(DE) and cos(DE).
    auto *sin_DE = builder.CreateAlloca(tp);
    auto *cos_DE = builder.CreateAlloca(tp);

    // Write the initial values for sin_DE and cos_DE.
    auto sin_cos_DE = llvm_sincos(s, builder.CreateLoad(tp, retval));
    builder.CreateStore(sin_cos_DE.first, sin_DE);
    builder.CreateStore(sin_cos_DE.second, cos_DE);

    // Helper to compute f(DE).
    auto fDE_compute = [&]() {
        // s0*(1 - cos(DE)).
        auto *one_c = llvm_constantfp(s, tp, 1.);
        auto *one_cDE = llvm_fsub(s, one_c, builder.CreateLoad(tp, cos_DE));
        auto *s0_one_cDE = llvm_fmul(s, s0, one_cDE);
        // c0*sin(DE).
        auto *c0_sDE = llvm_fmul(s, c0, builder.CreateLoad(tp, sin_DE));
        // DE - DM.
        auto *DE_DM = llvm_fsub(s, builder.CreateLoad(tp, retval), DM);
        // DE - DM + s0*(1 - cos(DE)) - c0*sin(DE).
        return llvm_fsub(s, llvm_fadd(s, DE_DM, s0_one_cDE), c0_sDE);
    };
    // Variable to hold the value of f(DE).
    auto *fDE = builder.CreateAlloca(tp);
    // Compute and store the initial value of f(DE).
    builder.CreateStore(fDE_compute(), fDE);

    // Create a variable to hold the result of the tolerance check
    // computed at the beginning of each iteration of the NR loop.
    // This is "true" if the loop needs to continue, or "false" if the
    // loop can stop because we achieved the desired tolerance.
    // NOTE: this is only allocated, it will be immediately written to
    // the first time loop_cond is evaluated.
    // NOTE: this needs to persist outside loop_cond because we will use it
    // after the loop in case we exceeded the max iter number.
    auto *vec_bool_t = make_vector_type(builder.getInt1Ty(), batch_size);
    auto *tol_check_ptr = builder.CreateAlloca(vec_bool_t);

    // Define the continuing condition functor. This will return "true"
    // if the NR loop needs to continue, "false" if it can be stopped
    // (either because a solution has been found, or because the max
    // number of iterations has been exceeded).
    // NOTE: hard-code max_iter for the time being. It would probably
    // make sense to make it dependent on the epsilon of fp_t though?
    auto *max_iter = builder.getInt32(50);
    auto loop_cond = [&]() -> llvm::Value * {
        // NOTE: we use an *absolute* tolerance of 4*eps for both the check
        // on the magnitude of f(DE) and on the magnitude of the bounding
        // range. This of course means that the final result is not guaranteed
        // to be the best possible approximation of DE, for at least the
        // following reasons:
        // - abs(f(DE)) < tol is a poor criterion when the derivative of
        //   f(DE) is small, because large variations in DE result in small
        //   variations of f(DE);
        // - the check on the bounding range should not use an absolute
        //   tolerance but a relative one (i.e., rescaled wrt the extrema
        //   of the bounds). This is what Boost's bisection implementation does,
        //   but I am not sure I understand if/how this works in case of a root
        //   equal to or very close to zero.
        auto *tol = llvm_codegen(s, tp, inv_kep_E_eps_like(s, fp_t) * number_like(s, fp_t, 4.));

        // Load the current values of f(DE) and DE.
        auto *cur_fDE = builder.CreateLoad(tp, fDE);
        auto *cur_DE = builder.CreateLoad(tp, retval);

        // Compute the sign of f(DE).
        // NOTE: this will be zero if f(DE) is NaN.
        auto *fDE_sgn = llvm_sgn(s, cur_fDE);

        // Update the bounds.
        // NOTE: if f(DE) is NaN, then new_ub/new_lb will also be set to NaN.
        auto *cur_ub = builder.CreateLoad(tp, ub_storage);
        auto *cur_lb = builder.CreateLoad(tp, lb_storage);
        auto *zero_c = vector_splat(builder, builder.getInt32(0), batch_size);
        auto *new_ub = builder.CreateSelect(builder.CreateICmpSGE(fDE_sgn, zero_c), cur_DE, cur_ub);
        auto *new_lb = builder.CreateSelect(builder.CreateICmpSLE(fDE_sgn, zero_c), cur_DE, cur_lb);
        builder.CreateStore(new_ub, ub_storage);
        builder.CreateStore(new_lb, lb_storage);

        // Compute the size of the new bounding range.
        auto *bsize = llvm_fsub(s, new_ub, new_lb);

        // First tolerance check: abs(f(DE)) > tol.
        // NOTE: if DE is NaN, then f(DE) is NaN and tol1_check is false.
        auto *tol1_check = llvm_fcmp_ogt(s, llvm_abs(s, cur_fDE), tol);

        // Second tolerance check: (ub - lb) > tol.
        // NOTE: if DE is NaN, then tol2_check is false.
        auto *tol2_check = llvm_fcmp_ogt(s, bsize, tol);

        // Put them together with a logical AND.
        auto *tol_check
            = builder.CreateSelect(tol1_check, tol2_check, llvm::ConstantInt::get(tol2_check->getType(), 0u));
        // NOTE: we need OR reduction in batch mode: continue if *any* element of the batch
        // needs more iterations.
        auto *tol_cond = (batch_size == 1u) ? tol_check : builder.CreateOrReduce(tol_check);

        // Store the result of the tolerance check.
        builder.CreateStore(tol_check, tol_check_ptr);

        // Check the number of iterations.
        auto *c_cond = builder.CreateICmpULT(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter);

        // Combine tolerance check and number of iterations check with a logical AND.
        return builder.CreateSelect(c_cond, tol_cond, llvm::ConstantInt::get(tol_cond->getType(), 0u));
    };

    // Run the loop.
    llvm_while_loop(s, loop_cond, [&]() {
        // Compute the new value via the Newton-Raphson formula.
        auto *old_val = builder.CreateLoad(tp, retval);
        auto *one_c = llvm_constantfp(s, tp, 1.);
        auto *diff = llvm_fadd(s, one_c, llvm_fmul(s, s0, builder.CreateLoad(tp, sin_DE)));
        diff = llvm_fsub(s, diff, llvm_fmul(s, c0, builder.CreateLoad(tp, cos_DE)));
        auto *fdiv = llvm_fdiv(s, builder.CreateLoad(tp, fDE), diff);
        auto *new_val = llvm_fsub(s, old_val, fdiv);

        // Bisect if new_val > cur_ub.
        auto *half_c = llvm_codegen(s, tp, number_like(s, fp_t, 1. / 2));
        auto *cur_ub = builder.CreateLoad(tp, ub_storage);
        auto *bcheck = llvm_fcmp_ogt(s, new_val, cur_ub);
        new_val = builder.CreateSelect(bcheck, llvm_fmul(s, half_c, llvm_fadd(s, old_val, cur_ub)), new_val);

        // Bisect if new_val < cur_lb.
        auto *cur_lb = builder.CreateLoad(tp, lb_storage);
        bcheck = llvm_fcmp_olt(s, new_val, cur_lb);
        new_val = builder.CreateSelect(bcheck, llvm_fmul(s, half_c, llvm_fadd(s, old_val, cur_lb)), new_val);

        // Store the new value.
        builder.CreateStore(new_val, retval);

        // Update sin_DE/cos_DE.
        sin_cos_DE = llvm_sincos(s, new_val);
        builder.CreateStore(sin_cos_DE.first, sin_DE);
        builder.CreateStore(sin_cos_DE.second, cos_DE);

        // Update f(DE).
        builder.CreateStore(fDE_compute(), fDE);

        // Update the counter.
        builder.CreateStore(builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), counter), builder.getInt32(1)),
                            counter);
    });

    // After exiting the NR loop, check the counter.
    llvm_if_then_else(
        s, builder.CreateICmpEQ(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter),
        [&]() {
            // Load the tol_check variable.
            auto *tol_check = builder.CreateLoad(vec_bool_t, tol_check_ptr);

            // Set to quiet NaN in the return value all the lanes for which tol_check is 1.
            auto *old_val = builder.CreateLoad(tp, retval);
            auto *new_val = builder.CreateSelect(
                tol_check, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), old_val);
            builder.CreateStore(new_val, retval);

            llvm_invoke_external(s, "heyoka_inv_kep_DE_max_iter", builder.getVoidTy(), {},
                                 llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                                          {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn}));
        },
        []() {});

    // Load the result.
    llvm::Value *ret = builder.CreateLoad(tp, retval);

    // Codegen 2pi, used below.
    auto *twopi_const = llvm_codegen(s, tp, twopi_num);

    // Reduce the result to the standard trigonometric range [0, 2pi).
    // NOTE: this reduction will not change ret if it is NaN.
    // Is ret < 0?
    auto *ret_lt_0 = llvm_fcmp_olt(s, ret, llvm_constantfp(s, tp, 0.));
    ret = builder.CreateSelect(ret_lt_0, llvm_fadd(s, twopi_const, ret), ret);

    // Is ret >= 2pi?
    auto *ret_ge_2pi = llvm_fcmp_oge(s, ret, twopi_const);
    ret = builder.CreateSelect(ret_ge_2pi, llvm_fsub(s, ret, twopi_const), ret);

    // Return the result.
    builder.CreateRet(ret);

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE

// LCOV_EXCL_START

// NOTE: these functions will be called when numerical root finding exceeds the maximum number
// of allowed iterations.
extern "C" {

HEYOKA_DLL_PUBLIC void heyoka_inv_kep_E_max_iter() noexcept
{
    heyoka::detail::get_logger()->warn("iteration limit exceeded while solving the elliptic inverse Kepler equation");
}

HEYOKA_DLL_PUBLIC void heyoka_inv_kep_F_max_iter() noexcept
{
    heyoka::detail::get_logger()->warn(
        "iteration limit exceeded while solving the inverse Kepler equation for the eccentric longitude F");
}

HEYOKA_DLL_PUBLIC void heyoka_inv_kep_DE_max_iter() noexcept
{
    heyoka::detail::get_logger()->warn("iteration limit exceeded while solving the inverse Kepler equation for DE");
}
}

// LCOV_EXCL_STOP
