// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128) || defined(HEYOKA_HAVE_REAL)

#include <mp++/integer.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/log.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

pow_impl::pow_impl(expression b, expression e) : func_base("pow", std::vector{std::move(b), std::move(e)}) {}

pow_impl::pow_impl() : pow_impl(1_dbl, 1_dbl) {}

void pow_impl::to_stream(std::ostringstream &oss) const
{
    assert(args().size() == 2u);

    const auto &base = args()[0];
    const auto &expo = args()[1];

    // NOTE: following Python's parsing rules, there are 2
    // situations in which we need to put brackets around the base:
    //
    // - the base is a negation: this is because the power operator **
    //   binds more tightly than unary operators on its left.
    //   Thus, if the base begins with a '-' sign, we cannot
    //   print here -x**y, but we must print instead (-x)**y;
    // - the base itself is a pow(): this is because in Python
    //   the power operator associates from right to left. Thus,
    //   pow(pow(x, y), z) cannot be written as x**y**z as that
    //   would be parsed as x**(y**z), and we need to write
    //   (x**y)**z instead.

    const auto base_is_negation = is_negation_prod(base);
    const auto base_is_pow = [&]() {
        const auto *fptr = std::get_if<func>(&base.value());

        return fptr != nullptr && fptr->extract<pow_impl>() != nullptr;
    }();

    if (base_is_negation || base_is_pow) {
        oss << '(';
    }

    stream_expression(oss, base);

    if (base_is_negation || base_is_pow) {
        oss << ')';
    }

    oss << "**";
    stream_expression(oss, expo);
}

namespace
{

// Exponentiation by squaring.
// NOLINTNEXTLINE(misc-no-recursion)
llvm::Value *pow_ebs(llvm_state &s, llvm::Value *base, std::uint32_t exp)
{
    if (exp == 0u) {
        return llvm_codegen(s, base->getType(), number{1.});
    }

    if (exp == 1u) {
        return base;
    }

    if (exp % 2u == 0u) {
        return pow_ebs(s, llvm_square(s, base), exp / 2u);
    } else {
        auto *tmp = pow_ebs(s, llvm_square(s, base), (exp - 1u) / 2u);
        return llvm_fmul(s, base, tmp);
    }
}

using safe_int64_t = boost::safe_numerics::safe<std::int64_t>;

// Check if ex is an integral number. If it is, its
// value will be returned.
std::optional<safe_int64_t> ex_is_integral(const expression &ex)
{
    return std::visit(
        [](const auto &v) -> std::optional<safe_int64_t> {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, number>) {
                return std::visit(
                    [](const auto &x) -> std::optional<safe_int64_t> {
                        using num_type = uncvref_t<decltype(x)>;

                        using std::trunc;
                        using std::isfinite;

                        if (!isfinite(x) || x != trunc(x)) {
                            // Non-finite or non-integral.
                            return {};
                        }

                        // NOTE: in these conversions we are ok with throwing
                        // if the integer is too large.
                        if constexpr (std::is_floating_point_v<num_type>) {
                            return boost::numeric_cast<std::int64_t>(x);
                        }

#if defined(HEYOKA_HAVE_REAL128)
                        else if constexpr (std::is_same_v<num_type, mppp::real128>) {
                            // NOTE: for mppp:: types, convert first into mppp::integer
                            // and then do a (checked) cast to std::int64_t.
                            return static_cast<std::int64_t>(static_cast<mppp::integer<1>>(x));
                        }
#endif

#if defined(HEYOKA_HAVE_REAL)
                        else if constexpr (std::is_same_v<num_type, mppp::real>) {
                            return static_cast<std::int64_t>(static_cast<mppp::integer<1>>(x));
                        }
#endif

                        // LCOV_EXCL_START
                        else {
                            static_assert(always_false_v<num_type>);
                            throw;
                        }
                        // LCOV_EXCL_STOP
                    },
                    v.value());
            } else {
                // Not a number.
                return {};
            }
        },
        ex.value());
}

// Check if ex is a number in the form n / 2,
// where n is an odd integral value. If it is, n
// will be returned.
std::optional<safe_int64_t> ex_is_odd_integral_half(const expression &ex)
{
    return std::visit(
        [](const auto &v) -> std::optional<safe_int64_t> {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, number>) {
                return std::visit(
                    [](const auto &x) -> std::optional<safe_int64_t> {
                        using num_type = uncvref_t<decltype(x)>;

                        using std::trunc;
                        using std::isfinite;

                        if (!isfinite(x) || x == trunc(x)) {
                            // x is not finite, or it is
                            // an integral value.
                            return {};
                        }

                        // NOTE: here we will be assuming that, for all supported
                        // float types, multiplication by 2 is exact.
                        // Since we are assuming IEEE binary floats anyway, we should be
                        // safe here.
                        const auto y = 2 * x;

                        // NOTE: y should never become infinity here for builtin FP types,
                        // because this would mean that x is integral (since large float
                        // values are all integrals anyway).
                        // The only potential exception is mppp::real, where I am not 100%
                        // sure what happens wrt the minimum/maximum exponent API in MPFR.
                        // Thus, in order to be sure, let's check the finiteness of y.
                        if (!isfinite(y)) {
                            // LCOV_EXCL_START
                            throw std::overflow_error("Overflow detected in ex_is_odd_integral_half()");
                            // LCOV_EXCL_STOP
                        }

                        if (y != trunc(y)) {
                            // x is not n/2.
                            return {};
                        }

                        // NOTE: in these conversions we are ok with throwing
                        // if the integer is too large.
                        if constexpr (std::is_floating_point_v<num_type>) {
                            return boost::numeric_cast<std::int64_t>(y);
                        }

#if defined(HEYOKA_HAVE_REAL128)
                        else if constexpr (std::is_same_v<num_type, mppp::real128>) {
                            return static_cast<std::int64_t>(static_cast<mppp::integer<1>>(y));
                        }
#endif

#if defined(HEYOKA_HAVE_REAL)
                        else if constexpr (std::is_same_v<num_type, mppp::real>) {
                            return static_cast<std::int64_t>(static_cast<mppp::integer<1>>(y));
                        }
#endif

                        // LCOV_EXCL_START
                        else {
                            static_assert(always_false_v<num_type>);
                            throw;
                        }
                        // LCOV_EXCL_STOP
                    },
                    v.value());
            } else {
                // Not a number.
                return {};
            }
        },
        ex.value());
}

} // namespace

// Construct a pow_eval_algo based on the exponentiation arguments of 'impl'.
pow_eval_algo get_pow_eval_algo(const pow_impl &impl)
{
    // Maximum integral exponent magnitude for which
    // pow() is transformed into multiplications and divisions.
    constexpr std::uint32_t pow_max_small_pow_n = 16;

    assert(impl.args().size() == 2u);

    // NOTE: check the special cases first, otherwise fall through
    // to the general case.

    // Small integral powers.
    if (const auto exp = ex_is_integral(impl.args()[1])) {
        if (*exp >= 0 && *exp <= pow_max_small_pow_n) {
            return {pow_eval_algo::type::pos_small_int,
                    [e = *exp](auto &s, const auto &args) { return pow_ebs(s, args[0], e); }, exp,
                    fmt::format("_pos_small_int_{}", static_cast<std::int64_t>(*exp))};
        }

        if (*exp < 0 && -*exp <= pow_max_small_pow_n) {
            return {pow_eval_algo::type::neg_small_int,
                    [e = *exp](auto &s, const auto &args) {
                        auto *tmp = pow_ebs(s, args[0], -e);
                        return llvm_fdiv(s, llvm_codegen(s, tmp->getType(), number{1.}), tmp);
                    },
                    exp, fmt::format("_neg_small_int_{}", static_cast<std::int64_t>(-*exp))};
        }
    }

    // Small half-integral powers.
    if (const auto exp2 = ex_is_odd_integral_half(impl.args()[1])) {
        if (*exp2 >= 0 && *exp2 <= pow_max_small_pow_n) {
            return {pow_eval_algo::type::pos_small_half,
                    [e2 = *exp2](auto &s, const auto &args) {
                        auto *tmp = llvm_sqrt(s, args[0]);
                        return pow_ebs(s, tmp, e2);
                    },
                    exp2, fmt::format("_pos_small_half_{}", static_cast<std::int64_t>(*exp2))};
        }

        if (*exp2 < 0 && -*exp2 <= pow_max_small_pow_n) {
            return {pow_eval_algo::type::neg_small_half,
                    [e2 = *exp2](auto &s, const auto &args) {
                        auto *tmp = llvm_sqrt(s, args[0]);
                        tmp = pow_ebs(s, tmp, -e2);
                        return llvm_fdiv(s, llvm_codegen(s, tmp->getType(), number{1.}), tmp);
                    },
                    exp2, fmt::format("_neg_small_half_{}", static_cast<std::int64_t>(-*exp2))};
        }
    }

    // The general case.
    return {
        pow_eval_algo::type::general, [](auto &s, const auto &args) { return llvm_pow(s, args[0], args[1]); }, {}, {}};
}

llvm::Value *pow_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    // Fetch the pow eval algo.
    const auto pea = get_pow_eval_algo(*this);

    return llvm_eval_helper([&](const std::vector<llvm::Value *> &args, bool) { return pea.eval_f(s, args); }, *this, s,
                            fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *pow_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    // Fetch the pow eval algo.
    const auto pea = get_pow_eval_algo(*this);

    // Build the function name.
    const std::string func_name = "pow" + pea.suffix;

    return llvm_c_eval_func_helper(
        func_name, [&](const std::vector<llvm::Value *> &args, bool) { return pea.eval_f(s, args); }, *this, s, fp_t,
        batch_size, high_accuracy);
}

namespace
{

// Derivative of pow(number, number).
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, llvm::Type *fp_t, const pow_impl &f, const U &num0, const V &num1,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (order == 0u) {
        // Fetch the pow eval algo.
        const auto pea = get_pow_eval_algo(f);

        return pea.eval_f(s, {taylor_codegen_numparam(s, fp_t, num0, par_ptr, batch_size),
                              taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size)});
    } else {
        return vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of square(variable).
llvm::Value *taylor_diff_square_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                     std::uint32_t n_uvars, std::uint32_t order)
{
    assert(order > 0u);

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // Compute the sum.
    std::vector<llvm::Value *> sum;
    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 0; j <= (order - 1u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(llvm_fmul(s, v0, v1));
        }

        auto *ret = pairwise_sum(s, sum);
        return llvm_fadd(s, ret, ret);
    } else {
        // Even order.
        auto *ak2 = taylor_fetch_diff(arr, u_idx, order / 2u, n_uvars);
        auto *sq_ak2 = llvm_fmul(s, ak2, ak2);

        for (std::uint32_t j = 0; j <= (order - 2u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(llvm_fmul(s, v0, v1));
        }

        auto *ret = pairwise_sum(s, sum);
        return llvm_fadd(s, llvm_fadd(s, ret, ret), sq_ak2);
    }
}

// Derivative of sqrt(variable).
// NOTE: this is derived by taking:
// a = sqrt(b) -> a**2 = b -> (a**2)^[n] = b^[n]
// and then using the squaring formula.
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx)
{
    assert(order > 0u);

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // Compute the divisor: 2*a^[0].
    auto *div = taylor_fetch_diff(arr, idx, 0, n_uvars);
    div = llvm_fadd(s, div, div);

    // Init the factor: b^[n].
    auto *fac = taylor_fetch_diff(arr, u_idx, order, n_uvars);

    std::vector<llvm::Value *> sum;
    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 1; j <= (order - 1u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

            sum.push_back(llvm_fmul(s, v0, v1));
        }
    } else {
        // Even order.
        for (std::uint32_t j = 1; j <= (order - 2u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

            sum.push_back(llvm_fmul(s, v0, v1));
        }

        auto *tmp = taylor_fetch_diff(arr, idx, order / 2u, n_uvars);
        tmp = llvm_fmul(s, tmp, tmp);

        fac = llvm_fsub(s, fac, tmp);
    }

    // Avoid summing if the sum is empty.
    if (!sum.empty()) {
        auto *tmp = pairwise_sum(s, sum);
        tmp = llvm_fadd(s, tmp, tmp);

        fac = llvm_fsub(s, fac, tmp);
    }

    return llvm_fdiv(s, fac, div);
}

// Derivative of pow(variable, number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, llvm::Type *fp_t, const pow_impl &f, const variable &var, const U &num,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // Fetch the pow eval algo.
    const auto pea = get_pow_eval_algo(f);

    if (order == 0u) {
        return pea.eval_f(
            s, {taylor_fetch_diff(arr, u_idx, 0, n_uvars), taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size)});
    }

    // Special case for sqrt().
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (pea.algo == pow_eval_algo::type::pos_small_half && *pea.exp == 1) {
        return taylor_diff_sqrt_impl(s, var, arr, n_uvars, order, idx);
    }

    // Special case for square().
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (pea.algo == pow_eval_algo::type::pos_small_int && *pea.exp == 2) {
        return taylor_diff_square_impl(s, var, arr, n_uvars, order);
    }

    // The general case.
    auto &builder = s.builder();

    // NOTE: iteration in the [0, order) range
    // (i.e., order *not* included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        auto *v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
        auto *v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

        // Compute the scalar factor: order * num - j * (num + 1).
        auto scal_f = [&]() -> llvm::Value * {
            if constexpr (std::is_same_v<U, number>) {
                return vector_splat(
                    builder,
                    llvm_codegen(s, fp_t,
                                 number_like(s, fp_t, static_cast<double>(order)) * num
                                     - number_like(s, fp_t, static_cast<double>(j)) * (num + number_like(s, fp_t, 1.))),
                    batch_size);
            } else {
                auto pc = taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size);
                auto *jvec = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);
                auto *ordvec
                    = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(order))), batch_size);
                auto *onevec = vector_splat(builder, llvm_codegen(s, fp_t, number(1.)), batch_size);

                auto tmp1 = llvm_fmul(s, ordvec, pc);
                auto tmp2 = llvm_fmul(s, jvec, llvm_fadd(s, pc, onevec));

                return llvm_fsub(s, tmp1, tmp2);
            }
        }();

        // Add scal_f*v0*v1 to the sum.
        sum.push_back(llvm_fmul(s, scal_f, llvm_fmul(s, v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto *ret_acc = pairwise_sum(s, sum);

    // Compute the final divisor: order * (zero-th derivative of u_idx).
    auto *ord_f = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(order))), batch_size);
    auto *b0 = taylor_fetch_diff(arr, u_idx, 0, n_uvars);
    auto *div = llvm_fmul(s, ord_f, b0);

    // Compute and return the result: ret_acc / div.
    return llvm_fdiv(s, ret_acc, div);
}

// All the other cases.
template <typename U1, typename U2, std::enable_if_t<!std::conjunction_v<is_num_param<U1>, is_num_param<U2>>, int> = 0>
llvm::Value *taylor_diff_pow_impl(llvm_state &, llvm::Type *, const pow_impl &, const U1 &, const U2 &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a pow()");
}

llvm::Value *taylor_diff_pow(llvm_state &s, llvm::Type *fp_t, const pow_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 2u);

    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the exponentiation, but a vector of size {} was passed "
                        "instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_pow_impl(s, fp_t, f, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value(), f.args()[1].value());
}

} // namespace

llvm::Value *pow_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool) const
{
    return taylor_diff_pow(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of pow(number, number).
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &s, llvm::Type *fp_t, const pow_impl &fn, const U &n0,
                                            const V &n1, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    // Fetch the pow eval algo.
    const auto pea = get_pow_eval_algo(fn);

    // Create the function name.
    const std::string func_name = "pow" + pea.suffix;

    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, func_name, 0, [&](const auto &args) { return pea.eval_f(s, args); }, n0, n1);
}

// Derivative of square(variable).
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &s, llvm::Type *fp_t, const variable &var,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair
        = taylor_c_diff_func_name_args(context, fp_t, "pow_square", n_uvars, batch_size,
                                       {var,
                                        // NOTE: as usual, here only the type is important, not the value.
                                        number{0.}});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *diff_ptr = f->args().begin() + 2;
        auto *var_idx = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    llvm_square(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Distinguish the odd/even cases for the order.
                llvm_if_then_else(
                    s, builder.CreateICmpEQ(builder.CreateURem(ord, builder.getInt32(2)), builder.getInt32(1)),
                    [&]() {
                        // Odd order.
                        auto *loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(1)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto *a_nj
                                = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                            auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx);

                            builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, a_nj, aj)),
                                                acc);
                        });

                        // Return 2 * acc.
                        auto *acc_load = builder.CreateLoad(val_t, acc);
                        builder.CreateStore(llvm_fadd(s, acc_load, acc_load), retval);
                    },
                    [&]() {
                        // Even order.

                        // Pre-compute the final term.
                        auto *ak2 = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars,
                                                       builder.CreateUDiv(ord, builder.getInt32(2)), var_idx);
                        auto *sq_ak2 = llvm_fmul(s, ak2, ak2);

                        auto *loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(2)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto *a_nj
                                = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                            auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx);

                            builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, a_nj, aj)),
                                                acc);
                        });

                        // Return 2 * acc + ak2 * ak2.
                        auto *acc_load = builder.CreateLoad(val_t, acc);
                        builder.CreateStore(llvm_fadd(s, llvm_fadd(s, acc_load, acc_load), sq_ak2), retval);
                    });
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Derivative of sqrt(variable).
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &s, llvm::Type *fp_t, const variable &var,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair
        = taylor_c_diff_func_name_args(context, fp_t, "pow_sqrt", n_uvars, batch_size,
                                       {var,
                                        // NOTE: as usual, here only the type is important, not the value.
                                        number{.5}});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *u_idx = f->args().begin() + 1;
        auto *diff_ptr = f->args().begin() + 2;
        auto *var_idx = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    llvm_sqrt(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            },
            [&]() {
                // Compute the divisor: 2*a^[0].
                auto *div = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), u_idx);
                div = llvm_fadd(s, div, div);

                // retval = b^[n].
                builder.CreateStore(taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, var_idx), retval);

                // Determine the upper index of the summation: (ord - 1)/2 if ord is odd, (ord - 2)/2 otherwise.
                auto *ord_even
                    = builder.CreateICmpEQ(builder.CreateURem(ord, builder.getInt32(2)), builder.getInt32(0));
                auto *upper = builder.CreateUDiv(
                    builder.CreateSub(ord, builder.CreateSelect(ord_even, builder.getInt32(2), builder.getInt32(1))),
                    builder.getInt32(2));

                // Perform the summation.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);
                llvm_loop_u32(
                    s, builder.getInt32(1), builder.CreateAdd(upper, builder.getInt32(1)), [&](llvm::Value *j) {
                        auto *a_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
                        auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, u_idx);

                        builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, a_nj, aj)), acc);
                    });
                builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), builder.CreateLoad(val_t, acc)), acc);

                llvm_if_then_else(
                    s, ord_even,
                    [&]() {
                        // retval -= (a^[n/2])**2.
                        auto *tmp = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars,
                                                       builder.CreateUDiv(ord, builder.getInt32(2)), u_idx);
                        tmp = llvm_fmul(s, tmp, tmp);

                        builder.CreateStore(llvm_fsub(s, builder.CreateLoad(val_t, retval), tmp), retval);
                    },
                    []() {});

                // retval -= acc.
                builder.CreateStore(llvm_fsub(s, builder.CreateLoad(val_t, retval), builder.CreateLoad(val_t, acc)),
                                    retval);

                // retval /= div.
                builder.CreateStore(llvm_fdiv(s, builder.CreateLoad(val_t, retval), div), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Derivative of pow(variable, number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &s, llvm::Type *fp_t, const pow_impl &fn, const variable &var,
                                            const U &n, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    // Fetch the pow eval algo.
    const auto pea = get_pow_eval_algo(fn);

    // Special case for sqrt().
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (pea.algo == pow_eval_algo::type::pos_small_half && *pea.exp == 1) {
        return taylor_c_diff_func_sqrt_impl(s, fp_t, var, n_uvars, batch_size);
    }

    // Special case for square().
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (pea.algo == pow_eval_algo::type::pos_small_int && *pea.exp == 2) {
        return taylor_c_diff_func_square_impl(s, fp_t, var, n_uvars, batch_size);
    }

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Create the function name.
    const std::string pow_name = "pow" + pea.suffix;

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, pow_name, n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 5;
        auto exponent = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                auto *pow_base = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx);
                auto *pow_exp = taylor_c_diff_numparam_codegen(s, fp_t, n, exponent, par_ptr, batch_size);

                builder.CreateStore(pea.eval_f(s, {pow_base, pow_exp}), retval);
            },
            [&]() {
                // Create FP vector versions of exponent and order.
                auto alpha_v = taylor_c_diff_numparam_codegen(s, fp_t, n, exponent, par_ptr, batch_size);
                auto ord_v = vector_splat(builder, llvm_ui_to_fp(s, ord, fp_t), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
                    auto b_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                    auto aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, u_idx);

                    // Compute the factor n*alpha-j*(alpha+1).
                    auto *j_v = vector_splat(builder, llvm_ui_to_fp(s, j, fp_t), batch_size);
                    auto fac = llvm_fsub(
                        s, llvm_fmul(s, ord_v, alpha_v),
                        llvm_fmul(s, j_v,
                                  llvm_fadd(s, alpha_v,
                                            vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size))));

                    builder.CreateStore(
                        llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, fac, llvm_fmul(s, b_nj, aj))), acc);
                });

                // Finalize the result: acc / (n*b0).
                builder.CreateStore(
                    llvm_fdiv(s, builder.CreateLoad(val_t, acc),
                              llvm_fmul(s, ord_v,
                                        taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx))),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// All the other cases.
template <typename U1, typename U2, std::enable_if_t<!std::conjunction_v<is_num_param<U1>, is_num_param<U2>>, int> = 0>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &, llvm::Type *, const pow_impl &, const U1 &, const U2 &,
                                            std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a pow() in compact mode");
}

llvm::Function *taylor_c_diff_func_pow(llvm_state &s, llvm::Type *fp_t, const pow_impl &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 2u);

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_c_diff_func_pow_impl(s, fp_t, fn, v1, v2, n_uvars, batch_size);
        },
        fn.args()[0].value(), fn.args()[1].value());
}

} // namespace

llvm::Function *pow_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_pow(s, fp_t, *this, n_uvars, batch_size);
}

std::vector<expression> pow_impl::gradient() const
{
    assert(args().size() == 2u);
    return {args()[1] * pow(args()[0], args()[1] - 1_dbl), pow(args()[0], args()[1]) * log(args()[0])};
}

[[nodiscard]] expression pow_impl::normalise() const
{
    assert(args().size() == 2u);
    return pow(args()[0], args()[1]);
}

namespace
{

// Type traits machinery to detect if two types can be used as base and exponent
// in an exponentiation via the pow() function.
namespace pow_detail
{

using std::pow;

template <typename T, typename U>
using pow_t = decltype(pow(std::declval<T>(), std::declval<U>()));

} // namespace pow_detail

template <typename T, typename U>
using is_exponentiable = is_detected<pow_detail::pow_t, T, U>;

// Wrapper for the implementation of the top-level pow() function.
// NOLINTNEXTLINE(misc-no-recursion)
expression pow_wrapper_impl(expression b, expression e)
{
    // Attempt constant folding first.
    if (const auto *b_num_ptr = std::get_if<number>(&b.value()), *e_num_ptr = std::get_if<number>(&e.value());
        (b_num_ptr != nullptr) && (e_num_ptr != nullptr)) {
        return std::visit(
            [](const auto &x, const auto &y) -> expression {
                if constexpr (detail::is_exponentiable<decltype(x), decltype(y)>::value) {
                    using std::pow;

                    return expression{pow(x, y)};
                } else {
                    // LCOV_EXCL_START
                    throw std::invalid_argument(
                        fmt::format("Cannot raise a base of type '{}' to an exponent of type '{}'",
                                    boost::core::demangle(typeid(x).name()), boost::core::demangle(typeid(y).name())));
                    // LCOV_EXCL_STOP
                }
            },
            b_num_ptr->value(), e_num_ptr->value());
    }

    // Handle special cases for a numerical exponent.
    if (const auto *num_ptr = std::get_if<number>(&e.value())) {
        if (is_zero(*num_ptr)) {
            return 1_dbl;
        }

        if (is_one(*num_ptr)) {
            return b;
        }

        // Handle special cases when the base is a pow().
        if (const auto *fptr = std::get_if<func>(&b.value()); fptr != nullptr && fptr->extract<pow_impl>() != nullptr) {
            assert(fptr->args().size() == 2u);

            const auto &b_base = fptr->args()[0];
            const auto &b_exp = fptr->args()[1];

            if (const auto *b_exp_num_ptr = std::get_if<number>(&b_exp.value())) {
                // b's exponent is a number, fold it together with e.
                return pow(b_base, expression{*b_exp_num_ptr * *num_ptr});
            } else {
                // b's exponent is not a number, multiply it by e.
                return pow(b_base, prod({b_exp, e}));
            }
        }

        // Handle special cases when the base is a prod() and the exponent is an integral value:
        // (x*y)**n -> x**n * y**n.
        if (const auto *fptr = std::get_if<func>(&b.value());
            fptr != nullptr && fptr->extract<prod_impl>() != nullptr && is_integer(*num_ptr)) {
            std::vector<expression> new_prod_args;
            new_prod_args.reserve(fptr->args().size());
            for (const auto &arg : fptr->args()) {
                new_prod_args.push_back(pow(arg, expression{*num_ptr}));
            }

            return prod(new_prod_args);
        }
    }

    // The general case.
    return expression{func{pow_impl{std::move(b), std::move(e)}}};
}

} // namespace

} // namespace detail

// NOLINTNEXTLINE(misc-no-recursion)
expression pow(expression b, expression e)
{
    return detail::pow_wrapper_impl(std::move(b), std::move(e));
}

expression pow(expression b, float e)
{
    return pow(std::move(b), expression{e});
}

expression pow(expression b, double e)
{
    return pow(std::move(b), expression{e});
}

expression pow(expression b, long double e)
{
    return pow(std::move(b), expression{e});
}

#if defined(HEYOKA_HAVE_REAL128)

expression pow(expression b, mppp::real128 e)
{
    return pow(std::move(b), expression{e});
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression pow(expression b, mppp::real e)
{
    return pow(std::move(b), expression{std::move(e)});
}

#endif

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::pow_impl)
