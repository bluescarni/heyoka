// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

atan2_impl::atan2_impl(expression y, expression x) : func_base("atan2", std::vector{std::move(y), std::move(x)}) {}

atan2_impl::atan2_impl() : atan2_impl(0_dbl, 1_dbl) {}

expression atan2_impl::diff(const std::string &s) const
{
    assert(args().size() == 2u);

    const auto &y = args()[0];
    const auto &x = args()[1];

    auto den = square(x) + square(y);

    return (x * heyoka::diff(y, s) - y * heyoka::diff(x, s)) / std::move(den);
}

taylor_dc_t::size_type atan2_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 2u);

    // Decompose the arguments.
    auto &y = *get_mutable_args_it().first;
    if (const auto dres = taylor_decompose_in_place(std::move(y), u_vars_defs)) {
        y = expression{"u_{}"_format(dres)};
    }
    auto &x = *(get_mutable_args_it().first + 1);
    if (const auto dres = taylor_decompose_in_place(std::move(x), u_vars_defs)) {
        x = expression{"u_{}"_format(dres)};
    }

    // Append x * x and y * y.
    u_vars_defs.emplace_back(square(x), std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(square(y), std::vector<std::uint32_t>{});

    // Append x*x + y*y.
    u_vars_defs.emplace_back(expression{"u_{}"_format(u_vars_defs.size() - 2u)}
                                 + expression{"u_{}"_format(u_vars_defs.size() - 1u)},
                             std::vector<std::uint32_t>{});

    // Append the atan2 decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed atan2).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Derivative of atan2(number, number).
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *taylor_diff_atan2_impl(llvm_state &s, const std::vector<std::uint32_t> &, const U &num0, const V &num1,
                                    const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                    std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (order == 0u) {
        // Do the number codegen.
        auto y = taylor_codegen_numparam<T>(s, num0, par_ptr, batch_size);
        auto x = taylor_codegen_numparam<T>(s, num1, par_ptr, batch_size);

        // Compute and return the atan2.
        return llvm_atan2(s, y, x);
    } else {
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of atan2(var, number).
template <typename T, typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Value *taylor_diff_atan2_impl(llvm_state &s, const std::vector<std::uint32_t> &deps, const variable &var,
                                    const U &num, const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the index of the y variable argument.
    const auto y_idx = uname_to_index(var.name());

    // Do the codegen for the x number argument.
    auto x = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

    if (order == 0u) {
        // Compute and return the atan2.
        return llvm_atan2(s, taylor_fetch_diff(arr, y_idx, 0, n_uvars), x);
    }

    // Splat the order.
    auto n = vector_splat(builder, codegen<T>(s, number{static_cast<T>(order)}), batch_size);

    // Compute the divisor: n * d^[0].
    const auto d_idx = deps[0];
    auto divisor = builder.CreateFMul(n, taylor_fetch_diff(arr, d_idx, 0, n_uvars));

    // Compute the first part of the dividend: n * c^[0] * b^[n].
    auto dividend = builder.CreateFMul(n, builder.CreateFMul(x, taylor_fetch_diff(arr, y_idx, order, n_uvars)));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto fac = vector_splat(builder, codegen<T>(s, number(-static_cast<T>(j))), batch_size);

            auto dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto tmp = builder.CreateFMul(dnj, aj);
            tmp = builder.CreateFMul(fac, tmp);
            sum.push_back(tmp);
        }

        // Update the dividend.
        dividend = builder.CreateFAdd(dividend, pairwise_sum(builder, sum));
    }

    return builder.CreateFDiv(dividend, divisor);
}

// Derivative of atan2(number, var).
template <typename T, typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Value *taylor_diff_atan2_impl(llvm_state &s, const std::vector<std::uint32_t> &deps, const U &num,
                                    const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the index of the x variable argument.
    const auto x_idx = uname_to_index(var.name());

    // Do the codegen for the y number argument.
    auto y = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

    if (order == 0u) {
        // Compute and return the atan2.
        return llvm_atan2(s, y, taylor_fetch_diff(arr, x_idx, 0, n_uvars));
    }

    // Splat the order.
    auto n = vector_splat(builder, codegen<T>(s, number{static_cast<T>(order)}), batch_size);

    // Compute the divisor: n * d^[0].
    const auto d_idx = deps[0];
    auto divisor = builder.CreateFMul(n, taylor_fetch_diff(arr, d_idx, 0, n_uvars));

    // Compute the first part of the dividend: -n * b^[0] * c^[n].
    auto dividend = builder.CreateFMul(builder.CreateFNeg(n),
                                       builder.CreateFMul(y, taylor_fetch_diff(arr, x_idx, order, n_uvars)));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto fac = vector_splat(builder, codegen<T>(s, number(-static_cast<T>(j))), batch_size);

            auto dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto tmp = builder.CreateFMul(dnj, aj);
            tmp = builder.CreateFMul(fac, tmp);
            sum.push_back(tmp);
        }

        // Update the dividend.
        dividend = builder.CreateFAdd(dividend, pairwise_sum(builder, sum));
    }

    return builder.CreateFDiv(dividend, divisor);
}

// Derivative of atan2(var, var).
template <typename T>
llvm::Value *taylor_diff_atan2_impl(llvm_state &s, const std::vector<std::uint32_t> &deps, const variable &var0,
                                    const variable &var1, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the indices of the y and x variable arguments.
    const auto y_idx = uname_to_index(var0.name());
    const auto x_idx = uname_to_index(var1.name());

    if (order == 0u) {
        // Compute and return the atan2.
        return llvm_atan2(s, taylor_fetch_diff(arr, y_idx, 0, n_uvars), taylor_fetch_diff(arr, x_idx, 0, n_uvars));
    }

    // Splat the order.
    auto n = vector_splat(builder, codegen<T>(s, number{static_cast<T>(order)}), batch_size);

    // Compute the divisor: n * d^[0].
    const auto d_idx = deps[0];
    auto divisor = builder.CreateFMul(n, taylor_fetch_diff(arr, d_idx, 0, n_uvars));

    // Compute the first part of the dividend: n * (c^[0] * b^[n] - b^[0] * c^[n]).
    auto dividend
        = builder.CreateFMul(taylor_fetch_diff(arr, x_idx, 0, n_uvars), taylor_fetch_diff(arr, y_idx, order, n_uvars));
    dividend = builder.CreateFSub(dividend, builder.CreateFMul(taylor_fetch_diff(arr, y_idx, 0, n_uvars),
                                                               taylor_fetch_diff(arr, x_idx, order, n_uvars)));
    dividend = builder.CreateFMul(n, dividend);

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

            auto cnj = taylor_fetch_diff(arr, x_idx, order - j, n_uvars);
            auto bj = taylor_fetch_diff(arr, y_idx, j, n_uvars);

            auto bnj = taylor_fetch_diff(arr, y_idx, order - j, n_uvars);
            auto cj = taylor_fetch_diff(arr, x_idx, j, n_uvars);

            auto dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto tmp1 = builder.CreateFMul(cnj, bj);
            auto tmp2 = builder.CreateFMul(bnj, cj);
            auto tmp3 = builder.CreateFMul(dnj, aj);
            auto tmp = builder.CreateFSub(builder.CreateFSub(tmp1, tmp2), tmp3);

            tmp = builder.CreateFMul(fac, tmp);
            sum.push_back(tmp);
        }

        // Update the dividend.
        dividend = builder.CreateFAdd(dividend, pairwise_sum(builder, sum));
    }

    return builder.CreateFDiv(dividend, divisor);
}

// All the other cases.
template <typename T, typename U, typename V, typename... Args>
llvm::Value *taylor_diff_atan2_impl(llvm_state &, const std::vector<std::uint32_t> &, const U &, const V &,
                                    const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                    std::uint32_t, std::uint32_t, const Args &...)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of atan2()");
}

template <typename T>
llvm::Value *taylor_diff_atan2(llvm_state &s, const atan2_impl &f, const std::vector<std::uint32_t> &deps,
                               const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                               std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 2u);

    if (deps.size() != 1u) {
        throw std::invalid_argument("A hidden dependency vector of size 1 is expected in order to compute the Taylor "
                                    "derivative of atan2(), but a vector of size {} was passed "
                                    "instead"_format(deps.size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_atan2_impl<T>(s, deps, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value(), f.args()[1].value());
}

} // namespace

llvm::Value *atan2_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size) const
{
    return taylor_diff_atan2<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *atan2_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                          std::uint32_t batch_size) const
{
    return taylor_diff_atan2<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *atan2_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                          std::uint32_t batch_size) const
{
    return taylor_diff_atan2<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

} // namespace detail

expression atan2(expression y, expression x)
{
    return expression{func{detail::atan2_impl(std::move(y), std::move(x))}};
}

} // namespace heyoka
