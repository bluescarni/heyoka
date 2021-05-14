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

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

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

kepE_impl::kepE_impl() : kepE_impl(0_dbl, 0_dbl) {}

kepE_impl::kepE_impl(expression e, expression M) : func_base("kepE", std::vector{std::move(e), std::move(M)}) {}

expression kepE_impl::diff(const std::string &s) const
{
    assert(args().size() == 2u);

    const auto &e = args()[0];
    const auto &M = args()[1];

    expression E{func{*this}};

    return (heyoka::diff(e, s) * sin(E) + heyoka::diff(M, s)) / (1_dbl - e * cos(E));
}

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
kepE_impl::taylor_decompose(std::vector<std::pair<expression, std::vector<std::uint32_t>>> &u_vars_defs) &&
{
    assert(args().size() == 2u);

    // Decompose the arguments.
    auto &e = *get_mutable_args_it().first;
    if (const auto dres = taylor_decompose_in_place(std::move(e), u_vars_defs)) {
        e = expression{variable{"u_{}"_format(dres)}};
    }

    auto &M = *(get_mutable_args_it().first + 1);
    if (const auto dres = taylor_decompose_in_place(std::move(M), u_vars_defs)) {
        M = expression{variable{"u_{}"_format(dres)}};
    }

    // Make a copy of e.
    auto e_copy = e;

    // Append the kepE decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Append the sin(a)/cos(a) decompositions.
    u_vars_defs.emplace_back(sin(expression{variable{"u_{}"_format(u_vars_defs.size() - 1u)}}),
                             std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(cos(expression{variable{"u_{}"_format(u_vars_defs.size() - 2u)}}),
                             std::vector<std::uint32_t>{});

    // Append the e*cos(a) decomposition.
    u_vars_defs.emplace_back(std::move(e_copy) * expression{variable{"u_{}"_format(u_vars_defs.size() - 1u)}},
                             std::vector<std::uint32_t>{});

    // Add the hidden deps.
    // NOTE: hidden deps on e*cos(a) and sin(a) (in this order).
    (u_vars_defs.end() - 4)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));
    (u_vars_defs.end() - 4)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 3u));

    // sin/cos hidden deps.
    (u_vars_defs.end() - 3)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));
    (u_vars_defs.end() - 2)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 3u));

    return u_vars_defs.size() - 4u;
}

namespace
{

template <typename T>
auto kepE_codegen_impl(llvm_state &s, const std::vector<llvm::Value *> &args)
{
    assert(args.size() == 2u);
    assert(args[0] != nullptr);
    assert(args[1] != nullptr);

    auto &builder = s.builder();

    // Determine whether we are operating on scalars or vectors.
    std::uint32_t batch_size = 1;
    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
    }

    // Create/fetch the inverse Kepler function solver.
    auto kepE_func = llvm_add_inv_kep_E<T>(s, batch_size);

    // Call it.
    return builder.CreateCall(kepE_func, {args[0], args[1]});
}

} // namespace

llvm::Value *kepE_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    return kepE_codegen_impl<double>(s, args);
}

llvm::Value *kepE_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    return kepE_codegen_impl<long double>(s, args);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *kepE_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    return kepE_codegen_impl<mppp::real128>(s, args);
}

#endif

namespace
{

// Derivative of kepE(number, number).
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, const std::vector<std::uint32_t> &, const U &num0, const V &num1,
                                   const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                   std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (order == 0u) {
        // Do the number codegen.
        auto e = taylor_codegen_numparam<T>(s, num0, par_ptr, batch_size);
        auto M = taylor_codegen_numparam<T>(s, num1, par_ptr, batch_size);

        // Create/fetch the Kepler solver.
        auto fkep = llvm_add_inv_kep_E<T>(s, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {e, M});
    } else {
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of kepE(var, number).
template <typename T, typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, const std::vector<std::uint32_t> &deps, const variable &var,
                                   const U &num, const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    assert(deps.size() == 2u);

    auto &builder = s.builder();

    // Fetch the index of the e variable argument.
    const auto e_idx = uname_to_index(var.name());

    // Do the codegen for the M number argument.
    auto M = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto fkep = llvm_add_inv_kep_E<T>(s, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {taylor_fetch_diff(arr, e_idx, 0, n_uvars), M});
    }

    // Splat the order.
    auto n = vector_splat(builder, codegen<T>(s, number{static_cast<T>(order)}), batch_size);

    // Compute the divisor: n * (1 - c^[0]).
    const auto c_idx = deps[0];
    auto one_fp = vector_splat(builder, codegen<T>(s, number{1.}), batch_size);
    auto divisor = builder.CreateFMul(n, builder.CreateFSub(one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars)));

    // Compute the first part of the dividend: n * e^[n] * d^[0] (the derivative of M is zero because
    // here M is a constant and the order is > 0).
    const auto d_idx = deps[1];
    auto dividend = builder.CreateFMul(n, builder.CreateFMul(taylor_fetch_diff(arr, e_idx, order, n_uvars),
                                                             taylor_fetch_diff(arr, d_idx, 0, n_uvars)));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

            auto cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto ej = taylor_fetch_diff(arr, e_idx, j, n_uvars);

            auto tmp = builder.CreateFMul(dnj, ej);
            tmp = builder.CreateFAdd(builder.CreateFMul(cnj, aj), tmp);
            sum.push_back(builder.CreateFMul(fac, tmp));
        }

        // Update the dividend.
        dividend = builder.CreateFAdd(dividend, pairwise_sum(builder, sum));
    }

    return builder.CreateFDiv(dividend, divisor);
}

// Derivative of kepE(number, var).
template <typename T, typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, const std::vector<std::uint32_t> &deps, const U &num,
                                   const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    assert(deps.size() == 2u);

    auto &builder = s.builder();

    // Fetch the index of the M variable argument.
    const auto M_idx = uname_to_index(var.name());

    // Do the codegen for the e number argument.
    auto e = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto fkep = llvm_add_inv_kep_E<T>(s, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {e, taylor_fetch_diff(arr, M_idx, 0, n_uvars)});
    }

    // Splat the order.
    auto n = vector_splat(builder, codegen<T>(s, number{static_cast<T>(order)}), batch_size);

    // Compute the divisor: n * (1 - c^[0]).
    const auto c_idx = deps[0];
    auto one_fp = vector_splat(builder, codegen<T>(s, number{1.}), batch_size);
    auto divisor = builder.CreateFMul(n, builder.CreateFSub(one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars)));

    // Compute the first part of the dividend: n * M^[n] (the derivative of e is zero because
    // here e is a constant and the order is > 0).
    auto dividend = builder.CreateFMul(n, taylor_fetch_diff(arr, M_idx, order, n_uvars));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

            auto cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto tmp = builder.CreateFMul(fac, builder.CreateFMul(cnj, aj));
            sum.push_back(tmp);
        }

        // Update the dividend.
        dividend = builder.CreateFAdd(dividend, pairwise_sum(builder, sum));
    }

    return builder.CreateFDiv(dividend, divisor);
}

// Derivative of kepE(var, var).
template <typename T>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, const std::vector<std::uint32_t> &deps, const variable &var0,
                                   const variable &var1, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    assert(deps.size() == 2u);

    auto &builder = s.builder();

    // Fetch the index of the e/M variable arguments.
    const auto e_idx = uname_to_index(var0.name());
    const auto M_idx = uname_to_index(var1.name());

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto fkep = llvm_add_inv_kep_E<T>(s, batch_size);

        // Invoke and return.
        return builder.CreateCall(
            fkep, {taylor_fetch_diff(arr, e_idx, 0, n_uvars), taylor_fetch_diff(arr, M_idx, 0, n_uvars)});
    }

    // Splat the order.
    auto n = vector_splat(builder, codegen<T>(s, number{static_cast<T>(order)}), batch_size);

    // Compute the divisor: n * (1 - c^[0]).
    const auto c_idx = deps[0];
    auto one_fp = vector_splat(builder, codegen<T>(s, number{1.}), batch_size);
    auto divisor = builder.CreateFMul(n, builder.CreateFSub(one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars)));

    // Compute the first part of the dividend: n * (e^[n] * d^[0] + M^[n]).
    const auto d_idx = deps[1];
    auto dividend
        = builder.CreateFMul(taylor_fetch_diff(arr, e_idx, order, n_uvars), taylor_fetch_diff(arr, d_idx, 0, n_uvars));
    dividend = builder.CreateFAdd(dividend, taylor_fetch_diff(arr, M_idx, order, n_uvars));
    dividend = builder.CreateFMul(n, dividend);

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

            auto cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto ej = taylor_fetch_diff(arr, e_idx, j, n_uvars);

            auto tmp = builder.CreateFMul(dnj, ej);
            tmp = builder.CreateFAdd(builder.CreateFMul(cnj, aj), tmp);
            sum.push_back(builder.CreateFMul(fac, tmp));
        }

        // Update the dividend.
        dividend = builder.CreateFAdd(dividend, pairwise_sum(builder, sum));
    }

    return builder.CreateFDiv(dividend, divisor);
}

// All the other cases.
template <typename T, typename U, typename V, typename... Args>
llvm::Value *taylor_diff_kepE_impl(llvm_state &, const std::vector<std::uint32_t> &, const U &, const V &,
                                   const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                   std::uint32_t, std::uint32_t, const Args &...)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of kepE()");
}

template <typename T>
llvm::Value *taylor_diff_kepE(llvm_state &s, const kepE_impl &f, const std::vector<std::uint32_t> &deps,
                              const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                              std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 2u);

    if (deps.size() != 2u) {
        throw std::invalid_argument("A hidden dependency vector of size 2 is expected in order to compute the Taylor "
                                    "derivative of kepE(), but a vector of size {} was passed "
                                    "instead"_format(deps.size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_kepE_impl<T>(s, deps, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value(), f.args()[1].value());
}

} // namespace

llvm::Value *kepE_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size) const
{
    return taylor_diff_kepE<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *kepE_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size) const
{
    return taylor_diff_kepE<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *kepE_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size) const
{
    return taylor_diff_kepE<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

} // namespace detail

expression kepE(expression e, expression M)
{
    return expression{func{detail::kepE_impl{std::move(e), std::move(M)}}};
}

} // namespace heyoka
