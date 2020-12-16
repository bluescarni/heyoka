// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <initializer_list>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Attributes.h>
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
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

square_impl::square_impl(expression e) : func_base("square", std::vector{std::move(e)}) {}

square_impl::square_impl() : square_impl(0_dbl) {}

llvm::Value *square_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return s.builder().CreateFMul(args[0], args[0]);
}

llvm::Value *square_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    // NOTE: codegen is identical as in dbl.
    return codegen_dbl(s, args);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *square_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    // NOTE: codegen is identical as in dbl.
    return codegen_dbl(s, args);
}

#endif

namespace
{

// Derivative of square(number).
template <typename T>
llvm::Value *taylor_diff_square_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of square(variable).
template <typename T>
llvm::Value *taylor_diff_square_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                     std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t)
{
    // NOTE: we are currently not allowing order 0 derivatives
    // in non-compact mode.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of square() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // Compute the sum.
    std::vector<llvm::Value *> sum;
    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 0; j <= (order - 1u) / 2u; ++j) {
            auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto ret = pairwise_sum(builder, sum);
        return builder.CreateFAdd(ret, ret);
    } else {
        // Even order.
        auto ak2 = taylor_fetch_diff(arr, u_idx, order / 2u, n_uvars);
        auto sq_ak2 = builder.CreateFMul(ak2, ak2);

        for (std::uint32_t j = 0; j <= (order - 2u) / 2u; ++j) {
            auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto ret = pairwise_sum(builder, sum);
        return builder.CreateFAdd(builder.CreateFAdd(ret, ret), sq_ak2);
    }
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_square_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a square");
}

template <typename T>
llvm::Value *taylor_diff_square(llvm_state &s, const square_impl &f, const std::vector<llvm::Value *> &arr,
                                std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_diff_square_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        f.args()[0].value());
}

} // namespace

llvm::Value *square_impl::taylor_diff_dbl(llvm_state &s, const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                          std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size) const
{
    return taylor_diff_square<double>(s, *this, arr, n_uvars, order, idx, batch_size);
}

llvm::Value *square_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                           std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size) const
{
    return taylor_diff_square<long double>(s, *this, arr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *square_impl::taylor_diff_f128(llvm_state &s, const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                           std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size) const
{
    return taylor_diff_square<mppp::real128>(s, *this, arr, n_uvars, order, idx, batch_size);
}

#endif

} // namespace detail

} // namespace heyoka
