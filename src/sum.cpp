// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/sum.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_diff_sum(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of sum() (the order must be at least one)");
    }

    // Filter-out the u variables in func's arguments and load the corresponding
    // derivatives at the desired order.
    std::vector<llvm::Value *> u_der;
    for (const auto &arg : func.args()) {
        if (auto pval = std::get_if<variable>(&arg.value())) {
            u_der.push_back(taylor_fetch_diff(arr, uname_to_index(pval->name()), order, n_uvars));
        } else if (!std::holds_alternative<number>(arg.value())) {
            throw std::invalid_argument(
                "An invalid argument type was encountered while trying to build the Taylor derivative "
                "of a summation");
        }
    }

    if (u_der.empty()) {
        // No u variables appear in the list of arguments, return zero.
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    } else {
        return pairwise_sum(s.builder(), u_der);
    }
}

template <typename T>
llvm::Value *taylor_c_diff_sum(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                               llvm::Value *order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Filter-out the u variables in func's arguments and load the corresponding
    // derivatives at the desired order.
    std::vector<llvm::Value *> u_der;
    for (const auto &arg : func.args()) {
        if (auto pval = std::get_if<variable>(&arg.value())) {
            u_der.push_back(taylor_c_load_diff(s, arr, n_uvars, order, builder.getInt32(uname_to_index(pval->name()))));
        } else if (!std::holds_alternative<number>(arg.value())) {
            throw std::invalid_argument(
                "An invalid argument type was encountered while trying to build the Taylor derivative "
                "of a summation in compact mode");
        }
    }

    if (u_der.empty()) {
        // No u variables appear in the list of arguments, return zero.
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    } else {
        return pairwise_sum(builder, u_der);
    }
}

} // namespace

} // namespace detail

expression sum(std::vector<expression> terms)
{
    function fc{std::move(terms)};
    fc.display_name() = "sum";

    auto codegen_impl = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.empty()) {
            throw std::invalid_argument("Cannot codegen a summation without terms");
        }

        auto args_copy(args);
        return detail::pairwise_sum(s.builder(), args_copy);
    };

    fc.codegen_dbl_f() = codegen_impl;
    fc.codegen_ldbl_f() = codegen_impl;
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = codegen_impl;
#endif

    fc.taylor_diff_dbl_f() = detail::taylor_diff_sum<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sum<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sum<mppp::real128>;
#endif

    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_sum<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_sum<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_sum<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
