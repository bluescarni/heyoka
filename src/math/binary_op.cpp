// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
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

binary_op::binary_op() : binary_op(type::add, 0_dbl, 0_dbl) {}

binary_op::binary_op(type t, expression a, expression b)
    : func_base("binary_op", std::vector{std::move(a), std::move(b)}), m_type(t)
{
    assert(m_type >= type::add && m_type <= type::div);
}

bool binary_op::extra_equal_to(const func &f) const
{
    // NOTE: this should be ensured by the
    // implementation of func's equality operator.
    assert(f.extract<binary_op>() == f.get_ptr());

    return static_cast<const binary_op *>(f.get_ptr())->m_type == m_type;
}

std::size_t binary_op::extra_hash() const
{
    return std::hash<type>{}(m_type);
}

void binary_op::to_stream(std::ostream &os) const
{
    assert(args().size() == 2u);
    assert(m_type >= type::add && m_type <= type::div);

    os << '(' << lhs() << ' ';

    switch (m_type) {
        case type::add:
            os << '+';
            break;
        case type::sub:
            os << '-';
            break;
        case type::mul:
            os << '*';
            break;
        default:
            os << '/';
            break;
    }

    os << ' ' << rhs() << ')';
}

binary_op::type binary_op::op() const
{
    return m_type;
}

const expression &binary_op::lhs() const
{
    assert(args().size() == 2u);
    return args()[0];
}

const expression &binary_op::rhs() const
{
    assert(args().size() == 2u);
    return args()[1];
}

template <typename T>
expression binary_op::diff_impl(std::unordered_map<const void *, expression> &func_map, const T &x) const
{
    assert(args().size() == 2u);
    assert(m_type >= type::add && m_type <= type::div);

    switch (m_type) {
        case type::add:
            return detail::diff(func_map, lhs(), x) + detail::diff(func_map, rhs(), x);
        case type::sub:
            return detail::diff(func_map, lhs(), x) - detail::diff(func_map, rhs(), x);
        case type::mul:
            return detail::diff(func_map, lhs(), x) * rhs() + lhs() * detail::diff(func_map, rhs(), x);
        default:
            return (detail::diff(func_map, lhs(), x) * rhs() - lhs() * detail::diff(func_map, rhs(), x))
                   / (rhs() * rhs());
    }
}

expression binary_op::diff(std::unordered_map<const void *, expression> &func_map, const std::string &s) const
{
    return diff_impl(func_map, s);
}

expression binary_op::diff(std::unordered_map<const void *, expression> &func_map, const param &p) const
{
    return diff_impl(func_map, p);
}

namespace
{

template <class T>
T eval_bo_impl(const binary_op &bo, const std::unordered_map<std::string, T> &map, const std::vector<T> &pars)
{
    assert(bo.args().size() == 2u);
    assert(bo.op() >= binary_op::type::add && bo.op() <= binary_op::type::div);

    switch (bo.op()) {
        case binary_op::type::add:
            return eval<T>(bo.lhs(), map, pars) + eval<T>(bo.rhs(), map, pars);
        case binary_op::type::sub:
            return eval<T>(bo.lhs(), map, pars) - eval<T>(bo.rhs(), map, pars);
        case binary_op::type::mul:
            return eval<T>(bo.lhs(), map, pars) * eval<T>(bo.rhs(), map, pars);
        default:
            return eval<T>(bo.lhs(), map, pars) / eval<T>(bo.rhs(), map, pars);
    }
}

} // namespace

double binary_op::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    return eval_bo_impl<double>(*this, map, pars);
}

long double binary_op::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                 const std::vector<long double> &pars) const
{
    return eval_bo_impl<long double>(*this, map, pars);
}

#if defined(HEYOKA_HAVE_REAL128)

mppp::real128 binary_op::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                   const std::vector<mppp::real128> &pars) const
{
    return eval_bo_impl<mppp::real128>(*this, map, pars);
}

#endif

void binary_op::eval_batch_dbl(std::vector<double> &out_values,
                               const std::unordered_map<std::string, std::vector<double>> &map,
                               const std::vector<double> &pars) const
{
    assert(args().size() == 2u);
    assert(m_type >= type::add && m_type <= type::div);

    auto tmp = out_values;
    heyoka::eval_batch_dbl(out_values, lhs(), map, pars);
    heyoka::eval_batch_dbl(tmp, rhs(), map, pars);
    switch (m_type) {
        case type::add:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::plus<>());
            break;
        case type::sub:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::minus<>());
            break;
        case type::mul:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::multiplies<>());
            break;
        default:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::divides<>());
            break;
    }
}

namespace
{

[[nodiscard]] llvm::Value *bo_llvm_eval(llvm_state &s, const std::vector<llvm::Value *> &args, binary_op::type op)
{
    assert(args.size() == 2u);

    switch (op) {
        case binary_op::type::add:
            return s.builder().CreateFAdd(args[0], args[1]);
        case binary_op::type::sub:
            return s.builder().CreateFSub(args[0], args[1]);
        case binary_op::type::mul:
            return s.builder().CreateFMul(args[0], args[1]);
        default:
            assert(op == binary_op::type::div);
            return s.builder().CreateFDiv(args[0], args[1]);
    }
}

} // namespace

llvm::Value *binary_op::llvm_eval_dbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                      std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<double>(
        [&s, this](const std::vector<llvm::Value *> &args, bool) { return bo_llvm_eval(s, args, op()); }, *this, s,
        eval_arr, par_ptr, batch_size, high_accuracy);
}

llvm::Value *binary_op::llvm_eval_ldbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                       std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<long double>(
        [&s, this](const std::vector<llvm::Value *> &args, bool) { return bo_llvm_eval(s, args, op()); }, *this, s,
        eval_arr, par_ptr, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *binary_op::llvm_eval_f128(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                       std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<mppp::real128>(
        [&s, this](const std::vector<llvm::Value *> &args, bool) { return bo_llvm_eval(s, args, op()); }, *this, s,
        eval_arr, par_ptr, batch_size, high_accuracy);
}

#endif

namespace
{

// Derivative of number +- number.
template <bool AddOrSub, typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *bo_taylor_diff_addsub_impl(llvm_state &s, const U &num0, const V &num1, const std::vector<llvm::Value *> &,
                                        llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                        std::uint32_t batch_size)
{
    if (order == 0u) {
        auto n0 = taylor_codegen_numparam<T>(s, num0, par_ptr, batch_size);
        auto n1 = taylor_codegen_numparam<T>(s, num1, par_ptr, batch_size);

        return AddOrSub ? s.builder().CreateFAdd(n0, n1) : s.builder().CreateFSub(n0, n1);
    } else {
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of number +- var.
template <bool AddOrSub, typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *bo_taylor_diff_addsub_impl(llvm_state &s, const U &num, const variable &var,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                        std::uint32_t batch_size)
{
    auto &builder = s.builder();

    auto ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);

    if (order == 0u) {
        auto n = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

        return AddOrSub ? builder.CreateFAdd(n, ret) : builder.CreateFSub(n, ret);
    } else {
        if constexpr (AddOrSub) {
            return ret;
        } else {
            // Negate if we are doing a subtraction.
            return builder.CreateFNeg(ret);
        }
    }
}

// Derivative of var +- number.
template <bool AddOrSub, typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *bo_taylor_diff_addsub_impl(llvm_state &s, const variable &var, const U &num,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                        std::uint32_t batch_size)
{
    auto ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);

    if (order == 0u) {
        auto &builder = s.builder();

        auto n = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

        return AddOrSub ? builder.CreateFAdd(ret, n) : builder.CreateFSub(ret, n);
    } else {
        return ret;
    }
}

// Derivative of var +- var.
template <bool AddOrSub, typename T>
llvm::Value *bo_taylor_diff_addsub_impl(llvm_state &s, const variable &var0, const variable &var1,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                        std::uint32_t order, std::uint32_t, std::uint32_t)
{
    auto v0 = taylor_fetch_diff(arr, uname_to_index(var0.name()), order, n_uvars);
    auto v1 = taylor_fetch_diff(arr, uname_to_index(var1.name()), order, n_uvars);

    if constexpr (AddOrSub) {
        return s.builder().CreateFAdd(v0, v1);
    } else {
        return s.builder().CreateFSub(v0, v1);
    }
}

// All the other cases.
// LCOV_EXCL_START
template <bool, typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Value *bo_taylor_diff_addsub_impl(llvm_state &, const V1 &, const V2 &, const std::vector<llvm::Value *> &,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of add()/sub()");
}
// LCOV_EXCL_STOP

template <typename T>
llvm::Value *bo_taylor_diff_add(llvm_state &s, const binary_op &bo, const std::vector<llvm::Value *> &arr,
                                llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_diff_addsub_impl<true, T>(s, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Value *bo_taylor_diff_sub(llvm_state &s, const binary_op &bo, const std::vector<llvm::Value *> &arr,
                                llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_diff_addsub_impl<false, T>(s, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number * number.
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const U &num0, const V &num1, const std::vector<llvm::Value *> &,
                                     llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                     std::uint32_t batch_size)
{
    if (order == 0u) {
        auto n0 = taylor_codegen_numparam<T>(s, num0, par_ptr, batch_size);
        auto n1 = taylor_codegen_numparam<T>(s, num1, par_ptr, batch_size);

        return s.builder().CreateFMul(n0, n1);
    } else {
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of var * number.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const variable &var, const U &num,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    auto ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);
    auto mul = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

    return builder.CreateFMul(mul, ret);
}

// Derivative of number * var.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const U &num, const variable &var,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    // Return the derivative of var * number.
    return bo_taylor_diff_mul_impl<T>(s, var, num, arr, par_ptr, n_uvars, order, idx, batch_size);
}

// Derivative of var * var.
template <typename T>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const variable &var0, const variable &var1,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t, std::uint32_t)
{
    // Fetch the indices of the u variables.
    const auto u_idx0 = uname_to_index(var0.name());
    const auto u_idx1 = uname_to_index(var1.name());

    // NOTE: iteration in the [0, order] range
    // (i.e., order inclusive).
    std::vector<llvm::Value *> sum;
    auto &builder = s.builder();
    for (std::uint32_t j = 0; j <= order; ++j) {
        auto v0 = taylor_fetch_diff(arr, u_idx0, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, u_idx1, j, n_uvars);

        // Add v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(v0, v1));
    }

    return pairwise_sum(builder, sum);
}

// All the other cases.
// LCOV_EXCL_START
template <typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &, const V1 &, const V2 &, const std::vector<llvm::Value *> &,
                                     llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of mul()");
}
// LCOV_EXCL_STOP

template <typename T>
llvm::Value *bo_taylor_diff_mul(llvm_state &s, const binary_op &bo, const std::vector<llvm::Value *> &arr,
                                llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_diff_mul_impl<T>(s, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number / number.
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &s, const U &num0, const V &num1, const std::vector<llvm::Value *> &,
                                     llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                     std::uint32_t batch_size)
{
    if (order == 0u) {
        auto n0 = taylor_codegen_numparam<T>(s, num0, par_ptr, batch_size);
        auto n1 = taylor_codegen_numparam<T>(s, num1, par_ptr, batch_size);

        return s.builder().CreateFDiv(n0, n1);
    } else {
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of variable / variable or number / variable. These two cases
// are quite similar, so we handle them together.
template <typename T, typename U,
          std::enable_if_t<
              std::disjunction_v<std::is_same<U, number>, std::is_same<U, variable>, std::is_same<U, param>>, int> = 0>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &s, const U &nv, const variable &var1,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Fetch the index of var1.
    const auto u_idx1 = uname_to_index(var1.name());

    if (order == 0u) {
        // Special casing for zero order.
        auto numerator = [&]() -> llvm::Value * {
            if constexpr (std::is_same_v<U, number> || std::is_same_v<U, param>) {
                return taylor_codegen_numparam<T>(s, nv, par_ptr, batch_size);
            } else {
                return taylor_fetch_diff(arr, uname_to_index(nv.name()), 0, n_uvars);
            }
        }();

        return builder.CreateFDiv(numerator, taylor_fetch_diff(arr, u_idx1, 0, n_uvars));
    }

    // NOTE: iteration in the [1, order] range
    // (i.e., order inclusive).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        auto v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, u_idx1, j, n_uvars);

        // Add v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(v0, v1));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Load the divisor for the quotient formula.
    // This is the zero-th order derivative of var1.
    auto div = taylor_fetch_diff(arr, u_idx1, 0, n_uvars);

    if constexpr (std::is_same_v<U, number> || std::is_same_v<U, param>) {
        // nv is a number/param. Negate the accumulator
        // and divide it by the divisor.
        return builder.CreateFDiv(builder.CreateFNeg(ret_acc), div);
    } else {
        // nv is a variable. We need to fetch its
        // derivative of order 'order' from the array of derivatives.
        auto diff_nv_v = taylor_fetch_diff(arr, uname_to_index(nv.name()), order, n_uvars);

        // Produce the result: (diff_nv_v - ret_acc) / div.
        return builder.CreateFDiv(builder.CreateFSub(diff_nv_v, ret_acc), div);
    }
}

// Derivative of variable / number.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &s, const variable &var, const U &num,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    auto ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);
    auto div = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);

    return builder.CreateFDiv(ret, div);
}

// All the other cases.
// LCOV_EXCL_START
template <typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &, const V1 &, const V2 &, const std::vector<llvm::Value *> &,
                                     llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of div()");
}
// LCOV_EXCL_STOP

template <typename T>
llvm::Value *bo_taylor_diff_div(llvm_state &s, const binary_op &bo, const std::vector<llvm::Value *> &arr,
                                llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_diff_div_impl<T>(s, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Value *taylor_diff_bo_impl(llvm_state &s, const binary_op &bo, const std::vector<std::uint32_t> &deps,
                                 const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                 std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(bo.args().size() == 2u);
    assert(bo.op() >= binary_op::type::add && bo.op() <= binary_op::type::div);

    if (!deps.empty()) {
        throw std::invalid_argument("The vector of hidden dependencies in the Taylor diff for a binary operator "
                                    "should be empty, but instead it has a size of {}"_format(deps.size()));
    }

    switch (bo.op()) {
        case binary_op::type::add:
            return bo_taylor_diff_add<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
        case binary_op::type::sub:
            return bo_taylor_diff_sub<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
        case binary_op::type::mul:
            return bo_taylor_diff_mul<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
        default:
            return bo_taylor_diff_div<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
    }
}

} // namespace

llvm::Value *binary_op::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size, bool) const
{

    return taylor_diff_bo_impl<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *binary_op::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size, bool) const
{
    return taylor_diff_bo_impl<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *binary_op::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size, bool) const
{
    return taylor_diff_bo_impl<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

namespace
{

// Helper to implement the function for the differentiation of
// 'number/param op number/param' in compact mode. The function will always return zero,
// unless the order is 0 (in which case it will return the result of the codegen).
template <typename T, typename U, typename V>
llvm::Function *bo_taylor_c_diff_func_num_num(llvm_state &s, const binary_op &bo, const U &n0, const V &n1,
                                              std::uint32_t n_uvars, std::uint32_t batch_size,
                                              const std::string &op_name)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, op_name, n_uvars, batch_size, {n0, n1});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto par_ptr = f->args().begin() + 3;
        auto num0 = f->args().begin() + 5;
        auto num1 = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, run the codegen.
                auto vnum0 = taylor_c_diff_numparam_codegen(s, n0, num0, par_ptr, batch_size);
                auto vnum1 = taylor_c_diff_numparam_codegen(s, n1, num1, par_ptr, batch_size);

                switch (bo.op()) {
                    case binary_op::type::add:
                        builder.CreateStore(builder.CreateFAdd(vnum0, vnum1), retval);
                        break;
                    case binary_op::type::sub:
                        builder.CreateStore(builder.CreateFSub(vnum0, vnum1), retval);
                        break;
                    case binary_op::type::mul:
                        builder.CreateStore(builder.CreateFMul(vnum0, vnum1), retval);
                        break;
                    default:
                        builder.CreateStore(builder.CreateFDiv(vnum0, vnum1), retval);
                }
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of {}() "
                                        "in compact mode detected"_format(op_name));
        }
    }

    return f;
}

// Derivative of number/param +- number/param.
template <bool AddOrSub, typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_op &bo, const U &num0, const V &num1,
                                                  std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return bo_taylor_c_diff_func_num_num<T>(s, bo, num0, num1, n_uvars, batch_size, AddOrSub ? "add" : "sub");
}

// Derivative of number +- var.
template <bool AddOrSub, typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_op &, const U &n, const variable &var,
                                                  std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair
        = taylor_c_diff_func_name_args<T>(context, AddOrSub ? "add" : "sub", n_uvars, batch_size, {n, var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto num = f->args().begin() + 5;
        auto var_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(0)),
            [&]() {
                // For order zero, run the codegen.
                auto num_vec = taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size);
                auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), var_idx);

                builder.CreateStore(AddOrSub ? builder.CreateFAdd(num_vec, ret) : builder.CreateFSub(num_vec, ret),
                                    retval);
            },
            [&]() {
                // Load the derivative.
                auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

                if constexpr (!AddOrSub) {
                    ret = builder.CreateFNeg(ret);
                }

                // Create the return value.
                builder.CreateStore(ret, retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of addition in compact mode detected");
        }
    }

    return f;
}

// Derivative of var +- number.
template <bool AddOrSub, typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_op &, const variable &var, const U &n,
                                                  std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair
        = taylor_c_diff_func_name_args<T>(context, AddOrSub ? "add" : "sub", n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 5;
        auto num = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(0)),
            [&]() {
                // For order zero, run the codegen.
                auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), var_idx);
                auto num_vec = taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size);

                builder.CreateStore(AddOrSub ? builder.CreateFAdd(ret, num_vec) : builder.CreateFSub(ret, num_vec),
                                    retval);
            },
            [&]() {
                // Create the return value.
                builder.CreateStore(taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of addition in compact mode detected");
        }
    }

    return f;
}

// Derivative of var +- var.
template <bool AddOrSub, typename T>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_op &, const variable &var0,
                                                  const variable &var1, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair
        = taylor_c_diff_func_name_args<T>(context, AddOrSub ? "add" : "sub", n_uvars, batch_size, {var0, var1});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto var_idx0 = f->args().begin() + 5;
        auto var_idx1 = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto v0 = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx0);
        auto v1 = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx1);

        // Create the return value.
        if constexpr (AddOrSub) {
            builder.CreateRet(builder.CreateFAdd(v0, v1));
        } else {
            builder.CreateRet(builder.CreateFSub(v0, v1));
        }

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of addition in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
// LCOV_EXCL_START
template <bool, typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &, const binary_op &, const V1 &, const V2 &,
                                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of add()/sub() in compact mode");
}
// LCOV_EXCL_STOP

template <typename T>
llvm::Function *bo_taylor_c_diff_func_add(llvm_state &s, const binary_op &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_addsub_impl<true, T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Function *bo_taylor_c_diff_func_sub(llvm_state &s, const binary_op &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_addsub_impl<false, T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number/param * number/param.
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_op &bo, const U &num0, const V &num1,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return bo_taylor_c_diff_func_num_num<T>(s, bo, num0, num1, n_uvars, batch_size, "mul");
}

// Derivative of var * number.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_op &, const variable &var, const U &n,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "mul", n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 5;
        auto num = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(builder.CreateFMul(ret, taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size)));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of multiplication in compact mode detected");
        }
    }

    return f;
}

// Derivative of number * var.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_op &, const U &n, const variable &var,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "mul", n_uvars, batch_size, {n, var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto num = f->args().begin() + 5;
        auto var_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(builder.CreateFMul(ret, taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size)));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of multiplication in compact mode detected");
        }
    }

    return f;
}

// Derivative of var * var.
template <typename T>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_op &, const variable &var0,
                                               const variable &var1, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "mul", n_uvars, batch_size, {var0, var1});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto diff_ptr = f->args().begin() + 2;
        auto idx0 = f->args().begin() + 5;
        auto idx1 = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(0), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx0);
            auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx1);
            builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(val_t, acc), builder.CreateFMul(b_nj, cj)), acc);
        });

        // Create the return value.
        builder.CreateRet(builder.CreateLoad(val_t, acc));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of multiplication in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
// LCOV_EXCL_START
template <typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &, const binary_op &, const V1 &, const V2 &, std::uint32_t,
                                               std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of mul() in compact mode");
}
// LCOV_EXCL_STOP

template <typename T>
llvm::Function *bo_taylor_c_diff_func_mul(llvm_state &s, const binary_op &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_mul_impl<T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number/param / number/param.
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_op &bo, const U &num0, const V &num1,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return bo_taylor_c_diff_func_num_num<T>(s, bo, num0, num1, n_uvars, batch_size, "div");
}

// Derivative of var / number.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_op &, const variable &var, const U &n,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "div", n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 5;
        auto num = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(builder.CreateFDiv(ret, taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size)));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of division in compact mode detected");
        }
    }

    return f;
}

// Derivative of number / var.
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_op &, const U &n, const variable &var,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "div", n_uvars, batch_size, {n, var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        // NOTE: we don't need the number argument because
        // we only need its derivative of order n >= 1,
        // which is always zero.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto num = f->args().begin() + 5;
        auto var_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order zero, run the codegen.
                auto num_vec = taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size);
                auto ret = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx);

                builder.CreateStore(builder.CreateFDiv(num_vec, ret), retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx);
                    auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
                    builder.CreateStore(
                        builder.CreateFAdd(builder.CreateLoad(val_t, acc), builder.CreateFMul(cj, a_nj)), acc);
                });

                // Negate the loop summation.
                auto ret = builder.CreateFNeg(builder.CreateLoad(val_t, acc));

                // Divide and return.
                builder.CreateStore(
                    builder.CreateFDiv(ret, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of division in compact mode detected");
        }
    }

    return f;
}

// Derivative of var / var.
template <typename T>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_op &, const variable &var0,
                                               const variable &var1, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "div", n_uvars, batch_size, {var0, var1});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx0 = f->args().begin() + 5;
        auto var_idx1 = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx1);
            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
            builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(val_t, acc), builder.CreateFMul(cj, a_nj)), acc);
        });

        auto ret = builder.CreateFSub(taylor_c_load_diff(s, diff_ptr, n_uvars, ord, var_idx0),
                                      builder.CreateLoad(val_t, acc));

        // Divide and return.
        builder.CreateRet(
            builder.CreateFDiv(ret, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx1)));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of division in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
// LCOV_EXCL_START
template <typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &, const binary_op &, const V1 &, const V2 &, std::uint32_t,
                                               std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of div() in compact mode");
}
// LCOV_EXCL_STOP

template <typename T>
llvm::Function *bo_taylor_c_diff_func_div(llvm_state &s, const binary_op &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_div_impl<T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Function *taylor_c_diff_func_bo_impl(llvm_state &s, const binary_op &bo, std::uint32_t n_uvars,
                                           std::uint32_t batch_size)
{
    switch (bo.op()) {
        case binary_op::type::add:
            return bo_taylor_c_diff_func_add<T>(s, bo, n_uvars, batch_size);
        case binary_op::type::sub:
            return bo_taylor_c_diff_func_sub<T>(s, bo, n_uvars, batch_size);
        case binary_op::type::mul:
            return bo_taylor_c_diff_func_mul<T>(s, bo, n_uvars, batch_size);
        default:
            return bo_taylor_c_diff_func_div<T>(s, bo, n_uvars, batch_size);
    }
}

} // namespace

llvm::Function *binary_op::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                  bool) const
{
    return taylor_c_diff_func_bo_impl<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *binary_op::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                   bool) const
{
    return taylor_c_diff_func_bo_impl<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *binary_op::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                   bool) const
{
    return taylor_c_diff_func_bo_impl<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

} // namespace detail

expression add(expression x, expression y)
{
    return expression{func{detail::binary_op(detail::binary_op::type::add, std::move(x), std::move(y))}};
}

expression sub(expression x, expression y)
{
    return expression{func{detail::binary_op(detail::binary_op::type::sub, std::move(x), std::move(y))}};
}

expression mul(expression x, expression y)
{
    return expression{func{detail::binary_op(detail::binary_op::type::mul, std::move(x), std::move(y))}};
}

expression div(expression x, expression y)
{
    return expression{func{detail::binary_op(detail::binary_op::type::div, std::move(x), std::move(y))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::binary_op)
