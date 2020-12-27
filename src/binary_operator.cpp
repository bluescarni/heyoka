// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

binary_operator::binary_operator(type t, expression e1, expression e2)
    : m_type(t),
      // NOTE: need to use naked new as make_unique won't work with aggregate
      // initialization.
      m_ops(new std::array<expression, 2>{std::move(e1), std::move(e2)})
{
}

binary_operator::binary_operator(const binary_operator &other)
    : m_type(other.m_type), m_ops(std::make_unique<std::array<expression, 2>>(*other.m_ops))
{
}

binary_operator::binary_operator(binary_operator &&) noexcept = default;

binary_operator::~binary_operator() = default;

binary_operator &binary_operator::operator=(const binary_operator &bo)
{
    if (this != &bo) {
        *this = binary_operator(bo);
    }
    return *this;
}

binary_operator &binary_operator::operator=(binary_operator &&) noexcept = default;

expression &binary_operator::lhs()
{
    assert(m_ops);
    return (*m_ops)[0];
}

expression &binary_operator::rhs()
{
    assert(m_ops);
    return (*m_ops)[1];
}

binary_operator::type &binary_operator::op()
{
    assert(m_type >= type::add && m_type <= type::div);
    return m_type;
}

std::array<expression, 2> &binary_operator::args()
{
    assert(m_ops);
    return *m_ops;
}

const expression &binary_operator::lhs() const
{
    assert(m_ops);
    return (*m_ops)[0];
}

const expression &binary_operator::rhs() const
{
    assert(m_ops);
    return (*m_ops)[1];
}

const binary_operator::type &binary_operator::op() const
{
    assert(m_type >= type::add && m_type <= type::div);
    return m_type;
}

const std::array<expression, 2> &binary_operator::args() const
{
    assert(m_ops);
    return *m_ops;
}

void swap(binary_operator &bo0, binary_operator &bo1) noexcept
{
    std::swap(bo0.m_type, bo1.m_type);
    std::swap(bo0.m_ops, bo1.m_ops);
}

std::size_t hash(const binary_operator &bo)
{
    return std::hash<binary_operator::type>{}(bo.op()) + hash(bo.lhs()) + hash(bo.rhs());
}

std::ostream &operator<<(std::ostream &os, const binary_operator &bo)
{
    os << '(' << bo.lhs() << ' ';

    switch (bo.op()) {
        case binary_operator::type::add:
            os << '+';
            break;
        case binary_operator::type::sub:
            os << '-';
            break;
        case binary_operator::type::mul:
            os << '*';
            break;
        case binary_operator::type::div:
            os << '/';
            break;
    }

    return os << ' ' << bo.rhs() << ')';
}

std::vector<std::string> get_variables(const binary_operator &bo)
{
    auto lhs_vars = get_variables(bo.lhs());
    auto rhs_vars = get_variables(bo.rhs());

    lhs_vars.insert(lhs_vars.end(), std::make_move_iterator(rhs_vars.begin()), std::make_move_iterator(rhs_vars.end()));

    std::sort(lhs_vars.begin(), lhs_vars.end());
    lhs_vars.erase(std::unique(lhs_vars.begin(), lhs_vars.end()), lhs_vars.end());

    return lhs_vars;
}

void rename_variables(binary_operator &bo, const std::unordered_map<std::string, std::string> &repl_map)
{
    rename_variables(bo.lhs(), repl_map);
    rename_variables(bo.rhs(), repl_map);
}

bool operator==(const binary_operator &o1, const binary_operator &o2)
{
    return o1.op() == o2.op() && o1.lhs() == o2.lhs() && o1.rhs() == o2.rhs();
}

bool operator!=(const binary_operator &o1, const binary_operator &o2)
{
    return !(o1 == o2);
}

expression subs(const binary_operator &bo, const std::unordered_map<std::string, expression> &smap)
{
    return expression{binary_operator{bo.op(), subs(bo.lhs(), smap), subs(bo.rhs(), smap)}};
}

expression diff(const binary_operator &bo, const std::string &s)
{
    switch (bo.op()) {
        case binary_operator::type::add:
            return diff(bo.lhs(), s) + diff(bo.rhs(), s);
        case binary_operator::type::sub:
            return diff(bo.lhs(), s) - diff(bo.rhs(), s);
        case binary_operator::type::mul:
            return diff(bo.lhs(), s) * bo.rhs() + bo.lhs() * diff(bo.rhs(), s);
        default:
            return (diff(bo.lhs(), s) * bo.rhs() - bo.lhs() * diff(bo.rhs(), s)) / (bo.rhs() * bo.rhs());
    }
}

double eval_dbl(const binary_operator &bo, const std::unordered_map<std::string, double> &map,
                const std::vector<double> &pars)
{
    switch (bo.op()) {
        case binary_operator::type::add:
            return eval_dbl(bo.lhs(), map, pars) + eval_dbl(bo.rhs(), map, pars);
        case binary_operator::type::sub:
            return eval_dbl(bo.lhs(), map, pars) - eval_dbl(bo.rhs(), map, pars);
        case binary_operator::type::mul:
            return eval_dbl(bo.lhs(), map, pars) * eval_dbl(bo.rhs(), map, pars);
        default:
            return eval_dbl(bo.lhs(), map, pars) / eval_dbl(bo.rhs(), map, pars);
    }
}

void eval_batch_dbl(std::vector<double> &out_values, const binary_operator &bo,
                    const std::unordered_map<std::string, std::vector<double>> &map, const std::vector<double> &pars)
{
    auto tmp = out_values;
    eval_batch_dbl(out_values, bo.lhs(), map, pars);
    eval_batch_dbl(tmp, bo.rhs(), map, pars);
    switch (bo.op()) {
        case binary_operator::type::add:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::plus<>());
            break;
        case binary_operator::type::sub:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::minus<>());
            break;
        case binary_operator::type::mul:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::multiplies<>());
            break;
        default:
            std::transform(out_values.begin(), out_values.end(), tmp.begin(), out_values.begin(), std::divides<>());
            break;
    }
}

void update_node_values_dbl(std::vector<double> &node_values, const binary_operator &bo,
                            const std::unordered_map<std::string, double> &map,
                            const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter)
{
    const auto node_id = node_counter;
    node_counter++;
    // We have to recurse first as to make sure out is filled before being accessed later.
    update_node_values_dbl(node_values, bo.lhs(), map, node_connections, node_counter);
    update_node_values_dbl(node_values, bo.rhs(), map, node_connections, node_counter);
    switch (bo.op()) {
        case binary_operator::type::add:
            node_values[node_id]
                = node_values[node_connections[node_id][0]] + node_values[node_connections[node_id][1]];
            break;
        case binary_operator::type::sub:
            node_values[node_id]
                = node_values[node_connections[node_id][0]] - node_values[node_connections[node_id][1]];
            break;
        case binary_operator::type::mul:
            node_values[node_id]
                = node_values[node_connections[node_id][0]] * node_values[node_connections[node_id][1]];
            break;
        default:
            node_values[node_id]
                = node_values[node_connections[node_id][0]] / node_values[node_connections[node_id][1]];
            break;
    }
}

void update_grad_dbl(std::unordered_map<std::string, double> &grad, const binary_operator &bo,
                     const std::unordered_map<std::string, double> &map, const std::vector<double> &node_values,
                     const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter,
                     double acc)
{
    const auto node_id = node_counter;
    node_counter++;
    switch (bo.op()) {
        case binary_operator::type::add:
            // lhs (a + b -> 1)
            update_grad_dbl(grad, bo.lhs(), map, node_values, node_connections, node_counter, acc);
            // rhs (a + b -> 1)
            update_grad_dbl(grad, bo.rhs(), map, node_values, node_connections, node_counter, acc);
            break;

        case binary_operator::type::sub:
            // lhs (a + b -> 1)
            update_grad_dbl(grad, bo.lhs(), map, node_values, node_connections, node_counter, acc);
            // rhs (a + b -> 1)
            update_grad_dbl(grad, bo.rhs(), map, node_values, node_connections, node_counter, -acc);
            break;

        case binary_operator::type::mul:
            // lhs (a*b -> b)
            update_grad_dbl(grad, bo.lhs(), map, node_values, node_connections, node_counter,
                            acc * node_values[node_connections[node_id][1]]);
            // rhs (a*b -> a)
            update_grad_dbl(grad, bo.rhs(), map, node_values, node_connections, node_counter,
                            acc * node_values[node_connections[node_id][0]]);
            break;

        default:
            // lhs (a/b -> 1/b)
            update_grad_dbl(grad, bo.lhs(), map, node_values, node_connections, node_counter,
                            acc / node_values[node_connections[node_id][1]]);
            // rhs (a/b -> -a/b^2)
            update_grad_dbl(grad, bo.rhs(), map, node_values, node_connections, node_counter,
                            -acc * node_values[node_connections[node_id][0]] / node_values[node_connections[node_id][1]]
                                / node_values[node_connections[node_id][1]]);
            break;
    }
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const binary_operator &bo,
                        std::size_t &node_counter)
{
    const auto node_id = node_counter;
    node_counter++;
    node_connections.push_back(std::vector<std::size_t>(2));
    node_connections[node_id][0] = node_counter;
    update_connections(node_connections, bo.lhs(), node_counter);
    node_connections[node_id][1] = node_counter;
    update_connections(node_connections, bo.rhs(), node_counter);
}

std::vector<expression>::size_type taylor_decompose_in_place(binary_operator &&bo, std::vector<expression> &u_vars_defs)
{
    if (const auto dres_lhs = taylor_decompose_in_place(std::move(bo.lhs()), u_vars_defs)) {
        // The lhs required decomposition, and its decomposition
        // was placed at index dres_lhs in u_vars_defs. Replace the lhs
        // a u variable pointing at index dres_lhs.
        bo.lhs() = expression{variable{"u_" + detail::li_to_string(dres_lhs)}};
    }

    if (const auto dres_rhs = taylor_decompose_in_place(std::move(bo.rhs()), u_vars_defs)) {
        bo.rhs() = expression{variable{"u_" + detail::li_to_string(dres_rhs)}};
    }

    // Append the binary operator after decomposition
    // of lhs and rhs.
    u_vars_defs.emplace_back(std::move(bo));

    // The decomposition of binary operators
    // results in a new u variable, whose definition
    // we added to u_vars_defs above.
    return u_vars_defs.size() - 1u;
}

namespace detail
{

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
        auto n0 = taylor_codegen_constant<T>(s, num0, par_ptr, batch_size);
        auto n1 = taylor_codegen_constant<T>(s, num1, par_ptr, batch_size);

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
        auto n = taylor_codegen_constant<T>(s, num, par_ptr, batch_size);

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

        auto n = taylor_codegen_constant<T>(s, num, par_ptr, batch_size);

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
template <bool, typename, typename V1, typename V2,
          std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Value *bo_taylor_diff_addsub_impl(llvm_state &, const V1 &, const V2 &, const std::vector<llvm::Value *> &,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Value *bo_taylor_diff_add(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
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
llvm::Value *bo_taylor_diff_sub(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
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
template <typename T>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const number &num0, const number &num1,
                                     const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, (order == 0u) ? num0 * num1 : number{0.}), batch_size);
}

// Derivative of var * number.
template <typename T>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const variable &var, const number &num,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    auto ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);
    auto mul = vector_splat(builder, codegen<T>(s, num), batch_size);

    return builder.CreateFMul(mul, ret);
}

// Derivative of number * var.
template <typename T>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &s, const number &num, const variable &var,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
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
template <typename, typename V1, typename V2>
llvm::Value *bo_taylor_diff_mul_impl(llvm_state &, const V1 &, const V2 &, const std::vector<llvm::Value *> &,
                                     llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Value *bo_taylor_diff_mul(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
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
template <typename T>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &s, const number &num0, const number &num1,
                                     const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, order == 0u ? num0 / num1 : number{0.}), batch_size);
}

// Derivative of variable / variable or number / variable. These two cases
// are quite similar, so we handle them together.
template <typename T, typename U,
          std::enable_if_t<std::disjunction_v<std::is_same<U, number>, std::is_same<U, variable>>, int> = 0>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &s, const U &nv, const variable &var1,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Fetch the index of var1.
    const auto u_idx1 = uname_to_index(var1.name());

    if (order == 0u) {
        // Special casing for zero order.
        auto numerator = [&]() -> llvm::Value * {
            if constexpr (std::is_same_v<U, number>) {
                return vector_splat(builder, codegen<T>(s, nv), batch_size);
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

    if constexpr (std::is_same_v<U, number>) {
        // nv is a number. Negate the accumulator
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
template <typename T>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &s, const variable &var, const number &num,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    auto ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);
    auto div = vector_splat(builder, codegen<T>(s, num), batch_size);

    return builder.CreateFDiv(ret, div);
}

// All the other cases.
template <typename, typename V1, typename V2>
llvm::Value *bo_taylor_diff_div_impl(llvm_state &, const V1 &, const V2 &, const std::vector<llvm::Value *> &,
                                     llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Value *bo_taylor_diff_div(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
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
llvm::Value *taylor_diff_bo_impl(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
                                 llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                 std::uint32_t batch_size)
{
    // lhs and rhs must be u vars or numbers.
    auto check_arg = [](const expression &e) {
        std::visit(
            [](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    // The expression is a variable. Check that it
                    // is a u variable.
                    const auto &var_name = v.name();
                    if (var_name.rfind("u_", 0) != 0) {
                        throw std::invalid_argument(
                            "Invalid variable name '" + var_name
                            + "' encountered in the Taylor diff phase for a binary operator expression (the name "
                              "must be in the form 'u_n', where n is a non-negative integer)");
                    }
                } else if constexpr (!std::is_same_v<type, number> && !std::is_same_v<type, param>) {
                    // Not a variable and not a number.
                    throw std::invalid_argument(
                        "An invalid expression type was passed to the Taylor diff phase of a binary operator (the "
                        "expression must be either a variable or a number/param, but it is neither)");
                }
            },
            e.value());
    };

    check_arg(bo.lhs());
    check_arg(bo.rhs());

    switch (bo.op()) {
        case binary_operator::type::add:
            return bo_taylor_diff_add<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
        case binary_operator::type::sub:
            return bo_taylor_diff_sub<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
        case binary_operator::type::mul:
            return bo_taylor_diff_mul<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
        default:
            return bo_taylor_diff_div<T>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
    }
}

} // namespace

} // namespace detail

llvm::Value *taylor_diff_dbl(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
                             llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                             std::uint32_t batch_size)
{
    return detail::taylor_diff_bo_impl<double>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *taylor_diff_ldbl(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
                              llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                              std::uint32_t batch_size)
{
    return detail::taylor_diff_bo_impl<long double>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_diff_f128(llvm_state &s, const binary_operator &bo, const std::vector<llvm::Value *> &arr,
                              llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                              std::uint32_t batch_size)
{
    return detail::taylor_diff_bo_impl<mppp::real128>(s, bo, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

namespace detail
{

namespace
{

// Helper to implement the function for the differentiation of
// 'number op number' in compact mode. The function will always return zero,
// unless the order is 0 (in which case it will return the result of the codegen).
template <typename T>
llvm::Function *bo_taylor_c_diff_func_num_num(llvm_state &s, const binary_operator &bo, std::uint32_t batch_size,
                                              const std::string &fname, const std::string &op_name)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - number arguments.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), to_llvm_type<T>(context),
                                    to_llvm_type<T>(context)};

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
        auto num0 = f->args().begin() + 3;
        auto num1 = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, run the codegen.
                auto vnum0 = vector_splat(builder, num0, batch_size);
                auto vnum1 = vector_splat(builder, num1, batch_size);

                switch (bo.op()) {
                    case binary_operator::type::add:
                        builder.CreateStore(builder.CreateFAdd(vnum0, vnum1), retval);
                        break;
                    case binary_operator::type::sub:
                        builder.CreateStore(builder.CreateFSub(vnum0, vnum1), retval);
                        break;
                    case binary_operator::type::mul:
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
        builder.CreateRet(builder.CreateLoad(retval));

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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of " + op_name
                                        + " in compact mode detected");
        }
    }

    return f;
}

// Derivative of number +- number.
template <bool, typename T>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_operator &bo, const number &,
                                                  const number &, std::uint32_t, std::uint32_t batch_size)
{
    return bo_taylor_c_diff_func_num_num<T>(s, bo, batch_size,
                                            "heyoka_taylor_diff_addsub_num_num_"
                                                + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                            "addition");
}

// Derivative of number +- var.
template <bool AddOrSub, typename T>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_operator &, const number &,
                                                  const variable &, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname = std::string{"heyoka_taylor_diff_"} + (AddOrSub ? "add" : "sub") + "_num_var_"
                       + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - number argument,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), to_llvm_type<T>(context),
                                    llvm::Type::getInt32Ty(context)};

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
        auto num = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(0)),
            [&]() {
                // For order zero, run the codegen.
                auto num_vec = vector_splat(builder, num, batch_size);
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
        builder.CreateRet(builder.CreateLoad(retval));

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
template <bool AddOrSub, typename T>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_operator &, const variable &,
                                                  const number &, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_addsub_var_num_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument,
    // - number argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    to_llvm_type<T>(context)};

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
        auto var_idx = f->args().begin() + 3;
        auto num = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(0)),
            [&]() {
                // For order zero, run the codegen.
                auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), var_idx);
                auto num_vec = vector_splat(builder, num, batch_size);

                builder.CreateStore(AddOrSub ? builder.CreateFAdd(ret, num_vec) : builder.CreateFSub(ret, num_vec),
                                    retval);
            },
            [&]() {
                // Create the return value.
                builder.CreateStore(taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &s, const binary_operator &, const variable &,
                                                  const variable &, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname = std::string{"heyoka_taylor_diff_"} + (AddOrSub ? "add" : "sub") + "_var_var_"
                       + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the first var argument,
    // - idx of the second var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    llvm::Type::getInt32Ty(context)};

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
        auto var_idx0 = f->args().begin() + 3;
        auto var_idx1 = f->args().begin() + 4;

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
template <bool, typename, typename V1, typename V2>
llvm::Function *bo_taylor_c_diff_func_addsub_impl(llvm_state &, const binary_operator &, const V1 &, const V2 &,
                                                  std::uint32_t, std::uint32_t)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Function *bo_taylor_c_diff_func_add(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_addsub_impl<true, T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Function *bo_taylor_c_diff_func_sub(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_addsub_impl<false, T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number * number.
template <typename T>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_operator &bo, const number &, const number &,
                                               std::uint32_t, std::uint32_t batch_size)
{
    return bo_taylor_c_diff_func_num_num<T>(s, bo, batch_size,
                                            "heyoka_taylor_diff_mul_num_num_"
                                                + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                            "multiplication");
}

// Derivative of var * number.
template <typename T>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_operator &, const variable &, const number &,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_mul_var_num_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument,
    // - number argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    to_llvm_type<T>(context)};

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
        auto var_idx = f->args().begin() + 3;
        auto num = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(builder.CreateFMul(ret, vector_splat(builder, num, batch_size)));

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
template <typename T>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_operator &, const number &, const variable &,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_mul_num_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - number argument,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), to_llvm_type<T>(context),
                                    llvm::Type::getInt32Ty(context)};

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
        auto num = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(builder.CreateFMul(ret, vector_splat(builder, num, batch_size)));

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
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &s, const binary_operator &, const variable &,
                                               const variable &, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_mul_var_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the first var argument,
    // - idx of the second var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    llvm::Type::getInt32Ty(context)};

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
        auto idx0 = f->args().begin() + 3;
        auto idx1 = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(0), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx0);
            auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx1);
            builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(b_nj, cj)), acc);
        });

        // Create the return value.
        builder.CreateRet(builder.CreateLoad(acc));

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
template <typename, typename V1, typename V2>
llvm::Function *bo_taylor_c_diff_func_mul_impl(llvm_state &, const binary_operator &, const V1 &, const V2 &,
                                               std::uint32_t, std::uint32_t)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Function *bo_taylor_c_diff_func_mul(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_mul_impl<T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number / number.
template <typename T>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_operator &bo, const number &, const number &,
                                               std::uint32_t, std::uint32_t batch_size)
{
    return bo_taylor_c_diff_func_num_num<T>(s, bo, batch_size,
                                            "heyoka_taylor_diff_div_num_num_"
                                                + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                            "division");
}

// Derivative of var / number.
template <typename T>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_operator &, const variable &, const number &,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_div_var_num_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument,
    // - number argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    to_llvm_type<T>(context)};

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
        auto var_idx = f->args().begin() + 3;
        auto num = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(builder.CreateFDiv(ret, vector_splat(builder, num, batch_size)));

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
template <typename T>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_operator &, const number &, const variable &,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_div_num_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - number argument,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), to_llvm_type<T>(context),
                                    llvm::Type::getInt32Ty(context)};

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
        auto num = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 4;

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
                auto num_vec = vector_splat(builder, num, batch_size);
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
                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(cj, a_nj)), acc);
                });

                // Negate the loop summation.
                auto ret = builder.CreateFNeg(builder.CreateLoad(acc));

                // Divide and return.
                builder.CreateStore(
                    builder.CreateFDiv(ret, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &s, const binary_operator &, const variable &,
                                               const variable &, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_div_var_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the first var argument,
    // - idx of the second var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    llvm::Type::getInt32Ty(context)};

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
        auto var_idx0 = f->args().begin() + 3;
        auto var_idx1 = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx1);
            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
            builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(cj, a_nj)), acc);
        });

        auto ret = builder.CreateFSub(taylor_c_load_diff(s, diff_ptr, n_uvars, ord, var_idx0), builder.CreateLoad(acc));

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
template <typename, typename V1, typename V2>
llvm::Function *bo_taylor_c_diff_func_div_impl(llvm_state &, const binary_operator &, const V1 &, const V2 &,
                                               std::uint32_t, std::uint32_t)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Function *bo_taylor_c_diff_func_div(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return bo_taylor_c_diff_func_div_impl<T>(s, bo, v1, v2, n_uvars, batch_size);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Function *taylor_c_diff_func_bo_impl(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                           std::uint32_t batch_size)
{
    // lhs and rhs must be u vars or numbers.
    auto check_arg = [](const expression &e) {
        std::visit(
            [](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    // The expression is a variable. Check that it
                    // is a u variable.
                    const auto &var_name = v.name();
                    if (var_name.rfind("u_", 0) != 0) {
                        throw std::invalid_argument("Invalid variable name '" + var_name
                                                    + "' encountered in the Taylor diff phase for a binary operator "
                                                      "expression in compact mode (the name "
                                                      "must be in the form 'u_n', where n is a non-negative integer)");
                    }
                } else if constexpr (!std::is_same_v<type, number>) {
                    // Not a variable and not a number.
                    throw std::invalid_argument("An invalid expression type was passed to the Taylor diff phase of a "
                                                "binary operator in compact mode (the "
                                                "expression must be either a variable or a number, but it is neither)");
                }
            },
            e.value());
    };

    check_arg(bo.lhs());
    check_arg(bo.rhs());

    switch (bo.op()) {
        case binary_operator::type::add:
            return bo_taylor_c_diff_func_add<T>(s, bo, n_uvars, batch_size);
        case binary_operator::type::sub:
            return bo_taylor_c_diff_func_sub<T>(s, bo, n_uvars, batch_size);
        case binary_operator::type::mul:
            return bo_taylor_c_diff_func_mul<T>(s, bo, n_uvars, batch_size);
        default:
            return bo_taylor_c_diff_func_div<T>(s, bo, n_uvars, batch_size);
    }
}

} // namespace

} // namespace detail

llvm::Function *taylor_c_diff_func_dbl(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    return detail::taylor_c_diff_func_bo_impl<double>(s, bo, n_uvars, batch_size);
}

llvm::Function *taylor_c_diff_func_ldbl(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    return detail::taylor_c_diff_func_bo_impl<long double>(s, bo, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *taylor_c_diff_func_f128(llvm_state &s, const binary_operator &bo, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    return detail::taylor_c_diff_func_bo_impl<mppp::real128>(s, bo, n_uvars, batch_size);
}

#endif

} // namespace heyoka
