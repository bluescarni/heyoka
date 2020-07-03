// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/assert_nonnull_ret.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

binary_operator::binary_operator(type t, expression e1, expression e2)
    : m_type(t),
      // NOTE: need to use naked new as make_unique won't work with aggregate
      // initialization.
      m_ops(::new std::array<expression, 2>{std::move(e1), std::move(e2)})
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

double eval_dbl(const binary_operator &bo, const std::unordered_map<std::string, double> &map)
{
    switch (bo.op()) {
        case binary_operator::type::add:
            return eval_dbl(bo.lhs(), map) + eval_dbl(bo.rhs(), map);
        case binary_operator::type::sub:
            return eval_dbl(bo.lhs(), map) - eval_dbl(bo.rhs(), map);
        case binary_operator::type::mul:
            return eval_dbl(bo.lhs(), map) * eval_dbl(bo.rhs(), map);
        default:
            return eval_dbl(bo.lhs(), map) / eval_dbl(bo.rhs(), map);
    }
}

void eval_batch_dbl(std::vector<double> &out_values, const binary_operator &bo,
                    const std::unordered_map<std::string, std::vector<double>> &map)
{
    auto tmp = out_values;
    eval_batch_dbl(out_values, bo.lhs(), map);
    eval_batch_dbl(tmp, bo.rhs(), map);
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

namespace detail
{

namespace
{

template <typename T>
llvm::Value *bo_codegen_impl(llvm_state &s, const binary_operator &bo)
{
    auto *l = invoke_codegen<T>(s, bo.lhs());
    auto *r = invoke_codegen<T>(s, bo.rhs());

    auto &builder = s.builder();

    switch (bo.op()) {
        case binary_operator::type::add:
            return builder.CreateFAdd(l, r, "addtmp");
        case binary_operator::type::sub:
            return builder.CreateFSub(l, r, "subtmp");
        case binary_operator::type::mul:
            return builder.CreateFMul(l, r, "multmp");
        default:
            return builder.CreateFDiv(l, r, "divtmp");
    }
}

} // namespace

} // namespace detail

llvm::Value *codegen_dbl(llvm_state &s, const binary_operator &bo)
{
    heyoka_assert_nonnull_ret(detail::bo_codegen_impl<double>(s, bo));
}

llvm::Value *codegen_ldbl(llvm_state &s, const binary_operator &bo)
{
    heyoka_assert_nonnull_ret(detail::bo_codegen_impl<long double>(s, bo));
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

template <typename T>
llvm::Value *taylor_init_impl(llvm_state &s, const binary_operator &bo, llvm::Value *arr)
{
    auto &builder = s.builder();

    // Do the Taylor init for lhs and rhs.
    auto l = invoke_taylor_init<T>(s, bo.lhs(), arr);
    auto r = invoke_taylor_init<T>(s, bo.rhs(), arr);

    // Do the codegen for the corresponding operation.
    switch (bo.op()) {
        case binary_operator::type::add:
            return builder.CreateFAdd(l, r, "taylor_init_add");
        case binary_operator::type::sub:
            return builder.CreateFSub(l, r, "taylor_init_sub");
        case binary_operator::type::mul:
            return builder.CreateFMul(l, r, "taylor_init_mul");
        default:
            return builder.CreateFDiv(l, r, "taylor_init_div");
    }
}

} // namespace

} // namespace detail

llvm::Value *taylor_init_dbl(llvm_state &s, const binary_operator &bo, llvm::Value *arr)
{
    heyoka_assert_nonnull_ret(detail::taylor_init_impl<double>(s, bo, arr));
}

llvm::Value *taylor_init_ldbl(llvm_state &s, const binary_operator &bo, llvm::Value *arr)
{
    heyoka_assert_nonnull_ret(detail::taylor_init_impl<long double>(s, bo, arr));
}

} // namespace heyoka
