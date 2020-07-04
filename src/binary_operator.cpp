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
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/assert_nonnull_ret.hpp>
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

namespace detail
{

namespace
{

// Common boilerplate for the implementation of
// the Taylor derivatives of binary operators.
template <typename T>
auto bo_taylor_diff_common(llvm_state &s, const std::string &name)
{
    auto &builder = s.builder();

    // Check the function name.
    if (s.module().getFunction(name) != nullptr) {
        throw std::invalid_argument("Cannot add the function '" + name
                                    + "' when building the Taylor derivative of a binary operator expression: the "
                                      "function already exists in the LLVM module");
    }

    // Prepare the function prototype. The arguments are:
    // - const float pointer to the derivatives array,
    // - 32-bit integer (order of the derivative).
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())), builder.getInt32Ty()};

    // The function will return the n-th derivative as a float.
    auto *ft = llvm::FunctionType::get(to_llvm_type<T>(s.context()), fargs, false);
    assert(ft != nullptr);

    // Now create the function. Don't need to call it from outside,
    // thus internal linkage.
    auto *f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, name, s.module());
    assert(f != nullptr);

    // Setup the function arugments.
    auto arg_it = f->args().begin();
    arg_it->setName("diff_ptr");
    arg_it->addAttr(llvm::Attribute::ReadOnly);
    arg_it->addAttr(llvm::Attribute::NoCapture);
    auto diff_ptr = arg_it;

    (++arg_it)->setName("order");
    auto order = arg_it;

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    return std::tuple{f, diff_ptr, order};
}

// Derivative of number +- number.
template <bool, typename T>
llvm::Function *bo_taylor_diff_addsub_impl(llvm_state &s, const number &, const number &, const std::string &name,
                                           std::uint32_t, const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // The derivative of a constant is always zero.
    builder.CreateRet(invoke_codegen<T>(s, number(0.)));

    s.verify_function(name);

    // NOTE: no need to add any of these functions
    // to s.m_sig_map, as they are all internal
    // functions.

    return f;
}

// Derivative of number +- var.
template <bool AddOrSub, typename T>
llvm::Function *bo_taylor_diff_addsub_impl(llvm_state &s, const number &, const variable &var, const std::string &name,
                                           std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // Compute the index for accessing the derivative. The index is order * n_uvars + u_idx.
    const auto u_idx = uname_to_index(var.name());
    auto arr_idx = builder.CreateAdd(builder.CreateMul(builder.getInt32(n_uvars), order), builder.getInt32(u_idx));
    // Convert into a pointer.
    auto arr_ptr = builder.CreateInBoundsGEP(diff_ptr, arr_idx, "diff_ptr");
    // Load the value.
    auto ret = builder.CreateLoad(arr_ptr, "diff_load");

    if constexpr (AddOrSub) {
        builder.CreateRet(ret);
    } else {
        // Negate if we are doing a subtraction.
        builder.CreateRet(builder.CreateFNeg(ret));
    }

    s.verify_function(name);

    return f;
}

// Derivative of var +- number.
template <bool, typename T>
llvm::Function *bo_taylor_diff_addsub_impl(llvm_state &s, const variable &var, const number &, const std::string &name,
                                           std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // Compute the index for accessing the derivative. The index is order * n_uvars + u_idx.
    const auto u_idx = uname_to_index(var.name());
    auto arr_idx = builder.CreateAdd(builder.CreateMul(builder.getInt32(n_uvars), order), builder.getInt32(u_idx));
    // Convert into a pointer.
    auto arr_ptr = builder.CreateInBoundsGEP(diff_ptr, arr_idx, "diff_ptr");
    // Load the value.
    auto ret = builder.CreateLoad(arr_ptr, "diff_load");

    // NOTE: does not matter here plus or minus.
    builder.CreateRet(ret);

    s.verify_function(name);

    return f;
}

// Derivative of var +- var.
template <bool AddOrSub, typename T>
llvm::Function *bo_taylor_diff_addsub_impl(llvm_state &s, const variable &var0, const variable &var1,
                                           const std::string &name, std::uint32_t n_uvars,
                                           const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // Compute the indices for accessing the derivatives. The indices are order * n_uvars + (idx0, idx1).
    const auto u_idx0 = uname_to_index(var0.name());
    const auto u_idx1 = uname_to_index(var1.name());
    auto arr_idx0 = builder.CreateAdd(builder.CreateMul(builder.getInt32(n_uvars), order), builder.getInt32(u_idx0));
    auto arr_idx1 = builder.CreateAdd(builder.CreateMul(builder.getInt32(n_uvars), order), builder.getInt32(u_idx1));
    // Convert into pointers.
    auto arr_ptr0 = builder.CreateInBoundsGEP(diff_ptr, arr_idx0, "diff_ptr0");
    auto arr_ptr1 = builder.CreateInBoundsGEP(diff_ptr, arr_idx1, "diff_ptr1");
    // Load the values.
    auto v0 = builder.CreateLoad(arr_ptr0, "diff_load0");
    auto v1 = builder.CreateLoad(arr_ptr1, "diff_load1");

    if constexpr (AddOrSub) {
        builder.CreateRet(builder.CreateFAdd(v0, v1));
    } else {
        builder.CreateRet(builder.CreateFSub(v0, v1));
    }

    s.verify_function(name);

    return f;
}

// All the other cases. We should never end up here.
template <bool, typename, typename V1, typename V2>
llvm::Function *bo_taylor_diff_addsub_impl(llvm_state &, const V1 &, const V2 &, const std::string &, std::uint32_t,
                                           const std::unordered_map<std::uint32_t, number> &)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Function *bo_taylor_diff_add(llvm_state &s, const binary_operator &bo, std::uint32_t, const std::string &name,
                                   std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return std::visit(
        [&s, &name, n_uvars, &cd_uvars](const auto &v1, const auto &v2) {
            return bo_taylor_diff_addsub_impl<true, T>(s, v1, v2, name, n_uvars, cd_uvars);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Function *bo_taylor_diff_sub(llvm_state &s, const binary_operator &bo, std::uint32_t, const std::string &name,
                                   std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return std::visit(
        [&s, &name, n_uvars, &cd_uvars](const auto &v1, const auto &v2) {
            return bo_taylor_diff_addsub_impl<false, T>(s, v1, v2, name, n_uvars, cd_uvars);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number * number.
template <typename T>
llvm::Function *bo_taylor_diff_mul_impl(llvm_state &s, const number &, const number &, const std::string &name,
                                        std::uint32_t, const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // The derivative of a constant is always zero.
    builder.CreateRet(invoke_codegen<T>(s, number(0.)));

    s.verify_function(name);

    return f;
}

// Derivative of number * var.
template <typename T>
llvm::Function *bo_taylor_diff_mul_impl(llvm_state &s, const number &num, const variable &var, const std::string &name,
                                        std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // Compute the index for accessing the derivative. The index is order * n_uvars + u_idx.
    const auto u_idx = uname_to_index(var.name());
    auto arr_idx = builder.CreateAdd(builder.CreateMul(builder.getInt32(n_uvars), order), builder.getInt32(u_idx));
    // Convert into a pointer.
    auto arr_ptr = builder.CreateInBoundsGEP(diff_ptr, arr_idx, "diff_ptr");
    // Load the value.
    auto ret = builder.CreateLoad(arr_ptr, "diff_load");

    builder.CreateRet(builder.CreateFMul(invoke_codegen<T>(s, num), ret));

    s.verify_function(name);

    return f;
}

// Derivative of var * number.
template <typename T>
llvm::Function *bo_taylor_diff_mul_impl(llvm_state &s, const variable &var, const number &num, const std::string &name,
                                        std::uint32_t n_uvars,
                                        const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return bo_taylor_diff_mul_impl<T>(s, num, var, name, n_uvars, cd_uvars);
}

// Derivative of var * var.
template <typename T>
llvm::Function *bo_taylor_diff_mul_impl(llvm_state &s, const variable &var0, const variable &var1,
                                        const std::string &name, std::uint32_t n_uvars,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // Accumulator for the result.
    auto ret_acc = builder.CreateAlloca(to_llvm_type<T>(s.context()), 0, "ret_acc");
    builder.CreateStore(invoke_codegen<T>(s, number(0.)), ret_acc);

    // Initial value for the for-loop. We will be operating
    // in the range [0, order] (i.e., order inclusive).
    auto start_val = builder.getInt32(0);

    // Make the new basic block for the loop header,
    // inserting after current block.
    auto *preheader_bb = builder.GetInsertBlock();
    auto *loop_bb = llvm::BasicBlock::Create(s.context(), "loop", f);

    // Insert an explicit fall through from the current block to the loop_bb.
    builder.CreateBr(loop_bb);

    // Start insertion in loop_bb.
    builder.SetInsertPoint(loop_bb);

    // Start the PHI node with an entry for Start.
    auto *j_var = builder.CreatePHI(builder.getInt32Ty(), 2, "j");
    j_var->addIncoming(start_val, preheader_bb);

    // Loop body.
    // Compute the indices for accessing the derivatives in this loop iteration.
    // The indices are:
    // - (order - j_var) * n_uvars + u_idx0,
    // - j_var * n_uvars + u_idx1.
    const auto u_idx0 = uname_to_index(var0.name());
    const auto u_idx1 = uname_to_index(var1.name());
    auto arr_idx0 = builder.CreateAdd(builder.CreateMul(builder.CreateSub(order, j_var), builder.getInt32(n_uvars)),
                                      builder.getInt32(u_idx0));
    auto arr_idx1 = builder.CreateAdd(builder.CreateMul(j_var, builder.getInt32(n_uvars)), builder.getInt32(u_idx1));
    // Convert into pointers.
    auto arr_ptr0 = builder.CreateInBoundsGEP(diff_ptr, arr_idx0, "diff_ptr0");
    auto arr_ptr1 = builder.CreateInBoundsGEP(diff_ptr, arr_idx1, "diff_ptr1");
    // Load the values.
    auto v0 = builder.CreateLoad(arr_ptr0, "diff_load0");
    auto v1 = builder.CreateLoad(arr_ptr1, "diff_load1");
    // Update ret_acc: ret_acc = ret_acc + v0*v1.
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ret_acc), builder.CreateFMul(v0, v1)), ret_acc);

    // Compute the next value of the iteration.
    // NOTE: addition works regardless of integral signedness.
    auto *next_j_var = builder.CreateAdd(j_var, builder.getInt32(1), "next_j");

    // Compute the end condition.
    // NOTE: we use the unsigned less-than-or-equal predicate.
    auto *end_cond = builder.CreateICmp(llvm::CmpInst::ICMP_ULE, next_j_var, order, "loopcond");

    // Create the "after loop" block and insert it.
    auto *loop_end_bb = builder.GetInsertBlock();
    auto *after_bb = llvm::BasicBlock::Create(s.context(), "afterloop", f);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(end_cond, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    j_var->addIncoming(next_j_var, loop_end_bb);

    builder.CreateRet(builder.CreateLoad(ret_acc));

    s.verify_function(name);

    return f;
}

// All the other cases. We should never end up here.
template <typename, typename V1, typename V2>
llvm::Function *bo_taylor_diff_mul_impl(llvm_state &, const V1 &, const V2 &, const std::string &, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Function *bo_taylor_diff_mul(llvm_state &s, const binary_operator &bo, std::uint32_t, const std::string &name,
                                   std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return std::visit(
        [&s, &name, n_uvars, &cd_uvars](const auto &v1, const auto &v2) {
            return bo_taylor_diff_mul_impl<T>(s, v1, v2, name, n_uvars, cd_uvars);
        },
        bo.lhs().value(), bo.rhs().value());
}

// Derivative of number / number.
template <typename T>
llvm::Function *bo_taylor_diff_div_impl(llvm_state &s, std::uint32_t, const number &, const number &,
                                        const std::string &name, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // The derivative of a constant is always zero.
    builder.CreateRet(invoke_codegen<T>(s, number(0.)));

    s.verify_function(name);

    return f;
}

// Derivative of variable / variable or number / variable. These two cases
// are quite similar, so we handle them together.
template <typename T, typename U,
          std::enable_if_t<std::disjunction_v<std::is_same<U, number>, std::is_same<U, variable>>, int> = 0>
llvm::Function *bo_taylor_diff_div_impl(llvm_state &s, std::uint32_t idx, const U &nv, const variable &var1,
                                        const std::string &name, std::uint32_t n_uvars,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    auto &builder = s.builder();

    auto [f, diff_ptr, order] = bo_taylor_diff_common<T>(s, name);

    // Let's build the result of the summation first.
    // Accumulator for the result.
    auto ret_acc = builder.CreateAlloca(to_llvm_type<T>(s.context()), 0, "sum_acc");
    builder.CreateStore(invoke_codegen<T>(s, number(0.)), ret_acc);

    // Initial value for the for-loop. We will be operating
    // in the range [1, order] (i.e., order inclusive).
    auto start_val = builder.getInt32(1);

    // Make the new basic block for the loop header,
    // inserting after current block.
    auto *preheader_bb = builder.GetInsertBlock();
    auto *loop_bb = llvm::BasicBlock::Create(s.context(), "loop", f);

    // Insert an explicit fall through from the current block to the loop_bb.
    builder.CreateBr(loop_bb);

    // Start insertion in loop_bb.
    builder.SetInsertPoint(loop_bb);

    // Start the PHI node with an entry for Start.
    auto *j_var = builder.CreatePHI(builder.getInt32Ty(), 2, "j");
    j_var->addIncoming(start_val, preheader_bb);

    // Loop body.
    // Compute the indices for accessing the derivatives in this loop iteration.
    // The indices are:
    // - (order - j_var) * n_uvars + idx,
    // - j_var * n_uvars + u_idx1.
    const auto u_idx1 = uname_to_index(var1.name());
    auto arr_idx0 = builder.CreateAdd(builder.CreateMul(builder.CreateSub(order, j_var), builder.getInt32(n_uvars)),
                                      builder.getInt32(idx));
    auto arr_idx1 = builder.CreateAdd(builder.CreateMul(j_var, builder.getInt32(n_uvars)), builder.getInt32(u_idx1));
    // Convert into pointers.
    auto arr_ptr0 = builder.CreateInBoundsGEP(diff_ptr, arr_idx0, "diff_ptr0");
    auto arr_ptr1 = builder.CreateInBoundsGEP(diff_ptr, arr_idx1, "diff_ptr1");
    // Load the values.
    auto v0 = builder.CreateLoad(arr_ptr0, "diff_load0");
    auto v1 = builder.CreateLoad(arr_ptr1, "diff_load1");
    // Update ret_acc: ret_acc = ret_acc + v0*v1.
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ret_acc), builder.CreateFMul(v0, v1)), ret_acc);

    // Compute the next value of the iteration.
    // NOTE: addition works regardless of integral signedness.
    auto *next_j_var = builder.CreateAdd(j_var, builder.getInt32(1), "next_j");

    // Compute the end condition.
    // NOTE: we use the unsigned less-than-or-equal predicate.
    auto *end_cond = builder.CreateICmp(llvm::CmpInst::ICMP_ULE, next_j_var, order, "loopcond");

    // Create the "after loop" block and insert it.
    auto *loop_end_bb = builder.GetInsertBlock();
    auto *after_bb = llvm::BasicBlock::Create(s.context(), "afterloop", f);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(end_cond, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    j_var->addIncoming(next_j_var, loop_end_bb);

    // Load the divisor for the quotient formula.
    // This is the zero-th order derivative of var1.
    // The index is thus just u_idx1.
    auto div_ptr = builder.CreateInBoundsGEP(diff_ptr, builder.getInt32(u_idx1), "div_ptr");
    auto div = builder.CreateLoad(div_ptr, "div");

    if constexpr (std::is_same_v<U, number>) {
        // nv is a number. Negate the accumulator
        // and divide it by the divisor.
        builder.CreateRet(builder.CreateFDiv(builder.CreateFNeg(builder.CreateLoad(ret_acc)), div));
    } else {
        // nv is a variable. We need to fetch its
        // derivative of order 'order' from the array of derivatives.
        // The index will be order * n_uvars + u_idx0.
        const auto u_idx0 = uname_to_index(nv.name());
        arr_idx0 = builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), builder.getInt32(u_idx0));
        arr_ptr0 = builder.CreateInBoundsGEP(diff_ptr, arr_idx0, "diff_nv_ptr");
        auto diff_nv_v = builder.CreateLoad(arr_ptr0, "diff_nv");

        // Produce the result: (diff_nv_v - ret_acc) / div.
        builder.CreateRet(builder.CreateFDiv(builder.CreateFSub(diff_nv_v, builder.CreateLoad(ret_acc)), div));
    }

    s.verify_function(name);

    return f;
}

// Derivative of var / number.
template <typename T>
llvm::Function *bo_taylor_diff_div_impl(llvm_state &s, std::uint32_t, const variable &var, const number &num,
                                        const std::string &name, std::uint32_t n_uvars,
                                        const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    // NOTE: implement as the derivative of var * (1 / number).
    return bo_taylor_diff_mul_impl<T>(s, var, number(1.) / num, name, n_uvars, cd_uvars);
}

// All the other cases. We should never end up here.
template <typename, typename V1, typename V2>
llvm::Function *bo_taylor_diff_div_impl(llvm_state &, std::uint32_t, const V1 &, const V2 &, const std::string &,
                                        std::uint32_t, const std::unordered_map<std::uint32_t, number> &)
{
    assert(false);

    return nullptr;
}

template <typename T>
llvm::Function *bo_taylor_diff_div(llvm_state &s, const binary_operator &bo, std::uint32_t idx, const std::string &name,
                                   std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return std::visit(
        [&s, idx, &name, n_uvars, &cd_uvars](const auto &v1, const auto &v2) {
            return bo_taylor_diff_div_impl<T>(s, idx, v1, v2, name, n_uvars, cd_uvars);
        },
        bo.lhs().value(), bo.rhs().value());
}

template <typename T>
llvm::Function *taylor_diff_bo_impl(llvm_state &s, const binary_operator &bo, std::uint32_t idx,
                                    const std::string &name, std::uint32_t n_uvars,
                                    const std::unordered_map<std::uint32_t, number> &cd_uvars)
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
                } else if constexpr (!std::is_same_v<type, number>) {
                    // Not a variable and not a number.
                    throw std::invalid_argument(
                        "An invalid expression type was passed to the Taylor diff phase of a binary operator (the "
                        "expression must be either a variable or a number, but it is neither)");
                }
            },
            e.value());
    };

    check_arg(bo.lhs());
    check_arg(bo.rhs());

    switch (bo.op()) {
        case binary_operator::type::add:
            return bo_taylor_diff_add<T>(s, bo, idx, name, n_uvars, cd_uvars);
        case binary_operator::type::sub:
            return bo_taylor_diff_sub<T>(s, bo, idx, name, n_uvars, cd_uvars);
        case binary_operator::type::mul:
            return bo_taylor_diff_mul<T>(s, bo, idx, name, n_uvars, cd_uvars);
        default:
            return bo_taylor_diff_div<T>(s, bo, idx, name, n_uvars, cd_uvars);
    }
}

} // namespace

} // namespace detail

llvm::Function *taylor_diff_dbl(llvm_state &s, const binary_operator &bo, std::uint32_t idx, const std::string &name,
                                std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return detail::taylor_diff_bo_impl<double>(s, bo, idx, name, n_uvars, cd_uvars);
}

llvm::Function *taylor_diff_ldbl(llvm_state &s, const binary_operator &bo, std::uint32_t idx, const std::string &name,
                                 std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    return detail::taylor_diff_bo_impl<long double>(s, bo, idx, name, n_uvars, cd_uvars);
}

} // namespace heyoka
