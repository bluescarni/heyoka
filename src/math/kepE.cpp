// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

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
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

kepE_impl::kepE_impl() : kepE_impl(0_dbl, 0_dbl) {}

kepE_impl::kepE_impl(expression e, expression M) : func_base("kepE", std::vector{std::move(e), std::move(M)}) {}

expression kepE_impl::diff(std::unordered_map<const void *, expression> &func_map, const std::string &s) const
{
    assert(args().size() == 2u);

    const auto &e = args()[0];
    const auto &M = args()[1];

    expression E{func{*this}};

    return (detail::diff(func_map, e, s) * sin(E) + detail::diff(func_map, M, s)) / (1_dbl - e * cos(E));
}

expression kepE_impl::diff(std::unordered_map<const void *, expression> &func_map, const param &p) const
{
    assert(args().size() == 2u);

    const auto &e = args()[0];
    const auto &M = args()[1];

    expression E{func{*this}};

    return (detail::diff(func_map, e, p) * sin(E) + detail::diff(func_map, M, p)) / (1_dbl - e * cos(E));
}

namespace
{

llvm::Value *kepE_llvm_eval_impl(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                 const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr, llvm::Value *stride,
                                 std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_eval_helper(
        [&s, fp_t, batch_size](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepE_func = llvm_add_inv_kep_E(s, fp_t, batch_size);

            return s.builder().CreateCall(kepE_func, {args[0], args[1]});
        },
        fb, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

} // namespace

llvm::Value *kepE_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                  llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                  bool high_accuracy) const
{
    return kepE_llvm_eval_impl(s, fp_t, *this, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *kepE_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                               std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "kepE",
        [&s, batch_size, fp_t](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepE_func = llvm_add_inv_kep_E(s, fp_t, batch_size);

            return s.builder().CreateCall(kepE_func, {args[0], args[1]});
        },
        fb, s, fp_t, batch_size, high_accuracy);
}

} // namespace

llvm::Function *kepE_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    return kepE_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

taylor_dc_t::size_type kepE_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 2u);

    // Make a copy of e.
    // NOTE: the arguments here have already been decomposed, thus
    // args()[0] is a non-function value that will be deep-copied.
    assert(!std::holds_alternative<func>(args()[0].value()));
    auto e_copy = args()[0];

    // Append the kepE decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Append the sin(a)/cos(a) decompositions.
    u_vars_defs.emplace_back(sin(expression{variable{fmt::format("u_{}", u_vars_defs.size() - 1u)}}),
                             std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(cos(expression{variable{fmt::format("u_{}", u_vars_defs.size() - 2u)}}),
                             std::vector<std::uint32_t>{});

    // Append the e*cos(a) decomposition.
    // NOTE: use mul() instead of * in order to avoid the automatic simplification
    // of 0 * cos(a) -> 0, which would result in an invalid entry in the Taylor decomposition
    // (i.e., a number entry).
    u_vars_defs.emplace_back(mul(std::move(e_copy), expression{variable{fmt::format("u_{}", u_vars_defs.size() - 1u)}}),
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

// Derivative of kepE(number, number).
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &, const U &num0,
                                   const V &num1, const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                   std::uint32_t, std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (order == 0u) {
        // Do the number codegen.
        auto e = taylor_codegen_numparam(s, fp_t, num0, par_ptr, batch_size);
        auto M = taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size);

        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {e, M});
    } else {
        return vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of kepE(var, number).
template <typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const variable &var, const U &num, const std::vector<llvm::Value *> &arr,
                                   llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    assert(deps.size() == 2u);

    auto &builder = s.builder();

    // Fetch the index of the e variable argument.
    const auto e_idx = uname_to_index(var.name());

    // Do the codegen for the M number argument.
    auto M = taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size);

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {taylor_fetch_diff(arr, e_idx, 0, n_uvars), M});
    }

    // Splat the order.
    auto *n = vector_splat(builder, llvm_codegen(s, fp_t, number{static_cast<double>(order)}), batch_size);

    // Compute the divisor: n * (1 - c^[0]).
    const auto c_idx = deps[0];
    auto *one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
    auto *divisor = llvm_fmul(s, n, llvm_fsub(s, one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars)));

    // Compute the first part of the dividend: n * e^[n] * d^[0] (the derivative of M is zero because
    // here M is a constant and the order is > 0).
    const auto d_idx = deps[1];
    auto *dividend = llvm_fmul(
        s, n, llvm_fmul(s, taylor_fetch_diff(arr, e_idx, order, n_uvars), taylor_fetch_diff(arr, d_idx, 0, n_uvars)));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

            auto *cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto *aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto *dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto *ej = taylor_fetch_diff(arr, e_idx, j, n_uvars);

            auto *tmp = llvm_fmul(s, dnj, ej);
            tmp = llvm_fadd(s, llvm_fmul(s, cnj, aj), tmp);
            sum.push_back(llvm_fmul(s, fac, tmp));
        }

        // Update the dividend.
        dividend = llvm_fadd(s, dividend, pairwise_sum(s, sum));
    }

    return llvm_fdiv(s, dividend, divisor);
}

// Derivative of kepE(number, var).
template <typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const U &num, const variable &var, const std::vector<llvm::Value *> &arr,
                                   llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    assert(deps.size() == 2u);

    auto &builder = s.builder();

    // Fetch the index of the M variable argument.
    const auto M_idx = uname_to_index(var.name());

    // Do the codegen for the e number argument.
    auto e = taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size);

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {e, taylor_fetch_diff(arr, M_idx, 0, n_uvars)});
    }

    // Splat the order.
    auto *n = vector_splat(builder, llvm_codegen(s, fp_t, number{static_cast<double>(order)}), batch_size);

    // Compute the divisor: n * (1 - c^[0]).
    const auto c_idx = deps[0];
    auto *one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
    auto *divisor = llvm_fmul(s, n, llvm_fsub(s, one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars)));

    // Compute the first part of the dividend: n * M^[n] (the derivative of e is zero because
    // here e is a constant and the order is > 0).
    auto *dividend = llvm_fmul(s, n, taylor_fetch_diff(arr, M_idx, order, n_uvars));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

            auto *cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto *aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto *tmp = llvm_fmul(s, fac, llvm_fmul(s, cnj, aj));
            sum.push_back(tmp);
        }

        // Update the dividend.
        dividend = llvm_fadd(s, dividend, pairwise_sum(s, sum));
    }

    return llvm_fdiv(s, dividend, divisor);
}

// Derivative of kepE(var, var).
llvm::Value *taylor_diff_kepE_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const variable &var0, const variable &var1, const std::vector<llvm::Value *> &arr,
                                   llvm::Value *, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    assert(deps.size() == 2u);

    auto &builder = s.builder();

    // Fetch the index of the e/M variable arguments.
    const auto e_idx = uname_to_index(var0.name());
    const auto M_idx = uname_to_index(var1.name());

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

        // Invoke and return.
        return builder.CreateCall(
            fkep, {taylor_fetch_diff(arr, e_idx, 0, n_uvars), taylor_fetch_diff(arr, M_idx, 0, n_uvars)});
    }

    // Splat the order.
    auto *n = vector_splat(builder, llvm_codegen(s, fp_t, number{static_cast<double>(order)}), batch_size);

    // Compute the divisor: n * (1 - c^[0]).
    const auto c_idx = deps[0];
    auto *one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
    auto *divisor = llvm_fmul(s, n, llvm_fsub(s, one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars)));

    // Compute the first part of the dividend: n * (e^[n] * d^[0] + M^[n]).
    const auto d_idx = deps[1];
    auto *dividend
        = llvm_fmul(s, taylor_fetch_diff(arr, e_idx, order, n_uvars), taylor_fetch_diff(arr, d_idx, 0, n_uvars));
    dividend = llvm_fadd(s, dividend, taylor_fetch_diff(arr, M_idx, order, n_uvars));
    dividend = llvm_fmul(s, n, dividend);

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

            auto *cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto *aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto *dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto *ej = taylor_fetch_diff(arr, e_idx, j, n_uvars);

            auto *tmp = llvm_fmul(s, dnj, ej);
            tmp = llvm_fadd(s, llvm_fmul(s, cnj, aj), tmp);
            sum.push_back(llvm_fmul(s, fac, tmp));
        }

        // Update the dividend.
        dividend = llvm_fadd(s, dividend, pairwise_sum(s, sum));
    }

    return llvm_fdiv(s, dividend, divisor);
}

// All the other cases.
template <typename U, typename V, typename... Args>
llvm::Value *taylor_diff_kepE_impl(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &, const U &, const V &,
                                   const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                   std::uint32_t, std::uint32_t, const Args &...)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of kepE()");
}

llvm::Value *taylor_diff_kepE(llvm_state &s, llvm::Type *fp_t, const kepE_impl &f,
                              const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                              llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                              std::uint32_t batch_size)
{
    assert(f.args().size() == 2u);

    if (deps.size() != 2u) {
        throw std::invalid_argument(
            fmt::format("A hidden dependency vector of size 2 is expected in order to compute the Taylor "
                        "derivative of kepE(), but a vector of size {} was passed "
                        "instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_kepE_impl(s, fp_t, deps, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value(), f.args()[1].value());
}

} // namespace

llvm::Value *kepE_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size, bool) const
{
    return taylor_diff_kepE(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of kepE(number, number).
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *taylor_c_diff_func_kepE_impl(llvm_state &s, llvm::Type *fp_t, const U &n0, const V &n1,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "kepE", 2,
        [&s, fp_t, batch_size](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 2u);
            assert(args[0] != nullptr);
            assert(args[1] != nullptr);
            // LCOV_EXCL_STOP

            // Create/fetch the Kepler solver.
            auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

            return s.builder().CreateCall(fkep, args);
        },
        n0, n1);
}

// Derivative of kepE(var, number).
template <typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Function *taylor_c_diff_func_kepE_impl(llvm_state &s, llvm::Type *fp_t, const variable &var, const U &n,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "kepE", n_uvars, batch_size, {var, n}, 2);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

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
        auto e_idx = f->args().begin() + 5;
        auto num_M = f->args().begin() + 6;
        auto c_idx = f->args().begin() + 7;
        auto d_idx = f->args().begin() + 8;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of e_idx.
                builder.CreateStore(
                    builder.CreateCall(fkep,
                                       {taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), e_idx),
                                        taylor_c_diff_numparam_codegen(s, fp_t, n, num_M, par_ptr, batch_size)}),
                    retval);
            },
            [&]() {
                // Create FP vector version of the order.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, fp_t), batch_size);

                // Compute the divisor: ord * (1 - c^[0]).
                auto one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
                auto divisor
                    = llvm_fsub(s, one_fp, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), c_idx));
                divisor = llvm_fmul(s, ord_v, divisor);

                // Init the dividend: ord * e^[ord] * d^[0] (M is constant here).
                auto dividend = llvm_fmul(s, ord_v, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, e_idx));
                dividend = llvm_fmul(s, dividend,
                                     taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), d_idx));

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, fp_t), batch_size);

                    auto c_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, u_idx);
                    auto tmp = llvm_fmul(s, c_nj, aj);

                    auto d_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), d_idx);
                    auto ej = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, e_idx);
                    tmp = llvm_fadd(s, llvm_fmul(s, d_nj, ej), tmp);

                    tmp = llvm_fmul(s, j_v, tmp);

                    builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), tmp), acc);
                });

                // Write the result.
                builder.CreateStore(llvm_fdiv(s, llvm_fadd(s, dividend, builder.CreateLoad(val_t, acc)), divisor),
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
                "Inconsistent function signatures for the Taylor derivative of kepE() in compact mode detected");
        }
    }

    return f;
}

// Derivative of kepE(number, var).
template <typename U, std::enable_if_t<is_num_param<U>::value, int> = 0>
llvm::Function *taylor_c_diff_func_kepE_impl(llvm_state &s, llvm::Type *fp_t, const U &n, const variable &var,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "kepE", n_uvars, batch_size, {n, var}, 2);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

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
        auto num_e = f->args().begin() + 5;
        auto M_idx = f->args().begin() + 6;
        auto c_idx = f->args().begin() + 7;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of M_idx.
                builder.CreateStore(
                    builder.CreateCall(fkep,
                                       {taylor_c_diff_numparam_codegen(s, fp_t, n, num_e, par_ptr, batch_size),
                                        taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), M_idx)}),
                    retval);
            },
            [&]() {
                // Create FP vector versions of the order.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, fp_t), batch_size);

                // Compute the divisor: ord * (1 - c^[0]).
                auto one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
                auto divisor
                    = llvm_fsub(s, one_fp, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), c_idx));
                divisor = llvm_fmul(s, ord_v, divisor);

                // Init the dividend: ord * M^[n] (e is constant here).
                auto dividend = llvm_fmul(s, ord_v, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, M_idx));

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, fp_t), batch_size);

                    auto c_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, u_idx);
                    auto tmp = llvm_fmul(s, c_nj, aj);
                    tmp = llvm_fmul(s, j_v, tmp);

                    builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), tmp), acc);
                });

                // Write the result.
                builder.CreateStore(llvm_fdiv(s, llvm_fadd(s, dividend, builder.CreateLoad(val_t, acc)), divisor),
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
                "Inconsistent function signatures for the Taylor derivative of kepE() in compact mode detected");
        }
    }

    return f;
}

// Derivative of kepE(var, var).
llvm::Function *taylor_c_diff_func_kepE_impl(llvm_state &s, llvm::Type *fp_t, const variable &var0,
                                             const variable &var1, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "kepE", n_uvars, batch_size, {var0, var1}, 2);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_E(s, fp_t, batch_size);

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
        auto e_idx = f->args().begin() + 5;
        auto M_idx = f->args().begin() + 6;
        auto c_idx = f->args().begin() + 7;
        auto d_idx = f->args().begin() + 8;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of M_idx/e_idx.
                builder.CreateStore(
                    builder.CreateCall(fkep,
                                       {taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), e_idx),
                                        taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), M_idx)}),
                    retval);
            },
            [&]() {
                // Create FP vector version of the order.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, fp_t), batch_size);

                // Compute the divisor: ord * (1 - c^[0]).
                auto one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
                auto divisor
                    = llvm_fsub(s, one_fp, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), c_idx));
                divisor = llvm_fmul(s, ord_v, divisor);

                // Init the dividend: ord * (e^[ord] * d^[0] + M^[ord]).
                auto dividend = llvm_fmul(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, e_idx),
                                          taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), d_idx));
                dividend = llvm_fadd(s, dividend, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, M_idx));
                dividend = llvm_fmul(s, ord_v, dividend);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, fp_t), batch_size);

                    auto c_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, u_idx);
                    auto tmp = llvm_fmul(s, c_nj, aj);

                    auto d_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), d_idx);
                    auto ej = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, e_idx);
                    tmp = llvm_fadd(s, llvm_fmul(s, d_nj, ej), tmp);

                    tmp = llvm_fmul(s, j_v, tmp);

                    builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), tmp), acc);
                });

                // Write the result.
                builder.CreateStore(llvm_fdiv(s, llvm_fadd(s, dividend, builder.CreateLoad(val_t, acc)), divisor),
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
                "Inconsistent function signatures for the Taylor derivative of kepE() in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename U, typename V, typename... Args>
llvm::Function *taylor_c_diff_func_kepE_impl(llvm_state &, llvm::Type *, const U &, const V &, std::uint32_t,
                                             std::uint32_t, const Args &...)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of kepE() in compact mode");
}

llvm::Function *taylor_c_diff_func_kepE(llvm_state &s, llvm::Type *fp_t, const kepE_impl &fn, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    assert(fn.args().size() == 2u);

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_c_diff_func_kepE_impl(s, fp_t, v1, v2, n_uvars, batch_size);
        },
        fn.args()[0].value(), fn.args()[1].value());
}

} // namespace

llvm::Function *kepE_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                              std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_kepE(s, fp_t, *this, n_uvars, batch_size);
}

} // namespace detail

expression kepE(expression e, expression M)
{
    return expression{func{detail::kepE_impl{std::move(e), std::move(M)}}};
}

expression kepE(expression e, double M)
{
    return kepE(std::move(e), expression(M));
}

expression kepE(expression e, long double M)
{
    return kepE(std::move(e), expression(M));
}

#if defined(HEYOKA_HAVE_REAL128)

expression kepE(expression e, mppp::real128 M)
{
    return kepE(std::move(e), expression(M));
}

#endif

expression kepE(double e, expression M)
{
    return kepE(expression(e), std::move(M));
}

expression kepE(long double e, expression M)
{
    return kepE(expression(e), std::move(M));
}

#if defined(HEYOKA_HAVE_REAL128)

expression kepE(mppp::real128 e, expression M)
{
    return kepE(expression(e), std::move(M));
}

#endif

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::kepE_impl)
