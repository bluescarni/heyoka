// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
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
#include <heyoka/math/atanh.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

atanh_impl::atanh_impl(expression e) : func_base("atanh", std::vector{std::move(e)}) {}

atanh_impl::atanh_impl() : atanh_impl(0_dbl) {}

std::vector<expression> atanh_impl::gradient() const
{
    return {pow(1_dbl - pow(args()[0], 2_dbl), -1_dbl)};
}

llvm::Value *atanh_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                   llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                   bool high_accuracy) const
{
    return llvm_eval_helper([&s](const std::vector<llvm::Value *> &args, bool) { return llvm_atanh(s, args[0]); },
                            *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *atanh_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                                std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "atanh", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_atanh(s, args[0]); }, fb, s, fp_t,
        batch_size, high_accuracy);
}

} // namespace

llvm::Function *atanh_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                             bool high_accuracy) const
{
    return atanh_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

taylor_dc_t::size_type atanh_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Append arg * arg.
    u_vars_defs.emplace_back(pow(args()[0], 2_dbl), std::vector<std::uint32_t>{});

    // Append the atanh decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed atanh).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Derivative of atanh(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_atanh_impl(llvm_state &s, llvm::Type *fp_t, const atanh_impl &,
                                    const std::vector<std::uint32_t> &, const U &num,
                                    const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                    std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return llvm_atanh(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

llvm::Value *taylor_diff_atanh_impl(llvm_state &s, llvm::Type *fp_t, const atanh_impl &,
                                    const std::vector<std::uint32_t> &deps, const variable &var,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the index of the variable argument.
    const auto b_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_atanh(s, taylor_fetch_diff(arr, b_idx, 0, n_uvars));
    }

    // Create the constant 1 in fp format.
    auto *one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);

    if (order == 1u) {
        // Special-case the first-order derivative, in order
        // to avoid an empty summation below.
        return llvm_fdiv(s, taylor_fetch_diff(arr, b_idx, 1, n_uvars),
                         llvm_fsub(s, one_fp, taylor_fetch_diff(arr, deps[0], 0, n_uvars)));
    }

    // Create the fp version of the order.
    auto *ord_fp = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(order))), batch_size);

    // Assemble the first part of the result: n*b^[n].
    auto *ret = llvm_fmul(s, ord_fp, taylor_fetch_diff(arr, b_idx, order, n_uvars));

    // Compute n*(1 - c^[0]).
    auto *n_c0_p1 = llvm_fmul(s, ord_fp, llvm_fsub(s, one_fp, taylor_fetch_diff(arr, deps[0], 0, n_uvars)));

    // NOTE: iteration in the [1, order) range.
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j < order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is var * var.
        auto *cnj = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);
        auto *aj = taylor_fetch_diff(arr, idx, j, n_uvars);

        auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

        // Add j*cnj*aj to the sum.
        sum.push_back(llvm_fmul(s, fac, llvm_fmul(s, cnj, aj)));
    }

    // Update ret.
    ret = llvm_fadd(s, ret, pairwise_sum(s, sum));

    // Divide by n*(1 - c^[0]) and return.
    return llvm_fdiv(s, ret, n_c0_p1);
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_atanh_impl(llvm_state &, llvm::Type *, const atanh_impl &, const std::vector<std::uint32_t> &,
                                    const U &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                    std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an inverse hyperbolic tangent");
}

llvm::Value *taylor_diff_atanh(llvm_state &s, llvm::Type *fp_t, const atanh_impl &f,
                               const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                               llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                               std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("A hidden dependency vector of size 1 is expected in order to compute the Taylor "
                        "derivative of the inverse hyperbolic tangent, but a vector of size {} was passed instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_atanh_impl(s, fp_t, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *atanh_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                     std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                     std::uint32_t batch_size, bool) const
{
    return taylor_diff_atanh(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of atanh(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_atanh_impl(llvm_state &s, llvm::Type *fp_t, const atanh_impl &, const U &num,
                                              std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "atanh", 1,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_atanh(s, args[0]);
        },
        num);
}

// Derivative of atanh(variable).
llvm::Function *taylor_c_diff_func_atanh_impl(llvm_state &s, llvm::Type *fp_t, const atanh_impl &, const variable &var,
                                              std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "atanh", n_uvars, batch_size, {var}, 1);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *a_idx = f->args().begin() + 1;
        auto *diff_ptr = f->args().begin() + 2;
        auto *b_idx = f->args().begin() + 5;
        auto *c_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of b_idx.
                builder.CreateStore(
                    llvm_atanh(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), b_idx)), retval);
            },
            [&]() {
                // Create the constant 1 in fp format.
                auto *one_fp = vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);

                // Compute the fp version of the order.
                auto *ord_fp = vector_splat(builder, llvm_ui_to_fp(s, ord, fp_t), batch_size);

                // Compute n*b^[n].
                auto *ret = llvm_fmul(s, ord_fp, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, b_idx));

                // Compute n*(1 - c^[0]).
                auto *n_c0_p1 = llvm_fmul(
                    s, ord_fp,
                    llvm_fsub(s, one_fp, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), c_idx)));

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto *c_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, a_idx);

                    auto *fac = vector_splat(builder, llvm_ui_to_fp(s, j, fp_t), batch_size);

                    builder.CreateStore(
                        llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, fac, llvm_fmul(s, c_nj, aj))), acc);
                });

                // Update ret.
                ret = llvm_fadd(s, ret, builder.CreateLoad(val_t, acc));

                // Divide by n*(1 - c^[0]).
                ret = llvm_fdiv(s, ret, n_c0_p1);

                // Store into retval.
                // NOLINTNEXTLINE(readability-suspicious-call-argument)
                builder.CreateStore(ret, retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_atanh_impl(llvm_state &, llvm::Type *, const atanh_impl &, const U &, std::uint32_t,
                                              std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an inverse hyperbolic tangent in compact mode");
}

llvm::Function *taylor_c_diff_func_atanh(llvm_state &s, llvm::Type *fp_t, const atanh_impl &fn, std::uint32_t n_uvars,
                                         std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_atanh_impl(s, fp_t, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *atanh_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                               std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_atanh(s, fp_t, *this, n_uvars, batch_size);
}

} // namespace detail

expression atanh(expression e)
{
    if (const auto *num_ptr = std::get_if<number>(&e.value())) {
        return std::visit(
            [](const auto &x) {
                using std::atanh;

                return expression{atanh(x)};
            },
            num_ptr->value());
    } else {
        return expression{func{detail::atanh_impl(std::move(e))}};
    }
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::atanh_impl)
