// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
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
#include <heyoka/math/log.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

log_impl::log_impl(expression e) : func_base("log", std::vector{std::move(e)}) {}

log_impl::log_impl() : log_impl(1_dbl) {}

llvm::Value *log_impl::llvm_evaluate(llvm_state &s, const std::vector<llvm::Value *> &args, llvm::Type *, llvm::Value *,
                                     bool)
{
    assert(args.size() == 1u);
    return llvm_log(s, args[0]);
}

namespace
{

// Derivative of log(number)/log1p(number).
template <typename F, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_log_impl(llvm_state &s, llvm::Type *fp_t, const F &, const U &num,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        auto *const arg = taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size);
        return std::same_as<F, log_impl> ? llvm_log(s, arg) : llvm_log1p(s, arg);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of log(variable)/log1p(variable).
template <typename F>
llvm::Value *taylor_diff_log_impl(llvm_state &s, llvm::Type *fp_t, const F &, const variable &var,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  std::uint32_t order, std::uint32_t a_idx, std::uint32_t batch_size)
{
    constexpr auto islog = std::same_as<F, log_impl>;

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto b_idx = uname_to_index(var.name());

    // Load b[0].
    auto *const b0 = taylor_fetch_diff(arr, b_idx, 0, n_uvars);

    if (order == 0u) {
        return islog ? llvm_log(s, b0) : llvm_log1p(s, b0);
    }

    // Create the fp version of the order.
    auto *ord_fp = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(order))), batch_size);

    // Compute the outside divisor (n*b^[0] or n*(b^[0]+1)).
    auto *const div
        = islog ? llvm_fmul(s, ord_fp, b0)
                : llvm_fmul(s, ord_fp,
                            llvm_fadd(s, b0, vector_splat(builder, llvm_codegen(s, fp_t, number(1.)), batch_size)));

    // Init ret with n*b^[n].
    auto *ret = llvm_fmul(s, ord_fp, taylor_fetch_diff(arr, b_idx, order, n_uvars));

    // Run the summation only if order is > 1, otherwise pairwise_sum() will error out.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        for (std::uint32_t j = 1; j < order; ++j) {
            auto *bnj = taylor_fetch_diff(arr, b_idx, order - j, n_uvars);
            auto *aj = taylor_fetch_diff(arr, a_idx, j, n_uvars);

            auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

            // Add j*bnj*aj to the sum.
            sum.push_back(llvm_fmul(s, fac, llvm_fmul(s, bnj, aj)));
        }

        // Compute the result of the summation and subtract it from ret.
        ret = llvm_fsub(s, ret, pairwise_sum(s, sum));
    }

    // Divide by div and return.
    return llvm_fdiv(s, ret, div);
}

// All the other cases.
template <typename F, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_log_impl(llvm_state &, llvm::Type *, const F &self, const U &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(fmt::format(
        "An invalid argument type was encountered while trying to build the Taylor derivative of {}", self.get_name()));
}

template <typename F>
llvm::Value *taylor_diff_log(llvm_state &s, llvm::Type *fp_t, const F &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of {}, but a vector of size {} was passed instead",
                        f.get_name(), deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_log_impl(s, fp_t, f, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *log_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool) const
{
    return taylor_diff_log(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of log(number)/log1p(number).
template <typename F, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_log_impl(llvm_state &s, llvm::Type *fp_t, const F &self, const U &num,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, self.get_name(), 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return std::same_as<F, log_impl> ? llvm_log(s, args[0]) : llvm_log1p(s, args[0]);
        },
        num);
}

// Derivative of log(variable)/log1p(variable).
template <typename F>
llvm::Function *taylor_c_diff_func_log_impl(llvm_state &s, llvm::Type *fp_t, const F &self, const variable &var,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    constexpr auto islog = std::same_as<F, log_impl>;

    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, self.get_name(), n_uvars, batch_size, {var});
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

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        // Load b[0].
        auto *const b0 = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), b_idx);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&] {
                // For order 0, invoke the function on the order 0 of b_idx.
                builder.CreateStore(islog ? llvm_log(s, b0) : llvm_log1p(s, b0), retval);
            },
            [&] {
                // Create the fp version of the order.
                auto *ord_fp = vector_splat(builder, llvm_ui_to_fp(s, ord, fp_t), batch_size);

                // Compute the outside divisor (n*b^[0] or n*(b^[0]+1)).
                auto *div = islog ? llvm_fmul(s, ord_fp, b0)
                                  : llvm_fmul(s, ord_fp, llvm_fadd(s, b0, llvm_codegen(s, val_t, number{1.})));

                // Compute n*b^[n].
                auto *nbn = llvm_fmul(s, ord_fp, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, b_idx));

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto *bnj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), b_idx);
                    auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, a_idx);

                    // Compute j.
                    auto *fac = vector_splat(builder, llvm_ui_to_fp(s, j, fp_t), batch_size);

                    builder.CreateStore(
                        llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, fac, llvm_fmul(s, bnj, aj))), acc);
                });

                // ret = (n*b^[n] - acc) / div.
                builder.CreateStore(llvm_fdiv(s, llvm_fsub(s, nbn, builder.CreateLoad(val_t, acc)), div), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// All the other cases.
template <typename F, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_log_impl(llvm_state &, llvm::Type *, const F &self, const U &, std::uint32_t,
                                            std::uint32_t)
{
    throw std::invalid_argument(
        fmt::format("An invalid argument type was encountered while trying to build the Taylor derivative "
                    "of {} in compact mode",
                    self.get_name()));
}

template <typename F>
llvm::Function *taylor_c_diff_func_log(llvm_state &s, llvm::Type *fp_t, const F &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_log_impl(s, fp_t, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *log_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_log(s, fp_t, *this, n_uvars, batch_size);
}

[[nodiscard]] std::vector<expression> log_impl::gradient() const
{
    assert(args().size() == 1u);

    return {pow(args()[0], -1_dbl)};
}

} // namespace detail

expression log(expression e)
{
    if (const auto *num_ptr = std::get_if<number>(&e.value())) {
        return std::visit(
            [](const auto &x) {
                using std::log;

                return expression{log(x)};
            },
            num_ptr->value());
    } else {
        return expression{func{detail::log_impl(std::move(e))}};
    }
}

namespace detail
{

log1p_impl::log1p_impl(expression e) : func_base("log1p", std::vector{std::move(e)}) {}

log1p_impl::log1p_impl() : log1p_impl(0_dbl) {}

llvm::Value *log1p_impl::llvm_evaluate(llvm_state &s, const std::vector<llvm::Value *> &args, llvm::Type *,
                                       llvm::Value *, bool)
{
    assert(args.size() == 1u);
    return llvm_log1p(s, args[0]);
}

llvm::Value *log1p_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                     std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                     std::uint32_t batch_size, bool) const
{
    return taylor_diff_log(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Function *log1p_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                               std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_log(s, fp_t, *this, n_uvars, batch_size);
}

[[nodiscard]] std::vector<expression> log1p_impl::gradient() const
{
    assert(args().size() == 1u);

    return {pow(1_dbl + args()[0], -1_dbl)};
}

} // namespace detail

expression log1p(expression e)
{
    if (const auto *num_ptr = std::get_if<number>(&e.value())) {
        return std::visit(
            [](const auto &x) {
                using std::log1p;

                return expression{log1p(x)};
            },
            num_ptr->value());
    } else {
        return expression{func{detail::log1p_impl(std::move(e))}};
    }
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp,bugprone-throwing-static-initialization)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::log_impl)
// NOLINTNEXTLINE(cert-err58-cpp,bugprone-throwing-static-initialization)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::log1p_impl)
