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
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>
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

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/erf.hpp>
#include <heyoka/math/exp.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

erf_impl::erf_impl(expression e) : func_base("erf", std::vector{std::move(e)}) {}

erf_impl::erf_impl() : erf_impl(0_dbl) {}

llvm::Value *erf_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    return llvm_eval_helper([&s](const std::vector<llvm::Value *> &args, bool) { return llvm_erf(s, args[0]); }, *this,
                            s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *erf_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                              std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "erf", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_erf(s, args[0]); }, fb, s, fp_t,
        batch_size, high_accuracy);
}

} // namespace

llvm::Function *erf_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    return erf_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

taylor_dc_t::size_type erf_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Append arg * arg.
    u_vars_defs.emplace_back(pow(args()[0], 2_dbl), std::vector<std::uint32_t>{});

    // Append - arg * arg.
    u_vars_defs.emplace_back(-expression{fmt::format("u_{}", u_vars_defs.size() - 1u)}, std::vector<std::uint32_t>{});

    // Append exp( - arg * arg).
    u_vars_defs.emplace_back(exp(expression{fmt::format("u_{}", u_vars_defs.size() - 1u)}),
                             std::vector<std::uint32_t>{});

    // Append the erf decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed erf).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Small helper to compute sqrt(pi) / 2 at the precision of the floating-point type tp.
// NOTE: this is a repetition of number_like(), perhaps we can abstract this?
number sqrt_pi_2_like(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);

    auto &context = s.context();

    if (tp == to_external_llvm_type<float>(context, false)) {
        return number{std::sqrt(boost::math::constants::pi<float>()) / 2};
    } else if (tp == to_external_llvm_type<double>(context, false)) {
        return number{std::sqrt(boost::math::constants::pi<double>()) / 2};
    } else if (tp == to_external_llvm_type<long double>(context, false)) {
        return number{std::sqrt(boost::math::constants::pi<long double>()) / 2};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (tp == to_external_llvm_type<mppp::real128>(context, false)) {
        return number{mppp::sqrt(mppp::pi_128) / 2};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto real_prec = llvm_is_real(tp)) {
        auto ret = mppp::sqrt(mppp::real_pi(real_prec));
        mppp::div_2ui(ret, ret, 1ul);

        return number{std::move(ret)};
#endif
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Unable to create a number of type '{}' from the constant sqrt(pi) / 2", llvm_type_name(tp)));
    // LCOV_EXCL_STOP
}

// Derivative of erf(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_erf_impl(llvm_state &s, llvm::Type *fp_t, const erf_impl &, const std::vector<std::uint32_t> &,
                                  const U &num, const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return llvm_erf(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of erf(variable).
llvm::Value *taylor_diff_erf_impl(llvm_state &s, llvm::Type *fp_t, const erf_impl &,
                                  const std::vector<std::uint32_t> &deps, const variable &var,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the index of the variable argument.
    const auto b_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_erf(s, taylor_fetch_diff(arr, b_idx, 0, n_uvars));
    }

    // NOTE: iteration in the [1, order] range.
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is exp(- var * var).
        auto *bj = taylor_fetch_diff(arr, b_idx, j, n_uvars);
        auto *cnj = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);

        auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

        // Add j*cnj*bj to the sum vector.
        sum.push_back(llvm_fmul(s, fac, llvm_fmul(s, cnj, bj)));
    }

    // Create ret with the sum.
    auto *ret = pairwise_sum(s, sum);

    // Generate the factor n * sqrt(pi) / 2.
    auto fac = number_like(s, fp_t, static_cast<double>(order)) * sqrt_pi_2_like(s, fp_t);
    auto *fac_s = vector_splat(builder, llvm_codegen(s, fp_t, fac), batch_size);

    // Multiply and return.
    return llvm_fdiv(s, ret, fac_s);
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_erf_impl(llvm_state &, llvm::Type *, const erf_impl &, const std::vector<std::uint32_t> &,
                                  const U &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of the error function");
}

llvm::Value *taylor_diff_erf(llvm_state &s, llvm::Type *fp_t, const erf_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("A hidden dependency vector of size 1 is expected in order to compute the Taylor "
                        "derivative of the error function, but a vector of size {} was passed instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_erf_impl(s, fp_t, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *erf_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool) const
{
    return taylor_diff_erf(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of erf(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_erf_impl(llvm_state &s, llvm::Type *fp_t, const erf_impl &, const U &num,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "erf", 1,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_erf(s, args[0]);
        },
        num);
}

// Derivative of erf(variable).
llvm::Function *taylor_c_diff_func_erf_impl(llvm_state &s, llvm::Type *fp_t, const erf_impl &, const variable &var,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "erf", n_uvars, batch_size, {var}, 1);
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
                    llvm_erf(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), b_idx)), retval);
            },
            [&]() {
                // Compute the fp version of the order.
                auto *ord_fp = vector_splat(builder, llvm_ui_to_fp(s, ord, fp_t), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    auto *c_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto *bj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, b_idx);

                    auto *fac = vector_splat(builder, llvm_ui_to_fp(s, j, fp_t), batch_size);

                    builder.CreateStore(
                        llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, fac, llvm_fmul(s, c_nj, bj))), acc);
                });

                // Generate the factor n * sqrt(pi) /2
                auto *t1_llvm = vector_splat(builder, llvm_codegen(s, fp_t, sqrt_pi_2_like(s, fp_t)), batch_size);
                auto *fac = llvm_fmul(s, t1_llvm, ord_fp);

                // Store into retval.
                builder.CreateStore(llvm_fdiv(s, builder.CreateLoad(val_t, acc), fac), retval);
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
llvm::Function *taylor_c_diff_func_erf_impl(llvm_state &, llvm::Type *, const erf_impl &, const U &, std::uint32_t,
                                            std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of the error function in compact mode");
}

llvm::Function *taylor_c_diff_func_erf(llvm_state &s, llvm::Type *fp_t, const erf_impl &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_erf_impl(s, fp_t, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *erf_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_erf(s, fp_t, *this, n_uvars, batch_size);
}

std::vector<expression> erf_impl::gradient() const
{
    assert(args().size() == 1u);
    return {2_dbl / sqrt(pi) * exp(-pow(args()[0], 2_dbl))};
}

} // namespace detail

expression erf(expression e)
{
    if (const auto *num_ptr = std::get_if<number>(&e.value())) {
        return std::visit(
            [](const auto &x) {
                using std::erf;

                return expression{erf(x)};
            },
            num_ptr->value());
    } else {
        return expression{func{detail::erf_impl(std::move(e))}};
    }
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::erf_impl)
