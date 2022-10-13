// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <initializer_list>
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
#include <heyoka/math/sqrt.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

sqrt_impl::sqrt_impl(expression e) : func_base("sqrt", std::vector{std::move(e)}) {}

sqrt_impl::sqrt_impl() : sqrt_impl(0_dbl) {}

double sqrt_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return std::sqrt(heyoka::eval_dbl(args()[0], map, pars));
}

long double sqrt_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                 const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return std::sqrt(heyoka::eval_ldbl(args()[0], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 sqrt_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                   const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return mppp::sqrt(heyoka::eval_f128(args()[0], map, pars));
}
#endif

void sqrt_impl::eval_batch_dbl(std::vector<double> &out,
                               const std::unordered_map<std::string, std::vector<double>> &map,
                               const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    heyoka::eval_batch_dbl(out, args()[0], map, pars);
    for (auto &el : out) {
        el = std::sqrt(el);
    }
}

double sqrt_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("Inconsistent number of arguments when computing the numerical value of the "
                        "square root over doubles (1 argument was expected, but {} arguments were provided",
                        a.size()));
    }

    return std::sqrt(a[0]);
}

double sqrt_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 1u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the square root");
    }

    return 1. / (2. * std::sqrt(a[0]));
}

llvm::Value *sqrt_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                  llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                  bool high_accuracy) const
{
    return llvm_eval_helper([&s](const std::vector<llvm::Value *> &args, bool) { return llvm_sqrt(s, args[0]); }, *this,
                            s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *sqrt_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                               std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "sqrt", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_sqrt(s, args[0]); }, fb, s, fp_t,
        batch_size, high_accuracy);
}

} // namespace

llvm::Function *sqrt_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    return sqrt_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

namespace
{

// Derivative of sqrt(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, llvm::Type *fp_t, const sqrt_impl &, const U &num,
                                   const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                   std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return llvm_sqrt(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of sqrt(variable).
// NOTE: this is derived by taking:
// a = sqrt(b) -> a**2 = b -> (a**2)^[n] = b^[n]
// and then using the squaring formula.
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, llvm::Type *, const sqrt_impl &, const variable &var,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                   std::uint32_t order, std::uint32_t idx, std::uint32_t)
{
    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_sqrt(s, taylor_fetch_diff(arr, u_idx, 0, n_uvars));
    }

    // Compute the divisor: 2*a^[0].
    auto *div = taylor_fetch_diff(arr, idx, 0, n_uvars);
    div = llvm_fadd(s, div, div);

    // Init the factor: b^[n].
    auto *fac = taylor_fetch_diff(arr, u_idx, order, n_uvars);

    std::vector<llvm::Value *> sum;
    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 1; j <= (order - 1u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }
    } else {
        // Even order.
        for (std::uint32_t j = 1; j <= (order - 2u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto *tmp = taylor_fetch_diff(arr, idx, order / 2u, n_uvars);
        tmp = builder.CreateFMul(tmp, tmp);

        fac = builder.CreateFSub(fac, tmp);
    }

    // Avoid summing if the sum is empty.
    if (!sum.empty()) {
        auto *tmp = pairwise_sum(builder, sum);
        tmp = llvm_fadd(s, tmp, tmp);

        fac = builder.CreateFSub(fac, tmp);
    }

    return builder.CreateFDiv(fac, div);
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &, llvm::Type *, const sqrt_impl &, const U &,
                                   const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                   std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a square root");
}

llvm::Value *taylor_diff_sqrt(llvm_state &s, llvm::Type *fp_t, const sqrt_impl &f,
                              const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                              llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                              std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the square root, but a vector of size {} was passed "
                        "instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_sqrt_impl(s, fp_t, f, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *sqrt_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size, bool) const
{
    return taylor_diff_sqrt(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of sqrt(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &s, llvm::Type *fp_t, const sqrt_impl &, const U &num,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "sqrt", 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_sqrt(s, args[0]);
        },
        num);
}

// Derivative of sqrt(variable).
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &s, llvm::Type *fp_t, const sqrt_impl &, const variable &var,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "sqrt", n_uvars, batch_size, {var});
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
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *u_idx = f->args().begin() + 1;
        auto *diff_ptr = f->args().begin() + 2;
        auto *var_idx = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    llvm_sqrt(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            },
            [&]() {
                // Compute the divisor: 2*a^[0].
                auto *div = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), u_idx);
                div = llvm_fadd(s, div, div);

                // retval = b^[n].
                builder.CreateStore(taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, var_idx), retval);

                // Determine the upper index of the summation: (ord - 1)/2 if ord is odd, (ord - 2)/2 otherwise.
                auto *ord_even
                    = builder.CreateICmpEQ(builder.CreateURem(ord, builder.getInt32(2)), builder.getInt32(0));
                auto *upper = builder.CreateUDiv(
                    builder.CreateSub(ord, builder.CreateSelect(ord_even, builder.getInt32(2), builder.getInt32(1))),
                    builder.getInt32(2));

                // Perform the summation.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);
                llvm_loop_u32(
                    s, builder.getInt32(1), builder.CreateAdd(upper, builder.getInt32(1)), [&](llvm::Value *j) {
                        auto *a_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
                        auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, u_idx);

                        builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), builder.CreateFMul(a_nj, aj)),
                                            acc);
                    });
                builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), builder.CreateLoad(val_t, acc)), acc);

                llvm_if_then_else(
                    s, ord_even,
                    [&]() {
                        // retval -= (a^[n/2])**2.
                        auto *tmp = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars,
                                                       builder.CreateUDiv(ord, builder.getInt32(2)), u_idx);
                        tmp = builder.CreateFMul(tmp, tmp);

                        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(val_t, retval), tmp), retval);
                    },
                    []() {});

                // retval -= acc.
                builder.CreateStore(
                    builder.CreateFSub(builder.CreateLoad(val_t, retval), builder.CreateLoad(val_t, acc)), retval);

                // retval /= div.
                builder.CreateStore(builder.CreateFDiv(builder.CreateLoad(val_t, retval), div), retval);
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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of the square root "
                                        "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &, llvm::Type *, const sqrt_impl &, const U &, std::uint32_t,
                                             std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a square root in compact mode");
}

llvm::Function *taylor_c_diff_func_sqrt(llvm_state &s, llvm::Type *fp_t, const sqrt_impl &fn, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_sqrt_impl(s, fp_t, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *sqrt_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                              std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_sqrt(s, fp_t, *this, n_uvars, batch_size);
}

expression sqrt_impl::diff(std::unordered_map<const void *, expression> &func_map, const std::string &s) const
{
    assert(args().size() == 1u);

    return detail::diff(func_map, args()[0], s) / (2_dbl * sqrt(args()[0]));
}

expression sqrt_impl::diff(std::unordered_map<const void *, expression> &func_map, const param &p) const
{
    assert(args().size() == 1u);

    return detail::diff(func_map, args()[0], p) / (2_dbl * sqrt(args()[0]));
}

} // namespace detail

expression sqrt(expression e)
{
    return expression{func{detail::sqrt_impl(std::move(e))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::sqrt_impl)
