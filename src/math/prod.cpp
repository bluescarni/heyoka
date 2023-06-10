// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

prod_impl::prod_impl() : prod_impl(std::vector<expression>{}) {}

prod_impl::prod_impl(std::vector<expression> v) : func_base("prod", std::move(v)) {}

namespace
{

// If ex is a pow() function whose exponent is a negative
// number, then a pointer to the exponent will be returned. Otherwise,
// nullptr will be returned.
const number *ex_is_negative_pow(const expression &ex)
{
    const auto *fptr = std::get_if<func>(&ex.value());

    if (fptr == nullptr) {
        // Not a function.
        return nullptr;
    }

    const auto *pow_ptr = fptr->extract<pow_impl>();

    if (pow_ptr == nullptr) {
        // Not a pow().
        return nullptr;
    }

    const auto *n_exp_ptr = std::get_if<number>(&(pow_ptr->args()[1].value()));

    if (n_exp_ptr == nullptr) {
        // pow() exponent is not a number.
        return nullptr;
    }

    // Helper to check if a number is negative.
    auto negative_checker = [](const auto &v) {
        using std::isnan;

        return !isnan(v) && v < 0;
    };

    if (std::visit(negative_checker, n_exp_ptr->value())) {
        return n_exp_ptr;
    } else {
        // pow() number exponent is either NaN or not negative.
        return nullptr;
    }
}

// Check if a product is a negation - that is, a product with at least
// 2 terms and whose only numerical coefficient is at the beginning with
// value -1.
bool prod_is_negation_impl(const prod_impl &p)
{
    const auto &args = p.args();

    if (args.size() < 2u) {
        return false;
    }

    if (const auto *num_ptr = std::get_if<number>(&args[0].value()); num_ptr != nullptr && is_negative_one(*num_ptr)) {
        return std::none_of(args.begin() + 1, args.end(),
                            [](const auto &ex) { return std::holds_alternative<number>(ex.value()); });
    } else {
        return false;
    }
}

} // namespace

// NOLINTNEXTLINE(misc-no-recursion)
void prod_impl::to_stream(std::ostringstream &oss) const
{
    // NOTE: prods which have 0 or 1 terms are not possible
    // when using the public API, but let's handle these special
    // cases anyway.
    if (args().empty()) {
        stream_expression(oss, 1_dbl);
        return;
    }

    if (args().size() == 1u) {
        stream_expression(oss, args()[0]);
        return;
    }

    // Special case for negation.
    if (prod_is_negation_impl(*this)) {
        oss << '-';
        prod_impl(std::vector(args().begin() + 1, args().end())).to_stream(oss);

        return;
    }

    // Partition the arguments so that pow()s with negative
    // exponents are at the end. These constitute the denominator
    // of the product.
    auto tmp_args = args();
    const auto den_it = std::stable_partition(tmp_args.begin(), tmp_args.end(),
                                              [](const auto &ex) { return ex_is_negative_pow(ex) == nullptr; });

    // Helper to stream the numerator of the product.
    auto stream_num = [&]() {
        assert(den_it != tmp_args.begin());

        // Is the numerator consisting of a single term?
        const auto single_num = (tmp_args.begin() + 1 == den_it);

        if (!single_num) {
            oss << '(';
        }

        for (auto it = tmp_args.begin(); it != den_it; ++it) {
            stream_expression(oss, *it);

            if (it + 1 != den_it) {
                oss << " * ";
            }
        }

        if (!single_num) {
            oss << ')';
        }
    };

    // Helper to stream the denominator of the product.
    auto stream_den = [&]() {
        assert(den_it != tmp_args.end());

        // Is the denominator consisting of a single term?
        const auto single_den = (den_it + 1 == tmp_args.end());

        if (!single_den) {
            oss << '(';
        }

        for (auto it = den_it; it != tmp_args.end(); ++it) {
            assert(std::holds_alternative<func>(it->value()));
            assert(std::get<func>(it->value()).extract<pow_impl>() != nullptr);
            assert(std::get<func>(it->value()).args().size() == 2u);

            // Fetch the pow()'s base and exponent.
            auto base = std::get<func>(it->value()).args()[0];
            auto exp = std::get<number>(std::get<func>(it->value()).args()[1].value());

            // Stream the pow() with negated exponent.
            stream_expression(oss, pow(std::move(base), expression{-exp}));

            if (it + 1 != tmp_args.end()) {
                oss << " * ";
            }
        }

        if (!single_den) {
            oss << ')';
        }
    };

    if (den_it == tmp_args.begin()) {
        // Product consists only of negative pow()s.
        oss << '(';
        stream_expression(oss, 1_dbl);
        oss << " / ";
        stream_den();
        oss << ')';
    } else if (den_it == tmp_args.end()) {
        // There are no negative pow()s in the prod.
        // NOTE: no need to wrap in '()' brackets here as the numerator's
        // output already contains them.
        stream_num();
    } else {
        // There are some negative pow()s in the prod.
        oss << '(';
        stream_num();
        oss << " / ";
        stream_den();
        oss << ')';
    }
}

std::vector<expression> prod_impl::gradient() const
{
    const auto n_args = args().size();

    std::vector<expression> retval, tmp;
    retval.reserve(n_args);
    tmp.reserve(n_args);

    for (decltype(args().size()) i = 0; i < n_args; ++i) {
        tmp.clear();

        for (decltype(i) j = 0; j < n_args; ++j) {
            if (i != j) {
                tmp.push_back(args()[j]);
            }
        }

        retval.push_back(prod(tmp));
    }

    return retval;
}

llvm::Value *prod_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                  llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                  bool high_accuracy) const
{
    if (prod_is_negation_impl(*this)) {
        // Special case for negation.
        assert(args().size() >= 2u);

        return llvm_eval_helper(
            [&s](const auto &args, bool) {
                auto args_copy = std::vector(args.begin() + 1, args.end());
                auto *res = pairwise_prod(s, args_copy);

                return llvm_fneg(s, res);
            },
            *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
    } else {
        return llvm_eval_helper(
            [&s, fp_t, batch_size](const auto &args, bool) {
                if (args.empty()) {
                    return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{1.}), batch_size);
                } else {
                    auto args_copy = args;
                    return pairwise_prod(s, args_copy);
                }
            },
            *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
    }
}

llvm::Function *prod_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    if (prod_is_negation_impl(*this)) {
        // Special case for negation.
        assert(args().size() >= 2u);

        return llvm_c_eval_func_helper(
            "prod_neg",
            [&s](const auto &args, bool) {
                auto args_copy = std::vector(args.begin() + 1, args.end());
                auto *res = pairwise_prod(s, args_copy);

                return llvm_fneg(s, res);
            },
            *this, s, fp_t, batch_size, high_accuracy);
    } else {
        return llvm_c_eval_func_helper(
            "prod",
            [&s, fp_t, batch_size](const auto &args, bool) {
                if (args.empty()) {
                    return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{1.}), batch_size);
                } else {
                    auto args_copy = args;
                    return pairwise_prod(s, args_copy);
                }
            },
            *this, s, fp_t, batch_size, high_accuracy);
    }
}

namespace
{

// Derivative of numpar * numpar.
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *prod_taylor_diff_impl(llvm_state &s, llvm::Type *fp_t, const U &num0, const V &num1,
                                   const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                   std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    // NOTE: use neg() if:
    //
    // - U is a negative_one() number,
    // - order is zero.
    //
    // Otherwise, fall through.
    if constexpr (std::is_same_v<U, number>) {
        if (is_negative_one(num0) && order == 0u) {
            return llvm_fneg(s, taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size));
        }
    }

    // The general case.
    if (order == 0u) {
        auto *n0 = taylor_codegen_numparam(s, fp_t, num0, par_ptr, batch_size);
        auto *n1 = taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size);

        return llvm_fmul(s, n0, n1);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of var * numpar.
// NOTE: no point in trying to optimise this with a negation,
// as the public API won't allow the creation of a var * number product
// (it will be immediately re-arranged into number * var).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *prod_taylor_diff_impl(llvm_state &s, llvm::Type *fp_t, const variable &var, const U &num,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                   std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto *ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);
    auto *mul = taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size);

    return llvm_fmul(s, mul, ret);
}

// Derivative of numpar * var.
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *prod_taylor_diff_impl(llvm_state &s, llvm::Type *fp_t, const U &num, const variable &var,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                   std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    // Special casing for neg().
    if constexpr (std::is_same_v<U, number>) {
        if (is_negative_one(num)) {
            return llvm_fneg(s, taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars));
        }
    }

    // Return the derivative of var * number.
    return prod_taylor_diff_impl(s, fp_t, var, num, arr, par_ptr, n_uvars, order, idx, batch_size);
}

// Derivative of var * var.
llvm::Value *prod_taylor_diff_impl(llvm_state &s, llvm::Type *, const variable &var0, const variable &var1,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                   std::uint32_t order, std::uint32_t, std::uint32_t)
{
    // Fetch the indices of the u variables.
    const auto u_idx0 = uname_to_index(var0.name());
    const auto u_idx1 = uname_to_index(var1.name());

    // NOTE: iteration in the [0, order] range
    // (i.e., order inclusive).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j <= order; ++j) {
        auto *v0 = taylor_fetch_diff(arr, u_idx0, order - j, n_uvars);
        auto *v1 = taylor_fetch_diff(arr, u_idx1, j, n_uvars);

        // Add v0*v1 to the sum.
        sum.push_back(llvm_fmul(s, v0, v1));
    }

    return pairwise_sum(s, sum);
}

// All the other cases.
// LCOV_EXCL_START
template <typename V1, typename V2, std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Value *prod_taylor_diff_impl(llvm_state &, llvm::Type *, const V1 &, const V2 &,
                                   const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                   std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of prod()");
}
// LCOV_EXCL_STOP

} // namespace

llvm::Value *prod_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size, bool) const
{
    // LCOV_EXCL_START
    if (!deps.empty()) {
        throw std::invalid_argument(fmt::format("The vector of hidden dependencies in the Taylor diff for a product "
                                                "should be empty, but instead it has a size of {}",
                                                deps.size()));
    }
    // LCOV_EXCL_STOP

    if (args().size() != 2u) {
        throw std::invalid_argument(fmt::format("The Taylor derivative of a product can be computed only for products "
                                                "of 2 terms, but the current product has {} term(s) instead",
                                                args().size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return prod_taylor_diff_impl(s, fp_t, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        args()[0].value(), args()[1].value());
}

namespace
{

// Derivative of numpar * numpar.
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *prod_taylor_c_diff_func_impl(llvm_state &s, llvm::Type *fp_t, const U &num0, const V &num1,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    // NOTE: use neg() if U is a negative_one() number.
    if constexpr (std::is_same_v<U, number>) {
        if (is_negative_one(num0)) {
            return taylor_c_diff_func_numpar(
                s, fp_t, n_uvars, batch_size, "prod_neg", 0,
                [&s](const auto &args) {
                    // LCOV_EXCL_START
                    assert(args.size() == 2u);
                    assert(args[1] != nullptr);
                    // LCOV_EXCL_STOP

                    return llvm_fneg(s, args[1]);
                },
                num0, num1);
        }
    }

    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "prod", 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 2u);
            assert(args[0] != nullptr);
            assert(args[1] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_fmul(s, args[0], args[1]);
        },
        num0, num1);
}

// Derivative of var * numpar.
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *prod_taylor_c_diff_func_impl(llvm_state &s, llvm::Type *fp_t, const variable &var, const U &n,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "prod", n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
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
        auto ret = taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(llvm_fmul(s, ret, taylor_c_diff_numparam_codegen(s, fp_t, n, num, par_ptr, batch_size)));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of prod() in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

// Derivative of neg(variable).
llvm::Function *taylor_c_diff_func_neg_impl(llvm_state &s, llvm::Type *fp_t, const variable &var, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair
        = taylor_c_diff_func_name_args(context, fp_t, "prod_neg", n_uvars, batch_size, {number{-1.}, var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *diff_ptr = f->args().begin() + 2;
        auto *var_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = llvm_fneg(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, var_idx));

        // Return the result.
        builder.CreateRet(retval);

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);

    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of the negation "
                                        "in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

// Derivative of numpar * var.
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *prod_taylor_c_diff_func_impl(llvm_state &s, llvm::Type *fp_t, const U &n, const variable &var,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<U, number>) {
        if (is_negative_one(n)) {
            // Special case for negation.
            return taylor_c_diff_func_neg_impl(s, fp_t, var, n_uvars, batch_size);
        }
    }

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "prod", n_uvars, batch_size, {n, var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
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
        auto ret = taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(llvm_fmul(s, ret, taylor_c_diff_numparam_codegen(s, fp_t, n, num, par_ptr, batch_size)));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of prod() in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

// Derivative of var * var.
llvm::Function *prod_taylor_c_diff_func_impl(llvm_state &s, llvm::Type *fp_t, const variable &var0,
                                             const variable &var1, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "prod", n_uvars, batch_size, {var0, var1});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *diff_ptr = f->args().begin() + 2;
        auto *idx0 = f->args().begin() + 5;
        auto *idx1 = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(0), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            auto *b_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx0);
            auto *cj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, idx1);
            builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, b_nj, cj)), acc);
        });

        // Create the return value.
        builder.CreateRet(builder.CreateLoad(val_t, acc));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of prod() in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

// All the other cases.
// LCOV_EXCL_START
template <typename V1, typename V2, std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Function *prod_taylor_c_diff_func_impl(llvm_state &, llvm::Type *, const V1 &, const V2 &, std::uint32_t,
                                             std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of prod() in compact mode");
}
// LCOV_EXCL_STOP

} // namespace

llvm::Function *prod_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                              std::uint32_t batch_size, bool) const
{
    if (args().size() != 2u) {
        throw std::invalid_argument(
            fmt::format("The Taylor derivative of a product in compact mode can be computed only for products "
                        "of 2 terms, but the current product has {} term(s) instead",
                        args().size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return prod_taylor_c_diff_func_impl(s, fp_t, v1, v2, n_uvars, batch_size);
        },
        args()[0].value(), args()[1].value());
}

// Simplify the arguments for a prod(). This function returns either the simplified vector of arguments,
// or a single expression directly representing the result of the product.
std::variant<std::vector<expression>, expression> prod_simplify_args(const std::vector<expression> &args_)
{
    // Step 1: flatten products in args.
    std::vector<expression> args;
    args.reserve(args_.size());
    for (const auto &arg : args_) {
        if (const auto *fptr = std::get_if<func>(&arg.value());
            fptr != nullptr && fptr->extract<detail::prod_impl>() != nullptr) {

            for (const auto &prod_arg : fptr->args()) {
                args.push_back(prod_arg);
            }
        } else {
            args.push_back(arg);
        }
    }

    // Step 2: gather common bases with numerical exponents.
    // NOTE: we use map instead of unordered_map because expression hashing always
    // requires traversing the whole expression, while std::less<expression> can
    // exit early.
    std::map<expression, number> base_exp_map;

    for (const auto &arg : args) {
        if (const auto *fptr = std::get_if<func>(&arg.value());
            fptr != nullptr && fptr->extract<detail::pow_impl>() != nullptr
            && std::holds_alternative<number>(fptr->args()[1].value())) {
            // The current argument is of the form x**exp, where exp is a number.
            const auto &exp = std::get<number>(fptr->args()[1].value());

            // Try to insert base and exponent into base_exp_map.
            const auto [it, new_item] = base_exp_map.try_emplace(fptr->args()[0], exp);

            if (!new_item) {
                // The base already existed in base_exp_map, update the exponent.
                it->second = it->second + exp;
            }
        } else {
            // The current argument is *NOT* a power with a numerical exponent. Let's try to insert
            // it into base_exp_map with an exponent of 1.
            const auto [it, new_item] = base_exp_map.try_emplace(arg, 1.);

            if (!new_item) {
                // The current argument was already in the map, update its exponent.
                it->second = it->second + number{1.};
            }
        }
    }

    // Reconstruct args from base_exp_map.
    args.clear();
    for (const auto &[base, exp] : base_exp_map) {
        args.push_back(pow(base, expression{exp}));
    }

    // Step 3: partition args so that all numbers are at the end.
    const auto n_end_it = std::stable_partition(
        args.begin(), args.end(), [](const expression &ex) { return !std::holds_alternative<number>(ex.value()); });

    // Constant fold the numbers.
    if (n_end_it != args.end()) {
        for (auto it = n_end_it + 1; it != args.end(); ++it) {
            // NOTE: do not use directly operator*() on expressions in order
            // to avoid recursion.
            *n_end_it = expression{std::get<number>(n_end_it->value()) * std::get<number>(it->value())};
        }

        // Remove all numbers but the first one.
        args.erase(n_end_it + 1, args.end());

        // Remove the remaining number if it is one, or
        // return zero if constant folding results in zero.
        if (is_one(std::get<number>(n_end_it->value()))) {
            args.pop_back();
        } else if (is_zero(std::get<number>(n_end_it->value()))) {
            return 0_dbl;
        }
    }

    // Special cases.
    if (args.empty()) {
        return 1_dbl;
    }

    if (args.size() == 1u) {
        return std::move(args[0]);
    }

    // Sort the operands in canonical order.
    std::stable_sort(args.begin(), args.end(), detail::comm_ops_lt);

    return args;
}

// Helper to split the input prod 'e' into nested prods, each
// of which will have at most 'split' arguments.
// If 'e' is not a prod, or if it is a prod with no more than
// 'split' terms, 'e' will be returned unmodified.
// NOTE: 'e' is assumed to be a function.
// NOTE: quite a bit of repetition with sum_split() here.
// NOLINTNEXTLINE(misc-no-recursion)
expression prod_split(const expression &e, std::uint32_t split)
{
    assert(split >= 2u);
    assert(std::holds_alternative<func>(e.value()));

    const auto *prod_ptr = std::get<func>(e.value()).extract<prod_impl>();

    // NOTE: return 'e' unchanged if it is not a prod,
    // or if it is a prod that does not need to be split.
    // The latter condition is also used to terminate the
    // recursion.
    if (prod_ptr == nullptr || prod_ptr->args().size() <= split) {
        return e;
    }

    // NOTE: ret_seq will be a list
    // of prods each containing 'split' terms.
    // tmp is a temporary vector
    // used to accumulate the arguments for each
    // prod in ret_seq.
    std::vector<expression> ret_seq, tmp;
    for (const auto &arg : prod_ptr->args()) {
        tmp.push_back(arg);

        if (tmp.size() == split) {
            ret_seq.emplace_back(func{detail::prod_impl{std::move(tmp)}});

            // NOTE: tmp is practically guaranteed to be empty, but let's
            // be paranoid.
            tmp.clear();
        }
    }

    // NOTE: tmp is not empty if 'split' does not divide
    // exactly prod_ptr->args().size(). In such a case, we need to do the
    // last iteration manually.
    if (!tmp.empty()) {
        // NOTE: contrary to the previous loop, here we could
        // in principle end up creating a prod_impl with only one
        // term. We don't want to create such a prod as it would
        // break the Taylor diff implementations (which are all assuming
        // binary products).
        if (tmp.size() == 1u) {
            ret_seq.push_back(std::move(tmp[0]));
        } else {
            ret_seq.emplace_back(func{detail::prod_impl{std::move(tmp)}});
        }
    }

    // Recurse to split further, if needed.
    return prod_split(expression{func{detail::prod_impl{std::move(ret_seq)}}}, split);
}

} // namespace detail

expression prod(const std::vector<expression> &args_)
{
    auto args = detail::prod_simplify_args(args_);

    if (std::holds_alternative<expression>(args)) {
        return std::move(std::get<expression>(args));
    } else {
        return expression{func{detail::prod_impl{std::move(std::get<std::vector<expression>>(args))}}};
    }
}

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::prod_impl)