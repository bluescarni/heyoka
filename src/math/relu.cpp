// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/math/relu.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Checker for the slope parameter of the leaky ReLU.
void relu_slope_check(double slope)
{
    if (!std::isfinite(slope) || slope < 0) {
        throw std::invalid_argument(fmt::format("The slope parameter for a leaky ReLU must be finite and non-negative, "
                                                "but the value {} was provided instead",
                                                slope));
    }
}

// Helper to build a unique name for a relu/relup function, depending
// on the slope value.
std::string relu_name(const char *base, double slope)
{
    if (slope == 0) {
        return base;
    } else {
        // NOTE: we print the slope value in hex format, then we replace
        // the decimal point '.' with an underscore '_' (as the '.' is used
        // as a separator in the name mangling scheme for compact mode functions).
        auto ret = fmt::format("{}_{:a}", base, slope);
        std::replace(ret.begin(), ret.end(), '.', '_');

        return ret;
    }
}

} // namespace

relu_impl::relu_impl() : relu_impl(0_dbl, 0.) {}

relu_impl::relu_impl(expression ex, double slope)
    : func_base(relu_name("relu", slope), std::vector{std::move(ex)}), m_slope(slope)
{
    relu_slope_check(slope);
}

double relu_impl::get_slope() const noexcept
{
    return m_slope;
}

void relu_impl::to_stream(std::ostringstream &oss) const
{
    assert(args().size() == 1u);

    if (m_slope == 0) {
        oss << "relu(";
        stream_expression(oss, args()[0]);
        oss << ')';
    } else {
        oss << "leaky_relu(";
        stream_expression(oss, args()[0]);
        oss << fmt::format(", {})", m_slope);
    }
}

[[nodiscard]] std::vector<expression> relu_impl::gradient() const
{
    assert(args().size() == 1u);
    return {relup(args()[0], m_slope)};
}

namespace
{

// LLVM implementation of relu.
llvm::Value *llvm_relu(llvm_state &s, llvm::Value *x, double slope)
{
    auto *zero_c = llvm_constantfp(s, x->getType(), 0.);

    if (slope == 0) {
        return s.builder().CreateSelect(llvm_fcmp_ogt(s, x, zero_c), x, zero_c);
    } else {
        auto *slope_c = llvm_constantfp(s, x->getType(), slope);
        return s.builder().CreateSelect(llvm_fcmp_ogt(s, x, zero_c), x, llvm_fmul(s, slope_c, x));
    }
}

} // namespace

[[nodiscard]] llvm::Value *relu_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t,
                                                const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                                llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                                bool high_accuracy) const
{
    return llvm_eval_helper(
        [&](const std::vector<llvm::Value *> &args, bool) {
            assert(args.size() == 1u);
            return llvm_relu(s, args[0], m_slope);
        },
        *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *relu_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    return llvm_c_eval_func_helper(
        get_name(),
        [&](const std::vector<llvm::Value *> &args, bool) {
            assert(args.size() == 1u);
            return llvm_relu(s, args[0], m_slope);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

namespace
{

// Derivative of relu(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_relu_impl(llvm_state &s, llvm::Type *fp_t, const relu_impl &,
                                   const std::vector<std::uint32_t> &, const U &num, const std::vector<llvm::Value *> &,
                                   llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                   std::uint32_t batch_size, double slope)
{
    if (order == 0u) {
        return llvm_relu(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size), slope);
    } else {
        return vector_splat(s.builder(), llvm_constantfp(s, fp_t, 0.), batch_size);
    }
}

// Derivative of relu(variable).
llvm::Value *taylor_diff_relu_impl(llvm_state &s, llvm::Type *, const relu_impl &, const std::vector<std::uint32_t> &,
                                   const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t,
                                   double slope)
{
    const auto u_idx = uname_to_index(var.name());

    auto *u_zero = taylor_fetch_diff(arr, u_idx, 0, n_uvars);
    auto *u_order = taylor_fetch_diff(arr, u_idx, order, n_uvars);

    auto *zero_c = llvm_constantfp(s, u_zero->getType(), 0.);

    if (slope == 0) {
        return s.builder().CreateSelect(llvm_fcmp_ogt(s, u_zero, zero_c), u_order, zero_c);
    } else {
        auto *slope_c = llvm_constantfp(s, u_zero->getType(), slope);
        return s.builder().CreateSelect(llvm_fcmp_ogt(s, u_zero, zero_c), u_order, llvm_fmul(s, slope_c, u_order));
    }
}

// LCOV_EXCL_START

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_relu_impl(llvm_state &, llvm::Type *, const relu_impl &, const std::vector<std::uint32_t> &,
                                   const U &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                   std::uint32_t, std::uint32_t, std::uint32_t, double)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a relu");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Value *relu_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    // LCOV_EXCL_START
    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the relu, but a vector of size {} was passed instead",
                        deps.size()));
    }
    // LCOV_EXCL_STOP

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_relu_impl(s, fp_t, *this, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size,
                                         m_slope);
        },
        args()[0].value());
}

namespace
{

// Derivative of relu(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_relu_impl(llvm_state &s, llvm::Type *fp_t, const relu_impl &r, const U &num,
                                             // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                             std::uint32_t n_uvars, std::uint32_t batch_size, double slope)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, r.get_name(), 0,
        [&](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_relu(s, args[0], slope);
        },
        num);
}

// Derivative of relu(variable).
llvm::Function *taylor_c_diff_func_relu_impl(llvm_state &s, llvm::Type *fp_t, const relu_impl &r, const variable &var,
                                             // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                             std::uint32_t n_uvars, std::uint32_t batch_size, double slope)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, r.get_name(), n_uvars, batch_size, {var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = module.getFunction(fname);

    if (f != nullptr) {
        // The function was created before, return it.
        return f;
    }

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
    auto *diff_ptr = f->args().begin() + 2;
    auto *var_idx = f->args().begin() + 5;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Load the orders 0 and ord of var_idx.
    auto *u_zero = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx);
    auto *u_ord = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, var_idx);

    auto *zero_c = llvm_constantfp(s, u_zero->getType(), 0.);

    if (slope == 0) {
        builder.CreateRet(builder.CreateSelect(llvm_fcmp_ogt(s, u_zero, zero_c), u_ord, zero_c));
    } else {
        auto *slope_c = llvm_constantfp(s, u_zero->getType(), slope);
        builder.CreateRet(builder.CreateSelect(llvm_fcmp_ogt(s, u_zero, zero_c), u_ord, llvm_fmul(s, slope_c, u_ord)));
    }

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

// LCOV_EXCL_START

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_relu_impl(llvm_state &, llvm::Type *, const relu_impl &, const U &, std::uint32_t,
                                             std::uint32_t, double)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a relu in compact mode");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Function *relu_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                              std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_func_relu_impl(s, fp_t, *this, v, n_uvars, batch_size, m_slope); },
        args()[0].value());
}

relup_impl::relup_impl() : relup_impl(0_dbl, 0.) {}

relup_impl::relup_impl(expression ex, double slope)
    : func_base(relu_name("relup", slope), std::vector{std::move(ex)}), m_slope(slope)
{
    relu_slope_check(slope);
}

double relup_impl::get_slope() const noexcept
{
    return m_slope;
}

void relup_impl::to_stream(std::ostringstream &oss) const
{
    assert(args().size() == 1u);

    if (m_slope == 0) {
        oss << "relup(";
        stream_expression(oss, args()[0]);
        oss << ')';
    } else {
        oss << "leaky_relup(";
        stream_expression(oss, args()[0]);
        oss << fmt::format(", {})", m_slope);
    }
}

[[nodiscard]] std::vector<expression> relup_impl::gradient() const
{
    assert(args().size() == 1u);
    return {0_dbl};
}

namespace
{

// LLVM implementation of relup.
llvm::Value *llvm_relup(llvm_state &s, llvm::Value *x, double slope)
{
    auto *zero_c = llvm_constantfp(s, x->getType(), 0.);
    auto *one_c = llvm_constantfp(s, x->getType(), 1.);

    if (slope == 0) {
        return s.builder().CreateSelect(llvm_fcmp_ogt(s, x, zero_c), one_c, zero_c);
    } else {
        auto *slope_c = llvm_constantfp(s, x->getType(), slope);
        return s.builder().CreateSelect(llvm_fcmp_ogt(s, x, zero_c), one_c, slope_c);
    }
}

} // namespace

[[nodiscard]] llvm::Value *relup_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t,
                                                 const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                                 llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                                 bool high_accuracy) const
{
    return llvm_eval_helper(
        [&](const std::vector<llvm::Value *> &args, bool) {
            assert(args.size() == 1u);
            return llvm_relup(s, args[0], m_slope);
        },
        *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *relup_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                             bool high_accuracy) const
{
    return llvm_c_eval_func_helper(
        get_name(),
        [&](const std::vector<llvm::Value *> &args, bool) {
            assert(args.size() == 1u);
            return llvm_relup(s, args[0], m_slope);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

namespace
{

// Derivative of relup(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_relup_impl(llvm_state &s, llvm::Type *fp_t, const relup_impl &,
                                    const std::vector<std::uint32_t> &, const U &num,
                                    const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                    std::uint32_t order, std::uint32_t, std::uint32_t batch_size, double slope)
{
    if (order == 0u) {
        return llvm_relup(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size), slope);
    } else {
        return vector_splat(s.builder(), llvm_constantfp(s, fp_t, 0.), batch_size);
    }
}

// Derivative of relup(variable).
llvm::Value *taylor_diff_relup_impl(llvm_state &s, llvm::Type *, const relup_impl &, const std::vector<std::uint32_t> &,
                                    const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t,
                                    double slope)
{
    const auto u_idx = uname_to_index(var.name());
    auto *u_zero = taylor_fetch_diff(arr, u_idx, 0, n_uvars);

    if (order == 0u) {
        return llvm_relup(s, u_zero, slope);
    } else {
        return llvm_constantfp(s, u_zero->getType(), 0.);
    }
}

// LCOV_EXCL_START

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_relup_impl(llvm_state &, llvm::Type *, const relup_impl &, const std::vector<std::uint32_t> &,
                                    const U &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                    std::uint32_t, std::uint32_t, std::uint32_t, double)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a relup");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Value *relup_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                     std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                     std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    // LCOV_EXCL_START
    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the relup, but a vector of size {} was passed instead",
                        deps.size()));
    }
    // LCOV_EXCL_STOP

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_relup_impl(s, fp_t, *this, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size,
                                          m_slope);
        },
        args()[0].value());
}

namespace
{

// Derivative of relup(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_relup_impl(llvm_state &s, llvm::Type *fp_t, const relup_impl &r, const U &num,
                                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                              std::uint32_t n_uvars, std::uint32_t batch_size, double slope)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, r.get_name(), 0,
        [&](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_relup(s, args[0], slope);
        },
        num);
}

// Derivative of relup(variable).
llvm::Function *taylor_c_diff_func_relup_impl(llvm_state &s, llvm::Type *fp_t, const relup_impl &r, const variable &var,
                                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                              std::uint32_t n_uvars, std::uint32_t batch_size, double slope)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, r.get_name(), n_uvars, batch_size, {var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = module.getFunction(fname);

    if (f != nullptr) {
        // The function was created before, return it.
        return f;
    }

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
    auto *diff_ptr = f->args().begin() + 2;
    auto *var_idx = f->args().begin() + 5;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Create the return value.
    auto *retval = builder.CreateAlloca(val_t);

    llvm_if_then_else(
        s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
        [&]() {
            // For order 0, invoke the function on the order 0 of var_idx.
            auto *ret
                = llvm_relup(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx), slope);
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
            builder.CreateStore(ret, retval);
        },
        [&]() {
            // For all the other orders, the result is zero.
            builder.CreateStore(llvm_constantfp(s, val_t, 0.), retval);
        });

    // Return the result.
    builder.CreateRet(builder.CreateLoad(val_t, retval));

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

// LCOV_EXCL_START

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_relup_impl(llvm_state &, llvm::Type *, const relup_impl &, const U &, std::uint32_t,
                                              std::uint32_t, double)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a relup in compact mode");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Function *relup_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                               std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_func_relup_impl(s, fp_t, *this, v, n_uvars, batch_size, m_slope); },
        args()[0].value());
}

} // namespace detail

expression relu(expression x, double slope)
{
    detail::relu_slope_check(slope);

    // Fold relu(number) to its value.
    if (const auto *num_ptr = std::get_if<number>(&x.value())) {
        return std::visit([slope](const auto &x) { return expression{x > 0 ? x : slope * x}; }, num_ptr->value());
    } else {
        return expression{func{detail::relu_impl{std::move(x), slope}}};
    }
}

expression relup(expression x, double slope)
{
    detail::relu_slope_check(slope);

    // Fold relup(number) to its value.
    if (const auto *num_ptr = std::get_if<number>(&x.value())) {
        return std::visit([slope](const auto &x) { return expression{x > 0 ? 1. : slope}; }, num_ptr->value());
    } else {
        return expression{func{detail::relup_impl{std::move(x), slope}}};
    }
}

leaky_relu::leaky_relu(double slope) : m_slope(slope)
{
    detail::relu_slope_check(slope);
}

expression leaky_relu::operator()(expression x) const
{
    return relu(std::move(x), m_slope);
}

leaky_relup::leaky_relup(double slope) : m_slope(slope)
{
    detail::relu_slope_check(slope);
}

expression leaky_relup::operator()(expression x) const
{
    return relup(std::move(x), m_slope);
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::relu_impl)

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::relup_impl)
