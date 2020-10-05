// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Helper to run the Taylor init phase of a unary
// function.
template <typename T>
llvm::Value *taylor_u_init_unary_func(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                      std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    auto &builder = s.builder();

    // Do the initialisation for the function argument.
    auto arg = taylor_u_init<T>(s, f.args()[0], arr, batch_size);

    // Decompose arg into scalars.
    auto scalars = vector_to_scalars(builder, arg);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> init_vals;
    for (auto scal : scalars) {
        init_vals.push_back(function_codegen_from_values<T>(s, f, {scal}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, init_vals);
}

// Helper to run the Taylor init phase of a unary
// function in compact mode.
template <typename T>
llvm::Value *taylor_c_u_init_unary_func(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    auto &builder = s.builder();

    // Do the initialisation for the function argument.
    auto arg = taylor_c_u_init<T>(s, f.args()[0], arr, batch_size);

    // Decompose arg into scalars.
    auto scalars = vector_to_scalars(builder, arg);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> init_vals;
    for (auto scal : scalars) {
        init_vals.push_back(function_codegen_from_values<T>(s, f, {scal}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, init_vals);
}

template <typename T>
llvm::Value *taylor_u_init_sin(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the sine (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of sin(number).
template <typename T>
llvm::Value *taylor_diff_sin_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of sin(variable).
template <typename T>
llvm::Value *taylor_diff_sin_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                  std::uint32_t batch_size)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of sin() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    if (order == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Overflow in the Taylor derivative of sin()");
    }
    std::vector<llvm::Value *> sum;
    auto &builder = s.builder();
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the +1 is because we are accessing the cosine
        // of the u var, which is conventionally placed
        // right after the sine in the decomposition.
        auto v0 = taylor_fetch_diff(arr, idx + 1u, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

        // Add j*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Compute and return the result: ret_acc / order
    auto div = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_sin_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a sine");
}

template <typename T>
llvm::Value *taylor_diff_sin(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sine (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_diff_sin_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

template <typename T>
llvm::Value *taylor_c_u_init_sin(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the sine in compact mode (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_c_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of sin(number).
template <typename T>
llvm::Value *taylor_c_diff_sin_impl(llvm_state &s, const number &, llvm::Value *, std::uint32_t, llvm::Value *,
                                    std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of sin(variable).
template <typename T>
llvm::Value *taylor_c_diff_sin_impl(llvm_state &s, const variable &var, llvm::Value *diff_arr, std::uint32_t n_uvars,
                                    llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the pointee type of diff_arr.
    auto val_t = pointee_type(diff_arr);

    // Get the function name for the current fp type and batch size.
    const auto fname = "heyoka_taylor_diff_sin_" + taylor_mangle_suffix(val_t);

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Prepare the function prototype. The arguments:
        // - indices of the variables,
        // - derivative order,
        // - diff array.
        std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                        llvm::Type::getInt32Ty(context), diff_arr->getType()};
        // The return type is the pointee type of diff_arr.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the function arguments.
        auto idx0 = f->args().begin();
        auto idx1 = f->args().begin() + 1;
        auto ord = f->args().begin() + 2;
        auto diff_ptr = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        auto bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        // Create an FP vector version of the order.
        auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            // NOTE: the +1 is because we are accessing the cosine
            // of the u var, which is conventionally placed
            // right after the sine in the decomposition.
            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j),
                                           builder.CreateAdd(idx0, builder.getInt32(1)));
            auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx1);

            auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

            builder.CreateStore(
                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(j_v, builder.CreateFMul(a_nj, cj))),
                acc);
        });

        // Divide by the order to produce the return value.
        builder.CreateRet(builder.CreateFDiv(builder.CreateLoad(acc), ord_v));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    // Invoke the function.
    return builder.CreateCall(f,
                              {builder.getInt32(idx), builder.getInt32(uname_to_index(var.name())), order, diff_arr});
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_c_diff_sin_impl(llvm_state &, const U &, llvm::Value *, std::uint32_t, llvm::Value *, std::uint32_t,
                                    std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a sine in compact mode");
}

template <typename T>
llvm::Value *taylor_c_diff_sin(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                               llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sine in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_sin_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression sin(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.sin";
    fc.name_ldbl() = "llvm.sin";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_sin128";
#endif
    fc.display_name() = "sin";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the sine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return cos(args[0]) * diff(args[0], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the sine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sin(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the sine in batches (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::sin(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "sine over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sin(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing "
                                        "the derivative of std::sin");
        }

        return std::cos(args[0]);
    };
    // NOTE: for sine/cosine we need a non-default decomposition because
    // we always need both sine *and* cosine in the decomposition
    // in order to compute the derivatives.
    fc.taylor_decompose_f() = [](function &&f, std::vector<expression> &u_vars_defs) {
        if (f.args().size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the Taylor decomposition of "
                                        "the sine (1 argument was expected, but "
                                        + std::to_string(f.args().size()) + " arguments were provided");
        }

        // Decompose the argument.
        auto &arg = f.args()[0];
        if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
            arg = expression{variable{"u_" + detail::li_to_string(dres)}};
        }

        // Save a copy of the decomposed argument.
        auto f_arg = arg;

        // Append the sine decomposition.
        u_vars_defs.emplace_back(std::move(f));

        // Compute the return value (pointing to the
        // decomposed sine).
        const auto retval = u_vars_defs.size() - 1u;

        // Append the cosine decomposition.
        u_vars_defs.emplace_back(cos(std::move(f_arg)));

        return retval;
    };
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_sin<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_sin<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_sin<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sin<mppp::real128>;
#endif
    fc.taylor_c_u_init_dbl_f() = detail::taylor_c_u_init_sin<double>;
    fc.taylor_c_u_init_ldbl_f() = detail::taylor_c_u_init_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_u_init_f128_f() = detail::taylor_c_u_init_sin<mppp::real128>;
#endif
    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_sin<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_sin<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_u_init_cos(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the cosine (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of cos(number).
template <typename T>
llvm::Value *taylor_diff_cos_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

template <typename T>
llvm::Value *taylor_diff_cos_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                  std::uint32_t batch_size)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of cos() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    if (order == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Overflow in the Taylor derivative of cos()");
    }
    std::vector<llvm::Value *> sum;
    auto &builder = s.builder();
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the -1 is because we are accessing the sine
        // of the u var, which is conventionally placed
        // right before the cosine in the decomposition.
        auto v0 = taylor_fetch_diff(arr, idx - 1u, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

        // Add j*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Compute and return the result: -ret_acc / order
    auto div = vector_splat(builder, codegen<T>(s, number(-static_cast<T>(order))), batch_size);

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_cos_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a cosine");
}

template <typename T>
llvm::Value *taylor_diff_cos(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the cosine (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_diff_cos_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

template <typename T>
llvm::Value *taylor_c_u_init_cos(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the cosine in compact mode (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_c_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of cos(number).
template <typename T>
llvm::Value *taylor_c_diff_cos_impl(llvm_state &s, const number &, llvm::Value *, std::uint32_t, llvm::Value *,
                                    std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of cos(variable).
template <typename T>
llvm::Value *taylor_c_diff_cos_impl(llvm_state &s, const variable &var, llvm::Value *diff_arr, std::uint32_t n_uvars,
                                    llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the pointee type of diff_arr.
    auto val_t = pointee_type(diff_arr);

    // Get the function name for the current fp type and batch size.
    const auto fname = "heyoka_taylor_diff_cos_" + taylor_mangle_suffix(val_t);

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Prepare the function prototype. The arguments:
        // - indices of the variables,
        // - derivative order,
        // - diff array.
        std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                        llvm::Type::getInt32Ty(context), diff_arr->getType()};
        // The return type is the pointee type of diff_arr.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the function arguments.
        auto idx0 = f->args().begin();
        auto idx1 = f->args().begin() + 1;
        auto ord = f->args().begin() + 2;
        auto diff_ptr = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        auto bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        // Create an FP vector version of the order.
        auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            // NOTE: the -1 is because we are accessing the sine
            // of the u var, which is conventionally placed
            // right before the cosine in the decomposition.
            auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j),
                                           builder.CreateSub(idx0, builder.getInt32(1)));
            auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx1);

            auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

            builder.CreateStore(
                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(j_v, builder.CreateFMul(b_nj, cj))),
                acc);
        });

        // Divide by the order and negate to produce the return value.
        builder.CreateRet(builder.CreateFDiv(builder.CreateLoad(acc), builder.CreateFNeg(ord_v)));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    // Invoke the function.
    return builder.CreateCall(f,
                              {builder.getInt32(idx), builder.getInt32(uname_to_index(var.name())), order, diff_arr});
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_c_diff_cos_impl(llvm_state &, const U &, llvm::Value *, std::uint32_t, llvm::Value *, std::uint32_t,
                                    std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a cosine in compact mode");
}

template <typename T>
llvm::Value *taylor_c_diff_cos(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                               llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the cosine in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_cos_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression cos(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.cos";
    fc.name_ldbl() = "llvm.cos";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_cos128";
#endif
    fc.display_name() = "cos";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when taking the derivative of the cosine (1 "
                                        "argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return -sin(args[0]) * diff(args[0], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the cosine from doubles (1 "
                                        "argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::cos(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the cosine in batches of "
                                        "doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::cos(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "cosine over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::cos(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments or derivative requested when computing the derivative of std::cos");
        }

        return -std::sin(args[0]);
    };
    fc.taylor_decompose_f() = [](function &&f, std::vector<expression> &u_vars_defs) {
        if (f.args().size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the Taylor decomposition of "
                                        "the cosine (1 argument was expected, but "
                                        + std::to_string(f.args().size()) + " arguments were provided");
        }

        // Decompose the argument.
        auto &arg = f.args()[0];
        if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
            arg = expression{variable{"u_" + detail::li_to_string(dres)}};
        }

        // Append the sine decomposition.
        u_vars_defs.emplace_back(sin(arg));

        // Append the cosine decomposition.
        u_vars_defs.emplace_back(std::move(f));

        return u_vars_defs.size() - 1u;
    };
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_cos<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_cos<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_cos<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_cos<mppp::real128>;
#endif
    fc.taylor_c_u_init_dbl_f() = detail::taylor_c_u_init_cos<double>;
    fc.taylor_c_u_init_ldbl_f() = detail::taylor_c_u_init_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_u_init_f128_f() = detail::taylor_c_u_init_cos<mppp::real128>;
#endif
    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_cos<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_cos<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_u_init_log(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the logarithm (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of log(number).
template <typename T>
llvm::Value *taylor_diff_log_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of log(variable).
template <typename T>
llvm::Value *taylor_diff_log_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                  std::uint32_t batch_size)
{
    // NOTE: not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of log() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    auto &builder = s.builder();

    // The result of the summation.
    llvm::Value *ret_acc;

    // NOTE: iteration in the [1, order) range
    // (i.e., order excluded). If order is 1,
    // we need to special case as the pairwise
    // summation function requires a series
    // with at least 1 element.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;
        for (std::uint32_t j = 1; j < order; ++j) {
            auto v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order - j))), batch_size);

            // Add (order-j)*v0*v1 to the sum.
            sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
        }

        // Compute the result of the summation.
        ret_acc = pairwise_sum(builder, sum);
    } else {
        // If the order is 1, the summation will be empty.
        // Init the result of the summation with zero.
        ret_acc = vector_splat(builder, codegen<T>(s, number(0.)), batch_size);
    }

    // Finalise the return value: (b^[n] - ret_acc / n) / b^[0]
    auto bn = taylor_fetch_diff(arr, u_idx, order, n_uvars);
    auto b0 = taylor_fetch_diff(arr, u_idx, 0, n_uvars);

    auto div = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);

    return builder.CreateFDiv(builder.CreateFSub(bn, builder.CreateFDiv(ret_acc, div)), b0);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_log_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a logarithm");
}

template <typename T>
llvm::Value *taylor_diff_log(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the logarithm (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_diff_log_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

template <typename T>
llvm::Value *taylor_c_u_init_log(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the logarithm in compact mode (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_c_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of log(number).
template <typename T>
llvm::Value *taylor_c_diff_log_impl(llvm_state &s, const number &, llvm::Value *, std::uint32_t, llvm::Value *,
                                    std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of log(variable).
template <typename T>
llvm::Value *taylor_c_diff_log_impl(llvm_state &s, const variable &var, llvm::Value *diff_arr, std::uint32_t n_uvars,
                                    llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the pointee type of diff_arr.
    auto val_t = pointee_type(diff_arr);

    // Get the function name for the current fp type and batch size.
    const auto fname = "heyoka_taylor_diff_log_" + taylor_mangle_suffix(val_t);

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Prepare the function prototype. The arguments:
        // - indices of the variables,
        // - derivative order,
        // - diff array.
        std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                        llvm::Type::getInt32Ty(context), diff_arr->getType()};
        // The return type is the pointee type of diff_arr.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the function arguments.
        auto idx0 = f->args().begin();
        auto idx1 = f->args().begin() + 1;
        auto ord = f->args().begin() + 2;
        auto diff_ptr = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        auto bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        // Create an FP vector version of the order.
        auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx0);
            auto bj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx1);

            // Compute the factor n - j.
            auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
            auto fac = builder.CreateFSub(ord_v, j_v);

            builder.CreateStore(
                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(fac, builder.CreateFMul(a_nj, bj))),
                acc);
        });

        // ret = bn - acc / n.
        auto ret = builder.CreateFSub(taylor_c_load_diff(s, diff_ptr, n_uvars, ord, idx1),
                                      builder.CreateFDiv(builder.CreateLoad(acc), ord_v));

        // Return ret / b0.
        builder.CreateRet(builder.CreateFDiv(ret, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), idx1)));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    // Invoke the function.
    return builder.CreateCall(f,
                              {builder.getInt32(idx), builder.getInt32(uname_to_index(var.name())), order, diff_arr});
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_c_diff_log_impl(llvm_state &, const U &, llvm::Value *, std::uint32_t, llvm::Value *, std::uint32_t,
                                    std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a logarithm in compact mode");
}

template <typename T>
llvm::Value *taylor_c_diff_log(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                               llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the logarithm in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_log_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression log(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.log";
    fc.name_ldbl() = "llvm.log";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_log128";
#endif
    fc.display_name() = "log";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the logarithm (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return expression{number(1.)} / args[0] * diff(args[0], s);
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the logarithm from doubles (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::log(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the logarithm in batches of "
                                        "doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::log(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "logarithm over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::log(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments or derivative requested when computing the derivative of std::log");
        }

        return 1. / args[0];
    };
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_log<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_log<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_log<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_log<mppp::real128>;
#endif
    fc.taylor_c_u_init_dbl_f() = detail::taylor_c_u_init_log<double>;
    fc.taylor_c_u_init_ldbl_f() = detail::taylor_c_u_init_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_u_init_f128_f() = detail::taylor_c_u_init_log<mppp::real128>;
#endif
    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_log<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_log<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_u_init_exp(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the exponential (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of exp(number).
template <typename T>
llvm::Value *taylor_diff_exp_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of exp(variable).
template <typename T>
llvm::Value *taylor_diff_exp_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                  std::uint32_t batch_size)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of exp() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [0, order) range
    // (i.e., order excluded).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        auto v0 = taylor_fetch_diff(arr, idx, j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order - j))), batch_size);

        // Add (order-j)*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Finalise the return value: ret_acc / n.
    auto div = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_exp_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of an exponential");
}

template <typename T>
llvm::Value *taylor_diff_exp(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the exponential (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_diff_exp_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

template <typename T>
llvm::Value *taylor_c_u_init_exp(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the exponential in compact mode (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_c_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of exp(number).
template <typename T>
llvm::Value *taylor_c_diff_exp_impl(llvm_state &s, const number &, llvm::Value *, std::uint32_t, llvm::Value *,
                                    std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of exp(variable).
template <typename T>
llvm::Value *taylor_c_diff_exp_impl(llvm_state &s, const variable &var, llvm::Value *diff_arr, std::uint32_t n_uvars,
                                    llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the pointee type of diff_arr.
    auto val_t = pointee_type(diff_arr);

    // Get the function name for the current fp type and batch size.
    const auto fname = "heyoka_taylor_diff_exp_" + taylor_mangle_suffix(val_t);

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Prepare the function prototype. The arguments:
        // - indices of the variables,
        // - derivative order,
        // - diff array.
        std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                        llvm::Type::getInt32Ty(context), diff_arr->getType()};
        // The return type is the pointee type of diff_arr.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the function arguments.
        auto idx0 = f->args().begin();
        auto idx1 = f->args().begin() + 1;
        auto ord = f->args().begin() + 2;
        auto diff_ptr = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        auto bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        // Create an FP vector version of the order.
        auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
            auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx0);
            auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx1);

            // Compute the factor n - j.
            auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
            auto fac = builder.CreateFSub(ord_v, j_v);

            builder.CreateStore(
                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(fac, builder.CreateFMul(aj, b_nj))),
                acc);
        });

        // Return acc / n.
        builder.CreateRet(builder.CreateFDiv(builder.CreateLoad(acc), ord_v));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    // Invoke the function.
    return builder.CreateCall(f,
                              {builder.getInt32(idx), builder.getInt32(uname_to_index(var.name())), order, diff_arr});
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_c_diff_exp_impl(llvm_state &, const U &, llvm::Value *, std::uint32_t, llvm::Value *, std::uint32_t,
                                    std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an exponential in compact mode");
}

template <typename T>
llvm::Value *taylor_c_diff_exp(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                               llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the exponential in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_exp_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression exp(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.exp";
    fc.name_ldbl() = "llvm.exp";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_exp128";
#endif
    fc.display_name() = "exp";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the exponential (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return exp(args[0]) * diff(args[0], s);
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the exponential from doubles (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::exp(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the exponential in batches of "
                "doubles (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::exp(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "exponential over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::exp(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                        "derivative of std::exp over doubles");
        }

        return std::exp(args[0]);
    };
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_exp<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_exp<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_exp<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_exp<mppp::real128>;
#endif
    fc.taylor_c_u_init_dbl_f() = detail::taylor_c_u_init_exp<double>;
    fc.taylor_c_u_init_ldbl_f() = detail::taylor_c_u_init_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_u_init_f128_f() = detail::taylor_c_u_init_exp<mppp::real128>;
#endif
    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_exp<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_exp<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_u_init_pow(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    if (f.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the pow() function (2 arguments were expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    auto &builder = s.builder();

    // Do the initialisation for the function arguments.
    auto arg0 = taylor_u_init<T>(s, f.args()[0], arr, batch_size);
    auto arg1 = taylor_u_init<T>(s, f.args()[1], arr, batch_size);

    // Decompose arg into scalars.
    auto scalars0 = vector_to_scalars(builder, arg0);
    auto scalars1 = vector_to_scalars(builder, arg1);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> init_vals;
    for (decltype(scalars0.size()) i = 0; i < scalars0.size(); ++i) {
        init_vals.push_back(function_codegen_from_values<T>(s, f, {scalars0[i], scalars1[i]}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, init_vals);
}

// Derivative of pow(number, number).
template <typename T>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, const number &, const number &, const std::vector<llvm::Value *> &,
                                  std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of pow(variable, number).
template <typename T>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, const variable &var, const number &num,
                                  const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars, std::uint32_t order,
                                  std::uint32_t idx, std::uint32_t batch_size)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of pow() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [0, order) range
    // (i.e., order *not* included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

        // Compute the scalar factor: order * num - j * (num + 1).
        auto scal_f = vector_splat(builder,
                                   codegen<T>(s, number(static_cast<T>(order)) * num
                                                     - number(static_cast<T>(j)) * (num + number(static_cast<T>(1)))),
                                   batch_size);

        // Add scal_f*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(scal_f, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Compute the final divisor: order * (zero-th derivative of u_idx).
    auto ord_f = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);
    auto b0 = taylor_fetch_diff(arr, u_idx, 0, n_uvars);
    auto div = builder.CreateFMul(ord_f, b0);

    // Compute and return the result: ret_acc / div.
    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U1, typename U2>
llvm::Value *taylor_diff_pow_impl(llvm_state &, const U1 &, const U2 &, const std::vector<llvm::Value *> &,
                                  std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a pow()");
}

template <typename T>
llvm::Value *taylor_diff_pow(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "pow() (2 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_pow_impl<T>(s, v1, v2, arr, n_uvars, order, idx, batch_size);
        },
        func.args()[0].value(), func.args()[1].value());
}

template <typename T>
llvm::Value *taylor_c_u_init_pow(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if (f.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the pow() function in compact mode (2 arguments were expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    auto &builder = s.builder();

    // Do the initialisation for the function arguments.
    auto arg0 = taylor_c_u_init<T>(s, f.args()[0], arr, batch_size);
    auto arg1 = taylor_c_u_init<T>(s, f.args()[1], arr, batch_size);

    // Decompose arg into scalars.
    auto scalars0 = vector_to_scalars(builder, arg0);
    auto scalars1 = vector_to_scalars(builder, arg1);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> init_vals;
    for (decltype(scalars0.size()) i = 0; i < scalars0.size(); ++i) {
        init_vals.push_back(function_codegen_from_values<T>(s, f, {scalars0[i], scalars1[i]}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, init_vals);
}

// Derivative of pow(number, number).
template <typename T>
llvm::Value *taylor_c_diff_pow_impl(llvm_state &s, const number &, const number &, llvm::Value *, std::uint32_t,
                                    llvm::Value *, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of pow(variable, number).
template <typename T>
llvm::Value *taylor_c_diff_pow_impl(llvm_state &s, const variable &var, const number &num, llvm::Value *diff_arr,
                                    std::uint32_t n_uvars, llvm::Value *order, std::uint32_t idx,
                                    std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the pointee type of diff_arr.
    auto val_t = pointee_type(diff_arr);

    // Get the function name for the current fp type and batch size.
    const auto fname = "heyoka_taylor_diff_pow_" + taylor_mangle_suffix(val_t);

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Prepare the function prototype. The arguments:
        // - indices of the variables,
        // - exponent,
        // - derivative order,
        // - diff array.
        std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                        to_llvm_type<T>(context), llvm::Type::getInt32Ty(context), diff_arr->getType()};
        // The return type is the pointee type of diff_arr.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the function arguments.
        auto idx0 = f->args().begin();
        auto idx1 = f->args().begin() + 1;
        auto exponent = f->args().begin() + 2;
        auto ord = f->args().begin() + 3;
        auto diff_ptr = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        auto bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        // Create FP vector versions of exponent and order.
        auto alpha_v = vector_splat(builder, exponent, batch_size);
        auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
            auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx0);
            auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, idx1);

            // Compute the factor n*alpha-j*(alpha+1).
            auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
            auto fac = builder.CreateFSub(
                builder.CreateFMul(ord_v, alpha_v),
                builder.CreateFMul(
                    j_v, builder.CreateFAdd(alpha_v, vector_splat(builder, codegen<T>(s, number{1.}), batch_size))));

            builder.CreateStore(
                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(fac, builder.CreateFMul(b_nj, aj))),
                acc);
        });

        // Finalize the result: acc / (n*b0).
        builder.CreateRet(builder.CreateFDiv(
            builder.CreateLoad(acc),
            builder.CreateFMul(ord_v, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), idx0))));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    // Invoke the function.
    return builder.CreateCall(
        f, {builder.getInt32(uname_to_index(var.name())), builder.getInt32(idx), codegen<T>(s, num), order, diff_arr});
}

// All the other cases.
template <typename T, typename U1, typename U2>
llvm::Value *taylor_c_diff_pow_impl(llvm_state &, const U1 &, const U2 &, llvm::Value *, std::uint32_t, llvm::Value *,
                                    std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a pow() in compact mode");
}

template <typename T>
llvm::Value *taylor_c_diff_pow(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                               llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "pow() in compact mode (2 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_c_diff_pow_impl<T>(s, v1, v2, arr, n_uvars, order, idx, batch_size);
        },
        func.args()[0].value(), func.args()[1].value());
}

} // namespace

} // namespace detail

expression pow(expression e1, expression e2)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e1));
    args.emplace_back(std::move(e2));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.pow";
    fc.name_ldbl() = "llvm.pow";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_pow128";
#endif
    fc.display_name() = "pow";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif

    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 2u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the exponentiation (2 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        return args[1] * pow(args[0], args[1] - expression{number(1.)}) * diff(args[0], s)
               + pow(args[0], args[1]) * log(args[0]) * diff(args[1], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 2u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the exponentiation from doubles (2 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        return std::pow(eval_dbl(args[0], map), eval_dbl(args[1], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the exponentiation in "
                                        "batches of doubles (2 arguments were expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }
        auto out0 = out; // is this allocation needed?
        eval_batch_dbl(out0, args[0], map);
        eval_batch_dbl(out, args[1], map);
        for (decltype(out.size()) i = 0u; i < out.size(); ++i) {
            out[i] = std::pow(out0[i], out[i]);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "exponentiation over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::pow(args[0], args[1]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 2u || i > 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments or derivative requested when computing the derivative of std::pow");
        }
        return args[1] * std::pow(args[0], args[1] - 1.) + std::log(args[0]) * std::pow(args[0], args[1]);
    };
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_pow<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_pow<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_pow<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_pow<mppp::real128>;
#endif
    fc.taylor_c_u_init_dbl_f() = detail::taylor_c_u_init_pow<double>;
    fc.taylor_c_u_init_ldbl_f() = detail::taylor_c_u_init_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_u_init_f128_f() = detail::taylor_c_u_init_pow<mppp::real128>;
#endif
    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_pow<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_pow<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_u_init_sqrt(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the square root (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of sqrt(number).
template <typename T>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                   std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of sqrt(variable).
template <typename T>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    return taylor_diff_pow_impl<T>(s, var, number{T(1) / 2}, arr, n_uvars, order, idx, batch_size);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                   std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a square root");
}

template <typename T>
llvm::Value *taylor_diff_sqrt(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the square root (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_diff_sqrt_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

template <typename T>
llvm::Value *taylor_c_u_init_sqrt(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the square root in compact mode (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_c_u_init_unary_func<T>(s, f, arr, batch_size);
}

// Derivative of sqrt(number).
template <typename T>
llvm::Value *taylor_c_diff_sqrt_impl(llvm_state &s, const number &, llvm::Value *, std::uint32_t, llvm::Value *,
                                     std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of sqrt(variable).
template <typename T>
llvm::Value *taylor_c_diff_sqrt_impl(llvm_state &s, const variable &var, llvm::Value *arr, std::uint32_t n_uvars,
                                     llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    return taylor_c_diff_pow_impl<T>(s, var, number{T(1) / 2}, arr, n_uvars, order, idx, batch_size);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_c_diff_sqrt_impl(llvm_state &, const U &, llvm::Value *, std::uint32_t, llvm::Value *,
                                     std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a square root in compact mode");
}

template <typename T>
llvm::Value *taylor_c_diff_sqrt(llvm_state &s, const function &func, llvm::Value *arr, std::uint32_t n_uvars,
                                llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the square root in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_sqrt_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression sqrt(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.sqrt";
    fc.name_ldbl() = "llvm.sqrt";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_sqrt128";
#endif
    fc.display_name() = "sqrt";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the square root (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return diff(args[0], s) / (expression{number(2.)} * sqrt(args[0]));
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the square root from doubles (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sqrt(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the square root in batches of "
                "doubles (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::sqrt(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "square root over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sqrt(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                        "derivative of std::sqrt over doubles");
        }

        return std::sqrt(args[0]);
    };

    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_sqrt<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_sqrt<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_sqrt<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_sqrt<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sqrt<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sqrt<mppp::real128>;
#endif
    fc.taylor_c_u_init_dbl_f() = detail::taylor_c_u_init_sqrt<double>;
    fc.taylor_c_u_init_ldbl_f() = detail::taylor_c_u_init_sqrt<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_u_init_f128_f() = detail::taylor_c_u_init_sqrt<mppp::real128>;
#endif
    fc.taylor_c_diff_dbl_f() = detail::taylor_c_diff_sqrt<double>;
    fc.taylor_c_diff_ldbl_f() = detail::taylor_c_diff_sqrt<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_f128_f() = detail::taylor_c_diff_sqrt<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
