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

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
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

// Derivative of sin(number).
template <typename T>
llvm::Function *taylor_c_diff_func_sin_impl(llvm_state &s, const function &func, const number &, std::uint32_t,
                                            std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num<T>(s, func, batch_size,
                                           "heyoka_taylor_diff_sin_num_"
                                               + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                           "the sine");
}

// Derivative of sin(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_sin_impl(llvm_state &s, const function &func, const variable &,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_sin_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)}),
                    retval);
            },
            [&]() {
                // Init the accumlator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    // NOTE: the +1 is because we are accessing the cosine
                    // of the u var, which is conventionally placed
                    // right after the sine in the decomposition.
                    auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j),
                                                   builder.CreateAdd(u_idx, builder.getInt32(1)));
                    auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx);

                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(j_v, builder.CreateFMul(a_nj, cj))),
                                        acc);
                });

                // Divide by the order to produce the return value.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);
                builder.CreateStore(builder.CreateFDiv(builder.CreateLoad(acc), ord_v), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
                "Inconsistent function signature for the Taylor derivative of the sine in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U>
llvm::Function *taylor_c_diff_func_sin_impl(llvm_state &, const function &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a sine in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_sin(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sine in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit([&](const auto &v) { return taylor_c_diff_func_sin_impl<T>(s, func, v, n_uvars, batch_size); },
                      func.args()[0].value());
}

} // namespace

} // namespace detail

expression sin(expression e)
{
    std::vector<expression> args;
    args.push_back(std::move(e));

    function fc{std::move(args)};
    fc.display_name() = "sin";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the double codegen of the sine "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
            if (const auto sfn
                = detail::sleef_function_name(s.context(), "sin", vec_t->getElementType(),
                                              boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return detail::llvm_invoke_external(
                    s, sfn, vec_t, args,
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.sin", {args[0]->getType()}, args);
    };
    fc.codegen_ldbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the long double codegen of the sine "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.sin", {args[0]->getType()}, args);
    };
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the float128 codegen of the sine "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        // Decompose the argument into scalars.
        auto scalars = detail::vector_to_scalars(builder, args[0]);

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> retvals;
        for (auto scal : scalars) {
            retvals.push_back(detail::llvm_invoke_external(
                s, "heyoka_sin128", scal->getType(), {scal},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Build a vector with the results.
        return detail::scalars_to_vector(builder, retvals);
    };
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
        u_vars_defs.push_back(cos(std::move(f_arg)));

        return retval;
    };
    fc.taylor_diff_dbl_f() = detail::taylor_diff_sin<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sin<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_sin<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_sin<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

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

// Derivative of cos(number).
template <typename T>
llvm::Function *taylor_c_diff_func_cos_impl(llvm_state &s, const function &func, const number &, std::uint32_t,
                                            std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num<T>(s, func, batch_size,
                                           "heyoka_taylor_diff_cos_num_"
                                               + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                           "the cosine");
}

// Derivative of cos(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_cos_impl(llvm_state &s, const function &func, const variable &,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_cos_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)}),
                    retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    // NOTE: the -1 is because we are accessing the sine
                    // of the u var, which is conventionally placed
                    // right before the cosine in the decomposition.
                    auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j),
                                                   builder.CreateSub(u_idx, builder.getInt32(1)));
                    auto cj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx);

                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(j_v, builder.CreateFMul(b_nj, cj))),
                                        acc);
                });

                // Divide by the order and negate to produce the return value.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);
                builder.CreateStore(builder.CreateFDiv(builder.CreateLoad(acc), builder.CreateFNeg(ord_v)), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
                "Inconsistent function signature for the Taylor derivative of the cosine in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U>
llvm::Function *taylor_c_diff_func_cos_impl(llvm_state &, const function &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a cosine in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_cos(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the cosine in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit([&](const auto &v) { return taylor_c_diff_func_cos_impl<T>(s, func, v, n_uvars, batch_size); },
                      func.args()[0].value());
}

} // namespace

} // namespace detail

expression cos(expression e)
{
    std::vector<expression> args;
    args.push_back(std::move(e));

    function fc{std::move(args)};
    fc.display_name() = "cos";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the double codegen of the cosine "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
            if (const auto sfn
                = detail::sleef_function_name(s.context(), "cos", vec_t->getElementType(),
                                              boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return detail::llvm_invoke_external(
                    s, sfn, vec_t, args,
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.cos", {args[0]->getType()}, args);
    };
    fc.codegen_ldbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the long double codegen of the cosine "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.cos", {args[0]->getType()}, args);
    };
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the float128 codegen of the cosine "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        // Decompose the argument into scalars.
        auto scalars = detail::vector_to_scalars(builder, args[0]);

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> retvals;
        for (auto scal : scalars) {
            retvals.push_back(detail::llvm_invoke_external(
                s, "heyoka_cos128", scal->getType(), {scal},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Build a vector with the results.
        return detail::scalars_to_vector(builder, retvals);
    };
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
        u_vars_defs.push_back(sin(arg));

        // Append the cosine decomposition.
        u_vars_defs.emplace_back(std::move(f));

        return u_vars_defs.size() - 1u;
    };
    fc.taylor_diff_dbl_f() = detail::taylor_diff_cos<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_cos<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_cos<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_cos<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

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

// Derivative of log(number).
template <typename T>
llvm::Function *taylor_c_diff_func_log_impl(llvm_state &s, const function &func, const number &, std::uint32_t,
                                            std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num<T>(s, func, batch_size,
                                           "heyoka_taylor_diff_log_num_"
                                               + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                           "the logarithm");
}

// Derivative of log(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_log_impl(llvm_state &s, const function &func, const variable &,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_log_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)}),
                    retval);
            },
            [&]() {
                // Create a vector version of ord.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
                    auto bj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx);

                    // Compute the factor n - j.
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
                    auto fac = builder.CreateFSub(ord_v, j_v);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(a_nj, bj))),
                                        acc);
                });

                // ret = bn - acc / n.
                auto ret = builder.CreateFSub(taylor_c_load_diff(s, diff_ptr, n_uvars, ord, var_idx),
                                              builder.CreateFDiv(builder.CreateLoad(acc), ord_v));

                // Return ret / b0.
                builder.CreateStore(
                    builder.CreateFDiv(ret, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
                "Inconsistent function signature for the Taylor derivative of the logarithm in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U>
llvm::Function *taylor_c_diff_func_log_impl(llvm_state &, const function &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a logarithm in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_log(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the logarithm in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit([&](const auto &v) { return taylor_c_diff_func_log_impl<T>(s, func, v, n_uvars, batch_size); },
                      func.args()[0].value());
}

} // namespace

} // namespace detail

expression log(expression e)
{
    std::vector<expression> args;
    args.push_back(std::move(e));

    function fc{std::move(args)};
    fc.display_name() = "log";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the double codegen of the logarithm "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
            if (const auto sfn
                = detail::sleef_function_name(s.context(), "log", vec_t->getElementType(),
                                              boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return detail::llvm_invoke_external(
                    s, sfn, vec_t, args,
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.log", {args[0]->getType()}, args);
    };
    fc.codegen_ldbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Invalid number of arguments passed to the long double codegen of the logarithm "
                "function: 1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were passed instead");
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.log", {args[0]->getType()}, args);
    };
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the float128 codegen of the logarithm "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        // Decompose the argument into scalars.
        auto scalars = detail::vector_to_scalars(builder, args[0]);

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> retvals;
        for (auto scal : scalars) {
            retvals.push_back(detail::llvm_invoke_external(
                s, "heyoka_log128", scal->getType(), {scal},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Build a vector with the results.
        return detail::scalars_to_vector(builder, retvals);
    };
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
    fc.taylor_diff_dbl_f() = detail::taylor_diff_log<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_log<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_log<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_log<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

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

// Derivative of exp(number).
template <typename T>
llvm::Function *taylor_c_diff_func_exp_impl(llvm_state &s, const function &func, const number &, std::uint32_t,
                                            std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num<T>(s, func, batch_size,
                                           "heyoka_taylor_diff_exp_num_"
                                               + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                           "the exponential");
}

// Derivative of exp(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_exp_impl(llvm_state &s, const function &func, const variable &,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_exp_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)}),
                    retval);
            },
            [&]() {
                // Create a vector version of ord.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
                    auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, u_idx);
                    auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);

                    // Compute the factor n - j.
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
                    auto fac = builder.CreateFSub(ord_v, j_v);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(aj, b_nj))),
                                        acc);
                });

                // Return acc / n.
                builder.CreateStore(builder.CreateFDiv(builder.CreateLoad(acc), ord_v), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of the exponential "
                                        "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U>
llvm::Function *taylor_c_diff_func_exp_impl(llvm_state &, const function &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an exponential in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_exp(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the exponential in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit([&](const auto &v) { return taylor_c_diff_func_exp_impl<T>(s, func, v, n_uvars, batch_size); },
                      func.args()[0].value());
}

} // namespace

} // namespace detail

expression exp(expression e)
{
    std::vector<expression> args;
    args.push_back(std::move(e));

    function fc{std::move(args)};
    fc.display_name() = "exp";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the double codegen of the exponential "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
            if (const auto sfn
                = detail::sleef_function_name(s.context(), "exp", vec_t->getElementType(),
                                              boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return detail::llvm_invoke_external(
                    s, sfn, vec_t, args,
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.exp", {args[0]->getType()}, args);
    };
    fc.codegen_ldbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Invalid number of arguments passed to the long double codegen of the exponential "
                "function: 1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were passed instead");
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.exp", {args[0]->getType()}, args);
    };
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the float128 codegen of the exponential "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        // Decompose the argument into scalars.
        auto scalars = detail::vector_to_scalars(builder, args[0]);

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> retvals;
        for (auto scal : scalars) {
            retvals.push_back(detail::llvm_invoke_external(
                s, "heyoka_exp128", scal->getType(), {scal},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Build a vector with the results.
        return detail::scalars_to_vector(builder, retvals);
    };
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
    fc.taylor_diff_dbl_f() = detail::taylor_diff_exp<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_exp<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_exp<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_exp<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

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

// Derivative of pow(number, number).
template <typename T>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &s, const function &func, const number &, const number &,
                                            std::uint32_t, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname = "heyoka_taylor_diff_pow_num_num_" + taylor_mangle_suffix(val_t);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - base argument,
    // - exp argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), to_llvm_type<T>(context),
                                    to_llvm_type<T>(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto num_base = f->args().begin() + 3;
        auto num_exp = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, run the codegen.
                builder.CreateStore(codegen_from_values<T>(s, func,
                                                           {vector_splat(builder, num_base, batch_size),
                                                            vector_splat(builder, num_exp, batch_size)}),
                                    retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
                "Inconsistent function signature for the Taylor derivative of pow() in compact mode detected");
        }
    }

    return f;
}

// Derivative of pow(variable, number).
template <typename T>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &s, const function &func, const variable &, const number &,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_pow_var_num_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument,
    // - exp argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    to_llvm_type<T>(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 3;
        auto exponent = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx),
                                            vector_splat(builder, exponent, batch_size)}),
                    retval);
            },
            [&]() {
                // Create FP vector versions of exponent and order.
                auto alpha_v = vector_splat(builder, exponent, batch_size);
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
                    auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                    auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, u_idx);

                    // Compute the factor n*alpha-j*(alpha+1).
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
                    auto fac = builder.CreateFSub(
                        builder.CreateFMul(ord_v, alpha_v),
                        builder.CreateFMul(
                            j_v,
                            builder.CreateFAdd(alpha_v, vector_splat(builder, codegen<T>(s, number{1.}), batch_size))));

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(b_nj, aj))),
                                        acc);
                });

                // Finalize the result: acc / (n*b0).
                builder.CreateStore(
                    builder.CreateFDiv(builder.CreateLoad(acc),
                                       builder.CreateFMul(ord_v, taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                                                    builder.getInt32(0), var_idx))),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
                "Inconsistent function signatures for the Taylor derivative of pow() in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U1, typename U2>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &, const function &, const U1 &, const U2 &, std::uint32_t,
                                            std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a pow() in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_pow(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    if (func.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "pow() in compact mode (2 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_c_diff_func_pow_impl<T>(s, func, v1, v2, n_uvars, batch_size);
        },
        func.args()[0].value(), func.args()[1].value());
}

} // namespace

} // namespace detail

expression pow(expression e1, expression e2)
{
    // NOTE: we want to allow approximate implementations of pow()
    // in the following cases:
    // - e2 is an integral number n (in which case we want to allow
    //   transformation in a sequence of multiplications),
    // - e2 is a value of type n / 2, with n an odd integral value (in which case
    //   we want to give the option of implementing pow() on top of sqrt()).
    const auto allow_approx = detail::is_integral(e2) || detail::is_odd_integral_half(e2);

    std::vector<expression> args;
    args.push_back(std::move(e1));
    args.push_back(std::move(e2));

    function fc{std::move(args)};
    fc.display_name() = "pow";

    fc.codegen_dbl_f() = [allow_approx](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Invalid number of arguments passed to the double codegen of the pow "
                                        "function: 2 arguments were expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        // NOTE: we want to try the SLEEF route only if we are *not* approximating
        // pow() with sqrt() or iterated multiplications (in which case we are fine
        // with the LLVM builtin).
        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType()); !allow_approx && vec_t != nullptr) {
            if (const auto sfn
                = detail::sleef_function_name(s.context(), "pow", vec_t->getElementType(),
                                              boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return detail::llvm_invoke_external(
                    s, sfn, vec_t, args,
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        auto ret = detail::llvm_invoke_intrinsic(s, "llvm.pow", {args[0]->getType()}, args);
        if (allow_approx) {
            llvm::cast<llvm::CallInst>(ret)->setHasApproxFunc(true);
        }

        return ret;
    };
    fc.codegen_ldbl_f() = [allow_approx](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Invalid number of arguments passed to the long double codegen of the pow "
                                        "function: 2 arguments were expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto ret = detail::llvm_invoke_intrinsic(s, "llvm.pow", {args[0]->getType()}, args);
        if (allow_approx) {
            llvm::cast<llvm::CallInst>(ret)->setHasApproxFunc(true);
        }

        return ret;
    };
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Invalid number of arguments passed to the float128 codegen of the pow "
                                        "function: 2 arguments were expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        // Decompose the arguments into scalars.
        auto scalars0 = detail::vector_to_scalars(builder, args[0]);
        auto scalars1 = detail::vector_to_scalars(builder, args[1]);

        // Invoke the function on the scalars.
        std::vector<llvm::Value *> retvals;
        for (decltype(scalars0.size()) i = 0; i < scalars0.size(); ++i) {
            retvals.push_back(detail::llvm_invoke_external(
                s, "heyoka_pow128", scalars0[i]->getType(), {scalars0[i], scalars1[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Build a vector with the results.
        return detail::scalars_to_vector(builder, retvals);
    };
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
    fc.taylor_diff_dbl_f() = detail::taylor_diff_pow<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_pow<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_pow<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_pow<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

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

// Derivative of sqrt(number).
template <typename T>
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &s, const function &func, const number &, std::uint32_t,
                                             std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num<T>(s, func, batch_size,
                                           "heyoka_taylor_diff_sqrt_num_"
                                               + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                           "the square root");
}

// Derivative of sqrt(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &s, const function &func, const variable &,
                                             std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_sqrt_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - idx of the var argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx)}),
                    retval);
            },
            [&]() {
                // NOTE: this is copy-pasted from the pow() implementation,
                // and alpha_v hard-coded to 1/2. Perhaps we can avoid repetition
                // with some refactoring.
                // Create FP vector versions of exponent and order.
                auto alpha_v = vector_splat(builder, codegen<T>(s, number{T(1) / 2}), batch_size);
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
                    auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                    auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, u_idx);

                    // Compute the factor n*alpha-j*(alpha+1).
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
                    auto fac = builder.CreateFSub(
                        builder.CreateFMul(ord_v, alpha_v),
                        builder.CreateFMul(
                            j_v,
                            builder.CreateFAdd(alpha_v, vector_splat(builder, codegen<T>(s, number{1.}), batch_size))));

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(b_nj, aj))),
                                        acc);
                });

                // Finalize the result: acc / (n*b0).
                builder.CreateStore(
                    builder.CreateFDiv(builder.CreateLoad(acc),
                                       builder.CreateFMul(ord_v, taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                                                    builder.getInt32(0), var_idx))),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
template <typename T, typename U>
llvm::Function *taylor_c_diff_func_sqrt_impl(llvm_state &, const function &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a square root in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_sqrt(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the square root in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit([&](const auto &v) { return taylor_c_diff_func_sqrt_impl<T>(s, func, v, n_uvars, batch_size); },
                      func.args()[0].value());
}

} // namespace

} // namespace detail

expression sqrt(expression e)
{
    std::vector<expression> args;
    args.push_back(std::move(e));

    function fc{std::move(args)};
    fc.display_name() = "sqrt";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the double codegen of the square root "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.sqrt", {args[0]->getType()}, args);
    };
    fc.codegen_ldbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Invalid number of arguments passed to the long double codegen of the square root "
                "function: 1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were passed instead");
        }

        return detail::llvm_invoke_intrinsic(s, "llvm.sqrt", {args[0]->getType()}, args);
    };
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the float128 codegen of the square root "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        // Decompose the argument into scalars.
        auto scalars = detail::vector_to_scalars(builder, args[0]);

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> retvals;
        for (auto scal : scalars) {
            retvals.push_back(detail::llvm_invoke_external(
                s, "heyoka_sqrt128", scal->getType(), {scal},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Build a vector with the results.
        return detail::scalars_to_vector(builder, retvals);
    };
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

    fc.taylor_diff_dbl_f() = detail::taylor_diff_sqrt<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sqrt<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sqrt<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_sqrt<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_sqrt<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_sqrt<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
