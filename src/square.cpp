// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/square.hpp>
#include <heyoka/taylor.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Derivative of square(number).
template <typename T>
llvm::Value *taylor_diff_square_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of square(variable).
template <typename T>
llvm::Value *taylor_diff_square_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                     std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t)
{
    // NOTE: we are currently not allowing order 0 derivatives
    // in non-compact mode.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of square() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // Compute the sum.
    std::vector<llvm::Value *> sum;
    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 0; j <= (order - 1u) / 2u; ++j) {
            auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto ret = pairwise_sum(builder, sum);
        return builder.CreateFAdd(ret, ret);
    } else {
        // Even order.
        auto ak2 = taylor_fetch_diff(arr, u_idx, order / 2u, n_uvars);
        auto sq_ak2 = builder.CreateFMul(ak2, ak2);

        for (std::uint32_t j = 0; j <= (order - 2u) / 2u; ++j) {
            auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto ret = pairwise_sum(builder, sum);
        return builder.CreateFAdd(builder.CreateFAdd(ret, ret), sq_ak2);
    }
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_square_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a square");
}

template <typename T>
llvm::Value *taylor_diff_square(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                                std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the square (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) { return taylor_diff_square_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        func.args()[0].value());
}

// Derivative of square(number).
template <typename T>
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &s, const function &func, const number &, std::uint32_t,
                                               std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num<T>(s, func, batch_size,
                                           "heyoka_taylor_diff_square_num_"
                                               + taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size)),
                                           "the square");
}

// Derivative of square(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &s, const function &func, const variable &,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_square_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

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

                // Distinguish the odd/even cases for the order.
                llvm_if_then_else(
                    s, builder.CreateICmpEQ(builder.CreateURem(ord, builder.getInt32(2)), builder.getInt32(1)),
                    [&]() {
                        // Odd order.
                        auto loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(1)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                            auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(a_nj, aj)), acc);
                        });

                        // Return 2 * acc.
                        auto acc_load = builder.CreateLoad(acc);
                        builder.CreateStore(builder.CreateFAdd(acc_load, acc_load), retval);
                    },
                    [&]() {
                        // Even order.
                        auto loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(2)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                            auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(a_nj, aj)), acc);
                        });

                        // Return 2 * acc + ak2 * ak2.
                        auto acc_load = builder.CreateLoad(acc);
                        auto ak2 = taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                      builder.CreateUDiv(ord, builder.getInt32(2)), var_idx);
                        builder.CreateStore(
                            builder.CreateFAdd(builder.CreateFAdd(acc_load, acc_load), builder.CreateFMul(ak2, ak2)),
                            retval);
                    });
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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of the square "
                                        "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U>
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &, const function &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a square in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_square(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the square in compact mode (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit([&](const auto &v) { return taylor_c_diff_func_square_impl<T>(s, func, v, n_uvars, batch_size); },
                      func.args()[0].value());
}

} // namespace

} // namespace detail

expression square(expression e)
{
    std::vector<expression> args;
    args.push_back(std::move(e));

    function fc{std::move(args)};
    fc.display_name() = "square";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Invalid number of arguments passed to the codegen of the square "
                                        "function: 1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        return s.builder().CreateFMul(args[0], args[0]);
    };
    fc.codegen_ldbl_f() = fc.codegen_dbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = fc.codegen_dbl_f();
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_square<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_square<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_square<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_square<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_square<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_square<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
