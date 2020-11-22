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

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/sos_3d.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Derivative of sos_3d(var, var, var).
template <typename T>
llvm::Value *taylor_diff_sos_3d_impl(llvm_state &s, const variable &var0, const variable &var1, const variable &var2,
                                     const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars, std::uint32_t order,
                                     std::uint32_t, std::uint32_t)
{
    // NOTE: we are currently not allowing order 0 derivatives
    // in non-compact mode.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of sos_3d() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the indices of the variables.
    const auto u_idx_0 = uname_to_index(var0.name());
    const auto u_idx_1 = uname_to_index(var1.name());
    const auto u_idx_2 = uname_to_index(var2.name());

    // Compute the sums.
    std::vector<llvm::Value *> sum0;
    std::vector<llvm::Value *> sum1;
    std::vector<llvm::Value *> sum2;

    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 0; j <= (order - 1u) / 2u; ++j) {
            auto v0 = taylor_fetch_diff(arr, u_idx_0, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx_0, j, n_uvars);
            sum0.push_back(builder.CreateFMul(v0, v1));

            v0 = taylor_fetch_diff(arr, u_idx_1, order - j, n_uvars);
            v1 = taylor_fetch_diff(arr, u_idx_1, j, n_uvars);
            sum1.push_back(builder.CreateFMul(v0, v1));

            v0 = taylor_fetch_diff(arr, u_idx_2, order - j, n_uvars);
            v1 = taylor_fetch_diff(arr, u_idx_2, j, n_uvars);
            sum2.push_back(builder.CreateFMul(v0, v1));
        }

        auto ret0 = pairwise_sum(builder, sum0);
        ret0 = builder.CreateFAdd(ret0, ret0);

        auto ret1 = pairwise_sum(builder, sum1);
        ret1 = builder.CreateFAdd(ret1, ret1);

        auto ret2 = pairwise_sum(builder, sum2);
        ret2 = builder.CreateFAdd(ret2, ret2);

        return builder.CreateFAdd(builder.CreateFAdd(ret0, ret1), ret2);
    } else {
        // Even order.
        auto ak0 = taylor_fetch_diff(arr, u_idx_0, order / 2u, n_uvars);
        auto sq_ak0 = builder.CreateFMul(ak0, ak0);

        auto ak1 = taylor_fetch_diff(arr, u_idx_1, order / 2u, n_uvars);
        auto sq_ak1 = builder.CreateFMul(ak1, ak1);

        auto ak2 = taylor_fetch_diff(arr, u_idx_2, order / 2u, n_uvars);
        auto sq_ak2 = builder.CreateFMul(ak2, ak2);

        for (std::uint32_t j = 0; j <= (order - 2u) / 2u; ++j) {
            auto v0 = taylor_fetch_diff(arr, u_idx_0, order - j, n_uvars);
            auto v1 = taylor_fetch_diff(arr, u_idx_0, j, n_uvars);
            sum0.push_back(builder.CreateFMul(v0, v1));

            v0 = taylor_fetch_diff(arr, u_idx_1, order - j, n_uvars);
            v1 = taylor_fetch_diff(arr, u_idx_1, j, n_uvars);
            sum1.push_back(builder.CreateFMul(v0, v1));

            v0 = taylor_fetch_diff(arr, u_idx_2, order - j, n_uvars);
            v1 = taylor_fetch_diff(arr, u_idx_2, j, n_uvars);
            sum2.push_back(builder.CreateFMul(v0, v1));
        }

        auto ret0 = pairwise_sum(builder, sum0);
        ret0 = builder.CreateFAdd(builder.CreateFAdd(ret0, ret0), sq_ak0);

        auto ret1 = pairwise_sum(builder, sum1);
        ret1 = builder.CreateFAdd(builder.CreateFAdd(ret1, ret1), sq_ak1);

        auto ret2 = pairwise_sum(builder, sum2);
        ret2 = builder.CreateFAdd(builder.CreateFAdd(ret2, ret2), sq_ak2);

        return builder.CreateFAdd(builder.CreateFAdd(ret0, ret1), ret2);
    }
}

// All the other cases.
template <typename T, typename U0, typename U1, typename U2>
llvm::Value *taylor_diff_sos_3d_impl(llvm_state &, const U0 &, const U1 &, const U2 &,
                                     const std::vector<llvm::Value *> &, std::uint32_t, std::uint32_t, std::uint32_t,
                                     std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a sos_3d");
}

template <typename T>
llvm::Value *taylor_diff_sos_3d(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                                std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if (func.args().size() != 3u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sos_3d (3 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v0, const auto &v1, const auto &v2) {
            return taylor_diff_sos_3d_impl<T>(s, v0, v1, v2, arr, n_uvars, order, idx, batch_size);
        },
        func.args()[0].value(), func.args()[1].value(), func.args()[2].value());
}

// Derivative of sos_3d(var, var, var).
template <typename T>
llvm::Function *taylor_c_diff_func_sos_3d_impl(llvm_state &s, const function &func, const variable &, const variable &,
                                               const variable &, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname
        = "heyoka_taylor_diff_sos_3d_var_var_var_" + taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - indices of the var arguments.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context),     llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::Type::getInt32Ty(context),
                                    llvm::Type::getInt32Ty(context),     llvm::Type::getInt32Ty(context)};

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
        auto var_idx_0 = f->args().begin() + 3;
        auto var_idx_1 = f->args().begin() + 4;
        auto var_idx_2 = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulators.
        auto acc0 = builder.CreateAlloca(val_t);
        auto acc1 = builder.CreateAlloca(val_t);
        auto acc2 = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0.
                builder.CreateStore(
                    codegen_from_values<T>(s, func,
                                           {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx_0),
                                            taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx_1),
                                            taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx_2)}),
                    retval);
            },
            [&]() {
                // Init the accumulators.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc0);
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc1);
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc2);

                // Distinguish the odd/even cases for the order.
                llvm_if_then_else(
                    s, builder.CreateICmpEQ(builder.CreateURem(ord, builder.getInt32(2)), builder.getInt32(1)),
                    [&]() {
                        // Odd order.
                        auto loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(1)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx_0);
                            auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx_0);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc0), builder.CreateFMul(a_nj, aj)), acc0);

                            a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx_1);
                            aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx_1);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc1), builder.CreateFMul(a_nj, aj)), acc1);

                            a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx_2);
                            aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx_2);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc2), builder.CreateFMul(a_nj, aj)), acc2);
                        });

                        llvm::Value *ret0 = builder.CreateLoad(acc0);
                        ret0 = builder.CreateFAdd(ret0, ret0);

                        llvm::Value *ret1 = builder.CreateLoad(acc1);
                        ret1 = builder.CreateFAdd(ret1, ret1);

                        llvm::Value *ret2 = builder.CreateLoad(acc2);
                        ret2 = builder.CreateFAdd(ret2, ret2);

                        builder.CreateStore(builder.CreateFAdd(builder.CreateFAdd(ret0, ret1), ret2), retval);
                    },
                    [&]() {
                        // Even order.

                        // Pre-compute the final terms.
                        auto ak0 = taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                      builder.CreateUDiv(ord, builder.getInt32(2)), var_idx_0);
                        auto sq_ak0 = builder.CreateFMul(ak0, ak0);

                        auto ak1 = taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                      builder.CreateUDiv(ord, builder.getInt32(2)), var_idx_1);
                        auto sq_ak1 = builder.CreateFMul(ak1, ak1);

                        auto ak2 = taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                      builder.CreateUDiv(ord, builder.getInt32(2)), var_idx_2);
                        auto sq_ak2 = builder.CreateFMul(ak2, ak2);

                        auto loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(2)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx_0);
                            auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx_0);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc0), builder.CreateFMul(a_nj, aj)), acc0);

                            a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx_1);
                            aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx_1);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc1), builder.CreateFMul(a_nj, aj)), acc1);

                            a_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx_2);
                            aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, var_idx_2);

                            builder.CreateStore(
                                builder.CreateFAdd(builder.CreateLoad(acc2), builder.CreateFMul(a_nj, aj)), acc2);
                        });

                        llvm::Value *ret0 = builder.CreateLoad(acc0);
                        ret0 = builder.CreateFAdd(builder.CreateFAdd(ret0, ret0), sq_ak0);

                        llvm::Value *ret1 = builder.CreateLoad(acc1);
                        ret1 = builder.CreateFAdd(builder.CreateFAdd(ret1, ret1), sq_ak1);

                        llvm::Value *ret2 = builder.CreateLoad(acc2);
                        ret2 = builder.CreateFAdd(builder.CreateFAdd(ret2, ret2), sq_ak2);

                        builder.CreateStore(builder.CreateFAdd(builder.CreateFAdd(ret0, ret1), ret2), retval);
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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of sos_3d "
                                        "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U0, typename U1, typename U2>
llvm::Function *taylor_c_diff_func_sos_3d_impl(llvm_state &, const function &, const U0 &, const U1 &, const U2 &,
                                               std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a sos_3d in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_sos_3d(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    if (func.args().size() != 3u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sos_3d in compact mode (3 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v0, const auto &v1, const auto &v2) {
            return taylor_c_diff_func_sos_3d_impl<T>(s, func, v0, v1, v2, n_uvars, batch_size);
        },
        func.args()[0].value(), func.args()[1].value(), func.args()[2].value());
}

} // namespace

} // namespace detail

expression sos_3d(expression e1, expression e2, expression e3)
{
    std::vector<expression> args;
    args.push_back(std::move(e1));
    args.push_back(std::move(e2));
    args.push_back(std::move(e3));

    function fc{std::move(args)};
    fc.display_name() = "sos_3d";

    fc.codegen_dbl_f() = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.size() != 3u) {
            throw std::invalid_argument("Invalid number of arguments passed to the codegen of the sos_3d "
                                        "function: 3 arguments were expected, but "
                                        + std::to_string(args.size()) + " arguments were passed instead");
        }

        auto &builder = s.builder();

        return builder.CreateFAdd(
            builder.CreateFAdd(builder.CreateFMul(args[0], args[0]), builder.CreateFMul(args[1], args[1])),
            builder.CreateFMul(args[2], args[2]));
    };
    fc.codegen_ldbl_f() = fc.codegen_dbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = fc.codegen_dbl_f();
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_sos_3d<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sos_3d<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sos_3d<mppp::real128>;
#endif
    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_sos_3d<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_sos_3d<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_sos_3d<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
