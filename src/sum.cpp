// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
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
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/sum.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_diff_sum(llvm_state &s, const function &func, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of sum() (the order must be at least one)");
    }

    // Filter-out the u variables in func's arguments and load the corresponding
    // derivatives at the desired order.
    std::vector<llvm::Value *> u_der;
    for (const auto &arg : func.args()) {
        if (auto pval = std::get_if<variable>(&arg.value())) {
            u_der.push_back(taylor_fetch_diff(arr, uname_to_index(pval->name()), order, n_uvars));
        } else if (!std::holds_alternative<number>(arg.value())) {
            throw std::invalid_argument(
                "An invalid argument type was encountered while trying to build the Taylor derivative "
                "of a summation");
        }
    }

    if (u_der.empty()) {
        // No u variables appear in the list of arguments, return zero.
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    } else {
        return pairwise_sum(s.builder(), u_der);
    }
}

template <typename T>
llvm::Function *taylor_c_diff_func_sum(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Init the function name.
    std::string fname{"heyoka_taylor_diff_sum_"};

    // Init the function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t)};

    // Vector to keep track of whether the sum arguments are uvars (0) or numbers (1).
    std::vector<int> arg_types;

    // Complete the function names/arguments and fill in arg_types.
    for (const auto &arg : func.args()) {
        std::visit(
            [&](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    fname += "var_";
                    fargs.push_back(llvm::Type::getInt32Ty(context));
                    arg_types.push_back(0);
                } else if constexpr (std::is_same_v<type, number>) {
                    fname += "num_";
                    fargs.push_back(to_llvm_type<T>(context));
                    arg_types.push_back(1);
                } else {
                    throw std::invalid_argument("Invalid argument type detected when building the Taylor derivative "
                                                "function in compact mode for the summation function");
                }
            },
            arg.value());
    }

    fname += taylor_mangle_suffix(val_t) + "_n_uvars_" + li_to_string(n_uvars);

    assert(arg_types.size() == func.args().size());

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

        // Fetch the necessary arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Distinguish order == 0 and order != 0.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(0)),
            [&]() {
                // For order zero, we load the order 0 derivatives for the uvars
                // and run the codegen for the numbers.
                std::vector<llvm::Value *> vals;

                for (decltype(arg_types.size()) i = 0; i < arg_types.size(); ++i) {
                    if (arg_types[i] == 0) {
                        vals.push_back(
                            taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), f->args().begin() + 3 + i));
                    } else {
                        assert(arg_types[i] == 1);
                        vals.push_back(vector_splat(builder, f->args().begin() + 3 + i, batch_size));
                    }
                }

                if (vals.empty()) {
                    // NOTE: this is a corner case for a sum without arguments.
                    // Not sure how likely it is to end up here.
                    builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
                } else {
                    builder.CreateStore(pairwise_sum(builder, vals), retval);
                }
            },
            [&]() {
                // For order nonzero, we load the derivatives for the uvars
                // and ignore the numbers.
                std::vector<llvm::Value *> vals;

                for (decltype(arg_types.size()) i = 0; i < arg_types.size(); ++i) {
                    if (arg_types[i] == 0) {
                        vals.push_back(taylor_c_load_diff(s, diff_arr, n_uvars, order, f->args().begin() + 3 + i));
                    } else {
                        assert(arg_types[i] == 1);
                    }
                }

                if (vals.empty()) {
                    // NOTE: there are no uvar arguments, return zero.
                    builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
                } else {
                    builder.CreateStore(pairwise_sum(builder, vals), retval);
                }
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
                "Inconsistent function signature for the Taylor derivative of summation in compact mode detected");
        }
    }

    return f;
}

} // namespace

} // namespace detail

expression sum(std::vector<expression> terms)
{
    function fc{std::move(terms)};
    fc.display_name() = "sum";

    auto codegen_impl = [](llvm_state &s, const std::vector<llvm::Value *> &args) {
        if (args.empty()) {
            throw std::invalid_argument("Cannot codegen a summation without terms");
        }

        auto args_copy(args);
        return detail::pairwise_sum(s.builder(), args_copy);
    };

    fc.codegen_dbl_f() = codegen_impl;
    fc.codegen_ldbl_f() = codegen_impl;
#if defined(HEYOKA_HAVE_REAL128)
    fc.codegen_f128_f() = codegen_impl;
#endif

    fc.taylor_diff_dbl_f() = detail::taylor_diff_sum<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sum<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sum<mppp::real128>;
#endif

    fc.taylor_c_diff_func_dbl_f() = detail::taylor_c_diff_func_sum<double>;
    fc.taylor_c_diff_func_ldbl_f() = detail::taylor_c_diff_func_sum<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_c_diff_func_f128_f() = detail::taylor_c_diff_func_sum<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
