// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TAYLOR_COMMON_HPP
#define HEYOKA_DETAIL_TAYLOR_COMMON_HPP

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Implementation detail for taylor_c_diff_func_numpar(). np_tup is a tuple containing a sequence of numbers and/or
// params, while numpar_begin points to the first number/param argument in the signature of a function for the
// computation of Taylor derivatives in compact mode. This function will generate a set of llvm values from the function
// arguments starting at numpar_begin, either splatting the number argument or loading the parameter from the parameter
// array (see the implementation of taylor_c_diff_numparam_codegen()).
template <typename Tup, typename ArgIter, std::size_t... I>
inline std::vector<llvm::Value *>
taylor_c_diff_func_numpar_codegen_impl(llvm_state &s, llvm::Type *fp_t, const Tup &np_tup, ArgIter numpar_begin,
                                       llvm::Value *par_ptr, std::uint32_t batch_size, std::index_sequence<I...>)
{
    return {taylor_c_diff_numparam_codegen(s, fp_t, std::get<I>(np_tup), numpar_begin + I, par_ptr, batch_size)...};
}

// Helper to implement the function for the differentiation of
// a function of number(s)/param(s) in compact mode. The function will always return zero,
// unless the order is 0 (in which case it will return the result of applying the functor cgen
// to the number(s)/param(s) arguments).
template <typename F, typename... NumPars>
inline llvm::Function *taylor_c_diff_func_numpar(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                 std::uint32_t batch_size, const std::string &name,
                                                 std::uint32_t n_hidden_deps, const F &cgen, const NumPars &...np)
{
    static_assert(sizeof...(np) > 0u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, name, n_uvars, batch_size, {np...}, n_hidden_deps);
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
        auto ord = f->args().begin();
        auto par_ptr = f->args().begin() + 3;
        auto numpar_begin = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // Generate the vector of num/param arguments.
                auto np_args = taylor_c_diff_func_numpar_codegen_impl(s, fp_t, std::make_tuple(std::cref(np)...),
                                                                      numpar_begin, par_ptr, batch_size,
                                                                      std::make_index_sequence<sizeof...(np)>{});

                // Run the codegen and store the result.
                builder.CreateStore(cgen(np_args), retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(llvm_constantfp(s, val_t, 0.), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

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
            throw std::invalid_argument(fmt::format(
                "Inconsistent function signature for the Taylor derivative of {}() in compact mode detected", name));
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
