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
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <fmt/format.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pi.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

namespace heyoka
{

namespace detail
{

pi_impl::pi_impl() : func_base("pi", std::vector<expression>{}) {}

namespace
{

// Implementation of codegen for pi.
template <typename T>
auto pi_impl_codegen(llvm_state &s, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>) {
        return vector_splat(s.builder(), codegen<T>(s, number{boost::math::constants::pi<T>()}), batch_size);
    }
#if defined(HEYOKA_HAVE_REAL128)
    else if constexpr (std::is_same_v<T, mppp::real128>) {
        return vector_splat(s.builder(), codegen<mppp::real128>(s, number{mppp::pi_128}), batch_size);
    }
#endif
}

} // namespace

llvm::Value *pi_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                      const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                      std::uint32_t order, std::uint32_t, std::uint32_t batch_size) const
{
    if (order == 0u) {
        return pi_impl_codegen<double>(s, batch_size);
    } else {
        return vector_splat(s.builder(), codegen<double>(s, number{0.}), batch_size);
    }
}

llvm::Value *pi_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                       const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                       std::uint32_t order, std::uint32_t, std::uint32_t batch_size) const
{
    if (order == 0u) {
        return pi_impl_codegen<long double>(s, batch_size);
    } else {
        return vector_splat(s.builder(), codegen<long double>(s, number{0.}), batch_size);
    }
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *pi_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &,
                                       const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                       std::uint32_t order, std::uint32_t, std::uint32_t batch_size) const
{
    if (order == 0u) {
        return pi_impl_codegen<mppp::real128>(s, batch_size);
    } else {
        return vector_splat(s.builder(), codegen<mppp::real128>(s, number{0.}), batch_size);
    }
}

#endif

namespace
{

template <typename T>
auto taylor_c_diff_func_pi(llvm_state &s, std::uint32_t batch_size)
{
    using namespace fmt::literals;

    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Compose the function name.
    const auto fname
        = "heyoka_taylor_diff_pi_{}"_format(taylor_mangle_suffix(to_llvm_vector_type<T>(context, batch_size)));

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - par ptr.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t),
                                    llvm::PointerType::getUnqual(to_llvm_type<T>(context))};

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

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, run the codegen.
                builder.CreateStore(pi_impl_codegen<T>(s, batch_size), retval);
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
                "Inconsistent function signature for the Taylor derivative of pi() in compact mode detected");
        }
    }

    return f;
}

} // namespace

llvm::Function *pi_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t, std::uint32_t batch_size) const
{
    return taylor_c_diff_func_pi<double>(s, batch_size);
}

llvm::Function *pi_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t, std::uint32_t batch_size) const
{
    return taylor_c_diff_func_pi<long double>(s, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *pi_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t, std::uint32_t batch_size) const
{
    return taylor_c_diff_func_pi<mppp::real128>(s, batch_size);
}

#endif

} // namespace detail

expression pi()
{
    return expression{func{detail::pi_impl()}};
}

} // namespace heyoka
