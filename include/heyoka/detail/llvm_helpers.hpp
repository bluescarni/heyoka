// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_HELPERS_HPP
#define HEYOKA_DETAIL_LLVM_HELPERS_HPP

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/tfp.hpp>

namespace heyoka::detail
{

template <typename T>
inline llvm::Type *to_llvm_type(llvm::LLVMContext &c)
{
    if constexpr (std::is_same_v<T, double>) {
        if constexpr (std::numeric_limits<T>::is_iec559 && std::numeric_limits<T>::digits == 53) {
            // IEEE double-precision type.
            auto ret = llvm::Type::getDoubleTy(c);
            assert(ret != nullptr);
            return ret;
        } else {
            static_assert(always_false_v<T>, "Cannot deduce the LLVM type corresponding to 'double' on this platform.");
        }
    } else if constexpr (std::is_same_v<T, long double>) {
        if constexpr (std::numeric_limits<T>::is_iec559 && std::numeric_limits<T>::digits == 53) {
            // IEEE double-precision type (this is the case on MSVC for instance).
            auto ret = llvm::Type::getDoubleTy(c);
            assert(ret != nullptr);
            return ret;
        } else if constexpr (std::numeric_limits<T>::is_iec559 && std::numeric_limits<T>::digits == 64) {
            // x86 extended precision format.
            auto ret = llvm::Type::getX86_FP80Ty(c);
            assert(ret != nullptr);
            return ret;
        } else {
            static_assert(always_false_v<T>,
                          "Cannot deduce the LLVM type corresponding to 'long double' on this platform.");
        }
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        auto ret = llvm::Type::getFP128Ty(c);
        assert(ret != nullptr);
        return ret;
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type in to_llvm_type().");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *create_constant_vector(llvm::IRBuilder<> &, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *load_vector_from_memory(llvm::IRBuilder<> &, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC void store_vector_to_memory(llvm::IRBuilder<> &, llvm::Value *, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC std::vector<llvm::Value *> vector_to_scalars(llvm::IRBuilder<> &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *scalars_to_vector(llvm::IRBuilder<> &, const std::vector<llvm::Value *> &);

// Helper to return the (null) Taylor derivative of a constant,
// as a scalar or as a vector.
template <typename T>
inline llvm::Value *taylor_diff_batch_zero(llvm_state &s, std::uint32_t vector_size)
{
    auto ret = codegen<T>(s, number{0.});

    if (vector_size > 0u) {
        ret = create_constant_vector(s.builder(), ret, vector_size);
    }

    return ret;
}

HEYOKA_DLL_PUBLIC llvm::Value *llvm_pairwise_sum(llvm::IRBuilder<> &, std::vector<llvm::Value *> &);

// Helper to load the value of the derivative of a u variable
// from an array in the computation of a Taylor jet:
// - u_idx is the index of the u variable,
// - order is the derivative order,
// - n_uvars the total number of u variables,
// - diff_arr is the array of derivatives,
// - batch_idx and batch_size the batch index/size,
// - vector_size the SIMD vector size (will be zero in scalar mode).
// NOTE: see a838c9b0d803b7ab13e83834ac46e3df0963b158 for a commit
// containing the cd_uvars machinery.
template <typename T>
inline llvm::Value *tjb_load_derivative(llvm_state &s, std::uint32_t u_idx, std::uint32_t order, std::uint32_t n_uvars,
                                        llvm::Value *diff_arr, std::uint32_t batch_idx, std::uint32_t batch_size,
                                        std::uint32_t vector_size)
{
    // Sanity checks.
    assert(u_idx < n_uvars);
    assert(batch_idx < batch_size);

    auto &builder = s.builder();

    auto arr_ptr = builder.CreateInBoundsGEP(
        diff_arr,
        {builder.getInt32(0), builder.getInt32(order * n_uvars * batch_size + u_idx * batch_size + batch_idx)},
        "diff_arr_ptr");

    return (vector_size == 0u) ? builder.CreateLoad(arr_ptr, "diff_arr_load")
                               : load_vector_from_memory(builder, arr_ptr, vector_size);
}

HEYOKA_DLL_PUBLIC llvm::Value *llvm_invoke_intrinsic(llvm_state &, const std::string &,
                                                     const std::vector<llvm::Type *> &,
                                                     const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_invoke_external(llvm_state &, const std::string &, llvm::Type *,
                                                    const std::vector<llvm::Value *> &,
                                                    const std::vector<llvm::Attribute::AttrKind> & = {});

// Helper to create a constant tfp.
template <typename T>
inline tfp tfp_constant(llvm_state &s, const number &num, std::uint32_t batch_size, bool high_accuracy)
{
    auto ret = create_constant_vector(s.builder(), codegen<T>(s, num), batch_size);

    return tfp_from_vector(s, ret, high_accuracy);
}

// Helper to create a zero tfp.
template <typename T>
inline tfp tfp_zero(llvm_state &s, std::uint32_t batch_size, bool high_accuracy)
{
    return tfp_constant<T>(s, number{0.}, batch_size, high_accuracy);
}

} // namespace heyoka::detail

#endif
