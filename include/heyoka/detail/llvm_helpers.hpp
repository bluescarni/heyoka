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
#include <vector>

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

HEYOKA_DLL_PUBLIC llvm::Value *load_vector_from_memory(llvm::IRBuilder<> &, llvm::Value *, std::uint32_t,
                                                       const std::string & = "");

HEYOKA_DLL_PUBLIC llvm::Value *store_vector_to_memory(llvm::IRBuilder<> &, llvm::Value *, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC std::vector<llvm::Value *> vector_to_scalars(llvm::IRBuilder<> &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *scalars_to_vector(llvm::IRBuilder<> &, const std::vector<llvm::Value *> &);

// Helper to return the (null) Taylor derivative of a constant,
// as a scalar or as a vector.
template <typename T>
inline llvm::Value *taylor_diff_batch_zero(llvm_state &s, std::uint32_t vector_size)
{
    auto ret = codegen<T>(s, number(0.));

    if (vector_size > 0u) {
        ret = create_constant_vector(s.builder(), ret, vector_size);
    }

    return ret;
}

} // namespace heyoka::detail

#endif
