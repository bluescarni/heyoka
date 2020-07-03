// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_HELPERS_HPP
#define HEYOKA_DETAIL_LLVM_HELPERS_HPP

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/llvm_state.hpp>

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
    } else {
        static_assert(always_false_v<T>, "Unhandled type in to_llvm_type().");
    }
}

template <typename T, typename U>
inline llvm::Value *invoke_codegen(llvm_state &s, const U &x)
{
    if constexpr (std::is_same_v<T, double>) {
        return codegen_dbl(s, x);
    } else if constexpr (std::is_same_v<T, long double>) {
        return codegen_ldbl(s, x);
    } else {
        static_assert(always_false_v<T>, "Unhandled type in invoke_codegen().");
    }
}

template <typename T, typename U>
inline llvm::Value *invoke_taylor_init(llvm_state &s, const U &x, llvm::Value *arr)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_init_dbl(s, x, arr);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_init_ldbl(s, x, arr);
    } else {
        static_assert(always_false_v<T>, "Unhandled type in invoke_taylor_init().");
    }
}

} // namespace heyoka::detail

#endif
