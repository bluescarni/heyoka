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
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
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

// Helper to associate a C++ type to an LLVM type.
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

HEYOKA_DLL_PUBLIC llvm::Value *load_vector_from_memory(llvm::IRBuilder<> &, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC void store_vector_to_memory(llvm::IRBuilder<> &, llvm::Value *, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *vector_splat(llvm::IRBuilder<> &, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC std::vector<llvm::Value *> vector_to_scalars(llvm::IRBuilder<> &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *scalars_to_vector(llvm::IRBuilder<> &, const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::Value *pairwise_sum(llvm::IRBuilder<> &, std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_invoke_intrinsic(llvm_state &, const std::string &,
                                                     const std::vector<llvm::Type *> &,
                                                     const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_invoke_external(llvm_state &, const std::string &, llvm::Type *,
                                                    const std::vector<llvm::Value *> &,
                                                    const std::vector<llvm::Attribute::AttrKind> & = {});

HEYOKA_DLL_PUBLIC llvm::Value *llvm_invoke_internal(llvm_state &, const std::string &,
                                                    const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC void llvm_loop_u32(llvm_state &, llvm::Value *, llvm::Value *,
                                     const std::function<void(llvm::Value *)> &);

HEYOKA_DLL_PUBLIC llvm::Type *pointee_type(llvm::Value *);

HEYOKA_DLL_PUBLIC std::string llvm_type_name(llvm::Type *);

HEYOKA_DLL_PUBLIC bool compare_function_signature(llvm::Function *, llvm::Type *, const std::vector<llvm::Type *> &);

} // namespace heyoka::detail

#endif
