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
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/type_traits.hpp>
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
    } else {
        static_assert(always_false_v<T>, "Unhandled type in to_llvm_type().");
    }
}

// Common boilerplate for the implementation of
// functions computing Taylor derivatives.
template <typename T>
inline auto taylor_diff_common(llvm_state &s, const std::string &name)
{
    auto &builder = s.builder();

    // Check the function name.
    if (s.module().getFunction(name) != nullptr) {
        throw std::invalid_argument("Cannot add the function '" + name
                                    + "' when building a function for the computation of a Taylor derivative: the "
                                      "function already exists in the LLVM module");
    }

    // Prepare the function prototype. The arguments are:
    // - const float pointer to the derivatives array,
    // - 32-bit integer (order of the derivative).
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())), builder.getInt32Ty()};

    // The function will return the n-th derivative as a float.
    auto *ft = llvm::FunctionType::get(to_llvm_type<T>(s.context()), fargs, false);
    assert(ft != nullptr);

    // Now create the function. Don't need to call it from outside,
    // thus internal linkage.
    auto *f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, name, s.module());
    assert(f != nullptr);

    // Setup the function arugments.
    auto arg_it = f->args().begin();
    arg_it->setName("diff_ptr");
    arg_it->addAttr(llvm::Attribute::ReadOnly);
    arg_it->addAttr(llvm::Attribute::NoCapture);
    auto diff_ptr = arg_it;

    (++arg_it)->setName("order");
    auto order = arg_it;

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    return std::tuple{f, diff_ptr, order};
}

} // namespace heyoka::detail

#endif
