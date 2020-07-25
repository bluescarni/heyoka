// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/llvm_helpers.hpp>

namespace heyoka::detail
{

// Turn a pointer to float into a pointer to a SIMD vector of float
// of the corresponding type.
// NOTE: the implementation is based on bitcasting
// the original pointer to the appropriate vector type.
// This approach is based on what LLVM does when optimising
// IR for x86. It's not clear if this strategy is 100%
// portable across architectures.
llvm::Value *to_vector_pointer(llvm::IRBuilder<> &builder, llvm::Value *v, std::uint32_t vector_size)
{
    assert(vector_size > 0u);

    // Fetch the pointer type (this will result in an assertion
    // failure if v is not a pointer).
    auto ptr_t = llvm::cast<llvm::PointerType>(v->getType());
    assert(ptr_t != nullptr);

    // Fetch the pointee type.
    auto scalar_t = ptr_t->getPointerElementType();
    assert(scalar_t != nullptr);

    // Create the corresponding vector type.
    auto vector_t = llvm::VectorType::get(scalar_t, vector_size);
    assert(vector_t != nullptr);

    // Do the bitcast.
    return builder.CreateBitCast(v, llvm::PointerType::getUnqual(vector_t));
}

// Create a SIMD vector of size vector_size filled with the constant c.
llvm::Value *create_constant_vector(llvm::IRBuilder<> &builder, llvm::Value *c, std::uint32_t vector_size)
{
    llvm::Value *vec = llvm::UndefValue::get(llvm::VectorType::get(c->getType(), vector_size));

    // Fill up the vector with insertelement.
    for (std::uint32_t i = 0; i < vector_size; ++i) {
        // NOTE: the insertelement instruction returns
        // a new vector with the element at index i changed.
        vec = builder.CreateInsertElement(vec, c, i);
    }

    return vec;
}

} // namespace heyoka::detail
