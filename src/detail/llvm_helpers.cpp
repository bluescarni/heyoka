// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/llvm_helpers.hpp>

namespace heyoka::detail
{

namespace
{

// Small helper to fetch the data layout
// of the current machine. This is used
// to deduce alignment values below.
const auto &get_native_dl()
{
    thread_local const auto ret = *llvm::orc::JITTargetMachineBuilder::detectHost()->getDefaultDataLayoutForTarget();

    return ret;
}

// Helper to turn a pointer to float into a pointer to a SIMD vector of float
// of the corresponding type via direct bitcasting.
auto to_vector_pointer(llvm::IRBuilder<> &builder, llvm::Value *ptr, std::uint32_t vector_size)
{
    assert(vector_size > 0u);

    // Fetch the pointer type (this will result in an assertion
    // failure if v is not a pointer).
    auto ptr_t = llvm::cast<llvm::PointerType>(ptr->getType());
    assert(ptr_t != nullptr);

    // Fetch the pointee type.
    auto scalar_t = ptr_t->getPointerElementType();
    assert(scalar_t != nullptr);

    // Create the corresponding vector type.
    if (vector_size > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("Overflow in to_vector_pointer()");
    }
    auto vector_t = llvm::VectorType::get(scalar_t, static_cast<unsigned>(vector_size));
    assert(vector_t != nullptr);

    // Establish the minimum alignment value
    // for the scalar type.
    auto align_val = get_native_dl().getABITypeAlignment(scalar_t);

    // Return the converted pointer and the scalar alignment value.
    return std::pair{builder.CreateBitCast(ptr, llvm::PointerType::getUnqual(vector_t)), align_val};
}

} // namespace

// NOTE: these helpers to load/store SIMD vectors from/to memory are
// based on the x86 intrinsics pattern: convert a pointer to float
// into a pointer to a SIMD vector of float, and then execute
// the load/store with the alignment value from the scalar type.
// It remains to be seen if this pattern is applicable to other ISAs.
// An alternative is to load the scalar values one by one and then
// use them to fill up the SIMD vector with insertelement. This is probably
// more portable, but it result in a serious degradation of the JIT
// compilation performance.
llvm::Value *load_vector_from_memory(llvm::IRBuilder<> &builder, llvm::Value *ptr, std::uint32_t vector_size,
                                     const std::string &name)
{
    // Convert the input scalar pointer to a vector pointer.
    const auto [vec_ptr, align_val] = to_vector_pointer(builder, ptr, vector_size);

    // Do the load, using the minimum alignment of the scalar type.
    auto ret = builder.CreateLoad(vec_ptr, name);
    if (align_val > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Overflow in load_vector_from_memory()");
    }
    ret->setAlignment(llvm::Align(static_cast<std::uint64_t>(align_val)));

    return ret;
}

llvm::Value *store_vector_to_memory(llvm::IRBuilder<> &builder, llvm::Value *ptr, llvm::Value *vec,
                                    std::uint32_t vector_size)
{
    // Convert the input scalar pointer to a vector pointer.
    const auto [vec_ptr, align_val] = to_vector_pointer(builder, ptr, vector_size);

    // Do the store, using the minimum alignment of the scalar type.
    auto ret = builder.CreateStore(vec, vec_ptr);
    if (align_val > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Overflow in store_vector_from_memory()");
    }
    ret->setAlignment(llvm::Align(static_cast<std::uint64_t>(align_val)));

    return ret;
}

// Create a SIMD vector of size vector_size filled with the constant c.
llvm::Value *create_constant_vector(llvm::IRBuilder<> &builder, llvm::Value *c, std::uint32_t vector_size)
{
    assert(vector_size > 0u);

    if (vector_size > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("Overflow in create_constant_vector()");
    }
    llvm::Value *vec = llvm::UndefValue::get(llvm::VectorType::get(c->getType(), static_cast<unsigned>(vector_size)));

    // Fill up the vector with insertelement.
    for (std::uint32_t i = 0; i < vector_size; ++i) {
        // NOTE: the insertelement instruction returns
        // a new vector with the element at index i changed.
        vec = builder.CreateInsertElement(vec, c, i);
    }

    return vec;
}

} // namespace heyoka::detail
