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
#include <vector>

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/llvm_helpers.hpp>

#if defined(__clang__)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-out-of-range-compare"

#endif

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

std::vector<llvm::Value *> vector_to_scalars(llvm::IRBuilder<> &builder, llvm::Value *vec)
{
    // Fetch the vector type.
    auto vec_t = llvm::cast<llvm::VectorType>(vec->getType());

    // Fetch the vector width.
    auto vector_size = vec_t->getNumElements();

    // Extract the vector elements one by one.
    std::vector<llvm::Value *> ret;
    if (vector_size > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Overflow in vector_to_scalars()");
    }
    for (decltype(vector_size) i = 0; i < vector_size; ++i) {
        ret.push_back(builder.CreateExtractElement(vec, static_cast<std::uint64_t>(i)));
        assert(ret.back() != nullptr);
    }

    return ret;
}

llvm::Value *scalars_to_vector(llvm::IRBuilder<> &builder, const std::vector<llvm::Value *> &scalars)
{
    assert(!scalars.empty());

    // Fetch the scalar type.
    auto scalar_t = scalars[0]->getType();

    // Fetch the vector size.
    const auto vector_size = scalars.size();

    // Create the corresponding vector type.
    if (vector_size > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("Overflow in scalars_to_vector()");
    }
    auto vector_t = llvm::VectorType::get(scalar_t, static_cast<unsigned>(vector_size));
    assert(vector_t != nullptr);

    // Create an empty vector.
    llvm::Value *vec = llvm::UndefValue::get(vector_t);
    assert(vec != nullptr);

    // Fill it up.
    for (auto i = 0u; i < vector_size; ++i) {
        assert(scalars[i]->getType() == scalar_t);

        vec = builder.CreateInsertElement(vec, scalars[i], i);
    }

    return vec;
}

// Pairwise summation of a vector of LLVM values.
// https://en.wikipedia.org/wiki/Pairwise_summation
llvm::Value *llvm_pairwise_sum(llvm::IRBuilder<> &builder, std::vector<llvm::Value *> &sum)
{
    assert(!sum.empty());

    while (sum.size() != 1u) {
        std::vector<llvm::Value *> new_sum;

        for (decltype(sum.size()) i = 0; i < sum.size(); i += 2u) {
            if (i + 1u == sum.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_sum.push_back(sum[i]);
            } else {
                new_sum.push_back(builder.CreateFAdd(sum[i], sum[i + 1u]));
            }
        }

        new_sum.swap(sum);
    }

    return sum[0];
}

// Helper to invoke an intrinsic function with arguments 'args'. 'types' are the argument type(s) for
// overloaded intrinsics.
llvm::Value *llvm_invoke_intrinsic(llvm_state &s, const std::string &name, const std::vector<llvm::Type *> &types,
                                   const std::vector<llvm::Value *> &args)
{
    // Fetch the intrinsic ID from the name.
    const auto intrinsic_ID = llvm::Function::lookupIntrinsicID(name);
    if (intrinsic_ID == 0) {
        throw std::invalid_argument("Cannot fetch the ID of the intrinsic '" + name + "'");
    }

    // Fetch the declaration.
    // NOTE: for generic intrinsics to work, we need to specify
    // the desired argument type(s). See:
    // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
    // And the docs of the getDeclaration() function.
    auto callee_f = llvm::Intrinsic::getDeclaration(&s.module(), intrinsic_ID, types);
    if (callee_f == nullptr) {
        throw std::invalid_argument("Error getting the declaration of the intrinsic '" + name + "'");
    }
    if (!callee_f->isDeclaration()) {
        // It does not make sense to have a definition of a builtin.
        throw std::invalid_argument("The intrinsic '" + name + "' must be only declared, not defined");
    }

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, args);
    assert(r != nullptr);

    return r;
}

#if defined(__clang__)

#pragma clang diagnostic pop

#endif

} // namespace heyoka::detail
