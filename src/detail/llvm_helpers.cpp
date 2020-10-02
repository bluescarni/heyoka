// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
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
#include <heyoka/llvm_state.hpp>

namespace heyoka::detail
{

// Helper to load the data from pointer ptr as a vector of size vector_size. If vector_size is
// 1, a scalar is loaded instead.
llvm::Value *load_vector_from_memory(llvm::IRBuilder<> &builder, llvm::Value *ptr, std::uint32_t vector_size)
{
    assert(vector_size > 0u);
    assert(llvm::isa<llvm::PointerType>(ptr->getType()));

    if (vector_size == 1u) {
        // Scalar case.
        return builder.CreateLoad(ptr);
    }

    // Fetch the pointer type (this will result in an assertion
    // failure if ptr is not a pointer).
    auto ptr_t = llvm::cast<llvm::PointerType>(ptr->getType());

    // Fetch the pointee type.
    auto scalar_t = ptr_t->getPointerElementType();
    assert(scalar_t != nullptr);

    // Create the corresponding vector type.
    auto vector_t = llvm::VectorType::get(scalar_t, boost::numeric_cast<unsigned>(vector_size));
    assert(vector_t != nullptr);

    // Create the output vector.
    auto ret = static_cast<llvm::Value *>(llvm::UndefValue::get(vector_t));

    // Fill it.
    for (std::uint32_t i = 0; i < vector_size; ++i) {
        ret = builder.CreateInsertElement(ret,
                                          builder.CreateLoad(builder.CreateInBoundsGEP(ptr, {builder.getInt32(i)})), i);
    }

    return ret;
}

// Helper to store the content of vector vec to the pointer ptr. If vec is not a vector,
// a plain store will be performed.
void store_vector_to_memory(llvm::IRBuilder<> &builder, llvm::Value *ptr, llvm::Value *vec)
{
    if (auto v_ptr_t = llvm::dyn_cast<llvm::VectorType>(vec->getType())) {
        // Determine the vector size.
        const auto vector_size = boost::numeric_cast<std::uint32_t>(v_ptr_t->getNumElements());

        for (std::uint32_t i = 0; i < vector_size; ++i) {
            builder.CreateStore(builder.CreateExtractElement(vec, i),
                                builder.CreateInBoundsGEP(ptr, {builder.getInt32(i)}));
        }
    } else {
        // Not a vector, store vec directly.
        builder.CreateStore(vec, ptr);
    }
}

// Create a SIMD vector of size vector_size filled with the value c. If vector_size is 1,
// c will be returned.
llvm::Value *vector_splat(llvm::IRBuilder<> &builder, llvm::Value *c, std::uint32_t vector_size)
{
    assert(vector_size > 0u);

    if (vector_size == 1u) {
        return c;
    }

    llvm::Value *vec
        = llvm::UndefValue::get(llvm::VectorType::get(c->getType(), boost::numeric_cast<unsigned>(vector_size)));

    // Fill up the vector with insertelement.
    for (std::uint32_t i = 0; i < vector_size; ++i) {
        // NOTE: the insertelement instruction returns
        // a new vector with the element at index i changed.
        vec = builder.CreateInsertElement(vec, c, i);
    }

    return vec;
}

// Convert the input LLVM vector to a std::vector of values. If vec is not a vector,
// return {vec}.
std::vector<llvm::Value *> vector_to_scalars(llvm::IRBuilder<> &builder, llvm::Value *vec)
{
    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(vec->getType())) {
        // Fetch the vector width.
        auto vector_size = vec_t->getNumElements();

        // Extract the vector elements one by one.
        std::vector<llvm::Value *> ret;
        for (decltype(vector_size) i = 0; i < vector_size; ++i) {
            ret.push_back(builder.CreateExtractElement(vec, boost::numeric_cast<std::uint64_t>(i)));
            assert(ret.back() != nullptr);
        }

        return ret;
    } else {
        return {vec};
    }
}

// Convert a std::vector of values into an LLVM vector of the corresponding size.
// If scalars contains only 1 value, return that value.
llvm::Value *scalars_to_vector(llvm::IRBuilder<> &builder, const std::vector<llvm::Value *> &scalars)
{
    assert(!scalars.empty());

    // Fetch the vector size.
    const auto vector_size = scalars.size();

    if (vector_size == 1u) {
        return scalars[0];
    }

    // Fetch the scalar type.
    auto scalar_t = scalars[0]->getType();

    // Create the corresponding vector type.
    auto vector_t = llvm::VectorType::get(scalar_t, boost::numeric_cast<unsigned>(vector_size));
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
llvm::Value *pairwise_sum(llvm::IRBuilder<> &builder, std::vector<llvm::Value *> &sum)
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

    // Check the number of arguments.
    if (callee_f->arg_size() != args.size()) {
        throw std::invalid_argument("Incorrect # of arguments passed while calling the intrinsic '" + name
                                    + "': " + std::to_string(callee_f->arg_size()) + " are expected, but "
                                    + std::to_string(args.size()) + " were provided instead");
    }

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, args);
    assert(r != nullptr);

    return r;
}

// Helper to invoke an external function called 'name' with arguments args and return type ret_type.
llvm::Value *llvm_invoke_external(llvm_state &s, const std::string &name, llvm::Type *ret_type,
                                  const std::vector<llvm::Value *> &args,
                                  const std::vector<llvm::Attribute::AttrKind> &attrs)
{
    // Look up the name in the global module table.
    auto callee_f = s.module().getFunction(name);

    if (callee_f == nullptr) {
        // The function does not exist yet, make the prototype.
        std::vector<llvm::Type *> arg_types;
        for (auto a : args) {
            arg_types.push_back(a->getType());
        }
        auto *ft = llvm::FunctionType::get(ret_type, arg_types, false);
        callee_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
        if (callee_f == nullptr) {
            throw std::invalid_argument("Unable to create the prototype for the external function '" + name + "'");
        }

        // Add the function attributes.
        for (const auto &att : attrs) {
            callee_f->addFnAttr(att);
        }
    } else {
        // The function declaration exists already. Check that it is only a
        // declaration and not a definition.
        if (!callee_f->isDeclaration()) {
            throw std::invalid_argument(
                "Cannot call the function '" + name
                + "' as an external function, because it is defined as an internal module function");
        }
        // Check the number of arguments.
        if (callee_f->arg_size() != args.size()) {
            throw std::invalid_argument("Incorrect # of arguments passed while calling the external function '" + name
                                        + "': " + std::to_string(callee_f->arg_size()) + " are expected, but "
                                        + std::to_string(args.size()) + " were provided instead");
        }
        // NOTE: perhaps in the future we should consider adding more checks here
        // (e.g., argument types, return type).
    }

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, args);
    assert(r != nullptr);
    // NOTE: we used to have r->setTailCall(true) here, but:
    // - when optimising, the tail call attribute is automatically
    //   added,
    // - it is not 100% clear to me whether it is always safe to enable it:
    // https://llvm.org/docs/CodeGenerator.html#tail-calls

    return r;
}

// Helper to invoke an internal module function called 'name' with arguments 'args'.
llvm::Value *llvm_invoke_internal(llvm_state &s, const std::string &name, const std::vector<llvm::Value *> &args)
{
    auto callee_f = s.module().getFunction(name);

    if (callee_f == nullptr) {
        throw std::invalid_argument("Unknown internal function: '" + name + "'");
    }

    if (callee_f->isDeclaration()) {
        throw std::invalid_argument("The internal function '" + name
                                    + "' cannot be just a declaration, a definition is needed");
    }

    // Check the number of arguments.
    if (callee_f->arg_size() != args.size()) {
        throw std::invalid_argument("Incorrect # of arguments passed while calling the internal function '" + name
                                    + "': " + std::to_string(callee_f->arg_size()) + " are expected, but "
                                    + std::to_string(args.size()) + " were provided instead");
    }
    // NOTE: perhaps in the future we should consider adding more checks here
    // (e.g., argument types).

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, args);
    assert(r != nullptr);
    // NOTE: we used to have r->setTailCall(true) here, but:
    // - when optimising, the tail call attribute is automatically
    //   added,
    // - it is not 100% clear to me whether it is always safe to enable it:
    // https://llvm.org/docs/CodeGenerator.html#tail-calls

    return r;
}

} // namespace heyoka::detail
