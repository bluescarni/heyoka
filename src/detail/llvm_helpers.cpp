// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka::detail
{

// Helper to load the data from pointer ptr as a vector of size vector_size. If vector_size is
// 1, a scalar is loaded instead.
llvm::Value *load_vector_from_memory(llvm::IRBuilder<> &builder, llvm::Value *ptr, std::uint32_t vector_size)
{
    assert(vector_size > 0u);

    if (vector_size == 1u) {
        // Scalar case.
        return builder.CreateLoad(ptr);
    }

    // Fetch the pointer type (this will result in an assertion
    // failure if ptr is not a pointer).
    auto ptr_t = llvm::cast<llvm::PointerType>(ptr->getType());

    // Create the vector type.
    auto vector_t = make_vector_type(ptr_t->getElementType(), vector_size);
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

    llvm::Value *vec = llvm::UndefValue::get(make_vector_type(c->getType(), vector_size));
    assert(vec != nullptr);

    // Fill up the vector with insertelement.
    for (std::uint32_t i = 0; i < vector_size; ++i) {
        // NOTE: the insertelement instruction returns
        // a new vector with the element at index i changed.
        vec = builder.CreateInsertElement(vec, c, i);
    }

    return vec;
}

llvm::Type *make_vector_type(llvm::Type *t, std::uint32_t vector_size)
{
    assert(t != nullptr);
    assert(vector_size > 0u);

    if (vector_size == 1u) {
        return t;
    } else {
        auto retval =
#if LLVM_VERSION_MAJOR == 10
            llvm::VectorType::get
#else
            llvm::FixedVectorType::get
#endif
            (t, boost::numeric_cast<unsigned>(vector_size));

        assert(retval != nullptr);

        return retval;
    }
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
    auto vector_t = make_vector_type(scalar_t, boost::numeric_cast<std::uint32_t>(vector_size));
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

    if (sum.size() == std::numeric_limits<decltype(sum.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_sum()");
    }

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

// Create an LLVM for loop in the form:
//
// for (auto i = begin; i < end; i = next_cur(i)) {
//   body(i);
// }
//
// The default implementation of i = next_cur(i),
// if next_cur is not provided, is ++i.
//
// begin/end must be 32-bit unsigned integer values.
void llvm_loop_u32(llvm_state &s, llvm::Value *begin, llvm::Value *end, const std::function<void(llvm::Value *)> &body,
                   const std::function<llvm::Value *(llvm::Value *)> &next_cur)
{
    assert(body);
    assert(begin->getType() == end->getType());
    assert(begin->getType() == s.builder().getInt32Ty());

    auto &context = s.context();
    auto &builder = s.builder();

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Pre-create loop and afterloop blocks. Note that these have just
    // been created, they have not been inserted yet in the IR.
    auto *loop_bb = llvm::BasicBlock::Create(context);
    auto *after_bb = llvm::BasicBlock::Create(context);

    // NOTE: we need a special case if the body of the loop is
    // never to be executed (that is, begin >= end).
    // In such a case, we will jump directly to after_bb.
    // NOTE: unsigned integral comparison.
    auto skip_cond = builder.CreateICmp(llvm::CmpInst::ICMP_UGE, begin, end);
    builder.CreateCondBr(skip_cond, after_bb, loop_bb);

    // Get a reference to the current block for
    // later usage in the phi node.
    auto preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    f->getBasicBlockList().push_back(loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto cur = builder.CreatePHI(builder.getInt32Ty(), 2);
    cur->addIncoming(begin, preheader_bb);

    // Execute the loop body and the post-body code.
    llvm::Value *next;
    try {
        body(cur);

        // Compute the next value of the iteration. Use the next_cur
        // function if provided, otherwise, by default, increase cur by 1.
        // NOTE: addition works regardless of integral signedness.
        next = next_cur ? next_cur(cur) : builder.CreateAdd(cur, builder.getInt32(1));
    } catch (...) {
        // NOTE: at this point after_bb has not been
        // inserted into any parent, and thus it will not
        // be cleaned up automatically. Do it manually.
        after_bb->deleteValue();

        throw;
    }

    // Compute the end condition.
    // NOTE: we use the unsigned less-than predicate.
    auto end_cond = builder.CreateICmp(llvm::CmpInst::ICMP_ULT, next, end);

    // Get a reference to the current block for later use,
    // and insert the "after loop" block.
    auto loop_end_bb = builder.GetInsertBlock();
    f->getBasicBlockList().push_back(after_bb);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(end_cond, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    cur->addIncoming(next, loop_end_bb);
}

// Given an input pointer value, return the
// pointed-to type.
llvm::Type *pointee_type(llvm::Value *ptr)
{
    return llvm::cast<llvm::PointerType>(ptr->getType())->getElementType();
}

// Small helper to fetch a string representation
// of an LLVM type.
std::string llvm_type_name(llvm::Type *t)
{
    assert(t != nullptr);

    std::string retval;
    llvm::raw_string_ostream ostr(retval);

    t->print(ostr, false, true);

    return ostr.str();
}

// This function will return true if:
//
// - the return type of f is ret, and
// - the argument types of f are the same as in 'args'.
//
// Otherwise, the function will return false.
bool compare_function_signature(llvm::Function *f, llvm::Type *ret, const std::vector<llvm::Type *> &args)
{
    assert(f != nullptr);
    assert(ret != nullptr);

    if (ret != f->getReturnType()) {
        // Mismatched return types.
        return false;
    }

    auto it = f->arg_begin();
    for (auto arg_type : args) {
        if (it == f->arg_end() || it->getType() != arg_type) {
            // f has fewer arguments than args, or the current
            // arguments' types do not match.
            return false;
        }
        ++it;
    }

    // In order for the signatures to match,
    // we must be at the end of f's arguments list
    // (otherwise f has more arguments than args).
    return it == f->arg_end();
}

// Create an LLVM if statement in the form:
// if (cond) {
//   then_f();
// } else {
//   else_f();
// }
void llvm_if_then_else(llvm_state &s, llvm::Value *cond, const std::function<void()> &then_f,
                       const std::function<void()> &else_f)
{
    auto &context = s.context();
    auto &builder = s.builder();

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Create and insert the "then" block.
    auto *then_bb = llvm::BasicBlock::Create(context, "", f);

    // Create but do not insert the "else" and merge blocks.
    auto *else_bb = llvm::BasicBlock::Create(context);
    auto *merge_bb = llvm::BasicBlock::Create(context);

    // Create the conditional jump.
    builder.CreateCondBr(cond, then_bb, else_bb);

    // Emit the code for the "then" branch.
    builder.SetInsertPoint(then_bb);
    try {
        then_f();
    } catch (...) {
        // NOTE: else_bb and merge_bb have not been
        // inserted into any parent yet, clean them
        // up manually.
        else_bb->deleteValue();
        merge_bb->deleteValue();

        throw;
    }

    // Jump to the merge block.
    builder.CreateBr(merge_bb);

    // Emit the "else" block.
    f->getBasicBlockList().push_back(else_bb);
    builder.SetInsertPoint(else_bb);
    try {
        else_f();
    } catch (...) {
        // NOTE: merge_bb has not been
        // inserted into any parent yet, clean it
        // up manually.
        merge_bb->deleteValue();

        throw;
    }

    // Jump to the merge block.
    builder.CreateBr(merge_bb);

    // Emit the merge block.
    f->getBasicBlockList().push_back(merge_bb);
    builder.SetInsertPoint(merge_bb);
}

} // namespace heyoka::detail
