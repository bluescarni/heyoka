// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka::detail
{

namespace
{

// The global type map to associate a C++ type to an LLVM type.
const auto type_map = []() {
    std::unordered_map<std::type_index, llvm::Type *(*)(llvm::LLVMContext &)> retval;

    // Try to associate C++ double to LLVM double.
    if (std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53) {
        retval[typeid(double)] = [](llvm::LLVMContext &c) {
            auto ret = llvm::Type::getDoubleTy(c);
            assert(ret != nullptr);
            return ret;
        };
    }

    // Try to associate C++ long double to LLVM double or x86_fp80.
    if (std::numeric_limits<long double>::is_iec559) {
        if (std::numeric_limits<long double>::digits == 53) {
            retval[typeid(long double)] = [](llvm::LLVMContext &c) {
                // IEEE double-precision type (this is the case on MSVC for instance).
                auto ret = llvm::Type::getDoubleTy(c);
                assert(ret != nullptr);
                return ret;
            };
        } else if (std::numeric_limits<long double>::digits == 64) {
            retval[typeid(long double)] = [](llvm::LLVMContext &c) {
                // x86 extended precision format.
                auto ret = llvm::Type::getX86_FP80Ty(c);
                assert(ret != nullptr);
                return ret;
            };
        }
    }

#if defined(HEYOKA_HAVE_REAL128)

    // Associate mppp::real128 to fp128.
    retval[typeid(mppp::real128)] = [](llvm::LLVMContext &c) {
        auto ret = llvm::Type::getFP128Ty(c);
        assert(ret != nullptr);
        return ret;
    };

#endif

    return retval;
}();

} // namespace

// Implementation of the function to associate a C++ type to
// an LLVM type.
llvm::Type *to_llvm_type_impl(llvm::LLVMContext &c, const std::type_info &tp)
{
    const auto it = type_map.find(tp);

    if (it == type_map.end()) {
        throw std::invalid_argument("Unable to associate the C++ type '{}' to an LLVM type"_format(tp.name()));
    } else {
        return it->second(c);
    }
}

// Helper to produce a unique string for the type t.
std::string llvm_mangle_type(llvm::Type *t)
{
    assert(t != nullptr);

    if (auto *v_t = llvm::dyn_cast<llvm::VectorType>(t)) {
        // If the type is a vector, get the name of the element type
        // and append the vector size.
        return "{}_{}"_format(llvm_type_name(v_t->getElementType()), v_t->getNumElements());
    } else {
        // Otherwise just return the type name.
        return llvm_type_name(t);
    }
}

// Helper to load the data from pointer ptr as a vector of size vector_size. If vector_size is
// 1, a scalar is loaded instead.
llvm::Value *load_vector_from_memory(ir_builder &builder, llvm::Value *ptr, std::uint32_t vector_size)
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
void store_vector_to_memory(ir_builder &builder, llvm::Value *ptr, llvm::Value *vec)
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
llvm::Value *vector_splat(ir_builder &builder, llvm::Value *c, std::uint32_t vector_size)
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
std::vector<llvm::Value *> vector_to_scalars(ir_builder &builder, llvm::Value *vec)
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
llvm::Value *scalars_to_vector(ir_builder &builder, const std::vector<llvm::Value *> &scalars)
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

// Pairwise reduction of a vector of LLVM values.
llvm::Value *pairwise_reduce(std::vector<llvm::Value *> &vals,
                             const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &f)
{
    assert(!vals.empty());
    assert(f);

    // LCOV_EXCL_START
    if (vals.size() == std::numeric_limits<decltype(vals.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_reduce()");
    }
    // LCOV_EXCL_STOP

    while (vals.size() != 1u) {
        std::vector<llvm::Value *> new_vals;

        for (decltype(vals.size()) i = 0; i < vals.size(); i += 2u) {
            if (i + 1u == vals.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_vals.push_back(vals[i]);
            } else {
                new_vals.push_back(f(vals[i], vals[i + 1u]));
            }
        }

        new_vals.swap(vals);
    }

    return vals[0];
}

// Pairwise summation of a vector of LLVM values.
// https://en.wikipedia.org/wiki/Pairwise_summation
llvm::Value *pairwise_sum(ir_builder &builder, std::vector<llvm::Value *> &sum)
{
    return pairwise_reduce(
        sum, [&builder](llvm::Value *a, llvm::Value *b) -> llvm::Value * { return builder.CreateFAdd(a, b); });
}

// Helper to invoke an intrinsic function with arguments 'args'. 'types' are the argument type(s) for
// overloaded intrinsics.
llvm::Value *llvm_invoke_intrinsic(llvm_state &s, const std::string &name, const std::vector<llvm::Type *> &types,
                                   const std::vector<llvm::Value *> &args)
{
    // Fetch the intrinsic ID from the name.
    const auto intrinsic_ID = llvm::Function::lookupIntrinsicID(name);
    if (intrinsic_ID == 0) {
        throw std::invalid_argument("Cannot fetch the ID of the intrinsic '{}'"_format(name));
    }

    // Fetch the declaration.
    // NOTE: for generic intrinsics to work, we need to specify
    // the desired argument type(s). See:
    // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
    // And the docs of the getDeclaration() function.
    auto callee_f = llvm::Intrinsic::getDeclaration(&s.module(), intrinsic_ID, types);
    if (callee_f == nullptr) {
        throw std::invalid_argument("Error getting the declaration of the intrinsic '{}'"_format(name));
    }
    if (!callee_f->isDeclaration()) {
        // It does not make sense to have a definition of a builtin.
        throw std::invalid_argument("The intrinsic '{}' must be only declared, not defined"_format(name));
    }

    // Check the number of arguments.
    if (callee_f->arg_size() != args.size()) {
        throw std::invalid_argument(
            "Incorrect # of arguments passed while calling the intrinsic '{}': {} are "
            "expected, but {} were provided instead"_format(name, callee_f->arg_size(), args.size()));
    }

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, args);
    assert(r != nullptr);

    return r;
}

// Helper to invoke an external function called 'name' with arguments args and return type ret_type.
llvm::Value *llvm_invoke_external(llvm_state &s, const std::string &name, llvm::Type *ret_type,
                                  const std::vector<llvm::Value *> &args, const std::vector<int> &attrs)
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
            throw std::invalid_argument("Unable to create the prototype for the external function '{}'"_format(name));
        }

        // Add the function attributes.
        for (const auto &att : attrs) {
            // NOTE: convert back to the LLVM attribute enum.
            callee_f->addFnAttr(boost::numeric_cast<llvm::Attribute::AttrKind>(att));
        }
    } else {
        // The function declaration exists already. Check that it is only a
        // declaration and not a definition.
        if (!callee_f->isDeclaration()) {
            throw std::invalid_argument("Cannot call the function '{}' as an external function, because "
                                        "it is defined as an internal module function"_format(name));
        }
        // Check the number of arguments.
        if (callee_f->arg_size() != args.size()) {
            throw std::invalid_argument(
                "Incorrect # of arguments passed while calling the external function '{}': {} "
                "are expected, but {} were provided instead"_format(name, callee_f->arg_size(), args.size()));
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
        throw std::invalid_argument("Unknown internal function: '{}'"_format(name));
    }

    if (callee_f->isDeclaration()) {
        throw std::invalid_argument("The internal function '{}' cannot be just a "
                                    "declaration, a definition is needed"_format(name));
    }

    // Check the number of arguments.
    if (callee_f->arg_size() != args.size()) {
        throw std::invalid_argument(
            "Incorrect # of arguments passed while calling the internal function '{}': {} are "
            "expected, but {} were provided instead"_format(name, callee_f->arg_size(), args.size()));
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

    assert(cond->getType() == builder.getInt1Ty());

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

// Helper to create a global zero-inited array variable in the module m
// with type t. The array is mutable and with internal linkage.
llvm::Value *make_global_zero_array(llvm::Module &m, llvm::ArrayType *t)
{
    assert(t != nullptr);

    // Make the global array.
    auto gl_arr = new llvm::GlobalVariable(m, t, false, llvm::GlobalVariable::InternalLinkage,
                                           llvm::ConstantAggregateZero::get(t));

    // Return it.
    return gl_arr;
}

// Helper to invoke an external function on a vector argument.
// The function will be called on each element of the vector separately,
// and the return will be re-assembled as a vector.
llvm::Value *call_extern_vec(llvm_state &s, llvm::Value *arg, const std::string &fname)
{
    auto &builder = s.builder();

    // Decompose the argument into scalars.
    auto scalars = vector_to_scalars(builder, arg);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> retvals;
    for (auto scal : scalars) {
        retvals.push_back(llvm_invoke_external(
            s, fname, scal->getType(), {scal},
            // NOTE: in theory we may add ReadNone here as well,
            // but for some reason, at least up to LLVM 10,
            // this causes strange codegen issues. Revisit
            // in the future.
            {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, retvals);
}

// Create an LLVM for loop in the form:
//
// while (cond()) {
//   body();
// }
void llvm_while_loop(llvm_state &s, const std::function<llvm::Value *()> &cond, const std::function<void()> &body)
{
    assert(body);
    assert(cond);

    auto &context = s.context();
    auto &builder = s.builder();

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Do a first evaluation of cond.
    // NOTE: if this throws, we have not created any block
    // yet, no need for manual cleanup.
    auto cmp = cond();
    assert(cmp != nullptr);
    assert(cmp->getType() == builder.getInt1Ty());

    // Pre-create loop and afterloop blocks. Note that these have just
    // been created, they have not been inserted yet in the IR.
    auto *loop_bb = llvm::BasicBlock::Create(context);
    auto *after_bb = llvm::BasicBlock::Create(context);

    // NOTE: we need a special case if the body of the loop is
    // never to be executed (that is, cond returns false).
    // In such a case, we will jump directly to after_bb.
    builder.CreateCondBr(builder.CreateNot(cmp), after_bb, loop_bb);

    // Get a reference to the current block for
    // later usage in the phi node.
    auto preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    f->getBasicBlockList().push_back(loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto cur = builder.CreatePHI(builder.getInt1Ty(), 2);
    cur->addIncoming(cmp, preheader_bb);

    // Execute the loop body and the post-body code.
    try {
        body();

        // Compute the end condition.
        cmp = cond();
        assert(cmp != nullptr);
        assert(cmp->getType() == builder.getInt1Ty());
    } catch (...) {
        // NOTE: at this point after_bb has not been
        // inserted into any parent, and thus it will not
        // be cleaned up automatically. Do it manually.
        after_bb->deleteValue();

        throw;
    }

    // Get a reference to the current block for later use,
    // and insert the "after loop" block.
    auto loop_end_bb = builder.GetInsertBlock();
    f->getBasicBlockList().push_back(after_bb);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(cmp, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    cur->addIncoming(cmp, loop_end_bb);
}

// Helper to compute sin and cos simultaneously.
std::pair<llvm::Value *, llvm::Value *> llvm_sincos(llvm_state &s, llvm::Value *x)
{
    auto &context = s.context();

#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == llvm::Type::getFP128Ty(context)) {
        // NOTE: for __float128 we cannot use the intrinsics, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector argument to scalars.
        auto x_scalars = vector_to_scalars(builder, x);

        // Execute the sincosq() function on the scalar values and store
        // the results in res_scalars.
        // NOTE: need temp storage because sincosq uses pointers
        // for output values.
        auto s_all = builder.CreateAlloca(x_t);
        auto c_all = builder.CreateAlloca(x_t);
        std::vector<llvm::Value *> res_sin, res_cos;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            llvm_invoke_external(
                s, "sincosq", builder.getVoidTy(), {x_scalars[i], s_all, c_all},
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});

            res_sin.emplace_back(builder.CreateLoad(s_all));
            res_cos.emplace_back(builder.CreateLoad(c_all));
        }

        // Reconstruct the return value as a vector.
        return {scalars_to_vector(builder, res_sin), scalars_to_vector(builder, res_cos)};
    } else {
#endif
        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(x->getType())) {
            // NOTE: although there exists a SLEEF function for computing sin/cos
            // at the same time, we cannot use it directly because it returns a pair
            // of SIMD vectors rather than a single one and that does not play
            // well with the calling conventions. In theory we could write a wrapper
            // for these sincos functions using pointers for output values,
            // but compiling such a wrapper requires correctly
            // setting up the SIMD compilation flags. Perhaps we can consider this in the
            // future to improve performance.
            const auto sfn_sin = sleef_function_name(context, "sin", vec_t->getElementType(),
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
            const auto sfn_cos = sleef_function_name(context, "cos", vec_t->getElementType(),
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));

            if (!sfn_sin.empty() && !sfn_cos.empty()) {
                auto ret_sin = llvm_invoke_external(
                    s, sfn_sin, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});

                auto ret_cos = llvm_invoke_external(
                    s, sfn_cos, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});

                return {ret_sin, ret_cos};
            }
        }

        // Compute sin and cos via intrinsics.
        auto *sin_x = llvm_invoke_intrinsic(s, "llvm.sin", {x->getType()}, {x});
        auto *cos_x = llvm_invoke_intrinsic(s, "llvm.cos", {x->getType()}, {x});

        return {sin_x, cos_x};
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to compute abs(x_v).
llvm::Value *llvm_abs(llvm_state &s, llvm::Value *x_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector argument.
    auto *x_t = x_v->getType()->getScalarType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v);

        // Execute the fabsq() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        res_scalars.reserve(x_scalars.size());
        for (auto *x_scal : x_scalars) {
            res_scalars.push_back(llvm_invoke_external(
                s, "fabsq", x_t, {x_scal},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        return llvm_invoke_intrinsic(s, "llvm.fabs", {x_v->getType()}, {x_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to reduce x modulo y, that is, to compute:
// x - y * floor(x / y).
llvm::Value *llvm_modulus(llvm_state &s, llvm::Value *x, llvm::Value *y)
{
    auto &builder = s.builder();

#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto *x_t = x->getType()->getScalarType();

    auto &context = s.context();

    if (x_t == llvm::Type::getFP128Ty(context)) {
        // NOTE: for __float128 we cannot use the intrinsics, we need
        // to call an external function.

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x), y_scalars = vector_to_scalars(builder, y);

        // Compute the modulus on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            auto quo = builder.CreateFDiv(x_scalars[i], y_scalars[i]);
            auto fl_quo = llvm_invoke_external(
                s, "floorq", x_t, {quo},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});

            res_scalars.push_back(builder.CreateFSub(x_scalars[i], builder.CreateFMul(y_scalars[i], fl_quo)));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        auto quo = builder.CreateFDiv(x, y);
        auto fl_quo = llvm_invoke_intrinsic(s, "llvm.floor", {quo->getType()}, {quo});

        return builder.CreateFSub(x, builder.CreateFMul(y, fl_quo));
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Minimum value.
llvm::Value *llvm_min(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto *x_t = x_v->getType()->getScalarType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v), y_scalars = vector_to_scalars(builder, y_v);

        // Execute the heyoka_minnum128() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_minnum128", x_t, {x_scalars[i], y_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        return llvm_invoke_intrinsic(s, "llvm.minnum", {x_v->getType()}, {x_v, y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Maximum value.
llvm::Value *llvm_max(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto *x_t = x_v->getType()->getScalarType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v), y_scalars = vector_to_scalars(builder, y_v);

        // Execute the heyoka_max128() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_max128", x_t, {x_scalars[i], y_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        // Return max(a, b).
        return llvm_invoke_intrinsic(s, "llvm.maxnum", {x_v->getType()}, {x_v, y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Branchless sign function.
// NOTE: requires FP value.
llvm::Value *llvm_sgn(llvm_state &s, llvm::Value *val)
{
    assert(val != nullptr);
    assert(val->getType()->getScalarType()->isFloatingPointTy());

    auto &builder = s.builder();

    // Build the zero constant.
    auto zero = llvm::Constant::getNullValue(val->getType());

    // Run the comparisons.
    auto cmp0 = builder.CreateFCmpOLT(zero, val);
    auto cmp1 = builder.CreateFCmpOLT(val, zero);

    // Convert to int32.
    llvm::Type *int_type;
    if (auto *v_t = llvm::dyn_cast<llvm::VectorType>(cmp0->getType())) {
        int_type = make_vector_type(builder.getInt32Ty(), boost::numeric_cast<std::uint32_t>(v_t->getNumElements()));
    } else {
        int_type = builder.getInt32Ty();
    }
    auto icmp0 = builder.CreateZExt(cmp0, int_type);
    auto icmp1 = builder.CreateZExt(cmp1, int_type);

    // Compute and return the result.
    return builder.CreateSub(icmp0, icmp1);
}

namespace
{

// Add a function to count the number of sign changes in the coefficients
// of a polynomial of degree n. The coefficients are SIMD vectors of size batch_size
// and scalar type T.
template <typename T>
llvm::Function *llvm_add_csc_impl(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding a sign changes counter function");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto tp = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name.
    const auto fname = "heyoka_csc_degree_{}_{}"_format(n, llvm_mangle_type(tp));

    // The function arguments:
    // - pointer to the return value,
    // - pointer to the array of coefficients.
    // NOTE: both pointers are to the scalar counterparts
    // of the vector types, so that we can call this from regular
    // C++ code.
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(builder.getInt32Ty()),
                                    llvm::PointerType::getUnqual(to_llvm_type<T>(context))};

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is void.
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto out_ptr = f->args().begin();
        out_ptr->setName("out_ptr");
        out_ptr->addAttr(llvm::Attribute::NoCapture);
        out_ptr->addAttr(llvm::Attribute::NoAlias);
        out_ptr->addAttr(llvm::Attribute::WriteOnly);

        auto cf_ptr = f->args().begin() + 1;
        cf_ptr->setName("cf_ptr");
        cf_ptr->addAttr(llvm::Attribute::NoCapture);
        cf_ptr->addAttr(llvm::Attribute::NoAlias);
        cf_ptr->addAttr(llvm::Attribute::ReadOnly);

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Fetch the type for storing the last_nz_idx variable.
        auto last_nz_idx_t = make_vector_type(builder.getInt32Ty(), batch_size);

        // The initial last nz idx is zero for all batch elements.
        auto last_nz_idx = builder.CreateAlloca(last_nz_idx_t);
        builder.CreateStore(llvm::Constant::getNullValue(last_nz_idx_t), last_nz_idx);

        // NOTE: last_nz_idx is an index into the poly coefficient vector. Thus, in batch
        // mode, when loading from a vector of indices, we will have to apply an offset.
        // For instance, for batch_size = 4 and last_nz_idx = [0, 1, 1, 2], the actual
        // memory indices to load the scalar coefficients from are:
        // - 0 * 4 + 0 = 0
        // - 1 * 4 + 1 = 5
        // - 1 * 4 + 2 = 6
        // - 2 * 4 + 3 = 11.
        // That is, last_nz_idx * batch_size + offset, where offset is [0, 1, 2, 3].
        llvm::Value *offset;
        if (batch_size == 1u) {
            // In scalar mode the offset is simply zero.
            offset = builder.getInt32(0);
        } else {
            offset = llvm::UndefValue::get(make_vector_type(builder.getInt32Ty(), batch_size));
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                offset = builder.CreateInsertElement(offset, builder.getInt32(i), i);
            }
        }

        // Init the vector of coefficient pointers with the base pointer value.
        auto cf_ptr_v = vector_splat(builder, cf_ptr, batch_size);

        // Init the return value with zero.
        auto retval = builder.CreateAlloca(last_nz_idx_t);
        builder.CreateStore(llvm::Constant::getNullValue(last_nz_idx_t), retval);

        // The iteration range is [1, n].
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n + 1u), [&](llvm::Value *cur_n) {
            // Load the current poly coefficient(s).
            auto cur_cf = load_vector_from_memory(
                builder, builder.CreateInBoundsGEP(cf_ptr, {builder.CreateMul(cur_n, builder.getInt32(batch_size))}),
                batch_size);

            // Load the last nonzero coefficient(s).
            auto last_nz_ptr_idx = builder.CreateAdd(
                offset, builder.CreateMul(builder.CreateLoad(last_nz_idx),
                                          vector_splat(builder, builder.getInt32(batch_size), batch_size)));
            auto last_nz_ptr = builder.CreateInBoundsGEP(cf_ptr_v, {last_nz_ptr_idx});
            auto last_nz_cf = batch_size > 1u ? static_cast<llvm::Value *>(builder.CreateMaskedGather(last_nz_ptr,
#if LLVM_VERSION_MAJOR == 10
                                                                                                      alignof(T)
#else
                                                                                                      llvm::Align(alignof(T))
#endif
                                                                                                          ))
                                              : static_cast<llvm::Value *>(builder.CreateLoad(last_nz_ptr));

            // Compute the sign of the current coefficient(s).
            auto cur_sgn = llvm_sgn(s, cur_cf);

            // Compute the sign of the last nonzero coefficient(s).
            auto last_nz_sgn = llvm_sgn(s, last_nz_cf);

            // Add them and check if the result is zero (this indicates a sign change).
            auto cmp = builder.CreateICmpEQ(builder.CreateAdd(cur_sgn, last_nz_sgn),
                                            llvm::Constant::getNullValue(cur_sgn->getType()));

            // We also need to check if last_nz_sgn is zero. If that is the case, it means
            // we haven't found any nonzero coefficient yet for the polynomial and we must
            // not modify retval yet.
            auto zero_cmp = builder.CreateICmpEQ(last_nz_sgn, llvm::Constant::getNullValue(last_nz_sgn->getType()));
            cmp = builder.CreateSelect(zero_cmp, llvm::Constant::getNullValue(cmp->getType()), cmp);

            // Update retval.
            builder.CreateStore(builder.CreateAdd(builder.CreateLoad(retval), builder.CreateZExt(cmp, last_nz_idx_t)),
                                retval);

            // Update last_nz_idx.
            builder.CreateStore(
                builder.CreateSelect(builder.CreateICmpEQ(cur_sgn, llvm::Constant::getNullValue(cur_sgn->getType())),
                                     builder.CreateLoad(last_nz_idx), vector_splat(builder, cur_n, batch_size)),
                last_nz_idx);
        });

        // Store the result.
        store_vector_to_memory(builder, out_ptr, builder.CreateLoad(retval));

        // Return.
        builder.CreateRetVoid();

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        if (!compare_function_signature(f, builder.getVoidTy(), fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the sign changes counter function detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace

llvm::Function *llvm_add_csc_dbl(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    return llvm_add_csc_impl<double>(s, n, batch_size);
}

llvm::Function *llvm_add_csc_ldbl(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    return llvm_add_csc_impl<long double>(s, n, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *llvm_add_csc_f128(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    return llvm_add_csc_impl<mppp::real128>(s, n, batch_size);
}

#endif

namespace
{

// Variable template for the constant pi at different levels of precision.
template <typename T>
const auto inv_kep_E_pi = boost::math::constants::pi<T>();

#if defined(HEYOKA_HAVE_REAL128)

template <>
const mppp::real128 inv_kep_E_pi<mppp::real128> = mppp::pi_128;

#endif

// Implementation of the inverse Kepler equation.
template <typename T>
llvm::Function *llvm_add_inv_kep_E_impl(llvm_state &s, std::uint32_t batch_size)
{
    using std::nextafter;

    assert(batch_size > 0u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto tp = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name.
    const auto fname = "heyoka_inv_kep_E_{}"_format(llvm_mangle_type(tp));

    // The function arguments:
    // - eccentricity,
    // - mean anomaly.
    std::vector<llvm::Type *> fargs{tp, tp};

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is tp.
        auto *ft = llvm::FunctionType::get(tp, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ecc = f->args().begin();
        auto M_arg = f->args().begin() + 1;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(tp);

        // Reduce M modulo 2*pi.
        auto M = llvm_modulus(s, M_arg, vector_splat(builder, codegen<T>(s, number{2 * inv_kep_E_pi<T>}), batch_size));

        // Compute the initial guess from the usual elliptic expansion
        // to the third order in eccentricities:
        // E = M + e*sin(M) + e**2*sin(M)*cos(M) + e**3*sin(M)*(3/2*cos(M)**2 - 1/2) + ...
        auto [sin_M, cos_M] = llvm_sincos(s, M);
        // e*sin(M).
        auto e_sin_M = builder.CreateFMul(ecc, sin_M);
        // e*cos(M).
        auto e_cos_M = builder.CreateFMul(ecc, cos_M);
        // e**2.
        auto e2 = builder.CreateFMul(ecc, ecc);
        // cos(M)**2.
        auto cos_M_2 = builder.CreateFMul(cos_M, cos_M);

        // 3/2 and 1/2 constants.
        auto c_3_2 = vector_splat(builder, codegen<T>(s, number{T(3) / 2}), batch_size);
        auto c_1_2 = vector_splat(builder, codegen<T>(s, number{T(1) / 2}), batch_size);

        // M + e*sin(M).
        auto tmp1 = builder.CreateFAdd(M, e_sin_M);
        // e**2*sin(M)*cos(M).
        auto tmp2 = builder.CreateFMul(e_sin_M, e_cos_M);
        // e**3*sin(M).
        auto tmp3 = builder.CreateFMul(e2, e_sin_M);
        // 3/2*cos(M)**2 - 1/2.
        auto tmp4 = builder.CreateFSub(builder.CreateFMul(c_3_2, cos_M_2), c_1_2);

        // Put it together.
        auto ig1 = builder.CreateFAdd(tmp1, tmp2);
        auto ig2 = builder.CreateFMul(tmp3, tmp4);
        auto ig = builder.CreateFAdd(ig1, ig2);

        // Make extra sure the initial guess is in the [0, 2*pi) range.
        auto lb = vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
        auto ub = vector_splat(builder, codegen<T>(s, number{nextafter(2 * inv_kep_E_pi<T>, T(0))}), batch_size);
        ig = llvm_max(s, ig, lb);
        ig = llvm_min(s, ig, ub);

        // Store it.
        builder.CreateStore(ig, retval);

        // Create the counter.
        auto *counter = builder.CreateAlloca(builder.getInt32Ty());
        builder.CreateStore(builder.getInt32(0), counter);

        // Variables to store sin(E) and cos(E).
        auto sin_E = builder.CreateAlloca(tp);
        auto cos_E = builder.CreateAlloca(tp);

        // Write the initial values for sin_E and cos_E.
        auto sin_cos_E = llvm_sincos(s, builder.CreateLoad(retval));
        builder.CreateStore(sin_cos_E.first, sin_E);
        builder.CreateStore(sin_cos_E.second, cos_E);

        // Variable to hold the value of f(E) = E - e*sin(E) - M.
        auto fE = builder.CreateAlloca(tp);
        // Helper to compute f(E).
        auto fE_compute = [&]() {
            auto ret = builder.CreateFMul(ecc, builder.CreateLoad(sin_E));
            ret = builder.CreateFSub(builder.CreateLoad(retval), ret);
            return builder.CreateFSub(ret, M);
        };
        // Compute and store the initial value of f(E).
        builder.CreateStore(fE_compute(), fE);

        // Define the stopping condition functor.
        // NOTE: hard-code this for the time being.
        auto max_iter = builder.getInt32(50);
        auto loop_cond = [&,
                          // NOTE: tolerance is 4 * eps.
                          tol = vector_splat(builder, codegen<T>(s, number{std::numeric_limits<T>::epsilon() * 4}),
                                             batch_size)]() -> llvm::Value * {
            auto c_cond = builder.CreateICmpULT(builder.CreateLoad(counter), max_iter);

            // Keep on iterating as long as abs(f(E)) > tol.
            // NOTE: need reduction only in batch mode.
            auto tol_check = builder.CreateFCmpOGT(llvm_abs(s, builder.CreateLoad(fE)), tol);
            auto tol_cond = (batch_size == 1u) ? tol_check : builder.CreateOrReduce(tol_check);

            // NOTE: this is a way of creating a logical AND.
            return builder.CreateSelect(c_cond, tol_cond, llvm::ConstantInt::getNullValue(tol_cond->getType()));
        };

        // Run the loop.
        llvm_while_loop(s, loop_cond, [&, one_c = vector_splat(builder, codegen<T>(s, number{1.}), batch_size)]() {
            // Compute the new value.
            auto old_val = builder.CreateLoad(retval);
            auto new_val = builder.CreateFDiv(
                builder.CreateLoad(fE), builder.CreateFSub(one_c, builder.CreateFMul(ecc, builder.CreateLoad(cos_E))));
            new_val = builder.CreateFSub(old_val, new_val);

            // Bisect if new_val > ub.
            // NOTE: '>' is fine here, ub is the maximum allowed value.
            auto bcheck = builder.CreateFCmpOGT(new_val, ub);
            new_val = builder.CreateSelect(
                bcheck,
                builder.CreateFMul(vector_splat(builder, codegen<T>(s, number{T(1) / 2}), batch_size),
                                   builder.CreateFAdd(old_val, ub)),
                new_val);

            // Bisect if new_val < lb.
            bcheck = builder.CreateFCmpOLT(new_val, lb);
            new_val = builder.CreateSelect(
                bcheck,
                builder.CreateFMul(vector_splat(builder, codegen<T>(s, number{T(1) / 2}), batch_size),
                                   builder.CreateFAdd(old_val, lb)),
                new_val);

            // Store the new value.
            builder.CreateStore(new_val, retval);

            // Update sin_E/cos_E.
            sin_cos_E = llvm_sincos(s, new_val);
            builder.CreateStore(sin_cos_E.first, sin_E);
            builder.CreateStore(sin_cos_E.second, cos_E);

            // Update f(E).
            builder.CreateStore(fE_compute(), fE);

            // Update the counter.
            builder.CreateStore(builder.CreateAdd(builder.CreateLoad(counter), builder.getInt32(1)), counter);
        });

        // Check the counter.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(builder.CreateLoad(counter), max_iter),
            [&]() {
                llvm_invoke_external(s, "heyoka_inv_kep_E_max_iter", builder.getVoidTy(), {},
                                     {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});
            },
            []() {});

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        if (!compare_function_signature(f, tp, fargs)) {
            throw std::invalid_argument("Inconsistent function signature for the inverse Kepler equation detected");
        }
    }

    return f;
}

} // namespace

llvm::Function *llvm_add_inv_kep_E_dbl(llvm_state &s, std::uint32_t batch_size)
{
    return llvm_add_inv_kep_E_impl<double>(s, batch_size);
}

llvm::Function *llvm_add_inv_kep_E_ldbl(llvm_state &s, std::uint32_t batch_size)
{
    return llvm_add_inv_kep_E_impl<long double>(s, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *llvm_add_inv_kep_E_f128(llvm_state &s, std::uint32_t batch_size)
{
    return llvm_add_inv_kep_E_impl<mppp::real128>(s, batch_size);
}

#endif

namespace
{

// Helper to create a global const array containing
// all binomial coefficients up to (n, n). The coefficients are stored
// as scalars and the return value is a pointer to the first coefficient.
// The array has shape (n + 1, n + 1) and it is stored in row-major format.
template <typename T>
llvm::Value *llvm_add_bc_array_impl(llvm_state &s, std::uint32_t n)
{
    // Overflow check.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || (n + 1u) > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding an array of binomial coefficients");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the array type.
    auto *arr_type
        = llvm::ArrayType::get(to_llvm_type<T>(context), boost::numeric_cast<std::uint64_t>((n + 1u) * (n + 1u)));

    // Generate the binomials as constants.
    std::vector<llvm::Constant *> bc_const;
    for (std::uint32_t i = 0; i <= n; ++i) {
        for (std::uint32_t j = 0; j <= n; ++j) {
            // NOTE: the Boost implementation requires j <= i. We don't care about
            // j > i anyway.
            const auto val = (j <= i) ? boost::math::binomial_coefficient<T>(boost::numeric_cast<unsigned>(i),
                                                                             boost::numeric_cast<unsigned>(j))
                                      : T(0);
            bc_const.push_back(llvm::cast<llvm::Constant>(codegen<T>(s, number{val})));
        }
    }

    // Create the global array.
    auto *bc_const_arr = llvm::ConstantArray::get(arr_type, bc_const);
    auto *g_bc_const_arr = new llvm::GlobalVariable(md, bc_const_arr->getType(), true,
                                                    llvm::GlobalVariable::InternalLinkage, bc_const_arr);

    // Get out a pointer to the beginning of the array.
    return builder.CreateInBoundsGEP(g_bc_const_arr, {builder.getInt32(0), builder.getInt32(0)});
}

} // namespace

llvm::Value *llvm_add_bc_array_dbl(llvm_state &s, std::uint32_t n)
{
    return llvm_add_bc_array_impl<double>(s, n);
}

llvm::Value *llvm_add_bc_array_ldbl(llvm_state &s, std::uint32_t n)
{
    return llvm_add_bc_array_impl<long double>(s, n);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *llvm_add_bc_array_f128(llvm_state &s, std::uint32_t n)
{
    return llvm_add_bc_array_impl<mppp::real128>(s, n);
}

#endif

} // namespace heyoka::detail

// NOTE: this function will be called by the LLVM implementation
// of the inverse Kepler function when the maximum number of iterations
// is exceeded.
extern "C" HEYOKA_DLL_PUBLIC void heyoka_inv_kep_E_max_iter() noexcept
{
    heyoka::detail::get_logger()->warn("iteration limit exceeded while solving the elliptic inverse Kepler equation");
}
