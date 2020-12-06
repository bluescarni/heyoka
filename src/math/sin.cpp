// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sin.hpp>

namespace heyoka
{

namespace detail
{

sin_impl::sin_impl(expression e) : func_base("sin", std::vector{std::move(e)}) {}

sin_impl::sin_impl() : sin_impl(0_dbl) {}

llvm::Value *sin_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);

    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
        if (const auto sfn = sleef_function_name(s.context(), "sin", vec_t->getElementType(),
                                                 boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
            !sfn.empty()) {
            return llvm_invoke_external(
                s, sfn, vec_t, args,
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
        }
    }

    return llvm_invoke_intrinsic(s, "llvm.sin", {args[0]->getType()}, args);
}

llvm::Value *sin_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);

    return llvm_invoke_intrinsic(s, "llvm.sin", {args[0]->getType()}, args);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *sin_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);

    auto &builder = s.builder();

    // Decompose the argument into scalars.
    auto scalars = vector_to_scalars(builder, args[0]);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> retvals;
    for (auto scal : scalars) {
        retvals.push_back(llvm_invoke_external(
            s, "heyoka_sin128", scal->getType(), {scal},
            // NOTE: in theory we may add ReadNone here as well,
            // but for some reason, at least up to LLVM 10,
            // this causes strange codegen issues. Revisit
            // in the future.
            {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, retvals);
}

#endif

} // namespace detail

} // namespace heyoka
