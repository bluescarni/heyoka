// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_FUNC_CREATE_HPP
#define HEYOKA_DETAIL_LLVM_FUNC_CREATE_HPP

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/core.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to create an LLVM function.
// NOTE: the purpose of this helper is to check that the function was created with
// the requested name: LLVM will silently change the name of the created function
// if it already exists in a module, and in some cases we want to prevent
// this from happening.
template <typename... Args>
llvm::Function *llvm_func_create(llvm::FunctionType *tp, llvm::Function::LinkageTypes linkage, const std::string &name,
                                 Args &&...args)
{
    llvm::Function *ret = llvm::Function::Create(tp, linkage, name, std::forward<Args>(args)...);
    assert(ret != nullptr);

    if (ret->getName() != name) {
        // Remove function before throwing.
        ret->eraseFromParent();

        throw std::invalid_argument(fmt::format("Unable to create an LLVM function with name '{}'", name));
    }

    return ret;
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
