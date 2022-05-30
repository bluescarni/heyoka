// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <llvm/IR/Function.h>

#include <heyoka/detail/cm_utils.hpp>

namespace heyoka::detail
{

bool llvm_func_name_compare::operator()(const llvm::Function *f0, const llvm::Function *f1) const
{
    return f0->getName() < f1->getName();
}

} // namespace heyoka::detail
