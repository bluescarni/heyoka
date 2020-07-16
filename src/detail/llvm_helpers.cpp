// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <utility>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/llvm_helpers.hpp>

namespace heyoka::detail
{

std::pair<llvm::Value *, llvm::Value *> kahan_acc(llvm::IRBuilder<> &builder, llvm::Value *x, llvm::Value *sum,
                                                  llvm::Value *c)
{
    llvm::FastMathFlags fmf;

    auto y = llvm::cast<llvm::Instruction>(builder.CreateFSub(x, c, "kahan_a"));
    assert(y != nullptr);
    y->setFastMathFlags(fmf);

    auto t = llvm::cast<llvm::Instruction>(builder.CreateFAdd(sum, y, "kahan_b"));
    assert(t != nullptr);
    t->setFastMathFlags(fmf);

    auto t_m_sum = llvm::cast<llvm::Instruction>(builder.CreateFSub(t, sum, "kahan_c"));
    assert(t_m_sum != nullptr);
    t_m_sum->setFastMathFlags(fmf);

    auto new_c = llvm::cast<llvm::Instruction>(builder.CreateFSub(t_m_sum, y, "kahan_d"));
    assert(new_c != nullptr);
    new_c->setFastMathFlags(fmf);

    return std::pair<llvm::Value *, llvm::Value *>{t, new_c};
}

} // namespace heyoka::detail
