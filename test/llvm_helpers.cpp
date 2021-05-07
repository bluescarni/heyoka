// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("while_loop")
{
    using detail::llvm_while_loop;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state s{kw::opt_level = opt_level};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto val_t = builder.getInt32Ty();

        std::vector<llvm::Type *> fargs{val_t};
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "count_n", &md);

        auto final_n = f->args().begin();

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto retval = builder.CreateAlloca(val_t);
        builder.CreateStore(builder.getInt32(0), retval);

        llvm_while_loop(
            s, [&]() -> llvm::Value * { return builder.CreateICmpULT(builder.CreateLoad(retval), final_n); },
            [&]() { builder.CreateStore(builder.CreateAdd(builder.CreateLoad(retval), builder.getInt32(1)), retval); });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

        // Verify.
        s.verify_function(f);

        // Run the optimisation pass.
        s.optimise();

        // Compile.
        s.compile();

        // Fetch the function pointerr
        auto f_ptr = reinterpret_cast<std::uint32_t (*)(std::uint32_t)>(s.jit_lookup("count_n"));

        REQUIRE(f_ptr(0) == 0u);
        REQUIRE(f_ptr(1) == 1u);
        REQUIRE(f_ptr(2) == 2u);
        REQUIRE(f_ptr(3) == 3u);
        REQUIRE(f_ptr(4) == 4u);
    }

    // Error handling.
    {
        llvm_state s;

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto val_t = builder.getInt32Ty();

        std::vector<llvm::Type *> fargs{val_t};
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "count_n", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto retval = builder.CreateAlloca(val_t);
        builder.CreateStore(builder.getInt32(0), retval);

        REQUIRE_THROWS_AS(llvm_while_loop(
                              s, [&]() -> llvm::Value * { throw std::runtime_error{"aa"}; }, [&]() {}),
                          std::runtime_error);
    }

    {
        llvm_state s;

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto val_t = builder.getInt32Ty();

        std::vector<llvm::Type *> fargs{val_t};
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "count_n", &md);

        auto final_n = f->args().begin();

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto retval = builder.CreateAlloca(val_t);
        builder.CreateStore(builder.getInt32(0), retval);

        REQUIRE_THROWS_AS(
            llvm_while_loop(
                s, [&]() -> llvm::Value * { return builder.CreateICmpULT(builder.CreateLoad(retval), final_n); },
                [&]() { throw std::runtime_error{"aa"}; }),
            std::runtime_error);
    }
}
