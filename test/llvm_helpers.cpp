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
#include <llvm/IR/Constants.h>
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

TEST_CASE("fp_pair")
{
    using detail::to_llvm_pair_type;
    using detail::to_llvm_vector_type;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state s{kw::opt_level = opt_level};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto fp_t = to_llvm_vector_type<double>(context, 1);
        auto fpp_t = to_llvm_pair_type<double>(context, 1);

        std::vector<llvm::Type *> fargs{fp_t, fp_t};
        auto *ft = llvm::FunctionType::get(fpp_t, fargs, false);
        REQUIRE(ft != nullptr);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "pair_test", &md);
        REQUIRE(f != nullptr);

        auto x = f->args().begin();
        auto y = f->args().begin() + 1;

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        llvm::Value *rv = llvm::UndefValue::get(fpp_t);
        rv = builder.CreateInsertValue(rv, builder.CreateFAdd(x, y), {0});
        rv = builder.CreateInsertValue(rv, builder.CreateFSub(x, y), {1});

        builder.CreateRet(rv);

        // Verify.
        s.verify_function(f);

        // Run the optimisation pass.
        s.optimise();

        // Compile.
        s.compile();

        // Fetch the function pointer.
        using p_double = struct {
            double x;
            double y;
        };
        auto f_ptr = reinterpret_cast<p_double (*)(double, double)>(s.jit_lookup("pair_test"));

        p_double res{0, 0};
        res = f_ptr(42, -1);

        REQUIRE(res.x == 41);
        REQUIRE(res.y == 43);
    }
}

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

        // Fetch the function pointer.
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

        // NOTE: if we don't do the cleanup of f before re-throwing,
        // on the OSX CI the test will hang on shutdown (i.e., all the tests
        // run correctly but the test program hangs on exit). Not sure what is
        // going on with that, perhaps another bad interaction between LLVM and
        // exceptions?
        auto thrower = [&]() {
            try {
                llvm_while_loop(
                    s, [&]() -> llvm::Value * { throw std::runtime_error{"aa"}; }, [&]() {});
            } catch (...) {
                f->eraseFromParent();

                throw;
            }
        };

        REQUIRE_THROWS_AS(thrower(), std::runtime_error);
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

        auto thrower = [&]() {
            try {
                llvm_while_loop(
                    s, [&]() -> llvm::Value * { return builder.CreateICmpULT(builder.CreateLoad(retval), final_n); },
                    [&]() { throw std::runtime_error{"aa"}; });
            } catch (...) {
                f->eraseFromParent();

                throw;
            }
        };

        REQUIRE_THROWS_AS(thrower(), std::runtime_error);
    }
}
