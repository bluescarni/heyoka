// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <cstdint>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state s{kw::opt_level = opt_level};

        auto &md = s.module();
        auto &bld = s.builder();
        auto &ctx = s.context();

        // Fetch a couple of types.
        auto *ptr_t = bld.getPtrTy();
        auto *i32_t = bld.getInt32Ty();

        // Null pointer constant.
        auto *nullp = llvm::ConstantPointerNull::get(ptr_t);

        // Define the function type for the coroutine.
        auto *coro_ft = llvm::FunctionType::get(ptr_t, {i32_t}, false);

        // Create the coroutine.
        auto *coro_f = llvm::Function::Create(coro_ft, llvm::Function::PrivateLinkage, "f", &md);
        coro_f->setPresplitCoroutine();

        // Fetch the function argument.
        auto *limit = coro_f->getArg(0);

        // Create a new basic block to start insertion into.
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", coro_f));

        // Prepare space for the promise.
        auto *coro_promise = bld.CreateAlloca(i32_t);
        coro_promise->setName("coro_promise");

        // Get the coro id.
        auto *coro_id = detail::llvm_invoke_intrinsic(
            bld, "llvm.coro.id", {},
            {bld.getInt32(boost::numeric_cast<std::uint32_t>(alignof(std::max_align_t))), coro_promise, nullp, nullp});
        coro_id->setName("coro_id");

        // Next, we determine if the coroutine needs allocation.
        auto *coro_need_alloc = detail::llvm_invoke_intrinsic(bld, "llvm.coro.alloc", {}, {coro_id});
        coro_need_alloc->setName("coro_need_alloc");

        // If we need allocation, perform it and return the pointer to the allocated memory. Otherwise, get a null
        // pointer.
        auto *coro_ptr_alloca = bld.CreateAlloca(ptr_t);
        detail::llvm_if_then_else(
            s, coro_need_alloc,
            [&bld, &s, i32_t, ptr_t, coro_ptr_alloca]() {
                // Determine how much storage we need to dynamically allocate with malloc.
                auto *coro_frame_size
                    = detail::to_size_t(s, detail::llvm_invoke_intrinsic(bld, "llvm.coro.size", {i32_t}, {}));

                // Allocate with malloc().
                // NOTE: will need a wrapper here that aborts in case of allocation failure.
                auto *coro_ptr = detail::llvm_invoke_external(s, "malloc", ptr_t, {coro_frame_size});

                // Store the result in the alloca.
                // NOLINTNEXTLINE(readability-suspicious-call-argument)
                bld.CreateStore(coro_ptr, coro_ptr_alloca);
            },
            [&bld, nullp, coro_ptr_alloca]() {
                // Store a null pointer in the alloca.
                bld.CreateStore(nullp, coro_ptr_alloca);
            });
        auto *coro_ptr = bld.CreateLoad(ptr_t, coro_ptr_alloca, "coro_ptr");

        // Create the coroutine handle.
        auto *coro_hdl = detail::llvm_invoke_intrinsic(bld, "llvm.coro.begin", {}, {coro_id, coro_ptr});
        coro_hdl->addRetAttr(llvm::Attribute::NoAlias);
        coro_hdl->setName("coro_hdl");

        // Create the resume, cleanup and suspend blocks.
        auto *resume_bb = llvm::BasicBlock::Create(ctx);
        resume_bb->setName("coro_resume_block");
        auto *cleanup_bb = llvm::BasicBlock::Create(ctx);
        cleanup_bb->setName("coro_cleanup_block");
        auto *suspend_bb = llvm::BasicBlock::Create(ctx);
        suspend_bb->setName("coro_suspend_block");

        // Initialise the promise value.
        bld.CreateStore(bld.getInt32(0), coro_promise);

        // The main loop.
        detail::llvm_loop_u32(
            s, bld.getInt32(0), limit,
            [&bld, coro_promise, &ctx, resume_bb, suspend_bb, cleanup_bb, coro_f](llvm::Value *cur_n) {
                // Store cur_n + 1 into coro_promise.
                bld.CreateStore(bld.CreateAdd(cur_n, bld.getInt32(1)), coro_promise);

                // Suspend the coroutine.
                auto *coro_suspend = detail::llvm_invoke_intrinsic(
                    bld, "llvm.coro.suspend", {}, {llvm::ConstantTokenNone::get(ctx), bld.getInt1(false)});
                coro_suspend->setName("coro_suspend");

                // Create the suspend/resume/cleanup switch statement.
                auto *sw = bld.CreateSwitch(coro_suspend, suspend_bb);
                sw->addCase(bld.getInt8(0), resume_bb);
                sw->addCase(bld.getInt8(1), cleanup_bb);

                // Insert the resume block, and set it as insertion point, so that the tail of the
                // loop statement (which increases the loop variable and checks whether or not we are
                // at the end of the iteration) will be generated in it.
                detail::llvm_append_block(coro_f, resume_bb);
                bld.SetInsertPoint(resume_bb);
            });

        // After exiting the loop, we invoke the final suspend.
        auto *coro_fsuspend = detail::llvm_invoke_intrinsic(bld, "llvm.coro.suspend", {},
                                                            {llvm::ConstantTokenNone::get(ctx), bld.getInt1(true)});
        coro_fsuspend->setName("coro_fsuspend");

        // Create the suspend/cleanup switch statement.
        // NOTE: since this is a final suspend, no resume is possible.
        auto *sw = bld.CreateSwitch(coro_fsuspend, suspend_bb);
        sw->addCase(bld.getInt8(1), cleanup_bb);

        // Insert and codegen the cleanup block.
        detail::llvm_append_block(coro_f, cleanup_bb);
        bld.SetInsertPoint(cleanup_bb);

        // Invoke the llvm.coro.free() intrinsic to determine if we need to free dynamically-allocated memory.
        auto *coro_free_ptr = detail::llvm_invoke_intrinsic(bld, "llvm.coro.free", {}, {coro_id, coro_hdl});
        coro_free_ptr->setName("coro_free_ptr");
        detail::llvm_if_then_else(
            s, bld.CreateICmpEQ(coro_free_ptr, nullp), []() {},
            [&s, &bld, coro_free_ptr]() { detail::llvm_invoke_external(s, "free", bld.getVoidTy(), {coro_free_ptr}); });

        // Jump to the suspend block.
        bld.CreateBr(suspend_bb);

        // Insert and codegen the suspend block.
        detail::llvm_append_block(coro_f, suspend_bb);
        bld.SetInsertPoint(suspend_bb);

        // End the coroutine.
        detail::llvm_invoke_intrinsic(bld, "llvm.coro.end", {},
                                      {coro_hdl, bld.getInt1(false), llvm::ConstantTokenNone::get(ctx)});

        // Return the handle.
        bld.CreateRet(coro_hdl);

        // Setup the main function.
        auto *main_ft = llvm::FunctionType::get(i32_t, {i32_t}, false);
        auto *main_f = llvm::Function::Create(main_ft, llvm::Function::ExternalLinkage, "main", &md);

        // Fetch the input argument.
        auto *input_n = main_f->getArg(0);

        // Create a new basic block to start insertion into.
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", main_f));

        // Setup the variable that will contain the return value.
        auto *ret_alloca = bld.CreateAlloca(i32_t);
        bld.CreateStore(bld.getInt32(0), ret_alloca);

        // Setup the array of coroutine handles.
        const std::uint32_t n_coro = 100;
        auto *coro_hdl_arr_type = llvm::ArrayType::get(ptr_t, n_coro);
        auto *coro_hdl_arr_alloca = bld.CreateAlloca(coro_hdl_arr_type);

        // Initial invocation of the coroutines.
        for (std::uint32_t i = 0; i < n_coro; ++i) {
            auto *hdl_ptr
                = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca, {bld.getInt32(0), bld.getInt32(i)});
            auto *hdl = bld.CreateCall(coro_f, {input_n});
            bld.CreateStore(hdl, hdl_ptr);
        }

        // Fetch a handle to the first coroutine in the array.
        auto *first_hdl_ptr
            = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca, {bld.getInt32(0), bld.getInt32(0)});
        auto *first_hdl = bld.CreateLoad(ptr_t, first_hdl_ptr);

        // Iterate until the first coroutine is done.
        detail::llvm_while_loop(
            s,
            [&]() {
                auto *first_coro_done = detail::llvm_invoke_intrinsic(bld, "llvm.coro.done", {}, {first_hdl});
                return bld.CreateNot(first_coro_done);
            },
            [&]() {
                detail::llvm_loop_u32(s, bld.getInt32(0), bld.getInt32(n_coro), [&](llvm::Value *cur_i) {
                    auto *hdl_ptr
                        = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca, {bld.getInt32(0), cur_i});
                    auto *hdl = bld.CreateLoad(ptr_t, hdl_ptr);
                    detail::llvm_invoke_intrinsic(bld, "llvm.coro.resume", {}, {hdl});
                });

#if 0
                // Resume all coroutines.
                for (std::uint32_t i = 0; i < n_coro; ++i) {
                    auto *hdl_ptr = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca,
                                                          {bld.getInt32(0), bld.getInt32(i)});
                    auto *hdl = bld.CreateLoad(ptr_t, hdl_ptr);
                    detail::llvm_invoke_intrinsic(bld, "llvm.coro.resume", {}, {hdl});
                }
#endif
            });

#if 0
        // Setup the variable containing the current size of the couroutine array.
        auto *coro_hdl_arr_size_alloca = bld.CreateAlloca(i32_t);
        bld.CreateStore(bld.getInt32(n_coro), coro_hdl_arr_size_alloca);



        // Create two nested loops:
        // - the outer loop iterates as long as the number of active coroutines is nonzero,
        // - the inner loop scans the array of active coroutines and checks if they are done. If a coroutine is done,
        //   its handle is moved at the end of the array and the number of active coroutines is decreased by one.
        //   Otherwise, the coroutine is resumed.
        detail::llvm_while_loop(
            s,
            [&]() {
                auto *n_active_coro = bld.CreateLoad(i32_t, coro_hdl_arr_size_alloca);
                return bld.CreateICmpNE(n_active_coro, bld.getInt32(0));
            },
            [&]() {
                auto *cur_array_idx_alloca = bld.CreateAlloca(i32_t);
                bld.CreateStore(bld.getInt32(0), cur_array_idx_alloca);

                detail::llvm_while_loop(
                    s,
                    [&]() {
                        auto *cur_array_idx = bld.CreateLoad(i32_t, cur_array_idx_alloca);
                        auto *n_active_coro = bld.CreateLoad(i32_t, coro_hdl_arr_size_alloca);
                        return bld.CreateICmpNE(cur_array_idx, n_active_coro);
                    },
                    [&]() {
                        // Fetch the current array index.
                        auto *cur_array_idx = bld.CreateLoad(i32_t, cur_array_idx_alloca);

                        // Load the corresponding coroutine handle.
                        auto *hdl_ptr = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca,
                                                              {bld.getInt32(0), cur_array_idx});
                        auto *hdl = bld.CreateLoad(ptr_t, hdl_ptr);

                        // Check if the coroutine is done.
                        auto *coro_done = detail::llvm_invoke_intrinsic(bld, "llvm.coro.done", {}, {hdl});

                        detail::llvm_if_then_else(
                            s, coro_done,
                            [&]() {
                                // The coroutine is done. We need to:
                                // - swap its handle with the last possibly-active handle in the array,
                                // - decrease the number of active coroutines by one.
                                //
                                // NOTE: we do *not* bump cur_array_idx by 1 because we need to examine the coroutine we
                                // just swapped from the tail.

                                // Fetch the current number of active coroutines.
                                auto *n_active_coro = bld.CreateLoad(i32_t, coro_hdl_arr_size_alloca);

                                // Fetch the pointer to the last possibly-active handle in the array.
                                auto *last_hdl_ptr = bld.CreateInBoundsGEP(
                                    coro_hdl_arr_type, coro_hdl_arr_alloca,
                                    {bld.getInt32(0), bld.CreateSub(n_active_coro, bld.getInt32(1))});
                                // Load it.
                                auto *last_hdl = bld.CreateLoad(ptr_t, last_hdl_ptr);

                                // Store the current handle into the last slot.
                                bld.CreateStore(hdl, last_hdl_ptr);

                                // Store the last handle into the current slot.
                                bld.CreateStore(last_hdl, hdl_ptr);

                                // Decrease the number of active coroutines by one.
                                bld.CreateStore(bld.CreateSub(n_active_coro, bld.getInt32(1)),
                                                coro_hdl_arr_size_alloca);
                            },
                            [&]() {
                                // The coroutine is not done. We need to:
                                // - resume it,
                                // - bump by one cur_array_idx.

                                // Resume the coroutine.
                                detail::llvm_invoke_intrinsic(bld, "llvm.coro.resume", {}, {hdl});

                                // Bump cur_array_idx.
                                bld.CreateStore(bld.CreateAdd(cur_array_idx, bld.getInt32(1)), cur_array_idx_alloca);
                            });
                    });
            });

#endif

        // Accumulate the values from the promises and destroy the coroutines.
        detail::llvm_loop_u32(
            s, bld.getInt32(0), bld.getInt32(n_coro), [&](llvm::Value *cur_i) { // Fetch the coroutine handle.
                auto *hdl_ptr = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca, {bld.getInt32(0), cur_i});
                auto *hdl = bld.CreateLoad(ptr_t, hdl_ptr);

                // Load the return value from the promise.
                auto *promise_addr = detail::llvm_invoke_intrinsic(
                    bld, "llvm.coro.promise", {},
                    {hdl, bld.getInt32(boost::numeric_cast<std::uint32_t>(detail::get_alignment(md, i32_t))),
                     bld.getInt1(false)});
                auto *promise_val = bld.CreateLoad(i32_t, promise_addr);

                // Update the return value.
                auto *cur_ret = bld.CreateLoad(i32_t, ret_alloca);
                auto *new_ret = bld.CreateAdd(cur_ret, promise_val);
                bld.CreateStore(new_ret, ret_alloca);

                // Destroy the coroutine.
                detail::llvm_invoke_intrinsic(bld, "llvm.coro.destroy", {}, {hdl});
            });

#if 0
        for (std::uint32_t i = 0; i < n_coro; ++i) {
            // Fetch the coroutine handle.
            auto *hdl_ptr
                = bld.CreateInBoundsGEP(coro_hdl_arr_type, coro_hdl_arr_alloca, {bld.getInt32(0), bld.getInt32(i)});
            auto *hdl = bld.CreateLoad(ptr_t, hdl_ptr);

            // Load the return value from the promise.
            auto *promise_addr = detail::llvm_invoke_intrinsic(
                bld, "llvm.coro.promise", {},
                {hdl, bld.getInt32(boost::numeric_cast<std::uint32_t>(detail::get_alignment(md, i32_t))),
                 bld.getInt1(false)});
            auto *promise_val = bld.CreateLoad(i32_t, promise_addr);

            // Update the return value.
            auto *cur_ret = bld.CreateLoad(i32_t, ret_alloca);
            auto *new_ret = bld.CreateAdd(cur_ret, promise_val);
            bld.CreateStore(new_ret, ret_alloca);

            // Destroy the coroutine.
            detail::llvm_invoke_intrinsic(bld, "llvm.coro.destroy", {}, {hdl});
        }
#endif

#if 0
        // Initial invocation of the coroutine.
        auto *hdl = bld.CreateCall(coro_f, {input_n});

        detail::llvm_while_loop(
            s,
            [&bld, hdl]() -> llvm::Value * {
                auto *coro_done = detail::llvm_invoke_intrinsic(bld, "llvm.coro.done", {}, {hdl});
                return bld.CreateNot(coro_done);
            },
            [&bld, hdl]() { detail::llvm_invoke_intrinsic(bld, "llvm.coro.resume", {}, {hdl}); });

        // Load the return value from the promise.
        auto *promise_addr = detail::llvm_invoke_intrinsic(
            bld, "llvm.coro.promise", {},
            {hdl, bld.getInt32(boost::numeric_cast<std::uint32_t>(detail::get_alignment(md, i32_t))),
             bld.getInt1(false)});

        // Fetch the promise value.
        auto *promise_val = bld.CreateLoad(i32_t, promise_addr);

        // Destroy the coroutine.
        detail::llvm_invoke_intrinsic(bld, "llvm.coro.destroy", {}, {hdl});

        bld.CreateRet(promise_val);
#endif

        bld.CreateRet(bld.CreateLoad(i32_t, ret_alloca));

        s.compile();

        std::cout << s.get_ir() << '\n';

        auto *main_ptr = reinterpret_cast<std::uint32_t (*)(std::uint32_t) noexcept>(s.jit_lookup("main"));
        std::cout << main_ptr(0) << '\n';
        std::cout << main_ptr(1) << '\n';
        std::cout << main_ptr(2) << '\n';
    }
}
