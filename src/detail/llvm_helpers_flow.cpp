// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Append bb to the list of blocks of the function f
void llvm_append_block(llvm::Function *f, llvm::BasicBlock *bb)
{
    f->insert(f->end(), bb);
}

} // namespace

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
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Pre-create loop and afterloop blocks. Note that these have just
    // been created, they have not been inserted yet in the IR.
    auto *loop_bb = llvm::BasicBlock::Create(context);
    auto *after_bb = llvm::BasicBlock::Create(context);

    // NOTE: we need a special case if the body of the loop is
    // never to be executed (that is, begin >= end).
    // In such a case, we will jump directly to after_bb.
    // NOTE: unsigned integral comparison.
    auto *skip_cond = builder.CreateICmp(llvm::CmpInst::ICMP_UGE, begin, end);
    builder.CreateCondBr(skip_cond, after_bb, loop_bb);

    // Get a reference to the current block for
    // later usage in the phi node.
    auto *preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    llvm_append_block(f, loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto *cur = builder.CreatePHI(builder.getInt32Ty(), 2);
    cur->addIncoming(begin, preheader_bb);

    // Execute the loop body and the post-body code.
    llvm::Value *next{};
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
    auto *end_cond = builder.CreateICmp(llvm::CmpInst::ICMP_ULT, next, end);

    // Get a reference to the current block for later use,
    // and insert the "after loop" block.
    auto *loop_end_bb = builder.GetInsertBlock();
    llvm_append_block(f, after_bb);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(end_cond, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    cur->addIncoming(next, loop_end_bb);
}

// Create a switch statement of the type:
//
// switch (val) {
//      default:
//          default_f();
//          break;
//      case_i:
//          case_i_f();
//          break;
// }
//
// where the pairs (case_i, case_i_f) are the elements in the 'cases' argument.
// val must be a 32-bit int.
void llvm_switch_u32(llvm_state &s, llvm::Value *val, const std::function<void()> &default_f,
                     const std::map<std::uint32_t, std::function<void()>> &cases)
{
    auto &context = s.context();
    auto &builder = s.builder();

    assert(val->getType() == builder.getInt32Ty());

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Prepare the blocks for the cases.
    std::deque<llvm::BasicBlock *> cases_blocks;
    for ([[maybe_unused]] const auto &cp : cases) {
        cases_blocks.push_back(llvm::BasicBlock::Create(context));
    }

    // Helper to clean up the uninserted blocks in case of exceptions.
    auto bb_cleanup = [&cases_blocks]() {
        for (auto *bb : cases_blocks) {
            bb->deleteValue();
        }
    };

    // Create and insert the default block.
    auto *default_bb = llvm::BasicBlock::Create(context, "", f);

    // Create but do not insert the merge block.
    auto *merge_bb = llvm::BasicBlock::Create(context);

    // Create the switch instruction.
    auto *sw_inst = builder.CreateSwitch(val, default_bb);

    // Emit the code for the default case.
    builder.SetInsertPoint(default_bb);
    try {
        default_f();
    } catch (...) {
        bb_cleanup();

        // NOTE: merge_bb has not been
        // inserted into any parent yet.
        merge_bb->deleteValue();

        throw;
    }

    // Jump to the merge block.
    builder.CreateBr(merge_bb);

    // Emit the cases blocks.
    for (const auto &[idx, case_f] : cases) {
        // Grab the block for the current case.
        auto *cur_bb = cases_blocks.front();

        // Insert it.
        llvm_append_block(f, cur_bb);
        builder.SetInsertPoint(cur_bb);

        // Pop it from cases_blocks, as now cur_bb is managed
        // by the builder and does not need cleanup in case
        // of exceptions any more.
        cases_blocks.pop_front();

        // Emit the code for the current case.
        try {
            case_f();
        } catch (...) {
            bb_cleanup();
            merge_bb->deleteValue();

            throw;
        }

        // Jump to the merge block.
        builder.CreateBr(merge_bb);

        // Add the case to the switch instruction.
        sw_inst->addCase(builder.getInt32(idx), cur_bb);
    }

    // Emit the merge block.
    llvm_append_block(f, merge_bb);
    builder.SetInsertPoint(merge_bb);
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
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Do a first evaluation of cond.
    // NOTE: if this throws, we have not created any block
    // yet, no need for manual cleanup.
    auto *cmp = cond();
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
    auto *preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    llvm_append_block(f, loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto *cur = builder.CreatePHI(builder.getInt1Ty(), 2);
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
    auto *loop_end_bb = builder.GetInsertBlock();
    llvm_append_block(f, after_bb);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(cmp, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    cur->addIncoming(cmp, loop_end_bb);
}

// Create an LLVM if statement in the form:
//
// if (cond) {
//   then_f();
// } else {
//   else_f();
// }
void llvm_if_then_else(llvm_state &s, llvm::Value *cond,
                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                       const std::function<void()> &then_f, const std::function<void()> &else_f)
{
    auto &context = s.context();
    auto &builder = s.builder();

    assert(cond->getType() == builder.getInt1Ty());

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto *f = builder.GetInsertBlock()->getParent();
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
    llvm_append_block(f, else_bb);
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
    llvm_append_block(f, merge_bb);
    builder.SetInsertPoint(merge_bb);
}

} // namespace detail

HEYOKA_END_NAMESPACE
