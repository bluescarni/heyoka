// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <concepts>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

std::string name_from_op(rel_op op)
{
    assert(op >= rel_op::eq);
    assert(op <= rel_op::gte);

    constexpr auto fstr = "rel_{}";

#define HEYOKA_MATH_REL_HANDLE_CASE(op)                                                                                \
    case rel_op::op:                                                                                                   \
        return fmt::format(fstr, #op);

    switch (op) {
        HEYOKA_MATH_REL_HANDLE_CASE(eq)
        HEYOKA_MATH_REL_HANDLE_CASE(neq)
        HEYOKA_MATH_REL_HANDLE_CASE(lt)
        HEYOKA_MATH_REL_HANDLE_CASE(gt)
        HEYOKA_MATH_REL_HANDLE_CASE(lte)
        HEYOKA_MATH_REL_HANDLE_CASE(gte)
    }

#undef HEYOKA_MATH_REL_HANDLE_CASE

    // LCOV_EXCL_START
    assert(false);

    throw;
    // LCOV_EXCL_STOP
}

} // namespace

rel_impl::rel_impl() : rel_impl(rel_op::eq, 1_dbl, 1_dbl) {}

rel_impl::rel_impl(rel_op op, expression a, expression b)
    : func_base(name_from_op(op), {std::move(a), std::move(b)}), m_op(op)
{
}

rel_op rel_impl::get_op() const noexcept
{
    return m_op;
}

void rel_impl::to_stream(std::ostringstream &oss) const
{
    assert(args().size() == 2u);

    const auto &a = args()[0];
    const auto &b = args()[1];

    oss << '(';
    stream_expression(oss, a);

    switch (m_op) {
        case rel_op::eq:
            oss << " == ";
            break;
        case rel_op::neq:
            oss << " != ";
            break;
        case rel_op::lt:
            oss << " < ";
            break;
        case rel_op::gt:
            oss << " > ";
            break;
        case rel_op::lte:
            oss << " <= ";
            break;
        default:
            assert(m_op == rel_op::gte);
            oss << " >= ";
            break;
    }

    stream_expression(oss, b);
    oss << ')';
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::vector<expression> rel_impl::gradient() const
{
    return {0_dbl, 0_dbl};
}

namespace
{

llvm::Value *rel_eval_impl(llvm_state &s, rel_op op, const std::vector<llvm::Value *> &args)
{
    assert(args.size() == 2u);

    llvm::Value *ret = nullptr;

    switch (op) {
        case rel_op::eq:
            ret = llvm_fcmp_oeq(s, args[0], args[1]);
            break;
        case rel_op::neq:
            ret = llvm_fcmp_one(s, args[0], args[1]);
            break;
        case rel_op::lt:
            ret = llvm_fcmp_olt(s, args[0], args[1]);
            break;
        case rel_op::gt:
            ret = llvm_fcmp_ogt(s, args[0], args[1]);
            break;
        case rel_op::lte:
            ret = llvm_fcmp_ole(s, args[0], args[1]);
            break;
        case rel_op::gte:
            ret = llvm_fcmp_oge(s, args[0], args[1]);
            break;
    }

    assert(ret != nullptr);

    // NOTE: the LLVM fp comparison primitives return booleans. Thus,
    // we need to convert back to the proper (vector) fp type on exit.
    // NOTE: we create a UI to FP conversion (rather than SI to FP)
    // so that we get either 1 or 0 from the conversion (with SI, true
    // would come out as -1).
    return llvm_ui_to_fp(s, ret, args[0]->getType());
}

} // namespace

llvm::Value *rel_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    return llvm_eval_helper(
        [&s, op = m_op](const std::vector<llvm::Value *> &args, bool) { return rel_eval_impl(s, op, args); }, *this, s,
        fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *rel_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    return llvm_c_eval_func_helper(
        name_from_op(m_op),
        [&s, op = m_op](const std::vector<llvm::Value *> &args, bool) { return rel_eval_impl(s, op, args); }, *this, s,
        fp_t, batch_size, high_accuracy);
}

llvm::Value *rel_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                   bool) const
{
    assert(args().size() == 2u);
    assert(deps.empty());

    // NOTE: we need to do something only at differentiation order 0.
    if (order == 0u) {
        std::vector<llvm::Value *> tmp;
        tmp.reserve(2);

        for (const auto &cur_arg : args()) {
            tmp.push_back(std::visit(
                [&]<typename T>(const T &v) -> llvm::Value * {
                    if constexpr (std::same_as<T, variable>) {
                        // Variable.
                        return taylor_fetch_diff(arr, uname_to_index(v.name()), 0, n_uvars);
                    } else if constexpr (is_num_param_v<T>) {
                        // Number/param.
                        return taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size);
                    } else {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            "An invalid argument type was encountered while trying to build the "
                            "Taylor derivative of a relational operation");
                        // LCOV_EXCL_STOP
                    }
                },
                cur_arg.value()));
        }

        return rel_eval_impl(s, m_op, tmp);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

llvm::Function *rel_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    assert(args().size() == 2u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Build the vector of arguments needed to determine the function name.
    std::vector<std::variant<variable, number, param>> nm_args;
    nm_args.reserve(static_cast<decltype(nm_args.size())>(args().size()));
    for (const auto &arg : args()) {
        nm_args.push_back(std::visit(
            []<typename T>(const T &v) -> std::variant<variable, number, param> {
                if constexpr (std::same_as<T, func>) {
                    // LCOV_EXCL_START
                    assert(false);
                    throw;
                    // LCOV_EXCL_STOP
                } else {
                    return v;
                }
            },
            arg.value()));
    }

    // Fetch the function name and arguments.
    const auto [fname, fargs]
        = taylor_c_diff_func_name_args(context, fp_t, name_from_op(m_op), n_uvars, batch_size, nm_args);

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f != nullptr) {
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The return type is val_t.
    auto *ft = llvm::FunctionType::get(val_t, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *order = f->args().begin();
    auto *diff_arr = f->args().begin() + 2;
    auto *par_ptr = f->args().begin() + 3;
    auto *operands = f->args().begin() + 5;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Create the return value.
    auto *retval = builder.CreateAlloca(val_t);

    llvm_if_then_else(
        s, builder.CreateICmpEQ(order, builder.getInt32(0)),
        [&]() {
            // For order zero, evaluate the relational operation.
            std::vector<llvm::Value *> vals;
            vals.reserve(2);

            for (decltype(args().size()) i = 0; i < args().size(); ++i) {
                vals.push_back(std::visit(
                    [&]<typename T>(const T &v) -> llvm::Value * {
                        if constexpr (std::same_as<T, variable>) {
                            return taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, operands + i);
                        } else if constexpr (is_num_param_v<T>) {
                            return taylor_c_diff_numparam_codegen(s, fp_t, v, operands + i, par_ptr, batch_size);
                        } else {
                            // LCOV_EXCL_START
                            throw std::invalid_argument(
                                "An invalid argument type was encountered while trying to build the "
                                "Taylor derivative of a relational operation");
                            // LCOV_EXCL_STOP
                        }
                    },
                    args()[i].value()));
            }

            builder.CreateStore(rel_eval_impl(s, m_op, vals), retval);
        },
        [&]() {
            // Otherwise, return zero.
            builder.CreateStore(llvm_constantfp(s, val_t, 0.), retval);
        });

    builder.CreateRet(builder.CreateLoad(val_t, retval));

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

} // namespace detail

#define HEYOKA_MATH_REL_IMPL(op)                                                                                       \
    expression op(expression a, expression b)                                                                          \
    {                                                                                                                  \
        return expression{func{detail::rel_impl{detail::rel_op::op, std::move(a), std::move(b)}}};                     \
    }

HEYOKA_MATH_REL_IMPL(eq)
HEYOKA_MATH_REL_IMPL(neq)
HEYOKA_MATH_REL_IMPL(lt)
HEYOKA_MATH_REL_IMPL(gt)
HEYOKA_MATH_REL_IMPL(lte)
HEYOKA_MATH_REL_IMPL(gte)

#undef HEYOKA_MATH_REL_IMPL

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::rel_impl)
