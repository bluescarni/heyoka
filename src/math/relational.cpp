// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include <fmt/core.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/s11n.hpp>

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
