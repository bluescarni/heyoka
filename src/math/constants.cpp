// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <fmt/format.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

using max_fp_t =
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128
#else
    long double
#endif
    ;

} // namespace

constant_impl::constant_impl() : constant_impl("null_constant", number(max_fp_t(0))) {}

constant_impl::constant_impl(std::string name, number val) : func_base(std::move(name), {}), m_value(std::move(val))
{
    if (!std::holds_alternative<max_fp_t>(m_value.value())) {
        throw std::invalid_argument(
            "A constant can be initialised only from a floating-point value with the maximum precision");
    }
}

const number &constant_impl::get_value() const
{
    return m_value;
}

void constant_impl::to_stream(std::ostream &os) const
{
    os << get_name();
}

std::vector<expression> constant_impl::gradient() const
{
    assert(args().empty());
    return {};
}

llvm::Value *constant_impl::llvm_eval_dbl(llvm_state &s, const std::vector<llvm::Value *> &, llvm::Value *,
                                          llvm::Value *, std::uint32_t batch_size, bool) const
{
    return vector_splat(s.builder(), codegen<double>(s, get_value()), batch_size);
}

llvm::Value *constant_impl::llvm_eval_ldbl(llvm_state &s, const std::vector<llvm::Value *> &, llvm::Value *,
                                           llvm::Value *, std::uint32_t batch_size, bool) const
{
    return vector_splat(s.builder(), codegen<long double>(s, get_value()), batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *constant_impl::llvm_eval_f128(llvm_state &s, const std::vector<llvm::Value *> &, llvm::Value *,
                                           llvm::Value *, std::uint32_t batch_size, bool) const
{
    return vector_splat(s.builder(), codegen<mppp::real128>(s, get_value()), batch_size);
}

#endif

namespace
{

template <typename T>
[[nodiscard]] llvm::Function *constant_llvm_c_eval(llvm_state &s, const constant_impl &ci, std::uint32_t batch_size,
                                                   bool high_accuracy)
{
    return llvm_c_eval_func_helper<T>(
        ci.get_name(),
        [&s, &ci, batch_size](const std::vector<llvm::Value *> &, bool) {
            return vector_splat(s.builder(), codegen<T>(s, ci.get_value()), batch_size);
        },
        ci, s, batch_size, high_accuracy);
}

} // namespace

llvm::Function *constant_impl::llvm_c_eval_func_dbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return constant_llvm_c_eval<double>(s, *this, batch_size, high_accuracy);
}

llvm::Function *constant_impl::llvm_c_eval_func_ldbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return constant_llvm_c_eval<long double>(s, *this, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *constant_impl::llvm_c_eval_func_f128(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return constant_llvm_c_eval<mppp::real128>(s, *this, batch_size, high_accuracy);
}

#endif

namespace
{

template <typename T>
llvm::Value *constant_taylor_diff_impl(const constant_impl &c, llvm_state &s, std::uint32_t order,
                                       std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // NOTE: no need for normalisation of the derivative,
    // as the only nonzero retval is for order 0
    // for which the normalised derivative coincides with
    // the non-normalised derivative.
    if (order == 0u) {
        return vector_splat(builder, codegen<T>(s, c.get_value()), batch_size);
    } else {
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    }
}

} // namespace

llvm::Value *constant_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                            const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                            std::uint32_t, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                            bool) const
{
    return constant_taylor_diff_impl<double>(*this, s, order, batch_size);
}

llvm::Value *constant_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                             std::uint32_t, std::uint32_t order, std::uint32_t,
                                             std::uint32_t batch_size, bool) const
{
    return constant_taylor_diff_impl<long double>(*this, s, order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *constant_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &,
                                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                             std::uint32_t, std::uint32_t order, std::uint32_t,
                                             std::uint32_t batch_size, bool) const
{
    return constant_taylor_diff_impl<mppp::real128>(*this, s, order, batch_size);
}

#endif

namespace
{

template <typename T>
llvm::Function *taylor_c_diff_constant_impl(const constant_impl &c, llvm_state &s, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair
        = taylor_c_diff_func_name_args<T>(context, fmt::format("constant_{}", c.get_name()), n_uvars, batch_size, {});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // NOTE: no need for normalisation of the derivative,
        // as the only nonzero retval is for order 0
        // for which the normalised derivative coincides with
        // the non-normalised derivative.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, return the constant itself.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, c.get_value()), batch_size), retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of a constant in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace

llvm::Function *constant_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                      bool) const
{
    return taylor_c_diff_constant_impl<double>(*this, s, n_uvars, batch_size);
}

llvm::Function *constant_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                       bool) const
{
    return taylor_c_diff_constant_impl<long double>(*this, s, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *constant_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                       bool) const
{
    return taylor_c_diff_constant_impl<mppp::real128>(*this, s, n_uvars, batch_size);
}

#endif

pi_impl::pi_impl()
    : constant_impl("pi", number(
#if defined(HEYOKA_HAVE_REAL128)
                              mppp::pi_128
#else
                              boost::math::constants::pi<long double>()
#endif
                              ))
{
}

void pi_impl::to_stream(std::ostream &os) const
{
    os << u8"π";
}

} // namespace detail

const expression pi{func{detail::pi_impl{}}};

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::pi_impl)
