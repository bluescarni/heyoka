// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

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

constant_impl::constant_impl(const constant_impl &) = default;

constant_impl::~constant_impl() = default;

const number &constant_impl::get_value() const
{
    return m_value;
}

void constant_impl::to_stream(std::ostream &os) const
{
    os << get_name();
}

expression constant_impl::diff(std::unordered_map<const void *, expression> &, const std::string &) const
{
    assert(args().empty());

    return 0_dbl;
}

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
                                            std::uint32_t, std::uint32_t order, std::uint32_t,
                                            std::uint32_t batch_size) const
{
    return constant_taylor_diff_impl<double>(*this, s, order, batch_size);
}

llvm::Value *constant_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                             std::uint32_t, std::uint32_t order, std::uint32_t,
                                             std::uint32_t batch_size) const
{
    return constant_taylor_diff_impl<long double>(*this, s, order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *constant_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &,
                                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                             std::uint32_t, std::uint32_t order, std::uint32_t,
                                             std::uint32_t batch_size) const
{
    return constant_taylor_diff_impl<mppp::real128>(*this, s, order, batch_size);
}

#endif

namespace
{

template <typename T>
llvm::Function *taylor_c_diff_constant_impl(const constant_impl &c, llvm_state &s, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Compose the function name.
    const auto fname = "heyoka_taylor_diff_constant_{}_{}"_format(c.get_name(), taylor_mangle_suffix(val_t));

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - par ptr,
    // - time ptr.
    std::vector<llvm::Type *> fargs{
        llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context), llvm::PointerType::getUnqual(val_t),
        llvm::PointerType::getUnqual(to_llvm_type<T>(context)), llvm::PointerType::getUnqual(to_llvm_type<T>(context))};

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
        builder.CreateRet(builder.CreateLoad(retval));

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

llvm::Function *constant_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t, std::uint32_t batch_size) const
{
    return taylor_c_diff_constant_impl<double>(*this, s, batch_size);
}

llvm::Function *constant_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t, std::uint32_t batch_size) const
{
    return taylor_c_diff_constant_impl<long double>(*this, s, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *constant_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t, std::uint32_t batch_size) const
{
    return taylor_c_diff_constant_impl<mppp::real128>(*this, s, batch_size);
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
