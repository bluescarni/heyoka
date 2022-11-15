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
#include <ios>
#include <limits>
#include <locale>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

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

std::string null_constant_func::operator()([[maybe_unused]] unsigned prec) const
{
    assert(prec > 0u);

    return "0";
}

std::string pi_constant_func::operator()(unsigned prec) const
{
    assert(prec > 0u);

    // NOTE: we assume double is always ieee style.
    static_assert(std::numeric_limits<double>::is_iec559);
    if (prec <= static_cast<unsigned>(std::numeric_limits<double>::digits)) {
        return fmt::format("{:.{}}", boost::math::constants::pi<double>(), std::numeric_limits<double>::max_digits10);
    }

    // Try with long double.
    if (std::numeric_limits<long double>::is_iec559
        && prec <= static_cast<unsigned>(std::numeric_limits<long double>::digits)) {
        // NOTE: fmt support for long double is sketchy, let's go with iostreams.
        std::ostringstream oss;
        oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        oss.imbue(std::locale::classic());
        oss << std::showpoint;

        oss.precision(std::numeric_limits<long double>::max_digits10);
        oss << boost::math::constants::pi<long double>();

        return oss.str();
    }

#if defined(HEYOKA_HAVE_REAL128)

    if (prec <= static_cast<unsigned>(std::numeric_limits<mppp::real128>::digits)) {
        return mppp::pi_128.to_string();
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    return mppp::real_pi(boost::numeric_cast<mpfr_prec_t>(prec)).to_string();

#else

    throw std::invalid_argument(fmt::format("Unable to generate a pi constant with a precision of {} bits", prec));

#endif
}

namespace
{

// Regex to match floating-point numbers. See:
// https://www.regular-expressions.info/floatingpoint.html
// NOLINTNEXTLINE(cert-err58-cpp)
const std::regex fp_regex(R"(^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$)");

} // namespace

} // namespace detail

constant::constant() : constant("null_constant", detail::null_constant_func{}) {}

constant::constant(std::string name, str_func_t f, std::optional<std::string> repr)
    : func_base(std::move(name), {}), m_str_func(std::move(f)), m_repr(std::move(repr))
{
    if (!m_str_func) {
        throw std::invalid_argument("Cannot construct a constant with an empty string function");
    }
}

// Fetch the internal type of the string
// function. This is intended to help discriminating
// different constants.
std::type_index constant::get_str_func_t() const
{
    return m_str_func.get_type_index();
}

void constant::to_stream(std::ostream &os) const
{
    if (m_repr) {
        os << *m_repr;
    } else {
        os << get_name();
    }
}

std::vector<expression> constant::gradient() const
{
    assert(args().empty());
    return {};
}

// This is a thin wrapper around m_str_func that checks
// the input prec value and validates the returned string.
std::string constant::operator()(unsigned prec) const
{
    if (prec == 0u) {
        throw std::invalid_argument("Cannot generate a constant with a precision of zero bits");
    }

    auto ret = m_str_func(prec);

    // Validate the return value.
    if (!std::regex_match(ret, detail::fp_regex)) {
        throw std::invalid_argument(fmt::format("The string '{}' returned by the implementation of a constant is not a "
                                                "valid representation of a floating-point number in base 10",
                                                ret));
    }

    return ret;
}

// Helper to generate the LLVM version of the constant for the type tp.
// tp is supposed to be a scalar type.
llvm::Constant *constant::make_llvm_const([[maybe_unused]] llvm_state &s, llvm::Type *tp) const
{
    assert(tp != nullptr);
    assert(!tp->isVectorTy());

    // NOTE: isIEEE() is only available since LLVM 13.
    // For earlier versions of LLVM, we check that
    // tp is not a double-double, all the other available
    // FP types should be IEEE.
    if (tp->isFloatingPointTy() &&
#if LLVM_VERSION_MAJOR >= 13
        tp->isIEEE()
#else
        !tp->isPPC_FP128Ty()
#endif
    ) {
        // Fetch the FP semantics and precision.
        const auto &sem = tp->getFltSemantics();
        const auto prec = llvm::APFloatBase::semanticsPrecision(sem);

        // Fetch the string representation at the desired precision.
        const auto str_rep = operator()(prec);

        // Generate the APFloat and the constant.
        return llvm::ConstantFP::get(tp, llvm::APFloat(sem, str_rep));
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto real_prec = detail::llvm_is_real(tp)) {
        // Fetch the string representation at the desired precision.
        const auto str_rep = operator()(boost::numeric_cast<unsigned>(real_prec));

        // Go through llvm_codegen().
        return llvm::cast<llvm::Constant>(llvm_codegen(s, tp, number{mppp::real{str_rep, real_prec}}));
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot generate an LLVM constant of type '{}'", detail::llvm_type_name(tp)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *constant::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &, llvm::Value *,
                                 llvm::Value *, std::uint32_t batch_size, bool) const
{
    return detail::vector_splat(s.builder(), make_llvm_const(s, fp_t), batch_size);
}

llvm::Function *constant::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    return detail::llvm_c_eval_func_helper(
        get_name(),
        [&s, batch_size, fp_t, this](const std::vector<llvm::Value *> &, bool) {
            return detail::vector_splat(s.builder(), make_llvm_const(s, fp_t), batch_size);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

llvm::Value *constant::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &,
                                   const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                   std::uint32_t order, std::uint32_t, std::uint32_t batch_size, bool) const
{
    auto &builder = s.builder();

    // NOTE: no need for normalisation of the derivative,
    // as the only nonzero retval is for order 0
    // for which the normalised derivative coincides with
    // the non-normalised derivative.
    if (order == 0u) {
        return detail::vector_splat(builder, make_llvm_const(s, fp_t), batch_size);
    } else {
        return detail::vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

llvm::Function *constant::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = detail::make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = detail::taylor_c_diff_func_name_args(context, fp_t, fmt::format("constant_{}", get_name()),
                                                              n_uvars, batch_size, {});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // NOTE: no need for normalisation of the derivative,
        // as the only nonzero retval is for order 0
        // for which the normalised derivative coincides with
        // the non-normalised derivative.
        detail::llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, return the constant itself.
                builder.CreateStore(detail::vector_splat(builder, make_llvm_const(s, fp_t), batch_size), retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(detail::vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size),
                                    retval);
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
        if (!detail::compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of a constant in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

// NOLINTNEXTLINE(cert-err58-cpp)
const expression pi{func{constant{"pi", detail::pi_constant_func{}, u8"π"}}};

} // namespace heyoka

HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::detail::null_constant_func, std::string, unsigned)

HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::detail::pi_constant_func, std::string, unsigned)

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::constant)
