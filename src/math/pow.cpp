// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>

namespace heyoka
{

namespace detail
{

pow_impl::pow_impl(expression b, expression e) : func_base("pow", std::vector{std::move(b), std::move(e)}) {}

pow_impl::pow_impl() : pow_impl(1_dbl, 1_dbl) {}

namespace
{

// NOTE: we want to allow approximate implementations of pow()
// in the following cases:
// - exponent is an integral number n (in which case we want to allow
//   transformation in a sequence of multiplications),
// - exponent is a value of type n / 2, with n an odd integral value (in which case
//   we want to give the option of implementing pow() on top of sqrt()).
bool pow_allow_approx(const pow_impl &pi)
{
    return is_integral(pi.args()[1]) || is_odd_integral_half(pi.args()[1]);
}

} // namespace

llvm::Value *pow_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 2u);
    assert(args[0] != nullptr);
    assert(args[1] != nullptr);

    const auto allow_approx = pow_allow_approx(*this);

    // NOTE: we want to try the SLEEF route only if we are *not* approximating
    // pow() with sqrt() or iterated multiplications (in which case we are fine
    // with the LLVM builtin).
    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType()); !allow_approx && vec_t != nullptr) {
        if (const auto sfn = sleef_function_name(s.context(), "pow", vec_t->getElementType(),
                                                 boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
            !sfn.empty()) {
            return llvm_invoke_external(
                s, sfn, vec_t, args,
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
        }
    }

    auto ret = llvm_invoke_intrinsic(s, "llvm.pow", {args[0]->getType()}, args);

    if (allow_approx) {
        llvm::cast<llvm::CallInst>(ret)->setHasApproxFunc(true);
    }

    return ret;
}

llvm::Value *pow_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 2u);
    assert(args[0] != nullptr);
    assert(args[1] != nullptr);

    const auto allow_approx = pow_allow_approx(*this);

    auto ret = llvm_invoke_intrinsic(s, "llvm.pow", {args[0]->getType()}, args);

    if (allow_approx) {
        llvm::cast<llvm::CallInst>(ret)->setHasApproxFunc(true);
    }

    return ret;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *pow_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 2u);
    assert(args[0] != nullptr);
    assert(args[1] != nullptr);

    auto &builder = s.builder();

    // Decompose the arguments into scalars.
    auto scalars0 = vector_to_scalars(builder, args[0]);
    auto scalars1 = vector_to_scalars(builder, args[1]);

    // Invoke the function on the scalars.
    std::vector<llvm::Value *> retvals;
    for (decltype(scalars0.size()) i = 0; i < scalars0.size(); ++i) {
        retvals.push_back(llvm_invoke_external(
            s, "heyoka_pow128", scalars0[i]->getType(), {scalars0[i], scalars1[i]},
            // NOTE: in theory we may add ReadNone here as well,
            // but for some reason, at least up to LLVM 10,
            // this causes strange codegen issues. Revisit
            // in the future.
            {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, retvals);
}

#endif

double pow_impl::eval_dbl(const std::unordered_map<std::string, double> &map) const
{
    assert(args().size() == 2u);

    return std::pow(heyoka::eval_dbl(args()[0], map), heyoka::eval_dbl(args()[1], map));
}

void pow_impl::eval_batch_dbl(std::vector<double> &out,
                              const std::unordered_map<std::string, std::vector<double>> &map) const
{
    assert(args().size() == 2u);

    auto out0 = out; // is this allocation needed?
    heyoka::eval_batch_dbl(out0, args()[0], map);
    heyoka::eval_batch_dbl(out, args()[1], map);
    for (decltype(out.size()) i = 0; i < out.size(); ++i) {
        out[i] = std::pow(out0[i], out[i]);
    }
}

double pow_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 2u) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Inconsistent number of arguments when computing the numerical value of the "
            "exponentiation over doubles (2 arguments were expected, but {} arguments were provided"_format(a.size()));
    }

    return std::pow(a[0], a[1]);
}

double pow_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 2u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the exponentiation");
    }

    return a[1] * std::pow(a[0], a[1] - 1.) + std::log(a[0]) * std::pow(a[0], a[1]);
}

} // namespace detail

} // namespace heyoka
