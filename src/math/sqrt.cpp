// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

sqrt_impl::sqrt_impl(expression e) : func_base("sqrt", std::vector{std::move(e)}) {}

sqrt_impl::sqrt_impl() : sqrt_impl(0_dbl) {}

llvm::Value *sqrt_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    // NOTE: no need for sleef here, SIMD instructions
    // sets usually have direct support for sqrt (but perhaps
    // double check in the future about non-x86).
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return llvm_invoke_intrinsic(s, "llvm.sqrt", {args[0]->getType()}, args);
}

llvm::Value *sqrt_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    // NOTE: codegen is identical as in dbl.
    return codegen_dbl(s, args);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *sqrt_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    auto &builder = s.builder();

    // Decompose the argument into scalars.
    auto scalars = vector_to_scalars(builder, args[0]);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> retvals;
    for (auto scal : scalars) {
        retvals.push_back(llvm_invoke_external(
            s, "heyoka_sqrt128", scal->getType(), {scal},
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

double sqrt_impl::eval_dbl(const std::unordered_map<std::string, double> &map) const
{
    assert(args().size() == 1u);

    return std::sqrt(heyoka::eval_dbl(args()[0], map));
}

void sqrt_impl::eval_batch_dbl(std::vector<double> &out,
                               const std::unordered_map<std::string, std::vector<double>> &map) const
{
    assert(args().size() == 1u);

    heyoka::eval_batch_dbl(out, args()[0], map);
    for (auto &el : out) {
        el = std::sqrt(el);
    }
}

double sqrt_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 1u) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Inconsistent number of arguments when computing the numerical value of the "
            "square root over doubles (1 argument was expected, but {} arguments were provided"_format(a.size()));
    }

    return std::sqrt(a[0]);
}

double sqrt_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 1u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the square root");
    }

    return 1. / (2. * std::sqrt(a[0]));
}

namespace
{

// Derivative of sqrt(number).
template <typename T>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, const number &, const std::vector<llvm::Value *> &, std::uint32_t,
                                   std::uint32_t, std::uint32_t, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of sqrt(variable).
template <typename T>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &s, const variable &var, const std::vector<llvm::Value *> &arr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size)
{
    return taylor_diff_pow_impl_det<T>(s, var, number{T(1) / 2}, arr, n_uvars, order, idx, batch_size);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_sqrt_impl(llvm_state &, const U &, const std::vector<llvm::Value *> &, std::uint32_t,
                                   std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a square root");
}

template <typename T>
llvm::Value *taylor_diff_sqrt(llvm_state &s, const sqrt_impl &f, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_diff_sqrt_impl<T>(s, v, arr, n_uvars, order, idx, batch_size); },
        f.args()[0].value());
}

} // namespace

llvm::Value *sqrt_impl::taylor_diff_dbl(llvm_state &s, const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                        std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size) const
{
    return taylor_diff_sqrt<double>(s, *this, arr, n_uvars, order, idx, batch_size);
}

llvm::Value *sqrt_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                         std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size) const
{
    return taylor_diff_sqrt<long double>(s, *this, arr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *sqrt_impl::taylor_diff_f128(llvm_state &s, const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                         std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size) const
{
    return taylor_diff_sqrt<mppp::real128>(s, *this, arr, n_uvars, order, idx, batch_size);
}

#endif

} // namespace detail

} // namespace heyoka
