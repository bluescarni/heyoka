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
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/variable.hpp>

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

kepE_impl::kepE_impl() : kepE_impl(0_dbl, 0_dbl) {}

kepE_impl::kepE_impl(expression e, expression M) : func_base("kepE", std::vector{std::move(e), std::move(M)}) {}

expression kepE_impl::diff(const std::string &s) const
{
    assert(args().size() == 2u);

    const auto &e = args()[0];
    const auto &M = args()[1];

    expression E{func{*this}};

    return (heyoka::diff(e, s) * sin(E) + heyoka::diff(M, s)) / (1_dbl - e * cos(E));
}

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
kepE_impl::taylor_decompose(std::vector<std::pair<expression, std::vector<std::uint32_t>>> &u_vars_defs) &&
{
    assert(args().size() == 2u);

    // Decompose the arguments.
    auto &e = *get_mutable_args_it().first;
    if (const auto dres = taylor_decompose_in_place(std::move(e), u_vars_defs)) {
        e = expression{variable{"u_{}"_format(dres)}};
    }

    auto &M = *(get_mutable_args_it().first + 1);
    if (const auto dres = taylor_decompose_in_place(std::move(M), u_vars_defs)) {
        M = expression{variable{"u_{}"_format(dres)}};
    }

    // Make a copy of e.
    auto e_copy = e;

    // Append the kepE decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Append the sin(a)/cos(a) decompositions.
    u_vars_defs.emplace_back(sin(expression{variable{"u_{}"_format(u_vars_defs.size() - 1u)}}),
                             std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(cos(expression{variable{"u_{}"_format(u_vars_defs.size() - 2u)}}),
                             std::vector<std::uint32_t>{});

    // Append the e*cos(a) decomposition.
    u_vars_defs.emplace_back(std::move(e_copy) * expression{variable{"u_{}"_format(u_vars_defs.size() - 1u)}},
                             std::vector<std::uint32_t>{});

    // Add the hidden deps.
    // NOTE: hidden deps on sin(a) and e*cos(a).
    (u_vars_defs.end() - 4)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 3u));
    (u_vars_defs.end() - 4)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));

    // sin/cos hidden deps.
    (u_vars_defs.end() - 3)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));
    (u_vars_defs.end() - 2)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 3u));

    return u_vars_defs.size() - 4u;
}

namespace
{

template <typename T>
auto kepE_codegen_impl(llvm_state &s, const std::vector<llvm::Value *> &args)
{
    assert(args.size() == 2u);
    assert(args[0] != nullptr);
    assert(args[1] != nullptr);

    auto &builder = s.builder();

    // Determine whether we are operating on scalars or vectors.
    std::uint32_t batch_size = 1;
    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
    }

    // Create/fetch the inverse Kepler function solver.
    auto kepE_func = llvm_add_inv_kep_E<T>(s, batch_size);

    // Call it.
    return builder.CreateCall(kepE_func, {args[0], args[1]});
}

} // namespace

llvm::Value *kepE_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    return kepE_codegen_impl<double>(s, args);
}

llvm::Value *kepE_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    return kepE_codegen_impl<long double>(s, args);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *kepE_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    return kepE_codegen_impl<mppp::real128>(s, args);
}

#endif

} // namespace detail

expression kepE(expression e, expression M)
{
    return expression{func{detail::kepE_impl{std::move(e), std::move(M)}}};
}

} // namespace heyoka
