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
