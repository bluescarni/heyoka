// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_COMPOSITE_FUNCTION_HPP
#define HEYOKA_DETAIL_COMPOSITE_FUNCTION_HPP

#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// A composite function wraps a function expression as a single function.
//
// For instance, the expression sin(x + y) + z, which originally consists of multiple functions nested into each other,
// is wrapped as a single function with input arguments [x, y, z].
//
// A unique (llvm) function name is assembled at runtime from the function names in the original expression. Composite
// functions support only llvm evaluation.
class HEYOKA_DLL_PUBLIC composite_function_impl : public func_base
{
    expression m_ex;

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    composite_function_impl();
    explicit composite_function_impl(const expression &ex);

    [[nodiscard]] llvm::Value *llvm_evaluate(llvm_state &, const std::vector<llvm::Value *> &, llvm::Type *,
                                             llvm::Value *, bool) const;
};

HEYOKA_DLL_PUBLIC expression composite_function(const expression &);

} // namespace detail

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::composite_function_impl)

#endif
