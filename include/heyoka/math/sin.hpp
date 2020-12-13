// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_SIN_HPP
#define HEYOKA_MATH_SIN_HPP

#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>

namespace heyoka
{

namespace detail
{

class HEYOKA_DLL_PUBLIC sin_impl : public func_base
{
public:
    sin_impl();
    explicit sin_impl(expression);

    llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const;
    llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const;
#endif
};

} // namespace detail

} // namespace heyoka

#endif
