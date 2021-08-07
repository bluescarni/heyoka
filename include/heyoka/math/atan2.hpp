// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_ATAN2_HPP
#define HEYOKA_MATH_ATAN2_HPP

#include <string>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>

namespace heyoka
{

namespace detail
{

class HEYOKA_DLL_PUBLIC atan2_impl : public func_base
{
public:
    atan2_impl();
    explicit atan2_impl(expression, expression);

    expression diff(const std::string &) const;
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression atan2(expression, expression);

} // namespace heyoka

#endif
