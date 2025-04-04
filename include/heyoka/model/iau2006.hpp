// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_IAU2006_HPP
#define HEYOKA_MODEL_IAU2006_HPP

#include <array>
#include <tuple>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/vsop2013.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

HEYOKA_DLL_PUBLIC std::array<expression, 3> iau2006_impl(const expression &, double);

// Default truncation threshold for the IAU2006 theory.
inline constexpr double iau2006_default_thresh = 1e-6;

} // namespace detail

template <typename... KwArgs>
std::array<expression, 3> iau2006(const KwArgs &...kw_args)
{
    return std::apply(detail::iau2006_impl, detail::vsop2013_common_opts(detail::iau2006_default_thresh, kw_args...));
}

} // namespace model

HEYOKA_END_NAMESPACE

#endif
