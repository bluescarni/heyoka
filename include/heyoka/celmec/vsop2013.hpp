// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_CELMEC_VSOP2013_HPP
#define HEYOKA_CELMEC_VSOP2013_HPP

#include <cstdint>
#include <utility>

#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/time.hpp>

namespace heyoka
{

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(vsop2013_time);
IGOR_MAKE_NAMED_ARGUMENT(vsop2013_thresh);

} // namespace kw

namespace detail
{

// Common options for the vsop2013 functions.
template <typename... KwArgs>
inline auto vsop2013_common_opts(KwArgs &&...kw_args)
{
    igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = [&p]() -> expression {
        if constexpr (p.has(kw::vsop2013_time)) {
            return std::forward<decltype(p(kw::vsop2013_time))>(p(kw::vsop2013_time));
        } else {
            return heyoka::time;
        }
    }();

    // Threshold value (defaults to 1e-9).
    auto thresh = [&p]() -> double {
        if constexpr (p.has(kw::vsop2013_thresh)) {
            return std::forward<decltype(p(kw::vsop2013_thresh))>(p(kw::vsop2013_thresh));
        } else {
            return 1e-9;
        }
    }();

    return std::tuple{std::move(time_expr), thresh};
}

HEYOKA_DLL_PUBLIC expression vsop2013_elliptic_impl(std::uint32_t, std::uint32_t, expression, double);

} // namespace detail

template <typename... KwArgs>
inline expression vsop2013_elliptic(std::uint32_t pl_idx, std::uint32_t var_idx, KwArgs &&...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(std::forward<KwArgs>(kw_args)...);

    return detail::vsop2013_elliptic_impl(pl_idx, var_idx, std::move(time_expr), thresh);
}

} // namespace heyoka

#endif
