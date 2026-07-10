// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_SW_HPP
#define HEYOKA_MODEL_SW_HPP

#include <tuple>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_impl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/sw_data.hpp>

// NOTE: for the representation of SW data, we adopt a piecewise linear approximation where the switch points are given
// by the dates in the SW dataset. Within each time interval, an SW quantity is approximated as SW(t) = c0 + c1*t (where
// the values of the c0 and c1 constants change from interval to interval). In the expression system, we implement, for
// each SW quantity, two unary functions which return respectively the SW quantity and its first-order derivative at the
// given input time.

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

template <typename... KwArgs>
auto sw_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    // SW data (defaults to def-cted).
    auto data = [&p]() {
        if constexpr (p.has(kw::sw_data)) {
            return p(kw::sw_data);
        } else {
            return sw_data{};
        }
    }();

    return std::tuple{std::move(time_expr), std::move(data)};
}

} // namespace detail

inline constexpr auto sw_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::time_expr>,
                                               kw::descr::same_as<kw::sw_data, sw_data>>{};

} // namespace model

HEYOKA_END_NAMESPACE

// NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
HEYOKA_MODEL_DECLARE_EOP_SW(Ap_avg, sw_data, sw_kw_cfg, sw_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(f107, sw_data, sw_kw_cfg, sw_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(f107a_center81, sw_data, sw_kw_cfg, sw_common_opts);
// NOLINTEND(cppcoreguidelines-missing-std-forward)

#endif
