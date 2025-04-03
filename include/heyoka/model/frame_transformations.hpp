// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_FRAME_TRANSFORMATIONS_HPP
#define HEYOKA_MODEL_FRAME_TRANSFORMATIONS_HPP

#include <array>
#include <tuple>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/iau2006.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> rot_fk5j2000_icrs(const std::array<expression, 3> &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> rot_icrs_fk5j2000(const std::array<expression, 3> &);

namespace detail
{

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3>
rot_itrs_icrs_impl(const std::array<expression, 3> &, const expression &, double, const eop_data &);

// Common options for the itrs/icrs rotations.
template <typename... KwArgs>
auto itrs_icrs_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = [&p]() -> expression {
        if constexpr (p.has(kw::time_expr)) {
            return p(kw::time_expr);
        } else {
            return heyoka::time;
        }
    }();

    // Threshold value for the iau2006 theory.
    auto thresh = [&]() -> double {
        if constexpr (p.has(kw::thresh)) {
            return p(kw::thresh);
        } else {
            return iau2006_default_thresh;
        }
    }();

    // EOP data (defaults to def-cted).
    auto data = [&p]() -> eop_data {
        if constexpr (p.has(kw::eop_data)) {
            return p(kw::eop_data);
        } else {
            return eop_data{};
        }
    }();

    return std::tuple{std::move(time_expr), thresh, std::move(data)};
}

} // namespace detail

inline constexpr auto rot_itrs_icrs = []<typename... KwArgs>
    requires(!igor::has_unnamed_arguments<KwArgs...>())
(const std::array<expression, 3> &xyz, const KwArgs &...kw_args) {
    return std::apply(detail::rot_itrs_icrs_impl,
                      std::tuple_cat(std::make_tuple(xyz), detail::itrs_icrs_common_opts(kw_args...)));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
