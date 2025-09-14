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
#include <functional>
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

// NOTE: for the future, we may be interested in implementing more frame rotations. The TEME,
// for instance, can be implemented as a rotation of the PEF, which itself is a couple of rotations
// away from the ITRF and very close to the TIRS (see the diagram on the Vallado book).
//
// We need however a way to progressively construct complex rotations by stacking together elementary
// rotations, similarly to what AstroPy and GODOT do, because otherwise we have a combinatorial
// explosion of possibilities. The general idea would be to implement step-by-step rotations (e.g.,
// by re-using the intermediate frame rotations we already have in the detail namespace) and then either allow
// the user to define a sequence of rotations manually, or perhaps even introduce a class to represent
// a "graph of rotations", which would allow to, e.g., select the "quickest" way (i.e., lowest number
// of elementary rotations) to go from one frame to another. Not clear yet on how this is to be implemented...

HEYOKA_BEGIN_NAMESPACE

namespace model
{

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> rot_fk5j2000_icrs(const std::array<expression, 3> &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> rot_icrs_fk5j2000(const std::array<expression, 3> &);

namespace detail
{

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3>
rot_itrs_icrs_impl(const std::array<expression, 3> &, const expression &, double, const eop_data &);

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3>
rot_icrs_itrs_impl(const std::array<expression, 3> &, const expression &, double, const eop_data &);

// Common options for the itrs/icrs rotations.
template <typename... KwArgs>
auto itrs_icrs_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    // Threshold value for the iau2006 theory.
    const auto thresh = static_cast<double>(p(kw::thresh, iau2006_default_thresh));

    // EOP data (defaults to def-cted).
    auto data = [&p]() {
        if constexpr (p.has(kw::eop_data)) {
            return p(kw::eop_data);
        } else {
            return eop_data{};
        }
    }();

    return std::tuple{std::move(time_expr), thresh, std::move(data)};
}

} // namespace detail

inline constexpr auto rot_itrs_icrs_kw_cfg
    = igor::config<kw::descr::constructible_from<expression, kw::time_expr>,
                   kw::descr::convertible_to<kw::thresh, double>, kw::descr::same_as<kw::eop_data, eop_data>>{};

inline constexpr auto rot_itrs_icrs = []<typename... KwArgs>
    requires igor::validate<rot_itrs_icrs_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    return std::apply(detail::rot_itrs_icrs_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::itrs_icrs_common_opts(kw_args...)));
};

inline constexpr auto rot_icrs_itrs = []<typename... KwArgs>
    requires igor::validate<rot_itrs_icrs_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    return std::apply(detail::rot_icrs_itrs_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::itrs_icrs_common_opts(kw_args...)));
};

namespace detail
{

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> rot_itrs_teme_impl(const std::array<expression, 3> &,
                                                                             const expression &, const eop_data &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> rot_teme_itrs_impl(const std::array<expression, 3> &,
                                                                             const expression &, const eop_data &);

// Common options for the itrs/teme rotations.
template <typename... KwArgs>
auto itrs_teme_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    // EOP data (defaults to def-cted).
    auto data = [&p]() {
        if constexpr (p.has(kw::eop_data)) {
            return p(kw::eop_data);
        } else {
            return eop_data{};
        }
    }();

    return std::tuple{std::move(time_expr), std::move(data)};
}

} // namespace detail

inline constexpr auto rot_itrs_teme_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::time_expr>,
                                                          kw::descr::same_as<kw::eop_data, eop_data>>{};

inline constexpr auto rot_itrs_teme = []<typename... KwArgs>
    requires igor::validate<rot_itrs_teme_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    return std::apply(detail::rot_itrs_teme_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::itrs_teme_common_opts(kw_args...)));
};

inline constexpr auto rot_teme_itrs = []<typename... KwArgs>
    requires igor::validate<rot_itrs_teme_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    return std::apply(detail::rot_teme_itrs_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::itrs_teme_common_opts(kw_args...)));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
