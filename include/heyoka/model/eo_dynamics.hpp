// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_EO_DYNAMICS_HPP
#define HEYOKA_MODEL_EO_DYNAMICS_HPP

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/make_optional.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Default values for the eo_dynamics kwargs.
inline constexpr auto eo_dynamics_default_iau2006_thresh = 1e-3;

// Parse the kwargs for eo_dynamics.
template <typename... KwArgs>
auto eo_dynamics_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Parse max geopotential degree and order.
    const auto max_geo_degree = boost::numeric_cast<std::uint32_t>(p(kw::max_geo_degree, 0));
    const auto max_geo_order = boost::numeric_cast<std::uint32_t>(p(kw::max_geo_order, 0));

    // Parse iau2006_thresh.
    const auto iau2006_thresh = static_cast<double>(p(kw::iau2006_thresh, eo_dynamics_default_iau2006_thresh));

    // Parse the eop data.
    auto eop_data = p(kw::eop_data, heyoka::eop_data{});

    // Parse the sw data.
    auto sw_data = p(kw::sw_data, heyoka::sw_data{});

    // Parse the ballistic coefficient expression.
    auto Cb_opt = [&p]() -> std::optional<expression> {
        if constexpr (p.has(kw::Cb)) {
            return make_optional<expression>(p(kw::Cb));
        } else {
            return {};
        }
    }();

    // Parse the ELP2000 and VSOP2013 thresholds.
    const auto elp2000_thresh_opt = [&p]() -> std::optional<double> {
        if constexpr (p.has(kw::elp2000_thresh)) {
            return make_optional<double>(p(kw::elp2000_thresh));
        } else {
            return {};
        }
    }();

    const auto vsop2013_thresh_opt = [&p]() -> std::optional<double> {
        if constexpr (p.has(kw::vsop2013_thresh)) {
            return make_optional<double>(p(kw::vsop2013_thresh));
        } else {
            return {};
        }
    }();

    return std::make_tuple(max_geo_degree, max_geo_order, iau2006_thresh, std::move(eop_data), std::move(sw_data),
                           std::move(Cb_opt), elp2000_thresh_opt, vsop2013_thresh_opt);
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>>
eo_dynamics_impl(std::uint32_t, std::uint32_t, double, const eop_data &, const sw_data &,
                 const std::optional<expression> &, const std::optional<double> &, const std::optional<double> &);

} // namespace detail

// Kwargs configuration for eo_dynamics.
inline constexpr auto eo_dynamics_kw_cfg
    = igor::config<kw::descr::integral<kw::max_geo_degree>, kw::descr::integral<kw::max_geo_order>,
                   kw::descr::convertible_to<kw::iau2006_thresh, double>, kw::descr::same_as<kw::eop_data, eop_data>,
                   kw::descr::same_as<kw::sw_data, sw_data>, kw::descr::optional_from<expression, kw::Cb>,
                   kw::descr::optional_from<double, kw::elp2000_thresh>,
                   kw::descr::optional_from<double, kw::vsop2013_thresh>>{};

// Function to formulate the dynamics of an Earth-orbiting satellite.
//
// The dynamics is formulated in terms of the Cartesian state variables [x, y, z, vx, vy, vz] in the GCRS, with
// distances measured in km and time in seconds elapsed since the epoch of J2000.
//
// The precise formulation of the dynamics is controlled by several optional keyword arguments. By default (i.e., if no
// arguments are passed in input), purely Keplerian dynamics is returned, with the Earth's gravitational parameter taken
// from the egm2008 model.
//
// Currently the following configuration options are available:
//
// - eop/sw data: default to default-constructed instances;
// - max geopotential degree/order: default to m=n=0 (Keplerian);
// - iau2006 precession-nutation truncation threshold: defaults to eo_dynamics_default_iau2006_thresh;
// - expression for the ballistic coefficient Cb: defaults to not-provided, which disables atmospheric drag altogether.
//   Cb is expected in m**2/kg;
// - elp2000/vsop2013 truncation thresholds: must be either both provided or not-provided, default to not-provided
//   (which disables third-body perturbations altogether).
inline constexpr auto eo_dynamics = []<typename... KwArgs>
    requires igor::validate<eo_dynamics_kw_cfg, KwArgs...>
(KwArgs &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::eo_dynamics_impl, detail::eo_dynamics_opts(std::forward<KwArgs>(kw_args)...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
