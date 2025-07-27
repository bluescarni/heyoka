// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_ELP2000_HPP
#define HEYOKA_MODEL_ELP2000_HPP

#include <array>
#include <tuple>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/vsop2013.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

HEYOKA_DLL_PUBLIC std::vector<expression> elp2000_spherical_impl(const expression &, double);
HEYOKA_DLL_PUBLIC std::vector<expression> elp2000_cartesian_impl(const expression &, double);
HEYOKA_DLL_PUBLIC std::vector<expression> elp2000_cartesian_e2000_impl(const expression &, double);
HEYOKA_DLL_PUBLIC std::vector<expression> elp2000_cartesian_fk5_impl(const expression &, double);

} // namespace detail

// NOTE: the elp2000 and vsop2013 theories have the same kwargs config.
inline constexpr auto elp2000_kw_cfg = vsop2013_kw_cfg;

template <typename... KwArgs>
    requires igor::validate<elp2000_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::vector<expression> elp2000_spherical(KwArgs &&...kw_args)
{
    // NOTE: we re-use detail::vsop2013_common_opts() here because the keyword options
    // for ELP2000 are the same.
    return std::apply(detail::elp2000_spherical_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

template <typename... KwArgs>
    requires igor::validate<elp2000_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::vector<expression> elp2000_cartesian(KwArgs &&...kw_args)
{
    return std::apply(detail::elp2000_cartesian_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

template <typename... KwArgs>
    requires igor::validate<elp2000_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::vector<expression> elp2000_cartesian_e2000(KwArgs &&...kw_args)
{
    return std::apply(detail::elp2000_cartesian_e2000_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

template <typename... KwArgs>
    requires igor::validate<elp2000_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::vector<expression> elp2000_cartesian_fk5(KwArgs &&...kw_args)
{
    return std::apply(detail::elp2000_cartesian_fk5_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

HEYOKA_DLL_PUBLIC std::array<double, 2> get_elp2000_mus();

} // namespace model

HEYOKA_END_NAMESPACE

#endif
