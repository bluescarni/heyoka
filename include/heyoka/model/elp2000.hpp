// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
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

template <typename... KwArgs>
std::vector<expression> elp2000_spherical(const KwArgs &...kw_args)
{
    // NOTE: we re-use detail::vsop2013_common_opts() here because the keyword options
    // for ELP2000 are the same.
    return std::apply(detail::elp2000_spherical_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

template <typename... KwArgs>
std::vector<expression> elp2000_cartesian(const KwArgs &...kw_args)
{
    return std::apply(detail::elp2000_cartesian_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

template <typename... KwArgs>
std::vector<expression> elp2000_cartesian_e2000(const KwArgs &...kw_args)
{
    return std::apply(detail::elp2000_cartesian_e2000_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

template <typename... KwArgs>
std::vector<expression> elp2000_cartesian_fk5(const KwArgs &...kw_args)
{
    return std::apply(detail::elp2000_cartesian_fk5_impl, detail::vsop2013_common_opts(1e-6, kw_args...));
}

HEYOKA_DLL_PUBLIC std::array<double, 2> get_elp2000_mus();

} // namespace model

HEYOKA_END_NAMESPACE

#endif
