// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_NBODY_HPP
#define HEYOKA_NBODY_HPP

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

namespace detail
{

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> make_nbody_sys_fixed_masses(std::uint32_t, number,
                                                                                             std::vector<number>);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> make_nbody_sys_par_masses(std::uint32_t, number,
                                                                                           std::uint32_t);

} // namespace detail

// Create an ODE system representing a Newtonian N-body problem.
// n is the number of bodies (>= 2), while the following optional kwargs
// can be passed:
//
// - 'masses', which contains the numerical values of the masses,
// - 'Gconst', which contains the numerical value of the gravitational constant.
//
// 'Gconst' defaults to a value of 1 if not specified.
// If 'masses' is not specified, all masses have a
// constant numerical value of 1.
//
// The returned system consists of the differential equations for the state of each body,
// in the following order:
// x_0' = ...
// y_0' = ...
// z_0' = ...
// vx_0' = ...
// vy_0' = ...
// vz_0' = ...
// x_1' = ...
// y_1' = ...
// etc.
template <typename... KwArgs>
inline std::vector<std::pair<expression, expression>> make_nbody_sys(std::uint32_t n, KwArgs &&...kw_args)
{
    if (n < 2u) {
        throw std::invalid_argument("At least 2 bodies are needed to construct an N-body system");
    }

    igor::parser p{kw_args...};

    if constexpr (p.has_unnamed_arguments()) {
        static_assert(detail::always_false_v<KwArgs...>,
                      "The variadic arguments in the construction of an N-body system contain "
                      "unnamed arguments.");
    } else {
        // G constant (defaults to 1).
        auto G_const = [&p]() {
            if constexpr (p.has(kw::Gconst)) {
                return number{std::forward<decltype(p(kw::Gconst))>(p(kw::Gconst))};
            } else {
                return number{1.};
            }
        }();

        std::vector<number> masses_vec;

        if constexpr (p.has(kw::masses)) {
            for (const auto &mass_value : p(kw::masses)) {
                masses_vec.emplace_back(mass_value);
            }
        } else {
            // If no masses are provided, fix all masses to 1.
            masses_vec.resize(static_cast<decltype(masses_vec.size())>(n), number{1.});
        }

        return detail::make_nbody_sys_fixed_masses(n, std::move(G_const), std::move(masses_vec));
    }
}

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(n_massive);

}

template <typename... KwArgs>
inline std::vector<std::pair<expression, expression>> make_nbody_par_sys(std::uint32_t n, KwArgs &&...kw_args)
{
    if (n < 2u) {
        throw std::invalid_argument("At least 2 bodies are needed to construct an N-body system");
    }

    igor::parser p{kw_args...};

    if constexpr (p.has_unnamed_arguments()) {
        static_assert(detail::always_false_v<KwArgs...>,
                      "The variadic arguments in the construction of an N-body system contain "
                      "unnamed arguments.");
    } else {
        // G constant (defaults to 1).
        auto G_const = [&p]() {
            if constexpr (p.has(kw::Gconst)) {
                return number{std::forward<decltype(p(kw::Gconst))>(p(kw::Gconst))};
            } else {
                return number{1.};
            }
        }();

        if constexpr (p.has(kw::n_massive)) {
            if constexpr (std::is_integral_v<detail::uncvref_t<decltype(p(kw::n_massive))>>) {
                return detail::make_nbody_sys_par_masses(n, std::move(G_const),
                                                         boost::numeric_cast<std::uint32_t>(p(kw::n_massive)));
            } else {
                static_assert(detail::always_false_v<KwArgs...>,
                              "The n_massive keyword argument must be of integral type.");
            }
        } else {
            return detail::make_nbody_sys_par_masses(n, std::move(G_const), n);
        }
    }
}

} // namespace heyoka

#endif
