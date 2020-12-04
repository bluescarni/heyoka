// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_NBODY_HPP
#define HEYOKA_NBODY_HPP

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(masses);
IGOR_MAKE_NAMED_ARGUMENT(Gconst);

} // namespace kw

namespace detail
{

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> make_nbody_sys_fixed_masses(std::uint32_t, number,
                                                                                             std::vector<number>);

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

HEYOKA_DLL_PUBLIC std::array<double, 6> random_elliptic_state(double, const std::array<std::pair<double, double>, 6> &,
                                                              unsigned = std::random_device{}());

HEYOKA_DLL_PUBLIC std::array<double, 6> cartesian_to_oe(double, const std::array<double, 6> &);

} // namespace heyoka

#endif
