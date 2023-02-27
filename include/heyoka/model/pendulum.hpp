// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_PENDULUM_HPP
#define HEYOKA_MODEL_PENDULUM_HPP

#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(gconst);
IGOR_MAKE_NAMED_ARGUMENT(l);

} // namespace kw

namespace model
{

namespace detail
{

template <typename... KwArgs>
auto pendulum_common_opts(KwArgs &&...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");
    static_assert(!p.has_other_than(kw::gconst, kw::l),
                  "This function accepts only the 'gconst' and 'l' named arguments.");

    // Gravitational constant (defaults to 1).
    auto gconst = [&p]() {
        if constexpr (p.has(kw::gconst)) {
            return expression{std::forward<decltype(p(kw::gconst))>(p(kw::gconst))};
        } else {
            return 1_dbl;
        }
    }();

    // Length (defaults to 1).
    auto l = [&p]() {
        if constexpr (p.has(kw::l)) {
            return expression{std::forward<decltype(p(kw::l))>(p(kw::l))};
        } else {
            return 1_dbl;
        }
    }();

    return std::tuple{std::move(gconst), std::move(l)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> pendulum_impl(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression pendulum_energy_impl(const expression &, const expression &);

} // namespace detail

template <typename... KwArgs>
std::vector<std::pair<expression, expression>> pendulum(KwArgs &&...kw_args)
{
    return std::apply(detail::pendulum_impl, detail::pendulum_common_opts(std::forward<KwArgs>(kw_args)...));
}

// NOTE: this returns the energy per unit of mass - the actual energy
// can be obtained by multiplying the result by the mass of the bob.
template <typename... KwArgs>
expression pendulum_energy(KwArgs &&...kw_args)
{
    return std::apply(detail::pendulum_energy_impl, detail::pendulum_common_opts(std::forward<KwArgs>(kw_args)...));
}

} // namespace model

HEYOKA_END_NAMESPACE

#endif
