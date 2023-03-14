// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_FIXED_CENTRES_HPP
#define HEYOKA_MODEL_FIXED_CENTRES_HPP

#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

template <typename... KwArgs>
auto fixed_centres_common_opts(KwArgs &&...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(),
                  "Unnamed arguments cannot be passed in the variadic pack to this function.");
    static_assert(!p.has_other_than(kw::Gconst, kw::masses),
                  "This function accepts only the 'Gconst' and 'masses' named arguments.");

    // G constant (defaults to 1).
    auto Gconst = [&p]() {
        if constexpr (p.has(kw::Gconst)) {
            return expression{std::forward<decltype(p(kw::Gconst))>(p(kw::Gconst))};
        } else {
            return 1_dbl;
        }
    }();

    // The vector of masses.
    std::vector<expression> masses_vec;
    if constexpr (p.has(kw::masses)) {
        for (const auto &mass_value : p(kw::masses)) {
            masses_vec.emplace_back(mass_value);
        }
    }

    if (masses_vec.empty()) {
        throw std::invalid_argument("In a fixed centres system the number of centres cannot be zero");
    }

    return std::tuple{std::move(Gconst), std::move(masses_vec)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> fixed_centres_impl(const expression &,
                                                                                    const std::vector<expression> &);

} // namespace detail

inline constexpr auto fixed_centres = [](auto &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::fixed_centres_impl,
                      detail::fixed_centres_common_opts(std::forward<decltype(kw_args)>(kw_args)...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
