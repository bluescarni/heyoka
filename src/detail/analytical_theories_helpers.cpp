// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/analytical_theories_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Complex multiplication.
std::array<expression, 2> ex_cmul(const std::array<expression, 2> &c1, const std::array<expression, 2> &c2)
{
    const auto &[a, b] = c1;
    const auto &[c, d] = c2;

    return {a * c - b * d, b * c + a * d};
}

// Complex inversion.
std::array<expression, 2> ex_cinv(const std::array<expression, 2> &c)
{
    const auto &[a, b] = c;

    const auto den = pow(a, 2_dbl) + pow(b, 2_dbl);

    return {a / den, -b / den};
}

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
std::array<expression, 2> ccpow_impl(pow_dict_t &pd, const std::array<expression, 2> &pow1,
                                     const std::array<expression, 2> &powm1, std::int8_t n)
{
    auto it = pd.find(n);

    if (it != pd.end()) {
        return it->second;
    }

    std::array<expression, 2> ret;
    if (n >= 0) {
        ret = ex_cmul(pow1, ccpow_impl(pd, pow1, powm1, static_cast<std::int8_t>(n - 1)));
    } else {
        ret = ex_cmul(powm1, ccpow_impl(pd, pow1, powm1, static_cast<std::int8_t>(n + 1)));
    }

    [[maybe_unused]] auto [_, flag] = pd.try_emplace(n, ret);
    assert(flag);

    return ret;
}

} // namespace

// Implementation of complex integral exponentiation of cos(ex) + im * sin(ex)
// supported by a cache.
std::array<expression, 2> ccpow(const expression &ex, trig_eval_dict_t &td, std::int8_t n)
{
    auto it = td.find(ex);
    assert(it != td.end());

    auto &pd = it->second;

    assert(pd.contains(1));
    assert(pd.contains(-1));

    const auto pow1 = pd[1];
    const auto powm1 = pd[-1];

    return ccpow_impl(pd, pow1, powm1, n);
}

// Pairwise complex product.
std::array<expression, 2> pairwise_cmul(std::vector<std::array<expression, 2>> &terms)
{
    namespace hd = heyoka::detail;

    if (terms.empty()) {
        return {1_dbl, 0_dbl};
    }

    // LCOV_EXCL_START
    if (terms.size() == std::numeric_limits<decltype(terms.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_cmul()");
    }
    // LCOV_EXCL_STOP

    while (terms.size() != 1u) {
        std::vector<std::array<expression, 2>> new_terms;

        for (decltype(terms.size()) i = 0; i < terms.size(); i += 2u) {
            if (i + 1u == terms.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_terms.push_back(terms[i]);
            } else {
                new_terms.push_back(hd::ex_cmul(terms[i], terms[i + 1u]));
            }
        }

        new_terms.swap(terms);
    }

    return terms[0];
}

} // namespace detail

HEYOKA_END_NAMESPACE
