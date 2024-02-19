// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/model/pendulum.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

std::vector<std::pair<expression, expression>> pendulum_impl(const expression &gconst, const expression &l)
{
    auto [x, v] = make_vars("x", "v");

    return {prime(x) = v, prime(v) = -gconst / l * sin(x)};
}

expression pendulum_energy_impl(const expression &gconst, const expression &l)
{
    auto [x, v] = make_vars("x", "v");

    return 0.5_dbl * (l * l) * (v * v) + gconst * l * (1_dbl - cos(x));
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
