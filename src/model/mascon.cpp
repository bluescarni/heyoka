// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/model/mascon.hpp>
#include <heyoka/model/rotating.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

std::vector<std::pair<expression, expression>> mascon_impl(const expression &Gconst,
                                                           const std::vector<expression> &masses,
                                                           const std::vector<expression> &positions,
                                                           const std::vector<expression> &omega)
{
    auto fc_dyn = fixed_centres_impl(Gconst, masses, positions);
    auto rot_dyn = rotating_impl(omega);

    assert(fc_dyn.size() == 6u);
    assert(rot_dyn.size() == 6u);

    // NOTE: need to alter only the accelerations.
    for (auto i = 3u; i < 6u; ++i) {
        fc_dyn[i].second = std::move(fc_dyn[i].second) + std::move(rot_dyn[i].second);
    }

    return fc_dyn;
}

expression mascon_energy_impl(const expression &Gconst, const std::vector<expression> &masses,
                              const std::vector<expression> &positions, const std::vector<expression> &omega)
{
    auto fc_en = fixed_centres_energy_impl(Gconst, masses, positions);
    auto rot_pot = rotating_potential_impl(omega);

    return std::move(fc_en) + std::move(rot_pot);
}

expression mascon_potential_impl(const expression &Gconst, const std::vector<expression> &masses,
                                 const std::vector<expression> &positions, const std::vector<expression> &omega)
{
    auto fc_pot = fixed_centres_potential_impl(Gconst, masses, positions);
    auto rot_pot = rotating_potential_impl(omega);

    return std::move(fc_pot) + std::move(rot_pot);
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
