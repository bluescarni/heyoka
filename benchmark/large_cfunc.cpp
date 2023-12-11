// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <random>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/mascon.hpp>

using namespace heyoka;

int main()
{
    set_logger_level_trace();

    std::uint32_t N = 20000ull;

    std::mt19937 rng;
    std::uniform_real_distribution<double> rdist;

    auto gen = [&rng, &rdist]() { return rdist(rng); };

    std::vector<double> pos_vals, masses_vals;

    std::generate_n(std::back_inserter(pos_vals), N * 3u, gen);
    std::generate_n(std::back_inserter(masses_vals), N, gen);

    std::vector<expression> pos, masses;

    for (std::uint32_t i = 0; i < N; ++i) {
        for (auto j = 0u; j < 3u; ++j) {
            pos.push_back(par[i * 3u + j]);
        }
    }

    for (std::uint32_t i = 3u * N; i < 3u * N + N; ++i) {
        masses.push_back(par[i]);
    }

    auto en = model::fixed_centres_energy(kw::positions = pos, kw::masses = masses);

    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    llvm_state s;

    add_cfunc<double>(s, "en", {en}, kw::vars = {x, y, z, vx, vy, vz}, kw::compact_mode = true);

    s.compile();

    [[maybe_unused]] auto fn = s.jit_lookup("en");
}
