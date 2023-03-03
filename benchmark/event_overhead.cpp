// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <iostream>
#include <vector>

#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main()
{
    auto sys = model::nbody(2, kw::masses = {1., 0.});

    const auto ic = std::vector{0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.};

    taylor_adaptive<double> ta{sys, ic, kw::compact_mode = true};

    taylor_adaptive<double> ta_ev{
        sys, ic, kw::compact_mode = true,
        kw::nt_events = {nt_event<double>("x_1"_var - 1000., [](taylor_adaptive<double> &, double, int) {})}};

    // NOTE: propagate for a while then reset, to prepare the caches.
    ta_ev.propagate_until(1000.);
    ta_ev.set_time(0.);
    std::copy(ic.begin(), ic.end(), ta_ev.get_state_data());

    warmup();

    auto start = std::chrono::high_resolution_clock::now();
    ta.propagate_until(100000.);
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Without events: " << elapsed << "μs\n";

    warmup();

    start = std::chrono::high_resolution_clock::now();
    ta_ev.propagate_until(100000.);
    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "With events: " << elapsed << "μs\n";
}
