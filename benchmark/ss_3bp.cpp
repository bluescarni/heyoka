// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double tol;
    bool with_dense = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(1e-15),
                                                       "tolerance")("with_dense", "enable dense output");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("with_dense")) {
        with_dense = true;
    }

    warmup();

    // Prepare the time grid, if needed.
    std::vector<double> grid;
    if (with_dense) {
        for (auto i = 0ull; i < 500000ull; ++i) {
            grid.push_back(100000. / 500000 * i);
        }
        grid.push_back(100000.);
    }

    auto masses = {1.00000597682, 1 / 1047.355, 1 / 3501.6};
    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = make_nbody_sys(3, kw::masses = masses, kw::Gconst = G);

    auto ic = {// Sun.
               -5.137271893918405e-03, -5.288891104344273e-03, 6.180743702483316e-06, 2.3859757364179156e-03,
               -2.3396779489468049e-03, -8.1384891821122709e-07,
               // Jupiter.
               3.404393156051084, 3.6305811472186558, 0.0342464685434024, -2.0433186406983279e+00,
               2.0141003472039567e+00, -9.7316724504621210e-04,
               // Saturn.
               6.606942557811084, 6.381645992310656, -0.1361381213577972, -1.5233982268351876, 1.4589658329821569,
               0.0061033600397708};

    taylor_adaptive<double> ta{std::move(sys), ic, kw::tol = tol};

    taylor_outcome oc;
    double h_min, h_max;
    std::size_t n_steps;
    std::vector<double> d_out;
    std::optional<continuous_output<double>> c_out;

    auto start = std::chrono::high_resolution_clock::now();

    if (with_dense) {
        std::tie(oc, h_min, h_max, n_steps, d_out) = ta.propagate_grid(std::move(grid));
    } else {
        std::tie(oc, h_min, h_max, n_steps, c_out) = ta.propagate_until(100000.);
    }

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";
    std::cout << "Outcome: " << oc << '\n';
    std::cout << "Min/max timestep: " << h_min << ", " << h_max << '\n';

    std::cout << ta << '\n';

    // Reference result from a quadruple-precision integration.
    const auto ref = std::vector{0.00529783352211642635986448512870267126,   0.00443801687663031764860286477965782292,
                                 -9.26927271770180624859940898715912866e-07, -0.00216531823419081350439333313986268641,
                                 0.00214032867924028123522294255311480690,   -8.67721509026440812519470549521477867e-06,
                                 -2.92497467718230374582976002426387292,     -4.36042856694508689271852041970916660,
                                 -0.0347154846850475113508604212948948637,   2.21048191345880323553795415392017512,
                                 -1.58848197474132095629314575660274164,     0.00454279037439684585301286724874262847,
                                 -8.77199825846701098536240728390136140,     -0.962123421465639669245658270137496032,
                                 0.119309299617985001156428107644659796,     0.191865835965930410443458136955585028,
                                 -2.18388123410681074311949955152592019,     0.0151965022008957683497311770695583142};

    double rms_err = 0;
    for (auto i = 0u; i < 18u; ++i) {
        const auto loc_err = ref[i] - ta.get_state()[i];
        rms_err += loc_err * loc_err;
    }
    std::cout << "RMS error: " << std::sqrt(rms_err / 18) << '\n';

    return 0;
}
