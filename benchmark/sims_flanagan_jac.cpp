// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepDE.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>

#include <fmt/ranges.h>

using namespace heyoka;

using v_ex_t = std::array<expression, 3>;

const auto mu = make_vars("mu");
const auto tm = make_vars("t");

std::pair<v_ex_t, v_ex_t> make_lp(const v_ex_t &pos_0, const v_ex_t &vel_0)
{
    const auto &[x0, y0, z0] = pos_0;
    const auto &[vx0, vy0, vz0] = vel_0;

    auto v02 = vx0 * vx0 + vy0 * vy0 + vz0 * vz0;
    auto r0 = sqrt(x0 * x0 + y0 * y0 + z0 * z0);
    auto eps = v02 * 0.5 - mu / r0;
    auto a = -mu / (2. * eps);

    auto sigma0 = (x0 * vx0 + y0 * vy0 + z0 * vz0) / sqrt(mu);
    auto s0 = sigma0 / sqrt(a);
    auto c0 = 1. - r0 / a;

    auto n = sqrt(mu / (a * a * a));
    auto DM = n * tm;

    auto DE = kepDE(s0, c0, DM);

    auto cDE = cos(DE);
    auto sDE = sin(DE);

    auto r = a + (r0 - a) * cDE + sigma0 * sqrt(a) * sDE;

    auto F = 1. - a / r0 * (1. - cDE);
    auto G = a * sigma0 / sqrt(mu) * (1. - cDE) + r0 * sqrt(a / mu) * sDE;
    auto Ft = -sqrt(mu * a) / (r * r0) * sDE;
    auto Gt = 1. - a / r * (1. - cDE);

    return {{{F * x0 + G * vx0, F * y0 + G * vy0, F * z0 + G * vz0}},
            {{Ft * x0 + Gt * vx0, Ft * y0 + Gt * vy0, Ft * z0 + Gt * vz0}}};
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    create_logger();
    auto logger = spdlog::get("heyoka");

    set_logger_level_trace();

    unsigned N{};

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("N", po::value<unsigned>(&N)->default_value(20u),
                                                       "number of segments");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (N == 0u) {
        throw std::invalid_argument("The number of segments cannot be zero");
    }

    auto [x0, y0, z0] = make_vars("x0", "y0", "z0");
    auto [vx0, vy0, vz0] = make_vars("vx0", "vy0", "vz0");

    std::vector<v_ex_t> Dvs;
    for (auto i = 0u; i < N; ++i) {
        auto [Dvx, Dvy, Dvz]
            = make_vars(fmt::format("Dvx{}", i + 1u), fmt::format("Dvy{}", i + 1u), fmt::format("Dvz{}", i + 1u));
        Dvs.push_back(v_ex_t{Dvx, Dvy, Dvz});
    }

    auto [pos_f, vel_f] = make_lp({x0, y0, z0}, {vx0, vy0, vz0});

    for (auto i = 0u; i < N; ++i) {
        vel_f[0] += Dvs[i][0];
        vel_f[1] += Dvs[i][1];
        vel_f[2] += Dvs[i][2];

        std::tie(pos_f, vel_f) = make_lp(pos_f, vel_f);
    }

    std::vector<expression> Dvs_list;
    for (const auto &Dv : Dvs) {
        Dvs_list.push_back(Dv[0]);
        Dvs_list.push_back(Dv[1]);
        Dvs_list.push_back(Dv[2]);
    }

    std::vector<expression> diff_vars_list = {x0, y0, z0, vx0, vy0, vz0};
    diff_vars_list.insert(diff_vars_list.end(), Dvs_list.begin(), Dvs_list.end());

    auto dt
        = diff_tensors({pos_f[0], pos_f[1], pos_f[2], vel_f[0], vel_f[1], vel_f[2]}, kw::diff_args = diff_vars_list);

    std::vector<expression> jac;
    auto jac_sr = dt.get_derivatives(1);
    std::transform(jac_sr.begin(), jac_sr.end(), std::back_inserter(jac), [](const auto &p) { return p.second; });

    llvm_state s;

    std::vector<expression> vars_list = {x0, y0, z0, vx0, vy0, vz0, mu, tm};
    vars_list.insert(vars_list.end(), Dvs_list.begin(), Dvs_list.end());

    logger->trace("Adding cfunc");

    add_cfunc<double>(s, "func", jac, kw::vars = vars_list);

    logger->trace("Compiling cfunc");

    s.compile();

    logger->trace("Looking up cfunc");

    auto *fptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("func"));

    logger->trace("All done");
}
