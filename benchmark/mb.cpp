// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/expression.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/taylor.hpp>

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;
    using namespace fmt::literals;
    using namespace heyoka;
    using std::sqrt;
    using t_ev_t = t_event<double>;

    unsigned N_sqrt;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("N_sqrt", po::value<unsigned>(&N_sqrt)->default_value(25u),
                                                       "square root of the number of particles");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (N_sqrt == 0u) {
        throw std::invalid_argument("N_sqrt cannot be zero");
    }

    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // Number of particles.
    static const auto N = N_sqrt * N_sqrt;

    // Particle radius.
    const auto p_radius = .75;

    // Box size.
    const auto box_size = N_sqrt * 10.;

    // The list of ODEs.
    std::vector<std::pair<expression, expression>> eqns;

    for (auto i = 0u; i < N; ++i) {
        auto [xi, yi, vxi, vyi] = make_vars("x_{}"_format(i), "y_{}"_format(i), "vx_{}"_format(i), "vy_{}"_format(i));

        eqns.push_back(prime(xi) = vxi);
        eqns.push_back(prime(yi) = vyi);
        eqns.push_back(prime(vxi) = 0_dbl);
        eqns.push_back(prime(vyi) = 0_dbl);
    }

    // Collisions with the left/right walls.
    struct cb_left_right {
        unsigned idx;

        bool operator()(taylor_adaptive<double> &ta, bool, int) const
        {
            ta.get_state_data()[idx * 4u + 2u] = -ta.get_state_data()[idx * 4u + 2u];

            return true;
        }
    };

    // Collisions with the top/bottom walls.
    struct cb_top_bottom {
        unsigned idx;

        bool operator()(taylor_adaptive<double> &ta, bool, int) const
        {
            ta.get_state_data()[idx * 4u + 3u] = -ta.get_state_data()[idx * 4u + 3u];

            return true;
        }
    };

    // Collision counter.
    static auto coll_counter = 0ull;

    // Sphere-sphere collision.
    struct cb_sph_sph {
        unsigned i, j;

        bool operator()(taylor_adaptive<double> &ta, bool, int) const
        {
            ++coll_counter;

            auto s_array = xt::adapt(ta.get_state_data(), {N, 4u});

            // Determine the unit vector uij
            // connecting particle i to particle j.
            auto ri = xt::view(s_array, i, xt::range(0, 2));
            auto rj = xt::view(s_array, j, xt::range(0, 2));
            auto rij = rj - ri;
            auto uij = rij / sqrt(xt::linalg::dot(rij, rij)[0]);

            // Fetch the velocities of the
            // two particles.
            auto vi = xt::view(s_array, i, xt::range(2, 4));
            auto vj = xt::view(s_array, j, xt::range(2, 4));

            // Project vi/vj across uij.
            const auto proj_i = xt::linalg::dot(vi, uij)[0];
            const auto proj_j = xt::linalg::dot(vj, uij)[0];

            // Flip the velocity components
            // across uij.
            vi -= proj_i * uij;
            vj -= proj_j * uij;
            vi += proj_j * uij;
            vj += proj_i * uij;

            return true;
        }
    };

    // The list of events.
    std::vector<t_ev_t> t_events;

    // Collisions with the box walls.
    for (auto i = 0u; i < N; ++i) {
        t_events.emplace_back(expression("y_{}"_format(i)) + (box_size / 2 - p_radius), kw::callback = cb_top_bottom{i},
                              kw::direction = event_direction::negative);
        t_events.emplace_back(expression("y_{}"_format(i)) - (box_size / 2 - p_radius), kw::callback = cb_top_bottom{i},
                              kw::direction = event_direction::positive);
        t_events.emplace_back(expression("x_{}"_format(i)) + (box_size / 2 - p_radius), kw::callback = cb_left_right{i},
                              kw::direction = event_direction::negative);
        t_events.emplace_back(expression("x_{}"_format(i)) - (box_size / 2 - p_radius), kw::callback = cb_left_right{i},
                              kw::direction = event_direction::positive);
    }

    // Sphere-sphere collisions.
    for (auto i = 0u; i < N; ++i) {
        auto [xi, yi] = make_vars("x_{}"_format(i), "y_{}"_format(i));

        for (auto j = i + 1u; j < N; ++j) {
            auto [xj, yj] = make_vars("x_{}"_format(j), "y_{}"_format(j));

            t_events.emplace_back(square(xi - xj) + square(yi - yj) - 4 * p_radius * p_radius,
                                  kw::callback = cb_sph_sph{i, j}, kw::direction = event_direction::negative);
        }
    }

    spdlog::stopwatch sw;

    // Create the integrator.
    auto ta = taylor_adaptive<double>(eqns, std::vector<double>(N * 4u), kw::compact_mode = true,
                                      kw::t_events = t_events, kw::tol = 1.);

    logger->trace("Integrator creation time: {}", sw);

    auto s_array = xt::adapt(ta.get_state_data(), {N, 4u});

    // Generate randomly velocities and positions.
    std::mt19937 rng(42u);
    std::uniform_real_distribution<double> rdist(-1., 1.);

    for (auto i = 0u; i < N; ++i) {
        const auto vx = rdist(rng), vy = rdist(rng);
        const auto vnorm = sqrt(vx * vx + vy * vy);

        auto vi = xt::view(s_array, i, xt::range(2, 4));
        vi[0] = vx / vnorm;
        vi[1] = vy / vnorm;
    }

    for (auto i = 0u; i < N_sqrt; ++i) {
        for (auto j = 0u; j < N_sqrt; ++j) {
            auto pos = xt::view(s_array, i * N_sqrt + j, xt::range(0, 2));

            pos[0] = (i * 10.) + 5 - box_size / 2;
            pos[1] = (j * 10.) + 5 - box_size / 2;
        }
    }

    // Create the time grid.
    const auto Ngrid = 200;
    auto lgrid = xt::linspace<double>(0., static_cast<double>(Ngrid), Ngrid);
    std::vector<double> t_grid(lgrid.begin(), lgrid.end());

    sw.reset();

    // Propagate.
    auto [oc, _1, _2, nsteps, res] = ta.propagate_grid(t_grid, kw::max_delta_t = 1.);

    logger->trace("Integration time   : {}", sw);
    logger->info("Integration outcome : {}", oc);
    logger->info("N of steps          : {}", nsteps);
    logger->info("Number of collisions: {}", coll_counter);
}
