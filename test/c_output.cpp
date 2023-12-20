// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

static std::mt19937 rng;

TEST_CASE("scalar")
{
    using std::cos;
    using std::sin;

    using Catch::Matchers::Message;

    auto tester = [](auto fp_x, unsigned opt_level, bool ha) {
        using fp_t = decltype(fp_x);

        std::stringstream oss;

        // Basic testing.
        continuous_output<fp_t> co;
        REQUIRE(co.get_output().empty());
        REQUIRE(co.get_times().empty());
        REQUIRE(co.get_tcs().empty());
        REQUIRE_THROWS_MATCHES(co(0.), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output object"));
        REQUIRE_THROWS_MATCHES(co.get_bounds(), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output object"));
        REQUIRE_THROWS_MATCHES(co.get_n_steps(), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output object"));

        auto co2 = co;
        REQUIRE(co2.get_output().empty());
        REQUIRE_THROWS_MATCHES(co2(0.), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output object"));

        oss << co;
        REQUIRE(boost::algorithm::contains(oss.str(), "Default-constructed continuous_output"));

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -x}, {0., 1.}, kw::opt_level = opt_level, kw::high_accuracy = ha};

        auto [_0, _1, _2, tot_steps, d_out, _3] = ta.propagate_until(10., kw::c_output = true);

        REQUIRE(d_out.has_value());
        REQUIRE(d_out->get_output().size() == 2u);
        REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
        REQUIRE(!d_out->get_tcs().empty());
        REQUIRE(!d_out->get_llvm_state().get_ir().empty());
        REQUIRE(tot_steps == d_out->get_n_steps());

        oss.str("");
        oss << *d_out;
        REQUIRE(boost::algorithm::contains(oss.str(), "forward"));

        // Reset time/state.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        // Run a grid propagation.
        auto t_grid = std::vector<fp_t>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
        auto grid_out = std::get<5>(ta.propagate_grid(t_grid));

        // Compare the two.
        for (auto i = 0u; i < 11u; ++i) {
            (*d_out)(t_grid[i]);
            REQUIRE(d_out->get_output()[0] == approximately(grid_out[2u * i], fp_t(10)));
            REQUIRE(d_out->get_output()[1] == approximately(grid_out[2u * i + 1u], fp_t(10)));
        }

        REQUIRE(d_out->get_bounds().first == 0.);
        REQUIRE(d_out->get_bounds().second == approximately(fp_t(10)));
        REQUIRE(d_out->get_n_steps() > 0u);

        // Try slightly outside the bounds.
        (*d_out)(fp_t(-.01));
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(-0.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(-0.01))));
        (*d_out)(fp_t(10.01));
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(10.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(10.01))));

        // Try making a copy too.
        auto co3 = *d_out;
        co3(4.);
        REQUIRE(co3.get_output()[0] == approximately(grid_out[2u * 4u], fp_t(10)));
        REQUIRE(co3.get_output()[1] == approximately(grid_out[2u * 4u + 1u], fp_t(10)));

        // Limiting case in which not steps are taken.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = std::numeric_limits<fp_t>::infinity();
        ta.set_time(0);
        d_out = std::get<4>(ta.propagate_until(10., kw::c_output = true));
        REQUIRE(!d_out.has_value());

        // Try with propagate_for() too.

        // Reset time/state.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        std::tie(_0, _1, _2, tot_steps, d_out, _3) = ta.propagate_for(10., kw::c_output = true);

        REQUIRE(d_out.has_value());
        REQUIRE(d_out->get_output().size() == 2u);
        REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
        REQUIRE(!d_out->get_tcs().empty());
        REQUIRE(!d_out->get_llvm_state().get_ir().empty());
        REQUIRE(tot_steps == d_out->get_n_steps());

        // Compare the two.
        for (auto i = 0u; i < 11u; ++i) {
            (*d_out)(t_grid[i]);
            REQUIRE(d_out->get_output()[0] == approximately(grid_out[2u * i], fp_t(10)));
            REQUIRE(d_out->get_output()[1] == approximately(grid_out[2u * i + 1u], fp_t(10)));
        }

        // Do it backwards too.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        d_out = std::get<4>(ta.propagate_until(-10., kw::c_output = true));

        REQUIRE(d_out.has_value());
        REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
        REQUIRE(!d_out->get_tcs().empty());

        oss.str("");
        oss << *d_out;
        REQUIRE(boost::algorithm::contains(oss.str(), "backward"));

        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        // Run a grid propagation.
        t_grid = std::vector<fp_t>{0., -1., -2., -3., -4., -5., -6., -7., -8., -9., -10.};
        grid_out = std::get<5>(ta.propagate_grid(t_grid));

        // Compare the two.
        for (auto i = 0u; i < 11u; ++i) {
            (*d_out)(t_grid[i]);
            REQUIRE(d_out->get_output()[0] == approximately(grid_out[2u * i], fp_t(10)));
            REQUIRE(d_out->get_output()[1] == approximately(grid_out[2u * i + 1u], fp_t(10)));
        }

        REQUIRE(d_out->get_bounds().first == 0.);
        REQUIRE(d_out->get_bounds().second == approximately(fp_t(-10)));
        REQUIRE(d_out->get_n_steps() > 0u);

        // Try slightly outside the bounds.
        (*d_out)(fp_t(.01));
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(0.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(0.01))));
        (*d_out)(fp_t(-10.01));
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(-10.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(-10.01))));

        // Try making a copy too.
        co = *d_out;
        co(-4.);
        REQUIRE(co.get_output()[0] == approximately(grid_out[2u * 4u], fp_t(10)));
        REQUIRE(co.get_output()[1] == approximately(grid_out[2u * 4u + 1u], fp_t(10)));

        co = *&co;
        co(-5.);
        REQUIRE(co.get_output()[0] == approximately(grid_out[2u * 5u], fp_t(10)));
        REQUIRE(co.get_output()[1] == approximately(grid_out[2u * 5u + 1u], fp_t(10)));

        // Limiting case in which not steps are taken.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = std::numeric_limits<fp_t>::infinity();
        ta.set_time(0);
        d_out = std::get<4>(ta.propagate_until(-10., kw::c_output = true));
        REQUIRE(!d_out.has_value());

        // Try with non-finite time.
        REQUIRE_THROWS_AS(co(std::numeric_limits<fp_t>::infinity()), std::invalid_argument);

        // s11n testing.
        oss.str("");

        {
            boost::archive::binary_oarchive oa(oss);
            oa << co;
        }

        co = continuous_output<fp_t>{};

        {
            boost::archive::binary_iarchive ia(oss);
            ia >> co;
        }

        REQUIRE(co.get_output()[0] == approximately(grid_out[2u * 5u], fp_t(10)));
        REQUIRE(co.get_output()[1] == approximately(grid_out[2u * 5u + 1u], fp_t(10)));

        // Try with a def-cted object too.
        oss.str("");

        continuous_output<fp_t> co4;

        {
            boost::archive::binary_oarchive oa(oss);
            oa << co4;
        }

        co4 = co;

        {
            boost::archive::binary_iarchive ia(oss);
            ia >> co4;
        }

        REQUIRE(co4.get_output().empty());
        REQUIRE_THROWS_MATCHES(co4(0.), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output object"));

        // Try with c_output=false too.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        std::tie(_0, _1, _2, tot_steps, d_out, _3) = ta.propagate_for(10., kw::c_output = false);

        REQUIRE(!d_out.has_value());
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto ha : {false, true}) {
            tuple_for_each(fp_types, [&tester, opt_level, ha](auto x) { tester(x, opt_level, ha); });
        }
    }
}

TEST_CASE("batch")
{
    using std::cos;
    using std::sin;

    using Catch::Matchers::Message;

    auto tester = [](auto fp_x, unsigned opt_level, bool ha, unsigned batch_size) {
        using fp_t = decltype(fp_x);

        std::stringstream oss;

        // Vector to pass to the call operator of the
        // continuous_output_batch objects.
        std::vector<fp_t> loc_time(batch_size);

        // Basic testing.
        continuous_output_batch<fp_t> co;
        REQUIRE(co.get_output().empty());
        REQUIRE(co.get_times().empty());
        REQUIRE(co.get_tcs().empty());
        REQUIRE(co.get_batch_size() == 0u);
        REQUIRE_THROWS_MATCHES(co(loc_time), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));
        REQUIRE_THROWS_MATCHES(co(loc_time[0]), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));
        REQUIRE_THROWS_MATCHES(co(loc_time.data()), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));
        REQUIRE_THROWS_MATCHES(co(std::vector<fp_t>{}), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));
        REQUIRE_THROWS_MATCHES(co.get_bounds(), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));
        REQUIRE_THROWS_MATCHES(co.get_n_steps(), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));

        auto co2 = co;
        REQUIRE(co2.get_output().empty());
        REQUIRE(co2.get_batch_size() == 0u);
        REQUIRE_THROWS_MATCHES(co2(loc_time), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));

        oss << co;
        REQUIRE(boost::algorithm::contains(oss.str(), "Default-constructed continuous_output_batch"));

        auto [x, v] = make_vars("x", "v");

        // Create the vector of initial conditions.
        std::vector<fp_t> ic;
        for (auto i = 0u; i < batch_size; ++i) {
            ic.push_back(fp_t(i) / 100);
        }
        for (auto i = 0u; i < batch_size; ++i) {
            ic.push_back(1 + fp_t(i) / 100);
        }

        // The vector of initial times.
        std::vector<fp_t> init_tm(batch_size, fp_t(0));

        // The vector of final times.
        std::vector<fp_t> final_tm;
        for (auto i = 0u; i < batch_size; ++i) {
            final_tm.push_back(fp_t(10.) + fp_t(i) / 100);
        }

        // Create a random batch grid.
        const auto n_points = 10u;
        std::vector<fp_t> grid(batch_size * n_points), tmp(n_points - 2u);
        for (auto i = 0u; i < batch_size; ++i) {
            // First point is always zero.
            grid[i] = 0;

            std::uniform_real_distribution<double> rdist(1e-6, 10. + i / 100. - 1e-6);
            for (auto j = 0u; j < n_points - 2u; ++j) {
                tmp[j] = static_cast<fp_t>(rdist(rng));
            }
            std::sort(tmp.begin(), tmp.end());

            for (auto j = 0u; j < n_points - 2u; ++j) {
                grid[(j + 1u) * batch_size + i] = tmp[j];
            }

            // Last point.
            grid[grid.size() - batch_size + i] = final_tm[i];
        }

        auto ta = taylor_adaptive_batch<fp_t>{
            {prime(x) = v, prime(v) = -x}, ic, batch_size, kw::opt_level = opt_level, kw::high_accuracy = ha};

        auto d_out = ta.propagate_until(final_tm, kw::c_output = true);

        REQUIRE(d_out.has_value());
        REQUIRE(d_out->get_output().size() == 2u * batch_size);
        REQUIRE(d_out->get_times().size() == (d_out->get_n_steps() + 2u) * batch_size);
        REQUIRE(!d_out->get_tcs().empty());
        REQUIRE(!d_out->get_llvm_state().get_ir().empty());

        oss.str("");
        oss << *d_out;
        REQUIRE(boost::algorithm::contains(oss.str(), "forward"));

        // Reset time/state.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);

        // Run a grid propagation.
        auto grid_out = ta.propagate_grid(grid);

        // Compare the two.
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            (*d_out)(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(d_out->get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(d_out->get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }

            // Try the scalar version too.
            for (auto j = 0u; j < batch_size; ++j) {
                (*d_out)(loc_time[j]);
                REQUIRE(d_out->get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(d_out->get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        REQUIRE(d_out->get_bounds().first.size() == batch_size);
        REQUIRE(d_out->get_bounds().second.size() == batch_size);
        for (std::size_t j = 0; j < batch_size; ++j) {
            REQUIRE(d_out->get_bounds().first[j] == 0.);
            REQUIRE(d_out->get_bounds().second[j] == approximately(final_tm[j]));
        }
        REQUIRE(d_out->get_n_steps() > 0u);

        // Try slightly outside the bounds.
        for (auto j = 0u; j < batch_size; ++j) {
            loc_time[j] = fp_t(-0.01);
        }
        (*d_out)(loc_time);
        for (auto j = 0u; j < batch_size; ++j) {
            REQUIRE(d_out->get_output()[j]
                    == approximately(ic[j] * cos(loc_time[j]) + ic[batch_size + j] * sin(loc_time[j])));
            REQUIRE(d_out->get_output()[batch_size + j]
                    == approximately(-ic[j] * sin(loc_time[j]) + ic[batch_size + j] * cos(loc_time[j])));
        }
        for (auto j = 0u; j < batch_size; ++j) {
            loc_time[j] = final_tm[j] + fp_t(0.01);
        }
        (*d_out)(loc_time);
        for (auto j = 0u; j < batch_size; ++j) {
            REQUIRE(d_out->get_output()[j]
                    == approximately(ic[j] * cos(loc_time[j]) + ic[batch_size + j] * sin(loc_time[j])));
            REQUIRE(d_out->get_output()[batch_size + j]
                    == approximately(-ic[j] * sin(loc_time[j]) + ic[batch_size + j] * cos(loc_time[j])));
        }

        // Try making a copy too.
        auto co3 = *d_out;
        REQUIRE(co3.get_output().size() == batch_size * 2u);
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            co3(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(co3.get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(co3.get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        // Limiting case in which not steps are taken.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);
        // Set the first velocity in the batch to infinity.
        ta.get_state_data()[batch_size] = std::numeric_limits<fp_t>::infinity();
        d_out = ta.propagate_until(final_tm, kw::c_output = true);
        REQUIRE(!d_out.has_value());

        // Try with propagate_for() too.

        // Reset time/state.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);

        d_out = ta.propagate_for(final_tm, kw::c_output = true);

        REQUIRE(d_out.has_value());
        REQUIRE(d_out->get_output().size() == 2u * batch_size);
        REQUIRE(d_out->get_times().size() == (d_out->get_n_steps() + 2u) * batch_size);
        REQUIRE(!d_out->get_tcs().empty());
        REQUIRE(!d_out->get_llvm_state().get_ir().empty());

        // Compare the two.
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            (*d_out)(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(d_out->get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(d_out->get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        REQUIRE(d_out->get_bounds().first.size() == batch_size);
        REQUIRE(d_out->get_bounds().second.size() == batch_size);
        for (std::size_t j = 0; j < batch_size; ++j) {
            REQUIRE(d_out->get_bounds().first[j] == 0.);
            REQUIRE(d_out->get_bounds().second[j] == approximately(final_tm[j]));
        }
        REQUIRE(d_out->get_n_steps() > 0u);

        // Integrate backwards in time.
        for (auto j = 0u; j < batch_size; ++j) {
            final_tm[j] = -final_tm[j];
        }
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                grid[i * batch_size + j] = -grid[i * batch_size + j];
            }
        }

        // Reset the integrator state.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);

        d_out = ta.propagate_until(final_tm, kw::c_output = true);

        REQUIRE(d_out->get_times().size() == d_out->get_n_steps() * batch_size + 2u * batch_size);
        REQUIRE(d_out.has_value());
        REQUIRE(!d_out->get_tcs().empty());

        oss.str("");
        oss << *d_out;
        REQUIRE(boost::algorithm::contains(oss.str(), "backward"));

        // Reset the integrator state.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);

        grid_out = ta.propagate_grid(grid);

        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            (*d_out)(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(d_out->get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(d_out->get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        REQUIRE(d_out->get_bounds().first.size() == batch_size);
        REQUIRE(d_out->get_bounds().second.size() == batch_size);
        for (std::size_t j = 0; j < batch_size; ++j) {
            REQUIRE(d_out->get_bounds().first[j] == 0.);
            REQUIRE(d_out->get_bounds().second[j] == approximately(final_tm[j]));
        }
        REQUIRE(d_out->get_n_steps() > 0u);

        // Try slightly outside the bounds.
        for (auto j = 0u; j < batch_size; ++j) {
            loc_time[j] = fp_t(0.01);
        }
        (*d_out)(loc_time);
        for (auto j = 0u; j < batch_size; ++j) {
            REQUIRE(d_out->get_output()[j]
                    == approximately(ic[j] * cos(loc_time[j]) + ic[batch_size + j] * sin(loc_time[j])));
            REQUIRE(d_out->get_output()[batch_size + j]
                    == approximately(-ic[j] * sin(loc_time[j]) + ic[batch_size + j] * cos(loc_time[j])));
        }
        for (auto j = 0u; j < batch_size; ++j) {
            loc_time[j] = final_tm[j] - fp_t(0.01);
        }
        (*d_out)(loc_time);
        for (auto j = 0u; j < batch_size; ++j) {
            REQUIRE(d_out->get_output()[j]
                    == approximately(ic[j] * cos(loc_time[j]) + ic[batch_size + j] * sin(loc_time[j])));
            REQUIRE(d_out->get_output()[batch_size + j]
                    == approximately(-ic[j] * sin(loc_time[j]) + ic[batch_size + j] * cos(loc_time[j])));
        }

        // Try making a copy too.
        co = *d_out;
        REQUIRE(co.get_output().size() == batch_size * 2u);
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            co(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(co.get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(co.get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        co = *&co;
        REQUIRE(co.get_output().size() == batch_size * 2u);
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            co(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(co.get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(co.get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        // Limiting case in which not steps are taken.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);
        // Set the first velocity in the batch to infinity.
        ta.get_state_data()[batch_size] = std::numeric_limits<fp_t>::infinity();
        d_out = ta.propagate_until(final_tm, kw::c_output = true);
        REQUIRE(!d_out.has_value());

        // Try with non-finite time.
        loc_time[0] = std::numeric_limits<fp_t>::infinity();
        REQUIRE_THROWS_AS(co(loc_time), std::invalid_argument);
        REQUIRE_THROWS_AS(co(loc_time[0]), std::invalid_argument);

        // s11n testing.
        oss.str("");

        {
            boost::archive::binary_oarchive oa(oss);
            oa << co;
        }

        co = continuous_output_batch<fp_t>{};

        {
            boost::archive::binary_iarchive ia(oss);
            ia >> co;
        }

        REQUIRE(co.get_output().size() == batch_size * 2u);
        for (auto i = 0u; i < n_points; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                loc_time[j] = grid[i * batch_size + j];
            }

            co(loc_time);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(co.get_output()[j] == approximately(grid_out[2u * i * batch_size + j], fp_t(10)));
                REQUIRE(co.get_output()[batch_size + j]
                        == approximately(grid_out[2u * i * batch_size + batch_size + j], fp_t(10)));
            }
        }

        // Try with a def-cted object too.
        oss.str("");

        continuous_output_batch<fp_t> co4;

        {
            boost::archive::binary_oarchive oa(oss);
            oa << co4;
        }

        co4 = co;

        {
            boost::archive::binary_iarchive ia(oss);
            ia >> co4;
        }

        REQUIRE(co4.get_output().empty());
        REQUIRE_THROWS_MATCHES(co4(loc_time), std::invalid_argument,
                               Message("Cannot use a default-constructed continuous_output_batch object"));

        // Try with an input vector to the call operator of the wrong size.
        REQUIRE_THROWS_MATCHES(
            co(std::vector<fp_t>{}), std::invalid_argument,
            Message(
                fmt::format("An invalid time vector was passed to the call operator of continuous_output_batch: the "
                            "vector size is 0, but a size of {} was expected instead",
                            batch_size)));

        // Try with c_output=false too.
        std::copy(ic.begin(), ic.end(), ta.get_state_data());
        ta.set_time(init_tm);

        d_out = ta.propagate_for(final_tm, kw::c_output = false);

        REQUIRE(!d_out.has_value());
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto ha : {false, true}) {
            for (auto batch_size : {1u, 2u, 4u, 5u}) {
                tuple_for_each(fp_types,
                               [&tester, opt_level, ha, batch_size](auto x) { tester(x, opt_level, ha, batch_size); });
            }
        }
    }
}
