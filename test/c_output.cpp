// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

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
        REQUIRE(oss.str() == "Default-constructed continuous_output");

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -x}, {0., 1.}, kw::opt_level = opt_level, kw::high_accuracy = ha};

        auto d_out = std::get<4>(ta.propagate_until(10., kw::c_output = true));

        REQUIRE(d_out.has_value());
        REQUIRE(d_out->get_output().size() == 2u);
        REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
        REQUIRE(!d_out->get_tcs().empty());
        REQUIRE(!d_out->get_llvm_state().get_ir().empty());

        oss.str("");
        oss << *d_out;
        REQUIRE(boost::algorithm::contains(oss.str(), "forward"));

        // Reset time/state.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        // Run a grid propagation.
        auto t_grid = std::vector<fp_t>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
        auto grid_out = std::get<4>(ta.propagate_grid(t_grid));

        // Compare the two.
        for (auto i = 0u; i < 11u; ++i) {
            (*d_out)(t_grid[i]);
            REQUIRE(d_out->get_output()[0] == grid_out[2u * i]);
            REQUIRE(d_out->get_output()[1] == grid_out[2u * i + 1u]);
        }

        REQUIRE(d_out->get_bounds().first == 0.);
        REQUIRE(d_out->get_bounds().second == approximately(fp_t(10)));
        REQUIRE(d_out->get_n_steps() > 0u);

        // Try slightly outside the bounds.
        (*d_out)(-.01);
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(-0.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(-0.01))));
        (*d_out)(10.01);
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(10.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(10.01))));

        // Try making a copy too.
        auto co3 = *d_out;
        co3(4.);
        REQUIRE(co3.get_output()[0] == grid_out[2u * 4u]);
        REQUIRE(co3.get_output()[1] == grid_out[2u * 4u + 1u]);

        // Limiting case in which not steps are taken.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = std::numeric_limits<fp_t>::infinity();
        ta.set_time(0);
        d_out = std::get<4>(ta.propagate_until(10., kw::c_output = true));
        REQUIRE(!d_out.has_value());

        // Do it backwards too.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        d_out = std::get<4>(ta.propagate_until(-10., kw::c_output = true));

        REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
        REQUIRE(!d_out->get_tcs().empty());

        REQUIRE(d_out.has_value());

        oss.str("");
        oss << *d_out;
        REQUIRE(boost::algorithm::contains(oss.str(), "backward"));

        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        // Run a grid propagation.
        t_grid = std::vector<fp_t>{0., -1., -2., -3., -4., -5., -6., -7., -8., -9., -10.};
        grid_out = std::get<4>(ta.propagate_grid(t_grid));

        // Compare the two.
        for (auto i = 0u; i < 11u; ++i) {
            (*d_out)(t_grid[i]);
            REQUIRE(d_out->get_output()[0] == grid_out[2u * i]);
            REQUIRE(d_out->get_output()[1] == grid_out[2u * i + 1u]);
        }

        REQUIRE(d_out->get_bounds().first == 0.);
        REQUIRE(d_out->get_bounds().second == approximately(fp_t(-10)));
        REQUIRE(d_out->get_n_steps() > 0u);

        // Try slightly outside the bounds.
        (*d_out)(.01);
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(0.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(0.01))));
        (*d_out)(-10.01);
        REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(-10.01))));
        REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(-10.01))));

        // Try making a copy too.
        co = *d_out;
        co(-4.);
        REQUIRE(co.get_output()[0] == grid_out[2u * 4u]);
        REQUIRE(co.get_output()[1] == grid_out[2u * 4u + 1u]);

        co = *&co;
        co(-5.);
        REQUIRE(co.get_output()[0] == grid_out[2u * 5u]);
        REQUIRE(co.get_output()[1] == grid_out[2u * 5u + 1u]);

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

        REQUIRE(co.get_output()[0] == grid_out[2u * 5u]);
        REQUIRE(co.get_output()[1] == grid_out[2u * 5u + 1u]);

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
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto ha : {false, true}) {
            tuple_for_each(fp_types, [&tester, opt_level, ha](auto x) { tester(x, opt_level, ha); });
        }
    }
}
