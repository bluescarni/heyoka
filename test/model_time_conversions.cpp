// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
#include <cmath>
#include <sstream>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/time_conversions.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("delta_tt_tai")
{
    // Single precision.
    {
        auto cf = cfunc<float>{{model::delta_tt_tai}, {}};
        std::array<float, 1> out{};
        std::array<float, 0> in{};
        cf(out, in);
        REQUIRE(out[0] == 32.184f);
    }

    // Double precision.
    {
        auto cf = cfunc<double>{{model::delta_tt_tai}, {}};
        std::array<double, 1> out{};
        std::array<double, 0> in{};
        cf(out, in);
        REQUIRE(out[0] == 32.184);
    }

#if defined(HEYOKA_HAVE_REAL128)

    // Quad precision.
    {
        auto cf = cfunc<mppp::real128>{{model::delta_tt_tai}, {}};
        std::array<mppp::real128, 1> out{};
        std::array<mppp::real128, 0> in{};
        cf(out, in);
        REQUIRE(out[0] == mppp::real128{"32.184"});
    }

#endif

    // Streaming test.
    {
        std::ostringstream oss;
        oss << model::delta_tt_tai;
        REQUIRE(oss.str() == "delta_tt_tai(32.184)");
    }

    // Serialisation test.
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = model::delta_tt_tai + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == model::delta_tt_tai + x);
}

TEST_CASE("delta_tdb_tt")
{
    // A few correctness tests computed via astropy, e.g.:
    //
    // >>> tm = Time(val=2451545.0, val2=100., format='jd', scale='tdb')
    // >>> (tm.jd - tm.tt.jd)*86400.
    // np.float64(0.0016495585441589355)
    {
        auto cf = cfunc<double>{{model::delta_tdb_tt()}, {}};
        std::array<double, 1> out{};
        std::array<double, 0> in{};

        cf(out, in, kw::time = 0.);
        REQUIRE(std::abs(out[0] - -8.046627044677734e-05) < 4e-5);

        cf(out, in, kw::time = 86400.);
        REQUIRE(std::abs(out[0] - -8.046627044677734e-05) < 4e-5);

        cf(out, in, kw::time = 2. * 86400.);
        REQUIRE(std::abs(out[0] - -4.023313522338867e-05) < 4e-5);

        cf(out, in, kw::time = 100 * 86400.);
        REQUIRE(std::abs(out[0] - 0.0016495585441589355) < 4e-5);

        cf(out, in, kw::time = 1000 * 86400.);
        REQUIRE(std::abs(out[0] - -0.0016093254089355469) < 4e-5);

        cf(out, in, kw::time = 3456 * 86400.);
        REQUIRE(std::abs(out[0] - 0.00048279762268066406) < 4e-5);

        cf(out, in, kw::time = 9456 * 86400.);
        REQUIRE(std::abs(out[0] - -0.0011265277862548828) < 4e-5);
    }

    // Streaming test.
    {
        std::ostringstream oss;
        oss << model::delta_tdb_tt();
        REQUIRE(boost::algorithm::contains(oss.str(), "delta_tdb_tt_K(1.657e-3)"));
        REQUIRE(boost::algorithm::contains(oss.str(), "delta_tdb_tt_EB(1.671e-2)"));
        REQUIRE(boost::algorithm::contains(oss.str(), "delta_tdb_tt_M0(6.239996)"));
        REQUIRE(boost::algorithm::contains(oss.str(), "delta_tdb_tt_M1(1.99096871e-7)"));
    }

    // Serialisation test.
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = model::delta_tdb_tt(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == model::delta_tdb_tt(x));
}
