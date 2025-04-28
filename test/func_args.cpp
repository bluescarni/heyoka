// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("func_args basics")
{
    using Catch::Matchers::Message;

    {
        func_args f;
        REQUIRE(!f.get_shared_args());
        REQUIRE(f.get_args().empty());

        func_args f2{f};
        REQUIRE(!f2.get_shared_args());
        REQUIRE(f2.get_args().empty());

        REQUIRE(!f.get_shared_args());
        REQUIRE(f.get_args().empty());
    }

    {
        func_args f{{"x"_var}};
        REQUIRE(!f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);

        func_args f2{std::move(f)};
        REQUIRE(!f2.get_shared_args());
        REQUIRE(f2.get_args().size() == 1u);
        REQUIRE(f2.get_args()[0] == "x"_var);

        f = f2;
        REQUIRE(!f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);
    }

    {
        func_args f{{"x"_var}, true};
        REQUIRE(f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);

        func_args f2;
        f2 = f;
        REQUIRE(f2.get_shared_args());
        REQUIRE(f2.get_args().size() == 1u);
        REQUIRE(f2.get_args()[0] == "x"_var);

        REQUIRE(f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);

        REQUIRE(f.get_args().data() == f2.get_args().data());
        auto f3(f2);
        REQUIRE(f.get_args().data() == f3.get_args().data());
    }

    {
        func_args f{std::make_shared<const std::vector<expression>>(std::vector{"x"_var})};
        REQUIRE(f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);

        func_args f2;
        f2 = std::move(f);
        REQUIRE(f2.get_shared_args());
        REQUIRE(f2.get_args().size() == 1u);
        REQUIRE(f2.get_args()[0] == "x"_var);

        f = std::move(f2);
        REQUIRE(f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);
    }

    REQUIRE_THROWS_MATCHES(func_args(nullptr), std::invalid_argument,
                           Message("Cannot initialise a func_args instance from a null pointer"));
}

TEST_CASE("s11n")
{
    {
        std::stringstream ss;

        func_args f{{"x"_var}};

        {
            boost::archive::binary_oarchive oa(ss);

            oa << f;
        }

        f = func_args{};

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> f;
        }

        REQUIRE(!f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);
    }

    {
        std::stringstream ss;

        func_args f{{"x"_var}, true};

        {
            boost::archive::binary_oarchive oa(ss);

            oa << f;
        }

        f = func_args{};

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> f;
        }

        REQUIRE(f.get_shared_args());
        REQUIRE(f.get_args().size() == 1u);
        REQUIRE(f.get_args()[0] == "x"_var);
    }
}
