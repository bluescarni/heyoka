// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <variant>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("number ctors")
{
    REQUIRE(std::get<number>(expression{1.1f}.value()) == number{1.1f});
    REQUIRE(std::get<number>(expression{1.1}.value()) == number{1.1});
    REQUIRE(std::get<number>(expression{1.1l}.value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(std::get<number>(expression{mppp::real128{"1.1"}}.value()) == number{mppp::real128{"1.1"}});

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(std::get<number>(expression{mppp::real{"1.1", 123}}.value()) == number{mppp::real{"1.1", 123}});

#endif
}

TEST_CASE("literals")
{
    REQUIRE(1.1_flt == expression{1.1f});
    REQUIRE(1111111111111111111_flt == expression{1111111111111111111.f});

    REQUIRE(1.1_dbl == expression{1.1});
    REQUIRE(1111111111111111111_dbl == expression{1111111111111111111.});

    REQUIRE(1.1_ldbl == expression{1.1l});
    REQUIRE(1111111111111111111_ldbl == expression{1111111111111111111.l});

#if defined(HEYOKA_HAVE_REAL128)

    using namespace mppp::literals;

    REQUIRE(1.1_f128 == expression{1.1_rq});
    REQUIRE(1111111111111111111_f128 == expression{1111111111111111111._rq});

#endif
}

TEST_CASE("number binary ops")
{

    REQUIRE(1_flt + 1.1f == expression{1.f + 1.1f});
    REQUIRE(1.1f + 1_flt == expression{1.f + 1.1f});

    REQUIRE(1_dbl + 1.1 == expression{1. + 1.1});
    REQUIRE(1.1 + 1_dbl == expression{1. + 1.1});

    REQUIRE(1_ldbl + 1.1l == expression{1.l + 1.1l});
    REQUIRE(1.1l + 1_ldbl == expression{1.l + 1.1l});

#if defined(HEYOKA_HAVE_REAL128)

    using namespace mppp::literals;

    REQUIRE(1_f128 + 1.1_rq == expression{1._rq + 1.1_rq});
    REQUIRE(1.1_rq + 1_f128 == expression{1._rq + 1.1_rq});

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(1_dbl + mppp::real{"1.1", 345} == expression{1. + mppp::real{"1.1", 345}});
    REQUIRE(mppp::real{"1.1", 345} + 1_dbl == expression{1. + mppp::real{"1.1", 345}});

#endif

    REQUIRE(1_flt - 1.1f == expression{1.f - 1.1f});
    REQUIRE(1.1f - 1_flt == expression{1.1f - 1.f});

    REQUIRE(1_dbl - 1.1 == expression{1. - 1.1});
    REQUIRE(1.1 - 1_dbl == expression{1.1 - 1.});

    REQUIRE(1_ldbl - 1.1l == expression{1.l - 1.1l});
    REQUIRE(1.1l - 1_ldbl == expression{1.1l - 1.l});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(1_f128 - 1.1_rq == expression{1._rq - 1.1_rq});
    REQUIRE(1.1_rq - 1_f128 == expression{1.1_rq - 1._rq});

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(1_dbl - mppp::real{"1.1", 345} == expression{1. - mppp::real{"1.1", 345}});
    REQUIRE(mppp::real{"1.1", 345} - 1_dbl == expression{mppp::real{"1.1", 345} - 1.});

#endif

    REQUIRE(1_flt * 1.1f == expression{1.f * 1.1f});
    REQUIRE(1.1f * 1_flt == expression{1.1f * 1.f});

    REQUIRE(1_dbl * 1.1 == expression{1. * 1.1});
    REQUIRE(1.1 * 1_dbl == expression{1.1 * 1.});

    REQUIRE(1_ldbl * 1.1l == expression{1.l * 1.1l});
    REQUIRE(1.1l * 1_ldbl == expression{1.1l * 1.l});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(1_f128 * 1.1_rq == expression{1._rq * 1.1_rq});
    REQUIRE(1.1_rq * 1_f128 == expression{1.1_rq * 1._rq});

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(1_dbl * mppp::real{"1.1", 345} == expression{1. * mppp::real{"1.1", 345}});
    REQUIRE(mppp::real{"1.1", 345} * 1_dbl == expression{mppp::real{"1.1", 345} * 1.});

#endif

    REQUIRE(1_flt / 1.1f == expression{1.f / 1.1f});
    REQUIRE(1.1f / 1_flt == expression{1.1f / 1.f});

    REQUIRE(1_dbl / 1.1 == expression{1. / 1.1});
    REQUIRE(1.1 / 1_dbl == expression{1.1 / 1.});

    REQUIRE(1_ldbl / 1.1l == expression{1.l / 1.1l});
    REQUIRE(1.1l / 1_ldbl == expression{1.1l / 1.l});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(1_f128 / 1.1_rq == expression{1._rq / 1.1_rq});
    REQUIRE(1.1_rq / 1_f128 == expression{1.1_rq / 1._rq});

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(1_dbl / mppp::real{"1.1", 345} == expression{1. / mppp::real{"1.1", 345}});
    REQUIRE(mppp::real{"1.1", 345} / 1_dbl == expression{mppp::real{"1.1", 345} / 1.});

#endif
}

TEST_CASE("number compound ops")
{
    {
        auto ex = 1_flt;
        ex += 1.1f;
        REQUIRE(ex == 1_flt + 1.1f);
    }

    {
        auto ex = 1_dbl;
        ex += 1.1;
        REQUIRE(ex == 1_dbl + 1.1);
    }

    {
        auto ex = 1_ldbl;
        ex += 1.1l;
        REQUIRE(ex == 1_ldbl + 1.1l);
    }

#if defined(HEYOKA_HAVE_REAL128)

    {
        using namespace mppp::literals;

        auto ex = 1_f128;
        ex += 1.1_rq;
        REQUIRE(ex == 1_f128 + 1.1_rq);
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    {
        auto ex = 1_dbl;
        ex += mppp::real{"1.1", 345};
        REQUIRE(ex == 1_dbl + mppp::real{"1.1", 345});
    }

#endif

    {
        auto ex = 1_flt;
        ex -= 1.1f;
        REQUIRE(ex == 1_flt - 1.1f);
    }

    {
        auto ex = 1_dbl;
        ex -= 1.1;
        REQUIRE(ex == 1_dbl - 1.1);
    }

    {
        auto ex = 1_ldbl;
        ex -= 1.1l;
        REQUIRE(ex == 1_ldbl - 1.1l);
    }

#if defined(HEYOKA_HAVE_REAL128)

    {
        using namespace mppp::literals;

        auto ex = 1_f128;
        ex -= 1.1_rq;
        REQUIRE(ex == 1_f128 - 1.1_rq);
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    {
        auto ex = 1_dbl;
        ex -= mppp::real{"1.1", 345};
        REQUIRE(ex == 1_dbl - mppp::real{"1.1", 345});
    }

#endif

    {
        auto ex = 1_flt;
        ex *= 1.1f;
        REQUIRE(ex == 1_flt * 1.1f);
    }

    {
        auto ex = 1_dbl;
        ex *= 1.1;
        REQUIRE(ex == 1_dbl * 1.1);
    }

    {
        auto ex = 1_ldbl;
        ex *= 1.1l;
        REQUIRE(ex == 1_ldbl * 1.1l);
    }

#if defined(HEYOKA_HAVE_REAL128)

    {
        using namespace mppp::literals;

        auto ex = 1_f128;
        ex *= 1.1_rq;
        REQUIRE(ex == 1_f128 * 1.1_rq);
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    {
        auto ex = 1_dbl;
        ex *= mppp::real{"1.1", 345};
        REQUIRE(ex == 1_dbl * mppp::real{"1.1", 345});
    }

#endif

    {
        auto ex = 1_flt;
        ex /= 1.1f;
        REQUIRE(ex == 1_flt / 1.1f);
    }

    {
        auto ex = 1_dbl;
        ex /= 1.1;
        REQUIRE(ex == 1_dbl / 1.1);
    }

    {
        auto ex = 1_ldbl;
        ex /= 1.1l;
        REQUIRE(ex == 1_ldbl / 1.1l);
    }

#if defined(HEYOKA_HAVE_REAL128)

    {
        using namespace mppp::literals;

        auto ex = 1_f128;
        ex /= 1.1_rq;
        REQUIRE(ex == 1_f128 / 1.1_rq);
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    {
        auto ex = 1_dbl;
        ex /= mppp::real{"1.1", 345};
        REQUIRE(ex == 1_dbl / mppp::real{"1.1", 345});
    }

#endif
}
