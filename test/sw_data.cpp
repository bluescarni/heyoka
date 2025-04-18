// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <initializer_list>
// #include <limits>
// #include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
// #include <vector>

// #include <boost/math/constants/constants.hpp>
// #include <boost/multiprecision/cpp_bin_float.hpp>

// #include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>

#include "catch.hpp"
// #include "test_utils.hpp"

using namespace heyoka;
// using namespace heyoka_test;

// static std::mt19937 rng;

// constexpr auto ntrials = 10000;

TEST_CASE("basic")
{
    sw_data idata;

    REQUIRE(!idata.get_table().empty());
    REQUIRE(!idata.get_timestamp().empty());
    REQUIRE(idata.get_identifier() == "celestrak_last_5_years");
}

TEST_CASE("parse_sw_data_celestrak test")
{
    using Catch::Matchers::Message;

    // Successful parses.
    {
        const std::string str = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,"
                                "AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_"
                                "LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.empty());
    }

    // With newline at the end.
    {
        const std::string str = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,"
                                "AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_"
                                "LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.empty());
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,68.1";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.size() == 1u);

        REQUIRE(data[0].mjd == 58849);
        REQUIRE(data[0].Ap_avg == 2);
        REQUIRE(data[0].f107 == 71.8);
        REQUIRE(data[0].f107a_center81 == 71.4);
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,68.1\n";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.size() == 1u);

        REQUIRE(data[0].mjd == 58849);
        REQUIRE(data[0].Ap_avg == 2);
        REQUIRE(data[0].f107 == 71.8);
        REQUIRE(data[0].f107a_center81 == 71.4);
    }

    // Non-monotonic dates.
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,68."
              "1";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Invalid SW data table detected: the MJD value 58849 "
                                       "on line 0 is not less than the MJD value in the next line (58849)"));
    }

    // Invalid flux values.
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,inf,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Invalid SW data table detected: the f107 value inf on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,-1,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Invalid SW data table detected: the f107 value -1 on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,3,69.4,OBS,inf,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(
            detail::parse_sw_data_celestrak(str), std::invalid_argument,
            Message("Invalid SW data table detected: the f107a_center81 value inf on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,1,69.4,OBS,-2,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(
            detail::parse_sw_data_celestrak(str), std::invalid_argument,
            Message("Invalid SW data table detected: the f107a_center81 value -2 on line 0 is invalid"));
    }

    // Unparsable data.
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,faf,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string 'faf' could "
                                       "not be parsed as a valid numerical value"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,fuf,69.4,OBS,71.4,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string 'fuf' could "
                                       "not be parsed as a valid numerical value"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,fef,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string 'fef' could "
                                       "not be parsed as a valid numerical value"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01-0,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,fef,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string '2020-01-0' could "
                                       "not be parsed as a valid ISO 8601 date"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-"
              "01a01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,fef,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string '2020-01a01' could "
                                       "not be parsed as a valid ISO 8601 date"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n202a-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,fef,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string '202a' could "
                                       "not be parsed as a valid year/month/day value"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "a1-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,fef,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string 'a1' could "
                                       "not be parsed as a valid year/month/day value"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "01-a2,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,fef,69.7,69.2,68.1\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: the string 'a2' could "
                                       "not be parsed as a valid year/month/day value"));
    }

    // Handling of missing values.
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,68.1\n";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.empty());
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,,69.4,OBS,71.4,69.7,69.2,68.1\n";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.empty());
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,,69.7,69.2,68.1\n";

        const auto data = detail::parse_sw_data_celestrak(str);

        REQUIRE(data.empty());
    }

    // Wrong number of fields.
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: at least 28 fields "
                                       "were expected in a data row, but 27 were found instead"));
    }
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2022-"
              "01-01,2542,22,3,0,0,7,7,13,10,7,47\n";

        REQUIRE_THROWS_MATCHES(detail::parse_sw_data_celestrak(str), std::invalid_argument,
                               Message("Error parsing a celestrak SW data file: at least 28 fields "
                                       "were expected in a data row, but 12 were found instead"));
    }
}

TEST_CASE("s11n")
{
    sw_data idata;

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);
        oa << idata;
    }

    // Save the original table/timestamp/identifier.
    const auto otable = idata.get_table();
    const auto otimestamep = idata.get_timestamp();
    const auto oidentifier = idata.get_identifier();

    // Reset via move.
    auto idat2 = std::move(idata);

    {
        boost::archive::binary_iarchive ia(ss);
        ia >> idata;
    }

    REQUIRE(otable == idata.get_table());
    REQUIRE(otimestamep == idata.get_timestamp());
    REQUIRE(oidentifier == idata.get_identifier());
}

TEST_CASE("sw_data_Ap_avg")
{
    // Fetch the default sw_data.
    const sw_data data;

    auto tester = [&data]<typename T>() {
        llvm_state s;

        auto &bld = s.builder();
        auto &ctx = s.context();
        auto &md = s.module();

        auto *scal_t = detail::to_external_llvm_type<T>(ctx);

        // Add dummy functions that use the arrays, returning pointers to the first elements.
        auto *ft = llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx), {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
        bld.CreateRet(detail::llvm_get_sw_data_Ap_avg(s, data, scal_t));

        // Compile and fetch the function pointer.
        s.compile();
        auto *fptr = reinterpret_cast<const T *(*)()>(s.jit_lookup("test"));

        // Fetch the array pointer.
        const auto *arr_ptr = fptr();

        // Just check the first value.
        REQUIRE(arr_ptr[0] == 2);
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

TEST_CASE("sw_data_f107")
{
    // Fetch the default sw_data.
    const sw_data data;

    auto tester = [&data]<typename T>() {
        llvm_state s;

        auto &bld = s.builder();
        auto &ctx = s.context();
        auto &md = s.module();

        auto *scal_t = detail::to_external_llvm_type<T>(ctx);

        // Add dummy functions that use the arrays, returning pointers to the first elements.
        auto *ft = llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx), {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
        bld.CreateRet(detail::llvm_get_sw_data_f107(s, data, scal_t));

        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test2", &md);
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
        bld.CreateRet(detail::llvm_get_sw_data_f107a_center81(s, data, scal_t));

        // Compile and fetch the function pointers.
        s.compile();
        auto *fptr1 = reinterpret_cast<const T *(*)()>(s.jit_lookup("test"));
        auto *fptr2 = reinterpret_cast<const T *(*)()>(s.jit_lookup("test2"));

        // Fetch the array pointers.
        const auto *arr_ptr1 = fptr1();
        const auto *arr_ptr2 = fptr2();

        // Just check the first values.
        REQUIRE(arr_ptr1[0] == static_cast<T>(71.8));
        REQUIRE(arr_ptr2[0] == static_cast<T>(71.4));
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

// Test to check the download code.
TEST_CASE("download")
{
    const auto data = sw_data::fetch_latest_celestrak();
    REQUIRE(!data.get_table().empty());
    REQUIRE(data.get_identifier() == "celestrak_last_5_years");
}

#if 0
TEST_CASE("eop_data upper_bound")
{
    auto tester = []<typename T>() {
        using fptr_t = void (*)(std::uint32_t *, const T *, std::uint32_t, const T *) noexcept;

        // Helper to add a test function that invokes the upper_bound implementation
        // and returns the result.
        auto add_test_func = [](llvm_state &s, std::uint32_t batch_size) {
            auto &ctx = s.context();
            auto &bld = s.builder();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);
            auto *ptr_t = llvm::PointerType::getUnqual(ctx);

            auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, bld.getInt32Ty(), ptr_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

            auto *out_ptr = f->getArg(0);
            auto *arr_ptr = f->getArg(1);
            auto *arr_size = f->getArg(2);
            auto *date_ptr = f->getArg(3);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            // Load the date value.
            auto *date = detail::ext_load_vector_from_memory(s, scal_t, date_ptr, batch_size);

            // Run the binary search.
            auto *ret = detail::llvm_eop_data_upper_bound(s, arr_ptr, arr_size, date);

            // Write the result to out_ptr.
            detail::ext_store_vector_to_memory(s, out_ptr, ret);

            bld.CreateRetVoid();
        };

        // Empty range.
        {
            llvm_state s;
            add_test_func(s, 1);

            s.compile();
            auto *fptr = reinterpret_cast<fptr_t>(s.jit_lookup("test"));

            std::uint32_t ret = 42;
            T date = 123;

            fptr(&ret, nullptr, 0, &date);
            REQUIRE(ret == 0u);
        }

        // Empty range batch.
        {
            llvm_state s;
            add_test_func(s, 2);

            s.compile();
            auto *fptr = reinterpret_cast<fptr_t>(s.jit_lookup("test"));

            std::uint32_t ret[] = {42, 43};
            T date[] = {123, 42};

            fptr(ret, nullptr, 0, date);
            REQUIRE(ret[0] == 0u);
            REQUIRE(ret[1] == 0u);
        }

        // Comprehensive randomised scalar testing.
        for (const auto batch_size : {1u, 2u, 4u, 5u, 8u}) {
            // Setup the compiled function.
            llvm_state s;
            add_test_func(s, batch_size);
            s.compile();
            auto *fptr = reinterpret_cast<fptr_t>(s.jit_lookup("test"));

            // Prepare the vector of input dates.
            std::vector<T> date_vector;
            date_vector.resize(batch_size);

            // Prepare the vector of output values.
            std::vector<std::uint32_t> outs;
            outs.resize(batch_size);

            // Prepare the array of sorted dates.
            std::vector<T> arr;
            std::uniform_int_distribution<unsigned> sdist(0, 20);

            // Distribution to generate random dates.
            std::uniform_real_distribution<T> rdist(0, 10.);

            // Distribution triggering with low probability.
            std::uniform_int_distribution<unsigned> lp_dist(0, 100);

            for (auto i = 0; i < ntrials; ++i) {
                // Generate random dates and sort them.
                arr.resize(sdist(rng));
                std::ranges::generate(arr, [&rdist]() { return rdist(rng); });
                std::ranges::sort(arr);

                // Generate random input dates.
                std::ranges::generate(date_vector, [&rdist, &lp_dist, &arr]() {
                    // Generate a nan with low probability.
                    if (lp_dist(rng) == 0u) {
                        return std::numeric_limits<T>::quiet_NaN();
                    }

                    // If arr is not empty, with low probability pick a random
                    // value within it so that we have dates in date_vector
                    // equal to some dates in arr.
                    if (!arr.empty() && lp_dist(rng) == 0u) {
                        std::uniform_int_distribution<unsigned> adist(0, static_cast<unsigned>(arr.size() - 1u));
                        return arr[adist(rng)];
                    }

                    return rdist(rng);
                });

                // Invoke the jitted function.
                fptr(outs.data(), arr.data(), static_cast<std::uint32_t>(arr.size()), date_vector.data());

                // For each date in the batch, run upper_bound and compare the result.
                for (auto j = 0u; j < batch_size; ++j) {
                    const auto it = std::ranges::upper_bound(arr, date_vector[j], [](auto a, auto b) {
                        if (std::isnan(a)) {
                            return false;
                        }

                        return std::isnan(b) ? true : (a < b);
                    });

                    REQUIRE(it - arr.begin() == outs[j]);
                }
            }
        }
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

TEST_CASE("eop_data locate_date")
{
    auto tester = []<typename T>() {
        using fptr_t = void (*)(std::uint32_t *, const T *, std::uint32_t, const T *) noexcept;

        // Helper to add a test function that invokes the locate_date implementation
        // and returns the result.
        auto add_test_func = [](llvm_state &s, std::uint32_t batch_size) {
            auto &ctx = s.context();
            auto &bld = s.builder();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);
            auto *ptr_t = llvm::PointerType::getUnqual(ctx);

            auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, bld.getInt32Ty(), ptr_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

            auto *out_ptr = f->getArg(0);
            auto *arr_ptr = f->getArg(1);
            auto *arr_size = f->getArg(2);
            auto *date_ptr = f->getArg(3);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            // Load the date value.
            auto *date = detail::ext_load_vector_from_memory(s, scal_t, date_ptr, batch_size);

            // Run the binary search.
            auto *ret = detail::llvm_eop_data_locate_date(s, arr_ptr, arr_size, date);

            // Write the result to out_ptr.
            detail::ext_store_vector_to_memory(s, out_ptr, ret);

            bld.CreateRetVoid();
        };

        // Empty range.
        {
            llvm_state s;
            add_test_func(s, 1);

            s.compile();
            auto *fptr = reinterpret_cast<fptr_t>(s.jit_lookup("test"));

            std::uint32_t ret = 42;
            T date = 123;

            fptr(&ret, nullptr, 0, &date);
            REQUIRE(ret == 0u);
        }

        // Empty range batch.
        {
            llvm_state s;
            add_test_func(s, 2);

            s.compile();
            auto *fptr = reinterpret_cast<fptr_t>(s.jit_lookup("test"));

            std::uint32_t ret[] = {42, 43};
            T date[] = {123, 42};

            fptr(ret, nullptr, 0, date);
            REQUIRE(ret[0] == 0u);
            REQUIRE(ret[1] == 0u);
        }

        // Comprehensive randomised scalar testing.
        for (const auto batch_size : {1u, 2u, 4u, 5u, 8u}) {
            // Setup the compiled function.
            llvm_state s;
            add_test_func(s, batch_size);
            s.compile();
            auto *fptr = reinterpret_cast<fptr_t>(s.jit_lookup("test"));

            // Prepare the vector of input dates.
            std::vector<T> date_vector;
            date_vector.resize(batch_size);

            // Prepare the vector of output values.
            std::vector<std::uint32_t> outs;
            outs.resize(batch_size);

            // Prepare the array of sorted dates.
            std::vector<T> arr;
            std::uniform_int_distribution<unsigned> sdist(0, 20);

            // Distribution to generate random dates.
            std::uniform_real_distribution<T> rdist(0, 10.);

            // Distribution triggering with low probability.
            std::uniform_int_distribution<unsigned> lp_dist(0, 100);

            for (auto i = 0; i < ntrials; ++i) {
                // Generate random dates and sort them.
                arr.resize(sdist(rng));
                std::ranges::generate(arr, [&rdist]() { return rdist(rng); });
                std::ranges::sort(arr);

                // Generate random input dates.
                std::ranges::generate(date_vector, [&rdist, &lp_dist, &arr]() {
                    // Generate a nan with low probability.
                    if (lp_dist(rng) == 0u) {
                        return std::numeric_limits<T>::quiet_NaN();
                    }

                    // If arr is not empty, with low probability pick a random
                    // value within it so that we have dates in date_vector
                    // equal to some dates in arr.
                    if (!arr.empty() && lp_dist(rng) == 0u) {
                        std::uniform_int_distribution<unsigned> adist(0, static_cast<unsigned>(arr.size() - 1u));
                        return arr[adist(rng)];
                    }

                    return rdist(rng);
                });

                // Invoke the jitted function.
                fptr(outs.data(), arr.data(), static_cast<std::uint32_t>(arr.size()), date_vector.data());

                // For each date in the batch, check that the result is within the correct interval.
                for (auto j = 0u; j < batch_size; ++j) {
                    const auto cur_out = outs[j];
                    const auto cur_date = date_vector[j];

                    if (cur_out == arr.size()) {
                        // NOTE: these are all the cases in which the function returns arr_size.
                        REQUIRE(
                            (arr.size() < 2u || std::isnan(cur_date) || cur_date < arr[0] || cur_date >= arr.back()));
                    } else {
                        REQUIRE(arr.size() >= 2u);
                        REQUIRE(cur_out < arr.size() - 1u);
                        REQUIRE(cur_date >= arr[cur_out]);
                        REQUIRE(cur_date < arr[cur_out + 1u]);
                    }
                }
            }
        }
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

#endif