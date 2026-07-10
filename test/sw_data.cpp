// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>

#include <fmt/core.h>

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

using namespace heyoka;

TEST_CASE("basic")
{
    sw_data idata;

    REQUIRE(!idata.get_table().empty());
    REQUIRE(!idata.get_timestamp().empty());
    REQUIRE(idata.get_identifier() == "celestrak_long_term");
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

        REQUIRE_THROWS_MATCHES(sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
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
              "68.1\n2020-01-02,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
                               Message("Invalid SW data table detected: the f107 value inf on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,-1,69.4,OBS,71.4,69.7,69.2,"
              "68.1\n2020-01-02,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
                               Message("Invalid SW data table detected: the f107 value -1 on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,3,69.4,OBS,inf,69.7,69.2,"
              "68.1\n2020-01-02,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(
            sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
            Message("Invalid SW data table detected: the f107a_center81 value inf on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,1,69.4,OBS,-2,69.7,69.2,"
              "68.1\n2020-01-02,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(
            sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
            Message("Invalid SW data table detected: the f107a_center81 value -2 on line 0 is invalid"));
    }

    // Invalid Ap_avg values.
    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,inf,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69."
              "2,"
              "68.1\n2020-01-02,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
                               Message("Invalid SW data table detected: the Ap_avg value inf on line 0 is invalid"));
    }

    {
        const std::string str
            = "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
              "F10.7_"
              "OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
              "LAST81\n2020-01-01,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,-1,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1\n2020-01-02,2542,22,3,0,0,7,7,13,10,7,47,2,0,0,3,3,5,4,3,2,0.0,0,6,71.8,69.4,OBS,71.4,69.7,69.2,"
              "68.1";

        REQUIRE_THROWS_MATCHES(sw_data(detail::parse_sw_data_celestrak(str), "ts", "id"), std::invalid_argument,
                               Message("Invalid SW data table detected: the Ap_avg value -1 on line 0 is invalid"));
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

TEST_CASE("reanchor_sw_data_celestrak error handling")
{
    using Catch::Matchers::Message;

    // Fewer than 3 rows.
    {
        sw_data_table tbl{{.mjd = 0.}, {.mjd = 1.}};
        REQUIRE_THROWS_MATCHES(detail::reanchor_sw_data_celestrak(tbl), std::invalid_argument,
                               Message("Invalid CelesTrak SW dataset detected: the minimum number of required "
                                       "rows is 3, but the dataset contains only 2 row(s)"));
    }

    // Non-finite mjd.
    {
        sw_data_table tbl{{.mjd = std::numeric_limits<double>::infinity()}, {.mjd = 1.}, {.mjd = 2.}};
        REQUIRE_THROWS_MATCHES(
            detail::reanchor_sw_data_celestrak(tbl), std::invalid_argument,
            Message("Invalid CelesTrak SW dataset detected: a non-finite mjd was found at row index 0"));
    }

    // Non-integral mjd.
    {
        sw_data_table tbl{{.mjd = 0.}, {.mjd = 1.5}, {.mjd = 2.}};
        REQUIRE_THROWS_MATCHES(
            detail::reanchor_sw_data_celestrak(tbl), std::invalid_argument,
            Message("Invalid CelesTrak SW dataset detected: a non-integral mjd was found at row index 1"));
    }

    // mjd too large in magnitude.
    {
        sw_data_table tbl{{.mjd = 4300000000.}, {.mjd = 4300000001.}, {.mjd = 4300000002.}};
        REQUIRE_THROWS_MATCHES(detail::reanchor_sw_data_celestrak(tbl), std::invalid_argument,
                               Message(fmt::format("Invalid CelesTrak SW dataset detected: the mjd value {} at row "
                                                   "index 0 is too large in magnitude",
                                                   4300000000.)));
    }

    // Non-monotonic mjd.
    {
        sw_data_table tbl{{.mjd = 0.}, {.mjd = 1.}, {.mjd = 1.}};
        REQUIRE_THROWS_MATCHES(detail::reanchor_sw_data_celestrak(tbl), std::invalid_argument,
                               Message("Invalid CelesTrak SW dataset detected: the mjd value at row index 2 is not "
                                       "greater than the mjd value at the previous row index"));
    }

    // Non-consecutive mjd.
    {
        sw_data_table tbl{{.mjd = 0.}, {.mjd = 1.}, {.mjd = 3.}};
        REQUIRE_THROWS_MATCHES(detail::reanchor_sw_data_celestrak(tbl), std::invalid_argument,
                               Message("Invalid CelesTrak SW dataset detected: the mjd value at row index 2 is not "
                                       "1 day larger than the mjd value at the previous row index"));
    }
}

TEST_CASE("reanchor_sw_data_celestrak")
{
    // NOTE: the interpolated f107 is not bit-exact (its 20/24 and 17/24 offsets are not dyadic), hence the numerical
    // tolerance (which has to be relatively large due to the large magnitudes of the mjds). Ap_avg and f107a_center81
    // use the exact 0.5 offset and stay bit-exact.
    const auto f107_close = [](double a, double b) { return std::abs(a - b) <= 1e-9; };

    // A table entirely after 1991-06-01 (f107 anchored at 20h UTC throughout). 4 rows -> 3 re-anchored rows.
    {
        sw_data_table tbl{{.mjd = 50000., .Ap_avg = 10., .f107 = 60., .f107a_center81 = 100.},
                          {.mjd = 50001., .Ap_avg = 20., .f107 = 66., .f107a_center81 = 110.},
                          {.mjd = 50002., .Ap_avg = 30., .f107 = 72., .f107a_center81 = 120.},
                          {.mjd = 50003., .Ap_avg = 40., .f107 = 78., .f107a_center81 = 130.}};

        detail::reanchor_sw_data_celestrak(tbl);

        // 4 rows in -> 3 out, with the first date (50000) dropped.
        REQUIRE(tbl.size() == 3u);
        REQUIRE(tbl[0].mjd == 50001.);
        REQUIRE(tbl[1].mjd == 50002.);
        REQUIRE(tbl[2].mjd == 50003.);

        // Ap_avg and f107a_center81 are anchored at 12h UTC, so the 0h value is exactly the average of the two
        // bracketing days.
        REQUIRE(tbl[0].Ap_avg == 15.);
        REQUIRE(tbl[1].Ap_avg == 25.);
        REQUIRE(tbl[2].Ap_avg == 35.);
        REQUIRE(tbl[0].f107a_center81 == 105.);
        REQUIRE(tbl[1].f107a_center81 == 115.);
        REQUIRE(tbl[2].f107a_center81 == 125.);

        // f107 is anchored at 20h UTC, so the 0h value is prev + (cur - prev) * (1 - 20/24) = prev + (cur - prev)/6.
        REQUIRE(f107_close(tbl[0].f107, 61.));
        REQUIRE(f107_close(tbl[1].f107, 67.));
        REQUIRE(f107_close(tbl[2].f107, 73.));
    }

    // A table straddling the 1991-06-01 f107 measurement-time change (mjd 48408): 17h UT before, 20h UT from then on. 3
    // rows -> 2 re-anchored rows, the first of which (target 48408) is the seam row.
    {
        sw_data_table tbl{{.mjd = 48407., .Ap_avg = 10., .f107 = 0., .f107a_center81 = 100.},
                          {.mjd = 48408., .Ap_avg = 20., .f107 = 27., .f107a_center81 = 110.},
                          {.mjd = 48409., .Ap_avg = 30., .f107 = 33., .f107a_center81 = 120.}};

        detail::reanchor_sw_data_celestrak(tbl);

        REQUIRE(tbl.size() == 2u);
        REQUIRE(tbl[0].mjd == 48408.);
        REQUIRE(tbl[1].mjd == 48409.);

        // Seam row (target 48408): left sample at 48407 + 17/24, right at 48408 + 20/24. The 0h value is f107_prev +
        // (f107_cur - f107_prev) * (1 - 17/24) / (1 + 20/24 - 17/24) = 0 + 27 * (7/24)/(27/24) = 7.
        REQUIRE(f107_close(tbl[0].f107, 7.));
        // Post-seam row (target 48409): both samples at 20h, so prev + (cur - prev)/6 = 27 + 6/6 = 28.
        REQUIRE(f107_close(tbl[1].f107, 28.));

        // Ap_avg / f107a_center81 use the 12h anchor on both sides, unaffected by the seam.
        REQUIRE(tbl[0].Ap_avg == 15.);
        REQUIRE(tbl[1].Ap_avg == 25.);
        REQUIRE(tbl[0].f107a_center81 == 105.);
        REQUIRE(tbl[1].f107a_center81 == 115.);
    }
}

const std::initializer_list<sw_data_row> sample_custom_data
    = {{.mjd = 123., .Ap_avg = 1, .f107 = 2., .f107a_center81 = 3.},
       {.mjd = 124., .Ap_avg = 4, .f107 = 5., .f107a_center81 = 6.}};

TEST_CASE("custom data error handling")
{
    using Catch::Matchers::Message;

    REQUIRE_THROWS_MATCHES(sw_data({}, "", ""), std::invalid_argument,
                           Message("Cannot initialise an SW data table from fewer than 2 rows (0 row(s) detected)"));
    REQUIRE_THROWS_MATCHES(
        sw_data(std::ranges::subrange(sample_custom_data.begin(), sample_custom_data.begin() + 1), "", ""),
        std::invalid_argument,
        Message("Cannot initialise an SW data table from fewer than 2 rows (1 row(s) detected)"));
    REQUIRE_THROWS_MATCHES(sw_data(sample_custom_data, "", ""), std::invalid_argument,
                           Message("Cannot construct an SW data instance with an empty timestamp"));
    REQUIRE_THROWS_MATCHES(sw_data(sample_custom_data, "a", ""), std::invalid_argument,
                           Message("Cannot construct an SW data instance with an empty identifier"));
    REQUIRE_THROWS_MATCHES(
        sw_data(sample_custom_data, "a.", "b"), std::invalid_argument,
        Message("Invalid timestamp 'a.' specified for an SW data instance: the timestamp cannot contain '.' or '-'"));
    REQUIRE_THROWS_MATCHES(
        sw_data(sample_custom_data, "a-", "b"), std::invalid_argument,
        Message("Invalid timestamp 'a-' specified for an SW data instance: the timestamp cannot contain '.' or '-'"));
    REQUIRE_THROWS_MATCHES(
        sw_data(sample_custom_data, "a", "b."), std::invalid_argument,
        Message("Invalid identifier 'b.' specified for an SW data instance: the identifier cannot contain '.' or '-'"));
    REQUIRE_THROWS_MATCHES(
        sw_data(sample_custom_data, "a", "b-"), std::invalid_argument,
        Message("Invalid identifier 'b-' specified for an SW data instance: the identifier cannot contain '.' or '-'"));
    REQUIRE_THROWS_MATCHES(sw_data(sample_custom_data, "a", "celestrak_a"), std::invalid_argument,
                           Message("Invalid identifier 'celestrak_a' specified for an SW data instance: the identifier "
                                   "cannot start with 'celestrak_', this prefix is reserved"));
}

TEST_CASE("custom data")
{
    const auto sdata = sw_data(sample_custom_data, "a", "b");
    REQUIRE(sdata.get_table().size() == 2u);
    REQUIRE(sdata.get_table()[0] == *sample_custom_data.begin());
    REQUIRE(sdata.get_table()[1] == *(sample_custom_data.begin() + 1));
    REQUIRE(sdata.get_timestamp() == "a");
    REQUIRE(sdata.get_identifier() == "b");
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

        // Check name mangling.
        REQUIRE(boost::algorithm::contains(s.get_ir(), "sw_data"));

        // Compile and fetch the function pointer.
        s.compile();
        auto *fptr = reinterpret_cast<const T *(*)()>(s.jit_lookup("test"));

        // Fetch the array pointer.
        const auto *arr_ptr = fptr();

        // Just check the first value against the source table.
        REQUIRE(arr_ptr[0] == static_cast<T>(data.get_table()[0].Ap_avg));
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

        // Just check the first values against the source table.
        REQUIRE(arr_ptr1[0] == static_cast<T>(data.get_table()[0].f107));
        REQUIRE(arr_ptr2[0] == static_cast<T>(data.get_table()[0].f107a_center81));
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

// Test to check the download code.
TEST_CASE("download")
{
    try {
        const auto data = sw_data::fetch_latest_celestrak();
        REQUIRE(!data.get_table().empty());
        REQUIRE(data.get_identifier() == "celestrak_last_5_years");
    } catch (const std::exception &e) {
        std::cout << "Exception caught during download test: " << e.what() << '\n';
    }
}
