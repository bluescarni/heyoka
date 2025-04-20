// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>

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

        // Check name mangling.
        REQUIRE(boost::algorithm::contains(s.get_ir(), "sw_data"));

        // Compile and fetch the function pointer.
        s.compile();
        auto *fptr = reinterpret_cast<const T *(*)()>(s.jit_lookup("test"));

        // Fetch the array pointer.
        const auto *arr_ptr = fptr();

        // Just check the first value.
        REQUIRE(arr_ptr[0] == 21);
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
        REQUIRE(arr_ptr1[0] == static_cast<T>(269.3));
        REQUIRE(arr_ptr2[0] == static_cast<T>(266.6));
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
