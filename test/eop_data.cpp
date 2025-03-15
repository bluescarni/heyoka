// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basic")
{
    eop_data idata;

    REQUIRE(!idata.get_table().empty());
    REQUIRE(!idata.get_timestamp().empty());
    REQUIRE(idata.get_identifier() == "iers_rapid_finals2000A_all");
}

TEST_CASE("parse_eop_data_iers_rapid test")
{
    using Catch::Matchers::Message;

    // Successful parses.
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        const auto data = detail::parse_eop_data_iers_rapid(str);

        REQUIRE(data.size() == 2u);

        REQUIRE(data[0].mjd == 41684);
        REQUIRE(data[0].delta_ut1_utc == .8075000);

        REQUIRE(data[1].mjd == 41685);
        REQUIRE(data[1].delta_ut1_utc == .8044000);
    }

    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.0461909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        const auto data = detail::parse_eop_data_iers_rapid(str);

        REQUIRE(data.size() == 2u);

        REQUIRE(data[0].mjd == 60709);
        REQUIRE(data[0].delta_ut1_utc == 0.0461909);

        REQUIRE(data[1].mjd == 60710);
        REQUIRE(data[1].delta_ut1_utc == 0.0456841);
    }

    // NOTE: newline at the end.
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.0461909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n";

        const auto data = detail::parse_eop_data_iers_rapid(str);

        REQUIRE(data.size() == 1u);

        REQUIRE(data[0].mjd == 60709);
        REQUIRE(data[0].delta_ut1_utc == 0.0461909);
    }

    // NOTE: missing both bulletin A and bulletin B data.
    {
        const std::string str
            = "26 315 61114.00                                                                                         "
              "                                                                                   \n26 316 61115.00    "
              "                                                                                                        "
              "                                                                ";

        const auto data = detail::parse_eop_data_iers_rapid(str);

        REQUIRE(data.empty());
    }

    // Parse errors.

    // Wrong line length.
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.0461909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Invalid line detected in a IERS rapid EOP data file: the expected number of "
                    "characters in the line is at least 185, but a line with 1 character(s) was detected instead"));
    }

    {
        const std::string str
            = "73 1 2\n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Invalid line detected in a IERS rapid EOP data file: the expected number of "
                    "characters in the line is at least 185, but a line with 6 character(s) was detected instead"));
    }

    // Invalid MJDs.
    {
        const std::string str
            = "73 1 2 4a684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
                               Message("Error parsing a IERS rapid EOP data file: the string '4a684.00' could "
                                       "not be parsed as a valid MJD"));
    }
    {
        const std::string str
            = "73 1 2 41684.0  I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
                               Message("Error parsing a IERS rapid EOP data file: the string '41684.0 ' could "
                                       "not be parsed as a valid MJD"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41684.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value 41684 "
                                       "on line 0 is not less than the MJD value in the next line (41684)"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41683.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value 41684 "
                                       "on line 0 is not less than the MJD value in the next line (41683)"));
    }
    {
        const std::string str
            = "73 1 2      inf I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41683.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value inf on line 0 is not finite"));
    }
    {
        const std::string str
            = "73 1 2 41683.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "     inf I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value inf on line 1 is not finite"));
    }

    // UT1-UTC
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .80a5000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Error parsing a IERS rapid EOP data file: the bulletin B string for the UT1-UTC "
                    "difference '.80a5000' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .801500    -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Error parsing a IERS rapid EOP data file: the bulletin B string for the UT1-UTC "
                    "difference '.801500 ' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.04a1909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Error parsing a IERS rapid EOP data file: the bulletin A string for the UT1-UTC "
                    "difference '0.04a1909' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.041190  0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Error parsing a IERS rapid EOP data file: the bulletin A string for the UT1-UTC "
                    "difference '0.041190 ' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I       inf 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Invalid EOP data table detected: the UT1-UTC value inf on line 0 is not finite"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8015000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000        inf   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_iers_rapid(str), std::invalid_argument,
            Message("Invalid EOP data table detected: the UT1-UTC value inf on line 1 is not finite"));
    }

    // A check for wrong filename for the download function.
    REQUIRE_THROWS_MATCHES(
        eop_data::fetch_latest_iers_rapid("helloworld"), std::invalid_argument,
        Message("Invalid filename 'helloworld' specified for a IERS rapid EOP data file: the valid names "
                "are {\"finals2000A.all\", "
                "\"finals2000A.daily\", \"finals2000A.daily.extended\", \"finals2000A.data\"}"));
}

TEST_CASE("s11n")
{
    eop_data idata;

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

TEST_CASE("eop_data_date_tt_cy_j2000")
{
    // Fetch the default eop_data.
    const eop_data data;

    auto tester = [&data]<typename T>() {
        llvm_state s;

        auto &bld = s.builder();
        auto &ctx = s.context();
        auto &md = s.module();

        auto *scal_t = detail::to_external_llvm_type<T>(ctx);

        // Add dummy function that uses the array, returning a pointer to the first element.
        auto *ft = llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx), {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
        bld.CreateRet(detail::llvm_get_eop_data_date_tt_cy_j2000(s, data, scal_t));

        // Compile and fetch the function pointer.
        s.compile();
        auto *fptr = reinterpret_cast<const T *(*)()>(s.jit_lookup("test"));

        // Check manually a few values. These values have been computed with astropy.
        REQUIRE(*fptr() == approximately(static_cast<T>(-0.2699657628640961)));
        REQUIRE(*(fptr() + 6308) == approximately(static_cast<T>(-0.09726213109235177)));
        REQUIRE(*(fptr() + 19429) == approximately(static_cast<T>(0.26197127448982177)));
    };

    tester.operator()<float>();
    tester.operator()<double>();

    // A test with llvm multi state to check proper linkonce behaviour.
    auto multi_tester = [&data]<typename T>() {
        std::vector<llvm_state> vs;

        for (auto i : {1, 2}) {
            llvm_state s;

            auto &bld = s.builder();
            auto &ctx = s.context();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);

            // Add dummy function that uses the array, returning a pointer to the first element.
            auto *ft = llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx), {}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, fmt::format("test_{}", i), &md);
            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
            bld.CreateRet(detail::llvm_get_eop_data_date_tt_cy_j2000(s, data, scal_t));

            vs.push_back(std::move(s));
        }

        llvm_multi_state ms(std::move(vs));

        // Compile and fetch the function pointer.
        ms.compile();
        auto *fptr1 = reinterpret_cast<const T *(*)()>(ms.jit_lookup("test_1"));
        auto *fptr2 = reinterpret_cast<const T *(*)()>(ms.jit_lookup("test_2"));

        // Check manually a few values. These values have been computed with astropy.
        REQUIRE(*fptr1() == approximately(static_cast<T>(-0.2699657628640961)));
        REQUIRE(*(fptr1() + 6308) == approximately(static_cast<T>(-0.09726213109235177)));
        REQUIRE(*(fptr1() + 19429) == approximately(static_cast<T>(0.26197127448982177)));

        REQUIRE(*fptr1() == *fptr2());
        REQUIRE(*(fptr1() + 6308) == *(fptr2() + 6308));
        REQUIRE(*(fptr1() + 19429) == *(fptr2() + 19429));
    };

    multi_tester.operator()<float>();
    multi_tester.operator()<double>();
}

TEST_CASE("eop_data_era")
{
    // Fetch the default eop_data.
    const eop_data data;

    auto tester = [&data]<typename T>() {
        llvm_state s;

        auto &bld = s.builder();
        auto &ctx = s.context();
        auto &md = s.module();

        auto *scal_t = detail::to_external_llvm_type<T>(ctx);

        // Add dummy function that uses the array, returning a pointer to the first element.
        auto *ft = llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx), {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
        bld.CreateRet(detail::llvm_get_eop_data_era(s, data, scal_t));

        // Compile and fetch the function pointer.
        s.compile();
        auto *fptr = reinterpret_cast<const T *(*)()>(s.jit_lookup("test"));

        // Check manually a few values. These values have been computed with astropy.
        // NOTE: these are only approximately true because apparently astropy is using a
        // slightly different dataset by default. Perhaps we can make these checks more precise
        // once we figure out exactly what astropy is using.
        REQUIRE(std::abs(*fptr() - 1.7773390613567774) < 1e-6);
        REQUIRE(std::abs(*(fptr() + 6308) - 3.4744869507397453) < 1e-6);
        REQUIRE(std::abs(*(fptr() + 19429) - 2.989612722143122) < 1e-6);
    };

    tester.operator()<float>();
    tester.operator()<double>();
}
