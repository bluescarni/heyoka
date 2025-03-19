// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
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

static std::mt19937 rng;

constexpr auto ntrials = 10000;

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

TEST_CASE("parse_eop_data_iers_long_term test")
{
    using Catch::Matchers::Message;

    // Successful parses.
    {
        const std::string str
            = "MJD;Year;Month;Day;Type;x_pole;sigma_x_pole;y_pole;sigma_y_pole;x_rate;sigma_x_rate;y_rate;sigma_y_rate;"
              "Type;UT1-UTC;sigma_UT1-UTC;LOD;sigma_LOD;Type;dPsi;sigma_dPsi;dEpsilon;sigma_dEpsilon;dX;sigma_dX;dY;"
              "sigma_dY\n37665;1962;01;01;;-0.012700;0.030000;0.213000;0.030000;0.000000;0.000000;0.000000;0.000000;;0."
              "0326338;0.0020000;0.0017230;0.0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n37666;1962;01;02;;-0."
              "015900;0.030000;0.214100;0.030000;0.000000;0.000000;0.000000;0.000000;;0.0320547;0.0020000;0.0016690;0."
              "0014000;;;;;;0.000000;0.004774;0.000000;0.002000";

        const auto data = detail::parse_eop_data_iers_long_term(str);

        REQUIRE(data.size() == 2u);

        REQUIRE(data[0].mjd == 37665);
        REQUIRE(data[0].delta_ut1_utc == 0.0326338);

        REQUIRE(data[1].mjd == 37666);
        REQUIRE(data[1].delta_ut1_utc == 0.0320547);
    }

    // NOTE: newline at the end.
    {
        const std::string str
            = "MJD;Year;Month;Day;Type;x_pole;sigma_x_pole;y_pole;sigma_y_pole;x_rate;sigma_x_rate;y_rate;sigma_y_rate;"
              "Type;UT1-UTC;sigma_UT1-UTC;LOD;sigma_LOD;Type;dPsi;sigma_dPsi;dEpsilon;sigma_dEpsilon;dX;sigma_dX;dY;"
              "sigma_dY\n37665;1962;01;01;;-0.012700;0.030000;0.213000;0.030000;0.000000;0.000000;0.000000;0.000000;;0."
              "0326338;0.0020000;0.0017230;0.0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n37666;1962;01;02;;-0."
              "015900;0.030000;0.214100;0.030000;0.000000;0.000000;0.000000;0.000000;;0.0320547;0.0020000;0.0016690;0."
              "0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n";

        const auto data = detail::parse_eop_data_iers_long_term(str);

        REQUIRE(data.size() == 2u);

        REQUIRE(data[0].mjd == 37665);
        REQUIRE(data[0].delta_ut1_utc == 0.0326338);

        REQUIRE(data[1].mjd == 37666);
        REQUIRE(data[1].delta_ut1_utc == 0.0320547);
    }

    // Invalid mjd.
    {
        const std::string str
            = "MJD;Year;Month;Day;Type;x_pole;sigma_x_pole;y_pole;sigma_y_pole;x_rate;sigma_x_rate;y_rate;sigma_y_rate;"
              "Type;UT1-UTC;sigma_UT1-UTC;LOD;sigma_LOD;Type;dPsi;sigma_dPsi;dEpsilon;sigma_dEpsilon;dX;sigma_dX;dY;"
              "sigma_dY\nhelloworld;1962;01;01;;-0.012700;0.030000;0.213000;0.030000;0.000000;0.000000;0.000000;0."
              "000000;;0."
              "0326338;0.0020000;0.0017230;0.0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n37666;1962;01;02;;-0."
              "015900;0.030000;0.214100;0.030000;0.000000;0.000000;0.000000;0.000000;;0.0320547;0.0020000;0.0016690;0."
              "0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_long_term(str), std::invalid_argument,
                               Message("Error parsing a IERS long term EOP data file: the string 'helloworld' could "
                                       "not be parsed as a valid double-precision value"));
    }

    // Invalid ut1-utc value.
    {
        const std::string str
            = "MJD;Year;Month;Day;Type;x_pole;sigma_x_pole;y_pole;sigma_y_pole;x_rate;sigma_x_rate;y_rate;sigma_y_rate;"
              "Type;UT1-UTC;sigma_UT1-UTC;LOD;sigma_LOD;Type;dPsi;sigma_dPsi;dEpsilon;sigma_dEpsilon;dX;sigma_dX;dY;"
              "sigma_dY\n37665;1962;01;01;;-0.012700;0.030000;0.213000;0.030000;0.000000;0.000000;0.000000;0."
              "000000;;0."
              "0326338;0.0020000;0.0017230;0.0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n37666;1962;01;02;;-0."
              "015900;0.030000;0.214100;0.030000;0.000000;0.000000;0.000000;0.000000;;goofy;0.0020000;0.0016690;0."
              "0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_long_term(str), std::invalid_argument,
                               Message("Error parsing a IERS long term EOP data file: the string 'goofy' could "
                                       "not be parsed as a valid double-precision value"));
    }

    // Invalid number of fields.
    {
        const std::string str
            = "MJD;Year;Month;Day;Type;x_pole;sigma_x_pole;y_pole;sigma_y_pole;x_rate;sigma_x_rate;y_rate;sigma_y_rate;"
              "Type;UT1-UTC;sigma_UT1-UTC;LOD;sigma_LOD;Type;dPsi;sigma_dPsi;dEpsilon;sigma_dEpsilon;dX;sigma_dX;dY;"
              "sigma_dY\n37665;1962;01;01;;-0.012700;0.030000;0.213000;0.030000;0.000000;0.000000;0.000000;0."
              "000000;;0."
              "0326338;0.0020000;0.0017230;0.0014000;;;;;;0.000000;0.004774;0.000000;0.002000\n37666;1962;01\n";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_iers_long_term(str), std::invalid_argument,
                               Message("Error parsing a IERS long term EOP data file: at least 15 fields "
                                       "were expected in a data row, but 3 were found instead"));
    }
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

        // Add a second function to test that we do not generate the data twice.
        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test2", &md);
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

// Test to check the download code. We pick a small file for testing.
TEST_CASE("download finals2000A.daily")
{
    const auto data = eop_data::fetch_latest_iers_rapid("finals2000A.daily");
    REQUIRE(!data.get_table().empty());
    REQUIRE(data.get_identifier() == "iers_rapid_finals2000A_daily");
}

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
