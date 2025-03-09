// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <limits>
#include <string>

#include <heyoka/model/iers.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("parse test")
{
    using Catch::Matchers::Message;

    // Successful parses.
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        const auto data = model::parse_iers_data(str);

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

        const auto data = model::parse_iers_data(str);

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

        const auto data = model::parse_iers_data(str);

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

        const auto data = model::parse_iers_data(str);

        REQUIRE(data.size() == 2u);

        REQUIRE(data[0].mjd == 61114);
        REQUIRE(std::isnan(data[0].delta_ut1_utc));

        REQUIRE(data[1].mjd == 61115);
        REQUIRE(std::isnan(data[1].delta_ut1_utc));
    }

    // Parse errors.

    // Wrong line length.
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.0461909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Invalid line detected in a finals2000A.all IERS data file: the expected number of "
                    "characters in the line is at least 185, but a line with 1 character(s) was detected instead"));
    }

    {
        const std::string str
            = "73 1 2\n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Invalid line detected in a finals2000A.all IERS data file: the expected number of "
                    "characters in the line is at least 185, but a line with 6 character(s) was detected instead"));
    }

    // Invalid MJDs.
    {
        const std::string str
            = "73 1 2 4a684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(model::parse_iers_data(str), std::invalid_argument,
                               Message("Error parsing a finals2000A.all IERS data file: the string '4a684.00' could "
                                       "not be parsed as a valid MJD"));
    }
    {
        const std::string str
            = "73 1 2 41684.0  I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(model::parse_iers_data(str), std::invalid_argument,
                               Message("Error parsing a finals2000A.all IERS data file: the string '41684.0 ' could "
                                       "not be parsed as a valid MJD"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41684.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(model::parse_iers_data(str), std::invalid_argument,
                               Message("Invalid finals2000A.all IERS data file detected: the MJD value 41684 "
                                       "on line 0 is not less than the MJD value in the next line (41684)"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41683.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(model::parse_iers_data(str), std::invalid_argument,
                               Message("Invalid finals2000A.all IERS data file detected: the MJD value 41684 "
                                       "on line 0 is not less than the MJD value in the next line (41683)"));
    }
    {
        const std::string str
            = "73 1 2      inf I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41683.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Invalid finals2000A.all IERS data file detected: the MJD value inf on line 0 is not finite"));
    }
    {
        const std::string str
            = "73 1 2 41683.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "     inf I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Invalid finals2000A.all IERS data file detected: the MJD value inf on line 1 is not finite"));
    }

    // UT1-UTC
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .80a5000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Error parsing a finals2000A.all IERS data file: the bulletin B string for the UT1-UTC "
                    "difference '.80a5000' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .801500    -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Error parsing a finals2000A.all IERS data file: the bulletin B string for the UT1-UTC "
                    "difference '.801500 ' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.04a1909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Error parsing a finals2000A.all IERS data file: the bulletin A string for the UT1-UTC "
                    "difference '0.04a1909' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.041190  0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Error parsing a finals2000A.all IERS data file: the bulletin A string for the UT1-UTC "
                    "difference '0.041190 ' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I       inf 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Invalid finals2000A.all IERS data file detected: the UT1-UTC value inf on line 0 is an infinity"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8015000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000        inf   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            model::parse_iers_data(str), std::invalid_argument,
            Message("Invalid finals2000A.all IERS data file detected: the UT1-UTC value inf on line 1 is an infinity"));
    }
}

TEST_CASE("set/get iers data")
{
    auto old_data = model::get_iers_data();
    REQUIRE(!old_data->empty());

    model::set_iers_data(*old_data);

    REQUIRE(*model::get_iers_data() == *old_data);

    REQUIRE(model::get_iers_data().get() != old_data.get());
}

TEST_CASE("iers_data_row cmp")
{
    REQUIRE(model::iers_data_row{.mjd = 1} != model::iers_data_row{.mjd = 2});
    REQUIRE(model::iers_data_row{.mjd = 1} == model::iers_data_row{.mjd = 1});

    REQUIRE(model::iers_data_row{.mjd = 1, .delta_ut1_utc = 1.} == model::iers_data_row{.mjd = 1, .delta_ut1_utc = 1.});
    REQUIRE(model::iers_data_row{.mjd = 1, .delta_ut1_utc = 2.} != model::iers_data_row{.mjd = 1, .delta_ut1_utc = 1.});
    REQUIRE(model::iers_data_row{.mjd = 1, .delta_ut1_utc = std::numeric_limits<double>::quiet_NaN()}
            != model::iers_data_row{.mjd = 1, .delta_ut1_utc = 1.});
    REQUIRE(model::iers_data_row{.mjd = 1, .delta_ut1_utc = 1.}
            != model::iers_data_row{.mjd = 1, .delta_ut1_utc = std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(model::iers_data_row{.mjd = 1, .delta_ut1_utc = std::numeric_limits<double>::quiet_NaN()}
            == model::iers_data_row{.mjd = 1, .delta_ut1_utc = std::numeric_limits<double>::quiet_NaN()});
}
