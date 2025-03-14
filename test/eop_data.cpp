// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <string>
#include <utility>

#include <heyoka/eop_data.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    eop_data idata;

    REQUIRE(!idata.get_table().empty());
    REQUIRE(!idata.get_timestamp().empty());
    REQUIRE(idata.get_identifier() == "usno_finals2000A_all");
}

TEST_CASE("parse_eop_data_usno test")
{
    using Catch::Matchers::Message;

    // Successful parses.
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        const auto data = detail::parse_eop_data_usno(str);

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

        const auto data = detail::parse_eop_data_usno(str);

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

        const auto data = detail::parse_eop_data_usno(str);

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

        const auto data = detail::parse_eop_data_usno(str);

        REQUIRE(data.empty());
    }

    // Parse errors.

    // Wrong line length.
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.0461909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_usno(str), std::invalid_argument,
            Message("Invalid line detected in a USNO EOP data file: the expected number of "
                    "characters in the line is at least 185, but a line with 1 character(s) was detected instead"));
    }

    {
        const std::string str
            = "73 1 2\n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_usno(str), std::invalid_argument,
            Message("Invalid line detected in a USNO EOP data file: the expected number of "
                    "characters in the line is at least 185, but a line with 6 character(s) was detected instead"));
    }

    // Invalid MJDs.
    {
        const std::string str
            = "73 1 2 4a684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Error parsing a USNO EOP data file: the string '4a684.00' could "
                                       "not be parsed as a valid MJD"));
    }
    {
        const std::string str
            = "73 1 2 41684.0  I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Error parsing a USNO EOP data file: the string '41684.0 ' could "
                                       "not be parsed as a valid MJD"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41684.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value 41684 "
                                       "on line 0 is not less than the MJD value in the next line (41684)"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41683.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value 41684 "
                                       "on line 0 is not less than the MJD value in the next line (41683)"));
    }
    {
        const std::string str
            = "73 1 2      inf I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "41683.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value inf on line 0 is not finite"));
    }
    {
        const std::string str
            = "73 1 2 41683.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n73 1 3 "
              "     inf I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Invalid EOP data table detected: the MJD value inf on line 1 is not finite"));
    }

    // UT1-UTC
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .80a5000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Error parsing a USNO EOP data file: the bulletin B string for the UT1-UTC "
                                       "difference '.80a5000' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .801500    -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000   .8044000   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Error parsing a USNO EOP data file: the bulletin B string for the UT1-UTC "
                                       "difference '.801500 ' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.04a1909 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Error parsing a USNO EOP data file: the bulletin A string for the UT1-UTC "
                                       "difference '0.04a1909' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I 0.041190  0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(detail::parse_eop_data_usno(str), std::invalid_argument,
                               Message("Error parsing a USNO EOP data file: the bulletin A string for the UT1-UTC "
                                       "difference '0.041190 ' could not be parsed as a floating-point value"));
    }
    {
        const std::string str
            = "25 2 3 60709.00 I  0.099700 0.000012  0.309126 0.000014  I       inf 0.0000082  0.5842 0.0078  I     "
              "0.383    0.375    -0.034    0.114                                                     \n25 2 4 60710.00 "
              "I  0.097391 0.000008  0.309488 0.000011  I 0.0456841 0.0000132  0.4075 0.0064  I     0.391    0.319    "
              "-0.022    0.109                                                     ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_usno(str), std::invalid_argument,
            Message("Invalid EOP data table detected: the UT1-UTC value inf on line 0 is not finite"));
    }
    {
        const std::string str
            = "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    "
              "-0.766    0.199    -0.720    0.300   .143000   .137000   .8015000   -18.637    -3.667  \n73 1 3 "
              "41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 0.1916  P    -0.751    "
              "0.199    -0.701    0.300   .141000   .134000        inf   -18.636    -3.571  ";

        REQUIRE_THROWS_MATCHES(
            detail::parse_eop_data_usno(str), std::invalid_argument,
            Message("Invalid EOP data table detected: the UT1-UTC value inf on line 1 is not finite"));
    }

    // A check for wrong filename for the download function.
    REQUIRE_THROWS_MATCHES(eop_data::fetch_latest_usno("helloworld"), std::invalid_argument,
                           Message("Invalid filename 'helloworld' specified for a USNO EOP data file: the valid names "
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
