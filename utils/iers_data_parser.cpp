// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>

#include <fmt/core.h>

#include <heyoka/iers_data.hpp>

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string input_file_path;
    std::string timestamp;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("input", po::value<std::string>(&input_file_path)->required(),
                                                       "path to the IERS data file 'finals2000A.all'")(
        "timestamp", po::value<std::string>(&timestamp)->required(),
        "UTC timestamp for the IERS data file 'finals2000A.all' (YYYY_MM_DD)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    po::notify(vm);

    // Validate the timestamp format.
    const std::regex ts_pattern(R"(\d{4}_\d{2}_\d{2})");
    if (!std::regex_match(timestamp, ts_pattern)) {
        throw std::invalid_argument(
            fmt::format("The string '{}' is not a valid timestamp in the YYYY_MM_DD format", timestamp));
    }

    // Read the contents of the file into a string.
    std::ifstream ifile(input_file_path);
    const std::string file_data{std::istreambuf_iterator<char>(ifile), std::istreambuf_iterator<char>()};

    // Parse it.
    const auto iers_data = heyoka::detail::parse_iers_data(file_data);

    // Create the header file first.
    std::ofstream oheader("builtin_iers_data.hpp");
    oheader << fmt::format(R"(#ifndef HEYOKA_DETAIL_IERS_BUILTIN_IERS_DATA_HPP
#define HEYOKA_DETAIL_IERS_BUILTIN_IERS_DATA_HPP

#include <heyoka/config.hpp>
#include <heyoka/iers_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{{

extern const char *const builtin_iers_data_ts;

extern const iers_row builtin_iers_data[{}];

}} // namespace detail

HEYOKA_END_NAMESPACE

#endif
)",
                           iers_data.size());

    // Now the cpp file.
    std::ofstream ocpp("builtin_iers_data.cpp");
    ocpp << fmt::format(R"(#include <limits>

#include <heyoka/config.hpp>
#include <heyoka/detail/iers/builtin_iers_data.hpp>
#include <heyoka/iers_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{{

const char *const builtin_iers_data_ts = "{}";

constinit const iers_row builtin_iers_data[{}] = {{)",
                        timestamp, iers_data.size());

    for (const auto &[mjd, cur_delta_ut1_utc] : iers_data) {
        if (std::isnan(cur_delta_ut1_utc)) {
            ocpp << fmt::format("{{.mjd={}, .delta_ut1_utc=std::numeric_limits<double>::quiet_NaN()}},\n", mjd);
        } else {
            ocpp << fmt::format("{{.mjd={}, .delta_ut1_utc={}}},\n", mjd, cur_delta_ut1_utc);
        }
    }

    ocpp << R"(};

} // namespace detail

HEYOKA_END_NAMESPACE
)";
}
