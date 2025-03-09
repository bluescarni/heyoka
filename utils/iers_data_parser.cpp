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
#include <string>

#include <boost/program_options.hpp>

#include <fmt/core.h>

#include <heyoka/model/iers.hpp>

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string input_file_path;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("input", po::value<std::string>(&input_file_path)->required(),
                                                       "path to the IERS data file 'finals2000A.all'");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    po::notify(vm);

    // Read the contents of the file into a string.
    std::ifstream ifile(input_file_path);
    const std::string file_data{std::istreambuf_iterator<char>(ifile), std::istreambuf_iterator<char>()};

    // Parse it.
    const auto iers_data = heyoka::model::parse_iers_data(file_data);

    // Create the header file first.
    std::ofstream oheader("builtin_iers_data.hpp");
    oheader << fmt::format(R"(#ifndef HEYOKA_DETAIL_IERS_BUILTIN_IERS_DATA_HPP
#define HEYOKA_DETAIL_IERS_BUILTIN_IERS_DATA_HPP

#include <heyoka/config.hpp>
#include <heyoka/model/iers.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{{

extern const iers_data_row init_iers_data[{}];

}}

HEYOKA_END_NAMESPACE

#endif
)",
                           iers_data.size());

    // Now the cpp file.
    std::ofstream ocpp("builtin_iers_data.cpp");
    ocpp << fmt::format(R"(#include <limits>

#include <heyoka/config.hpp>
#include <heyoka/detail/iers/builtin_iers_data.hpp>
#include <heyoka/model/iers.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{{

constinit const iers_data_row init_iers_data[{}] = {{)",
                        iers_data.size());

    for (const auto &[mjd, cur_delta_ut1_utc] : iers_data) {
        if (std::isnan(cur_delta_ut1_utc)) {
            ocpp << fmt::format("{{.mjd={}, .delta_ut1_utc=std::numeric_limits<double>::quiet_NaN()}},\n", mjd);
        } else {
            ocpp << fmt::format("{{.mjd={}, .delta_ut1_utc={}}},\n", mjd, cur_delta_ut1_utc);
        }
    }

    ocpp << R"(};

}

HEYOKA_END_NAMESPACE
)";
}
