// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <fstream>

#include <fmt/core.h>

#include <heyoka/iers_data.hpp>

int main()
{
    // Download the latest iers data.
    const auto idata = heyoka::iers_data::fetch_latest();

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
                           idata.get_table().size());

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
                        idata.get_timestamp(), idata.get_table().size());

    for (const auto &[mjd, cur_delta_ut1_utc] : idata.get_table()) {
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
