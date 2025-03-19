// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <future>
#include <stdexcept>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

// Fetch the latest IERS rapid-prediction and long-term data and assemble
// it into a single eop_data dataset.
eop_data eop_data::fetch_latest_iers_combined()
{
    // Async download.
    auto fut_rapid = std::async(std::launch::async, []() { return eop_data::fetch_latest_iers_rapid(); });
    auto fut_long_term = std::async(std::launch::async, &eop_data::fetch_latest_iers_long_term);

    // Fetch the downloaded data.
    const auto data_rapid = fut_rapid.get();
    const auto data_long_term = fut_long_term.get();

    // Fetch a reference to the rapid-prediction data.
    const auto &rapid_table = data_rapid.get_table();

    // NOTE: the return value is built by combining the long-term data with the rapid-prediction data.
    // The rapid-prediction data will augment the long-term data where it is not available.
    auto ret_table = data_long_term.get_table();
    if (ret_table.empty()) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            "Error while combining long-term and rapid-prediction IERS data: the long-term dataset is empty");
        // LCOV_EXCL_STOP
    }

    // Locate the first row in rapid_table in which the date is *greater than* the
    // date in the last row of data_long_term.
    auto it = std::ranges::upper_bound(rapid_table, ret_table.back().mjd, {}, &eop_data_row::mjd);

    // Append to ret_table.
    ret_table.insert(ret_table.end(), it, rapid_table.end());

    // Assemble the result, combining the timestamps and identifiers from the two datasets.
    return eop_data(std::move(ret_table),
                    fmt::format("{}_{}", data_long_term.get_timestamp(), data_rapid.get_timestamp()),
                    fmt::format("{}_{}", data_long_term.get_identifier(), data_rapid.get_identifier()));
}

HEYOKA_END_NAMESPACE
