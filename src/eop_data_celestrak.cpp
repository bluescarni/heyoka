// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <boost/charconv.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/http_download.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to parse a double-precision value from a field in a row of a celestrak EOP data file.
double parse_eop_data_celestrak_double(const std::ranges::contiguous_range auto &cur_field)
{
    // Fetch the range of data.
    const auto *const begin = std::ranges::data(cur_field);
    const auto *const end = begin + std::ranges::size(cur_field);

    // Try to parse.
    double retval{};
    const auto parse_res = boost::charconv::from_chars(begin, end, retval);
    if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Error parsing a CelesTrak EOP data file: the string '{}' could "
                                                "not be parsed as a valid double-precision value",
                                                std::string_view(begin, end)));
    }

    return retval;
}

} // namespace

// NOTE: the expected format for the data in str is described here:
//
// https://celestrak.org/SpaceData/EOP-format.php
eop_data_table parse_eop_data_celestrak(const std::string &str)
{
    // Parse line by line, splitting on newlines.
    eop_data_table retval;
    bool past_first_line = false;
    for (const auto cur_line : str | std::views::split('\n')) {
        // NOTE: skip the first line which contains the CSV header.
        if (!past_first_line) {
            past_first_line = true;
            continue;
        }

        // NOTE: celestrak data files may have a newline at the end. When we encounter it, just break out.
        if (std::ranges::empty(cur_line)) {
            break;
        }

        // Init the EOP values.
        double mjd{}, delta_ut1_utc{}, pm_x{}, pm_y{}, dX{}, dY{};

        // This is the index of the last field we will be reading from.
        constexpr auto last_field_idx = 9u;

        // Within an individual line, we split on the CSV separator ','.
        auto field_range = cur_line | std::views::split(',');
        // NOTE: keep track of the field index.
        boost::safe_numerics::safe<unsigned> field_idx = 0;
        for (auto field_it = std::ranges::begin(field_range);
             // NOTE: here we check with <= in order to make sure we read the last field.
             // This means that, on a correct parse, field_idx will have a value of
             // last_field_idx + 1 when exiting the loop.
             field_idx <= last_field_idx && field_it != std::ranges::end(field_range); ++field_it, ++field_idx) {
            switch (static_cast<unsigned>(field_idx)) {
                case 1u:
                    // Parse the mjd.
                    mjd = parse_eop_data_celestrak_double(*field_it);
                    break;
                case 2u:
                    // Parse pm_x.
                    pm_x = parse_eop_data_celestrak_double(*field_it);
                    break;
                case 3u:
                    // Parse pm_y.
                    pm_y = parse_eop_data_celestrak_double(*field_it);
                    break;
                case 4u:
                    // Parse the UT1-UTC value.
                    delta_ut1_utc = parse_eop_data_celestrak_double(*field_it);
                    break;
                case 8u:
                    // Parse the dX value.
                    //
                    // NOTE: celestrak provides dX/dY in arcsecs but we want them in milliarcsecs.
                    dX = parse_eop_data_celestrak_double(*field_it) * 1000.;
                    break;
                case 9u:
                    // Parse the dY value.
                    dY = parse_eop_data_celestrak_double(*field_it) * 1000.;
                    break;
                default:;
            }
        }

        // Check if we parsed everything we needed to.
        if (field_idx != last_field_idx + 1u) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("Error parsing a CelesTrak EOP data file: at least {} fields "
                                                    // LCOV_EXCL_STOP
                                                    "were expected in a data row, but {} were found instead",
                                                    last_field_idx + 1u, static_cast<unsigned>(field_idx)));
        }

        // Add the line to retval.
        retval.push_back({.mjd = mjd, .delta_ut1_utc = delta_ut1_utc, .pm_x = pm_x, .pm_y = pm_y, .dX = dX, .dY = dY});
    }

    // Validate the output.
    validate_eop_data_table(retval);

    return retval;
}

} // namespace detail

eop_data eop_data::fetch_latest_celestrak(bool long_term)
{
    // Download the file.
    const auto *filename = long_term ? "EOP-All.csv" : "EOP-Last5Years.csv";
    auto [text, timestamp] = detail::http_download("celestrak.org", 80, fmt::format("/SpaceData/{}", filename));

    // Build the identifier string.
    const auto *identifier = long_term ? "celestrak_long_term" : "celestrak_last_5_years";

    // Parse, validate and return.
    return eop_data(detail::parse_eop_data_celestrak(text), std::move(timestamp), identifier);
}

HEYOKA_END_NAMESPACE
