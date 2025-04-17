// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <iterator>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <boost/safe_numerics/safe_integer.hpp>

#include <boost/charconv.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/detail/http_download.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to parse a value from a field in a row of a celestrak SW data.
template <typename T>
std::optional<T> parse_sw_data_celestrak_value(const std::ranges::contiguous_range auto &cur_field)
{
    // NOTE: an empty field means missing data.
    if (std::ranges::empty(cur_field)) {
        return {};
    }

    // Fetch the range of data.
    const auto *const begin = std::ranges::data(cur_field);
    const auto *const end = begin + std::ranges::size(cur_field);

    // Try to parse.
    T retval{};
    const auto parse_res = boost::charconv::from_chars(begin, end, retval);
    if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Error parsing a celestrak SW data file: the string '{}' could "
                                                "not be parsed as a valid numerical value",
                                                std::string_view(begin, end)));
    }

    return retval;
}

// Helper to parse a UTC date from a celestrak SW datafile as a UTC mjd.
// NOTE: we assume that the date is always present.
double parse_sw_data_celestrak_mjd(const std::ranges::contiguous_range auto &cur_field)
{
    // Fetch the range of data.
    const auto *const begin = std::ranges::data(cur_field);
    const auto *const end = begin + std::ranges::size(cur_field);

    // We are expecting 10 characters in the format yyyy-mm-dd.
    if (std::ranges::size(cur_field) != 10 || std::ranges::distance(cur_field | std::views::split('-')) != 3)
        [[unlikely]] {
        throw std::invalid_argument(fmt::format("Error parsing a celestrak SW data file: the string '{}' could "
                                                "not be parsed as a valid ISO 8601 date",
                                                std::string_view(begin, end)));
    }

    // Helper to parse an int representing a year/month/day.
    const auto parse_ymd = [](auto *begin, const auto *end) {
        // Remove leading zeroes.
        if (end - begin > 1) {
            while (begin != end && *begin == '0') {
                ++begin;
            }
        }

        // Parse.
        int retval{};
        const auto parse_res = boost::charconv::from_chars(begin, end, retval);
        if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
            throw std::invalid_argument(fmt::format("Error parsing a celestrak SW data file: the string '{}' could "
                                                    "not be parsed as a valid year/month/day value",
                                                    std::string_view(begin, end)));
        }

        return retval;
    };

    // Parse year, month, day.
    const auto year = parse_ymd(begin, begin + 4);
    const auto month = parse_ymd(begin + 5, begin + 7);
    const auto day = parse_ymd(begin + 8, begin + 10);

    // Convert into a Julian date.
    double djm0{}, djm{};
    const auto ret = ::eraCal2jd(year, month, day, &djm0, &djm);
    if (ret != 0) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Error parsing a celestrak SW data file: the conversion of the ISO "
                                                "8601 date string '{}' to a Julian date produced the error code {}",
                                                std::string_view(begin, end), ret));
    }

    // NOTE: the eraCal2jd() function returns directly the mjd as djm.
    return djm;
}

} // namespace

// NOTE: the expected format for the data in str is described here:
//
// https://celestrak.org/SpaceData/SpaceWx-format.php
sw_data_table parse_sw_data_celestrak(const std::string &str)
{
    // Parse line by line, splitting on newlines.
    sw_data_table retval;
    for (const auto cur_line : str
                                   | std::views::split('\n')
                                   // NOTE: drop the first line which contains the CSV header.
                                   | std::views::drop(1)) {
        // NOTE: celestrak data files may have a newline at the end. When we encounter it, just break out.
        if (std::ranges::empty(cur_line)) {
            break;
        }

        // Init the SW values.
        double mjd{}, f107{}, f107a_center81{};
        std::uint16_t Ap_avg{};

        // This is the index of the last field we will be reading from.
        constexpr auto last_field_idx = 27u;

        // Within an individual line, we split on the CSV separator ','.
        auto field_range = cur_line | std::views::split(',');
        // NOTE: keep track of the field index.
        boost::safe_numerics::safe<unsigned> field_idx = 0;
        // NOTE: keep track of missing data.
        bool missing_data = false;
        for (auto field_it = std::ranges::begin(field_range); field_it != std::ranges::end(field_range);
             ++field_it, ++field_idx) {
            switch (static_cast<unsigned>(field_idx)) {
                case 0u:
                    // Parse the mjd.
                    mjd = parse_sw_data_celestrak_mjd(*field_it);
                    break;
                case 20u:
                    // Parse Ap_avg.
                    if (const auto val = parse_sw_data_celestrak_value<std::uint16_t>(*field_it)) {
                        Ap_avg = *val;
                    } else {
                        missing_data = true;
                    }

                    break;
                case 24u:
                    // Parse f107.
                    if (const auto val = parse_sw_data_celestrak_value<double>(*field_it)) {
                        f107 = *val;
                    } else {
                        missing_data = true;
                    }

                    break;
                case 27u:
                    // Parse f107a_center81.
                    if (const auto val = parse_sw_data_celestrak_value<double>(*field_it)) {
                        f107a_center81 = *val;
                    } else {
                        missing_data = true;
                    }

                    break;
                default:;
            }

            // Break out if we parsed the last field.
            if (field_idx == last_field_idx) {
                break;
            }
        }

        // Check if we parsed everything we needed to.
        if (field_idx != last_field_idx) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("Error parsing a celestrak SW data file: at least {} fields "
                                                    // LCOV_EXCL_STOP
                                                    "were expected in a data row, but {} were found instead",
                                                    last_field_idx + 1u, static_cast<unsigned>(field_idx)));
        }

        // If we have missing data, break out.
        if (missing_data) {
            break;
        }

        // Add the line to retval.
        retval.push_back({.mjd = mjd, .Ap_avg = Ap_avg, .f107 = f107, .f107a_center81 = f107a_center81});
    }

    // Validate the output.
    validate_sw_data_table(retval);

    return retval;
}

} // namespace detail

sw_data sw_data::fetch_latest_celestrak(bool long_term)
{
    // Download the file.
    const auto *filename = long_term ? "SW-All.csv" : "SW-Last5Years.csv";
    auto [text, timestamp] = detail::http_download("celestrak.org", 80, fmt::format("/SpaceData/{}", filename));

    // Build the identifier string.
    const auto *identifier = long_term ? "celestrak_long_term" : "celestrak_last_5_years";

    // Parse, validate and return.
    return sw_data(detail::parse_sw_data_celestrak(text), std::move(timestamp), identifier);
}

HEYOKA_END_NAMESPACE
