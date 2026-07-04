// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <boost/charconv.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/detail/http_download.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to parse a value from a field in a row of a celestrak SW data file.
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

// Helper to parse a UTC calendar date from a celestrak SW datafile as a UTC mjd.
//
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
    const auto parse_ymd = [](const auto *begin, const auto *end) {
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
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Error parsing a celestrak SW data file: the conversion of the ISO "
                                                "8601 date string '{}' to a Julian date produced the error code {}",
                                                std::string_view(begin, end), ret));
        // LCOV_EXCL_STOP
    }

    // NOTE: the eraCal2jd() function returns directly the mjd as djm.
    return djm;
}

} // namespace

// NOTE: the expected format for the data in str is described here:
//
// https://celestrak.org/SpaceData/SpaceWx-format.php
//
// NOTE: this function parses the CelesTrak data as-is - the time re-anchoring is performed in a separate
// post-processing function.
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
        double mjd{}, f107{}, f107a_center81{}, Ap_avg{};

        // This is the index of the last field we will be reading from.
        constexpr auto last_field_idx = 27u;

        // Within an individual line, we split on the CSV separator ','.
        auto field_range = cur_line | std::views::split(',');
        // NOTE: keep track of the field index.
        boost::safe_numerics::safe<unsigned> field_idx = 0;
        // NOTE: keep track of missing data.
        bool missing_data = false;
        for (auto field_it = std::ranges::begin(field_range);
             // NOTE: here we check with <= in order to make sure we read the last field.
             // This means that, on a correct parse, field_idx will have a value of
             // last_field_idx + 1 when exiting the loop.
             field_idx <= last_field_idx && field_it != std::ranges::end(field_range); ++field_it, ++field_idx) {
            switch (static_cast<unsigned>(field_idx)) {
                case 0u:
                    // Parse the mjd.
                    mjd = parse_sw_data_celestrak_mjd(*field_it);
                    break;
                case 20u:
                    // Parse Ap_avg.
                    if (const auto val = parse_sw_data_celestrak_value<double>(*field_it)) {
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
        }

        // Check if we parsed everything we needed to.
        if (field_idx != last_field_idx + 1u) [[unlikely]] {
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

    return retval;
}

// This function re-anchors the SW data from CelesTrak so that each quantity refers to the 0h UTC time of the
// corresponding mjd. Re-anchoring brings them all to a common instant, as required for a single data row.
//
// The re-anchoring is done via linear interpolation, thus after re-anchoring tbl will be one row shorter (e.g., from 3
// original data points we would output 2 re-interpolated data points).
void reanchor_sw_data_celestrak(sw_data_table &tbl)
{
    // As a first check, we need at least 3 rows: 2 is the bare minimum for linear interpolation, 3 produces 2 rows in
    // the post-processed datasets, which is the bare minimum in the rest of the API.
    const auto n_orig_rows = tbl.size();
    if (n_orig_rows < 3u) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid CelesTrak SW dataset detected: the minimum number of required "
                                                "rows is 3, but the dataset contains only {} row(s)",
                                                n_orig_rows));
    }

    // As a second check, we want the dates to be strictly consecutive integral values. This checks our assumption that
    // the CelesTrak dataset is a gap-free daily series: exactly one row per calendar day, each dated at an integral (0h
    // UTC / midnight) mjd. The integral mjd is what lets us treat the native anchors below (12h, 17h/20h) as fixed
    // offsets from it.
    for (decltype(tbl.size()) i = 0; i < n_orig_rows; ++i) {
        const auto cur_mjd = tbl[i].mjd;

        if (!std::isfinite(cur_mjd)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid CelesTrak SW dataset detected: a non-finite mjd was found at row index {}", i));
        }

        if (std::trunc(cur_mjd) != cur_mjd) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid CelesTrak SW dataset detected: a non-integral mjd was found at row index {}", i));
        }

        // NOTE: we want to make sure that the magnitudes of the mjd values are small enough to always allow exact
        // subtraction computations.
        if (std::abs(cur_mjd) > std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid CelesTrak SW dataset detected: the mjd value {} at row index {} is too large in magnitude",
                cur_mjd, i));
        }

        if (i == 0u) {
            continue;
        }

        // Now we can check for monotonicity and 1 day delta between consecutive values.
        const auto prev_mjd = tbl[i - 1u].mjd;

        if (!(prev_mjd < cur_mjd)) [[unlikely]] {
            throw std::invalid_argument(fmt::format("Invalid CelesTrak SW dataset detected: the mjd value at row index "
                                                    "{} is not greater than the mjd value at the previous row index",
                                                    i));
        }

        if (cur_mjd - prev_mjd != 1.0) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid CelesTrak SW dataset detected: the mjd value at row index "
                            "{} is not 1 day larger than the mjd value at the previous row index",
                            i));
        }
    }

    // We can now proceed to the re-anchoring.
    //
    // The raw CelesTrak quantities are natively anchored at different times of the day, expressed here as a fraction of
    // a day past 0h UTC:
    //
    // - Ap_avg and f107a_center81 are daily/centred averages, hence anchored at the centre of the day (12h UTC),
    // - f107 is an instantaneous measurement whose UT time changed on 1991-06-01, from 17:00 UT (Ottawa) to 20:00 UT
    //   (Penticton). The mjd of 1991-06-01 is 48408.
    //
    // NOTE: in these offsets and in the rest of the function too, we ignore the existence of leap seconds. These would
    // introduce very minor changes, far below the measurement accuracy of the space weather indices and the atmospheric
    // models in which they are used.
    constexpr double Ap_avg_off = 0.5;
    constexpr double f107a_center81_off = 0.5;
    const auto f107_off = [](const double mjd) noexcept {
        constexpr double f107_switch_mjd = 48408;
        return mjd < f107_switch_mjd ? 17. / 24 : 20. / 24;
    };

    // This helper first determines the line passing through (t0, q0) and (t1, q1), and then uses it to evaluate q at
    // the arbitrary time t.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const auto linterp = [](const double t0, const double q0, const double t1, const double q1, const double t) {
        const auto slope = (q1 - q0) / (t1 - t0);
        return q0 + (slope * (t - t0));
    };

    // Re-anchor in-place.
    for (decltype(tbl.size()) i = 1; i < n_orig_rows; ++i) {
        // Fetch the data for the previous and current rows.
        const auto [prev_mjd, prev_Ap_avg, prev_f107, prev_f107a_center81] = tbl[i - 1u];
        const auto [cur_mjd, cur_Ap_avg, cur_f107, cur_f107a_center81] = tbl[i];

        // Interpolate.
        const auto Ap_avg = linterp(prev_mjd + Ap_avg_off, prev_Ap_avg, cur_mjd + Ap_avg_off, cur_Ap_avg, cur_mjd);
        const auto f107
            = linterp(prev_mjd + f107_off(prev_mjd), prev_f107, cur_mjd + f107_off(cur_mjd), cur_f107, cur_mjd);
        const auto f107a_center81 = linterp(prev_mjd + f107a_center81_off, prev_f107a_center81,
                                            cur_mjd + f107a_center81_off, cur_f107a_center81, cur_mjd);

        // NOTE: the re-anchored data is written in the *previous* slot. The current slot needs to stay pristine in
        // order for the next interpolation to take place.
        tbl[i - 1u] = {.mjd = cur_mjd, .Ap_avg = Ap_avg, .f107 = f107, .f107a_center81 = f107a_center81};
    }

    // Remove the last line, which still contains the original data.
    tbl.pop_back();
}

} // namespace detail

sw_data sw_data::fetch_latest_celestrak(const bool long_term)
{
    // Download the file.
    const auto *filename = long_term ? "SW-All.csv" : "SW-Last5Years.csv";
    auto [text, timestamp] = detail::http_download("celestrak.org", 80, fmt::format("/SpaceData/{}", filename));

    // Build the identifier string.
    const auto *identifier = long_term ? "celestrak_long_term" : "celestrak_last_5_years";

    // Parse the raw data, re-anchor it to 0h UTC, construct and return.
    auto tbl = detail::parse_sw_data_celestrak(text);
    detail::reanchor_sw_data_celestrak(tbl);
    return sw_data(std::move(tbl), std::move(timestamp), identifier, true);
}

HEYOKA_END_NAMESPACE
