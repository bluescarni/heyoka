// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <boost/charconv.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// The expected line length in a IERS rapid EOP data file.
constexpr auto eop_data_iers_rapid_expected_line_length = 185u;

// Helper to parse the MJD from a line in a IERS rapid EOP data file.
double parse_eop_data_iers_rapid_mjd(auto cur_line)
{
    static_assert(std::ranges::contiguous_range<decltype(cur_line)>);
    assert(std::ranges::size(cur_line) >= eop_data_iers_rapid_expected_line_length);

    // Fetch the range of mjd data.
    const auto *mjd_begin = std::ranges::data(cur_line) + 7;
    const auto *const mjd_end = mjd_begin + 8;

    // Ignore leading whitespaces.
    for (; mjd_begin != mjd_end && *mjd_begin == ' '; ++mjd_begin) {
    }

    // Try to parse.
    double mjd{};
    const auto mjd_parse_res = boost::charconv::from_chars(mjd_begin, mjd_end, mjd);
    if (mjd_parse_res.ec != std::errc{} || mjd_parse_res.ptr != mjd_end) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Error parsing a IERS rapid EOP data file: the string '{}' could not be parsed as a valid MJD",
                        std::string_view(mjd_begin, mjd_end)));
    }

    return mjd;
}

// Helper to parse the UT1-UTC difference from a line in a IERS rapid EOP data file.
std::optional<double> parse_eop_data_iers_rapid_delta_ut1_utc(auto cur_line)
{
    static_assert(std::ranges::contiguous_range<decltype(cur_line)>);
    assert(std::ranges::size(cur_line) >= eop_data_iers_rapid_expected_line_length);

    // We first try to fetch the UT1-UTC from the bulletin B data.
    // This is the post-processed most accurate data, usually lagging
    // behind the present time.
    const auto *begin = std::ranges::data(cur_line) + 154;
    const auto *end = begin + 11;

    // Ignore leading whitespaces.
    for (; begin != end && *begin == ' '; ++begin) {
    }
    const auto bullB_sv = std::string_view(begin, end);

    // Try to parse, but only if the bulletin B data is not empty: if empty, it means
    // it is not available yet.
    if (begin != end) {
        double delta_ut1_utc{};
        const auto parse_res = boost::charconv::from_chars(begin, end, delta_ut1_utc);
        if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Error parsing a IERS rapid EOP data file: the bulletin B string for the UT1-UTC "
                            "difference '{}' could not be parsed as a floating-point value",
                            bullB_sv));
        }

        return delta_ut1_utc;
    }

    // Try the bulletin A data.
    begin = std::ranges::data(cur_line) + 58;
    end = begin + 10;

    // Ignore leading whitespaces.
    for (; begin != end && *begin == ' '; ++begin) {
    }
    const auto bullA_sv = std::string_view(begin, end);

    // Try to parse, but only if the bulletin A data is not empty: if empty, it means
    // that it is not available yet and we will return an empty optional instead.
    if (begin == end) {
        return {};
    }
    double delta_ut1_utc{};
    const auto parse_res = boost::charconv::from_chars(begin, end, delta_ut1_utc);
    if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Error parsing a IERS rapid EOP data file: the bulletin A string for the UT1-UTC "
                        "difference '{}' could not be parsed as a floating-point value",
                        bullA_sv));
    }

    return delta_ut1_utc;
}

} // namespace

// NOTE: the expected format for the data in str is described here:
//
// https://maia.usno.navy.mil/ser7/readme.finals2000A
eop_data_table parse_eop_data_iers_rapid(const std::string &str)
{
    // Parse line by line, splitting on newlines.
    eop_data_table retval;
    for (const auto cur_line : str | std::views::split('\n')) {
        // NOTE: IERS rapid EOP data files may have a newline at the end. When we encounter it, just break out.
        if (std::ranges::empty(cur_line)) {
            break;
        }

        // Check the line length.
        if (std::ranges::size(cur_line) < eop_data_iers_rapid_expected_line_length) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid line detected in a IERS rapid EOP data file: the expected number of "
                "characters in the line is at least {}, but a line with {} character(s) was detected instead",
                eop_data_iers_rapid_expected_line_length, std::ranges::size(cur_line)));
        }

        // Parse the mjd.
        const auto mjd = parse_eop_data_iers_rapid_mjd(cur_line);

        // Parse the UT1-UTC difference.
        const auto delta_ut1_utc = parse_eop_data_iers_rapid_delta_ut1_utc(cur_line);
        if (!delta_ut1_utc) {
            // No data available on this line, break out.
            break;
        }

        // Add the line to retval.
        retval.push_back({.mjd = mjd, .delta_ut1_utc = *delta_ut1_utc});
    }

    // Validate the output.
    validate_eop_data_table(retval);

    return retval;
}

namespace
{

// NOLINTNEXTLINE(cert-err58-cpp)
const std::set<std::string> eop_data_iers_rapid_filenames
    = {"finals2000A.all", "finals2000A.daily", "finals2000A.daily.extended", "finals2000A.data"};

} // namespace

} // namespace detail

eop_data eop_data::fetch_latest_iers_rapid(const std::string &filename)
{
    // Check the provided filename.
    if (!detail::eop_data_iers_rapid_filenames.contains(filename)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid filename '{}' specified for a IERS rapid EOP data file: the valid names are {}",
                        filename, detail::eop_data_iers_rapid_filenames));
    }

    // Download it.
    auto [text, timestamp] = download("maia.usno.navy.mil", 443, fmt::format("/ser7/{}", filename));

    // Build the identifier string.
    auto identifier = fmt::format("iers_rapid_{}", filename);
    // NOTE: we transform '.' into '_' so that we can use the identifier
    // to construct the mangled name of compact-mode primitives (which
    // use '.' as a separator).
    std::ranges::replace(identifier, '.', '_');

    // Parse, validate and return.
    return eop_data(detail::parse_eop_data_iers_rapid(text), std::move(timestamp), std::move(identifier));
}

HEYOKA_END_NAMESPACE
