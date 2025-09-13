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
#include <tuple>
#include <utility>

#include <boost/charconv.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/http_download.hpp>
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// The expected line length in a IERS rapid EOP data file.
constexpr auto eop_data_iers_rapid_expected_line_length = 185u;

// Helper to parse the MJD from a line in a IERS rapid EOP data file.
double parse_eop_data_iers_rapid_mjd(const std::ranges::contiguous_range auto &cur_line)
{
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

// Helper to parse a field from a line in a IERS rapid EOP data file.
//
// In the rapid datasets, there may be both bulletin A and bulletin B data available. We first try to parse
// the B data as it is supposed to be more accurate, and we fall back to A data if B data is not available.
// If neither A nor B data is available, we will return an empty optional.
//
// The beginA/B and sizeA/B indicate the [begin, begin + size) column range for the bulletin A/B data.
std::optional<double> parse_eop_data_iers_rapid_field(const std::ranges::contiguous_range auto &cur_line,
                                                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                      int beginB, int sizeB, int beginA, int sizeA, const char *name)
{
    assert(std::ranges::size(cur_line) >= eop_data_iers_rapid_expected_line_length);

    // We first try to fetch the bulletin B data. This is the post-processed most accurate data, usually lagging
    // behind the present time.
    const auto *begin = std::ranges::data(cur_line) + beginB;
    const auto *end = begin + sizeB;

    // Ignore leading whitespaces.
    for (; begin != end && *begin == ' '; ++begin) {
    }
    const auto bullB_sv = std::string_view(begin, end);

    // Try to parse, but only if the bulletin B data is not empty: if empty, it means
    // it is not available yet.
    if (begin != end) {
        double retval{};
        const auto parse_res = boost::charconv::from_chars(begin, end, retval);
        if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Error parsing a IERS rapid EOP data file: the bulletin B string for the {} field '{}' "
                            "could not be parsed as a floating-point value",
                            name, bullB_sv));
        }

        return retval;
    }

    // Try the bulletin A data.
    begin = std::ranges::data(cur_line) + beginA;
    end = begin + sizeA;

    // Ignore leading whitespaces.
    for (; begin != end && *begin == ' '; ++begin) {
    }
    const auto bullA_sv = std::string_view(begin, end);

    // Try to parse, but only if the bulletin A data is not empty: if empty, it means
    // that it is not available yet and we will return an empty optional instead.
    if (begin == end) {
        return {};
    }
    double retval{};
    const auto parse_res = boost::charconv::from_chars(begin, end, retval);
    if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Error parsing a IERS rapid EOP data file: the bulletin A string for "
                                                "the {} field '{}' could not be parsed as a floating-point value",
                                                name, bullA_sv));
    }

    return retval;
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
        const auto delta_ut1_utc = parse_eop_data_iers_rapid_field(cur_line, 154, 11, 58, 10, "UT1-UTC difference");
        if (!delta_ut1_utc) {
            // No data available on this line, break out.
            break;
        }

        // Parse the PM x/y values.
        const auto pm_x = parse_eop_data_iers_rapid_field(cur_line, 134, 10, 18, 9, "pm_x");
        if (!pm_x) {
            // No data available on this line, break out.
            break;
        }
        const auto pm_y = parse_eop_data_iers_rapid_field(cur_line, 144, 10, 37, 9, "pm_y");
        if (!pm_y) {
            // No data available on this line, break out.
            break;
        }

        // Parse the dX/dY values.
        const auto dX = parse_eop_data_iers_rapid_field(cur_line, 165, 10, 97, 9, "dX");
        if (!dX) {
            // No data available on this line, break out.
            break;
        }
        const auto dY = parse_eop_data_iers_rapid_field(cur_line, 175, 10, 116, 9, "dY");
        if (!dY) {
            // No data available on this line, break out.
            break;
        }

        // Add the line to retval.
        retval.push_back(
            {.mjd = mjd, .delta_ut1_utc = *delta_ut1_utc, .pm_x = *pm_x, .pm_y = *pm_y, .dX = *dX, .dY = *dY});
    }

    // Validate the output.
    validate_eop_data_table(retval);

    return retval;
}

namespace
{

// NOLINTNEXTLINE(cert-err58-cpp)
const std::set<std::string> eop_data_iers_rapid_filenames_usno
    = {"finals2000A.all", "finals2000A.daily", "finals2000A.daily.extended", "finals2000A.data"};

// NOTE: on the IERS website, there's no file corresponding to USNO's finals2000A.daily.extended.
//
// NOLINTNEXTLINE(cert-err58-cpp)
const std::set<std::string> eop_data_iers_rapid_filenames_iers
    = {"finals.all.iau2000.txt", "finals.daily.iau2000.txt", "finals.data.iau2000.txt"};

} // namespace

} // namespace detail

eop_data eop_data::fetch_latest_iers_rapid(const std::string &origin, const std::string &filename)
{
    // Check the origin.
    if (origin != "usno" && origin != "iers") [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid origin '{}' specified for a IERS rapid EOP data file: the valid origins are 'usno' and 'iers'",
            origin));
    }

    // Check the provided filename.
    if (origin == "usno") {
        if (!detail::eop_data_iers_rapid_filenames_usno.contains(filename)) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid filename '{}' specified for a IERS rapid EOP data file (USNO origin): the valid names are {}",
                filename, detail::eop_data_iers_rapid_filenames_usno));
        }
    } else {
        if (!detail::eop_data_iers_rapid_filenames_iers.contains(filename)) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid filename '{}' specified for a IERS rapid EOP data file (IERS origin): the valid names are {}",
                filename, detail::eop_data_iers_rapid_filenames_iers));
        }
    }

    std::string text, timestamp;

    // Download it.
    if (origin == "usno") {
        std::tie(text, timestamp)
            = detail::https_download("maia.usno.navy.mil", 443, fmt::format("/ser7/{}", filename));
    } else {
        std::tie(text, timestamp)
            = detail::https_download("datacenter.iers.org", 443, fmt::format("/data/latestVersion/{}", filename));
    }

    // Build the identifier string.
    //
    // NOTE: we want to mangle the origin together with the filename. Even if the two origins contain the same files,
    // the timestamps will be in general different, which could lead to confusion about which data is exactly being
    // used.
    auto identifier = fmt::format("iers_rapid_{}_{}", origin, filename);
    // NOTE: we transform '.' into '_' so that we can use the identifier to construct the mangled name of compact-mode
    // primitives (which use '.' as a separator).
    std::ranges::replace(identifier, '.', '_');

    // Parse, validate and return.
    return eop_data(detail::parse_eop_data_iers_rapid(text), std::move(timestamp), std::move(identifier));
}

HEYOKA_END_NAMESPACE
