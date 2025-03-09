// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <charconv>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <boost/smart_ptr/make_shared.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

// NOTE: clang implements std::from_chars() for floating-point types
// only since clang 20. Thus, we will be resorting to the Boost implementation.
#if defined(__clang__)

#include <boost/charconv.hpp>

#endif

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/iers/iers.hpp>
#include <heyoka/model/iers.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

bool iers_data_row::operator==(const iers_data_row &other) const noexcept
{
    // NOTE: mjd cannot be nan.
    if (mjd != other.mjd) {
        return false;
    }

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    if (std::isnan(delta_ut1_utc)) {
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        return std::isnan(other.delta_ut1_utc);
    }

    return delta_ut1_utc == other.delta_ut1_utc;
}

namespace detail
{

namespace
{

// The expected line length in a IERS data file.
constexpr auto iers_data_expected_line_length = 185u;

// NOTE: small wrapper to workaround the lack of std::from_chars for floating-point
// values on clang.
auto from_chars(const char *first, const char *last, double &value)
{
#if defined(__clang__)

    return boost::charconv::from_chars(first, last, value);

#else

    return std::from_chars(first, last, value);

#endif
}

// Helper to parse the MJD from a line in a IERS data file.
double parse_iers_data_mjd(const auto &cur_line)
{
    assert(cur_line.size() >= iers_data_expected_line_length);

    // Fetch the range of mjd data.
    const auto *mjd_begin = cur_line.data() + 7;
    const auto *const mjd_end = mjd_begin + 8;

    // Ignore leading whitespaces.
    for (; mjd_begin != mjd_end && *mjd_begin == ' '; ++mjd_begin) {
    }

    // Try to parse.
    double mjd{};
    const auto mjd_parse_res = from_chars(mjd_begin, mjd_end, mjd);
    if (mjd_parse_res.ec != std::errc{} || mjd_parse_res.ptr != mjd_end) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Error parsing a finals2000A.all IERS data file: the string '{}' could not be parsed as a valid MJD",
            std::string_view(mjd_begin, mjd_end)));
    }

    return mjd;
}

// Helper to parse the UT1-UTC difference from a line in a IERS data file.
double parse_iers_data_delta_ut1_utc(const auto &cur_line)
{
    assert(cur_line.size() >= iers_data_expected_line_length);

    // We first try to fetch the UT1-UTC from the bulletin B data.
    // This is the post-processed most accurate data, usually lagging
    // behind the present time.
    const auto *begin = cur_line.data() + 154;
    const auto *end = begin + 11;

    // Ignore leading whitespaces.
    for (; begin != end && *begin == ' '; ++begin) {
    }
    const auto bullB_sv = std::string_view(begin, end);

    // Try to parse, but only if the bulletin B data is not empty: if empty, it means
    // it is not available yet.
    if (begin != end) {
        double delta_ut1_utc{};
        const auto parse_res = from_chars(begin, end, delta_ut1_utc);
        if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Error parsing a finals2000A.all IERS data file: the bulletin B string for the UT1-UTC "
                            "difference '{}' could not be parsed as a floating-point value",
                            bullB_sv));
        }

        return delta_ut1_utc;
    }

    // Try the bulletin A data.
    begin = cur_line.data() + 58;
    end = begin + 10;

    // Ignore leading whitespaces.
    for (; begin != end && *begin == ' '; ++begin) {
    }
    const auto bullA_sv = std::string_view(begin, end);

    // Try to parse, but only if the bulletin A data is not empty: if empty, it means
    // that it is not available yet and we will return NaN instead.
    if (begin == end) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double delta_ut1_utc{};
    const auto parse_res = from_chars(begin, end, delta_ut1_utc);
    if (parse_res.ec != std::errc{} || parse_res.ptr != end) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Error parsing a finals2000A.all IERS data file: the bulletin A string for the UT1-UTC "
                        "difference '{}' could not be parsed as a floating-point value",
                        bullA_sv));
    }

    return delta_ut1_utc;
}

// Helper to validate IERS data.
void validate_iers_data(const std::vector<iers_data_row> &data)
{
    const auto n_entries = data.size();

    for (decltype(data.size()) i = 0; i < n_entries; ++i) {
        // All mjd values must be finite and ordered in strictly ascending order.
        const auto cur_mjd = data[i].mjd;
        if (!std::isfinite(cur_mjd)) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid finals2000A.all IERS data file detected: the MJD value {} on line {} is not finite", cur_mjd,
                i));
        }
        // NOTE: if data[i + 1u].mjd is NaN, then cur_mjd >= data[i + 1u].mjd evaluates
        // to false and we will throw on the next iteration when we detect a non-finite
        // value for the mjd.
        if (i + 1u != n_entries && cur_mjd >= data[i + 1u].mjd) [[unlikely]] {
            throw std::invalid_argument(fmt::format("Invalid finals2000A.all IERS data file detected: the MJD value {} "
                                                    "on line {} is not less than the MJD value in the next line ({})",
                                                    cur_mjd, i, data[i + 1u].mjd));
        }

        // UT1-UTC values cannot be inf (but they can be NaN if they are missing).
        const auto cur_delta_ut1_utc = data[i].delta_ut1_utc;
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        if (std::isinf(cur_delta_ut1_utc)) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid finals2000A.all IERS data file detected: the UT1-UTC value {} on line {} is an infinity",
                cur_delta_ut1_utc, i));
        }
    }
}

} // namespace

} // namespace detail

// NOTE: it is expected that str stores the content of the data file finals2000A.all from:
//
// https://maia.usno.navy.mil/ser7/finals2000A.all
//
// This data file contains the history of IERS data from 1973 up to the present time,
// also including predictions for the future. The format of the file is described here:
//
// https://maia.usno.navy.mil/ser7/readme.finals2000A
iers_data_t parse_iers_data(const std::string &str)
{
    // Parse line by line, splitting on newlines.
    iers_data_t retval;
    for (const auto cur_line : str | std::views::split('\n')) {
        // NOTE: finals2000A.all files may have a newline at the end, when we encounter
        // it just break out.
        if (std::ranges::empty(cur_line)) {
            break;
        }

        // Check the line length.
        if (std::ranges::size(cur_line) < detail::iers_data_expected_line_length) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid line detected in a finals2000A.all IERS data file: the expected number of "
                "characters in the line is at least {}, but a line with {} character(s) was detected instead",
                detail::iers_data_expected_line_length, std::ranges::size(cur_line)));
        }

        // Parse the mjd.
        const auto mjd = detail::parse_iers_data_mjd(cur_line);

        // Parse the UT1-UTC difference.
        const auto delta_ut1_utc = detail::parse_iers_data_delta_ut1_utc(cur_line);

        // Add the line to retval.
        retval.push_back({.mjd = mjd, .delta_ut1_utc = delta_ut1_utc});
    }

    // Validate the output.
    detail::validate_iers_data(retval);

    return retval;
}

boost::shared_ptr<const iers_data_t> get_iers_data()
{
    return heyoka::detail::cur_iers_data;
}

void set_iers_data(iers_data_t new_data)
{
    // Validate.
    detail::validate_iers_data(new_data);

    // Assign.
    heyoka::detail::cur_iers_data = boost::make_shared<const iers_data_t>(std::move(new_data));
}

} // namespace model

HEYOKA_END_NAMESPACE