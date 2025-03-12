// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <charconv>
#include <exception>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/iers_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// List of abbreviated month names.
constexpr std::array<std::string_view, 12> iers_data_month_names
    = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

// Map to associate abbreviated month names to [1, 12] indices.
// NOLINTNEXTLINE(cert-err58-cpp)
const auto iers_data_month_names_map = []() {
    std::unordered_map<std::string_view, unsigned> retval;

    for (auto i = 0u; i < 12u; ++i) {
        retval.emplace(iers_data_month_names[i], i + 1u);
    }

    return retval;
}();

// Regular expression to parse a date in the RFC 7231 format:
//
// https://stackoverflow.com/questions/54927845/what-is-valid-rfc1123-date-format
//
// NOLINTNEXTLINE(cert-err58-cpp)
const auto iers_data_date_regexp = std::regex(
    fmt::format(R"((Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{{2}}) ({}) (\d{{4}}) (\d{{2}}):(\d{{2}}):(\d{{2}}) GMT)",
                fmt::join(iers_data_month_names, "|")));

// Helper to extract from the "Last-Modified" field of an http response the timestamp in the format required
// by the iers_data class.
std::string iers_data_parse_last_modified(std::string_view lm_field)
{
    std::cmatch matches;
    if (std::regex_match(lm_field.data(), lm_field.data() + lm_field.size(), matches, iers_data_date_regexp))
        [[likely]] {
        // Check if all groups matched.
        // NOTE: 7 + 1 because the first match is the entire string.
        if (matches.size() == 8u) [[likely]] {
            // Parse the day
            unsigned day{};
            auto res = std::from_chars(matches[2].first, matches[2].second, day);
            if (res.ec != std::errc{} || res.ptr != matches[2].second) [[unlikely]] {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("Could not parse the string '{}' as a day",
                                                        std::string_view(matches[2].first, matches[2].second)));
                // LCOV_EXCL_STOP
            }

            // Parse the month.
            const auto month_str = std::string_view(matches[3].first, matches[3].second);
            // NOTE: this must hold because the regex matched.
            assert(iers_data_month_names_map.contains(month_str));
            const auto month = iers_data_month_names_map.find(month_str)->second;

            // Parse the year.
            unsigned year{};
            res = std::from_chars(matches[4].first, matches[4].second, year);
            if (res.ec != std::errc{} || res.ptr != matches[4].second) [[unlikely]] {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("Could not parse the string '{}' as a year",
                                                        std::string_view(matches[4].first, matches[4].second)));
                // LCOV_EXCL_STOP
            }

            // Assemble the year-month-day string.
            return fmt::format("{:04}_{:02}_{:02}", year, month, day);
        }
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Could not parse the string '{}' as the 'Last-Modified' field of an http header", lm_field));
    // LCOV_EXCL_STOP
}

} // namespace

} // namespace detail

iers_data iers_data::fetch_latest()
{
    try {
        // NOTE: code adapted from here:
        // https://github.com/boostorg/beast/blob/develop/example/http/client/sync-ssl/http_client_sync_ssl.cpp
        namespace net = boost::asio;
        namespace ssl = net::ssl;
        using tcp = net::ip::tcp;
        namespace beast = boost::beast;
        namespace http = beast::http;

        // Parameters for the connection.
        constexpr auto host = "maia.usno.navy.mil";
        constexpr auto port = "443";
        constexpr auto target = "/ser7/finals2000A.all";
        constexpr int version = 11;

        // The io_context is required for all I/O.
        net::io_context ioc;

        // The SSL context is required, and holds certificates.
        ssl::context ctx(ssl::context::tlsv12_client);

        // Set default verification paths (uses system's certificate store).
        ctx.set_default_verify_paths();

        // Set verification mode.
        ctx.set_verify_mode(ssl::verify_peer);

        // These objects perform our I/O.
        tcp::resolver resolver(ioc);
        ssl::stream<beast::tcp_stream> stream(ioc, ctx);

        // Set the expected hostname in the peer certificate for verification.
        stream.set_verify_callback(ssl::host_name_verification(host));

        // Look up the domain name.
        auto const results = resolver.resolve(host, port);

        // Make the connection on the IP address we get from a lookup.
        beast::get_lowest_layer(stream).connect(results);

        // Perform the SSL handshake.
        stream.handshake(ssl::stream_base::client);

        // Set up an HTTP GET request message.
        http::request<http::string_body> req{http::verb::get, target, version};
        req.set(http::field::host, host);
        req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

        // Send the HTTP request to the remote host.
        http::write(stream, req);

        // This buffer is used for reading and must be persisted.
        beast::flat_buffer buffer;

        // Declare a container to hold the response.
        http::response<http::dynamic_body> res;

        // Receive the HTTP response.
        http::read(stream, buffer, res);

        // Parse the "last modified field" to construct the timestamp.
        auto timestamp = detail::iers_data_parse_last_modified(res[http::field::last_modified]);

        // Fetch the message body as a string. See:
        // https://github.com/boostorg/beast/issues/819
        const auto body = boost::beast::buffers_to_string(res.body().data());

        // Gracefully close the stream.
        stream.shutdown();

        // Construct and return.
        return iers_data{detail::parse_iers_data(body), std::move(timestamp)};

        // LCOV_EXCL_START
    } catch (const std::exception &ex) {
        throw std::invalid_argument(fmt::format("Error while trying to download the latest IERS data: {}", ex.what()));
    } catch (...) {
        throw std::invalid_argument("Error while trying to download the latest IERS data");
    }
    // LCOV_EXCL_STOP
}

HEYOKA_END_NAMESPACE
