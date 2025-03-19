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
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// List of abbreviated month names.
constexpr std::array<std::string_view, 12> eop_data_month_names
    = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

// Map to associate abbreviated month names to [1, 12] indices.
// NOLINTNEXTLINE(cert-err58-cpp)
const auto eop_data_month_names_map = []() {
    std::unordered_map<std::string_view, unsigned> retval;

    for (auto i = 0u; i < 12u; ++i) {
        retval.emplace(eop_data_month_names[i], i + 1u);
    }

    return retval;
}(); // LCOV_EXCL_LINE

// Regular expression to parse a date in the RFC 7231 format:
//
// https://stackoverflow.com/questions/54927845/what-is-valid-rfc1123-date-format
//
// NOLINTNEXTLINE(cert-err58-cpp)
const auto eop_data_date_regexp = std::regex(
    fmt::format(R"((Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{{2}}) ({}) (\d{{4}}) (\d{{2}}):(\d{{2}}):(\d{{2}}) GMT)",
                fmt::join(eop_data_month_names, "|")));

// Helper to extract from the "Last-Modified" field of an http response the timestamp in the format required
// by the eop_data class.
std::string eop_data_parse_last_modified(std::string_view lm_field)
{
    // Helper to parse an unsigned integral quantity from the range [begin, end). 'name'
    // is the name of the quantity, to be used only for error reporting.
    auto uint_parse = [](const char *begin, const char *end, std::string_view name) {
        unsigned out{};
        const auto res = std::from_chars(begin, end, out);
        if (res.ec != std::errc{} || res.ptr != end) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("Could not parse the string '{}' as an integral {} value",
                                                    std::string_view(begin, end), name));
            // LCOV_EXCL_STOP
        }

        return out;
    };

    std::cmatch matches;
    if (std::regex_match(lm_field.data(), lm_field.data() + lm_field.size(), matches, eop_data_date_regexp))
        [[likely]] {
        // Check if all groups matched.
        // NOTE: 7 + 1 because the first match is the entire string.
        if (matches.size() == 8u) [[likely]] {
            // Parse the day
            const auto day = uint_parse(matches[2].first, matches[2].second, "day");

            // Parse the month.
            const auto month_str = std::string_view(matches[3].first, matches[3].second);
            // NOTE: this must hold because the regex matched.
            assert(eop_data_month_names_map.contains(month_str));
            const auto month = eop_data_month_names_map.find(month_str)->second;

            // Parse the year.
            const auto year = uint_parse(matches[4].first, matches[4].second, "year");

            // Parse the hour.
            const auto hour = uint_parse(matches[5].first, matches[5].second, "hour");

            // Parse the minute.
            const auto minute = uint_parse(matches[6].first, matches[6].second, "minute");

            // Parse the second.
            const auto second = uint_parse(matches[7].first, matches[7].second, "second");

            // Assemble the timestamp.
            return fmt::format("{:04}_{:02}_{:02}_{:02}_{:02}_{:02}", year, month, day, hour, minute, second);
        }
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Could not parse the string '{}' as the 'Last-Modified' field of an http header", lm_field));
    // LCOV_EXCL_STOP
}

} // namespace

} // namespace detail

std::pair<std::string, std::string> eop_data::download(const std::string &host, unsigned port,
                                                       const std::string &target)
{
    try {
        // NOTE: code adapted from here:
        // https://github.com/boostorg/beast/blob/develop/example/http/client/sync-ssl/http_client_sync_ssl.cpp
        namespace net = boost::asio;
        namespace ssl = net::ssl;
        using tcp = net::ip::tcp;
        namespace beast = boost::beast;
        namespace http = beast::http;

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
        auto const results = resolver.resolve(host, fmt::format("{}", port));

        // Make the connection on the IP address we get from a lookup.
        beast::get_lowest_layer(stream).connect(results);

        // Perform the SSL handshake.
        stream.handshake(ssl::stream_base::client);

        // Set up an HTTP GET request message.
        // NOTE: the 11 stands for HTTP 1.1.
        http::request<http::string_body> req{http::verb::get, target, 11};
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
        auto timestamp = detail::eop_data_parse_last_modified(res[http::field::last_modified]);

        // Fetch the message body as a string. See:
        // https://github.com/boostorg/beast/issues/819
        auto body = boost::beast::buffers_to_string(res.body().data());

        // Gracefully close the stream.
        beast::error_code ec;
        stream.shutdown(ec);

        // ssl::error::stream_truncated, also known as an SSL "short read",
        // indicates the peer closed the connection without performing the
        // required closing handshake (for example, Google does this to
        // improve performance). Generally this can be a security issue,
        // but if your communication protocol is self-terminated (as
        // it is with both HTTP and WebSocket) then you may simply
        // ignore the lack of close_notify.
        //
        // https://github.com/boostorg/beast/issues/38
        //
        // https://security.stackexchange.com/questions/91435/how-to-handle-a-malicious-ssl-tls-shutdown
        //
        // When a short read would cut off the end of an HTTP message,
        // Beast returns the error beast::http::error::partial_message.
        // Therefore, if we see a short read here, it has occurred
        // after the message has been completed, so it is safe to ignore it.
        if (ec != beast::error_code{} && ec != net::ssl::error::stream_truncated) [[unlikely]] {
            // LCOV_EXCL_START
            throw beast::system_error{ec};
            // LCOV_EXCL_STOP
        }

        // Construct and return.
        return std::make_pair(std::move(body), std::move(timestamp));

        // LCOV_EXCL_START
    } catch (const std::exception &ex) {
        throw std::invalid_argument(fmt::format("Error while trying to download EOP data: {}", ex.what()));
    } catch (...) {
        throw std::invalid_argument("Error while trying to download EOP data");
    }
    // LCOV_EXCL_STOP
}

HEYOKA_END_NAMESPACE
