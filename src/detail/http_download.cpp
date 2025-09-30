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
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/regex.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/http_download.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// List of abbreviated month names.
constexpr std::array<std::string_view, 12> http_download_month_names
    = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

// Map to associate abbreviated month names to [1, 12] indices.
// NOLINTNEXTLINE(cert-err58-cpp)
const auto http_download_month_names_map = []() {
    std::unordered_map<std::string_view, unsigned> retval;

    for (auto i = 0u; i < 12u; ++i) {
        retval.emplace(http_download_month_names[i], i + 1u);
    }

    return retval;
}(); // LCOV_EXCL_LINE

// Regular expression to parse a date in the RFC 7231 format:
//
// https://stackoverflow.com/questions/54927845/what-is-valid-rfc1123-date-format
//
// NOLINTNEXTLINE(cert-err58-cpp)
const auto http_download_date_regexp = boost::regex(
    fmt::format(R"((Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{{2}}) ({}) (\d{{4}}) (\d{{2}}):(\d{{2}}):(\d{{2}}) GMT)",
                fmt::join(http_download_month_names, "|")));

// Helper to extract from the "Last-Modified" field of an http response the timestamp.
std::string http_download_parse_last_modified(std::string_view lm_field)
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

    boost::cmatch matches;
    if (boost::regex_match(lm_field.data(), lm_field.data() + lm_field.size(), matches, http_download_date_regexp))
        [[likely]] {
        // Check if all groups matched.
        // NOTE: 7 + 1 because the first match is the entire string.
        if (matches.size() == 8u) [[likely]] {
            // Parse the day
            const auto day = uint_parse(matches[2].first, matches[2].second, "day");

            // Parse the month.
            const auto month_str = std::string_view(matches[3].first, matches[3].second);
            // NOTE: this must hold because the regex matched.
            assert(http_download_month_names_map.contains(month_str));
            const auto month = http_download_month_names_map.find(month_str)->second;

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

// NOTE: code adapted from here:
//
// https://github.com/boostorg/beast/blob/develop/example/http/client/async-ssl/http_client_async_ssl.cpp
// https://github.com/boostorg/beast/blob/develop/example/http/client/async/http_client_async.cpp
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;
namespace beast = boost::beast;
namespace http = beast::http;

// Timeout duration for network operations.
constexpr auto timeout_duration = std::chrono::seconds(60);

// Helper to check a beast error code and throw as needed.
void check_error(beast::error_code ec, const char *id)
{
    if (ec) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::runtime_error(
            fmt::format("Error in the {} phase while trying to download a file: {}", id, ec.message()));
        // LCOV_EXCL_STOP
    }
}

class https_session : public std::enable_shared_from_this<https_session>
{
    tcp::resolver resolver_;
    ssl::stream<beast::tcp_stream> stream_;
    beast::flat_buffer buffer_; // (Must persist between reads)
    http::request<http::empty_body> req_;
    http::response<http::string_body> res_;
    // This is the data we will extract from the http message.
    std::string timestamp_;
    std::string body_;

public:
    explicit https_session(net::any_io_executor ex, ssl::context &ctx) : resolver_(ex), stream_(ex, ctx) {}

    // Helpers to move the data out after download is completed.
    auto move_timestamp()
    {
        return std::move(timestamp_);
    }
    auto move_body()
    {
        return std::move(body_);
    }

    // Start the asynchronous operation.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void run(char const *host, char const *port, char const *target)
    {
        // Set the expected hostname in the peer certificate for verification.
        stream_.set_verify_callback(ssl::host_name_verification(host));

        // Set up an HTTP GET request message.
        // NOTE: the '11' here means HTTP 1.1.
        req_.version(11);
        req_.method(http::verb::get);
        req_.target(target);
        req_.set(http::field::host, host);
        req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

        // Look up the domain name.
        resolver_.async_resolve(host, port, beast::bind_front_handler(&https_session::on_resolve, shared_from_this()));
    }

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    void on_resolve(beast::error_code ec, tcp::resolver::results_type results)
    {
        check_error(ec, "resolve");

        // Set a timeout on the operation.
        beast::get_lowest_layer(stream_).expires_after(timeout_duration);

        // Make the connection on the IP address we get from a lookup.
        beast::get_lowest_layer(stream_).async_connect(
            results, beast::bind_front_handler(&https_session::on_connect, shared_from_this()));
    }

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type)
    {
        check_error(ec, "connect");

        // Perform the SSL handshake.
        stream_.async_handshake(ssl::stream_base::client,
                                beast::bind_front_handler(&https_session::on_handshake, shared_from_this()));
    }

    void on_handshake(beast::error_code ec)
    {
        check_error(ec, "handshake");

        // Set a timeout on the operation.
        beast::get_lowest_layer(stream_).expires_after(timeout_duration);

        // Send the HTTP request to the remote host.
        http::async_write(stream_, req_, beast::bind_front_handler(&https_session::on_write, shared_from_this()));
    }

    void on_write(beast::error_code ec, std::size_t)
    {
        check_error(ec, "write");

        // Receive the HTTP response.
        http::async_read(stream_, buffer_, res_,
                         beast::bind_front_handler(&https_session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t)
    {
        check_error(ec, "read");

        if (res_.result() != http::status::ok) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::runtime_error(
                fmt::format("HTTP error {}, the response is: '{}'", res_.result_int(), res_.body()));
            // LCOV_EXCL_STOP
        }

        // Parse the "last modified field" to construct the timestamp.
        timestamp_ = http_download_parse_last_modified(res_[http::field::last_modified]);

        // Fetch the message body.
        body_ = res_.body();

        // Set a timeout on the operation.
        beast::get_lowest_layer(stream_).expires_after(timeout_duration);

        // Gracefully close the stream.
        stream_.async_shutdown(beast::bind_front_handler(&https_session::on_shutdown, shared_from_this()));
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void on_shutdown(beast::error_code ec)
    {
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
        if (ec != net::ssl::error::stream_truncated) {
            check_error(ec, "shutdown");
        }
    }
};

class http_session : public std::enable_shared_from_this<http_session>
{
    tcp::resolver resolver_;
    beast::tcp_stream stream_;
    beast::flat_buffer buffer_; // (Must persist between reads)
    http::request<http::empty_body> req_;
    http::response<http::string_body> res_;
    // This is the data we will extract from the http message.
    std::string timestamp_;
    std::string body_;

public:
    explicit http_session(net::io_context &ioc) : resolver_(net::make_strand(ioc)), stream_(net::make_strand(ioc)) {}

    // Helpers to move the data out after download is completed.
    auto move_timestamp()
    {
        return std::move(timestamp_);
    }
    auto move_body()
    {
        return std::move(body_);
    }

    // Start the asynchronous operation.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void run(char const *host, char const *port, char const *target)
    {
        // Set up an HTTP GET request message.
        // NOTE: the '11' here means HTTP 1.1.
        req_.version(11);
        req_.method(http::verb::get);
        req_.target(target);
        req_.set(http::field::host, host);
        req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

        // Look up the domain name.
        resolver_.async_resolve(host, port, beast::bind_front_handler(&http_session::on_resolve, shared_from_this()));
    }

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    void on_resolve(beast::error_code ec, tcp::resolver::results_type results)
    {
        check_error(ec, "resolve");

        // Set a timeout on the operation.
        stream_.expires_after(timeout_duration);

        // Make the connection on the IP address we get from a lookup.
        stream_.async_connect(results, beast::bind_front_handler(&http_session::on_connect, shared_from_this()));
    }

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type)
    {
        check_error(ec, "connect");

        // Set a timeout on the operation.
        stream_.expires_after(timeout_duration);

        // Send the HTTP request to the remote host.
        http::async_write(stream_, req_, beast::bind_front_handler(&http_session::on_write, shared_from_this()));
    }

    void on_write(beast::error_code ec, std::size_t)
    {
        check_error(ec, "write");

        // Receive the HTTP response.
        http::async_read(stream_, buffer_, res_, beast::bind_front_handler(&http_session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t)
    {
        check_error(ec, "read");

        if (res_.result() != http::status::ok) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::runtime_error(
                fmt::format("HTTP error {}, the response is: '{}'", res_.result_int(), res_.body()));
            // LCOV_EXCL_STOP
        }

        // Parse the "last modified field" to construct the timestamp.
        timestamp_ = http_download_parse_last_modified(res_[http::field::last_modified]);

        // Fetch the message body.
        body_ = res_.body();

        // Gracefully close the socket.
        stream_.socket().shutdown(tcp::socket::shutdown_both, ec);

        // not_connected happens sometimes so don't bother reporting it.
        if (ec != beast::errc::not_connected) {
            check_error(ec, "shutdown");
        }

        // If we get here then the connection is closed gracefully.
    }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::mutex ssl_verify_file_mutex;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::optional<std::string> ssl_verify_file;

} // namespace

// Machinery to get/set a global SSL verify file.
//
// This can be used in cases where the system's certificate store is not working properly (e.g., this can happen if a
// vendored SSL library uses a hard-coded certificate store path different from the one in use on the current system).
std::optional<std::string> get_ssl_verify_file()
{
    const std::scoped_lock lock(ssl_verify_file_mutex);

    return ssl_verify_file;
}

// LCOV_EXCL_START

void set_ssl_verify_file(std::string path)
{
    const std::scoped_lock lock(ssl_verify_file_mutex);

    if (path.empty()) {
        ssl_verify_file.reset();
    } else {
        ssl_verify_file.emplace(std::move(path));
    }
}

// LCOV_EXCL_STOP

// NOTE: this is a function to download a file from a remote server via https. In addition to the file,
// its timestamp on the remote server will also be returned in the format year_month_day_hour_minute_second.
std::pair<std::string, std::string> https_download(const std::string &host, unsigned port, const std::string &target)
{
    try {
        namespace net = boost::asio;
        namespace ssl = net::ssl;

        // The io_context is required for all I/O.
        net::io_context ioc;

        // The SSL context is required, and holds certificates.
        ssl::context ctx(ssl::context::tlsv12_client);

        // Set default verification paths (uses system's certificate store).
        ctx.set_default_verify_paths();

        // Load a custom verify file, if provided by the user.
        if (const auto vfile = get_ssl_verify_file()) {
            ctx.load_verify_file(*vfile); // LCOV_EXCL_LINE
        }

        // Set verification mode.
        ctx.set_verify_mode(ssl::verify_peer);

        // Launch the asynchronous operation. The session is constructed with a strand to
        // ensure that handlers do not execute concurrently.
        const auto port_str = fmt::format("{}", port);
        auto sesh = std::make_shared<detail::https_session>(net::make_strand(ioc), ctx);
        sesh->run(host.c_str(), port_str.c_str(), target.c_str());

        // Run the I/O service. The call will return when the get operation is complete.
        ioc.run();

        // Construct and return.
        return std::make_pair(sesh->move_body(), sesh->move_timestamp());

        // LCOV_EXCL_START
    } catch (const std::exception &ex) {
        throw std::invalid_argument(fmt::format("Error invoking https_download(): {}", ex.what()));
    } catch (...) {
        throw std::invalid_argument("Error invoking https_download()");
    }
    // LCOV_EXCL_STOP
}

// NOTE: this is a function to download a file from a remote server via http. In addition to the file,
// its timestamp on the remote server will also be returned in the format year_month_day_hour_minute_second.
std::pair<std::string, std::string> http_download(const std::string &host, unsigned port, const std::string &target)
{
    try {
        namespace net = boost::asio;

        // The io_context is required for all I/O.
        net::io_context ioc;

        // Launch the asynchronous operation.
        const auto port_str = fmt::format("{}", port);
        auto sesh = std::make_shared<detail::http_session>(ioc);
        sesh->run(host.c_str(), port_str.c_str(), target.c_str());

        // Run the I/O service. The call will return when the get operation is complete.
        ioc.run();

        // Construct and return.
        return std::make_pair(sesh->move_body(), sesh->move_timestamp());

        // LCOV_EXCL_START
    } catch (const std::exception &ex) {
        throw std::invalid_argument(fmt::format("Error invoking http_download(): {}", ex.what()));
    } catch (...) {
        throw std::invalid_argument("Error invoking http_download()");
    }
    // LCOV_EXCL_STOP
}

} // namespace detail

HEYOKA_END_NAMESPACE
