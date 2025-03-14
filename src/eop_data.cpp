// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_data/builtin_eop_data.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

void eop_data_row::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << mjd;
    oa << delta_ut1_utc;
}

void eop_data_row::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> mjd;
    ia >> delta_ut1_utc;
}

namespace detail
{

// Helper to validate a EOP data table.
// NOTE: this must be called by every fetch_latest_*() function before passing the table to the eop_data constructor.
void validate_eop_data_table(const eop_data_table &data)
{
    const auto n_entries = data.size();

    for (decltype(data.size()) i = 0; i < n_entries; ++i) {
        // All mjd values must be finite and ordered in strictly ascending order.
        const auto cur_mjd = data[i].mjd;
        if (!std::isfinite(cur_mjd)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid EOP data table detected: the MJD value {} on line {} is not finite", cur_mjd, i));
        }
        // NOTE: if data[i + 1u].mjd is NaN, then cur_mjd >= data[i + 1u].mjd evaluates
        // to false and we will throw on the next iteration when we detect a non-finite
        // value for the mjd.
        if (i + 1u != n_entries && cur_mjd >= data[i + 1u].mjd) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("Invalid EOP data table detected: the MJD value {} "
                                                    "on line {} is not less than the MJD value in the next line ({})",
                                                    // LCOV_EXCL_STOP
                                                    cur_mjd, i, data[i + 1u].mjd));
        }

        // UT1-UTC values must be finite.
        const auto cur_delta_ut1_utc = data[i].delta_ut1_utc;
        if (!std::isfinite(cur_delta_ut1_utc)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid EOP data table detected: the UT1-UTC value {} on line {} is not finite",
                            cur_delta_ut1_utc, i));
        }
    }
}

} // namespace detail

struct eop_data::impl {
    eop_data_table m_data;
    // NOTE: timestamp and identifier are meant to uniquely identify
    // the data. The identifier indicates the data source, while the
    // timestamp is used to identify the version of the data. The timestamp
    // is always built from the "Last-Modified" property of the file on the
    // remote server.
    std::string m_timestamp;
    std::string m_identifier;

    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_data;
        ar & m_timestamp;
        ar & m_identifier;
    }
};

void eop_data::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << m_impl;
}

void eop_data::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> m_impl;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
eop_data::eop_data(eop_data_table data, std::string timestamp, std::string identifier)
    : m_impl(std::make_shared<const impl>(std::move(data), std::move(timestamp), std::move(identifier)))
{
}

eop_data::eop_data()
    : m_impl(std::make_shared<const impl>(
          eop_data_table(std::ranges::begin(detail::builtin_eop_data), std::ranges::end(detail::builtin_eop_data)),
          // NOTE: the builtin EOP data is from USNO's finals2000A.all file.
          detail::builtin_eop_data_ts, "usno_finals2000A_all"))
{
}

const eop_data_table &eop_data::get_table() const noexcept
{
    return m_impl->m_data;
}

const std::string &eop_data::get_timestamp() const noexcept
{
    return m_impl->m_timestamp;
}

const std::string &eop_data::get_identifier() const noexcept
{
    return m_impl->m_identifier;
}

HEYOKA_END_NAMESPACE
