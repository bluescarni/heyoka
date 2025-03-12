// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_IERS_DATA_HPP
#define HEYOKA_IERS_DATA_HPP

#include <string>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

// Single row in a IERS data table.
struct HEYOKA_DLL_PUBLIC iers_row {
    // UTC modified Julian date.
    double mjd = 0;
    // UT1-UTC (seconds).
    double delta_ut1_utc = 0;

    bool operator==(const iers_row &) const noexcept;
};

// The IERS data table.
using iers_table = std::vector<iers_row>;

HEYOKA_DLL_PUBLIC class iers_data
{
    iers_table m_data;
    std::string m_timestamp;

    explicit iers_data(iers_table, std::string);

public:
    iers_data();
    [[nodiscard]] const iers_table &get_table() const noexcept;

    [[nodiscard]] const std::string &get_timestamp() const noexcept;

    HEYOKA_DLL_PUBLIC static iers_data fetch_latest();
};

namespace detail
{

[[nodiscard]] HEYOKA_DLL_PUBLIC iers_table parse_iers_data(const std::string &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
