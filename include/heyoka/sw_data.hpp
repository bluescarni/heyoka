// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_SW_DATA_HPP
#define HEYOKA_SW_DATA_HPP

#include <compare>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

// Single row in a SW data table.
struct HEYOKA_DLL_PUBLIC sw_data_row {
    // UTC modified Julian date.
    double mjd = 0;
    // Arithmetic average of the 8 Ap indices for the day.
    std::uint16_t Ap_avg = 0;
    // Observed 10.7-cm solar radio flux (F10.7).
    double f107 = 0;
    // 81-day arithmetic average of observed F10.7 centred on the date.
    double f107a_center81 = 0;

    // NOTE: used for testing.
    auto operator<=>(const sw_data_row &) const = default;

private:
    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

// The SW data table.
using sw_data_table = std::vector<sw_data_row>;

// The SW data class.
//
// This data class stores internally a table of SW data. The default constructor
// uses a builtin copy of the SW data from celestrak. Factory functions are
// available to download the latest datafiles from celestrak and other SW data providers.
class HEYOKA_DLL_PUBLIC sw_data
{
    struct impl;
    std::shared_ptr<const impl> m_impl;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: private implementation-detail constructor.
    explicit sw_data(sw_data_table, std::string, std::string);

public:
    sw_data();

    [[nodiscard]] const sw_data_table &get_table() const noexcept;
    [[nodiscard]] const std::string &get_timestamp() const noexcept;
    [[nodiscard]] const std::string &get_identifier() const noexcept;

    static sw_data fetch_latest_celestrak(bool = false);
};

namespace detail
{

void validate_sw_data_table(const sw_data_table &);

[[nodiscard]] HEYOKA_DLL_PUBLIC sw_data_table parse_sw_data_celestrak(const std::string &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
