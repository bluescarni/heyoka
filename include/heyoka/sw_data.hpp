// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_SW_DATA_HPP
#define HEYOKA_SW_DATA_HPP

#include <compare>
#include <concepts>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

// Single row in an SW data table.
//
// Each row gives the values assumed by a set of space weather quantities at the instant specified by the modified
// Julian date mjd.
struct HEYOKA_DLL_PUBLIC sw_data_row {
    // UTC modified Julian date.
    double mjd = 0;
    // 24-hour running average of the Ap index centred on the date.
    double Ap_avg = 0;
    // Observed 10.7-cm solar radio flux (F10.7).
    double f107 = 0;
    // 81-day running average of observed F10.7 centred on the date.
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
    explicit sw_data(sw_data_table, std::string, std::string, bool);

public:
    using row_type = sw_data_row;

    sw_data();
    explicit sw_data(std::initializer_list<sw_data_row>, std::string, std::string);
    template <typename R>
        requires std::ranges::input_range<R>
                 && std::same_as<sw_data_row, std::remove_cvref_t<std::ranges::range_reference_t<R>>>
    explicit sw_data(R &&r, std::string timestamp, std::string identifier)
        : sw_data(detail::eop_sw_table_from_range<sw_data_table>(std::forward<R>(r)), std::move(timestamp),
                  std::move(identifier), false)
    {
    }

    [[nodiscard]] const sw_data_table &get_table() const noexcept;
    [[nodiscard]] const std::string &get_timestamp() const noexcept;
    [[nodiscard]] const std::string &get_identifier() const noexcept;

    static sw_data fetch_latest_celestrak(bool = false);
};

namespace detail
{

[[nodiscard]] HEYOKA_DLL_PUBLIC sw_data_table parse_sw_data_celestrak(const std::string &);
HEYOKA_DLL_PUBLIC void reanchor_sw_data_celestrak(sw_data_table &);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_sw_data_Ap_avg(llvm_state &, const sw_data &, llvm::Type *);
[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_sw_data_f107(llvm_state &, const sw_data &, llvm::Type *);
[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_sw_data_f107a_center81(llvm_state &, const sw_data &,
                                                                             llvm::Type *);

} // namespace detail

HEYOKA_END_NAMESPACE

// Version changelog:
//
// - version 1: switched AP_avg from std::uint16_t to double.
BOOST_CLASS_VERSION(heyoka::sw_data_row, 1)

#endif
