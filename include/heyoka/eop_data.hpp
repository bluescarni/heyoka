// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EOP_DATA_HPP
#define HEYOKA_EOP_DATA_HPP

#include <compare>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

// Single row in a EOP data table.
struct HEYOKA_DLL_PUBLIC eop_data_row {
    // UTC modified Julian date.
    double mjd = 0;
    // UT1-UTC (seconds).
    double delta_ut1_utc = 0;

    // NOTE: used in testing.
    auto operator<=>(const eop_data_row &) const = default;

private:
    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

// The EOP data table.
using eop_data_table = std::vector<eop_data_row>;

// The EOP data class.
//
// This data class stores internally a table of EOP data. The default constructor
// uses a builtin copy of the finals2000A.all file from IERS. Factory functions are
// available to download the latest datafiles from IERS and other EOP data providers.
class HEYOKA_DLL_PUBLIC eop_data
{
    struct impl;
    std::shared_ptr<const impl> m_impl;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: private implementation-detail constructor.
    explicit eop_data(eop_data_table, std::string, std::string);

public:
    eop_data();

    [[nodiscard]] const eop_data_table &get_table() const noexcept;
    [[nodiscard]] const std::string &get_timestamp() const noexcept;
    [[nodiscard]] const std::string &get_identifier() const noexcept;

private:
    static std::pair<std::string, std::string> download(const std::string &, unsigned, const std::string &);

public:
    static eop_data fetch_latest_iers_rapid(const std::string & = "finals2000A.all");
    static eop_data fetch_latest_iers_long_term();
};

namespace detail
{

void validate_eop_data_table(const eop_data_table &);

[[nodiscard]] HEYOKA_DLL_PUBLIC eop_data_table parse_eop_data_iers_rapid(const std::string &);
[[nodiscard]] HEYOKA_DLL_PUBLIC eop_data_table parse_eop_data_iers_long_term(const std::string &);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_eop_data_date_tt_cy_j2000(llvm_state &, const eop_data &,
                                                                                llvm::Type *);
[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_eop_data_era(llvm_state &, const eop_data &, llvm::Type *);
[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_eop_data_upper_bound(llvm_state &, llvm::Value *, llvm::Value *,
                                                                       llvm::Value *);
[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_eop_data_locate_date(llvm_state &, llvm::Value *, llvm::Value *,
                                                                       llvm::Value *);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
