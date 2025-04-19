// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/core.h>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/sw_data/builtin_sw_data.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

void sw_data_row::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << mjd;
    oa << Ap_avg;
    oa << f107;
    oa << f107a_center81;
}

void sw_data_row::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> mjd;
    ia >> Ap_avg;
    ia >> f107;
    ia >> f107a_center81;
}

namespace detail
{

// Helper to validate a SW data table.
// NOTE: this must be called by every fetch_latest_*() function before passing the table to the sw_data constructor.
void validate_sw_data_table(const sw_data_table &data)
{
    const auto n_entries = data.size();

    for (decltype(data.size()) i = 0; i < n_entries; ++i) {
        // All mjd values must be finite and ordered in strictly ascending order.
        const auto cur_mjd = data[i].mjd;
        if (!std::isfinite(cur_mjd)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(
                fmt::format("Invalid SW data table detected: the MJD value {} on line {} is not finite", cur_mjd, i));
            // LCOV_EXCL_STOP
        }
        // NOTE: if data[i + 1u].mjd is NaN, then cur_mjd >= data[i + 1u].mjd evaluates
        // to false and we will throw on the next iteration when we detect a non-finite
        // value for the mjd.
        if (i + 1u != n_entries && cur_mjd >= data[i + 1u].mjd) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("Invalid SW data table detected: the MJD value {} "
                                                    "on line {} is not less than the MJD value in the next line ({})",
                                                    // LCOV_EXCL_STOP
                                                    cur_mjd, i, data[i + 1u].mjd));
        }

        // f107 values must be finite and non-negative.
        const auto cur_f107 = data[i].f107;
        if (!std::isfinite(cur_f107) || cur_f107 < 0) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid SW data table detected: the f107 value {} on line {} is invalid", cur_f107, i));
        }

        // f107a_center81 values must be finite and non-negative.
        const auto cur_f107a_center81 = data[i].f107a_center81;
        if (!std::isfinite(cur_f107a_center81) || cur_f107a_center81 < 0) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid SW data table detected: the f107a_center81 value {} on line {} is invalid",
                            cur_f107a_center81, i));
        }
    }
}

} // namespace detail

struct sw_data::impl {
    sw_data_table m_data;
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

void sw_data::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << m_impl;
}

void sw_data::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> m_impl;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
sw_data::sw_data(sw_data_table data, std::string timestamp, std::string identifier)
    : m_impl(std::make_shared<const impl>(std::move(data), std::move(timestamp), std::move(identifier)))
{
}

sw_data::sw_data()
    : m_impl(std::make_shared<const impl>(
          sw_data_table(std::ranges::begin(detail::builtin_sw_data), std::ranges::end(detail::builtin_sw_data)),
          // NOTE: the builtin SW data is from celestrak's SW-Last5Years.csv file.
          detail::builtin_sw_data_ts, "celestrak_long_term"))
{
}

const sw_data_table &sw_data::get_table() const noexcept
{
    return m_impl->m_data;
}

const std::string &sw_data::get_timestamp() const noexcept
{
    return m_impl->m_timestamp;
}

const std::string &sw_data::get_identifier() const noexcept
{
    return m_impl->m_identifier;
}

namespace detail
{

llvm::Value *llvm_get_sw_data_Ap_avg(llvm_state &s, const sw_data &data, llvm::Type *scal_t)
{
    return llvm_get_eop_sw_data(
        s, data, scal_t, "Ap_avg",
        [&s, scal_t](const sw_data_row &r) {
            return llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{static_cast<double>(r.Ap_avg)}));
        },
        "sw");
}

// NOTE: these are all similar, use a macro (yuck) to avoid repetition.
#define HEYOKA_LLVM_GET_SW_DATA_IMPL(name)                                                                             \
    llvm::Value *llvm_get_sw_data_##name(llvm_state &s, const sw_data &data, llvm::Type *scal_t)                       \
    {                                                                                                                  \
        return llvm_get_eop_sw_data(                                                                                   \
            s, data, scal_t, #name,                                                                                    \
            [&s, scal_t](const sw_data_row &r) {                                                                       \
                return llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{r.name}));                            \
            },                                                                                                         \
            "sw");                                                                                                     \
    }

HEYOKA_LLVM_GET_SW_DATA_IMPL(f107)
HEYOKA_LLVM_GET_SW_DATA_IMPL(f107a_center81)

#undef HEYOKA_LLVM_GET_SW_DATA_IMPL

} // namespace detail

HEYOKA_END_NAMESPACE
