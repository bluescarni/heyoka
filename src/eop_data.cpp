// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/eop_data/builtin_eop_data.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

void eop_data_row::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << mjd;
    oa << delta_ut1_utc;
    oa << pm_x;
    oa << pm_y;
    oa << dX;
    oa << dY;
}

void eop_data_row::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> mjd;
    ia >> delta_ut1_utc;
    ia >> pm_x;
    ia >> pm_y;
    ia >> dX;
    ia >> dY;
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

        // PM values must be finite.
        const auto pm_x = data[i].pm_x;
        if (!std::isfinite(pm_x)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid EOP data table detected: the pm_x value {} on line {} is not finite", pm_x, i));
        }
        const auto pm_y = data[i].pm_y;
        if (!std::isfinite(pm_y)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid EOP data table detected: the pm_y value {} on line {} is not finite", pm_y, i));
        }

        // dX/dY values must be finite.
        const auto dX = data[i].dX;
        if (!std::isfinite(dX)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid EOP data table detected: the dX value {} on line {} is not finite", dX, i));
        }
        const auto dY = data[i].dY;
        if (!std::isfinite(dY)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid EOP data table detected: the dY value {} on line {} is not finite", dY, i));
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
          // NOTE: the builtin EOP data is from IERS' rapid finals2000A.all file downloaded from USNO.
          detail::builtin_eop_data_ts, "iers_rapid_usno_finals2000A_all"))
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

namespace detail
{

// eop data getter for the ERA.
//
// The ERA will be represented as a double-length floating-point, hence the value type of the
// global array is itself a 2-elements array (of type scal_t).
llvm::Value *llvm_get_eop_data_era(llvm_state &s, const eop_data &data, llvm::Type *scal_t)
{
    // Determine the value type.
    auto *value_t = llvm::ArrayType::get(scal_t, 2);

    const auto value_getter = [&s, scal_t, value_t](const eop_data_row &r) {
        // Fetch the UTC mjd.
        const auto utc_mjd = r.mjd;

        // Turn it into a UTC jd.
        const auto utc_jd1 = 2400000.5;
        const auto utc_jd2 = utc_mjd;

        // Fetch the UT1-UTC difference.
        const auto dut1 = r.delta_ut1_utc;

        // Convert the UTC jd into UT1.
        // LCOV_EXCL_START
        double ut1_jd1{}, ut1_jd2{};
        const auto ret = ::eraUtcut1(utc_jd1, utc_jd2, dut1, &ut1_jd1, &ut1_jd2);
        if (ret == -1) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Unable to convert the UTC Julian date ({}, {}) into a UT1 Julian date", utc_jd1, utc_jd2));
        }
        // LCOV_EXCL_STOP

        // Compute the ERA. See:
        //
        // https://en.wikipedia.org/wiki/Sidereal_time#ERA
        //
        // The approach we follow here is to first compute the ERA in multiprecision (without any reduction to the [0,
        // 2pi] range), and then to truncate and store the result as a double-double number. This gives us the ERA with
        // roughly 30 decimal digits of precision. Then, in the implementation of the era/erap functions in the
        // expression system, we will perform double-double interpolation and reduction to the [0, 2pi] range, and
        // finally we will truncate the result back to single-length representation.
        //
        // NOTE: at this time the ERA has an intrinsic accuracy no better than ~1e-10 rad, due to the ~1e-5s
        // uncertainty in the measurement of the UT1-UTC difference. Thus it seems like there's no point
        // in attempting to generalise this process to precisions higher than double. We can always generalise this
        // to higher precision in the future if needed.

        // Use octuple precision as a safety margin.
        //
        // NOTE: cpp_bin_float_oct is defined as having expression templates turned off, thus we are ok with the use of
        // auto throughout the computation.
        using oct_t = boost::multiprecision::cpp_bin_float_oct;
        const auto tU = oct_t{ut1_jd1} + ut1_jd2 - 2451545.0;
        const auto twopi = 2 * boost::math::constants::pi<oct_t>();
        const auto era = twopi * (0.7790572732640 + 1.00273781191135448 * tU);

        // Compute the hi/lo components of the double-double era approximation.
        auto era_hi = static_cast<double>(era);
        auto era_lo = static_cast<double>(era - era_hi);

        // Normalise them for peace of mind. This should not be necessary assuming Boost multiprecision rounds
        // correctly, but better safe than sorry.
        std::tie(era_hi, era_lo) = eft_add_knuth(era_hi, era_lo);

        if (!std::isfinite(era_hi) || !std::isfinite(era_lo)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "The computation of the ERA at the UTC MJD {} produced the non-finite double-length value ({}, {})",
                utc_mjd, era_hi, era_lo));
            // LCOV_EXCL_STOP
        }

        // Pack the values into an array and return.
        return llvm::ConstantArray::get(value_t, {llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{era_hi})),
                                                  llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{era_lo}))});
    };

    return llvm_get_eop_sw_data(s, data, value_t, "era", value_getter, "eop");
}

// eop data getter for the gmst82.
//
// The gmst82 will be represented as a double-length floating-point, hence the value type of the
// global array is itself a 2-elements array (of type scal_t).
llvm::Value *llvm_get_eop_data_gmst82(llvm_state &s, const eop_data &data, llvm::Type *scal_t)
{
    // Determine the value type.
    auto *value_t = llvm::ArrayType::get(scal_t, 2);

    const auto value_getter = [&s, scal_t, value_t](const eop_data_row &r) {
        // Fetch the UTC mjd.
        const auto utc_mjd = r.mjd;

        // Turn it into a UTC jd.
        const auto utc_jd1 = 2400000.5;
        const auto utc_jd2 = utc_mjd;

        // Fetch the UT1-UTC difference.
        const auto dut1 = r.delta_ut1_utc;

        // Convert the UTC jd into UT1.
        // LCOV_EXCL_START
        double ut1_jd1{}, ut1_jd2{};
        const auto ret = ::eraUtcut1(utc_jd1, utc_jd2, dut1, &ut1_jd1, &ut1_jd2);
        if (ret == -1) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Unable to convert the UTC Julian date ({}, {}) into a UT1 Julian date", utc_jd1, utc_jd2));
        }
        // LCOV_EXCL_STOP

        // Compute the gmst82. We use eq. (2) from:
        //
        // https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf
        //
        // The approach we follow here is to first compute the gmst82 in multiprecision (without any reduction to the
        // [0, 2pi] range), and then to truncate and store the result as a double-double number. This gives us the
        // gmst82 with roughly 30 decimal digits of precision. Then, in the implementation of the gmst82/gmst82p
        // functions in the expression system, we will perform double-double interpolation and reduction to the [0, 2pi]
        // range, and finally we will truncate the result back to single-length representation.
        //
        // NOTE: at this time the gmst82 has an intrinsic accuracy no better than ~1e-10 rad, due to the ~1e-5s
        // uncertainty in the measurement of the UT1-UTC difference. Thus it seems like there's no point
        // in attempting to generalise this process to precisions higher than double. We can always generalise this
        // to higher precision in the future if needed.

        // Use octuple precision as a safety margin.
        //
        // NOTE: cpp_bin_float_oct is defined as having expression templates turned off, thus we are ok with the use of
        // auto throughout the computation.
        using oct_t = boost::multiprecision::cpp_bin_float_oct;

        // Assemble the UT1 Julian date.
        auto tUT1 = (oct_t{ut1_jd1} + ut1_jd2);

        // Turn it into Julian centuries elapsed since JD 2451545.0, as requested by the formula. See also the "Sidereal
        // Time" section in the Vallado book.
        tUT1 = (tUT1 - 2451545.0) / 36525;

        // The coefficients for the expression of the gmst82, in seconds.
        const auto c0 = 67310.54841;
        const auto c1 = oct_t{876600.} * 3600 + 8640184.812866;
        const auto c2 = 0.093104;
        const auto c3 = -6.2e-6;

        // Compute the value of the gmst82.
        auto gmst82 = c0 + (c1 + (c2 + c3 * tUT1) * tUT1) * tUT1;

        // Convert it to radians.
        gmst82 = gmst82 * boost::math::constants::pi<oct_t>() / 43200;

        // Compute the hi/lo components of the double-double gmst82 approximation.
        auto gmst82_hi = static_cast<double>(gmst82);
        auto gmst82_lo = static_cast<double>(gmst82 - gmst82_hi);

        // Normalise them for peace of mind. This should not be necessary assuming Boost multiprecision rounds
        // correctly, but better safe than sorry.
        std::tie(gmst82_hi, gmst82_lo) = eft_add_knuth(gmst82_hi, gmst82_lo);

        if (!std::isfinite(gmst82_hi) || !std::isfinite(gmst82_lo)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "The computation of the gmst82 at the UTC MJD {} produced the non-finite double-length value ({}, {})",
                utc_mjd, gmst82_hi, gmst82_lo));
            // LCOV_EXCL_STOP
        }

        // Pack the values into an array and return.
        return llvm::ConstantArray::get(value_t,
                                        {llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{gmst82_hi})),
                                         llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{gmst82_lo}))});
    };

    return llvm_get_eop_sw_data(s, data, value_t, "gmst82", value_getter, "eop");
}

// Getters for the polar motion and dX/dY data.
// NOTE: these are all similar, use a macro (yuck) to avoid repetition.
// NOTE: like with the ERA, perform the computations in double-precision.
#define HEYOKA_LLVM_GET_EOP_DATA_IMPL(name, conv_factor)                                                               \
    llvm::Value *llvm_get_eop_data_##name(llvm_state &s, const eop_data &data, llvm::Type *scal_t)                     \
    {                                                                                                                  \
        return llvm_get_eop_sw_data(                                                                                   \
            s, data, scal_t, #name,                                                                                    \
            [&s, scal_t](const eop_data_row &r) {                                                                      \
                /* Fetch the value and convert. */                                                                     \
                const auto val = r.name * (conv_factor);                                                               \
                return llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{val}));                               \
            },                                                                                                         \
            "eop");                                                                                                    \
    }

// NOTE: PM data is in arcsec, dX/dY data in milliarcsec.
HEYOKA_LLVM_GET_EOP_DATA_IMPL(pm_x, boost::math::constants::pi<double>() / (180. * 3600));
HEYOKA_LLVM_GET_EOP_DATA_IMPL(pm_y, boost::math::constants::pi<double>() / (180. * 3600));
HEYOKA_LLVM_GET_EOP_DATA_IMPL(dX, boost::math::constants::pi<double>() / (180. * 3600 * 1000));
HEYOKA_LLVM_GET_EOP_DATA_IMPL(dY, boost::math::constants::pi<double>() / (180. * 3600 * 1000));

#undef HEYOKA_LLVM_GET_EOP_DATA_IMPL

} // namespace detail

HEYOKA_END_NAMESPACE
