// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/eop_data/builtin_eop_data.hpp>
#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>

// NOTE: GCC warns about use of mismatched new/delete
// when creating global variables. I am not sure this is
// a real issue, as it looks like we are adopting the "canonical"
// approach for the creation of global variables (at least
// according to various sources online)
// and clang is not complaining. But let us revisit
// this issue in later LLVM versions.
#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

#endif

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
          // NOTE: the builtin EOP data is from IERS' rapid finals2000A.all file.
          detail::builtin_eop_data_ts, "iers_rapid_finals2000A_all"))
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

namespace
{

// This is a helper that will construct into an LLVM module a global
// array of scalars from the input eop data. A pointer to the beginning of
// the array will be returned. If the array already exists, a pointer
// to the existing array will be returned instead.
//
// 'data' is the source of eop data. scal_t is the scalar floating-point type
// to be used to codegen the data. arr_name is a name uniquely identifying
// the array to be created. value_getter is the function object that will be used
// to construct an array value from a row in the input eop data table.
//
// NOTE: this function will ensure that the size of the global array is representable
// as a 32-bit int.
llvm::Value *llvm_get_eop_data(llvm_state &s, const eop_data &data, llvm::Type *scal_t,
                               const std::string_view &arr_name,
                               const std::function<double(const eop_data_row &)> &value_getter)
{
    assert(scal_t != nullptr);
    assert(!llvm::isa<llvm::FixedVectorType>(scal_t));
    assert(value_getter);

    auto &md = s.module();

    // Fetch the table.
    const auto &table = data.get_table();

    // Assemble the mangled name for the array. The mangled name will be based on:
    //
    // - arr_name,
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data,
    // - the scalar type used for the codegen.
    const auto name = fmt::format("heyoka.eop_data_{}.{}.{}_{}.{}", arr_name, table.size(), data.get_timestamp(),
                                  data.get_identifier(), llvm_mangle_type(scal_t));

    // Helper to return a pointer to the first element of the array.
    const auto fetch_ptr = [&bld = s.builder()](llvm::GlobalVariable *g_arr) {
        return bld.CreateInBoundsGEP(g_arr->getValueType(), g_arr, {bld.getInt32(0), bld.getInt32(0)});
    };

    // Check if we already codegenned the array.
    if (auto *gv = md.getGlobalVariable(name)) {
        // We already codegenned the array, return a pointer to the first element.
        return fetch_ptr(gv);
    }

    // We need to create a new array. Begin with the array type.
    // NOTE: array size needs a 64-bit int, but we want to guarantee that the array size fits in a 32-bit int.
    auto *arr_type = llvm::ArrayType::get(scal_t, boost::numeric_cast<std::uint32_t>(table.size()));

    // Construct the vector of initialisers.
    std::vector<llvm::Constant *> data_init;
    data_init.reserve(table.size());
    for (const auto &row : table) {
        // Get the value from the row.
        const auto value = value_getter(row);

        // Add it to the vector of initialisers.
        data_init.push_back(llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{value})));
    }

    // Create the array.
    // NOTE: we use linkonce_odr linkage so that we do not get duplicate definitions of the
    // same data in multiple llvm modules.
    auto *arr = llvm::ConstantArray::get(arr_type, data_init);
    auto *g_arr = new llvm::GlobalVariable(md, arr_type, true, llvm::GlobalVariable::LinkOnceODRLinkage, arr, name);

    // Return a pointer to the first element.
    return fetch_ptr(g_arr);
}

} // namespace

// eop data getter for the date measured in TT Julian centuries since J2000.0.
llvm::Value *llvm_get_eop_data_date_tt_cy_j2000(llvm_state &s, const eop_data &data, llvm::Type *scal_t)
{
    auto *logger = get_logger();

    const auto value_getter = [logger](const eop_data_row &r) {
        // Fetch the UTC mjd.
        const auto utc_mjd = r.mjd;

        // Turn it into a UTC jd.
        const auto utc_jd1 = 2400000.5;
        const auto utc_jd2 = utc_mjd;

        // Convert it into a TAI julian date.
        double tai_jd1{}, tai_jd2{};
        const auto ret = eraUtctai(utc_jd1, utc_jd2, &tai_jd1, &tai_jd2);
        // LCOV_EXCL_START
        if (ret == -1) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Unable to convert the UTC Julian date ({}, {}) into a TAI Julian date", utc_jd1, utc_jd2));
        }
        if (ret == 1) [[unlikely]] {
            logger->warn("Potentially inaccurate UTC->TAI conversion detected for the UTC Julian date ({}, {})",
                         utc_jd1, utc_jd2);
        }
        // LCOV_EXCL_STOP

        // Transform TAI into TT.
        double tt_jd1{}, tt_jd2{};
        eraTaitt(tai_jd1, tai_jd2, &tt_jd1, &tt_jd2);

        // Normalise.
        std::tie(tt_jd1, tt_jd2) = eft_add_knuth(tt_jd1, tt_jd2);

        // Compute in dfloat the number of days since J2000.0.
        const auto ndays = dfloat(tt_jd1, tt_jd2) - dfloat(2451545.0, 0.);

        // Convert to Julian centuries and return.
        const auto retval = static_cast<double>(ndays) / 36525;
        if (!std::isfinite(retval)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "The conversion of the UTC MJD {} to TT centuries since J2000.0 produced the non-finite value {}",
                utc_mjd, retval));
            // LCOV_EXCL_STOP
        }
        return retval;
    };

    return llvm_get_eop_data(s, data, scal_t, "date_tt_cy_j2000", value_getter);
}

// eop data getter for the ERA.
llvm::Value *llvm_get_eop_data_era(llvm_state &s, const eop_data &data, llvm::Type *scal_t)
{
    auto *logger = get_logger();

    const auto value_getter = [logger](const eop_data_row &r) {
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
        const auto ret = eraUtcut1(utc_jd1, utc_jd2, dut1, &ut1_jd1, &ut1_jd2);
        if (ret == -1) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Unable to convert the UTC Julian date ({}, {}) into a UT1 Julian date", utc_jd1, utc_jd2));
        }
        if (ret == 1) [[unlikely]] {
            logger->warn("Potentially inaccurate UTC->UT1 conversion detected for the UTC Julian date ({}, {})",
                         utc_jd1, utc_jd2);
        }
        // LCOV_EXCL_STOP

        // Compute the ERA and return.
        const auto retval = eraEra00(ut1_jd1, ut1_jd2);
        if (!std::isfinite(retval)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "The computation of the ERA at the UTC MJD {} produced the non-finite value {}", utc_mjd, retval));
            // LCOV_EXCL_STOP
        }
        return retval;
    };

    return llvm_get_eop_data(s, data, scal_t, "era", value_getter);
}

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
