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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
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
#include <llvm/Support/Alignment.h>
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
// array of values from the input eop data. A pointer to the beginning of
// the array will be returned. If the array already exists, a pointer
// to the existing array will be returned instead.
//
// 'data' is the source of eop data. value_t is the value type of the array. arr_name
// is a name uniquely identifying the array to be created. value_getter is the function
// object that will be used to construct an array value from a row in the input eop data table.
//
// NOTE: this function will ensure that the size of the global array is representable
// as a 32-bit int.
llvm::Value *llvm_get_eop_data(llvm_state &s, const eop_data &data, llvm::Type *value_t,
                               const std::string_view &arr_name,
                               const std::function<llvm::Constant *(const eop_data_row &)> &value_getter)
{
    assert(value_t != nullptr);
    assert(value_getter);

    auto &md = s.module();

    // Fetch the table.
    const auto &table = data.get_table();

    // Assemble the mangled name for the array. The mangled name will be based on:
    //
    // - arr_name,
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data,
    // - the value type of the array.
    const auto name = fmt::format("heyoka.eop_data_{}.{}.{}_{}.{}", arr_name, table.size(), data.get_timestamp(),
                                  data.get_identifier(), llvm_mangle_type(value_t));

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
    auto *arr_type = llvm::ArrayType::get(value_t, boost::numeric_cast<std::uint32_t>(table.size()));

    // Construct the vector of initialisers.
    std::vector<llvm::Constant *> data_init;
    data_init.reserve(table.size());
    for (const auto &row : table) {
        // Get the value from the row.
        auto *value = value_getter(row);
        assert(value->getType() == value_t);

        // Add it to the vector of initialisers.
        data_init.push_back(value);
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

    const auto value_getter = [logger, &s, scal_t](const eop_data_row &r) {
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
        // NOTE: in principle the truncation of retval to lower-than-double precision could
        // result in an infinity being produced. This is still ok, as the important thing for the
        // dates array is the absence of NaNs. If infinities are generated, the bisection algorithm
        // will still work (although we will likely generate NaNs once we start using the infinite
        // date in calculations).
        return llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{retval}));
    };

    return llvm_get_eop_data(s, data, scal_t, "date_tt_cy_j2000", value_getter);
}

// eop data getter for the ERA.
//
// The ERA will be represented as a double-length floating-point, hence the value type of the
// global array is itself a 2-elements array (of type scal_t).
llvm::Value *llvm_get_eop_data_era(llvm_state &s, const eop_data &data, llvm::Type *scal_t)
{
    auto *logger = get_logger();

    // NOTE: for the ERA data specifically, we want to make sure that the array size x 2 is representable
    // as a 32-bit int. The reason for this is that in the implementation of the era/erap functions,
    // we will be reinterpreting the array of size-2 arrays as a 1D flattened array, into which we want
    // to be able to index via 32-bit ints.
    if (data.get_table().size() > std::numeric_limits<std::uint32_t>::max() / 2u) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::overflow_error("Overflow detected while generating the LLVM ERA data");
        // LCOV_EXCL_STOP
    }

    // Determine the value type.
    auto *value_t = llvm::ArrayType::get(scal_t, 2);

    const auto value_getter = [logger, &s, scal_t, value_t](const eop_data_row &r) {
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

        // Compute the ERA and return. See:
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
        // in attempting to generalise this process to precisions higher than double.

        // Use octuple precision as a safety margin.
        using oct_t = boost::multiprecision::cpp_bin_float_oct;
        const auto tU = oct_t{ut1_jd1} + ut1_jd2 - 2451545.0;
        const auto twopi = 2 * boost::math::constants::pi<oct_t>();
        const auto era = twopi * (0.7790572732640 + 1.00273781191135448 * tU);

        // Compute the hi/lo components of the double-double era approximation.
        auto era_hi = static_cast<double>(era);
        auto era_lo = static_cast<double>(era - era_hi);

        // Normalise them for peace of mind. This should not be necessary assuming Boost multiprecision
        // rounds correctly, but better safe than sorry.
        std::tie(era_hi, era_lo) = eft_add_knuth(era_hi, era_lo);

        if (!std::isfinite(era_hi) || !std::isfinite(era_lo)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "The computation of the ERA at the UTC MJD {} produced the non-finite double-length value ({}, {})",
                utc_mjd, era_hi, era_lo));
            // LCOV_EXCL_STOP
        }

        // Pack the values into an array and return.
        auto *retval
            = llvm::ConstantArray::get(value_t, {llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{era_hi})),
                                                 llvm::cast<llvm::Constant>(llvm_codegen(s, scal_t, number{era_lo}))});
        return retval;
    };

    return llvm_get_eop_data(s, data, value_t, "era", value_getter);
}

// Implementation of the std::upper_bound() algorithm for use in llvm_eop_data_locate_date().
//
// Given an array of sorted scalar values beginning at ptr and of size arr_size (a 32-bit int), this function
// will return the index of the first element in the array that is *greater than* v. If no such element exists,
// arr_size will be returned. v can be a scalar or a vector.
//
// The algorithm is short enough to be reproduced here:
//
// template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type, class Compare>
// ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
// {
//     ForwardIt it;
//     typename std::iterator_traits<ForwardIt>::difference_type count, step;
//     count = std::distance(first, last);
//
//     while (count > 0)
//     {
//         it = first;
//         step = count / 2;
//         std::advance(it, step);
//
//         if (!comp(value, *it))
//         {
//             first = ++it;
//             count -= step + 1;
//         }
//         else
//             count = step;
//     }
//
//     return first;
// }
//
// Particular care must be taken for the vector implementation: while in a scalar implementation
// the bisection loop is never entered if count == 0, in the vector implementation we will be entering
// the bisection loop with count == 0 whenever a SIMD lane has finished but the other SIMD lanes have not.
// In a loop iteration with count == 0, the following happens:
//
// - step is set to 0 and 'it' remains inited to 'first';
// - 'it' may be pointing one past the end of the array, and thus we must take care
//   of *not* dereferencing it if that is the case. We thus have two possibilities:
//   - 'it' points somewhere in the array: in this case, we know that 'it' points to
//     the first array element greater than 'v', and thus '!comp(value, *it)' evaluates
//     to false, we end up in the 'count = step' branch, where count is set again
//     to 0. Thus, neither 'first' nor 'count' are altered;
//   - 'it' points one past the end: in this case, we must avoid reading from it and we must
//     replace the condition '!comp(value, *it)' with 'false', so that we end up in the
//     'count = step' branch.
llvm::Value *llvm_eop_data_upper_bound(llvm_state &s, llvm::Value *ptr, llvm::Value *arr_size, llvm::Value *v)
{
    assert(ptr != nullptr);
    // NOTE: this will also check that ptr is not a vector of pointers.
    assert(ptr->getType()->isPointerTy());
    assert(arr_size != nullptr);
    assert(v != nullptr);

    auto &bld = s.builder();

    // NOTE: infer the scalar type from v.
    auto *scal_t = v->getType()->getScalarType();

    // Fetch the 32-bit int type.
    auto *int32_tp = bld.getInt32Ty();
    // NOTE: this will also check that arr_size is not a vector of values.
    assert(arr_size->getType() == int32_tp);

    // Determine the batch size.
    std::uint32_t batch_size = 1;
    if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(v->getType())) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
        assert(batch_size != 0u);
    }

    // Splat ptr to the batch size.
    auto *arr_ptr = vector_splat(bld, ptr, batch_size);

    // The type to be used for indexing into the scalar array.
    auto *idx_vec_t = make_vector_type(int32_tp, batch_size);

    // The type that will be loaded from the scalar array.
    auto *fp_vec_t = make_vector_type(scal_t, batch_size);

    // Create arrays of 32-bit constants for use below.
    auto *zero_vec_i32 = vector_splat(bld, bld.getInt32(0), batch_size);
    auto *one_vec_i32 = vector_splat(bld, bld.getInt32(1), batch_size);
    auto *two_vec_i32 = vector_splat(bld, bld.getInt32(2), batch_size);

    // Create the 'first' and 'count' variables.
    // NOTE: as "iterator" type we will be using a 32-bit int.
    auto *first = bld.CreateAlloca(idx_vec_t);
    auto *count = bld.CreateAlloca(idx_vec_t);

    // The 'first' iterator is inited with zeros (as it points to the beginning of the array).
    bld.CreateStore(zero_vec_i32, first);

    // 'count' is inited with the size of the array.
    auto *arr_size_splat = vector_splat(bld, arr_size, batch_size);
    bld.CreateStore(arr_size_splat, count);

    // Iterate as long as all elements of 'count' are > 0.
    llvm_while_loop(
        s,
        [&bld, count, idx_vec_t, zero_vec_i32, batch_size]() -> llvm::Value * {
            auto *cmp = bld.CreateICmpUGT(bld.CreateLoad(idx_vec_t, count), zero_vec_i32);

            // NOTE: in scalar mode, no reduction is needed.
            return (batch_size == 1u) ? cmp : bld.CreateOrReduce(cmp);
        },
        [&bld, &s, first, count, idx_vec_t, batch_size, scal_t, fp_vec_t, v, one_vec_i32, arr_ptr, arr_size_splat,
         two_vec_i32]() {
            // Load the value stored in 'first' - this will be the iterator we will
            // be using in the current iteration of the loop.
            llvm::Value *cur_first = bld.CreateLoad(idx_vec_t, first);
            auto *it = cur_first;

            // Compute the step value for the current iteration: step = count / 2.
            llvm::Value *cur_count = bld.CreateLoad(idx_vec_t, count);
            auto *step = bld.CreateUDiv(cur_count, two_vec_i32);

            // Advance 'it' by step.
            it = bld.CreateAdd(it, step);

            // Load the value(s) from 'it' into 'cur_value'. 'mask' is used only in vector mode, otherwise it
            // remains null.
            llvm::Value *cur_value{}, *mask{};
            if (batch_size == 1u) {
                // Normal scalar load.
                cur_value = bld.CreateLoad(scal_t, bld.CreateInBoundsGEP(scal_t, arr_ptr, {it}));
            } else {
                // NOTE: as explained above, in vector mode we must take care to avoid loading from 'it'
                // if it points one past the end. We accomplish this with a masked gather.

                // Fetch the alignment of the scalar type.
                const auto align = get_alignment(s.module(), scal_t);

                // Identify the SIMD lane(s) which are *not* reading past the end of the array, storing the
                // result in 'mask'.
                mask = bld.CreateICmpNE(it, arr_size_splat);

                // As a passthru value for the masked gather, use the v value itself.
                auto *passthru = v;

                // Masked gather with passthru.
                cur_value = bld.CreateMaskedGather(fp_vec_t, bld.CreateInBoundsGEP(scal_t, arr_ptr, {it}),
                                                   llvm::Align(align), mask, passthru);
            }

            // Run the comparison.
            // NOTE: the original comparison would be '!comp(value, *it)', which translates to '!(v < *it)' in
            // the current code. *it can never be NaN (apart from a corner case in vector mode when the passthru
            // value is NaN, but we take care of this later ANDing the mask). v could be NaN, in which case we want
            // '!(v < *it)' to evaluate to true because we want to consider NaN greater than non-NaN. In order to do
            // this, we flip the comparison around to '*it <= v', and we implement it via the ULE predicate, which
            // returns true if either v is NaN or '*it <= v'.
            auto *cmp = llvm_fcmp_ule(s, cur_value, v);
            if (batch_size != 1u) {
                // NOTE: in vector mode, we must take care that cmp for the masked-out lanes evaluates to false.
                assert(mask != nullptr);
                cmp = bld.CreateAnd(cmp, mask);
            }

            // We now need to update 'first' and 'count'. Branch on the batch size for efficiency.
            if (batch_size == 1u) {
                // Scalar implementation.
                llvm_if_then_else(
                    s, cmp,
                    [&bld, one_vec_i32, it, first, count, step, cur_count]() {
                        // Assign it + 1 to first.
                        auto *itp1 = bld.CreateAdd(it, one_vec_i32);
                        bld.CreateStore(itp1, first);

                        // Compute new_count = count - (step + 1).
                        auto *stepp1 = bld.CreateAdd(step, one_vec_i32);
                        auto *new_count = bld.CreateSub(cur_count, stepp1);

                        // Assign new_count to count.
                        bld.CreateStore(new_count, count);
                    },
                    [&bld, step, count]() {
                        // NOTE: no update of 'first' needed here.
                        // Assign step to count.
                        bld.CreateStore(step, count);
                    });
            } else {
                // Vector implementation.

                // Compute it + 1.
                auto *itp1 = bld.CreateAdd(it, one_vec_i32);

                // Compute step + 1.
                auto *stepp1 = bld.CreateAdd(step, one_vec_i32);

                // Compute count - (step + 1).
                auto *count_m_stepp1 = bld.CreateSub(cur_count, stepp1);

                // Compute the new first = cmp ? (it + 1) : cur_first.
                auto *new_first = bld.CreateSelect(cmp, itp1, cur_first);

                // Compute the new count = cmp ? (count - (step + 1)) : step.
                auto *new_count = bld.CreateSelect(cmp, count_m_stepp1, step);

                // Store the new first and the new count.
                bld.CreateStore(new_count, count);
                bld.CreateStore(new_first, first);
            }
        });

    // The return value is the value stored in 'first'.
    return bld.CreateLoad(idx_vec_t, first);
}

// Given an array of sorted scalar date values beginning at ptr and of size arr_size (a 32-bit int), and a date value
// date_value, this function will return the index idx in the array such that ptr[idx] <= date_value < ptr[idx + 1].
//
// In other words, this function will regard a sorted array of N dates as a sequence of N-1 contiguous half-open time
// intervals, and it will return the index of the interval containing the input date_value.
//
// If either:
//
// - arr_size == 0, or
// - date_value is less than the first value in the array, or
// - date_value is greater than or equal to the last value in the array, or
// - date_value is NaN,
//
// then arr_size will be returned. date_value can be a scalar or a vector.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_eop_data_locate_date(llvm_state &s, llvm::Value *ptr, llvm::Value *arr_size, llvm::Value *date_value)
{
    auto &bld = s.builder();

    // As a first step, we look for the first date in the array which is *greater than* date_value.
    auto *idx = llvm_eop_data_upper_bound(s, ptr, arr_size, date_value);

    // The happy path requires the computation of idx - 1. The exceptions are:
    //
    // - idx == 0 -> this means that date_value is before any time interval,
    // - idx == arr_size -> this means that data_value is after any time interval.
    //
    // In these two cases, we want to return arr_size, rather than idx - 1.

    // Determine the batch size from date_value.
    std::uint32_t batch_size = 1;
    if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(date_value->getType())) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
        assert(batch_size != 0u);
    }

    // Splat the array size.
    auto *arr_size_splat = vector_splat(bld, arr_size, batch_size);

    // First step: ret = (idx == 0) ? arr_size : idx.
    auto *idx_is_zero = bld.CreateICmpEQ(idx, llvm::ConstantInt::get(idx->getType(), 0));
    auto *ret = bld.CreateSelect(idx_is_zero, arr_size_splat, idx);

    // Second step: ret = (ret == arr_size) ? ret : (ret - 1).
    auto *ret_eq_arr_size = bld.CreateICmpEQ(ret, arr_size_splat);
    ret = bld.CreateSelect(ret_eq_arr_size, ret, bld.CreateSub(ret, llvm::ConstantInt::get(ret->getType(), 1)));

    return ret;
}

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
