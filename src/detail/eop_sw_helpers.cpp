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
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/sw_data.hpp>

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

namespace detail
{

// This is a helper that will construct into an LLVM module a global
// array of values from the input eop/sw data. A pointer to the beginning of
// the array will be returned. If the array already exists, a pointer
// to the existing array will be returned instead.
//
// 'data' is the source of eop/sw data. value_t is the value type of the array. arr_name
// is a name uniquely identifying the array to be created. value_getter is the function
// object that will be used to construct a single value in the array from a row in the
// input data table. data_id is a string identifying the type of data (eop/sw).
//
// NOTE: this function will ensure that the size of the global array is representable
// as a 32-bit int.
template <typename Data>
llvm::Value *llvm_get_eop_sw_data(llvm_state &s, const Data &data, llvm::Type *value_t,
                                  const std::string_view &arr_name,
                                  const std::function<llvm::Constant *(const typename Data::row_type &)> &value_getter,
                                  const std::string_view &data_id)
{
    assert(value_t != nullptr);
    assert(value_getter);

    auto &md = s.module();

    // Fetch the table.
    const auto &table = data.get_table();

    // Assemble the mangled name for the array. The mangled name will be based on:
    //
    // - data_id,
    // - arr_name,
    // - the total number of rows in the eop/sw data table,
    // - the timestamp and identifier of the eop/sw data,
    // - the value type of the array.
    const auto name = fmt::format("heyoka.{}_data_{}.{}.{}_{}.{}", data_id, arr_name, table.size(),
                                  data.get_timestamp(), data.get_identifier(), llvm_mangle_type(value_t));

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
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *g_arr = new llvm::GlobalVariable(md, arr_type, true, llvm::GlobalVariable::LinkOnceODRLinkage, arr, name);

    // Return a pointer to the first element.
    return fetch_ptr(g_arr);
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Value *
llvm_get_eop_sw_data(llvm_state &, const eop_data &, llvm::Type *, const std::string_view &,
                     const std::function<llvm::Constant *(const eop_data::row_type &)> &, const std::string_view &);
template HEYOKA_DLL_PUBLIC llvm::Value *
llvm_get_eop_sw_data(llvm_state &, const sw_data &, llvm::Type *, const std::string_view &,
                     const std::function<llvm::Constant *(const sw_data::row_type &)> &, const std::string_view &);

// eop/sw data getter for the date measured in TT Julian centuries since J2000.0.
template <typename Data>
llvm::Value *llvm_get_eop_sw_data_date_tt_cy_j2000(llvm_state &s, const Data &data, llvm::Type *scal_t,
                                                   const std::string_view &data_id)
{
    const auto value_getter = [&s, scal_t](const auto &r) {
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
        // LCOV_EXCL_STOP

        // Transform TAI into TT.
        double tt_jd1{}, tt_jd2{};
        ::eraTaitt(tai_jd1, tai_jd2, &tt_jd1, &tt_jd2);

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

    return llvm_get_eop_sw_data(s, data, scal_t, "date_tt_cy_j2000", value_getter, data_id);
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_eop_sw_data_date_tt_cy_j2000(llvm_state &, const eop_data &,
                                                                              llvm::Type *, const std::string_view &);
template HEYOKA_DLL_PUBLIC llvm::Value *llvm_get_eop_sw_data_date_tt_cy_j2000(llvm_state &, const sw_data &,
                                                                              llvm::Type *, const std::string_view &);

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
llvm::Value *llvm_eop_sw_data_locate_date(llvm_state &s, llvm::Value *ptr, llvm::Value *arr_size,
                                          llvm::Value *date_value)
{
    assert(ptr != nullptr);
    assert(ptr->getType()->isPointerTy());

    auto &bld = s.builder();
    auto &md = s.module();
    auto &ctx = s.context();

    assert(arr_size != nullptr);
    assert(arr_size->getType() == bld.getInt32Ty());

    // NOTE: we want to define a function to perform the computation in order to help LLVM elide redundant computations.
    // For instance, if we are computing both era/erap and pm_x/pm_xp for the *same* input date,
    // we would like LLVM to locate that date only once for the two calls.

    // Create the mangled name of the function. The mangled name will be based on the type of date_value,
    // which encodes both the value type of the array and the batch size.
    auto *val_t = date_value->getType();
    const auto fname = fmt::format("heyoka.eop_sw_data_locate_date.{}", llvm_mangle_type(val_t));

    // Check if we already created the function.
    if (auto *fptr = md.getFunction(fname)) {
        // The function was created already. Call it and return the result.
        return bld.CreateCall(fptr, {ptr, arr_size, date_value});
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // Determine the batch size from date_value.
    std::uint32_t batch_size = 1;
    if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(val_t)) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
        assert(batch_size != 0u);
    }

    // Determine the return type: either a single 32-bit int or a vector thereof.
    auto *ret_t = make_vector_type(bld.getInt32Ty(), batch_size);

    // Build the function prototype.
    auto *ft = llvm::FunctionType::get(ret_t, {bld.getPtrTy(), bld.getInt32Ty(), val_t}, false);

    // Create the function
    auto *f = llvm_func_create(ft, llvm::Function::PrivateLinkage, fname, &md);
    f->addFnAttr(llvm::Attribute::NoRecurse);
    f->addFnAttr(llvm::Attribute::NoUnwind);
    f->addFnAttr(llvm::Attribute::Speculatable);
    f->addFnAttr(llvm::Attribute::WillReturn);

    // Fetch the arguments.
    auto *ptr_arg = f->getArg(0);
    auto *arr_size_arg = f->getArg(1);
    auto *date_value_arg = f->getArg(2);

    // The pointer argument is read-only.
    ptr_arg->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // As a first step, we look for the first date in the array which is *greater than* date_value.
    auto *idx = llvm_upper_bound(s, ptr_arg, arr_size_arg, date_value_arg);

    // The happy path requires the computation of idx - 1. The exceptions are:
    //
    // - idx == 0 -> this means that date_value is before any time interval,
    // - idx == arr_size -> this means that data_value is after any time interval.
    //
    // In these two cases, we want to return arr_size, rather than idx - 1.

    // Splat the array size.
    auto *arr_size_splat = vector_splat(bld, arr_size_arg, batch_size);

    // First step: ret = (idx == 0) ? arr_size : idx.
    auto *idx_is_zero = bld.CreateICmpEQ(idx, llvm::ConstantInt::get(idx->getType(), 0));
    auto *ret = bld.CreateSelect(idx_is_zero, arr_size_splat, idx);

    // Second step: ret = (ret == arr_size) ? ret : (ret - 1).
    auto *ret_eq_arr_size = bld.CreateICmpEQ(ret, arr_size_splat);
    ret = bld.CreateSelect(ret_eq_arr_size, ret, bld.CreateSub(ret, llvm::ConstantInt::get(ret->getType(), 1)));

    // Create the return value.
    bld.CreateRet(ret);

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    // Invoke the function and return the result.
    return bld.CreateCall(f, {ptr, arr_size, date_value});
}

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
