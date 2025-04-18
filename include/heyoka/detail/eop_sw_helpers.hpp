// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EOP_SW_HELPERS_HPP
#define HEYOKA_DETAIL_EOP_SW_HELPERS_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <cassert>
#include <cstdint>
#include <functional>
#include <string_view>
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

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

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

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif

#endif
