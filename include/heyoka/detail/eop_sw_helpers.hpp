// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EOP_SW_HELPERS_HPP
#define HEYOKA_DETAIL_EOP_SW_HELPERS_HPP

#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>

// NOTE: this header contains implementation details common to EOP and SW data.

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename Data>
[[nodiscard]] llvm::Value *
llvm_get_eop_sw_data(llvm_state &, const Data &, llvm::Type *, const std::string_view &,
                     const std::function<llvm::Constant *(const typename Data::row_type &)> &,
                     const std::string_view &);

template <typename Data>
[[nodiscard]] llvm::Value *llvm_get_eop_sw_data_date_tt_cy_j2000(llvm_state &, const Data &, llvm::Type *,
                                                                 const std::string_view &);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *llvm_eop_sw_data_locate_date(llvm_state &, llvm::Value *, llvm::Value *,
                                                                          llvm::Value *);

// NOTE: small generic helper to turn a range of rows into an EOP/SW data table of type T. This assumes that R is an
// input range whose reference type is the row type of T.
template <typename T, typename R>
T eop_sw_table_from_range(R &&r)
{
    static_assert(std::ranges::input_range<T>);
    static_assert(std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<R>>, typename T::value_type>);

    if constexpr (std::same_as<T, std::remove_cvref_t<R>>) {
        return std::forward<R>(r);
    } else {
        return std::ranges::to<T>(r);
    }
}

void eop_sw_check_ts_id(std::string_view, const std::string &, const std::string &, bool,
                        std::span<const std::string_view>);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
