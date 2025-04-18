// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EOP_SW_HELPERS_HPP
#define HEYOKA_DETAIL_EOP_SW_HELPERS_HPP

#include <functional>
#include <string_view>

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

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
