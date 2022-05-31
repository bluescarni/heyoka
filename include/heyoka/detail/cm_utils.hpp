// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_CM_UTILS_HPP
#define HEYOKA_DETAIL_CM_UTILS_HPP

#include <cstdint>
#include <variant>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>

namespace heyoka::detail
{

// Comparision operator for LLVM functions based on their names.
struct llvm_func_name_compare {
    bool operator()(const llvm::Function *, const llvm::Function *) const;
};

std::vector<std::variant<std::uint32_t, number>> udef_to_variants(const expression &,
                                                                  const std::vector<std::uint32_t> &);

} // namespace heyoka::detail

#endif
