// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_CM_UTILS_HPP
#define HEYOKA_DETAIL_CM_UTILS_HPP

#include <heyoka/detail/llvm_fwd.hpp>

namespace heyoka::detail
{

// Comparision operator for LLVM functions based on their names.
struct llvm_func_name_compare {
    bool operator()(const llvm::Function *, const llvm::Function *) const;
};

} // namespace heyoka::detail

#endif
