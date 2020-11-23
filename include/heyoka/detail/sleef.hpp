// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_SLEEF_HPP
#define HEYOKA_DETAIL_SLEEF_HPP

#include <cstdint>
#include <string>

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka::detail
{

HEYOKA_DLL_PUBLIC std::string sleef_function_name(llvm::LLVMContext &, const std::string &, llvm::Type *,
                                                  std::uint32_t);

} // namespace heyoka::detail

#endif
