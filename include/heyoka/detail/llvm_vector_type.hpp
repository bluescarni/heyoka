// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_VECTOR_TYPE_HPP
#define HEYOKA_DETAIL_LLVM_VECTOR_TYPE_HPP

#include <llvm/IR/DerivedTypes.h>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: this is a convenience typedef for the LLVM
// vector type used internally by heyoka. It is a vector
// whose size is fixed at compile time.
using llvm_vector_type = llvm::FixedVectorType;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
