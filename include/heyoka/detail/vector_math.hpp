// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VECTOR_MATH_HPP
#define HEYOKA_DETAIL_VECTOR_MATH_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

struct vf_info {
    // The name of the vector function to be invoked (e.g., from SLEEF).
    std::string name;
    // The vfabi attribute corresponding to the vector function.
    std::string vf_abi_attr;
    // The corresponding low-precision versions of the above. These will be empty if the low-precision counterpart is
    // not available.
    std::string lp_name;
    std::string lp_vf_abi_attr;
    // Number of SIMD lanes.
    std::uint32_t width = 0;
    // Number of arguments.
    std::uint32_t nargs = 0;
    // An optional function to emit IR code necessary to the invocation of the vector variants.
    using gen_t = std::function<void(llvm_state &)>;
    gen_t gen;
};

// Lookup the vector info for a scalar function.
//
// This function takes in input the name of a scalar function - which could be either an LLVM function (including
// intrinsics) or a function from the C runtime - and returns a set of vf_info instances, one for each vector variant
// available for that function. The list of returned vf_info instances is guaranteed to be sorted in ascending 'width'
// order.
//
// NOTE: this returns a const reference to a global, annotate it with HEYOKA_NO_DANGLING because we know that this does
// not return dangling refs, but GCC gets confused.
HEYOKA_NO_DANGLING const std::vector<vf_info> &lookup_vf_info(const std::string &);

#if defined(HEYOKA_WITH_SLEEF)

void make_combined_sleef_functions(llvm_state &, const std::string &, const std::string &, const std::string &,
                                   std::uint32_t, std::uint32_t, const std::string &);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
