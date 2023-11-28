// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VECTOR_MATH_HPP
#define HEYOKA_DETAIL_VECTOR_MATH_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

struct vf_info {
    // The name of the vector function
    // to be invoked (e.g., from SLEEF).
    std::string name;
    // The vfabi attribute corresponding
    // to the vector function.
    std::string vf_abi_attr;
    // The corresponding low-precision versions
    // of the above. These will be empty if
    // the low-precision counterpart is
    // not available.
    std::string lp_name;
    std::string lp_vf_abi_attr;
    // Number of SIMD lanes.
    std::uint32_t width = 0;
    // Number of arguments.
    std::uint32_t nargs = 0;
};

const std::vector<vf_info> &lookup_vf_info(const std::string &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
