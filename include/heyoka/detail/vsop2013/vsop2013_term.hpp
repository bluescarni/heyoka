// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_TERM_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_TERM_HPP

#include <cstdint>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

// NOTE: this represents a single line in the VSOP2013 data files.
// Each term is made of a sequence of indices (a1-a17) coupled to a pair of
// coefficients (S and C). Most of the indices have a small range and thus they
// can be representted with 8-bit ints, but some require 16 bits and
// one index requires 32 bits.
struct vsop2013_term {
    std::int8_t a1_a9[9]{};
    std::int8_t a15_a17[3]{};
    std::int16_t a10_a13[4]{};
    std::int32_t a14{};
    double S{}, C{};
};

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
