// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <stdexcept>
#include <utility>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/dfloat.hpp>

namespace heyoka::detail
{

#if defined(HEYOKA_HAVE_REAL)

dfloat<mppp::real>::dfloat() = default;

dfloat<mppp::real>::dfloat(mppp::real x) : hi(std::move(x)), lo(mppp::real_kind::zero, hi.get_prec()) {}

dfloat<mppp::real>::dfloat(mppp::real h, mppp::real l) : hi(std::move(h)), lo(std::move(l))
{
    if (hi.get_prec() != lo.get_prec()) {
        // LCOV_EXCL_START
        throw std::invalid_argument("Mismatched precisions in the components of a dfloat<mppp::real>");
        // LCOV_EXCL_STOP
    }
}

dfloat<mppp::real>::operator mppp::real() const
{
    return hi;
}

#endif

} // namespace heyoka::detail
