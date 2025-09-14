// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this is a workaround for a compilation issue on OSX, where clang complains that malloc()/free() declarations
// (used somewhere inside fmt) are not available. See:
// https://github.com/fmtlib/fmt/pull/4477
#include <cstdlib>

#include <heyoka/config.hpp>

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/dfloat.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

#if defined(HEYOKA_HAVE_REAL)

dfloat<mppp::real>::dfloat() = default;

dfloat<mppp::real>::dfloat(mppp::real x) : hi(std::move(x)), lo(mppp::real_kind::zero, hi.get_prec()) {}

dfloat<mppp::real>::dfloat(mppp::real h, mppp::real l) : hi(std::move(h)), lo(std::move(l))
{
    if (hi.get_prec() != lo.get_prec()) {
        throw std::invalid_argument(
            fmt::format("Mismatched precisions in the components of a dfloat<mppp::real>: the high component has a "
                        "precision of {}, while the low component has a precision of {}",
                        hi.get_prec(), lo.get_prec()));
    }
}

dfloat<mppp::real>::operator mppp::real() const &
{
    return hi;
}

dfloat<mppp::real>::operator mppp::real() &&
{
    return std::move(hi);
}

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
