// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_DFLOAT_HPP
#define HEYOKA_DETAIL_DFLOAT_HPP

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <string>
#include <tuple>
#include <utility>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

namespace heyoka::detail
{

// A minimal class for double-length float arithmetic.
// F can be any floating-point type supported by heyoka.
template <typename F>
struct dfloat {
    F hi, lo;

    dfloat() : hi(0), lo(0) {}
    explicit dfloat(F x) : hi(x), lo(0) {}
    explicit dfloat(F h, F l) : hi(h), lo(l) {}

    explicit operator F() const
    {
        return hi;
    }

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &hi;
        ar &lo;
    }
};

#if defined(HEYOKA_HAVE_REAL)

// NOTE: we have a specialised implementation for this
// in order to ensure the ctors always set the components
// to the same precision (and if they can't, they will throw).
template <>
struct HEYOKA_DLL_PUBLIC dfloat<mppp::real> {
    mppp::real hi, lo;

    dfloat();
    explicit dfloat(mppp::real);
    explicit dfloat(mppp::real, mppp::real);

    explicit operator mppp::real() const;

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &hi;
        ar &lo;
    }
};

#endif

template <typename F>
inline bool isfinite(const dfloat<F> &x)
{
    using std::isfinite;

    return isfinite(x.hi) && isfinite(x.lo);
}

// Error-free transformation of the sum of two floating point numbers.
// This is Dekker's algorithm, which requires abs(a) >= abs(b). See algorithm 2.2 here:
// https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
template <typename F>
inline std::pair<F, F> eft_add_dekker(F a, F b)
{
    auto x = a + b;
    auto y = (a - x) + b;

    return {x, y};
}

// Error-free transformation of the sum of two floating point numbers.
// This is Knuth's algorithm. See algorithm 2.1 here:
// https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
template <typename F>
inline std::pair<F, F> eft_add_knuth(F a, F b)
{
    auto x = a + b;
    auto z = x - a;
    auto y = (a - (x - z)) + (b - z);

    return {x, y};
}

// Normalise a double-length float.
// Taken from:
// https://github.com/fhajji/ntl/blob/6918e6b80336cee34f2131fcf71a58c72b931174/src/quad_float.cpp#L125
// NOTE: this is based on the error-free trasformation requiring abs(x.hi) >= abs(x.lo).
template <typename F>
inline dfloat<F> normalise(const dfloat<F> &x)
{
    // LCOV_EXCL_START
#if !defined(NDEBUG)
    using std::abs;

    if (isfinite(x)) {
        assert(abs(x.hi) >= abs(x.lo));
    }
#endif
    // LCOV_EXCL_STOP

    auto [u, v] = eft_add_dekker(x.hi, x.lo);

    return dfloat<F>(u, v);
}

// NOTE: taken with minimal adaptations from NTL.
template <typename F>
inline dfloat<F> operator+(const dfloat<F> &a, const dfloat<F> &b)
{
    // NOTE: x_hi + y_hi is the exact result of a.hi + b.hi.
    // x_lo + y_lo  is the exact result of a.lo + b.lo.
    auto [x_hi, y_hi] = eft_add_knuth(a.hi, b.hi);
    auto [x_lo, y_lo] = eft_add_knuth(a.lo, b.lo);

    // The plan is now to:
    // - add x_lo to y_hi, and normalise;
    // - add y_lo to v, and normalise again.
    // NOTE: this is different from Dekker's algorithm, and I am not
    // 100% sure why this works as Dekker's EFT has requirements on the
    // magnitudes of the operands. However, this is essentially the
    // original code from NTL and testing also indicates that
    // this works.
    auto [u, v] = eft_add_dekker(x_hi, y_hi + x_lo);
    std::tie(u, v) = eft_add_dekker(u, v + y_lo);

    return dfloat<F>(u, v);
}

// Subtraction.
template <typename F>
inline dfloat<F> operator-(const dfloat<F> &x, const dfloat<F> &y)
{
    return x + dfloat<F>(-y.hi, -y.lo);
}

// A few convenience overloads.
template <typename F>
inline dfloat<F> operator+(const dfloat<F> &x, const F &y)
{
    return x + dfloat<F>(y);
}

template <typename F>
inline dfloat<F> &operator+=(dfloat<F> &x, const dfloat<F> &y)
{
    return x = x + y;
}

template <typename F>
inline dfloat<F> &operator+=(dfloat<F> &x, const F &y)
{
    return x = x + dfloat<F>(y);
}

template <typename F>
inline dfloat<F> operator-(const dfloat<F> &x, const F &y)
{
    return x - dfloat<F>(y);
}

template <typename F>
inline dfloat<F> operator-(const F &x, const dfloat<F> &y)
{
    return dfloat<F>(x) - y;
}

// Comparisons.
template <typename F>
inline bool operator==(const dfloat<F> &x, const dfloat<F> &y)
{
    return x.hi == y.hi && x.lo == y.lo;
}

template <typename F>
inline bool operator==(const dfloat<F> &x, const F &y)
{
    return x == dfloat<F>(y);
}

template <typename F>
inline bool operator!=(const dfloat<F> &x, const dfloat<F> &y)
{
    return !(x == y);
}

template <typename F>
inline bool operator<(const dfloat<F> &x, const dfloat<F> &y)
{
    return (x.hi < y.hi) || (x.hi == y.hi && x.lo < y.lo);
}

template <typename F>
inline bool operator>(const dfloat<F> &x, const dfloat<F> &y)
{
    return (x.hi > y.hi) || (x.hi == y.hi && x.lo > y.lo);
}

template <typename F>
inline bool operator>(const F &x, const dfloat<F> &y)
{
    return dfloat<F>(x) > y;
}

template <typename F>
inline bool operator<=(const dfloat<F> &x, const dfloat<F> &y)
{
    return (x.hi < y.hi) || (x.hi == y.hi && x.lo <= y.lo);
}

template <typename F>
inline bool operator<=(const F &x, const dfloat<F> &y)
{
    return dfloat<F>(x) <= y;
}

template <typename F>
inline bool operator>=(const dfloat<F> &x, const dfloat<F> &y)
{
    return (x.hi > y.hi) || (x.hi == y.hi && x.lo >= y.lo);
}

template <typename F>
inline bool operator>=(const dfloat<F> &x, const F &y)
{
    return x >= dfloat<F>(y);
}

template <typename F>
inline bool operator>=(const F &x, const dfloat<F> &y)
{
    return dfloat<F>(x) >= y;
}

} // namespace heyoka::detail

#endif
