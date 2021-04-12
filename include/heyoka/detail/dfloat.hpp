// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_DFLOAT_HPP
#define HEYOKA_DETAIL_DFLOAT_HPP

#include <cmath>
#include <string>

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
};

// NOTE: taken with minimal adaptations from NTL. My understanding
// here is the following: given input (x, xx) and (y, yy), the error-free
// additions x + y and xx + yy are computed. The components of xx + yy
// are then added to the low part of the result of x + y, each addition
// being followed by a normalisation step.
template <typename F>
inline dfloat<F> operator+(const dfloat<F> &x, const dfloat<F> &y)
{
    F H, h, T, t, S, s, e, f;
    F t1;

    S = x.hi + y.hi;
    T = x.lo + y.lo;
    e = S - x.hi;
    f = T - x.lo;

    t1 = S - e;
    t1 = x.hi - t1;
    s = y.hi - e;
    s = s + t1;

    t1 = T - f;
    t1 = x.lo - t1;
    t = y.lo - f;
    t = t + t1;

    s = s + T;
    H = S + s;
    h = S - H;
    h = h + s;

    h = h + t;
    e = H + h;
    f = H - e;
    f = f + h;

    return dfloat<F>(e, f);
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

template <typename F>
inline bool isfinite(const dfloat<F> &x)
{
    using std::isfinite;

    return isfinite(x.hi) && isfinite(x.lo);
}

// Comparisons.
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
inline bool operator>=(const F &x, const dfloat<F> &y)
{
    return dfloat<F>(x) >= y;
}

} // namespace heyoka::detail

#endif
