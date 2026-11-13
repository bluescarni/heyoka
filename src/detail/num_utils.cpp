// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <mp++/real.hpp>

#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/real_helpers.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
T num_zero_like([[maybe_unused]] const T &x)
{
    if constexpr (std::is_same_v<T, mppp::real>) {
        return mppp::real{mppp::real_kind::zero, x.get_prec()};
    } else {
        return 0;
    }
}

template float num_zero_like(const float &);

template double num_zero_like(const double &);

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

template long double num_zero_like(const long double &);

#endif

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_zero_like(const mppp::real128 &);

#endif

template mppp::real num_zero_like(const mppp::real &);

template <typename T>
T num_one_like([[maybe_unused]] const T &x)
{
    if constexpr (std::is_same_v<T, mppp::real>) {
        return mppp::real{1u, x.get_prec()};
    } else {
        return 1;
    }
}

template float num_one_like(const float &);

template double num_one_like(const double &);

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

template long double num_one_like(const long double &);

#endif

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_one_like(const mppp::real128 &);

#endif

template mppp::real num_one_like(const mppp::real &);

template <typename T>
T num_eps_like([[maybe_unused]] const T &x)
{
    if constexpr (std::is_same_v<T, mppp::real>) {
        return eps_from_prec(x.get_prec());
    } else {
        return std::numeric_limits<T>::epsilon();
    }
}

template float num_eps_like(const float &);

template double num_eps_like(const double &);

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

template long double num_eps_like(const long double &);

#endif

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_eps_like(const mppp::real128 &);

#endif

template mppp::real num_eps_like(const mppp::real &);

template <typename T>
T num_inf_like([[maybe_unused]] const T &x)
{
    if constexpr (std::is_same_v<T, mppp::real>) {
        return mppp::real{mppp::real_kind::inf, x.get_prec()};
    } else {
        return std::numeric_limits<T>::infinity();
    }
}

template float num_inf_like(const float &);

template double num_inf_like(const double &);

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

template long double num_inf_like(const long double &);

#endif

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_inf_like(const mppp::real128 &);

#endif

template mppp::real num_inf_like(const mppp::real &);

// NOTE: helpers to compute the next/previous representable floating-point numbers.
template <typename T>
T num_nextup(T x)
{
    using std::nextafter;
    // NOTE: for mppp::real, the second argument will be an infinity in single precision. This is ok, as
    // mppp::nextafter() ignores the precision of the second argument and outputs a value with the same precision as the
    // first argument.
    return nextafter(std::move(x), static_cast<T>(std::numeric_limits<float>::infinity()));
}

template <typename T>
T num_nextdown(T x)
{
    using std::nextafter;
    return nextafter(std::move(x), static_cast<T>(-std::numeric_limits<float>::infinity()));
}

// Explicit instantiations.
#define HEYOKA_NUM_NEXT_UP_DOWN_INST(F)                                                                                \
    template F num_nextup<F>(F);                                                                                       \
    template F num_nextdown<F>(F);

HEYOKA_NUM_NEXT_UP_DOWN_INST(float)
HEYOKA_NUM_NEXT_UP_DOWN_INST(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_NUM_NEXT_UP_DOWN_INST(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_NUM_NEXT_UP_DOWN_INST(mppp::real128)

#endif

HEYOKA_NUM_NEXT_UP_DOWN_INST(mppp::real)

#undef HEYOKA_NUM_NEXT_UP_DOWN_INST

// Small helper to divide an input value by 2, with mppp::real specialisation.
template <typename T>
T num_div_2(T x)
{
    if constexpr (std::is_same_v<T, mppp::real>) {
        // NOTE: the specialisation for mppp::real is guaranteed to produce a result with the same precision as x. It
        // also uses a specialised division primitive for performance.
        return mppp::div_2ui(std::move(x), 1u);
    } else {
        return x / 2;
    }
}

// Explicit instantiations.
#define HEYOKA_NUM_DIV_2_INST(F) template F num_div_2<F>(F);

HEYOKA_NUM_DIV_2_INST(float)
HEYOKA_NUM_DIV_2_INST(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_NUM_DIV_2_INST(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_NUM_DIV_2_INST(mppp::real128)

#endif

HEYOKA_NUM_DIV_2_INST(mppp::real)

#undef HEYOKA_NUM_DIV_2_INST

} // namespace detail

HEYOKA_END_NAMESPACE
