// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <type_traits>
#include <utility>
#include <variant>

#include <heyoka/config.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

HEYOKA_BEGIN_NAMESPACE

expression operator+(expression e)
{
    return e;
}

expression operator-(const expression &e)
{
    return prod({expression{number{-1.}}, e});
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator+(const expression &e1, const expression &e2)
{
    return sum({e1, e2});
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator-(const expression &e1, const expression &e2)
{
    return e1 + -e2;
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator*(const expression &e1, const expression &e2)
{
    return prod({e1, e2});
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator/(const expression &e1, const expression &e2)
{
    return prod({e1, pow(e2, -1_dbl)});
}

expression operator+(const expression &ex, double x)
{
    return ex + expression{x};
}

expression operator+(const expression &ex, long double x)
{
    return ex + expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator+(const expression &ex, mppp::real128 x)
{
    return ex + expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator+(const expression &ex, mppp::real x)
{
    return ex + expression{std::move(x)};
}

#endif

expression operator+(double x, const expression &ex)
{
    return expression{x} + ex;
}

expression operator+(long double x, const expression &ex)
{
    return expression{x} + ex;
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator+(mppp::real128 x, const expression &ex)
{
    return expression{x} + ex;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator+(mppp::real x, const expression &ex)
{
    return expression{std::move(x)} + ex;
}

#endif

expression operator-(const expression &ex, double x)
{
    return ex - expression{x};
}

expression operator-(const expression &ex, long double x)
{
    return ex - expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator-(const expression &ex, mppp::real128 x)
{
    return ex - expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator-(const expression &ex, mppp::real x)
{
    return ex - expression{std::move(x)};
}

#endif

expression operator-(double x, const expression &ex)
{
    return expression{x} - ex;
}

expression operator-(long double x, const expression &ex)
{
    return expression{x} - ex;
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator-(mppp::real128 x, const expression &ex)
{
    return expression{x} - ex;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator-(mppp::real x, const expression &ex)
{
    return expression{std::move(x)} - ex;
}

#endif

expression operator*(const expression &ex, double x)
{
    return ex * expression{x};
}

expression operator*(const expression &ex, long double x)
{
    return ex * expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator*(const expression &ex, mppp::real128 x)
{
    return ex * expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator*(const expression &ex, mppp::real x)
{
    return ex * expression{std::move(x)};
}

#endif

expression operator*(double x, const expression &ex)
{
    return expression{x} * ex;
}

expression operator*(long double x, const expression &ex)
{
    return expression{x} * ex;
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator*(mppp::real128 x, const expression &ex)
{
    return expression{x} * ex;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator*(mppp::real x, const expression &ex)
{
    return expression{std::move(x)} * ex;
}

#endif

expression operator/(const expression &ex, double x)
{
    return ex / expression{x};
}

expression operator/(const expression &ex, long double x)
{
    return ex / expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator/(const expression &ex, mppp::real128 x)
{
    return ex / expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator/(const expression &ex, mppp::real x)
{
    return ex / expression{std::move(x)};
}

#endif

expression operator/(double x, const expression &ex)
{
    return expression{x} / ex;
}

expression operator/(long double x, const expression &ex)
{
    return expression{x} / ex;
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator/(mppp::real128 x, const expression &ex)
{
    return expression{x} / ex;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression operator/(mppp::real x, const expression &ex)
{
    return expression{std::move(x)} / ex;
}

#endif

expression &operator+=(expression &x, const expression &e)
{
    // NOTE: it is important that compound operators
    // are implemented as x = x op e, so that we properly
    // take into account arithmetic promotions for
    // numbers (and, in case of mppp::real numbers,
    // precision propagation).
    return x = x + e;
}

expression &operator-=(expression &x, const expression &e)
{
    return x = x - e;
}

expression &operator*=(expression &x, const expression &e)
{
    return x = x * e;
}

expression &operator/=(expression &x, const expression &e)
{
    return x = x / e;
}

expression &operator+=(expression &ex, double x)
{
    return ex += expression{x};
}

expression &operator+=(expression &ex, long double x)
{
    return ex += expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator+=(expression &ex, mppp::real128 x)
{
    return ex += expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression &operator+=(expression &ex, mppp::real x)
{
    return ex += expression{std::move(x)};
}

#endif

expression &operator-=(expression &ex, double x)
{
    return ex -= expression{x};
}

expression &operator-=(expression &ex, long double x)
{
    return ex -= expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator-=(expression &ex, mppp::real128 x)
{
    return ex -= expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression &operator-=(expression &ex, mppp::real x)
{
    return ex -= expression{std::move(x)};
}

#endif

expression &operator*=(expression &ex, double x)
{
    return ex *= expression{x};
}

expression &operator*=(expression &ex, long double x)
{
    return ex *= expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator*=(expression &ex, mppp::real128 x)
{
    return ex *= expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression &operator*=(expression &ex, mppp::real x)
{
    return ex *= expression{std::move(x)};
}

#endif

expression &operator/=(expression &ex, double x)
{
    return ex /= expression{x};
}

expression &operator/=(expression &ex, long double x)
{
    return ex /= expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator/=(expression &ex, mppp::real128 x)
{
    return ex /= expression{x};
}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression &operator/=(expression &ex, mppp::real x)
{
    return ex /= expression{std::move(x)};
}

#endif

bool operator==(const expression &e1, const expression &e2)
{
    auto visitor = [](const auto &v1, const auto &v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, type2>) {
            return v1 == v2;
        } else {
            return false;
        }
    };

    return std::visit(visitor, e1.value(), e2.value());
}

bool operator!=(const expression &e1, const expression &e2)
{
    return !(e1 == e2);
}

HEYOKA_END_NAMESPACE
