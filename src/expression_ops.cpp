// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

// TODO review headers.

#include <heyoka/config.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

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

// TODO: pass by copy + move?
expression operator-(const expression &e)
{
    return prod({expression{number{-1.}}, e});
}

namespace detail
{

// A comparison operator intended for sorting in a canonical
// way the operands to a commutative operator/function.
// NOTE: this is similar to std::less<expression>, except that
// it does not sort functions. The reason for this is that
// function comparison is platform-dependent as it relies on
// std::type_index comparison. We want to avoid having a
// platform-dependent canonical order for the way expressions
// are constructed.
bool comm_ops_lt(const expression &e1, const expression &e2)
{
    return std::visit(
        [](const auto &v1, const auto &v2) {
            using type1 = uncvref_t<decltype(v1)>;
            using type2 = uncvref_t<decltype(v2)>;

            // Phase 1: handle the cases where v1 and v2
            // are the same type.

            // Both arguments are variables: use lexicographic comparison.
            if constexpr (std::is_same_v<variable, type1> && std::is_same_v<variable, type2>) {
                return v1.name() < v2.name();
            }

            // Both arguments are params: compare the indices.
            if constexpr (std::is_same_v<param, type1> && std::is_same_v<param, type2>) {
                return v1.idx() < v2.idx();
            }

            // Both arguments are numbers: compare.
            if constexpr (std::is_same_v<number, type1> && std::is_same_v<number, type2>) {
                return v1 < v2;
            }

            // Both arguments are functions: equivalent.
            if constexpr (std::is_same_v<func, type1> && std::is_same_v<func, type2>) {
                return false;
            }

            // Phase 2: handle mixed types.

            // Number is always less than non-number.
            if constexpr (std::is_same_v<number, type1>) {
                return true;
            }

            // Function never less than non-function.
            if constexpr (std::is_same_v<func, type1>) {
                return false;
            }

            // Variable less than function, greater than anything elses.
            if constexpr (std::is_same_v<variable, type1>) {
                return std::is_same_v<type2, func>;
            }

            // Param greater than number, less than anything else.
            if constexpr (std::is_same_v<param, type1>) {
                return !std::is_same_v<type2, number>;
            }

            // LCOV_EXCL_START
            assert(false);

            return false;
            // LCOV_EXCL_STOP
        },
        e1.value(), e2.value());
}

} // namespace detail

// TODO pass by copy + move?
// NOLINTNEXTLINE(misc-no-recursion)
expression operator+(const expression &e1, const expression &e2)
{
    return sum({e1, e2});
}

// TODO pass by copy + move?
// NOLINTNEXTLINE(misc-no-recursion)
expression operator-(const expression &e1, const expression &e2)
{
    return e1 + -e2;
}

// TODO pass by copy + move?
// NOLINTNEXTLINE(misc-no-recursion)
expression operator*(const expression &e1, const expression &e2)
{
    return prod({e1, e2});
}

// TODO pass by copy + move?
// NOLINTNEXTLINE(misc-no-recursion)
expression operator/(const expression &e1, const expression &e2)
{
    return prod({e1, pow(e2, -1_dbl)});
}

// TODO pass by copy + move?
// TODO everywhere below.
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
