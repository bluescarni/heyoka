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

// NOTE: in these operators we check for number arguments
// immediately, before forwarding to the underlying implementation.
// We do this in order to avoid accidental promotions and incorrect
// precision propagation due to the use of double-precision constants
// in the implementations of the primitives.
expression operator-(const expression &e)
{
    if (const auto *nptr = std::get_if<number>(&e.value())) {
        return expression{-*nptr};
    } else {
        return prod({expression{number{-1.}}, e});
    }
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator+(const expression &e1, const expression &e2)
{
    if (std::holds_alternative<number>(e1.value()) && std::holds_alternative<number>(e2.value())) {
        return expression{std::get<number>(e1.value()) + std::get<number>(e2.value())};
    } else {
        return sum({e1, e2});
    }
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator-(const expression &e1, const expression &e2)
{
    if (std::holds_alternative<number>(e1.value()) && std::holds_alternative<number>(e2.value())) {
        return expression{std::get<number>(e1.value()) - std::get<number>(e2.value())};
    } else {
        return e1 + -e2;
    }
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator*(const expression &e1, const expression &e2)
{
    if (std::holds_alternative<number>(e1.value()) && std::holds_alternative<number>(e2.value())) {
        return expression{std::get<number>(e1.value()) * std::get<number>(e2.value())};
    } else {
        return prod({e1, e2});
    }
}

// NOLINTNEXTLINE(misc-no-recursion)
expression operator/(const expression &e1, const expression &e2)
{
    if (std::holds_alternative<number>(e1.value()) && std::holds_alternative<number>(e2.value())) {
        return expression{std::get<number>(e1.value()) / std::get<number>(e2.value())};
    } else {
        return prod({e1, pow(e2, -1_dbl)});
    }
}

#define HEYOKA_EX_BINARY_OP_R(op, type)                                                                                \
    expression operator op(const expression &ex, type x)                                                               \
    {                                                                                                                  \
        return ex op expression{std::move(x)};                                                                         \
    }

#define HEYOKA_EX_BINARY_OP_L(op, type)                                                                                \
    expression operator op(type x, const expression &ex)                                                               \
    {                                                                                                                  \
        return expression{std::move(x)} op ex;                                                                         \
    }

HEYOKA_EX_BINARY_OP_R(+, float)
HEYOKA_EX_BINARY_OP_R(+, double)
HEYOKA_EX_BINARY_OP_R(+, long double)
HEYOKA_EX_BINARY_OP_R(-, float)
HEYOKA_EX_BINARY_OP_R(-, double)
HEYOKA_EX_BINARY_OP_R(-, long double)
HEYOKA_EX_BINARY_OP_R(*, float)
HEYOKA_EX_BINARY_OP_R(*, double)
HEYOKA_EX_BINARY_OP_R(*, long double)
HEYOKA_EX_BINARY_OP_R(/, float)
HEYOKA_EX_BINARY_OP_R(/, double)
HEYOKA_EX_BINARY_OP_R(/, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_EX_BINARY_OP_R(+, mppp::real128)
HEYOKA_EX_BINARY_OP_R(-, mppp::real128)
HEYOKA_EX_BINARY_OP_R(*, mppp::real128)
HEYOKA_EX_BINARY_OP_R(/, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_EX_BINARY_OP_R(+, mppp::real)
HEYOKA_EX_BINARY_OP_R(-, mppp::real)
HEYOKA_EX_BINARY_OP_R(*, mppp::real)
HEYOKA_EX_BINARY_OP_R(/, mppp::real)

#endif

HEYOKA_EX_BINARY_OP_L(+, float)
HEYOKA_EX_BINARY_OP_L(+, double)
HEYOKA_EX_BINARY_OP_L(+, long double)
HEYOKA_EX_BINARY_OP_L(-, float)
HEYOKA_EX_BINARY_OP_L(-, double)
HEYOKA_EX_BINARY_OP_L(-, long double)
HEYOKA_EX_BINARY_OP_L(*, float)
HEYOKA_EX_BINARY_OP_L(*, double)
HEYOKA_EX_BINARY_OP_L(*, long double)
HEYOKA_EX_BINARY_OP_L(/, float)
HEYOKA_EX_BINARY_OP_L(/, double)
HEYOKA_EX_BINARY_OP_L(/, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_EX_BINARY_OP_L(+, mppp::real128)
HEYOKA_EX_BINARY_OP_L(-, mppp::real128)
HEYOKA_EX_BINARY_OP_L(*, mppp::real128)
HEYOKA_EX_BINARY_OP_L(/, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_EX_BINARY_OP_L(+, mppp::real)
HEYOKA_EX_BINARY_OP_L(-, mppp::real)
HEYOKA_EX_BINARY_OP_L(*, mppp::real)
HEYOKA_EX_BINARY_OP_L(/, mppp::real)

#endif

#undef HEYOKA_EX_BINARY_OP_R
#undef HEYOKA_EX_BINARY_OP_L

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

// NOLINTBEGIN
#define HEYOKA_EX_COMPOUND_OP(op, type)                                                                                \
    expression &operator op(expression & ex, type x)                                                                   \
    {                                                                                                                  \
        return ex op expression{std::move(x)};                                                                         \
    }
// NOLINTEND

HEYOKA_EX_COMPOUND_OP(+=, float)
HEYOKA_EX_COMPOUND_OP(+=, double)
HEYOKA_EX_COMPOUND_OP(+=, long double)
HEYOKA_EX_COMPOUND_OP(-=, float)
HEYOKA_EX_COMPOUND_OP(-=, double)
HEYOKA_EX_COMPOUND_OP(-=, long double)
HEYOKA_EX_COMPOUND_OP(*=, float)
HEYOKA_EX_COMPOUND_OP(*=, double)
HEYOKA_EX_COMPOUND_OP(*=, long double)
HEYOKA_EX_COMPOUND_OP(/=, float)
HEYOKA_EX_COMPOUND_OP(/=, double)
HEYOKA_EX_COMPOUND_OP(/=, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_EX_COMPOUND_OP(+=, mppp::real128)
HEYOKA_EX_COMPOUND_OP(-=, mppp::real128)
HEYOKA_EX_COMPOUND_OP(*=, mppp::real128)
HEYOKA_EX_COMPOUND_OP(/=, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_EX_COMPOUND_OP(+=, mppp::real)
HEYOKA_EX_COMPOUND_OP(-=, mppp::real)
HEYOKA_EX_COMPOUND_OP(*=, mppp::real)
HEYOKA_EX_COMPOUND_OP(/=, mppp::real)

#endif

#undef HEYOKA_EX_COMPOUND_OP

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
