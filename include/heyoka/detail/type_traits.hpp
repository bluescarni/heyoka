// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TYPE_TRAITS_HPP
#define HEYOKA_DETAIL_TYPE_TRAITS_HPP

#include <heyoka/config.hpp>

#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
using uncvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename, typename...>
inline constexpr bool always_false_v = false;

// http://en.cppreference.com/w/cpp/experimental/is_detected
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

// http://en.cppreference.com/w/cpp/experimental/nonesuch
struct nonesuch {
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const &) = delete;
    nonesuch(nonesuch &&) noexcept = delete;
    void operator=(nonesuch const &) = delete;
    void operator=(nonesuch &&) noexcept = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

template <template <class...> class Op, class... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

// Helper to detect if T is a supported floating-point type.
template <typename>
struct is_supported_fp : std::false_type {
};

template <>
struct is_supported_fp<double> : std::true_type {
};

template <>
struct is_supported_fp<long double> : std::true_type {
};

#if defined(HEYOKA_HAVE_REAL128)

template <>
struct is_supported_fp<mppp::real128> : std::true_type {
};

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
struct is_supported_fp<mppp::real> : std::true_type {
};

#endif

template <typename T>
inline constexpr bool is_supported_fp_v = is_supported_fp<T>::value;

// Detect vector type.
template <typename>
struct is_any_vector : std::false_type {
};

template <typename T>
struct is_any_vector<std::vector<T>> : std::true_type {
};

template <typename T>
inline constexpr bool is_any_vector_v = is_any_vector<T>::value;

// Detect initializer_list type.
template <typename>
struct is_any_ilist : std::false_type {
};

template <typename T>
struct is_any_ilist<std::initializer_list<T>> : std::true_type {
};

template <typename T>
inline constexpr bool is_any_ilist_v = is_any_ilist<T>::value;

// Detection of IEEE-style floating-point types.
template <typename T, int N>
constexpr bool is_ieee754_binaryN()
{
    if constexpr (std::is_floating_point_v<T> && std::numeric_limits<T>::is_specialized) {
        return std::numeric_limits<T>::radix == 2 && std::numeric_limits<T>::is_iec559
               && std::numeric_limits<T>::digits == N;
    } else {
        return false;
    }
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
constexpr bool is_ieee754_binaryN<mppp::real128, 113>()
{
    return true;
}

#endif

template <typename T>
inline constexpr bool is_ieee754_binary32 = is_ieee754_binaryN<T, 24>();

template <typename T>
inline constexpr bool is_ieee754_binary64 = is_ieee754_binaryN<T, 53>();

template <typename T>
inline constexpr bool is_x86_fp80 = is_ieee754_binaryN<T, 64>();

template <typename T>
inline constexpr bool is_ieee754_binary128 = is_ieee754_binaryN<T, 113>();

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
