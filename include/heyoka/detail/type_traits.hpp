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
#include <string>
#include <string_view>
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
    void operator=(nonesuch const &) = delete;
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

// NOTE: remove_pointer_t removes the top level qualifiers of the pointer as well:
// http://en.cppreference.com/w/cpp/types/remove_pointer
// After removal of pointer, we could still have a type which is cv qualified. Thus,
// we remove cv qualifications after pointer removal.
template <typename T>
using is_char_pointer
    = std::conjunction<std::is_pointer<T>, std::is_same<std::remove_cv_t<std::remove_pointer_t<T>>, char>>;

// This type trait is satisfied by C++ string-like types. Specifically, the type trait will be true if T, after the
// removal of cv qualifiers, is one of the following types:
// - std::string,
// - a pointer to (possibly cv qualified) char,
// - a char array of any size,
// - std::string_view.
template <typename T>
using is_string_type = std::disjunction<
    std::is_same<std::remove_cv_t<T>, std::string>, is_char_pointer<T>,
    // NOTE: std::remove_cv_t does remove cv qualifiers from arrays.
    std::conjunction<std::is_array<std::remove_cv_t<T>>, std::is_same<std::remove_extent_t<std::remove_cv_t<T>>, char>>,
    std::is_same<std::remove_cv_t<T>, std::string_view>>;

template <typename T>
inline constexpr bool is_string_type_v = is_string_type<T>::value;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
