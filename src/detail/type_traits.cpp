// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <concepts>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/type_traits.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// A function to compute a rough estimate of the cost of performing
// an elementary operation (e.g., addition/multiplication) on a scalar
// floating-point value of type T.
//
// The cost is calibrated to be 1 for single/double precision values,
// so that the unit of measure for the cost is a (very rough) approximation
// of clock cycles.
template <typename T>
double get_fp_unit_cost()
{
    if constexpr (std::same_as<float, T> || std::same_as<double, T>) {
        // float and double.
        return 1;
    } else if constexpr (std::same_as<long double, T>) {
        // long double.
        if constexpr (is_ieee754_binary64<T>) {
            return 1;
        } else if constexpr (is_x86_fp80<T>) {
            return 5;
        } else if constexpr (is_ieee754_binary128<T>) {
#if defined(HEYOKA_ARCH_PPC)
            return 10;
#else
            return 100;
#endif
        } else {
#if defined(HEYOKA_ARCH_PPC)
            // Double-double implementation.
            return 5;
#else
            static_assert(always_false_v<T>, "Unknown fp cost model for long double.");
#endif
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    else if constexpr (std::same_as<mppp::real128, T>) {
#if defined(HEYOKA_ARCH_PPC)
        return 10;
#else
        return 100;
#endif
    }
#endif
#if defined(HEYOKA_HAVE_REAL)
    else if constexpr (std::same_as<mppp::real, T>) {
        // NOTE: this should be improved to take into account
        // the selected precision.
        // NOTE: for reference, mppp::real with 113 bits of precision
        // is slightly slower than software-implemented quadmath.
        return 1000;
    }
#endif
    else {
        static_assert(always_false_v<T>, "Unknown fp cost model for an unsupported floating-point type.");
    }
}

// Explicit instantiations.
template double get_fp_unit_cost<float>();
template double get_fp_unit_cost<double>();
template double get_fp_unit_cost<long double>();

#if defined(HEYOKA_HAVE_REAL128)

template double get_fp_unit_cost<mppp::real128>();

#endif

#if defined(HEYOKA_HAVE_REAL)

template double get_fp_unit_cost<mppp::real>();

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
