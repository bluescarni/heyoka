// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <boost/core/demangle.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/predef/architecture.h>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Simple recursive implementation of the binomial coefficient via boost's cpp_int.
//
// NOLINTNEXTLINE(misc-no-recursion)
boost::multiprecision::cpp_int cpp_int_choose(const std::uint32_t n, const std::uint32_t k)
{
    assert(k <= n);

    if (k == 0u) {
        return 1;
    } else {
        return (n * cpp_int_choose(n - 1u, k - 1u)) / k;
    }
}

} // namespace

// Helper to produce an interval [lo, hi] of floating-point type T guaranteed to include the binomial coefficient of the
// inputs n and k. The x argument is used only for mppp::real in order to infer the desired precision.
//
// The output is guaranteed to be finite (an exception will be thrown in case of overflow). The function will throw if T
// is a builtin floating-point type whose radix is not 2 or which is not an IEC 559/IEEE 754 floating-point type.
template <typename T>
std::pair<T, T> ed_make_ival_binomial_cf_like(const std::uint32_t n, const std::uint32_t k, [[maybe_unused]] const T &x)
{
    using boost::multiprecision::cpp_int;
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;
    using std::isfinite;
    using std::nextafter;

    // Compute the exact result.
    const auto C = cpp_int_choose(n, k);

    // Determine the number of bits required for the exact representation of C.
    const auto C_nbits = safe_uint32_t(boost::multiprecision::msb(C)) + 1;

    constexpr bool has_num_limits = std::is_floating_point_v<T>
#if defined(HEYOKA_HAVE_REAL128)
                                    || std::same_as<T, mppp::real128>
#endif
        ;

    // Determine the bit width such that an integer with that many binary digits can always be represented exactly by T.
    const auto exact_digits = [&x] {
        static_cast<void>(x);

        if constexpr (has_num_limits) {
            // LCOV_EXCL_START
            if constexpr (std::numeric_limits<T>::radix != 2 || !std::numeric_limits<T>::is_iec559) {
                throw std::invalid_argument(fmt::format(
                    "Cannot construct a rigorous enclosure for a binomial coefficient using the floating-point type "
                    "'{}': the radix of the type is not 2 or the type is not an IEC 559/IEEE 754 floating-point type",
                    boost::core::demangle(typeid(T).name())));
            }
            // NOTE: some standard libraries (e.g., libstdc++) consider the double-double long double type available on
            // PPC as an IEC 559/IEEE 754 floating-point type, even though it is not. Thus, we add special casing for
            // such situations.
#if BOOST_ARCH_PPC && __LONG_DOUBLE_IBM128__
            if constexpr (std::same_as<T, long double>) {
                throw std::invalid_argument("Cannot construct a rigorous enclosure for a binomial coefficient using "
                                            "the IBM double-double long double type");
            }
#endif
            // LCOV_EXCL_STOP

            return safe_uint32_t(std::numeric_limits<T>::digits);
#if defined(HEYOKA_HAVE_REAL)
        } else if constexpr (std::same_as<T, mppp::real>) {
            return safe_uint32_t(x.get_prec());
#endif
        } else {
            static_assert(always_false_v<T>, "Unsupported type");
        }
    }();

    // Small helper to convert a cpp_int to the target floating-point type.
    const auto to_fp_type = [&x](const cpp_int &in) -> T {
        static_cast<void>(x);

        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(in);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::same_as<T, mppp::real128>) {
            return mppp::real128{in.str()};
#endif
#if defined(HEYOKA_HAVE_REAL)
        } else if constexpr (std::same_as<T, mppp::real>) {
            return mppp::real{in.str(), x.get_prec()};
#endif
        } else {
            static_assert(always_false_v<T>, "Unsupported type");
        }
    };

    // Small helper to validate the output for finiteness.
    const auto validate_ret = [n, k](std::pair<T, T> p) {
        if (!isfinite(p.first) || !isfinite(p.second)) [[unlikely]] {
            throw std::overflow_error(
                fmt::format("The computation of a binomial coefficient with arguments ({}, {}) overflows the "
                            "target floating-point type in ed_make_ival_binomial_cf_like()",
                            n, k));
        }
        return p;
    };

    if (C_nbits <= exact_digits) {
        // C is guaranteed to be exactly representable whatever its value.
        const auto fp_C = to_fp_type(C);
        return validate_ret({fp_C, fp_C});
    }

    // C may not be representable exactly.

    // Compute the lo part of the interval by zeroing out in C the least-significant bits that cannot be represented by
    // the floating-point type.
    const auto extra_bits = static_cast<std::uint32_t>(C_nbits - exact_digits);
    const cpp_int C_lo = (C >> extra_bits) << extra_bits;

    // Convert to the target floating-point type.
    const auto fp_C_lo = to_fp_type(C_lo);

    if (C_lo == C) {
        // The extra_bits least-significant bits of C are zero, thus C is exactly representable.
        return validate_ret({fp_C_lo, fp_C_lo});
    }

    // C is not representable exactly. The lower bound has been computed already, the upper bound can be computed by
    // taking the next representable value.

    // NOTE: for mppp::real, nextafter ignores the precision of the second argument, and uses the precision of the first
    // argument for the result, which is the correct outcome.
    return validate_ret({fp_C_lo, nextafter(fp_C_lo, static_cast<T>(std::numeric_limits<double>::infinity()))});
}

// Explicit instantiations.
#define HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST(T)                                                                        \
    template HEYOKA_DLL_PUBLIC std::pair<T, T> ed_make_ival_binomial_cf_like(std::uint32_t, std::uint32_t, const T &)

HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST(float);
HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST(double);

HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST(mppp::real);

#endif

#undef HEYOKA_ED_MAKE_IVAL_BINOMIAL_CF_INST

} // namespace detail

HEYOKA_END_NAMESPACE
