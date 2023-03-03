// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <limits>
#include <type_traits>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/num_utils.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
T num_zero_like([[maybe_unused]] const T &x)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return mppp::real{mppp::real_kind::zero, x.get_prec()};
    } else {
#endif
        return 0;
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

template double num_zero_like(const double &);

template long double num_zero_like(const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_zero_like(const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template mppp::real num_zero_like(const mppp::real &);

#endif

template <typename T>
T num_one_like([[maybe_unused]] const T &x)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return mppp::real{1u, x.get_prec()};
    } else {
#endif
        return 1;
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

template double num_one_like(const double &);

template long double num_one_like(const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_one_like(const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template mppp::real num_one_like(const mppp::real &);

#endif

template <typename T>
T num_eps_like([[maybe_unused]] const T &x)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return eps_from_prec(x.get_prec());
    } else {
#endif
        return std::numeric_limits<T>::epsilon();
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

template double num_eps_like(const double &);

template long double num_eps_like(const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_eps_like(const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template mppp::real num_eps_like(const mppp::real &);

#endif

template <typename T>
T num_inf_like([[maybe_unused]] const T &x)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return mppp::real{mppp::real_kind::inf, x.get_prec()};
    } else {
#endif
        return std::numeric_limits<T>::infinity();
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

template double num_inf_like(const double &);

template long double num_inf_like(const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template mppp::real128 num_inf_like(const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template mppp::real num_inf_like(const mppp::real &);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
