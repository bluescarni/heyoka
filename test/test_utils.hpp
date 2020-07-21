// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TEST_UTILS_HPP
#define HEYOKA_TEST_UTILS_HPP

#include <limits>
#include <ostream>
#include <sstream>

#include <heyoka/detail/math_wrappers.hpp>

namespace heyoka_test
{

template <typename T>
struct approximately {
    const T m_value;
    const T m_eps_mul;

    explicit approximately(T x, T eps_mul = T(10)) : m_value(x), m_eps_mul(eps_mul) {}
};

template <typename T>
inline bool operator==(const T &cmp, const approximately<T> &a)
{
    const auto tol = std::numeric_limits<T>::epsilon() * a.m_eps_mul;

    if (heyoka::detail::abs(cmp) < tol) {
        return heyoka::detail::abs(cmp - a.m_value) <= tol;
    } else {
        return heyoka::detail::abs((cmp - a.m_value) / cmp) <= tol;
    }
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const approximately<T> &a)
{
    std::ostringstream oss;
    oss.precision(std::numeric_limits<T>::max_digits10);
    oss << a.m_value;

    return os << oss.str();
}

} // namespace heyoka_test

#endif
