// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_BENCHMARK_UTILS_HPP
#define HEYOKA_BENCHMARK_UTILS_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xshape.hpp>

namespace heyoka_benchmark
{

template <typename T>
inline std::array<T, 3> cross(std::array<T, 3> a, std::array<T, 3> b)
{
    auto [a1, a2, a3] = a;
    auto [b1, b2, b3] = b;

    return {a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1};
}

template <typename T>
inline T dot(std::array<T, 3> a, std::array<T, 3> b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline T norm2(std::array<T, 3> x)
{
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
}

template <typename T>
inline T norm(std::array<T, 3> x)
{
    using std::sqrt;

    return sqrt(norm2(x));
}

template <typename T>
inline std::array<T, 3> sub(std::array<T, 3> a, std::array<T, 3> b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template <typename T>
inline std::array<T, 3> div(std::array<T, 3> a, T c)
{
    return {a[0] / c, a[1] / c, a[2] / c};
}

template <typename T>
inline std::array<T, 6> cart_to_kep(std::array<T, 3> x, std::array<T, 3> v, T mu)
{
    using std::acos;

    const auto h = cross(x, v);
    const auto e_v = sub(div(cross(v, h), mu), div(x, norm(x)));
    const auto n = std::array{-h[1], h[0], T(0)};

    auto nu = acos(dot(e_v, x) / (norm(e_v) * norm(x)));
    if (dot(x, v) < 0) {
        nu = 2 * acos(T(-1)) - nu;
    }

    const auto i = acos(h[2] / norm(h));

    const auto e = norm(e_v);

    auto Om = acos(n[0] / norm(n));
    if (n[1] < 0) {
        Om = 2 * acos(T(-1)) - Om;
    }

    auto om = acos(dot(n, e_v) / (norm(n) * norm(e_v)));
    if (e_v[2] < 0) {
        om = 2 * acos(T(-1)) - om;
    }

    const auto a = 1 / (2 / norm(x) - norm2(v) / mu);

    return {a, e, i, om, Om, nu};
}

template <typename T>
inline std::pair<std::array<T, 3>, std::array<T, 3>> kep_to_cart(std::array<T, 6> kep, T mu)
{
    using std::atan;
    using std::cos;
    using std::sin;
    using std::sqrt;
    using std::tan;

    auto [a, e, i, om, Om, nu] = kep;

    const auto E = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(nu / 2));

    const auto n = sqrt(mu / (a * a * a));

    const std::array<T, 3> q = {a * (cos(E) - e), a * sqrt(1 - e * e) * sin(E), T(0)};
    const std::array<T, 3> vq
        = {-n * a * sin(E) / (1 - e * cos(E)), n * a * sqrt(1 - e * e) * cos(E) / (1 - e * cos(E)), T(0)};

    const std::array<T, 3> r1 = {cos(Om) * cos(om) - sin(Om) * cos(i) * sin(om),
                                 -cos(Om) * sin(om) - sin(Om) * cos(i) * cos(om), sin(Om) * sin(i)};
    const std::array<T, 3> r2 = {sin(Om) * cos(om) + cos(Om) * cos(i) * sin(om),
                                 -sin(Om) * sin(om) + cos(Om) * cos(i) * cos(om), -cos(Om) * sin(i)};
    const std::array<T, 3> r3 = {sin(i) * sin(om), sin(i) * cos(om), cos(i)};

    std::array<T, 3> x = {dot(r1, q), dot(r2, q), dot(r3, q)};
    std::array<T, 3> v = {dot(r1, vq), dot(r2, vq), dot(r3, vq)};

    return std::pair{x, v};
}

// 1-D array type of fixed size N.
template <typename T, std::size_t N>
using vNd = xt::xtensor_fixed<T, xt::xshape<N>>;

// Custom implementation of norm() to avoid calling into lapack,
// which does not support types other than float/double.
template <typename E>
inline auto norm(const E &e)
{
    using std::sqrt;

    return sqrt(xt::linalg::dot(e, e)[0]);
}

template <typename E1, typename E2, typename T>
inline vNd<T, 6> cart_to_kep(const E1 &x, const E2 &v, T mu)
{
    static_assert(std::is_same_v<typename E1::value_type, T>);
    static_assert(std::is_same_v<typename E2::value_type, T>);

    using std::acos;

    const auto h = xt::linalg::cross(x, v);
    const auto e_v = xt::linalg::cross(v, h) / mu - x / norm(x);
    const vNd<T, 3> n = {-h[1], h[0], T(0)};

    auto nu = acos(xt::linalg::dot(e_v, x)[0] / (norm(e_v) * norm(x)));
    if (xt::linalg::dot(x, v)[0] < 0) {
        nu = 2 * acos(T(-1)) - nu;
    }

    const auto i = acos(h[2] / norm(h));

    const auto e = norm(e_v);

    auto Om = acos(n[0] / norm(n));
    if (n[1] < 0) {
        Om = 2 * acos(T(-1)) - Om;
    }

    auto om = acos(xt::linalg::dot(n, e_v)[0] / (norm(n) * norm(e_v)));
    if (e_v[2] < 0) {
        om = 2 * acos(T(-1)) - om;
    }

    const auto a = 1 / (2 / norm(x) - xt::linalg::dot(v, v)[0] / mu);

    return {a, e, i, om, Om, nu};
}

void warmup()
{
    std::cout << "Warming up";
    for (auto volatile counter = 0ull; counter < 1000000000ull; ++counter) {
        if (counter % 100000000ull == 0u) {
            std::cout << '.';
            std::cout.flush();
        }
    }
    std::cout << " Done\n";
}

} // namespace heyoka_benchmark

#endif
