// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this is a prototype of collision detection
// based on the jet of derivatives in an N-body problem.
// References:
// https://en.wikipedia.org/wiki/Real-root_isolation
// https://en.wikipedia.org/wiki/Descartes%27_rule_of_signs
// https://en.wikipedia.org/wiki/Budan%27s_theorem

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

#include <boost/iterator/filter_iterator.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/llvm_state.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include <iostream>

// Compute all binomial coefficients (n choose k) up to
// n = max_n.
template <typename T>
auto make_binomial_coefficients(std::uint32_t max_n)
{
    assert(max_n >= 2u);

    std::vector<T> retval;
    retval.resize(((max_n + 1u) * (max_n + 2u)) / 2u);

    // Fill up to n = 2.

    // 0 choose 0.
    retval[0] = 1;

    // 1 choose 0.
    retval[1] = 1;
    // 1 choose 1.
    retval[2] = 1;

    // 2 choose 0.
    retval[3] = 1;
    // 2 choose 1.
    retval[4] = 2;
    // 2 choose 2.
    retval[5] = 1;

    // Iterate using the recursion formula.
    std::uint32_t base_idx = 6;
    for (std::uint32_t n = 3; n <= max_n; base_idx += ++n) {
        // n choose 0 = 1.
        retval[base_idx] = 1;

        // NOTE: the recursion formula is valid up to k = n - 1.
        const auto prev_base_idx = base_idx - n;
        for (std::uint32_t k = 1; k < n; ++k) {
            retval[base_idx + k] = retval[prev_base_idx + k] + retval[prev_base_idx + k - 1u];
        }

        // n choose n = 1.
        retval[base_idx + n] = 1;
    }

    return retval;
}

// Fetch the index of (n choose k) in a vector produced by
// make_binomial_coefficients().
std::uint32_t bc_idx(std::uint32_t n, std::uint32_t k)
{
    assert(k <= n);

    return (n * (n + 1u)) / 2u + k;
}

// Polynomial add.
// NOTE: aliasing allowed.
template <typename Ret, typename T, typename U>
void poly_add(Ret &ret, const T &a, const U &b, std::uint32_t n)
{
    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = a[i] + b[i];
    }
}

// NOTE: this is not really a polynomial squaring, as we
// discard in the result all terms of degree higher
// than n. More like truncated power series squaring.
// NOTE: aliasing NOT allowed.
template <typename Ret, typename T>
void poly_square(Ret &ret, const T &a, std::uint32_t n)
{
    // Zero out the return value.
    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = 0;
    }

    for (std::uint32_t i = 0; i <= n / 2u; ++i) {
        ret[2u * i] += a[i] * a[i];

        for (auto j = i + 1u; j <= n - i; ++j) {
            ret[i + j] += 2 * a[i] * a[j];
        }
    }
}

// Revert the order of the coefficients in a polynomial.
// This corresponds to changing the variable from x
// to 1/x**n and then multiplying the resulting
// expression by x**n.
// NOTE: aliasing NOT allowed.
template <typename Ret, typename T>
void poly_reverse(Ret &ret, const T &a, std::uint32_t n)
{
    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = a[n - i];
    }
}

// Substitute the polynomial variable x with x + 1,
// and write the resulting polynomial in ret. bcs
// is a vector containing the binomial coefficients
// up to to (n choose n) as produced by make_binomial_coefficients().
// NOTE: aliasing NOT allowed.
template <typename Ret, typename T, typename U>
void poly_translate_1(Ret &ret, const T &a, std::uint32_t n, const U &bcs)
{
    // Zero out the return value.
    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = 0;
    }

    for (std::uint32_t i = 0; i <= n; ++i) {
        const auto ai = a[i];
        for (std::uint32_t k = 0; k <= i; ++k) {
            ret[k] += ai * bcs[bc_idx(i, k)];
        }
    }
}

// Transform the polynomial a(x) into scal**n * a(x / scal).
// NOTE: aliasing allowed.
template <typename Ret, typename T, typename U>
void poly_rescale1(Ret &ret, const T &a, const U &scal, std::uint32_t n)
{
    U cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[n - i] = cur_f * a[n - i];
        cur_f *= scal;
    }
}

// Transform the polynomial a(x) into a(x * scal).
// NOTE: aliasing allowed.
template <typename Ret, typename T, typename U>
void poly_rescale2(Ret &ret, const T &a, const U &scal, std::uint32_t n)
{
    U cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = cur_f * a[i];
        cur_f *= scal;
    }
}

struct zero_filter {
    template <typename U>
    bool operator()(const U &x) const
    {
        return x != 0;
    }
};

// Count the number of sign changes in the coefficients of polynomial a.
// Zero coefficients are skipped.
template <typename T>
std::uint32_t count_sign_changes(const T &a, std::uint32_t n)
{
    // Create iterators for skipping zero coefficients in the polynomial.
    auto begin = boost::make_filter_iterator<zero_filter>(a.begin(), a.begin() + (n + 1u));
    const auto end = boost::make_filter_iterator<zero_filter>(a.begin() + (n + 1u), a.begin() + (n + 1u));

    if (begin == end) {
        return 0;
    }

    std::uint32_t retval = 0;

    auto prev = begin;
    for (++begin; begin != end; ++begin, ++prev) {
        retval += (*begin > 0) != (*prev > 0);
    }

    return retval;
}

// Horner's polynomial evaluation.
template <typename T, typename U>
auto horner_eval(const T &a, const U &x, std::uint32_t n)
{
    assert(n > 0u);

    auto retval(a[n]);

    for (std::uint32_t i = 1; i <= n; ++i) {
        retval = a[n - i] + retval * x;
    }

    return retval;
}

using namespace heyoka;

int main()
{
    std::cout.precision(16);

    const auto order = 20;

    const auto bcs = make_binomial_coefficients<double>(order);

    llvm_state s;

    taylor_add_jet<double>(
        s, "jet",
        make_nbody_sys(6, kw::masses = {1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09},
                       kw::Gconst = 0.01720209895 * 0.01720209895),
        order, 1, true, true);

    s.compile();

    auto jptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

    auto state = std::vector{// Sun.
                             -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6, +6.69048890636161e-6,
                             -6.33922479583593e-6, -3.13202145590767e-9,
                             // Jupiter.
                             +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2, -5.59797969310664e-3,
                             +5.51815399480116e-3, -2.66711392865591e-6,
                             // Saturn.
                             +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1, -4.17354020307064e-3,
                             +3.99723751748116e-3, +1.67206320571441e-5,
                             // Uranus.
                             +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1, -3.25884806151064e-3,
                             +2.06438412905916e-3, -2.17699042180559e-5,
                             // Neptune.
                             -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1, -2.17471785045538e-4,
                             -3.11361111025884e-3, +3.58344705491441e-5,
                             // Pluto.
                             -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0, -1.76936577252484e-3,
                             -2.06720938381724e-3, +6.58091931493844e-4};
    state.resize(36 * (order + 1));

    auto s_array = xt::adapt(state, {order + 1, 6, 6});

    // Prepare temporary storage.
    std::vector<double> dx2(order + 1), dy2(order + 1), dz2(order + 1), r2(order + 1);
    std::vector<double> tmp1(order + 1), tmp2(order + 1), tmp3(order + 1);

    // The integration timestep.
    const auto h = 180.;

    // Compute the jet of derivatives.
    jptr(state.data(), nullptr, nullptr);

    // Compute the distance**2 between 2 bodies.
    poly_square(dx2, xt::view(s_array, xt::all(), 1, 0) - xt::view(s_array, xt::all(), 2, 0), order);
    poly_square(dy2, xt::view(s_array, xt::all(), 1, 1) - xt::view(s_array, xt::all(), 2, 1), order);
    poly_square(dz2, xt::view(s_array, xt::all(), 1, 2) - xt::view(s_array, xt::all(), 2, 2), order);
    poly_add(r2, dx2, dy2, order);
    poly_add(r2, r2, dz2, order);

    // NOTE: as long as this line is commented, the code below will detect
    // perfect overlaps of the 2 bodies. Uncomment this to detect the distance
    // square of the two bodies becoming, e.g., 16 within the integration timestep.
    // r2[0] -= 16;

    // Rescale r2 so that the range [0, h]
    // becomes [0, 1].
    poly_rescale2(r2, r2, h, order);

    // Init the working list.
    std::vector<std::tuple<std::uint32_t, std::uint32_t, std::vector<double>>> w_list;
    // The initial element in the list is the original range [0, 1]
    // and the initial polynomial.
    w_list.emplace_back(0, 0, r2);

    // The list of isolating intervals.
    std::vector<std::tuple<std::uint32_t, std::uint32_t>> isol;

    while (!w_list.empty()) {
        // Fetch the current inverval/polynomial from the working list.
        auto [c, k, q] = std::move(w_list.back());
        w_list.pop_back();

        // Transform q(x) into (x+1)**n * q(1/(x+1)).
        // NOTE: need to reverse first and then do the translation.
        poly_reverse(tmp1, q, order);
        poly_translate_1(tmp2, tmp1, order, bcs);

        // std::cout << "New eval: " << horner_eval(tmp2, .5, order) << '\n';
        // std::cout << "Compare: " << pow(1 + .5, 20) * horner_eval(q, 1 / (1 + .5), order) << '\n';

        // Count the number of sign changes in tmp2.
        const auto n_sc = count_sign_changes(tmp2, order);

        std::cout << "n_sc: " << n_sc << '\n';

        if (n_sc == 1u) {
            // Found isolating interval, add it to isol.
            isol.emplace_back(c, k);
        } else if (n_sc > 1u) {
            // No isolating interval found, bisect.
            poly_rescale1(tmp2, q, 2., order);
            w_list.emplace_back(2u * c, k + 1u, tmp2);

            // NOTE: tmp2 is q rescaled, need only
            // to translate it.
            poly_translate_1(tmp3, tmp2, order, bcs);
            w_list.emplace_back(2u * c + 1u, k + 1u, tmp3);
        }

        std::cout << "w_list size: " << w_list.size() << '\n';
    }

    std::cout << "Isolating invervals:\n";
    for (const auto &[c, k] : isol) {
        std::cout << "[" << (static_cast<double>(c) / (1u << k)) * h << ", "
                  << ((static_cast<double>(c) + 1) / (1u << k)) * h << "]\n";
    }
}
