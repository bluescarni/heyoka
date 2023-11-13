// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/elp2000/elp2000_1_3.hpp>
#include <heyoka/detail/fast_unordered.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/elp2000.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

constexpr std::array W1 = {3.8103444305883079, 8399.6847317739157, -2.8547283984772807e-05, 3.2017095500473753e-08,
                           -1.5363745554361197e-10};
// constexpr std::array W2 = {1.4547885323225087, 70.993304818359618, -0.00018557504160038375, -2.1839401892941265e-07,
//                            1.0327016221314225e-09};
// constexpr std::array W3
//     = {2.1824391972168398, -33.781426356625921, 3.08448160195509e-05, 3.6967043184602116e-08,
//     -1.738541860458796e-10};
// constexpr std::array T
//     = {1.753470343150658, 628.30758496215537,
//     -9.7932363584126268e-08, 4.3633231299858238e-11, 7.2722052166430391e-13};

constexpr std::array D = {5.1984667410274437, 7771.3771468120494, -2.8449351621188683e-05, 3.1973462269173901e-08,
                          -1.5436467606527627e-10};
constexpr std::array lp = {6.2400601269714615, 628.30195516800313, -2.680534842854624e-06, 7.1267611123101784e-10};
constexpr std::array l
    = {2.3555558982657985, 8328.6914269553617, 0.00015702775761561094, 2.5041111442988642e-07, -1.1863390776750345e-09};
constexpr std::array F
    = {1.6279052333714679, 8433.4661581308319, -5.9392100004323707e-05, -4.9499476841283623e-09, 2.021673050226765e-11};

// Create the expression for the evaluation of the polynomial with coefficients
// stored in dense form in cfs according to Horner's scheme.
template <typename T, std::size_t N>
expression horner_eval(const std::array<T, N> &cfs, const expression &x)
{
    static_assert(N > 0u);

    auto ret = expression(cfs[N - 1u]);

    for (std::size_t i = 1; i < N; ++i) {
        ret = fix_nn(cfs[N - i - 1u] + ret * x);
    }

    return ret;
}

// Complex multiplication.
std::array<expression, 2> ex_cmul(const std::array<expression, 2> &c1, const std::array<expression, 2> &c2)
{
    const auto &[a, b] = c1;
    const auto &[c, d] = c2;

    return {fix_nn(a * c - b * d), fix_nn(b * c + a * d)};
}

// Complex inversion.
std::array<expression, 2> ex_cinv(const std::array<expression, 2> &c)
{
    const auto &[a, b] = c;

    const auto den = fix_nn(a * a + b * b);

    return {fix_nn(a / den), fix_nn(-b / den)};
}

// Dictionary to map an integral exponent to the corresponding integral power of a complex expression.
using pow_dict_t = heyoka::detail::fast_umap<std::int8_t, std::array<expression, 2>>;

// Dictionary to map an expression "ex" to a a dictionary of integral powers of
// cos(ex) + im * sin(ex).
using trig_eval_dict_t = heyoka::detail::fast_umap<expression, pow_dict_t, std::hash<expression>>;

// NOLINTNEXTLINE(misc-no-recursion)
std::array<expression, 2> ccpow_impl(pow_dict_t &pd, const std::array<expression, 2> &pow1,
                                     const std::array<expression, 2> &powm1, std::int8_t n)
{
    auto it = pd.find(n);

    if (it != pd.end()) {
        return it->second;
    }

    std::array<expression, 2> ret;
    if (n >= 0) {
        ret = ex_cmul(pow1, ccpow_impl(pd, pow1, powm1, static_cast<std::int8_t>(n - 1)));
    } else {
        ret = ex_cmul(powm1, ccpow_impl(pd, pow1, powm1, static_cast<std::int8_t>(n + 1)));
    }

    [[maybe_unused]] auto [_, flag] = pd.try_emplace(n, ret);
    assert(flag);

    return ret;
}

// Implementation of complex integral exponentiation of cos(ex) + im * sin(ex)
// supported by a cache.
std::array<expression, 2> ccpow(const expression &ex, trig_eval_dict_t &td, std::int8_t n)
{
    auto it = td.find(ex);
    assert(it != td.end());

    auto &pd = it->second;

    assert(pd.find(1) != pd.end());
    assert(pd.find(-1) != pd.end());

    const auto pow1 = pd[1];
    const auto powm1 = pd[-1];

    return ccpow_impl(pd, pow1, powm1, n);
}

// Pairwise complex product.
std::array<expression, 2> pairwise_cmul(std::vector<std::array<expression, 2>> &terms)
{
    if (terms.empty()) {
        return {1_dbl, 0_dbl};
    }

    // LCOV_EXCL_START
    if (terms.size() == std::numeric_limits<decltype(terms.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_cmul()");
    }
    // LCOV_EXCL_STOP

    while (terms.size() != 1u) {
        std::vector<std::array<expression, 2>> new_terms;

        for (decltype(terms.size()) i = 0; i < terms.size(); i += 2u) {
            if (i + 1u == terms.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_terms.push_back(terms[i]);
            } else {
                new_terms.push_back(ex_cmul(terms[i], terms[i + 1u]));
            }
        }

        new_terms.swap(terms);
    }

    return terms[0];
}

} // namespace

std::vector<expression> elp2000_spherical_impl(const expression &tm, double thresh)
{
    if (!std::isfinite(thresh) || thresh < 0.) {
        throw std::invalid_argument(fmt::format("Invalid threshold value passed to elp2000_spherical(): "
                                                "the value must be finite and non-negative, but it is {} instead",
                                                thresh));
    }

    // Evaluate the arguments and time-dependent quantities.
    const auto W1_eval = horner_eval(W1, tm);
    const auto D_eval = horner_eval(D, tm);
    const auto lp_eval = horner_eval(lp, tm);
    const auto l_eval = horner_eval(l, tm);
    const auto F_eval = horner_eval(F, tm);

    // Seed the trig eval dictionary with powers of 0, 1 and -1 for each
    // argument.
    trig_eval_dict_t trig_eval;

    auto seed_trig_eval = [&trig_eval](const expression &arg) {
        const auto [it, flag] = trig_eval.insert({arg, {}});
        assert(flag);
        auto &pd = it->second;

        const std::array c_arg = {cos(arg), sin(arg)};

        pd[0] = {1_dbl, 0_dbl};
        pd[1] = c_arg;
        pd[-1] = ex_cinv(c_arg);
    };

    seed_trig_eval(D_eval);
    seed_trig_eval(lp_eval);
    seed_trig_eval(l_eval);
    seed_trig_eval(F_eval);

    // Temporary accumulation list for products.
    std::vector<std::array<expression, 2>> tmp_cprod;

    // Longitude.
    std::vector<expression> V_terms{W1_eval};

    // ELP1.
    {
        const std::array args = {D_eval, lp_eval, l_eval, F_eval};
        for (std::size_t i = 0; i < std::size(elp2000_idx_1); ++i) {
            const auto &[cur_A] = elp2000_A_1[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_1[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(fix_nn(cur_A * cprod[1]));
            }
        }
    }

    // Latitude.
    std::vector<expression> U_terms;

    // ELP2.
    {
        const std::array args = {D_eval, lp_eval, l_eval, F_eval};
        for (std::size_t i = 0; i < std::size(elp2000_idx_2); ++i) {
            const auto &[cur_A] = elp2000_A_2[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_2[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(fix_nn(cur_A * cprod[1]));
            }
        }
    }

    // Distance.
    std::vector<expression> r_terms;

    // ELP3.
    {
        const std::array args = {D_eval, lp_eval, l_eval, F_eval};
        for (std::size_t i = 0; i < std::size(elp2000_idx_3); ++i) {
            const auto &[cur_A] = elp2000_A_3[i];

            if (std::abs(cur_A / 384400.) > thresh) {
                const auto &cur_idx_v = elp2000_idx_3[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(fix_nn(cur_A * cprod[0]));
            }
        }
    }

    return {sum(r_terms), sum(U_terms), sum(V_terms)};
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
