// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <fmt/format.h>

#include <heyoka/celmec/vsop2013.hpp>
#include <heyoka/config.hpp>
#include <heyoka/detail/vsop2013/vsop2013_1.hpp>
#include <heyoka/detail/vsop2013/vsop2013_2.hpp>
#include <heyoka/detail/vsop2013/vsop2013_3.hpp>
#include <heyoka/detail/vsop2013/vsop2013_4.hpp>
#include <heyoka/detail/vsop2013/vsop2013_5.hpp>
#include <heyoka/detail/vsop2013/vsop2013_6.hpp>
#include <heyoka/detail/vsop2013/vsop2013_7.hpp>
#include <heyoka/detail/vsop2013/vsop2013_8.hpp>
#include <heyoka/detail/vsop2013/vsop2013_9.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Hasher for use in the map below.
struct vsop2013_hasher {
    std::size_t operator()(const std::pair<std::uint32_t, std::uint32_t> &p) const
    {
        std::size_t seed = std::hash<std::uint32_t>{}(p.first);
        boost::hash_combine(seed, p.second);
        return seed;
    }
};

// This dictionary will map a pair of indices (i, j)
// (planet index and variable index) to a tuple containing:
// - the maximum value of alpha (the time power) + 1,
// - a pointer to an array containing the sizes of the series'
//   chunks for each value of alpha,
// - a pointer to an array of pointers, i.e., a jagged 2D array
//   in which each row contains the data necessary to build
//   the chunks for each value of alpha, from 0 to max_alpha.
using vsop2013_data_t
    = std::unordered_map<std::pair<std::uint32_t, std::uint32_t>,
                         std::tuple<std::size_t, const unsigned long *, const double *const *>, vsop2013_hasher>;

// Helper to construct the data dictionary that
// we will be querying for constructing the series
// at runtime.
auto build_vsop2103_data()
{
    vsop2013_data_t retval;

#define HEYOKA_VSOP2013_RECORD_DATA(pl_idx, var_idx)                                                                   \
    retval[{pl_idx, var_idx}]                                                                                          \
        = std::tuple{std::size(vsop2013_##pl_idx##_##var_idx##_sizes), &vsop2013_##pl_idx##_##var_idx##_sizes[0],      \
                     &vsop2013_##pl_idx##_##var_idx##_data[0]}

    HEYOKA_VSOP2013_RECORD_DATA(1, 1);
    HEYOKA_VSOP2013_RECORD_DATA(1, 2);
    HEYOKA_VSOP2013_RECORD_DATA(1, 3);
    HEYOKA_VSOP2013_RECORD_DATA(1, 4);
    HEYOKA_VSOP2013_RECORD_DATA(1, 5);
    HEYOKA_VSOP2013_RECORD_DATA(1, 6);

    HEYOKA_VSOP2013_RECORD_DATA(2, 1);
    HEYOKA_VSOP2013_RECORD_DATA(2, 2);
    HEYOKA_VSOP2013_RECORD_DATA(2, 3);
    HEYOKA_VSOP2013_RECORD_DATA(2, 4);
    HEYOKA_VSOP2013_RECORD_DATA(2, 5);
    HEYOKA_VSOP2013_RECORD_DATA(2, 6);

    HEYOKA_VSOP2013_RECORD_DATA(3, 1);
    HEYOKA_VSOP2013_RECORD_DATA(3, 2);
    HEYOKA_VSOP2013_RECORD_DATA(3, 3);
    HEYOKA_VSOP2013_RECORD_DATA(3, 4);
    HEYOKA_VSOP2013_RECORD_DATA(3, 5);
    HEYOKA_VSOP2013_RECORD_DATA(3, 6);

    HEYOKA_VSOP2013_RECORD_DATA(4, 1);
    HEYOKA_VSOP2013_RECORD_DATA(4, 2);
    HEYOKA_VSOP2013_RECORD_DATA(4, 3);
    HEYOKA_VSOP2013_RECORD_DATA(4, 4);
    HEYOKA_VSOP2013_RECORD_DATA(4, 5);
    HEYOKA_VSOP2013_RECORD_DATA(4, 6);

    HEYOKA_VSOP2013_RECORD_DATA(5, 1);
    HEYOKA_VSOP2013_RECORD_DATA(5, 2);
    HEYOKA_VSOP2013_RECORD_DATA(5, 3);
    HEYOKA_VSOP2013_RECORD_DATA(5, 4);
    HEYOKA_VSOP2013_RECORD_DATA(5, 5);
    HEYOKA_VSOP2013_RECORD_DATA(5, 6);

    HEYOKA_VSOP2013_RECORD_DATA(6, 1);
    HEYOKA_VSOP2013_RECORD_DATA(6, 2);
    HEYOKA_VSOP2013_RECORD_DATA(6, 3);
    HEYOKA_VSOP2013_RECORD_DATA(6, 4);
    HEYOKA_VSOP2013_RECORD_DATA(6, 5);
    HEYOKA_VSOP2013_RECORD_DATA(6, 6);

    HEYOKA_VSOP2013_RECORD_DATA(7, 1);
    HEYOKA_VSOP2013_RECORD_DATA(7, 2);
    HEYOKA_VSOP2013_RECORD_DATA(7, 3);
    HEYOKA_VSOP2013_RECORD_DATA(7, 4);
    HEYOKA_VSOP2013_RECORD_DATA(7, 5);
    HEYOKA_VSOP2013_RECORD_DATA(7, 6);

    HEYOKA_VSOP2013_RECORD_DATA(8, 1);
    HEYOKA_VSOP2013_RECORD_DATA(8, 2);
    HEYOKA_VSOP2013_RECORD_DATA(8, 3);
    HEYOKA_VSOP2013_RECORD_DATA(8, 4);
    HEYOKA_VSOP2013_RECORD_DATA(8, 5);
    HEYOKA_VSOP2013_RECORD_DATA(8, 6);

    HEYOKA_VSOP2013_RECORD_DATA(9, 1);
    HEYOKA_VSOP2013_RECORD_DATA(9, 2);
    HEYOKA_VSOP2013_RECORD_DATA(9, 3);
    HEYOKA_VSOP2013_RECORD_DATA(9, 4);
    HEYOKA_VSOP2013_RECORD_DATA(9, 5);
    HEYOKA_VSOP2013_RECORD_DATA(9, 6);

#undef HEYOKA_VSOP2013_RECORD_DATA

    return retval;
}

} // namespace

// Implementation of the function constructing the VSOP2013 elliptic series as heyoka expressions. The elements
// are referred to the Dynamical Frame J2000.
expression vsop2013_elliptic_impl(std::uint32_t pl_idx, std::uint32_t var_idx, expression t_expr, double thresh)
{
    // Check the input values.
    if (pl_idx < 1u || pl_idx > 9u) {
        throw std::invalid_argument(fmt::format("Invalid planet index passed to vsop2013_elliptic(): "
                                                "the index must be in the [1, 9] range, but it is {} instead",
                                                pl_idx));
    }

    if (var_idx < 1u || var_idx > 6u) {
        throw std::invalid_argument(fmt::format("Invalid variable index passed to vsop2013_elliptic(): "
                                                "the index must be in the [1, 6] range, but it is {} instead",
                                                var_idx));
    }

    if (!std::isfinite(thresh) || thresh < 0.) {
        throw std::invalid_argument(fmt::format("Invalid threshold value passed to vsop2013_elliptic(): "
                                                "the value must be finite and non-negative, but it is {} instead",
                                                thresh));
    }

    // The lambda_l values (constant + linear term).
    constexpr std::array<std::array<double, 2>, 17> lam_l_data = {{{4.402608631669, 26087.90314068555},
                                                                   {3.176134461576, 10213.28554743445},
                                                                   {1.753470369433, 6283.075850353215},
                                                                   {6.203500014141, 3340.612434145457},
                                                                   {4.091360003050, 1731.170452721855},
                                                                   {1.713740719173, 1704.450855027201},
                                                                   {5.598641292287, 1428.948917844273},
                                                                   {2.805136360408, 1364.756513629990},
                                                                   {2.326989734620, 1361.923207632842},
                                                                   {0.599546107035, 529.6909615623250},
                                                                   {0.874018510107, 213.2990861084880},
                                                                   {5.481225395663, 74.78165903077800},
                                                                   {5.311897933164, 38.13297222612500},
                                                                   {0, 0.3595362285049309},
                                                                   {5.198466400630, 77713.7714481804},
                                                                   {1.627905136020, 84334.6615717837},
                                                                   {2.355555638750, 83286.9142477147}}};

    // Fetch the data.
    static const auto data = build_vsop2103_data();

    // Locate the data entry for the current planet and variable.
    const auto data_it = data.find({pl_idx, var_idx});
    assert(data_it != data.end()); // LCOV_EXCL_LINE
    // NOTE: avoid structured bindings due to the usual
    // issues with lambda capture.
    const auto n_alpha = std::get<0>(data_it->second);
    const auto *const sizes_ptr = std::get<1>(data_it->second);
    const auto *const val_ptr = std::get<2>(data_it->second);

    // This vector will contain the chunks of the series
    // for different values of alpha.
    std::vector<expression> parts(boost::numeric_cast<std::vector<expression>::size_type>(n_alpha));

    tbb::parallel_for(tbb::blocked_range(static_cast<std::size_t>(0), n_alpha), [&](const auto &r) {
        for (auto alpha = r.begin(); alpha != r.end(); ++alpha) {
            // Fetch the number of terms for this chunk.
            const auto cur_size = sizes_ptr[alpha];

            // This vector will contain the terms of the chunk
            // for the current value of alpha.
            std::vector<expression> cur(boost::numeric_cast<std::vector<expression>::size_type>(cur_size));

            tbb::parallel_for(tbb::blocked_range(0ul, cur_size), [&](const auto &r_in) {
                // trig will contain the components of the
                // sin/cos trigonometric argument.
                auto trig = std::vector<expression>(17u);

                for (auto i = r_in.begin(); i != r_in.end(); ++i) {
                    // Load the C/S values from the table.
                    const auto Sval = val_ptr[alpha][i * 19u + 17u];
                    const auto Cval = val_ptr[alpha][i * 19u + 18u];

                    // Check if the term is too small.
                    if (std::sqrt(Cval * Cval + Sval * Sval) < thresh) {
                        continue;
                    }

                    for (std::size_t j = 0; j < 17u; ++j) {
                        // Compute lambda_l for the current element
                        // of the trigonometric argument.
                        auto cur_lam = lam_l_data[j][0] + t_expr * lam_l_data[j][1];

                        // Multiply it by the current value in the table.
                        trig[j] = std::move(cur_lam) * val_ptr[alpha][i * 19u + j];
                    }

                    // Compute the trig arg.
                    auto trig_arg = sum(trig);

                    // Add the term to the chunk.
                    auto tmp = Sval * sin(trig_arg);
                    cur[i] = std::move(tmp) + Cval * cos(std::move(trig_arg));
                }
            });

            // Partition cur so that all zero expressions (i.e., VSOP2013 terms which have
            // been skipped) are at the end. Use stable_partition so that the original ordering
            // is preserved.
            const auto new_end = std::stable_partition(cur.begin(), cur.end(), [](const expression &e) {
                if (const auto *num_ptr = std::get_if<number>(&e.value()); num_ptr != nullptr && is_zero(*num_ptr)) {
                    return false;
                } else {
                    return true;
                }
            });

            // Erase the skipped terms.
            cur.erase(new_end, cur.end());

            // Sum the terms in the chunk and multiply them by t**alpha.
            parts[alpha] = powi(t_expr, boost::numeric_cast<std::uint32_t>(alpha)) * sum(std::move(cur));
        }
    });

    // Sum the chunks and return them.
    return sum(std::move(parts));
}

namespace
{

// G*M values for the planets.
constexpr double vsop2013_gm_pl[] = {4.9125474514508118699e-11, 7.2434524861627027000e-10, 8.9970116036316091182e-10,
                                     9.5495351057792580598e-11, 2.8253458420837780000e-07, 8.4597151856806587398e-08,
                                     1.2920249167819693900e-08, 1.5243589007842762800e-08, 2.1886997654259696800e-12};

// G*M value for the Sun.
constexpr double vsop2013_gm_sun = 2.9591220836841438269e-04;

} // namespace

// Implementation of the function constructing the VSOP2013 Cartesian series as heyoka expressions. The coordinates
// are referred to the Dynamical Frame J2000.
std::vector<expression> vsop2013_cartesian_impl(std::uint32_t pl_idx, expression t_expr, double thresh)
{
    // Get the elliptic orbital elements.
    expression a, lam, k, h, q, p;

    tbb::parallel_invoke([&]() { a = vsop2013_elliptic_impl(pl_idx, 1, t_expr, thresh); },
                         [&]() { lam = vsop2013_elliptic_impl(pl_idx, 2, t_expr, thresh); },
                         [&]() { k = vsop2013_elliptic_impl(pl_idx, 3, t_expr, thresh); },
                         [&]() { h = vsop2013_elliptic_impl(pl_idx, 4, t_expr, thresh); },
                         [&]() { q = vsop2013_elliptic_impl(pl_idx, 5, t_expr, thresh); },
                         [&]() { p = vsop2013_elliptic_impl(pl_idx, 6, t_expr, thresh); });

    // M, k**2 + h**2, q**2 + p**2, sqrt(q**2 + p**2).
    expression M, kh_2, qp_2, qp;
    tbb::parallel_invoke([&]() { M = lam - atan2(h, k); }, [&]() { kh_2 = k * k + h * h; },
                         [&]() {
                             qp_2 = q * q + p * p;
                             qp = sqrt(qp_2);
                         });

    // E, e, sqrt(1 - e**2), cos(i), sin(i), cos(Om), sin(Om), sin(E), cos(E)
    expression E, e, sqrt_1me2, ci, si, cOm, sOm, sin_E, cos_E;
    tbb::parallel_invoke(
        [&]() {
            e = sqrt(kh_2);
            E = kepE(e, M);
            tbb::parallel_invoke([&]() { sin_E = sin(E); }, [&]() { cos_E = cos(E); });
        },
        [&]() { sqrt_1me2 = sqrt(1_dbl - kh_2); },
        [&]() {
            ci = 1_dbl - 2_dbl * qp_2;
            si = sqrt(1_dbl - ci * ci);
        },
        [&]() { cOm = q / qp; }, [&]() { sOm = p / qp; });

    // cos(om), sin(om), q1/a, q2/a.
    expression com, som, q1_a, q2_a;
    tbb::parallel_invoke([&]() { com = (k * cOm + h * sOm) / e; }, [&]() { som = (h * cOm - k * sOm) / e; },
                         [&]() { q1_a = cos_E - e; }, [&]() { q2_a = sqrt_1me2 * sin_E; });

    // Prepare the entries of the rotation matrix, and a few auxiliary quantities.
    expression R00, R01, R10, R11, R20, R21, v_num, v_den;
    tbb::parallel_invoke([&]() { R00 = cOm * com - sOm * ci * som; }, [&]() { R01 = cOm * som + sOm * ci * com; },
                         [&]() { R10 = sOm * com + cOm * ci * som; }, [&]() { R11 = sOm * som - cOm * ci * com; },
                         [&]() { R20 = si * som; }, [&]() { R21 = si * com; }, [&]() { v_num = sqrt_1me2 * cos_E; },
                         [&]() { v_den = sqrt(a) * (1_dbl - e * cos_E); });

    // Compute the gravitational parameter for pl_idx.
    assert(pl_idx >= 1u && pl_idx <= 9u); // LCOV_EXCL_LINE
    const auto mu = std::sqrt(vsop2013_gm_sun + vsop2013_gm_pl[pl_idx - 1u]);

    // Prepare the return value.
    std::vector<expression> retval(6u);

    tbb::parallel_invoke([&]() { retval[0] = a * (q1_a * R00 - q2_a * R01); },
                         [&]() { retval[1] = a * (q1_a * R10 - q2_a * R11); },
                         [&]() { retval[2] = a * (q1_a * R20 + q2_a * R21); },
                         [&]() { retval[3] = mu * (-sin_E * R00 - v_num * R01) / v_den; },
                         [&]() { retval[4] = mu * (-sin_E * R10 - v_num * R11) / v_den; },
                         [&]() { retval[5] = mu * (-sin_E * R20 + v_num * R21) / v_den; });

    return retval;
}

// Implementation of the function constructing the VSOP2013 Cartesian series as heyoka expressions. The coordinates
// are referred to ICRF frame.
std::vector<expression> vsop2013_cartesian_icrf_impl(std::uint32_t pl_idx, expression t_expr, double thresh)
{
    // Compute the Cartesian coordinates in the Dynamical Frame J2000.
    const auto cart_dfj2000 = vsop2013_cartesian_impl(pl_idx, std::move(t_expr), thresh);

    // The two rotation angles for the transition Dynamical Frame J2000 -> ICRF.
    const auto eps = 0.4090926265865962;
    const auto phi = -2.5152133775962285e-07;

    // Perform the rotation.
    const auto &xe = cart_dfj2000[0];
    const auto &ye = cart_dfj2000[1];
    const auto &ze = cart_dfj2000[2];
    const auto &vxe = cart_dfj2000[3];
    const auto &vye = cart_dfj2000[4];
    const auto &vze = cart_dfj2000[5];

    std::vector<expression> retval(6u);

    tbb::parallel_invoke(
        [&]() {
            retval[0] = std::cos(phi) * xe - std::sin(phi) * std::cos(eps) * ye + std::sin(phi) * std::sin(eps) * ze;
        },
        [&]() {
            retval[1] = std::sin(phi) * xe + std::cos(phi) * std::cos(eps) * ye - std::cos(phi) * std::sin(eps) * ze;
        },
        [&]() { retval[2] = std::sin(eps) * ye + std::cos(eps) * ze; },
        [&]() {
            retval[3] = std::cos(phi) * vxe - std::sin(phi) * std::cos(eps) * vye + std::sin(phi) * std::sin(eps) * vze;
        },
        [&]() {
            retval[4] = std::sin(phi) * vxe + std::cos(phi) * std::cos(eps) * vye - std::cos(phi) * std::sin(eps) * vze;
        },
        [&]() { retval[5] = std::sin(eps) * vye + std::cos(eps) * vze; });

    return retval;
}

} // namespace detail

std::array<double, 10> get_vsop2013_mus()
{
    std::array<double, 10> retval{};

    retval[0] = heyoka::detail::vsop2013_gm_sun;

    std::copy(std::begin(heyoka::detail::vsop2013_gm_pl), std::end(heyoka::detail::vsop2013_gm_pl), retval.data() + 1);

    return retval;
}

HEYOKA_END_NAMESPACE
