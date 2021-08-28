// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <boost/numeric/conversion/cast.hpp>
#include <boost/version.hpp>

// NOTE: the header for hash_combine changed in version 1.67.
#if (BOOST_VERSION / 100000 > 1) || (BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 >= 67)

#include <boost/container_hash/hash.hpp>

#else

#include <boost/functional/hash.hpp>

#endif

#include <tbb/blocked_range.h>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <fmt/format.h>

#include <heyoka/celmec/vsop2013.hpp>
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
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka::detail
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
        throw std::invalid_argument("Invalid planet index passed to vsop2013_elliptic(): "
                                    "the index must be in the [1, 9] range, but it is {} instead"_format(pl_idx));
    }

    if (var_idx < 1u || var_idx > 6u) {
        throw std::invalid_argument("Invalid variable index passed to vsop2013_elliptic(): "
                                    "the index must be in the [1, 6] range, but it is {} instead"_format(var_idx));
    }

    if (!std::isfinite(thresh) || thresh < 0.) {
        throw std::invalid_argument("Invalid threshold value passed to vsop2013_elliptic(): "
                                    "the value must be finite and non-negative, but it is {} instead"_format(thresh));
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
    const auto sizes_ptr = std::get<1>(data_it->second);
    const auto val_ptr = std::get<2>(data_it->second);

    // This vector will contain the chunks of the series
    // for different values of alpha.
    std::vector<expression> parts(boost::numeric_cast<std::vector<expression>::size_type>(n_alpha));

    tbb::parallel_for(tbb::blocked_range(std::size_t(0), n_alpha), [&](const auto &r) {
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
                    auto trig_arg = pairwise_sum(trig);

                    // Add the term to the chunk.
                    auto tmp = Sval * sin(trig_arg);
                    cur[i] = std::move(tmp) + Cval * cos(std::move(trig_arg));
                }
            });

            // Partition cur so that all zero expressions (i.e., VSOP2013 terms which have
            // been skipped) are at the end. Use stable_partition so that the original ordering
            // is preserved.
            const auto new_end = std::stable_partition(cur.begin(), cur.end(), [](const expression &e) {
                if (auto num_ptr = std::get_if<number>(&e.value()); num_ptr != nullptr && is_zero(*num_ptr)) {
                    return false;
                } else {
                    return true;
                }
            });

            // Erase the skipped terms.
            cur.erase(new_end, cur.end());

            // Sum the terms in the chunk and multiply them by t**alpha.
            parts[alpha] = powi(t_expr, boost::numeric_cast<std::uint32_t>(alpha)) * pairwise_sum(std::move(cur));
        }
    });

    // Sum the chunks and return them.
    return pairwise_sum(std::move(parts));
}

// Implementation of the function constructing the VSOP2013 Cartesian series as heyoka expressions. The coordinates
// are referred to the Dynamical Frame J2000.
std::vector<expression> vsop2013_cartesian_impl(std::uint32_t pl_idx, expression t_expr, double thresh)
{
#if 0
    namespace flow = tbb::flow;

    // G*M values for the planets.
    constexpr double gm_pl[] = {4.9125474514508118699e-11, 7.2434524861627027000e-10, 8.9970116036316091182e-10,
                                9.5495351057792580598e-11, 2.8253458420837780000e-07, 8.4597151856806587398e-08,
                                1.2920249167819693900e-08, 1.5243589007842762800e-08, 2.1886997654259696800e-12};

    // G*M value for the Sun.
    constexpr double gm_sun = 2.9591220836841438269e-04;

    // Compute the gravitational parameter for pl_idx.
    assert(pl_idx >= 1u && pl_idx <= 9u); // LCOV_EXCL_LINE
    const auto mu = std::sqrt(gm_sun + gm_pl[pl_idx - 1u]);

    // Prepare the return value.
    std::vector<expression> retval(6u);

    expression a, lam, k, h, q, p, M, kh_2, qp_2, qp, E, e, sqrt_1me2, ci, si, cOm, sOm, sin_E, cos_E, com, som, q1_a,
        q2_a, R00, R01, R10, R11, R20, R21, v_num, v_den;

    flow::graph g;

    flow::broadcast_node<flow::continue_msg> start(g);

    // Get the elliptic orbital elements.
    flow::continue_node<flow::continue_msg> a_node(
        g, [&](auto) { a = vsop2013_elliptic_impl(pl_idx, 1, t_expr, thresh); });
    flow::continue_node<flow::continue_msg> lam_node(
        g, [&](auto) { lam = vsop2013_elliptic_impl(pl_idx, 2, t_expr, thresh); });
    flow::continue_node<flow::continue_msg> k_node(
        g, [&](auto) { k = vsop2013_elliptic_impl(pl_idx, 3, t_expr, thresh); });
    flow::continue_node<flow::continue_msg> h_node(
        g, [&](auto) { h = vsop2013_elliptic_impl(pl_idx, 4, t_expr, thresh); });
    flow::continue_node<flow::continue_msg> q_node(
        g, [&](auto) { q = vsop2013_elliptic_impl(pl_idx, 5, t_expr, thresh); });
    flow::continue_node<flow::continue_msg> p_node(
        g, [&](auto) { p = vsop2013_elliptic_impl(pl_idx, 6, t_expr, thresh); });
    flow::make_edge(start, a_node);
    flow::make_edge(start, lam_node);
    flow::make_edge(start, k_node);
    flow::make_edge(start, h_node);
    flow::make_edge(start, q_node);
    flow::make_edge(start, p_node);

    // M.
    flow::continue_node<flow::continue_msg> M_node(g, [&](auto) { M = lam - atan2(h, k); });
    flow::make_edge(lam_node, M_node);
    flow::make_edge(h_node, M_node);
    flow::make_edge(k_node, M_node);

    // k**2 + h**2.
    flow::continue_node<flow::continue_msg> kh2_node(g, [&](auto) { kh_2 = square(k) + square(h); });
    flow::make_edge(k_node, kh2_node);
    flow::make_edge(h_node, kh2_node);

    // q**2 + p**2.
    flow::continue_node<flow::continue_msg> qp2_node(g, [&](auto) { qp_2 = square(q) + square(p); });
    flow::make_edge(q_node, qp2_node);
    flow::make_edge(p_node, qp2_node);

    // sqrt(q**2 + p**2).
    flow::continue_node<flow::continue_msg> qp_node(g, [&](auto) { qp = sqrt(qp_2); });
    flow::make_edge(qp2_node, qp_node);

    // e.
    flow::continue_node<flow::continue_msg> e_node(g, [&](auto) { e = sqrt(kh_2); });
    flow::make_edge(kh2_node, e_node);

    // E.
    flow::continue_node<flow::continue_msg> E_node(g, [&](auto) { E = kepE(e, M); });
    flow::make_edge(e_node, E_node);
    flow::make_edge(M_node, E_node);

    // sin(E)/cos(E).
    flow::continue_node<flow::continue_msg> sin_E_node(g, [&](auto) { sin_E = sin(E); });
    flow::make_edge(E_node, sin_E_node);
    flow::continue_node<flow::continue_msg> cos_E_node(g, [&](auto) { cos_E = cos(E); });
    flow::make_edge(E_node, cos_E_node);

    // sqrt(1 - e**2).
    flow::continue_node<flow::continue_msg> sqrt_1me2_node(g, [&](auto) { sqrt_1me2 = sqrt(1_dbl - kh_2); });
    flow::make_edge(kh2_node, sqrt_1me2_node);

    // sin(i)/cos(i).
    flow::continue_node<flow::continue_msg> ci_node(g, [&](auto) { ci = 1_dbl - 2_dbl * qp_2; });
    flow::make_edge(qp2_node, ci_node);
    flow::continue_node<flow::continue_msg> si_node(g, [&](auto) { si = sqrt(1_dbl - square(ci)); });
    flow::make_edge(ci_node, si_node);

    // cos(Om)/sin(Om).
    flow::continue_node<flow::continue_msg> cOm_node(g, [&](auto) { cOm = q / qp; });
    flow::make_edge(q_node, cOm_node);
    flow::make_edge(qp_node, cOm_node);
    flow::continue_node<flow::continue_msg> sOm_node(g, [&](auto) { sOm = p / qp; });
    flow::make_edge(p_node, sOm_node);
    flow::make_edge(qp_node, sOm_node);

    // cos(om)/sin(om).
    flow::continue_node<flow::continue_msg> com_node(g, [&](auto) { com = (k * cOm + h * sOm) / e; });
    flow::make_edge(k_node, com_node);
    flow::make_edge(cOm_node, com_node);
    flow::make_edge(h_node, com_node);
    flow::make_edge(sOm_node, com_node);
    flow::make_edge(e_node, com_node);
    flow::continue_node<flow::continue_msg> som_node(g, [&](auto) { som = (h * cOm - k * sOm) / e; });
    flow::make_edge(h_node, som_node);
    flow::make_edge(cOm_node, som_node);
    flow::make_edge(k_node, som_node);
    flow::make_edge(sOm_node, som_node);
    flow::make_edge(e_node, som_node);

    // q1/a.
    flow::continue_node<flow::continue_msg> q1_a_node(g, [&](auto) { q1_a = cos_E - e; });
    flow::make_edge(cos_E_node, q1_a_node);
    flow::make_edge(e_node, q1_a_node);

    // q2/a
    flow::continue_node<flow::continue_msg> q2_a_node(g, [&](auto) { q2_a = sqrt_1me2 * sin_E; });
    flow::make_edge(sqrt_1me2_node, q2_a_node);
    flow::make_edge(sin_E_node, q2_a_node);

    // R00.
    flow::continue_node<flow::continue_msg> R00_node(g, [&](auto) { R00 = cOm * com - sOm * ci * som; });
    flow::make_edge(cOm_node, R00_node);
    flow::make_edge(com_node, R00_node);
    flow::make_edge(sOm_node, R00_node);
    flow::make_edge(ci_node, R00_node);
    flow::make_edge(som_node, R00_node);

    // R01.
    flow::continue_node<flow::continue_msg> R01_node(g, [&](auto) { R01 = cOm * som + sOm * ci * com; });
    flow::make_edge(cOm_node, R01_node);
    flow::make_edge(som_node, R01_node);
    flow::make_edge(sOm_node, R01_node);
    flow::make_edge(ci_node, R01_node);
    flow::make_edge(com_node, R01_node);

    // R10.
    flow::continue_node<flow::continue_msg> R10_node(g, [&](auto) { R10 = sOm * com + cOm * ci * som; });
    flow::make_edge(sOm_node, R10_node);
    flow::make_edge(com_node, R10_node);
    flow::make_edge(cOm_node, R10_node);
    flow::make_edge(ci_node, R10_node);
    flow::make_edge(som_node, R10_node);

    // R11.
    flow::continue_node<flow::continue_msg> R11_node(g, [&](auto) { R11 = sOm * som - cOm * ci * com; });
    flow::make_edge(sOm_node, R11_node);
    flow::make_edge(som_node, R11_node);
    flow::make_edge(cOm_node, R11_node);
    flow::make_edge(ci_node, R11_node);
    flow::make_edge(com_node, R11_node);

    // R20.
    flow::continue_node<flow::continue_msg> R20_node(g, [&](auto) { R20 = si * som; });
    flow::make_edge(si_node, R20_node);
    flow::make_edge(som_node, R20_node);

    // R21.
    flow::continue_node<flow::continue_msg> R21_node(g, [&](auto) { R21 = si * com; });
    flow::make_edge(si_node, R21_node);
    flow::make_edge(com_node, R21_node);

    // v_num.
    flow::continue_node<flow::continue_msg> v_num_node(g, [&](auto) { v_num = sqrt_1me2 * cos_E; });
    flow::make_edge(sqrt_1me2_node, v_num_node);
    flow::make_edge(cos_E_node, v_num_node);

    // v_den.
    flow::continue_node<flow::continue_msg> v_den_node(g, [&](auto) { v_den = sqrt(a) * (1_dbl - e * cos_E); });
    flow::make_edge(a_node, v_den_node);
    flow::make_edge(e_node, v_den_node);
    flow::make_edge(cos_E_node, v_den_node);

    // x.
    flow::continue_node<flow::continue_msg> x_node(g, [&](auto) { retval[0] = a * (q1_a * R00 - q2_a * R01); });
    flow::make_edge(a_node, x_node);
    flow::make_edge(q1_a_node, x_node);
    flow::make_edge(R00_node, x_node);
    flow::make_edge(q2_a_node, x_node);
    flow::make_edge(R01_node, x_node);

    // y.
    flow::continue_node<flow::continue_msg> y_node(g, [&](auto) { retval[1] = a * (q1_a * R10 - q2_a * R11); });
    flow::make_edge(a_node, y_node);
    flow::make_edge(q1_a_node, y_node);
    flow::make_edge(R10_node, y_node);
    flow::make_edge(q2_a_node, y_node);
    flow::make_edge(R11_node, y_node);

    // z.
    flow::continue_node<flow::continue_msg> z_node(g, [&](auto) { retval[2] = a * (q1_a * R20 + q2_a * R21); });
    flow::make_edge(a_node, z_node);
    flow::make_edge(q1_a_node, z_node);
    flow::make_edge(R20_node, z_node);
    flow::make_edge(q2_a_node, z_node);
    flow::make_edge(R21_node, z_node);

    // vx.
    flow::continue_node<flow::continue_msg> vx_node(
        g, [&](auto) { retval[3] = mu * (-sin_E * R00 - v_num * R01) / v_den; });
    flow::make_edge(sin_E_node, vx_node);
    flow::make_edge(R00_node, vx_node);
    flow::make_edge(v_num_node, vx_node);
    flow::make_edge(R01_node, vx_node);
    flow::make_edge(v_den_node, vx_node);

    // vy.
    flow::continue_node<flow::continue_msg> vy_node(
        g, [&](auto) { retval[4] = mu * (-sin_E * R10 - v_num * R11) / v_den; });
    flow::make_edge(sin_E_node, vy_node);
    flow::make_edge(R10_node, vy_node);
    flow::make_edge(v_num_node, vy_node);
    flow::make_edge(R11_node, vy_node);
    flow::make_edge(v_den_node, vy_node);

    // vz.
    flow::continue_node<flow::continue_msg> vz_node(
        g, [&](auto) { retval[5] = mu * (-sin_E * R20 + v_num * R21) / v_den; });
    flow::make_edge(sin_E_node, vz_node);
    flow::make_edge(R20_node, vz_node);
    flow::make_edge(v_num_node, vz_node);
    flow::make_edge(R21_node, vz_node);
    flow::make_edge(v_den_node, vz_node);

    start.try_put(flow::continue_msg{});
    g.wait_for_all();

    return retval;
#else
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
    tbb::parallel_invoke([&]() { M = lam - atan2(h, k); }, [&]() { kh_2 = square(k) + square(h); },
                         [&]() {
                             qp_2 = square(q) + square(p);
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
            si = sqrt(1_dbl - square(ci));
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

    // G*M values for the planets.
    constexpr double gm_pl[] = {4.9125474514508118699e-11, 7.2434524861627027000e-10, 8.9970116036316091182e-10,
                                9.5495351057792580598e-11, 2.8253458420837780000e-07, 8.4597151856806587398e-08,
                                1.2920249167819693900e-08, 1.5243589007842762800e-08, 2.1886997654259696800e-12};

    // G*M value for the Sun.
    constexpr double gm_sun = 2.9591220836841438269e-04;

    // Compute the gravitational parameter for pl_idx.
    assert(pl_idx >= 1u && pl_idx <= 9u); // LCOV_EXCL_LINE
    const auto mu = std::sqrt(gm_sun + gm_pl[pl_idx - 1u]);

    // Prepare the return value.
    std::vector<expression> retval(6u);

    tbb::parallel_invoke([&]() { retval[0] = a * (q1_a * R00 - q2_a * R01); },
                         [&]() { retval[1] = a * (q1_a * R10 - q2_a * R11); },
                         [&]() { retval[2] = a * (q1_a * R20 + q2_a * R21); },
                         [&]() { retval[3] = mu * (-sin_E * R00 - v_num * R01) / v_den; },
                         [&]() { retval[4] = mu * (-sin_E * R10 - v_num * R11) / v_den; },
                         [&]() { retval[5] = mu * (-sin_E * R20 + v_num * R21) / v_den; });

    return retval;
#endif
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

} // namespace heyoka::detail
