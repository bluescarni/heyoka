// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/version.hpp>

// NOTE: the header for hash_combine changed in version 1.67.
#if (BOOST_VERSION / 100000 > 1) || (BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 >= 67)

#include <boost/container_hash/hash.hpp>

#else

#include <boost/functional/hash.hpp>

#endif

#include <fmt/format.h>

#include <heyoka/celmec/vsop2013.hpp>
#include <heyoka/detail/vsop2013/vsop2013_1.hpp>
#include <heyoka/detail/vsop2013/vsop2013_2.hpp>
#include <heyoka/detail/vsop2013/vsop2013_3.hpp>
#include <heyoka/detail/vsop2013/vsop2013_4.hpp>
#include <heyoka/detail/vsop2013/vsop2013_5.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>

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

#undef HEYOKA_VSOP2013_RECORD_DATA

    return retval;
}

// The lambda_l values (constant + linear term).
const std::array<std::array<double, 2>, 17> lam_l_data = {{{4.402608631669, 26087.90314068555},
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

} // namespace

// Implementation of the function constructing the VSOP2013 series as heyoka expressions.
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

    // Fetch the data.
    static const auto data = build_vsop2103_data();

    // Locate the data entry for the current planet and variable.
    const auto data_it = data.find({pl_idx, var_idx});
    assert(data_it != data.end());
    const auto [n_alpha, sizes_ptr, val_ptr] = data_it->second;

    // This vector will contain the chunks of the series
    // for different values of alpha.
    std::vector<expression> parts;
    for (std::size_t alpha = 0; alpha < n_alpha; ++alpha) {
        // This vector will contain the terms of the chunk
        // for the current value of alpha.
        std::vector<expression> cur;

        // Fetch the number of terms for this chunk.
        const auto cur_size = sizes_ptr[alpha];

        for (std::size_t i = 0; i < cur_size; ++i) {
            // Load the C/S values from the table.
            const auto Sval = val_ptr[alpha][i * 19u + 17u];
            const auto Cval = val_ptr[alpha][i * 19u + 18u];

            // Check if we have reached a term which is too small.
            if (std::sqrt(Cval * Cval + Sval * Sval) < thresh) {
                break;
            }

            // tmp will contain the components of the
            // sin/cos trigonometric argument.
            std::vector<expression> tmp;

            for (std::size_t j = 0; j < 17u; ++j) {
                // Compute lambda_l for the current element
                // of the trigonometric argument.
                auto cur_lam = lam_l_data[j][0] + t_expr * lam_l_data[j][1];

                // Multiply it by the current value in the table.
                tmp.push_back(std::move(cur_lam) * val_ptr[alpha][i * 19u + j]);
            }

            // Compute the trig arg.
            const auto trig_arg = pairwise_sum(std::move(tmp));

            // Add the term to the chunk.
            cur.push_back(Sval * sin(trig_arg) + Cval * cos(trig_arg));
        }

        // Sum the terms in the chunk and multiply them by t**alpha.
        parts.push_back(powi(t_expr, boost::numeric_cast<std::uint32_t>(alpha)) * pairwise_sum(std::move(cur)));
    }

    // Sum the chunks and return them.
    return pairwise_sum(std::move(parts));
}

} // namespace heyoka::detail
