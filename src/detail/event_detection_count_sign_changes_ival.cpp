// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// NOTE: we don't need the zero option here as it is always filtered out when we use this enum.
//
// NOLINTNEXTLINE(performance-enum-size)
enum class ed_csc_ival_sign : unsigned {
    undecided = 0,
    plus = 1,
    minus = 2,
};

// Small helper to compute the sign of a nonzero interval coefficient represented as the (mlo, hi) pair.
ed_csc_ival_sign ed_csc_get_ival_sign(const auto &p)
{
    assert(p.first != 0 || p.second != 0);
    return ed_csc_ival_sign{static_cast<unsigned>((p.first < 0) + ((p.second < 0) << 1u))};
}

} // namespace

// Helper to count the number of sign changes in a list of interval coefficients.
//
// The coefficients are stored flattened-out in cfs, as usual with negated lower bounds:
//
// [-c0_lo, c0_hi, -c1_lo, c1_hi, -c2_lo, c2_hi, ...]
//
// Because the coefficients are intervals, we have 4 options for their sign:
//
// - '+' if mlo < 0,
// - '-' if hi < 0,
// - '0' if the interval is exactly [0, 0],
// - '?' otherwise.
//
// A '?' sign means that the sign of the coefficient could be any of ['+', '-', '0']. As a consequence, the return value
// of this function is also an interval representing the minimum/maximum number of sign changes.
//
// cfs is assumed to be non-empty, to have an even size and to represent valid intervals (in particular, it must not
// contain NaN values).
template <typename T>
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> ed_count_sign_changes_ival(const std::span<const T> cfs)
{
    assert(!cfs.empty());
    assert(cfs.size() % 2u == 0u);
    assert(std::ranges::none_of(cfs, [](const auto &x) {
        using std::isnan;
        return isnan(x);
    }));

    // Fetch the poly degree + 1, making sure that it is representable as a 32-bit int. This will ensure that the
    // computation of the sign changes cannot overflow.
    const auto op1 = boost::numeric_cast<std::uint32_t>(cfs.size() / 2u);

    // NOTE: in order to compute the return value correctly, we have to be careful dealing with undecided '?' signs.
    //
    // First of all, the computation of the sign changes ignores zero coefficients, which will be filtered out below.
    //
    // Within a group of one or more consecutive undecided signs '???...', the maximum number of sign changes occurs
    // with alternating signs: either '+-+-...' or '-+-+...'. Between the two possibilities, choosing the one that
    // begins with a sign which is the opposite of the sign *preceding* the group maximises the number of sign changes.
    // If nothing precedes the group, the sign alternation must be chosen so that the last sign in the group is the
    // opposite of the first sign *following* the group. If the entire list of nonzero coefficients is undecided, either
    // choice produces the maximum number of sign changes.
    //
    // The minimum number of sign changes occurs when all undecided signs in a group are regarded as zero: no sign
    // changes occur within the group by definition, nor does any sign change occur when crossing the group boundaries.
    // At most, a single sign change can occur between the two decided signs enclosing the group. This sign change can
    // never be eliminated with any choice for the undecided group.

    // Create a view over the coefficients as intervals, skipping over the zero ones.
    auto ival_view = std::views::iota(static_cast<std::uint32_t>(0), op1)
                     | std::views::transform([&cfs](const auto idx) {
                           return std::make_pair(std::cref(cfs[idx * 2u]), std::cref(cfs[(idx * 2u) + 1u]));
                       })
                     | std::views::filter([](const auto &p) { return p.first != 0 || p.second != 0; });

    // Init the counters for the min/max number of sign changes.
    std::uint32_t min_n_sc = 0, max_n_sc = 0;

    // The last decided sign found (if any).
    std::optional<ed_csc_ival_sign> last_decided_sign;

    // The size of the current group of undecided signs.
    std::uint32_t undecided_group_size = 0;

    // Iterate over the coefficients.
    const auto ival_view_end = std::ranges::end(ival_view);
    for (auto it = std::ranges::begin(ival_view); it != ival_view_end;) {
        // Compute the sign of the current coefficient.
        const auto cur_sign = ed_csc_get_ival_sign(*it);

        // Move immediately to the next coefficient.
        ++it;

        if (cur_sign == ed_csc_ival_sign::undecided) {
            // We are within an undecided group, update the group size.
            ++undecided_group_size;

            if (it == ival_view_end) {
                // The undecided group ends the list of coefficients.

                // For max_n_sc, we have (undecided_group_size - 1) sign changes from the group itself, plus 1 from
                // choosing the first sign in the group as the opposite of the sign preceding the group (if it exists).
                max_n_sc += undecided_group_size - 1u + static_cast<bool>(last_decided_sign);

                // For min_n_sc, we don't have additional sign changes.
                break;
            }

            // Nothing else to do, we can move to the next coefficient.
            continue;
        }

        if (last_decided_sign) {
            // We have two coefficients with decided signs bracketing a group of undecided signs, possibly of size 0 (in
            // which case, we are dealing with consecutive decided signs).
            //
            // In the consecutive decided signs case, we have exactly 0 or 1 sign changes (depending on the two signs).
            //
            // In the nonzero undecided group size case, for max_n_sc we have (undecided_group_size - 1) sign changes
            // from the group itself, one change from the left bracket and one change on the right bracket if
            // undecided_group_size is odd and the two decided signs are equal, or if undecided_group_size is even and
            // the two decided signs differ. min_n_sc depends only on the two decided signs.

            // Check if the two decided signs differ.
            const auto sign_changed = (cur_sign != *last_decided_sign);

            // Update max_n_sc/min_n_sc.
            //
            // NOTE: for undecided_group_size == 0 this formula collapses to max_n_sc += sign_changed.
            max_n_sc += undecided_group_size + (sign_changed != (undecided_group_size % 2u));
            min_n_sc += sign_changed;
        } else {
            // We have zero or more undecided signs at the beginning of the nonzero coefficients list before the first
            // decided sign. If the group is not empty, for max_n_sc we have (undecided_group_size - 1) sign changes
            // from the group itself plus the change on the right bracket, otherwise there is nothing to add. In both
            // cases, max_n_sc increases by undecided_group_size. For min_n_sc, we have nothing to add.
            max_n_sc += undecided_group_size;
        }

        // Update last_decided_sign.
        last_decided_sign = cur_sign;

        // Reset undecided_group_size.
        undecided_group_size = 0;
    }

    return {min_n_sc, max_n_sc};
}

// Explicit instantiations.

#define HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL(T)                                                                      \
    template HEYOKA_DLL_PUBLIC std::pair<std::uint32_t, std::uint32_t> ed_count_sign_changes_ival(std::span<const T>)

HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL(float);
HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL(double);
HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL(mppp::real);

#endif

#undef HEYOKA_INST_ED_COUNT_SIGN_CHANGES_IVAL

} // namespace detail

HEYOKA_END_NAMESPACE
