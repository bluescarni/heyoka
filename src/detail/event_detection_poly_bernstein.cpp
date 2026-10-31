// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to convert an input polynomial from the monomial basis into the interval Bernstein form required by event
// detection.
//
// cfs contains the scalar coefficients in the monomial basis of an input polynomial p(t) of degree n, stored in
// ascending order (that is, n = cfs.size() - 1). This function will compute intervals enclosing the Bernstein
// coefficients b_k (k = 0, ..., n) of the rescaled polynomial q(x) = p(h*x), defined by
//
// q(x) = sum_k b_k * C(n,k) * x**k * (1-x)**(n-k),
//
// where C(n,k) is the binomial coefficient. The rescaling maps the [0, h] interval (or [h, 0], if h < 0) of the
// original independent variable t into the [0, 1] interval of the new independent variable x.
//
// The first step of the conversion is the computation of interval enclosures for the coefficients of q(x) via
// llvm_ed_ival_poly_rescale().
//
// The second step reverses q(x), shifts the result by 1 via llvm_ed_ival_poly_shift_1(), and reverses the result of
// the shift. This produces a polynomial whose coefficients are C(n,k)*b_k. Indeed, with the substitution x = 1/(1+y)
// we have x**k * (1-x)**(n-k) = y**(n-k) / (1+y)**n, and thus (1+y)**n * q(1/(1+y)) = sum_k C(n,k) * b_k * y**(n-k).
// The left-hand side is the reversal of q(x) shifted by 1, and the final reversal brings the coefficients C(n,k)*b_k
// into ascending order of k.
//
// The last step recovers the b_k by dividing the coefficients by C(n,k). Because C(n,k) may not be representable, the
// division is by the enclosure of C(n,k) computed by ed_make_ival_binomial_cf_like(). For k = 0 and k = n, C(n,k) = 1
// and no division is needed.
//
// Compared to the direct formula b_k = sum_{i<=k} C(k,i)/C(n,i) * c_i (where c_i are the coefficients of q(x)), which
// requires O(n**2) interval multiplications, the O(n**2) part of this approach consists only of interval additions
// (in llvm_ed_ival_poly_shift_1()).
//
// The return value is a pair of lists of interval polynomial coefficients, stored in the (-lo, hi) format (that is, the
// lower bound of the interval is carried around negated), in ascending order:
//
// [(-c0_lo, c0_hi), (-c1_lo, c1_hi), (-c2_lo, c2_hi), ...]
//
// The first element of the pair contains the coefficients of the interval Bernstein form of the original polynomial.
// The second element of the pair contains the coefficients of q(x), that is, the original polynomial after the
// rescaling step.
//
// The calculations will be performed using the arithmetic primitives from the input guard ivg. The IR is emitted at the
// current insertion point, which must be within the function guarded by ivg.
//
// cfs is assumed non-empty. h and all elements of cfs are assumed to have the same type as ivg.get_val_t(). All input
// values are assumed to be finite. The output intervals enclose the exact coefficients, and they have no NaN bounds.
// Infinite bounds may be returned in case of overflow, but never in the invalid forms [-inf, -inf] and [+inf, +inf].
std::pair<std::vector<llvm::Value *>, std::vector<llvm::Value *>>
llvm_ed_ival_poly_bernstein(const llvm_ival_guard &ivg, const std::vector<llvm::Value *> &cfs, llvm::Value *const h)
{
    assert(!cfs.empty());
    assert(h != nullptr);
    assert(h->getType() == ivg.get_val_t());
    assert(std::ranges::all_of(cfs, [h](auto *const x) { return x != nullptr && x->getType() == h->getType(); }));

    auto &s = ivg.get_state();
    auto *const val_t = ivg.get_val_t();

    // Apply the rescaling.
    auto ret = llvm_ed_ival_poly_rescale(ivg, cfs, h);

    // Store the rescaled polynomial (this will be the second element of the return value).
    auto ret_rescaled = ret;

    // Reverse.
    std::ranges::reverse(ret);

    // Shift by 1.
    ret = llvm_ed_ival_poly_shift_1(ivg, ret);

    // Reverse again.
    std::ranges::reverse(ret);

    // Final division.
    const auto op1 = cfs.size();
    const auto n = boost::numeric_cast<std::uint32_t>(op1 - 1u);

    // Create a number whose internal value is of the C++ type corresponding to the scalar type of val_t. We will use
    // this to create binomial coefficient intervals of the correct target type.
    const auto x = number_like(s, val_t->getScalarType(), 0.);

    // NOTE: we can skip the first and the last coefficients since for k = 0 and k = n, C(n,k) = 1.
    for (decltype(cfs.size()) k = 1; k < op1 - 1u; ++k) {
        // Compute the binomial coefficient interval, checking if it is exact.
        //
        // NOTE: C_lo and C_hi are finite and >= 1, which satisfies the preconditions of the division primitives.
        const auto [C_lo, C_hi, exact] = std::visit(
            [n, k](const auto &v) {
                const auto [lo, hi] = ed_make_ival_binomial_cf_like(n, boost::numeric_cast<std::uint32_t>(k), v);
                return std::tuple{number{lo}, number{hi}, lo == hi};
            },
            x.value());

        // Generate the lower bound of C.
        auto *const C_lo_val = llvm_codegen(s, val_t, C_lo);

        // Fetch the current interval coefficient.
        auto &cur_b = ret[k];

        if (exact) {
            // Division by the positive point C_lo.
            cur_b = ivg.ival_div_pos_val(cur_b, C_lo_val);
        } else {
            // Generate the upper bound of C.
            auto *const C_hi_val = llvm_codegen(s, val_t, C_hi);

            // Build the C interval.
            auto *const C_ival = ivg.ival_construct(ivg.val_neg(C_lo_val), C_hi_val);

            // Division by the positive interval C_ival.
            cur_b = ivg.ival_div_pos_ival(cur_b, C_ival);
        }
    }

    return std::make_pair(std::move(ret), std::move(ret_rescaled));
}

} // namespace detail

HEYOKA_END_NAMESPACE
