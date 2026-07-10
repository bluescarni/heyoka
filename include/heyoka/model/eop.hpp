// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_EOP_HPP
#define HEYOKA_MODEL_EOP_HPP

#include <cstdint>
#include <tuple>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_impl.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>

// NOTE: for the representation of EOP data, we adopt a piecewise linear approximation where the switch points are given
// by the dates in the eop dataset. Within each time interval, an EOP quantity is approximated as EOP(t) = c0 + c1*t
// (where the values of the c0 and c1 constants change from interval to interval). In the expression system, we
// implement, for each EOP quantity, two unary functions which return respectively the EOP quantity and its first-order
// derivative at the given input time.
//
// NOTE: for the computation of angles such as the ERA and gmst82 (which do not show up directly in the EOP data, but
// rather are derived from it), we use a different approach wrt astropy/ERFA. Specifically, we precompute values at the
// dates in the eop dataset and then we interpolate directly between these precomputed values. Both the precomputation
// and the interpolation are performed in extended precision, and the result is cast back to the original precision only
// at the very end. By contrast, the canonical approach in astropy/ERFA is to first interpolate the ut1-utc difference,
// and then use the result of the interpolation to compute the desired value. This difference in approach is most likely
// at the root of the slight discrepancies we see wrt astropy/ERFA, which amount to < 1 part over 1 billion (circa
// millimetre level in LEO).
//
// NOTE: the linear interpolation approach should be adequate for astrodynamical applications. In principle, however,
// for both the polar motion and the UT1-UTC difference, there are short-term (diurnal and semi-diurnal) variations
// due to ocean tides and high-frequency nutation terms that should be accounted for. For this purpose, IERS provides
// a routine (interp.f) that accounts for these short-term variations while at the same time interpolating at higher
// orders. In principle, we could think in the future about implementing this code in the expression system. For more
// information see:
//
// https://hpiers.obspm.fr/iers/bul/bulb/explanatory.html

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *llvm_get_era_erap_func(llvm_state &, llvm::Type *, std::uint32_t,
                                                                       const eop_data &);
[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *llvm_get_gmst82_gmst82p_func(llvm_state &, llvm::Type *, std::uint32_t,
                                                                             const eop_data &);

template <typename... KwArgs>
auto eop_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    // EOP data (defaults to def-cted).
    auto data = [&p]() {
        if constexpr (p.has(kw::eop_data)) {
            return p(kw::eop_data);
        } else {
            return eop_data{};
        }
    }();

    return std::tuple{std::move(time_expr), std::move(data)};
}

} // namespace detail

inline constexpr auto eop_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::time_expr>,
                                                kw::descr::same_as<kw::eop_data, eop_data>>{};

} // namespace model

HEYOKA_END_NAMESPACE

// NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
HEYOKA_MODEL_DECLARE_EOP_SW(era, eop_data, eop_kw_cfg, eop_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(gmst82, eop_data, eop_kw_cfg, eop_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(pm_x, eop_data, eop_kw_cfg, eop_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(pm_y, eop_data, eop_kw_cfg, eop_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(dX, eop_data, eop_kw_cfg, eop_common_opts);
HEYOKA_MODEL_DECLARE_EOP_SW(dY, eop_data, eop_kw_cfg, eop_common_opts);
// NOLINTEND(cppcoreguidelines-missing-std-forward)

#endif
