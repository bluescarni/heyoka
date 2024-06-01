// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/taylor.hpp>
#include <heyoka/var_ode_sys.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
void setup_variational_ics_varpar(std::vector<T> &state, const var_ode_sys &vsys, std::uint32_t batch_size)
{
    const auto &sys = vsys.get_sys();

    assert(!state.empty());
    assert(batch_size > 0u);
    assert(state.size() % batch_size == 0u);
    assert(state.size() / batch_size == vsys.get_n_orig_sv());
    assert(sys.size() > state.size() / batch_size);

    // Prepare the state vector.
    // NOTE: the appended items will all be zeroes.
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::same_as<T, mppp::real>) {
        // NOTE: at this point we know that all the elements in state
        // are set up properly with the correct precision.
        state.resize(boost::safe_numerics::safe<std::uint32_t>(batch_size) * sys.size(),
                     mppp::real{0, state[0].get_prec()});
    } else {
#endif
        state.resize(boost::safe_numerics::safe<std::uint32_t>(batch_size) * sys.size());
#if defined(HEYOKA_HAVE_REAL)
    }
#endif

    // Fetch the range of multiindices.
    const auto didx_rng = vsys.get_didx_range();
    using didx_rng_diff_t = std::ranges::range_difference_t<decltype(vsys.get_didx_range())>;

    // Fetch the list of variational arguments.
    const auto &vargs = vsys.get_vargs();

    for (decltype(sys.size()) i = vsys.get_n_orig_sv(); i < sys.size(); ++i) {
        // Fetch the current multiindex.
        const auto &cur_didx = *(didx_rng.begin() + boost::numeric_cast<didx_rng_diff_t>(i));

        // We need to do something only for first-order derivatives, otherwise
        // we leave the zero in the state vector.
        if (cur_didx.second.size() != 1u) {
            continue;
        }

        // Fetch the diff index and order from the only element in cur_didx.second.
        const auto [didx, order] = cur_didx.second[0];

        // Fetch the state variable the current multiindex refers to.
        const auto &cur_sv = sys[cur_didx.first].first;
        assert(std::holds_alternative<variable>(cur_sv.value()));

        // If the diff order is greater than 1 or the derivative is not with respect
        // to the current state variable, move on.
        if (order > 1u || cur_sv != vargs[didx]) {
            continue;
        }

        // Establish the index for writing into state.
        const auto state_idx = static_cast<decltype(state.size())>(i) * batch_size;

        for (std::uint32_t j = 0; j < batch_size; ++j) {
#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::same_as<T, mppp::real>) {
                state[state_idx + j].set(1);
            } else {
#endif
                state[state_idx + j] = 1;
#if defined(HEYOKA_HAVE_REAL)
            }
#endif
        }
    }
}

// Explicit instantiations.
template void setup_variational_ics_varpar(std::vector<float> &, const var_ode_sys &, std::uint32_t);
template void setup_variational_ics_varpar(std::vector<double> &, const var_ode_sys &, std::uint32_t);
template void setup_variational_ics_varpar(std::vector<long double> &, const var_ode_sys &, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

template void setup_variational_ics_varpar(std::vector<mppp::real128> &, const var_ode_sys &, std::uint32_t);

#endif

#if defined(HEYOKA_HAVE_REAL)

template void setup_variational_ics_varpar(std::vector<mppp::real> &, const var_ode_sys &, std::uint32_t);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
