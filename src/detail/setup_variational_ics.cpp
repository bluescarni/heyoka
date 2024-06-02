// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/var_ode_sys.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
void setup_variational_ics_varpar(std::vector<T> &state, const var_ode_sys &vsys, std::uint32_t batch_size)
{
    using safe_state_size_t = boost::safe_numerics::safe<decltype(state.size())>;

    const auto &sys = vsys.get_sys();

    // LCOV_EXCL_START
    assert(!state.empty());
    assert(batch_size > 0u);
    assert(state.size() % batch_size == 0u);
    assert(state.size() / batch_size == vsys.get_n_orig_sv());
    assert(sys.size() > state.size() / batch_size);
    // LCOV_EXCL_STOP

    // Prepare the state vector.
    // NOTE: the appended items will all be zeroes.
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::same_as<T, mppp::real>) {
        // NOTE: at this point we know that all the elements in state
        // are set up properly with the correct precision.
        state.resize(safe_state_size_t(batch_size) * sys.size(), mppp::real{0, state[0].get_prec()});
    } else {
#endif
        state.resize(safe_state_size_t(batch_size) * sys.size());
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
        assert(std::holds_alternative<variable>(cur_sv.value())); // LCOV_EXCL_LINE

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

template <typename T>
void setup_variational_ics_t0(const llvm_state &s, std::vector<T> &state, const std::vector<T> &pars, const T *time,
                              const var_ode_sys &vsys, std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    using state_size_t = decltype(state.size());

    const auto &sys = vsys.get_sys();

    // LCOV_EXCL_START
    assert(!state.empty());
    assert(batch_size > 0u);
    assert(state.size() % batch_size == 0u);
    assert(state.size() / batch_size == sys.size());
    // LCOV_EXCL_STOP

    // Need to do anything only if derivatives wrt the initial time are requested.
    const auto time_arg_it = std::ranges::find(vsys.get_vargs(), heyoka::time);
    if (time_arg_it == vsys.get_vargs().end()) {
        return;
    }

    // Only first-order derivatives are supported at this time.
    if (vsys.get_order() > 1u) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "In a variational integrator the automatic setup of the initial conditions for the derivatives with "
            "respect to the initial time are currently supported only at the first order, but an order of {} was "
            "specified instead",
            vsys.get_order()));
    }

    // Determine the index of the time argument into vargs().
    // NOTE: the cast is safe because we know that state is large enough
    // to store all first-order derivatives.
    const auto time_arg_idx = static_cast<state_size_t>(std::ranges::distance(vsys.get_vargs().begin(), time_arg_it));

    // Fetch the number of original state variables.
    const auto n_orig_sv = vsys.get_n_orig_sv();

    // We need to evaluate the negative of the original rhs. Prepare the expressions for evaluation.
    std::vector<expression> v_ex, orig_sv;
    v_ex.reserve(n_orig_sv);
    orig_sv.reserve(n_orig_sv);
    for (decltype(vsys.get_n_orig_sv()) i = 0; i < n_orig_sv; ++i) {
        orig_sv.push_back(sys[i].first);
        v_ex.push_back(-sys[i].second);
    }

    // Make an llvm state similar to s.
    auto st = s.make_similar();

    // Setup the prec argument for the compiled function.
    const auto prec = [&]() -> long long {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::same_as<mppp::real, T>) {
            // NOTE: here we know that m_state is not empty and that it contains
            // values with the correct precision.
            return boost::numeric_cast<long long>(state[0].get_prec());
        } else {
#endif
            return 0;
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    }();

    // Add the compiled function to st.
    add_cfunc<T>(st, "f", v_ex, orig_sv, batch_size, high_accuracy, compact_mode, false, prec, false);

    // Compile.
    st.compile();

    // Fetch it.
    using cfunc_ptr_t = void (*)(T *, const T *, const T *, const T *) noexcept;
    auto *fptr = reinterpret_cast<cfunc_ptr_t>(st.jit_lookup("f"));

    // Widen batch_size in order to avoid overflows when indexing into
    // std::vector<T>.
    const auto bs = static_cast<state_size_t>(batch_size);

    // Prepare the output vector. Copy the values from state so that we know that
    // the precision is already set up correctly for mppp::real.
    std::vector<T> out(state.data(), state.data() + bs * n_orig_sv);

    // Evaluate.
    fptr(out.data(), state.data(), pars.data(), time);

    // Write the result into state.
    for (decltype(vsys.get_n_orig_sv()) i = 0; i < n_orig_sv; ++i) {
        // Compute the index for writing into state.
        const auto state_idx =
            // Initial offset for the original state variables.
            bs * n_orig_sv +
            // Offset due to the state variable index.
            bs * i * vsys.get_vargs().size() +
            // Offset due to time_arg_idx.
            bs * time_arg_idx;

        // Compute the offset for reading from out.
        const auto out_idx = bs * i;

        for (std::uint32_t j = 0; j < batch_size; ++j) {
            state[state_idx + j] = out[out_idx + j];
        }
    }
}

// Explicit instantiations.
#define HEYOKA_INST_SETUP_VARIATIONAL_ICS(F)                                                                           \
    template void setup_variational_ics_varpar(std::vector<F> &, const var_ode_sys &, std::uint32_t);                  \
    template void setup_variational_ics_t0(const llvm_state &, std::vector<F> &, const std::vector<F> &, const F *,    \
                                           const var_ode_sys &, std::uint32_t, bool, bool);

HEYOKA_INST_SETUP_VARIATIONAL_ICS(float)
HEYOKA_INST_SETUP_VARIATIONAL_ICS(double)
HEYOKA_INST_SETUP_VARIATIONAL_ICS(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_INST_SETUP_VARIATIONAL_ICS(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_INST_SETUP_VARIATIONAL_ICS(mppp::real)

#endif

#undef HEYOKA_INST_SETUP_VARIATIONAL_ICS

} // namespace detail

HEYOKA_END_NAMESPACE
