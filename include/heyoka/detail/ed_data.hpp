// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ED_DATA_HPP
#define HEYOKA_DETAIL_ED_DATA_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/container/vector.hpp>

#include <heyoka/config.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// The JIT bits for ed_data. Used both in scalar and batch variants.
template <typename T>
struct ed_jit_data {
    // fex_check function type.
    using fex_check_t = void (*)(const T *, const T *, std::uint32_t *, T *) noexcept;

    // The LLVM state.
    llvm_state m_state;
    // The JIT-compiled fast exclusion check used during event detection.
    fex_check_t m_fex_check = nullptr;

    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

// Enum to describe the sign of the Taylor polynomial of an event function at the end of an integration step.
//
// This is used in the implementation of cross step event detection.
enum class event_sign_end_step : std::int8_t {
    zero = 0,
    positive = 1,
    negative = -1,
    unknown = 2,
};

template <typename T>
struct ed_data {
    // Recover the event typedefs.
    using t_event_t = taylor_adaptive<T>::t_event_t;
    using nt_event_t = taylor_adaptive<T>::nt_event_t;

    // The vector of terminal events.
    std::vector<t_event_t> m_tes;
    // The vector of non-terminal events.
    std::vector<nt_event_t> m_ntes;
    // The jet of derivatives for the state variables and the events.
    std::vector<T> m_ev_jet;
    // Vector of detected terminal events.
    std::vector<std::tuple<std::uint32_t, T, int, T>> m_d_tes;
    // The vector of cooldowns for the terminal events.
    //
    // If an event is on cooldown, the corresponding optional in this vector will contain the total time elapsed since
    // the cooldown started and the absolute value of the cooldown duration.
    std::vector<std::optional<std::pair<T, T>>> m_te_cooldowns;
    // Vector of detected non-terminal events.
    std::vector<std::tuple<std::uint32_t, T, int>> m_d_ntes;
    // The JIT data.
    std::shared_ptr<ed_jit_data<T>> m_jit_data;
    // The workspace for poly root finding.
    frpr_workspace<T> m_frpr_ws;
    // The output for poly root finding.
    std::vector<real_poly_root<T>> m_poly_roots;
    // Vector to store the results of the fast exclusion check for all events (terminal and non-terminal). This is
    // guaranteed to be filled in after a successful invocation of detect_events(), except for a zero timestep (in which
    // case, nothing will be written to m_ev_fex_check_res).
    //
    // NOTE: use Boost's vector in order to avoid the bool specialisation trap.
    boost::container::vector<bool> m_ev_fex_check_res;
    // Vector to track the sign of all event functions (terminal and non-terminal) at the end of a step.
    std::vector<event_sign_end_step> m_ev_signs_end_step;

    // Constructors.
    ed_data(const llvm_state &, std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t,
            const T &);
    ed_data(const ed_data &);
    ~ed_data();

    // Delete unused bits.
    ed_data(ed_data &&) = delete;
    ed_data &operator=(const ed_data &) = delete;
    ed_data &operator=(ed_data &&) = delete;

    // The event detection function.
    void detect_events(const T &, std::uint32_t, std::uint32_t);

private:
    // Serialisation.
    //
    // NOTE: the def ctor is used only during deserialisation via pointer.
    ed_data();
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template <typename T>
struct ed_data_batch {
    // Recover the event typedefs.
    using t_event_t = taylor_adaptive_batch<T>::t_event_t;
    using nt_event_t = taylor_adaptive_batch<T>::nt_event_t;

    // The vector of terminal events.
    std::vector<t_event_t> m_tes;
    // The vector of non-terminal events.
    std::vector<nt_event_t> m_ntes;
    // The jet of derivatives for the state variables and the events.
    std::vector<T> m_ev_jet;
    // Vector of boolean flags signalling, for each batch lane, whether all input data for event detection is finite or
    // not.
    //
    // NOTE: we use int instead of bool in order to avoid the usual std::vector<bool> pitfalls.
    std::vector<int> m_ed_finite;
    // The vector to store the norm infinity of the state vector when using the stepper with events.
    std::vector<T> m_max_abs_state;
    // The vector to store the maximum absolute error on the Taylor series of the event equations.
    std::vector<T> m_g_eps;
    // Vector of detected terminal events.
    std::vector<std::vector<std::tuple<std::uint32_t, T, int, T>>> m_d_tes;
    // The vector of cooldowns for the terminal events.
    //
    // If an event is on cooldown, the corresponding optional in this vector will contain the total time elapsed since
    // the cooldown started and the absolute value of the cooldown duration.
    std::vector<std::vector<std::optional<std::pair<T, T>>>> m_te_cooldowns;
    // Vector of detected non-terminal events.
    std::vector<std::vector<std::tuple<std::uint32_t, T, int>>> m_d_ntes;
    // The JIT data.
    std::shared_ptr<ed_jit_data<T>> m_jit_data;
    // Outputs of the fast exclusion check for a single event.
    std::vector<std::uint32_t> m_fex_check_res;
    std::vector<T> m_fex_check_enc;
    // Vector in which exceptions raised by the event callbacks will be temporarily stored.
    std::vector<std::pair<std::uint32_t, std::exception_ptr>> m_cb_eptrs;
    // Vector to store the polynomial on which real root detection operates.
    //
    // While the fast exclusion check is run on the batched polynomials, real root finding operates on scalar
    // polynomials.
    std::vector<T> m_scal_poly;
    // The workspace for poly root finding.
    frpr_workspace<T> m_frpr_ws;
    // The output for poly root finding.
    std::vector<real_poly_root<T>> m_poly_roots;
    // Vector to store the results of the fast exclusion check for all events (terminal and non-terminal). This is
    // guaranteed to be filled in after a successful invocation of detect_events(), except for batch lanes in which the
    // timestep h is 0 or which contain non-finite data. Such lanes will be untouched.
    //
    // NOTE: use Boost's vector in order to avoid the bool specialisation trap.
    boost::container::vector<bool> m_ev_fex_check_res;
    // Vector to track the sign of all event functions (terminal and non-terminal) at the end of a step.
    std::vector<event_sign_end_step> m_ev_signs_end_step;

    // Constructors.
    ed_data_batch(const llvm_state &, std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t,
                  std::uint32_t);
    ed_data_batch(const ed_data_batch &);
    ~ed_data_batch();

    // Delete unused bits.
    ed_data_batch(ed_data_batch &&) = delete;
    ed_data_batch &operator=(const ed_data_batch &) = delete;
    ed_data_batch &operator=(ed_data_batch &&) = delete;

    // The event detection function.
    void detect_events(const T *, std::uint32_t, std::uint32_t, std::uint32_t);

private:
    // Serialisation.
    //
    // NOTE: the def ctor is used only during deserialisation via pointer.
    ed_data_batch();
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
