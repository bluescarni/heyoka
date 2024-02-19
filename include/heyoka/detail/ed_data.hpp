// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// A RAII helper to extract polys from a cache and
// return them to the cache upon destruction. Used
// during event detection.
template <typename>
class taylor_pwrap;

// Polynomial cache type. Each entry is a polynomial
// represented as a vector of coefficients. Used
// during event detection.
template <typename T>
using taylor_poly_cache = std::vector<std::vector<T>>;

} // namespace detail

template <typename T>
struct taylor_adaptive<T>::ed_data {
    // The working list type used during real root isolation.
    using wlist_t = std::vector<std::tuple<T, T, detail::taylor_pwrap<T>>>;
    // The type used to store the list of isolating intervals.
    using isol_t = std::vector<std::tuple<T, T>>;
    // Polynomial translation function type.
    using pt_t = void (*)(T *, const T *) noexcept;
    // rtscc function type.
    using rtscc_t = void (*)(T *, T *, std::uint32_t *, const T *) noexcept;
    // fex_check function type.
    using fex_check_t = void (*)(const T *, const T *, const std::uint32_t *, std::uint32_t *) noexcept;

    // The vector of terminal events.
    std::vector<t_event_t> m_tes;
    // The vector of non-terminal events.
    std::vector<nt_event_t> m_ntes;
    // The jet of derivatives for the state variables
    // and the events.
    std::vector<T> m_ev_jet;
    // Vector of detected terminal events.
    std::vector<std::tuple<std::uint32_t, T, int, T>> m_d_tes;
    // The vector of cooldowns for the terminal events.
    // If an event is on cooldown, the corresponding optional
    // in this vector will contain the total time elapsed
    // since the cooldown started and the absolute value
    // of the cooldown duration.
    std::vector<std::optional<std::pair<T, T>>> m_te_cooldowns;
    // Vector of detected non-terminal events.
    std::vector<std::tuple<std::uint32_t, T, int>> m_d_ntes;
    // The LLVM state.
    llvm_state m_state;
    // The JIT compiled functions used during root finding.
    // NOTE: use default member initializers to ensure that
    // these are zero-inited by the default constructor
    // (which is defaulted).
    pt_t m_pt = nullptr;
    rtscc_t m_rtscc = nullptr;
    fex_check_t m_fex_check = nullptr;
    // The polynomial cache.
    // NOTE: it is *really* important that this is declared
    // *before* m_wlist, because m_wlist will contain references
    // to and interact with m_poly_cache during destruction,
    // and we must be sure that m_wlist is destroyed *before*
    // m_poly_cache.
    detail::taylor_poly_cache<T> m_poly_cache;
    // The working list.
    wlist_t m_wlist;
    // The list of isolating intervals.
    isol_t m_isol;

    // Constructors.
    ed_data(llvm_state, std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t, const T &);
    ed_data(const ed_data &);
    ~ed_data();

    // Delete unused bits.
    ed_data(ed_data &&) = delete;
    ed_data &operator=(const ed_data &) = delete;
    ed_data &operator=(ed_data &&) = delete;

    // The event detection function.
    void detect_events(const T &, std::uint32_t, std::uint32_t, const T &);

private:
    // Serialisation.
    // NOTE: the def ctor is used only during deserialisation
    // via pointer.
    ed_data();
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template <typename T>
struct taylor_adaptive_batch<T>::ed_data {
    // The working list type used during real root isolation.
    using wlist_t = std::vector<std::tuple<T, T, detail::taylor_pwrap<T>>>;
    // The type used to store the list of isolating intervals.
    using isol_t = std::vector<std::tuple<T, T>>;
    // Polynomial translation function type.
    using pt_t = void (*)(T *, const T *) noexcept;
    // rtscc function type.
    using rtscc_t = void (*)(T *, T *, std::uint32_t *, const T *) noexcept;
    // fex_check function type.
    using fex_check_t = void (*)(const T *, const T *, const std::uint32_t *, std::uint32_t *) noexcept;

    // The vector of terminal events.
    std::vector<t_event_t> m_tes;
    // The vector of non-terminal events.
    std::vector<nt_event_t> m_ntes;
    // The jet of derivatives for the state variables
    // and the events.
    std::vector<T> m_ev_jet;
    // The vector to store the norm infinity of the state
    // vector when using the stepper with events.
    std::vector<T> m_max_abs_state;
    // The vector to store the the maximum absolute error
    // on the Taylor series of the event equations.
    std::vector<T> m_g_eps;
    // Vector of detected terminal events.
    std::vector<std::vector<std::tuple<std::uint32_t, T, int, T>>> m_d_tes;
    // The vector of cooldowns for the terminal events.
    // If an event is on cooldown, the corresponding optional
    // in this vector will contain the total time elapsed
    // since the cooldown started and the absolute value
    // of the cooldown duration.
    std::vector<std::vector<std::optional<std::pair<T, T>>>> m_te_cooldowns;
    // Vector of detected non-terminal events.
    std::vector<std::vector<std::tuple<std::uint32_t, T, int>>> m_d_ntes;
    // The LLVM state.
    llvm_state m_state;
    // Flags to signal if we are integrating backwards in time.
    std::vector<std::uint32_t> m_back_int;
    // Output of the fast exclusion check.
    std::vector<std::uint32_t> m_fex_check_res;
    // The JIT compiled functions used during root finding.
    // NOTE: use default member initializers to ensure that
    // these are zero-inited by the default constructor
    // (which is defaulted).
    pt_t m_pt = nullptr;
    rtscc_t m_rtscc = nullptr;
    fex_check_t m_fex_check = nullptr;
    // The polynomial cache.
    // NOTE: it is *really* important that this is declared
    // *before* m_wlist, because m_wlist will contain references
    // to and interact with m_poly_cache during destruction,
    // and we must be sure that m_wlist is destroyed *before*
    // m_poly_cache.
    detail::taylor_poly_cache<T> m_poly_cache;
    // The working list.
    wlist_t m_wlist;
    // The list of isolating intervals.
    isol_t m_isol;

    // Constructors.
    ed_data(llvm_state, std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t, std::uint32_t);
    ed_data(const ed_data &);
    ~ed_data();

    // Delete unused bits.
    ed_data(ed_data &&) = delete;
    ed_data &operator=(const ed_data &) = delete;
    ed_data &operator=(ed_data &&) = delete;

    // The event detection function.
    void detect_events(const T *, std::uint32_t, std::uint32_t, std::uint32_t);

private:
    // Serialisation.
    // NOTE: the def ctor is used only during deserialisation
    // via pointer.
    ed_data();
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

HEYOKA_END_NAMESPACE

#endif
