// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_POLY_ROOT_FINDING_FRPR_WORKSPACE_IMPL_HPP
#define HEYOKA_DETAIL_POLY_ROOT_FINDING_FRPR_WORKSPACE_IMPL_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <heyoka/config.hpp>

#include <concepts>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/poly_root_finding/vector_cache.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Internal struct encapsulating the JIT data for frpr_workspace_impl.
//
// By keeping it separate, we can include it in frpr_workspace_impl wrapped in a shared_ptr, so that copying a
// frpr_workspace does not deep-copy the JIT state.
template <typename T>
struct frpr_workspace_jit_data {
    using bernstein_func_t = void (*)(T *, T *, const T *, const T *) noexcept;
    using decasteljau_func_t = void (*)(T *, T *, const T *) noexcept;
#if defined(HEYOKA_HAVE_REAL)
    // NOTE: these are the extended-precision analogs of the above functions, used in the extended-precision pass of the
    // real root isolation (see isolate_real_roots()). They work at 3 times the number of binary digits of T (3 times
    // the workspace precision for mppp::real).
    using bernstein_func_3n_t = void (*)(mppp::real *, mppp::real *, const mppp::real *, const mppp::real *) noexcept;
    using decasteljau_func_3n_t = void (*)(mppp::real *, mppp::real *, const mppp::real *) noexcept;
#endif

    // The LLVM state.
    llvm_state m_state;
    // The JIT compiled functions used during root finding.
    bernstein_func_t m_bernstein_func = nullptr;
    decasteljau_func_t m_decasteljau_func = nullptr;
#if defined(HEYOKA_HAVE_REAL)
    bernstein_func_3n_t m_bernstein_func_3n = nullptr;
    decasteljau_func_3n_t m_decasteljau_func_3n = nullptr;
#endif

    // NOTE: needed only for s11n.
    frpr_workspace_jit_data();

    // Main ctor.
    explicit frpr_workspace_jit_data(llvm_state);
    frpr_workspace_jit_data(const frpr_workspace_jit_data &) = delete;
    frpr_workspace_jit_data(frpr_workspace_jit_data &&) noexcept = delete;
    frpr_workspace_jit_data &operator=(const frpr_workspace_jit_data &) = delete;
    frpr_workspace_jit_data &operator=(frpr_workspace_jit_data &&) noexcept = delete;
    ~frpr_workspace_jit_data();

    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template <typename T>
struct frpr_workspace_impl {
    // The working precision, used only if T is mppp::real.
    long long m_prec = 0;
    // The polynomial degree.
    std::uint32_t m_degree = 0;
    // The JIT data.
    std::shared_ptr<frpr_workspace_jit_data<T>> m_jit_data;
    // Temporary buffer to store the intervals identified by isolate_real_roots().
    std::vector<irr_interval<T>> m_intervals;
    // Vector cache.
    //
    // NOTE: it is *very* important that this is created before, and destroyed after, any data member containing
    // cached_vectors from this cache (e.g., m_wlist).
    vector_cache<T> m_vector_cache;
    // Value type for the working list m_wlist. It contains a half-open interval [lo, hi) paired with the rescaled
    // polynomial q(tau) = p(h*tau) represented in the Bernstein basis of [lo, hi] with interval coefficients. The
    // polynomial is stored in the usual flattened out layout with negated lower bounds for the interval coefficients.
    struct wlist_item {
        T lo{};
        T hi{};
        cached_vector<T> poly;
    };
    // Working list for the real root isolation loop.
    std::vector<wlist_item> m_wlist;
#if defined(HEYOKA_HAVE_REAL)
    // Vector cache for the extended-precision pass during real root isolation.
    vector_cache<mppp::real> m_vector_cache_3n;
    // These are the counterparts to wlist_item and m_wlist for the extended-precision pass. As usual, m_wlist_3n must
    // be declared *after* m_vector_cache_3n.
    struct wlist_item_3n {
        // NOTE: lo/hi are stored using the original type T, only the poly coefficients are stored in increased
        // precision.
        T lo{};
        T hi{};
        cached_vector<mppp::real> poly;
    };
    std::vector<wlist_item_3n> m_wlist_3n;
#endif

    // Serialisation.
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: needed only for s11n.
    frpr_workspace_impl();

    // Main constructor.
    explicit frpr_workspace_impl(long long, std::uint32_t, const llvm_state *);
    frpr_workspace_impl(const frpr_workspace_impl &);
    // NOTE: this is never needed, as the class is managed via a unique_ptr.
    frpr_workspace_impl(frpr_workspace_impl &&) noexcept = delete;
    frpr_workspace_impl &operator=(const frpr_workspace_impl &) = delete;
    frpr_workspace_impl &operator=(frpr_workspace_impl &&) noexcept = delete;
    ~frpr_workspace_impl();

#if defined(HEYOKA_HAVE_REAL)

    // Getter for the precision member, relevant only for multiprecision computations.
    [[nodiscard]] mpfr_prec_t get_prec() const noexcept
        requires std::same_as<mppp::real, T>;

#endif

    // Getter for the polynomial degree.
    [[nodiscard]] std::uint32_t get_degree() const noexcept;
};

#if defined(HEYOKA_HAVE_REAL)

// Helper to deduce the precision to use for the extended-precision pass during real root isolation.
template <typename T>
long long frpr_workspace_impl_get_prec_3n(long long);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
