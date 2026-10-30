// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FIND_REAL_POLY_ROOTS_HPP
#define HEYOKA_FIND_REAL_POLY_ROOTS_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

// Fwd declaration.
template <typename>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS frpr_workspace;

namespace detail
{

// Enum describing the type of an interval returned by isolate_real_roots().
//
// NOLINTNEXTLINE(performance-enum-size)
enum class irr_interval_type {
    // The interval may or may not contain real roots.
    undecided,
    // The interval contains exactly one real root.
    isolating,
};

// Struct representing the elements of the list returned by isolate_real_roots().
//
// If the type is 'isolating', there are two possibilities:
//
// - lo<hi: the interval is to be regarded as an open interval (lo, hi) within which a single simple real root is
//   guaranteed to exist;
// - lo==hi: a root of undetermined multiplicity exists *exactly* at lo/hi.
//
// For isolating intervals, the direction of the root (that is, the sign of the derivative at the root), if available,
// is given by 'sign'. The sign is guaranteed to be -1 or 1 for lo<hi, or zero for lo==hi (indicating that the sign is
// undetermined).
//
// If the interval type is 'undecided', it means that the real root isolation function shrunk the interval to a small
// size without being able to fully determine if and how many roots exist in the interval. In this case, the interval is
// to be regarded as a half-open interval [lo, hi) and the sign field is zero.
//
// NOTE: see the docs of isolate_real_roots() for the units of lo, hi and sign.
template <typename T>
struct irr_interval {
    T lo{};
    T hi{};
    irr_interval_type type = irr_interval_type::undecided;
    int sign = 0;
};

// Isolate the real roots for a polynomial.
template <typename T>
const std::vector<irr_interval<T>> &isolate_real_roots(frpr_workspace<T> &, std::span<const T>, const T &, const T &);

// Forward declaration of the pimpl implementation of frpr_workspace.
template <typename T>
struct frpr_workspace_impl;

} // namespace detail

// Struct representing a polynomial root returned by find_real_poly_roots().
template <typename T>
struct real_poly_root {
    // The root value.
    T t{};
    // The direction: +1 if the derivative of p(t) is positive at the root, -1 if the derivative is negative, 0 if the
    // sign is undetermined.
    int dir = 0;
};

// Find the real roots of a polynomial within an interval.
template <typename T>
void find_real_poly_roots(std::vector<real_poly_root<T>> &, frpr_workspace<T> &, std::span<const T>, const T &, T = 0);

// Workspace for find_real_poly_roots().
//
// This is an opaque class that encapsulates state required by find_real_poly_roots().
template <typename T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS frpr_workspace
{
    std::unique_ptr<detail::frpr_workspace_impl<T>> m_impl;

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // Allow several functions to access the internals of the workspace.
    friend const std::vector<detail::irr_interval<T>> &
    detail::isolate_real_roots<T>(frpr_workspace<T> &, std::span<const T>, const T &, const T &);
    friend void find_real_poly_roots<T>(std::vector<real_poly_root<T>> &, frpr_workspace<T> &, std::span<const T>,
                                        const T &, T);

public:
    explicit frpr_workspace(std::uint32_t, long long = 0, const llvm_state * = nullptr);
    frpr_workspace(const frpr_workspace &);
    frpr_workspace(frpr_workspace &&) noexcept;
    frpr_workspace &operator=(const frpr_workspace &);
    frpr_workspace &operator=(frpr_workspace &&) noexcept;
    ~frpr_workspace();
};

HEYOKA_END_NAMESPACE

#endif
