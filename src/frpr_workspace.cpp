// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <memory>
#include <utility>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/poly_root_finding/frpr_workspace_impl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

template <typename T>
void frpr_workspace<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_impl;
}

template <typename T>
void frpr_workspace<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    // Save the current impl for exception safety.
    auto old = std::move(m_impl);

    try {
        ar >> m_impl;
        // LCOV_EXCL_START
    } catch (...) {
        // Restore the original impl before re-throwing.
        m_impl = std::move(old);
        throw;
    }
    // LCOV_EXCL_STOP
}

// Main constructor.
//
// This constructor will set up a workspace for polynomials of degree 'degree'. The 'prec' argument (defaulting to zero)
// specifies the required precision for multiprecision computations. If T is a fixed-precision datatype, 'prec' is
// ignored, otherwise it must be a valid precision value. The optional s argument is a template llvm_state: if provided,
// the internal llvm_state will be constructed with the same settings as s, otherwise a default-constructed llvm_state
// will be used.
template <typename T>
frpr_workspace<T>::frpr_workspace(const std::uint32_t degree, const long long prec, const llvm_state *const s)
    : m_impl(std::make_unique<detail::frpr_workspace_impl<T>>(prec, degree, s))
{
}

template <typename T>
frpr_workspace<T>::frpr_workspace(const frpr_workspace &other)
    : m_impl(std::make_unique<detail::frpr_workspace_impl<T>>(*other.m_impl))
{
}

template <typename T>
frpr_workspace<T>::frpr_workspace(frpr_workspace &&) noexcept = default;

template <typename T>
frpr_workspace<T> &frpr_workspace<T>::operator=(const frpr_workspace &other)
{
    if (&other != this) [[likely]] {
        *this = frpr_workspace(other);
    }

    return *this;
}

template <typename T>
frpr_workspace<T> &frpr_workspace<T>::operator=(frpr_workspace &&) noexcept = default;

template <typename T>
frpr_workspace<T>::~frpr_workspace() = default;

#define HEYOKA_FRPR_WORKSPACE_INST(F) template class HEYOKA_DLL_PUBLIC frpr_workspace<F>

HEYOKA_FRPR_WORKSPACE_INST(float);
HEYOKA_FRPR_WORKSPACE_INST(double);
HEYOKA_FRPR_WORKSPACE_INST(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_FRPR_WORKSPACE_INST(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_FRPR_WORKSPACE_INST(mppp::real);

#endif

#undef HEYOKA_FRPR_WORKSPACE_INST

HEYOKA_END_NAMESPACE
