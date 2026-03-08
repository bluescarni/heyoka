// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/ed_data.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
void ed_jit_data<T>::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << m_state;
}

template <typename T>
void ed_jit_data<T>::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> m_state;

    // Fetch the function pointers from the LLVM state.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));
}

// Explicit instantiations.
template struct ed_jit_data<float>;
template struct ed_jit_data<double>;
template struct ed_jit_data<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template struct ed_jit_data<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template struct ed_jit_data<mppp::real>;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
