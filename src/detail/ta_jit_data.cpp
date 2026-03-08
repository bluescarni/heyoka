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

#include <heyoka/detail/ta_jit_data.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
void ta_jit_data<T>::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << m_llvm_state;
}

template <typename T>
void ta_jit_data<T>::load(boost::archive::binary_iarchive &ia, unsigned)
{
    // NOTE: here we are recovering only the JIT state, recovery of the function pointers is performed in parent
    // classes.
    ia >> m_llvm_state;
}

// Explicit instantiations.
template struct ta_jit_data<float>;
template struct ta_jit_data<double>;
template struct ta_jit_data<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template struct ta_jit_data<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template struct ta_jit_data<mppp::real>;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
