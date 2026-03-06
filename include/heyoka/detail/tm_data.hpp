// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TM_DATA_HPP
#define HEYOKA_DETAIL_TM_DATA_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <cstdint>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/var_ode_sys.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Data for Taylor map computation.
template <typename T>
struct tm_data {
    using tm_func_t = void (*)(T *, const T *, const T *) noexcept;
    llvm_state m_state;
    tm_func_t m_tm_func{};
    std::vector<T> m_output;

    // NOTE: this is used only for serialisation.
    tm_data();
    explicit tm_data(const var_ode_sys &, long long, const llvm_state &, std::uint32_t);
    tm_data(const tm_data &);
    tm_data(tm_data &&) noexcept = delete;
    tm_data &operator=(const tm_data &) = delete;
    tm_data &operator=(tm_data &&) noexcept = delete;
    ~tm_data();

    // Serialisation.
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
