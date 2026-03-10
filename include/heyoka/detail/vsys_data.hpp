// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSYS_DATA_HPP
#define HEYOKA_DETAIL_VSYS_DATA_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <cstdint>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/var_ode_sys.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Struct encapsulating data pertaining variational systems in Taylor integrators.
template <typename T>
struct vsys_data {
    // Compiled function for the evaluation of the Taylor map.
    cfunc<T> m_tm_cfunc;
    // Buffer storing the result of the evaluation of the Taylor map.
    std::vector<T> m_tm_output;

    // NOTE: this is used only for serialisation.
    vsys_data();
    explicit vsys_data(const var_ode_sys &, long long, const llvm_state &, std::uint32_t, bool, bool, bool);
    vsys_data(const vsys_data &);
    vsys_data(vsys_data &&) noexcept = delete;
    vsys_data &operator=(const vsys_data &) = delete;
    vsys_data &operator=(vsys_data &&) noexcept = delete;
    ~vsys_data();

    // Serialisation.
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
