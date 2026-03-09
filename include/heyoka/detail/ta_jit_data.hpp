// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TA_JIT_DATA_HPP
#define HEYOKA_DETAIL_TA_JIT_DATA_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <variant>

#include <heyoka/config.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Struct containing the JIT-compiled code for Taylor integrators.
template <typename T>
struct ta_jit_data {
    // The LLVM (multi)state.
    std::variant<llvm_state, llvm_multi_state> m_llvm_state;
    // The stepper types (non-compact mode).
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *) noexcept;
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *) noexcept;
    // The stepper types (compact mode). These have an additional argument - the tape pointer.
    using c_step_f_t = void (*)(T *, const T *, const T *, T *, T *, void *) noexcept;
    using c_step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *, void *) noexcept;
    // The stepper.
    std::variant<step_f_t, step_f_e_t, c_step_f_t, c_step_f_e_t> m_step_f{};
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *) noexcept;
    d_out_f_t m_d_out_f = nullptr;

    // Serialisation.
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
