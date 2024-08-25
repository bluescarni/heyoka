// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_I_DATA_HPP
#define HEYOKA_DETAIL_I_DATA_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/aligned_buffer.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
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

template <typename T>
struct taylor_adaptive<T>::i_data {
    // State vector.
    std::vector<T> m_state;
    // Time.
    detail::dfloat<T> m_time;
    // The LLVM (multi)state.
    std::variant<llvm_state, llvm_multi_state> m_llvm_state;
    // Dimension of the system.
    std::uint32_t m_dim{};
    // Taylor decomposition.
    taylor_dc_t m_dc;
    // Taylor order.
    std::uint32_t m_order{};
    // Tolerance.
    T m_tol{};
    // High accuracy.
    bool m_high_accuracy{};
    // Compact mode.
    bool m_compact_mode{};
    // The stepper types (non-compact mode).
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *) noexcept;
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *) noexcept;
    // The stepper types (compact mode). These have an additional argument - the tape pointer.
    using c_step_f_t = void (*)(T *, const T *, const T *, T *, T *, void *) noexcept;
    using c_step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *, void *) noexcept;
    // The stepper.
    std::variant<step_f_t, step_f_e_t, c_step_f_t, c_step_f_e_t> m_step_f;
    // Size/alignment for the compact mode tape.
    std::array<std::size_t, 2> m_tape_sa{};
    // Compact mode tape.
    detail::aligned_buffer_t m_tape;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // Size of the last timestep taken.
    T m_last_h{};
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *) noexcept;
    d_out_f_t m_d_out_f{};
    // The vector for the dense output.
    std::vector<T> m_d_out;
    // The ODE sys.
    sys_t m_vsys;
    // Taylor map data.
    std::optional<detail::tm_data<T>> m_tm_data;

private:
    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: used only for serialisation.
    i_data();

public:
    explicit i_data(llvm_state);

    i_data(const i_data &);

    i_data(i_data &&) noexcept = delete;
    i_data &operator=(const i_data &) = delete;
    i_data &operator=(i_data &&) noexcept = delete;

    ~i_data();

    void init_cm_tape();
};

template <typename T>
struct taylor_adaptive_batch<T>::i_data {
    // The batch size.
    std::uint32_t m_batch_size{};
    // State vectors.
    std::vector<T> m_state;
    // Times.
    std::vector<T> m_time_hi, m_time_lo;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Dimension of the system.
    std::uint32_t m_dim{};
    // Taylor decomposition.
    taylor_dc_t m_dc;
    // Taylor order.
    std::uint32_t m_order{};
    // Tolerance.
    T m_tol{};
    // High accuracy.
    bool m_high_accuracy{};
    // Compact mode.
    bool m_compact_mode{};
    // The steppers.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *) noexcept;
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *) noexcept;
    std::variant<step_f_t, step_f_e_t> m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // The sizes of the last timesteps taken.
    std::vector<T> m_last_h;
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *) noexcept;
    d_out_f_t m_d_out_f{};
    // The vector for the dense output.
    std::vector<T> m_d_out;
    // Temporary vectors for use
    // in the timestepping functions.
    // These two are used as default values,
    // they must never be modified.
    std::vector<T> m_pinf, m_minf;
    // This is used as temporary storage in step_impl().
    std::vector<T> m_delta_ts;
    // The vectors used to store the results of the step
    // and propagate functions.
    std::vector<std::tuple<taylor_outcome, T>> m_step_res;
    std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> m_prop_res;
    // Temporary vectors used in the step()/propagate_*() implementations.
    std::vector<std::size_t> m_ts_count;
    std::vector<T> m_min_abs_h, m_max_abs_h;
    std::vector<T> m_cur_max_delta_ts;
    std::vector<detail::dfloat<T>> m_pfor_ts;
    std::vector<int> m_t_dir;
    std::vector<detail::dfloat<T>> m_rem_time;
    std::vector<T> m_time_copy_hi, m_time_copy_lo;
    std::vector<int> m_nf_detected;
    // Temporary vector used in the dense output implementation.
    std::vector<T> m_d_out_time;
    // The ODE sys.
    sys_t m_vsys;
    // Taylor map data.
    std::optional<detail::tm_data<T>> m_tm_data;

private:
    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: used only for serialisation.
    i_data();

public:
    explicit i_data(llvm_state);

    i_data(const i_data &);

    i_data(i_data &&) noexcept = delete;
    i_data &operator=(const i_data &) = delete;
    i_data &operator=(i_data &&) noexcept = delete;

    ~i_data();
};

HEYOKA_END_NAMESPACE

#endif
