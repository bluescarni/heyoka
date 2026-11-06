// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_CONTINUOUS_OUTPUT_HPP
#define HEYOKA_CONTINUOUS_OUTPUT_HPP

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/supported_fp_type.hpp>

HEYOKA_BEGIN_NAMESPACE

template <typename T>
std::ostream &operator<<(std::ostream &, const continuous_output<T> &);

template <supported_fp_type T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS continuous_output
{
    friend class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive<T>;
    friend std::ostream &operator<< <T>(std::ostream &, const continuous_output<T> &);

    llvm_state m_llvm_state;
    std::vector<T> m_tcs;
    std::vector<T> m_times_hi, m_times_lo;
    std::vector<T> m_output;
    using fptr_t = void (*)(T *, T *, const T *, const T *, const T *) noexcept;
    fptr_t m_f_ptr = nullptr;

    HEYOKA_DLL_LOCAL void add_c_out_function(std::uint32_t, std::uint32_t, bool);
    void call_impl(T);

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    continuous_output();
    explicit continuous_output(llvm_state &&);
    continuous_output(const continuous_output &);
    continuous_output(continuous_output &&) noexcept;
    ~continuous_output();

    continuous_output &operator=(const continuous_output &);
    continuous_output &operator=(continuous_output &&) noexcept;

    [[nodiscard]] const llvm_state &get_llvm_state() const;

    const std::vector<T> &operator()(T);
    [[nodiscard]] const std::vector<T> &get_output() const;
    [[nodiscard]] const std::vector<T> &get_times() const;
    [[nodiscard]] const std::vector<T> &get_tcs() const;

    [[nodiscard]] std::pair<T, T> get_bounds() const;
    [[nodiscard]] std::size_t get_n_steps() const;
};

template <typename T>
std::ostream &operator<<(std::ostream &, const continuous_output_batch<T> &);

template <supported_batch_fp_type T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS continuous_output_batch
{
    friend class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive_batch<T>;
    friend std::ostream &operator<< <T>(std::ostream &, const continuous_output_batch<T> &);

    std::uint32_t m_batch_size = 0;
    llvm_state m_llvm_state;
    std::vector<T> m_tcs;
    std::vector<T> m_times_hi, m_times_lo;
    std::vector<T> m_output;
    std::vector<T> m_tmp_tm;
    using fptr_t = void (*)(T *, const T *, const T *, const T *, const T *) noexcept;
    fptr_t m_f_ptr = nullptr;

    HEYOKA_DLL_LOCAL void add_c_out_function(std::uint32_t, std::uint32_t, bool);
    void call_impl(const T *);

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    continuous_output_batch();
    explicit continuous_output_batch(llvm_state &&);
    continuous_output_batch(const continuous_output_batch &);
    continuous_output_batch(continuous_output_batch &&) noexcept;
    ~continuous_output_batch();

    continuous_output_batch &operator=(const continuous_output_batch &);
    continuous_output_batch &operator=(continuous_output_batch &&) noexcept;

    [[nodiscard]] const llvm_state &get_llvm_state() const;

    const std::vector<T> &operator()(const T *);
    const std::vector<T> &operator()(const std::vector<T> &);
    const std::vector<T> &operator()(T);

    [[nodiscard]] const std::vector<T> &get_output() const;
    // NOTE: when documenting this function, we need to warn about the padding.
    [[nodiscard]] const std::vector<T> &get_times() const;
    [[nodiscard]] const std::vector<T> &get_tcs() const;
    [[nodiscard]] std::uint32_t get_batch_size() const;

    [[nodiscard]] std::pair<std::vector<T>, std::vector<T>> get_bounds() const;
    [[nodiscard]] std::size_t get_n_steps() const;
};

HEYOKA_END_NAMESPACE

#endif
