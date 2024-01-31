// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <oneapi/tbb/parallel_invoke.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

template <typename T>
struct cfunc<T>::impl {
    // The compiled function types.
    using cfunc_ptr_t = void (*)(T *, const T *, const T *, const T *) noexcept;
    using cfunc_ptr_s_t = void (*)(T *, const T *, const T *, const T *, std::size_t) noexcept;

    // Data members.
    std::vector<expression> fn;
    std::vector<expression> vars;
    llvm_state s_scal;
    llvm_state s_batch;
    llvm_state s_scal_s;
    llvm_state s_batch_s;
    std::uint32_t simd_size = 0;
    std::vector<expression> dc;
    cfunc_ptr_t fptr_scal = nullptr;
    cfunc_ptr_t fptr_batch = nullptr;
    cfunc_ptr_s_t fptr_scal_s = nullptr;
    cfunc_ptr_s_t fptr_batch_s = nullptr;

    // Serialization.
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        ar << fn;
        ar << vars;
        ar << s_scal;
        ar << s_batch;
        ar << s_scal_s;
        ar << s_batch_s;
        ar << simd_size;
        ar << dc;
    }
    void load(boost::archive::binary_iarchive &ar, unsigned)
    {
        ar >> fn;
        ar >> vars;
        ar >> s_scal;
        ar >> s_batch;
        ar >> s_scal_s;
        ar >> s_batch_s;
        ar >> simd_size;
        ar >> dc;

        // Recover the function pointers.
        fptr_scal = reinterpret_cast<cfunc_ptr_t>(s_scal.jit_lookup("cfunc"));
        fptr_batch = reinterpret_cast<cfunc_ptr_t>(s_batch.jit_lookup("cfunc"));
        fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(s_scal_s.jit_lookup("cfunc.strided"));
        fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(s_batch_s.jit_lookup("cfunc.strided"));
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: this is necessary only for s11n.
    impl() = default;

    // NOTE: we use a single llvm_state for construction - all the internal
    // llvm_state instances will be copied/moved from s.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    explicit impl(std::vector<expression> fn, std::vector<expression> vars, llvm_state s,
                  std::optional<std::uint32_t> batch_size, bool high_accuracy, bool compact_mode, bool parallel_mode,
                  long long prec)
        : fn(std::move(fn)), vars(std::move(vars)), s_scal(std::move(s)), s_batch(s_scal), s_scal_s(s_scal),
          s_batch_s(s_scal)
    {
        // Compute the SIMD size.
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        simd_size = batch_size ? *batch_size : recommended_simd_size<T>();

#if defined(HEYOKA_HAVE_REAL)

        // NOTE: batch size > 1u not supported for real.
        if (std::same_as<T, mppp::real> && simd_size > 1u) {
            throw std::invalid_argument("Batch size > 1 is not supported for mppp::real");
        }

#endif

        // Add the compiled functions.
        oneapi::tbb::parallel_invoke(
            [&]() {
                // Scalar unstrided.
                // NOTE: we fetch the decomposition from the scalar
                // unstrided invocation of add_cfunc().
                dc = add_cfunc<T>(s_scal, "cfunc", fn, vars, kw::high_accuracy = high_accuracy,
                                  kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode, kw::prec = prec);

                s_scal.compile();

                fptr_scal = reinterpret_cast<cfunc_ptr_t>(s_scal.jit_lookup("cfunc"));
            },
            [&]() {
                // Batch unstrided.
                add_cfunc<T>(s_batch, "cfunc", fn, vars, kw::batch_size = simd_size, kw::high_accuracy = high_accuracy,
                             kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode, kw::prec = prec);

                s_batch.compile();

                fptr_batch = reinterpret_cast<cfunc_ptr_t>(s_batch.jit_lookup("cfunc"));
            },
            [&]() {
                // Scalar strided.
                add_cfunc<T>(s_scal_s, "cfunc.strided", fn, vars, kw::high_accuracy = high_accuracy,
                             kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode, kw::prec = prec,
                             kw::strided = true);

                s_scal_s.compile();

                fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(s_scal_s.jit_lookup("cfunc.strided"));
            },
            [&]() {
                // Batch strided.
                add_cfunc<T>(s_batch_s, "cfunc.strided", fn, vars, kw::batch_size = simd_size,
                             kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                             kw::parallel_mode = parallel_mode, kw::prec = prec, kw::strided = true);

                s_batch_s.compile();

                fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(s_batch_s.jit_lookup("cfunc.strided"));
            });
    }
    impl(const impl &other)
        : fn(other.fn), vars(other.vars), s_scal(other.s_scal), s_batch(other.s_batch), s_scal_s(other.s_scal_s),
          s_batch_s(other.s_batch_s), simd_size(other.simd_size), dc(other.dc)
    {
        // Recover the function pointers.
        fptr_scal = reinterpret_cast<cfunc_ptr_t>(s_scal.jit_lookup("cfunc"));
        fptr_batch = reinterpret_cast<cfunc_ptr_t>(s_batch.jit_lookup("cfunc"));
        fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(s_scal_s.jit_lookup("cfunc.strided"));
        fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(s_batch_s.jit_lookup("cfunc.strided"));
    }

    // These are never needed.
    impl(impl &&) noexcept = delete;
    impl &operator=(const impl &) = delete;
    impl &operator=(impl &&) noexcept = delete;

    ~impl() = default;
};

template <typename T>
cfunc<T>::cfunc() = default;

template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
cfunc<T>::cfunc(std::vector<expression> fn, std::vector<expression> vars,
                // NOLINTNEXTLINE(performance-unnecessary-value-param)
                std::tuple<bool, bool, bool, long long, std::optional<std::uint32_t>, llvm_state> tup)
{
    // Unpack the tuple.
    auto &[high_accuracy, compact_mode, parallel_mode, prec, batch_size, s] = tup;

    // Construct the impl.
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_impl = std::make_unique<impl>(std::move(fn), std::move(vars), std::move(s), batch_size, high_accuracy,
                                    compact_mode, parallel_mode, prec);
}

template <typename T>
cfunc<T>::cfunc(const cfunc &other) : m_impl(std::make_unique<impl>(*other.m_impl))
{
}

template <typename T>
cfunc<T>::cfunc(cfunc &&) noexcept = default;

template <typename T>
cfunc<T> &cfunc<T>::operator=(const cfunc &other)
{
    if (this != &other) {
        *this = cfunc(other);
    }

    return *this;
}

template <typename T>
cfunc<T> &cfunc<T>::operator=(cfunc &&) noexcept = default;

template <typename T>
cfunc<T>::~cfunc() = default;

template <typename T>
void cfunc<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_impl;
}

template <typename T>
void cfunc<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    try {
        ar >> m_impl;
        // LCOV_EXCL_START
    } catch (...) {
        *this = cfunc{};

        throw;
    }
    // LCOV_EXCL_STOP
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC cfunc<float>;
template class HEYOKA_DLL_PUBLIC cfunc<double>;
template class HEYOKA_DLL_PUBLIC cfunc<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class HEYOKA_DLL_PUBLIC cfunc<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class HEYOKA_DLL_PUBLIC cfunc<mppp::real>;

#endif

HEYOKA_END_NAMESPACE
