// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

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
    std::vector<expression> m_fn;
    std::vector<expression> m_vars;
    llvm_state m_s_scal;
    llvm_state m_s_batch;
    llvm_state m_s_scal_s;
    llvm_state m_s_batch_s;
    std::uint32_t m_simd_size = 0;
    std::vector<expression> m_dc;
    cfunc_ptr_t m_fptr_scal = nullptr;
    cfunc_ptr_t m_fptr_batch = nullptr;
    cfunc_ptr_s_t m_fptr_scal_s = nullptr;
    cfunc_ptr_s_t m_fptr_batch_s = nullptr;
    std::uint32_t m_nparams = 0;
    bool m_is_time_dependent = false;
    std::uint32_t m_nouts = 0;
    std::uint32_t m_nvars = 0;
    long long m_prec = 0;
    bool m_check_prec = false;

    // Serialization.
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        ar << m_fn;
        ar << m_vars;
        ar << m_s_scal;
        ar << m_s_batch;
        ar << m_s_scal_s;
        ar << m_s_batch_s;
        ar << m_simd_size;
        ar << m_dc;
        ar << m_nparams;
        ar << m_is_time_dependent;
        ar << m_nouts;
        ar << m_nvars;
        ar << m_prec;
        ar << m_check_prec;
    }
    void load(boost::archive::binary_iarchive &ar, unsigned)
    {
        ar >> m_fn;
        ar >> m_vars;
        ar >> m_s_scal;
        ar >> m_s_batch;
        ar >> m_s_scal_s;
        ar >> m_s_batch_s;
        ar >> m_simd_size;
        ar >> m_dc;
        ar >> m_nparams;
        ar >> m_is_time_dependent;
        ar >> m_nouts;
        ar >> m_nvars;
        ar >> m_prec;
        ar >> m_check_prec;

        // Recover the function pointers.
        m_fptr_scal = reinterpret_cast<cfunc_ptr_t>(m_s_scal.jit_lookup("cfunc"));
        m_fptr_batch = reinterpret_cast<cfunc_ptr_t>(m_s_batch.jit_lookup("cfunc"));
        m_fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(m_s_scal_s.jit_lookup("cfunc.strided"));
        m_fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(m_s_batch_s.jit_lookup("cfunc.strided"));
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // NOTE: this is necessary only for s11n.
    impl() = default;

    // NOTE: we use a single llvm_state for construction - all the internal
    // llvm_state instances will be copied/moved from s.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    explicit impl(std::vector<expression> fn, std::vector<expression> vars, llvm_state s,
                  std::optional<std::uint32_t> batch_size, bool high_accuracy, bool compact_mode, bool parallel_mode,
                  long long prec, bool check_prec)
        : m_fn(std::move(fn)), m_vars(std::move(vars)), m_s_scal(std::move(s)), m_s_batch(m_s_scal),
          m_s_scal_s(m_s_scal), m_s_batch_s(m_s_scal), m_prec(prec), m_check_prec(check_prec)
    {
        // Compute the SIMD size.
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        m_simd_size = batch_size ? *batch_size : recommended_simd_size<T>();

#if defined(HEYOKA_HAVE_REAL)

        // NOTE: batch size > 1u not supported for real.
        if (std::same_as<T, mppp::real> && m_simd_size > 1u) {
            throw std::invalid_argument("Batch size > 1 is not supported for mppp::real");
        }

#endif

        // Add the compiled functions.
        oneapi::tbb::parallel_invoke(
            [&]() {
                // Scalar unstrided.
                // NOTE: we fetch the decomposition from the scalar
                // unstrided invocation of add_cfunc().
                m_dc = add_cfunc<T>(m_s_scal, "cfunc", m_fn, m_vars, kw::high_accuracy = high_accuracy,
                                    kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode,
                                    kw::prec = m_prec);

                m_s_scal.compile();

                m_fptr_scal = reinterpret_cast<cfunc_ptr_t>(m_s_scal.jit_lookup("cfunc"));
            },
            [&]() {
                // Batch unstrided.
                add_cfunc<T>(m_s_batch, "cfunc", m_fn, m_vars, kw::batch_size = m_simd_size,
                             kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                             kw::parallel_mode = parallel_mode, kw::prec = m_prec);

                m_s_batch.compile();

                m_fptr_batch = reinterpret_cast<cfunc_ptr_t>(m_s_batch.jit_lookup("cfunc"));
            },
            [&]() {
                // Scalar strided.
                add_cfunc<T>(m_s_scal_s, "cfunc.strided", m_fn, m_vars, kw::high_accuracy = high_accuracy,
                             kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode, kw::prec = m_prec,
                             kw::strided = true);

                m_s_scal_s.compile();

                m_fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(m_s_scal_s.jit_lookup("cfunc.strided"));
            },
            [&]() {
                // Batch strided.
                add_cfunc<T>(m_s_batch_s, "cfunc.strided", m_fn, m_vars, kw::batch_size = m_simd_size,
                             kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                             kw::parallel_mode = parallel_mode, kw::prec = m_prec, kw::strided = true);

                m_s_batch_s.compile();

                m_fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(m_s_batch_s.jit_lookup("cfunc.strided"));
            });

        // Let's figure out if fn contains params and if it is time-dependent.
        m_nparams = get_param_size(m_fn);
        m_is_time_dependent = heyoka::is_time_dependent(m_fn);

        // Cache the number of variables and outputs.
        // NOTE: static casts should also be fine here, because add_cfunc()
        // succeeded and that guarantees that the number of vars and outputs
        // fits in a 32-bit int.
        m_nouts = boost::numeric_cast<std::uint32_t>(m_fn.size());
        m_nvars = boost::numeric_cast<std::uint32_t>(m_vars.size());
    }
    impl(const impl &other)
        : m_fn(other.m_fn), m_vars(other.m_vars), m_s_scal(other.m_s_scal), m_s_batch(other.m_s_batch),
          m_s_scal_s(other.m_s_scal_s), m_s_batch_s(other.m_s_batch_s), m_simd_size(other.m_simd_size),
          m_dc(other.m_dc), m_nparams(other.m_nparams), m_is_time_dependent(other.m_is_time_dependent),
          m_nouts(other.m_nouts), m_nvars(other.m_nvars), m_prec(other.m_prec), m_check_prec(other.m_check_prec)
    {
        // Recover the function pointers.
        m_fptr_scal = reinterpret_cast<cfunc_ptr_t>(m_s_scal.jit_lookup("cfunc"));
        m_fptr_batch = reinterpret_cast<cfunc_ptr_t>(m_s_batch.jit_lookup("cfunc"));
        m_fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(m_s_scal_s.jit_lookup("cfunc.strided"));
        m_fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(m_s_batch_s.jit_lookup("cfunc.strided"));
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
                std::tuple<bool, bool, bool, long long, std::optional<std::uint32_t>, llvm_state, bool> tup)
{
    // Unpack the tuple.
    auto &[high_accuracy, compact_mode, parallel_mode, prec, batch_size, s, check_prec] = tup;

    // Construct the impl.
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_impl = std::make_unique<impl>(std::move(fn), std::move(vars), std::move(s), batch_size, high_accuracy,
                                    compact_mode, parallel_mode, prec, check_prec);
}

template <typename T>
cfunc<T>::cfunc(const cfunc &other)
    : m_impl(
        // NOTE: support copy-construction from invalid object.
        other.m_impl ? std::make_unique<impl>(*other.m_impl) : nullptr)
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

template <typename T>
void cfunc<T>::check_valid(const char *name) const
{
    if (!m_impl) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The function '{}' cannot be invoked on an invalid cfunc object", name));
    }
}

template <typename T>
void cfunc<T>::operator()(out_1d outputs, in_1d inputs, std::optional<in_1d> pars, std::optional<T> time)
{
    check_valid(__func__);

    // Check the arguments.
    if (outputs.size() != m_impl->m_nouts) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid outputs array passed to a cfunc: the number of function "
                                                "outputs is {}, but the outputs array has a size of {}",
                                                m_impl->m_nouts, outputs.size()));
    }

    if (inputs.size() != m_impl->m_nvars) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid inputs array passed to a cfunc: the number of function "
                                                "inputs is {}, but the inputs array has a size of {}",
                                                m_impl->m_nvars, inputs.size()));
    }

    if (m_impl->m_nparams != 0u && !pars) [[unlikely]] {
        throw std::invalid_argument(
            "An array of parameter values must be passed in order to evaluate a function with parameters");
    }

    if (pars && pars->size() != m_impl->m_nparams) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The array of parameter values provided for the evaluation "
                                                "of a compiled function has {} element(s), "
                                                "but the number of parameters in the function is {}",
                                                pars->size(), m_impl->m_nparams));
    }

    if (m_impl->m_is_time_dependent && !time) [[unlikely]] {
        throw std::invalid_argument("A time value must be passed in order to evaluate a time-dependent function");
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::same_as<T, mppp::real>) {
        if (m_impl->m_check_prec) {
            const auto prec_checker = [&](const auto &x) {
                if (x.get_prec() != m_impl->m_prec) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "An mppp::real with an invalid precision of {} was detected in the arguments to the evaluation "
                        "of a compiled function - the expected precision value is {}",
                        x.get_prec(), m_impl->m_prec));
                }
            };

            std::ranges::for_each(outputs, prec_checker);
            std::ranges::for_each(inputs, prec_checker);

            if (pars) {
                std::ranges::for_each(*pars, prec_checker);
            }

            if (time) {
                prec_checker(*time);
            }
        }
    }

#endif

    // Invoke the compiled function.
    m_impl->m_fptr_scal(outputs.data(), inputs.data(), pars ? pars->data() : nullptr, time ? &*time : nullptr);
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
