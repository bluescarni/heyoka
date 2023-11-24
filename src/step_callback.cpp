// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/visibility.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: default construction builds an empty callback.
template <typename TA>
step_callback_impl<TA>::step_callback_impl() = default;

template <typename TA>
step_callback_impl<TA>::step_callback_impl(const step_callback_impl &other)
    : m_ptr(other ? other.m_ptr->clone() : nullptr){};

template <typename TA>
step_callback_impl<TA>::step_callback_impl(step_callback_impl &&) noexcept = default;

template <typename TA>
step_callback_impl<TA> &step_callback_impl<TA>::operator=(const step_callback_impl &other)
{
    if (this != &other) {
        *this = step_callback_impl(other);
    }

    return *this;
}

template <typename TA>
step_callback_impl<TA> &step_callback_impl<TA>::operator=(step_callback_impl &&) noexcept = default;

template <typename TA>
step_callback_impl<TA>::~step_callback_impl() = default;

template <typename TA>
step_callback_impl<TA>::operator bool() const noexcept
{
    return static_cast<bool>(m_ptr);
}

template <typename TA>
bool step_callback_impl<TA>::operator()(TA &ta)
{
    if (!m_ptr) {
        throw std::bad_function_call();
    }

    return m_ptr->operator()(ta);
}

template <typename TA>
void step_callback_impl<TA>::pre_hook(TA &ta)
{
    if (!m_ptr) {
        throw std::bad_function_call();
    }

    m_ptr->pre_hook(ta);
}

template <typename TA>
void step_callback_impl<TA>::swap(step_callback_impl &other) noexcept
{
    std::swap(m_ptr, other.m_ptr);
}

template <typename TA>
std::type_index step_callback_impl<TA>::get_type_index() const
{
    if (m_ptr) {
        return m_ptr->get_type_index();
    } else {
        return typeid(void);
    }
}

template <typename TA>
void swap(step_callback_impl<TA> &a, step_callback_impl<TA> &b) noexcept
{
    a.swap(b);
}

// Explicit instantiations.
template class step_callback_impl<taylor_adaptive<float>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<float>> &,
                                     step_callback_impl<taylor_adaptive<float>> &) noexcept;

template class step_callback_impl<taylor_adaptive<double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<double>> &,
                                     step_callback_impl<taylor_adaptive<double>> &) noexcept;

template class step_callback_impl<taylor_adaptive<long double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<long double>> &,
                                     step_callback_impl<taylor_adaptive<long double>> &) noexcept;

#if defined(HEYOKA_HAVE_REAL128)

template class step_callback_impl<taylor_adaptive<mppp::real128>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<mppp::real128>> &,
                                     step_callback_impl<taylor_adaptive<mppp::real128>> &) noexcept;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class step_callback_impl<taylor_adaptive<mppp::real>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<mppp::real>> &,
                                     step_callback_impl<taylor_adaptive<mppp::real>> &) noexcept;

#endif

template class step_callback_impl<taylor_adaptive_batch<float>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<float>> &,
                                     step_callback_impl<taylor_adaptive_batch<float>> &) noexcept;

template class step_callback_impl<taylor_adaptive_batch<double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<double>> &,
                                     step_callback_impl<taylor_adaptive_batch<double>> &) noexcept;

template class step_callback_impl<taylor_adaptive_batch<long double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<long double>> &,
                                     step_callback_impl<taylor_adaptive_batch<long double>> &) noexcept;

#if defined(HEYOKA_HAVE_REAL128)

template class step_callback_impl<taylor_adaptive_batch<mppp::real128>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<mppp::real128>> &,
                                     step_callback_impl<taylor_adaptive_batch<mppp::real128>> &) noexcept;

#endif

template <typename T, bool Batch>
step_callback_set_impl<T, Batch>::step_callback_set_impl() noexcept = default;

template <typename T, bool Batch>
step_callback_set_impl<T, Batch>::step_callback_set_impl(std::vector<step_cb_t> cbs) : m_cbs(std::move(cbs))
{
    for (const auto &cb : m_cbs) {
        if (!cb) {
            throw std::invalid_argument("Cannot construct a callback set containing one or more empty callbacks");
        }
    }
}

template <typename T, bool Batch>
step_callback_set_impl<T, Batch>::step_callback_set_impl(std::initializer_list<step_cb_t> cbs)
    : step_callback_set_impl(std::vector<step_cb_t>{cbs})
{
}

template <typename T, bool Batch>
step_callback_set_impl<T, Batch>::step_callback_set_impl(const step_callback_set_impl &) = default;

template <typename T, bool Batch>
step_callback_set_impl<T, Batch>::step_callback_set_impl(step_callback_set_impl &&) noexcept = default;

template <typename T, bool Batch>
step_callback_set_impl<T, Batch> &step_callback_set_impl<T, Batch>::operator=(const step_callback_set_impl &) = default;

template <typename T, bool Batch>
step_callback_set_impl<T, Batch> &step_callback_set_impl<T, Batch>::operator=(step_callback_set_impl &&) noexcept
    = default;

template <typename T, bool Batch>
step_callback_set_impl<T, Batch>::~step_callback_set_impl() = default;

template <typename T, bool Batch>
typename step_callback_set_impl<T, Batch>::size_type step_callback_set_impl<T, Batch>::size() const noexcept
{
    return m_cbs.size();
}

namespace
{

constexpr auto scs_index_err_msg = "Out of range index {} when accessing a step callback set of size {}";

} // namespace

template <typename T, bool Batch>
const typename step_callback_set_impl<T, Batch>::step_cb_t &
step_callback_set_impl<T, Batch>::operator[](size_type i) const
{
    if (i >= size()) {
        throw std::out_of_range(fmt::format(scs_index_err_msg, i, size()));
    }

    return m_cbs[i];
}

template <typename T, bool Batch>
typename step_callback_set_impl<T, Batch>::step_cb_t &step_callback_set_impl<T, Batch>::operator[](size_type i)
{
    if (i >= size()) {
        throw std::out_of_range(fmt::format(scs_index_err_msg, i, size()));
    }

    return m_cbs[i];
}

template <typename T, bool Batch>
bool step_callback_set_impl<T, Batch>::operator()(ta_t &ta)
{
    bool retval = true;

    for (auto &cb : m_cbs) {
        // NOTE: always execute the callback first,
        // and only then combine the outcome with retval.
        retval = cb(ta) && retval;
    }

    return retval;
}

template <typename T, bool Batch>
void step_callback_set_impl<T, Batch>::pre_hook(ta_t &ta)
{
    for (auto &cb : m_cbs) {
        cb.pre_hook(ta);
    }
}

template <typename T, bool Batch>
void swap(step_callback_set_impl<T, Batch> &c1, step_callback_set_impl<T, Batch> &c2) noexcept
{
    c1.m_cbs.swap(c2.m_cbs);
}

// Explicit instantiations.
template class step_callback_set_impl<float, true>;
template class step_callback_set_impl<float, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<float, true> &,
                                     step_callback_set_impl<float, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<float, false> &,
                                     step_callback_set_impl<float, false> &) noexcept;

template class step_callback_set_impl<double, true>;
template class step_callback_set_impl<double, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<double, true> &,
                                     step_callback_set_impl<double, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<double, false> &,
                                     step_callback_set_impl<double, false> &) noexcept;

template class step_callback_set_impl<long double, true>;
template class step_callback_set_impl<long double, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<long double, true> &,
                                     step_callback_set_impl<long double, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<long double, false> &,
                                     step_callback_set_impl<long double, false> &) noexcept;

#if defined(HEYOKA_HAVE_REAL128)

template class step_callback_set_impl<mppp::real128, true>;
template class step_callback_set_impl<mppp::real128, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<mppp::real128, true> &,
                                     step_callback_set_impl<mppp::real128, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<mppp::real128, false> &,
                                     step_callback_set_impl<mppp::real128, false> &) noexcept;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class step_callback_set_impl<mppp::real, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<mppp::real, false> &,
                                     step_callback_set_impl<mppp::real, false> &) noexcept;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE

// Implementation of s11n support for step_callback_set.
HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::step_callback_set<float>, float)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::step_callback_set<double>, double)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::step_callback_set<long double>, long double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::step_callback_batch_set<float>, float)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::step_callback_batch_set<double>, double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::step_callback_batch_set<long double>, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::step_callback_set<mppp::real128>, mppp::real128)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::step_callback_batch_set<mppp::real128>, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::step_callback_set<mppp::real>, mppp::real)

#endif
