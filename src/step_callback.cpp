// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>

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
template class step_callback_impl<taylor_adaptive<double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<double>> &,
                                     step_callback_impl<taylor_adaptive<double>> &);

template class step_callback_impl<taylor_adaptive<long double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<long double>> &,
                                     step_callback_impl<taylor_adaptive<long double>> &);

#if defined(HEYOKA_HAVE_REAL128)

template class step_callback_impl<taylor_adaptive<mppp::real128>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<mppp::real128>> &,
                                     step_callback_impl<taylor_adaptive<mppp::real128>> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template class step_callback_impl<taylor_adaptive<mppp::real>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive<mppp::real>> &,
                                     step_callback_impl<taylor_adaptive<mppp::real>> &);

#endif

template class step_callback_impl<taylor_adaptive_batch<double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<double>> &,
                                     step_callback_impl<taylor_adaptive_batch<double>> &);

template class step_callback_impl<taylor_adaptive_batch<long double>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<long double>> &,
                                     step_callback_impl<taylor_adaptive_batch<long double>> &);

#if defined(HEYOKA_HAVE_REAL128)

template class step_callback_impl<taylor_adaptive_batch<mppp::real128>>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_impl<taylor_adaptive_batch<mppp::real128>> &,
                                     step_callback_impl<taylor_adaptive_batch<mppp::real128>> &);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
