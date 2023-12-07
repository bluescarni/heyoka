// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <initializer_list>
#include <stdexcept>
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

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

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
template class HEYOKA_DLL_PUBLIC step_callback_set_impl<float, true>;
template class HEYOKA_DLL_PUBLIC step_callback_set_impl<float, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<float, true> &,
                                     step_callback_set_impl<float, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<float, false> &,
                                     step_callback_set_impl<float, false> &) noexcept;

template class HEYOKA_DLL_PUBLIC step_callback_set_impl<double, true>;
template class HEYOKA_DLL_PUBLIC step_callback_set_impl<double, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<double, true> &,
                                     step_callback_set_impl<double, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<double, false> &,
                                     step_callback_set_impl<double, false> &) noexcept;

template class HEYOKA_DLL_PUBLIC step_callback_set_impl<long double, true>;
template class HEYOKA_DLL_PUBLIC step_callback_set_impl<long double, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<long double, true> &,
                                     step_callback_set_impl<long double, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<long double, false> &,
                                     step_callback_set_impl<long double, false> &) noexcept;

#if defined(HEYOKA_HAVE_REAL128)

template class HEYOKA_DLL_PUBLIC step_callback_set_impl<mppp::real128, true>;
template class HEYOKA_DLL_PUBLIC step_callback_set_impl<mppp::real128, false>;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<mppp::real128, true> &,
                                     step_callback_set_impl<mppp::real128, true> &) noexcept;
template HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<mppp::real128, false> &,
                                     step_callback_set_impl<mppp::real128, false> &) noexcept;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class HEYOKA_DLL_PUBLIC step_callback_set_impl<mppp::real, false>;
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
