// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>

#include <heyoka/callable.hpp>
#include <heyoka/config.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

// NOTE: this file contains the code for registering default-constructed
// event and step callbacks in the serialisation system.

// NOLINTBEGIN(cert-err58-cpp)

// Scalar and batch event callbacks.
#define HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS(T)                                                                       \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, void, heyoka::taylor_adaptive<T> &, T, int)  \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, bool, heyoka::taylor_adaptive<T> &, int)

#define HEYOKA_S11N_IMPLEMENT_BATCH_EVENT_CALLBACKS(T)                                                                 \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, void, heyoka::taylor_adaptive_batch<T> &, T, \
                                          int, std::uint32_t)                                                          \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, bool, heyoka::taylor_adaptive_batch<T> &,    \
                                          int, std::uint32_t)

HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS(float)
HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS(double)
HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS(long double)

HEYOKA_S11N_IMPLEMENT_BATCH_EVENT_CALLBACKS(float)
HEYOKA_S11N_IMPLEMENT_BATCH_EVENT_CALLBACKS(double)
HEYOKA_S11N_IMPLEMENT_BATCH_EVENT_CALLBACKS(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS(mppp::real128)
HEYOKA_S11N_IMPLEMENT_BATCH_EVENT_CALLBACKS(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS(mppp::real)

#endif

#undef HEYOKA_S11N_IMPLEMENT_EVENT_CALLBACKS
#undef HEYOKA_S11N_IMPLEMENT_BATCH_EVENT_CALLBACKS

// Scalar and batch step callbacks.
HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, float)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, double)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, long double)

HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, float)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, mppp::real128)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::detail::empty_callable, mppp::real)

#endif

// NOLINTEND(cert-err58-cpp)
