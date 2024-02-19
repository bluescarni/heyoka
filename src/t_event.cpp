// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <ostream>
#include <stdexcept>

#include <boost/core/demangle.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T, bool B>
void t_event_impl<T, B>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << eq;
    ar << callback;
    ar << cooldown;
    ar << dir;
}

template <typename T, bool B>
void t_event_impl<T, B>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> eq;
    ar >> callback;
    ar >> cooldown;
    ar >> dir;
}

template <typename T, bool B>
t_event_impl<T, B>::t_event_impl() : t_event_impl(expression{})
{
}

template <typename T, bool B>
void t_event_impl<T, B>::finalise_ctor(callback_t cb, T cd, event_direction d)
{
    using std::isfinite;

    callback = std::move(cb);

    if (!isfinite(cd)) {
        throw std::invalid_argument("Cannot set a non-finite cooldown value for a terminal event");
    }
    cooldown = cd;

    if (d < event_direction::negative || d > event_direction::positive) {
        throw std::invalid_argument("Invalid value selected for the direction of a terminal event");
    }

    dir = d;
}

template <typename T, bool B>
t_event_impl<T, B>::t_event_impl(const t_event_impl &) = default;

template <typename T, bool B>
t_event_impl<T, B>::t_event_impl(t_event_impl &&) noexcept = default;

template <typename T, bool B>
t_event_impl<T, B> &t_event_impl<T, B>::operator=(const t_event_impl &o)
{
    if (this != &o) {
        *this = t_event_impl(o);
    }

    return *this;
}

template <typename T, bool B>
t_event_impl<T, B> &t_event_impl<T, B>::operator=(t_event_impl &&) noexcept = default;

template <typename T, bool B>
t_event_impl<T, B>::~t_event_impl() = default;

template <typename T, bool B>
const expression &t_event_impl<T, B>::get_expression() const
{
    return eq;
}

template <typename T, bool B>
typename t_event_impl<T, B>::callback_t &t_event_impl<T, B>::get_callback()
{
    return callback;
}

template <typename T, bool B>
const typename t_event_impl<T, B>::callback_t &t_event_impl<T, B>::get_callback() const
{
    return callback;
}

template <typename T, bool B>
event_direction t_event_impl<T, B>::get_direction() const
{
    return dir;
}

template <typename T, bool B>
T t_event_impl<T, B>::get_cooldown() const
{
    return cooldown;
}

namespace
{

// Implementation of stream insertion for the terminal event class.
template <typename C, typename T>
std::ostream &t_event_impl_stream_impl(std::ostream &os, const expression &eq, event_direction dir, const C &callback,
                                       const T &cooldown)
{
    os << "C++ datatype   : " << boost::core::demangle(typeid(T).name()) << '\n';
    os << "Event type     : terminal\n";
    os << "Event equation : " << eq << '\n';
    os << "Event direction: " << dir << '\n';
    os << "With callback  : " << (callback ? "yes" : "no") << '\n';
    os << "Cooldown       : " << (cooldown < 0 ? "auto" : fp_to_string(cooldown)) << '\n';

    return os;
}

} // namespace

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<float, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<float, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<double, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<double, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<long double, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<long double, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<mppp::real128, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<mppp::real128, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<mppp::real, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

#endif

// Explicit instantiation.
#define HEYOKA_T_EVENT_INST(F)                                                                                         \
    template class HEYOKA_DLL_PUBLIC t_event_impl<F, true>;                                                            \
    template class HEYOKA_DLL_PUBLIC t_event_impl<F, false>;

HEYOKA_T_EVENT_INST(float)
HEYOKA_T_EVENT_INST(double)
HEYOKA_T_EVENT_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_T_EVENT_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

template class HEYOKA_DLL_PUBLIC t_event_impl<mppp::real, false>;

#endif

#undef HEYOKA_T_EVENT_INST

} // namespace detail

HEYOKA_END_NAMESPACE
