// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <mp++/real.hpp>

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

    if (!isfinite(cd)) [[unlikely]] {
        throw std::invalid_argument("Cannot set a non-finite cooldown value for a terminal event");
    }
    if (cd == 0) [[unlikely]] {
        throw std::invalid_argument("A zero cooldown is invalid: please provide a positive value or leave it "
                                    "unspecified for automatic cooldown deduction");
    }
    cooldown = std::move(cd);

    if (d < event_direction::negative || d > event_direction::positive) [[unlikely]] {
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
    if (this != &o) [[likely]] {
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
t_event_impl<T, B>::callback_t &t_event_impl<T, B>::get_callback()
{
    return callback;
}

template <typename T, bool B>
const t_event_impl<T, B>::callback_t &t_event_impl<T, B>::get_callback() const
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

// Implementation of stream insertion for the terminal event class.
template <typename T, bool B>
std::ostream &operator<<(std::ostream &os, const t_event_impl<T, B> &ev)
{
    os << "C++ datatype   : " << boost::core::demangle(typeid(T).name()) << '\n';
    os << "Event type     : terminal\n";
    os << "Event equation : " << ev.get_expression() << '\n';
    os << "Event direction: " << ev.get_direction() << '\n';
    os << "With callback  : " << (ev.get_callback() ? "yes" : "no") << '\n';
    os << "Cooldown       : " << (ev.get_cooldown() < 0 ? "auto" : fp_to_string(ev.get_cooldown())) << '\n';

    return os;
}

// Explicit instantiations.
#define HEYOKA_T_EVENT_INST_SCALAR(F)                                                                                  \
    template class HEYOKA_DLL_PUBLIC t_event_impl<F, false>;                                                           \
    template HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<F, false> &);

#define HEYOKA_T_EVENT_INST_BATCH(F)                                                                                   \
    template class HEYOKA_DLL_PUBLIC t_event_impl<F, true>;                                                            \
    template HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<F, true> &);

#define HEYOKA_T_EVENT_INST(F)                                                                                         \
    HEYOKA_T_EVENT_INST_SCALAR(F)                                                                                      \
    HEYOKA_T_EVENT_INST_BATCH(F)

HEYOKA_T_EVENT_INST(float)
HEYOKA_T_EVENT_INST(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_T_EVENT_INST(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_T_EVENT_INST(mppp::real128)

#endif

HEYOKA_T_EVENT_INST_SCALAR(mppp::real)

#undef HEYOKA_T_EVENT_INST_SCALAR
#undef HEYOKA_T_EVENT_INST_BATCH
#undef HEYOKA_T_EVENT_INST

} // namespace detail

HEYOKA_END_NAMESPACE
