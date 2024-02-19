// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
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
#include <heyoka/detail/visibility.hpp>
#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

#define HEYOKA_TAYLOR_ENUM_STREAM_CASE(val)                                                                            \
    case val:                                                                                                          \
        os << #val;                                                                                                    \
        break

std::ostream &operator<<(std::ostream &os, event_direction dir)
{
    switch (dir) {
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(event_direction::any);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(event_direction::positive);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(event_direction::negative);
        default:
            // Unknown value.
            os << "event_direction::??";
    }

    return os;
}

#undef HEYOKA_TAYLOR_ENUM_STREAM_CASE

namespace detail
{

namespace
{

// Helper to create the callback used in the default
// constructor of a non-terminal event.
template <typename T, bool B>
auto nt_event_def_cb()
{
    if constexpr (B) {
        return [](taylor_adaptive_batch<T> &, T, int, std::uint32_t) {};
    } else {
        return [](taylor_adaptive<T> &, T, int) {};
    }
}

} // namespace

template <typename T, bool B>
void nt_event_impl<T, B>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << eq;
    ar << callback;
    ar << dir;
}

template <typename T, bool B>
void nt_event_impl<T, B>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> eq;
    ar >> callback;
    ar >> dir;
}

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl() : nt_event_impl(expression{}, nt_event_def_cb<T, B>())
{
}

template <typename T, bool B>
void nt_event_impl<T, B>::finalise_ctor(event_direction d)
{
    if (!callback) {
        throw std::invalid_argument("Cannot construct a non-terminal event with an empty callback");
    }

    if (d < event_direction::negative || d > event_direction::positive) {
        throw std::invalid_argument("Invalid value selected for the direction of a non-terminal event");
    }

    dir = d;
}

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl(const nt_event_impl &) = default;

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl(nt_event_impl &&) noexcept = default;

template <typename T, bool B>
nt_event_impl<T, B> &nt_event_impl<T, B>::operator=(const nt_event_impl &o)
{
    if (this != &o) {
        *this = nt_event_impl(o);
    }

    return *this;
}

template <typename T, bool B>
nt_event_impl<T, B> &nt_event_impl<T, B>::operator=(nt_event_impl &&) noexcept = default;

template <typename T, bool B>
nt_event_impl<T, B>::~nt_event_impl() = default;

template <typename T, bool B>
const expression &nt_event_impl<T, B>::get_expression() const
{
    return eq;
}

template <typename T, bool B>
typename nt_event_impl<T, B>::callback_t &nt_event_impl<T, B>::get_callback()
{
    return callback;
}

template <typename T, bool B>
const typename nt_event_impl<T, B>::callback_t &nt_event_impl<T, B>::get_callback() const
{
    return callback;
}

template <typename T, bool B>
event_direction nt_event_impl<T, B>::get_direction() const
{
    return dir;
}

namespace
{

// Implementation of stream insertion for the non-terminal event class.
template <typename T>
std::ostream &nt_event_impl_stream_impl(std::ostream &os, const expression &eq, event_direction dir)
{
    os << "C++ datatype   : " << boost::core::demangle(typeid(T).name()) << '\n';
    os << "Event type     : non-terminal\n";
    os << "Event equation : " << eq << '\n';
    os << "Event direction: " << dir << '\n';

    return os;
}

} // namespace

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<float, false> &e)
{
    return nt_event_impl_stream_impl<float>(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<float, true> &e)
{
    return nt_event_impl_stream_impl<float>(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<double, false> &e)
{
    return nt_event_impl_stream_impl<double>(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<double, true> &e)
{
    return nt_event_impl_stream_impl<double>(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<long double, false> &e)
{
    return nt_event_impl_stream_impl<long double>(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<long double, true> &e)
{
    return nt_event_impl_stream_impl<long double>(os, e.get_expression(), e.get_direction());
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<mppp::real128, false> &e)
{
    return nt_event_impl_stream_impl<mppp::real128>(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<mppp::real128, true> &e)
{
    return nt_event_impl_stream_impl<mppp::real128>(os, e.get_expression(), e.get_direction());
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<mppp::real, false> &e)
{
    return nt_event_impl_stream_impl<mppp::real>(os, e.get_expression(), e.get_direction());
}

#endif

// Explicit instantiation.
#define HEYOKA_NT_EVENT_INST(F)                                                                                        \
    template class HEYOKA_DLL_PUBLIC nt_event_impl<F, true>;                                                           \
    template class HEYOKA_DLL_PUBLIC nt_event_impl<F, false>;

HEYOKA_NT_EVENT_INST(float)
HEYOKA_NT_EVENT_INST(double)
HEYOKA_NT_EVENT_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_NT_EVENT_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

template class HEYOKA_DLL_PUBLIC nt_event_impl<mppp::real, false>;

#endif

#undef HEYOKA_NT_EVENT_INST

} // namespace detail

HEYOKA_END_NAMESPACE
