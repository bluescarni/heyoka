// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <mp++/real.hpp>

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
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
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
    ar << cross_step;
}

template <typename T, bool B>
void nt_event_impl<T, B>::load(boost::archive::binary_iarchive &ar, const unsigned version)
{
    if (version < static_cast<unsigned>(boost::serialization::version<nt_event_impl>::type::value)) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Unable to load a non-terminal event object: the archive version ({}) is too old", version));
        // LCOV_EXCL_STOP
    }

    ar >> eq;
    ar >> callback;
    ar >> dir;
    ar >> cross_step;
}

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl() : nt_event_impl(expression{}, nt_event_def_cb<T, B>())
{
}

template <typename T, bool B>
void nt_event_impl<T, B>::finalise_ctor(const event_direction d, const bool cstep)
{
    if (!callback) [[unlikely]] {
        throw std::invalid_argument("Cannot construct a non-terminal event with an empty callback");
    }

    if (d < event_direction::negative || d > event_direction::positive) [[unlikely]] {
        throw std::invalid_argument("Invalid value selected for the direction of a non-terminal event");
    }

    dir = d;
    cross_step = cstep;
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
const expression &nt_event_impl<T, B>::get_expression() const noexcept
{
    return eq;
}

template <typename T, bool B>
nt_event_impl<T, B>::callback_t &nt_event_impl<T, B>::get_callback() noexcept
{
    return callback;
}

template <typename T, bool B>
const nt_event_impl<T, B>::callback_t &nt_event_impl<T, B>::get_callback() const noexcept
{
    return callback;
}

template <typename T, bool B>
event_direction nt_event_impl<T, B>::get_direction() const noexcept
{
    return dir;
}

template <typename T, bool B>
bool nt_event_impl<T, B>::get_cross_step() const noexcept
{
    return cross_step;
}

// Implementation of stream insertion for the non-terminal event class.
template <typename T, bool B>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<T, B> &ev)
{
    os << "C++ datatype   : " << boost::core::demangle(typeid(T).name()) << '\n';
    os << "Event type     : non-terminal\n";
    os << "Event equation : " << ev.get_expression() << '\n';
    os << "Event direction: " << ev.get_direction() << '\n';
    os << "Cross step     : " << (ev.get_cross_step() ? "yes" : "no") << '\n';

    return os;
}

// Explicit instantiations.
#define HEYOKA_NT_EVENT_INST_SCALAR(F)                                                                                 \
    template class HEYOKA_DLL_PUBLIC nt_event_impl<F, false>;                                                          \
    template HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<F, false> &);

#define HEYOKA_NT_EVENT_INST_BATCH(F)                                                                                  \
    template class HEYOKA_DLL_PUBLIC nt_event_impl<F, true>;                                                           \
    template HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<F, true> &);

#define HEYOKA_NT_EVENT_INST(F)                                                                                        \
    HEYOKA_NT_EVENT_INST_SCALAR(F)                                                                                     \
    HEYOKA_NT_EVENT_INST_BATCH(F)

HEYOKA_NT_EVENT_INST(float)
HEYOKA_NT_EVENT_INST(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_NT_EVENT_INST(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_NT_EVENT_INST(mppp::real128)

#endif

HEYOKA_NT_EVENT_INST_SCALAR(mppp::real)

#undef HEYOKA_NT_EVENT_INST_SCALAR
#undef HEYOKA_NT_EVENT_INST_BATCH
#undef HEYOKA_NT_EVENT_INST

} // namespace detail

HEYOKA_END_NAMESPACE
