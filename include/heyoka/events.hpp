// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EVENTS_HPP
#define HEYOKA_EVENTS_HPP

#include <heyoka/config.hpp>

#include <concepts>
#include <cstdint>
#include <ostream>
#include <type_traits>
#include <utility>

#include <fmt/core.h>
#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS t_event_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using callback_t = callable<
        std::conditional_t<B, bool(taylor_adaptive_batch<T> &, int, std::uint32_t), bool(taylor_adaptive<T> &, int)>>;

private:
    expression eq;
    callback_t callback;
    T cooldown = 0;
    event_direction dir = event_direction::any;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void finalise_ctor(callback_t, T, event_direction);

public:
    t_event_impl();

    template <typename... KwArgs>
    explicit t_event_impl(expression e, const KwArgs &...kw_args) : eq(std::move(e))
    {
        igor::parser p{kw_args...};

        static_assert(!p.has_unnamed_arguments(),
                      "The variadic arguments in the construction of a terminal event contain "
                      "unnamed arguments.");

        // Callback (defaults to empty).
        auto cb = [&p]() -> callback_t {
            if constexpr (p.has(kw::callback)) {
                if constexpr (std::convertible_to<decltype(p(kw::callback)), callback_t>) {
                    return p(kw::callback);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'callback' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        // Cooldown (defaults to -1).
        auto cd = [&p]() -> T {
            if constexpr (p.has(kw::cooldown)) {
                if constexpr (std::convertible_to<decltype(p(kw::cooldown)), T>) {
                    return p(kw::cooldown);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'cooldown' keyword argument.");
                }
            } else {
                return T(-1);
            }
        }();

        // Direction (defaults to any).
        auto d = [&p]() -> event_direction {
            if constexpr (p.has(kw::direction)) {
                if constexpr (std::convertible_to<decltype(p(kw::direction)), event_direction>) {
                    return p(kw::direction);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'direction' keyword argument.");
                }
            } else {
                return event_direction::any;
            }
        }();

        finalise_ctor(std::move(cb), cd, d);
    }

    t_event_impl(const t_event_impl &);
    t_event_impl(t_event_impl &&) noexcept;

    t_event_impl &operator=(const t_event_impl &);
    t_event_impl &operator=(t_event_impl &&) noexcept;

    ~t_event_impl();

    [[nodiscard]] const expression &get_expression() const;
    callback_t &get_callback();
    [[nodiscard]] const callback_t &get_callback() const;
    [[nodiscard]] event_direction get_direction() const;
    [[nodiscard]] T get_cooldown() const;
};

// Prevent implicit instantiations.
#define HEYOKA_T_EVENT_EXTERN_INST(F)                                                                                  \
    extern template class t_event_impl<F, true>;                                                                       \
    extern template class t_event_impl<F, false>;

HEYOKA_T_EVENT_EXTERN_INST(float)
HEYOKA_T_EVENT_EXTERN_INST(double)
HEYOKA_T_EVENT_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_T_EVENT_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_T_EVENT_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_T_EVENT_EXTERN_INST

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS nt_event_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using callback_t = callable<std::conditional_t<B, void(taylor_adaptive_batch<T> &, T, int, std::uint32_t),
                                                   void(taylor_adaptive<T> &, T, int)>>;

private:
    expression eq;
    callback_t callback;
    event_direction dir = event_direction::any;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void finalise_ctor(event_direction);

public:
    nt_event_impl();

    template <typename... KwArgs>
    explicit nt_event_impl(expression e, callback_t cb, const KwArgs &...kw_args)
        : eq(std::move(e)), callback(std::move(cb))
    {
        igor::parser p{kw_args...};

        static_assert(!p.has_unnamed_arguments(),
                      "The variadic arguments in the construction of a non-terminal event contain "
                      "unnamed arguments.");

        // Direction (defaults to any).
        auto d = [&p]() -> event_direction {
            if constexpr (p.has(kw::direction)) {
                if constexpr (std::convertible_to<decltype(p(kw::direction)), event_direction>) {
                    return p(kw::direction);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'direction' keyword argument.");
                }
            } else {
                return event_direction::any;
            }
        }();

        finalise_ctor(d);
    }

    nt_event_impl(const nt_event_impl &);
    nt_event_impl(nt_event_impl &&) noexcept;

    nt_event_impl &operator=(const nt_event_impl &);
    nt_event_impl &operator=(nt_event_impl &&) noexcept;

    ~nt_event_impl();

    [[nodiscard]] const expression &get_expression() const;
    callback_t &get_callback();
    [[nodiscard]] const callback_t &get_callback() const;
    [[nodiscard]] event_direction get_direction() const;
};

// Prevent implicit instantiations.
#define HEYOKA_NT_EVENT_EXTERN_INST(F)                                                                                 \
    extern template class nt_event_impl<F, true>;                                                                      \
    extern template class nt_event_impl<F, false>;

HEYOKA_NT_EVENT_EXTERN_INST(float)
HEYOKA_NT_EVENT_EXTERN_INST(double)
HEYOKA_NT_EVENT_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_NT_EVENT_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_NT_EVENT_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_NT_EVENT_EXTERN_INST

template <typename T, bool B>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<T, B> &)
{
    static_assert(always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<float, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<float, true> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<double, true> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<long double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<long double, true> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<mppp::real128, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<mppp::real128, true> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<mppp::real, false> &);

#endif

template <typename T, bool B>
std::ostream &operator<<(std::ostream &os, const t_event_impl<T, B> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<float, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<float, true> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<double, true> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<long double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<long double, true> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<mppp::real128, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<mppp::real128, true> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<mppp::real, false> &);

#endif

} // namespace detail

template <typename T>
using nt_event = detail::nt_event_impl<T, false>;

template <typename T>
using t_event = detail::t_event_impl<T, false>;

template <typename T>
using nt_event_batch = detail::nt_event_impl<T, true>;

template <typename T>
using t_event_batch = detail::t_event_impl<T, true>;

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, event_direction);

HEYOKA_END_NAMESPACE

// fmt formatter for event_direction, implemented on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::event_direction> : fmt::ostream_formatter {
};

} // namespace fmt

// Export the s11n keys for default-constructed event callbacks.
#define HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY(T)                                                                      \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::detail::empty_callable, void, heyoka::taylor_adaptive<T> &, T, int)        \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::detail::empty_callable, bool, heyoka::taylor_adaptive<T> &, int)

#define HEYOKA_S11N_BATCH_EVENT_CALLBACKS_EXPORT_KEY(T)                                                                \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::detail::empty_callable, void, heyoka::taylor_adaptive_batch<T> &, T, int,  \
                                    std::uint32_t)                                                                     \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::detail::empty_callable, bool, heyoka::taylor_adaptive_batch<T> &, int,     \
                                    std::uint32_t)

HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY(float)
HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY(double)
HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY(long double)

HEYOKA_S11N_BATCH_EVENT_CALLBACKS_EXPORT_KEY(float)
HEYOKA_S11N_BATCH_EVENT_CALLBACKS_EXPORT_KEY(double)
HEYOKA_S11N_BATCH_EVENT_CALLBACKS_EXPORT_KEY(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY(mppp::real128)
HEYOKA_S11N_BATCH_EVENT_CALLBACKS_EXPORT_KEY(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY(mppp::real)

#endif

#undef HEYOKA_S11N_EVENT_CALLBACKS_EXPORT_KEY
#undef HEYOKA_S11N_BATCH_EVENT_CALLBACKS_EXPORT_KEY

#endif
