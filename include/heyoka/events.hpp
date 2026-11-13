// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EVENTS_HPP
#define HEYOKA_EVENTS_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>

#include <fmt/core.h>
#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/make_optional.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/supported_fp_type.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Common descriptor used in the validation of keyword arguments for both terminal and non-terminal events.
inline constexpr auto event_direction_descr = kw::descr::same_as<kw::direction, event_direction>;

// Metaprogramming for selecting the callback type for t_event_impl.
template <typename T, bool B>
struct t_event_impl_callback_t {
    using type = bool(taylor_adaptive_batch<T> &, int, std::uint32_t);
};

template <typename T>
struct t_event_impl_callback_t<T, false> {
    using type = bool(taylor_adaptive<T> &, int);
};

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS t_event_impl
{
public:
    using callback_t = callable<typename t_event_impl_callback_t<T, B>::type>;

private:
    expression eq;
    callback_t callback;
    std::optional<T> cooldown;
    event_direction dir = event_direction::any;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // Private implementation-detail ctor.
    struct ptag {};
    explicit t_event_impl(ptag, std::tuple<expression, callback_t, std::optional<T>, event_direction>);

    // Parser for the kwargs.
    template <typename... KwArgs>
    static auto parse_ctor_args(expression e, const KwArgs &...kw_args)
    {
        const igor::parser p{kw_args...};

        // Callback (defaults to empty).
        callback_t cb = p(kw::callback, callback_t{});

        // Cooldown (defaults to not-provided).
        auto cd_opt = [&p] -> std::optional<T> {
            if constexpr (p.has(kw::cooldown)) {
                return make_optional<T>(p(kw::cooldown));
            } else {
                return {};
            }
        }();

        // Direction (defaults to any).
        const auto d = p(kw::direction, event_direction::any);

        return std::make_tuple(std::move(e), std::move(cb), std::move(cd_opt), d);
    }

    // Configuration for the keyword arguments.
    static constexpr auto kw_cfg = igor::config<kw::descr::convertible_to<kw::callback, callback_t>,
                                                kw::descr::optional_from<T, kw::cooldown>, event_direction_descr>{};

public:
    t_event_impl();

    template <typename... KwArgs>
        requires igor::validate<kw_cfg, KwArgs...>
    // NOTE: accept forwarding references here to highlight that kw_args may in general be moved and that thus it is not
    // safe to re-use them.
    explicit t_event_impl(expression e, KwArgs &&...kw_args)
        : t_event_impl(ptag{}, parse_ctor_args(std::move(e), std::forward<KwArgs>(kw_args)...))
    {
    }

    t_event_impl(const t_event_impl &);
    t_event_impl(t_event_impl &&) noexcept;

    t_event_impl &operator=(const t_event_impl &);
    t_event_impl &operator=(t_event_impl &&) noexcept;

    ~t_event_impl();

    [[nodiscard]] const expression &get_expression() const noexcept;
    [[nodiscard]] callback_t &get_callback() noexcept;
    [[nodiscard]] const callback_t &get_callback() const noexcept;
    [[nodiscard]] event_direction get_direction() const noexcept;
    [[nodiscard]] const std::optional<T> &get_cooldown() const noexcept;
};

// Metaprogramming for selecting the callback type for nt_event_impl.
template <typename T, bool B>
struct nt_event_impl_callback_t {
    using type = void(taylor_adaptive_batch<T> &, T, int, std::uint32_t);
};

template <typename T>
struct nt_event_impl_callback_t<T, false> {
    using type = void(taylor_adaptive<T> &, T, int);
};

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS nt_event_impl
{
public:
    using callback_t = callable<typename nt_event_impl_callback_t<T, B>::type>;

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

    // Configuration for the keyword arguments.
    static constexpr auto kw_cfg = igor::config<event_direction_descr>{};

public:
    nt_event_impl();

    template <typename... KwArgs>
        requires igor::validate<kw_cfg, KwArgs...>
    explicit nt_event_impl(expression e, callback_t cb, const KwArgs &...kw_args)
        : eq(std::move(e)), callback(std::move(cb))
    {
        const igor::parser p{kw_args...};

        // Direction (defaults to any).
        const auto d = p(kw::direction, event_direction::any);

        finalise_ctor(d);
    }

    nt_event_impl(const nt_event_impl &);
    nt_event_impl(nt_event_impl &&) noexcept;

    nt_event_impl &operator=(const nt_event_impl &);
    nt_event_impl &operator=(nt_event_impl &&) noexcept;

    ~nt_event_impl();

    [[nodiscard]] const expression &get_expression() const noexcept;
    [[nodiscard]] callback_t &get_callback() noexcept;
    [[nodiscard]] const callback_t &get_callback() const noexcept;
    [[nodiscard]] event_direction get_direction() const noexcept;
};

template <typename T, bool B>
std::ostream &operator<<(std::ostream &, const nt_event_impl<T, B> &);

template <typename T, bool B>
std::ostream &operator<<(std::ostream &, const t_event_impl<T, B> &);

// Boost s11n class version history for t_event_impl:
//
// - 1: the cooldown is now stored in a std::optional.
inline constexpr int t_event_impl_s11n_version = 1;

} // namespace detail

template <supported_fp_type T>
using nt_event = detail::nt_event_impl<T, false>;

template <supported_fp_type T>
using t_event = detail::t_event_impl<T, false>;

template <supported_batch_fp_type T>
using nt_event_batch = detail::nt_event_impl<T, true>;

template <supported_batch_fp_type T>
using t_event_batch = detail::t_event_impl<T, true>;

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, event_direction);

HEYOKA_END_NAMESPACE

// fmt formatter for event_direction, implemented on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::event_direction> : fmt::ostream_formatter {};

} // namespace fmt

#define HEYOKA_T_EVENT_IMPL_CLASS_VERSION(T)                                                                           \
    BOOST_CLASS_VERSION(heyoka::t_event<T>, heyoka::detail::t_event_impl_s11n_version);                                \
    BOOST_CLASS_VERSION(heyoka::t_event_batch<T>, heyoka::detail::t_event_impl_s11n_version);

HEYOKA_T_EVENT_IMPL_CLASS_VERSION(float)
HEYOKA_T_EVENT_IMPL_CLASS_VERSION(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_T_EVENT_IMPL_CLASS_VERSION(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_T_EVENT_IMPL_CLASS_VERSION(mppp::real128)

#endif

BOOST_CLASS_VERSION(heyoka::t_event<mppp::real>, heyoka::detail::t_event_impl_s11n_version);

#undef HEYOKA_T_EVENT_IMPL_CLASS_VERSION

#endif
