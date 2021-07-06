// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_CALLABLE_HPP
#define HEYOKA_CALLABLE_HPP

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

namespace detail
{

template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_inner_base {
    virtual ~callable_inner_base() {}

    virtual std::unique_ptr<callable_inner_base> clone() const = 0;

    virtual R operator()(Args...) const = 0;
};

template <typename T, typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_inner final : callable_inner_base<R, Args...> {
    T m_value;

    using base = callable_inner_base<R, Args...>;

    // Constructors from T (copy and move variants).
    explicit callable_inner(const T &x) : m_value(x) {}
    explicit callable_inner(T &&x) : m_value(std::move(x)) {}

    // The clone method, used in the copy constructor.
    std::unique_ptr<base> clone() const final
    {
        return std::make_unique<callable_inner>(m_value);
    }

    R operator()(Args... args) const final
    {
        return m_value(std::forward<Args>(args)...);
    }
};

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS callable
{
    static_assert(detail::always_false_v<T>);
};

template <typename R, typename... Args>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS callable<R(Args...)>
{
    std::unique_ptr<detail::callable_inner_base<R, Args...>> m_ptr;

    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    auto ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    auto ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

    template <typename T>
    explicit callable(T &&f, std::false_type)
        : m_ptr(std::make_unique<detail::callable_inner<detail::uncvref_t<T>, R, Args...>>(std::forward<T>(f)))
    {
    }
    template <typename T>
    explicit callable(T &&f, std::true_type)
        : callable(static_cast<R (*)(Args...)>(std::forward<T>(f)), std::false_type{})
    {
    }

public:
    template <typename T, std::enable_if_t<std::negation_v<std::is_same<callable, detail::uncvref_t<T>>>, int> = 0>
    explicit callable(T &&f) : callable(std::forward<T>(f), std::is_same<R(Args...), detail::uncvref_t<T>>{})
    {
    }

    R operator()(Args... args) const
    {
        return ptr()->operator()(std::forward<Args>(args)...);
    }
};

} // namespace heyoka

#endif
