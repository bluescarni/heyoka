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
#include <functional>
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

    // We just need the def ctor, delete everything else.
    callable_inner() = default;
    callable_inner(const callable_inner &) = delete;
    callable_inner(callable_inner &&) = delete;
    callable_inner &operator=(const callable_inner &) = delete;
    callable_inner &operator=(callable_inner &&) = delete;

    // Constructors from T (copy and move variants).
    explicit callable_inner(const T &x) : m_value(x) {}
    explicit callable_inner(T &&x) : m_value(std::move(x)) {}

    // The clone method, used in the copy constructor.
    std::unique_ptr<callable_inner_base<R, Args...>> clone() const final
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

    // Dispatching of the generic constructor with specialisation
    // for construction from a function (second overload).
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
    // NOTE: default construction builds an empty callable.
    callable() = default;
    callable(const callable &other) : m_ptr(other ? other.m_ptr->clone() : nullptr) {}
    callable(callable &&) = default;

    // NOTE: generic ctor is enabled only if it does not
    // compete with copy/move ctors.
    template <typename T, std::enable_if_t<std::negation_v<std::is_same<callable, detail::uncvref_t<T>>>, int> = 0>
    callable(T &&f) : callable(std::forward<T>(f), std::is_same<R(Args...), detail::uncvref_t<T>>{})
    {
    }

    callable &operator=(const callable &other)
    {
        return *this = callable(other);
    }
    callable &operator=(callable &&) = default;

    R operator()(Args... args) const
    {
        if (!m_ptr) {
            throw std::bad_function_call();
        }

        return m_ptr->operator()(std::forward<Args>(args)...);
    }

    void swap(callable &other) noexcept
    {
        std::swap(m_ptr, other.m_ptr);
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(m_ptr);
    }
};

template <typename R, typename... Args>
inline void swap(callable<R(Args...)> &c0, callable<R(Args...)> &c1) noexcept
{
    c0.swap(c1);
}

} // namespace heyoka

#endif
