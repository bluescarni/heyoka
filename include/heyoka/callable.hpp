// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_CALLABLE_HPP
#define HEYOKA_CALLABLE_HPP

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/tanuki.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Machinery to detect std::function.
template <typename>
inline constexpr bool is_any_std_func = false;

template <typename R, typename... Args>
inline constexpr bool is_any_std_func<std::function<R(Args...)>> = true;

// Detect callable instances.
template <typename>
inline constexpr bool is_any_callable = false;

// An empty struct used in the default initialisation of callable objects.
//
// NOTE: we use this rather than, e.g., a null function pointer so that we can enable serialisation of
// default-constructed callables.
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS empty_callable {
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

// Base interface for callable objects.
//
// The base interface contains the bool conversion operator.
//
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions,cppcoreguidelines-virtual-class-destructor)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS base_callable_iface {
    virtual explicit operator bool() const noexcept = 0;

    // Default implementation.
    template <typename Base, typename Holder, typename T>
    struct impl : public Base {
        explicit operator bool() const noexcept final
        {
            using unrefT = std::remove_reference_t<std::unwrap_reference_t<T>>;

            if constexpr (std::is_pointer_v<unrefT> || std::is_member_pointer_v<unrefT>) {
                return getval<Holder>(this) != nullptr;
            } else if constexpr (is_any_callable<unrefT> || is_any_std_func<unrefT>) {
                return static_cast<bool>(getval<Holder>(this));
            } else {
                return true;
            }
        }
    };

    // Implementation for empty_callable.
    template <typename Base, typename Holder>
    struct impl<Base, Holder, empty_callable> : public Base {
        // NOTE: empty_callable is always empty.
        explicit operator bool() const noexcept final
        {
            return false;
        }
    };
};

// The two interfaces for const and mutable callable objects.
template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS const_callable_iface : base_callable_iface {
    virtual R operator()(Args... args) const = 0;
};

template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS mutable_callable_iface : base_callable_iface {
    virtual R operator()(Args... args) = 0;
};

// Implementation of the call operator for the callable interface. We need both a const and a mutable variant with
// identical code, so we move the implementation outside.
template <typename Holder, typename T, typename R, typename Impl, typename... Args>
R callable_call_operator(Impl &self, Args &&...args)
{
    using unrefT = std::remove_reference_t<std::unwrap_reference_t<T>>;

    // Check if this is empty before invoking the call operator.
    //
    // NOTE: no check needed here for std::function or callable: in case of an empty object, the
    // std::bad_function_call exception will be thrown by the call operator of the object.
    if constexpr (std::is_pointer_v<unrefT> || std::is_member_pointer_v<unrefT>) {
        if (getval<Holder>(self) == nullptr) [[unlikely]] {
            throw std::bad_function_call{};
        }
    }

    if constexpr (std::is_same_v<R, void>) {
        static_cast<void>(std::invoke(getval<Holder>(self), std::forward<Args>(args)...));
    } else {
        return std::invoke(getval<Holder>(self), std::forward<Args>(args)...);
    }
}

// Definition of the callable interface.
//
// This inherits from either const_callable_iface or mutable_callable_iface, depending on the Const flag.
template <bool Const, typename R, typename... Args>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface
    : std::conditional_t<Const, const_callable_iface<R, Args...>, mutable_callable_iface<R, Args...>> {
    // Default (empty) implementation.
    template <typename, typename, typename>
    struct impl {
    };

    // Implementation for mutable invocable objects.
    template <typename Base, typename Holder, typename T>
        requires(!Const)
                && std::is_invocable_r_v<R, std::remove_reference_t<std::unwrap_reference_t<T>> &, Args...>
                // NOTE: also require copy constructibility like std::function does.
                && std::copy_constructible<T>
    struct impl<Base, Holder, T> : base_callable_iface::impl<Base, Holder, T> {
        R operator()(Args... args) final
        {
            return callable_call_operator<Holder, T, R>(*this, std::forward<Args>(args)...);
        }
    };

    // Implementation for const invocable objects.
    template <typename Base, typename Holder, typename T>
        requires Const && std::is_invocable_r_v<R, const std::remove_reference_t<std::unwrap_reference_t<T>> &, Args...>
                 && std::copy_constructible<T>
    struct impl<Base, Holder, T> : base_callable_iface::impl<Base, Holder, T> {
        R operator()(Args... args) const final
        {
            return callable_call_operator<Holder, T, R>(*this, std::forward<Args>(args)...);
        }
    };

    // Implementation for empty_callable (mutable).
    template <typename Base, typename Holder>
        requires(!Const)
    struct impl<Base, Holder, empty_callable> : base_callable_iface::impl<Base, Holder, empty_callable> {
        // NOTE: the empty callable always results in an exception if called.
        [[noreturn]] R operator()(Args...) final
        {
            throw std::bad_function_call{};
        }
    };

    // Implementation for empty_callable (const).
    template <typename Base, typename Holder>
        requires Const
    struct impl<Base, Holder, empty_callable> : base_callable_iface::impl<Base, Holder, empty_callable> {
        [[noreturn]] R operator()(Args...) const final
        {
            throw std::bad_function_call{};
        }
    };
};

// Implementation of the reference interface.
template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_ref_iface {
    // NOTE: thanks to the "deducing this" feature, we need just one implementation of the call operator, which works
    // for both the const and mutable variants.
    template <typename Wrap>
    R operator()(this Wrap &&self, Args... args)
    {
        // NOTE: a wrap in invalid state is considered empty.
        if (is_invalid(self)) [[unlikely]] {
            throw std::bad_function_call{};
        }

        return iface_ptr(std::forward<Wrap>(self))->operator()(std::forward<Args>(args)...);
    }

    template <typename Wrap>
    explicit operator bool(this const Wrap &self) noexcept
    {
        // NOTE: a wrap in invalid state is considered empty.
        if (is_invalid(self)) {
            return false;
        } else {
            return static_cast<bool>(*iface_ptr(self));
        }
    }
};

// Configuration of the callable wrap.
template <bool Const, typename R, typename... Args>
inline constexpr auto callable_wrap_config = tanuki::config<empty_callable, callable_ref_iface<R, Args...>>{
    // Similarly to std::function, ensure that callable can store in static storage pointers and reference wrappers.
    //
    // NOTE: reference wrappers are not guaranteed to have the size of a pointer, but in practice that should always be
    // the case. In case this is a concern, static asserts can be added in the callable interface implementation.
    .static_size = tanuki::holder_size<R (*)(Args...), callable_iface<Const, R, Args...>>,
    .pointer_interface = false,
    .explicit_ctor = tanuki::wrap_ctor::always_implicit};

// Definition of the callable wrap.
template <bool Const, typename R, typename... Args>
using callable_wrap_t = tanuki::wrap<callable_iface<Const, R, Args...>, callable_wrap_config<Const, R, Args...>>;

// Specialise is_any_callable to detect callables.
template <bool Const, typename R, typename... Args>
inline constexpr bool is_any_callable<detail::callable_wrap_t<Const, R, Args...>> = true;

// Helper to select the const or mutable callable wrap variant.
template <typename T>
struct callable_impl_selector {
};

template <typename R, typename... Args>
struct callable_impl_selector<R(Args...)> {
    using type = callable_wrap_t<false, R, Args...>;
};

template <typename R, typename... Args>
struct callable_impl_selector<R(Args...) const> {
    using type = callable_wrap_t<true, R, Args...>;
};

} // namespace detail

template <typename T>
    requires(requires() { typename detail::callable_impl_selector<T>::type; })
using callable = typename detail::callable_impl_selector<T>::type;

HEYOKA_END_NAMESPACE

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

// Serialisation macros.
//
// NOTE: by default, we build a custom name and pass it to TANUKI_S11N_WRAP_EXPORT_KEY2. This allows us to reduce the
// size of the final guid wrt to what TANUKI_S11N_WRAP_EXPORT_KEY would synthesise, and thus to ameliorate the "class
// name too long" issue.
#define HEYOKA_S11N_CALLABLE_EXPORT_KEY(udc, ...)                                                                      \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, "heyoka::callable<" #__VA_ARGS__ ">@" #udc,                                      \
                                 heyoka::detail::callable_iface<__VA_ARGS__>)

#define HEYOKA_S11N_CALLABLE_EXPORT_KEY2(udc, gid, ...)                                                                \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, gid, heyoka::detail::callable_iface<__VA_ARGS__>)

#define HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(udc, ...)                                                                \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(udc, heyoka::detail::callable_iface<__VA_ARGS__>)

#define HEYOKA_S11N_CALLABLE_EXPORT(udc, ...)                                                                          \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(udc, __VA_ARGS__)                                                                  \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(udc, __VA_ARGS__)

#define HEYOKA_S11N_CALLABLE_EXPORT2(udc, gid, ...)                                                                    \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY2(udc, gid, __VA_ARGS__)                                                            \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(udc, __VA_ARGS__)

#endif
