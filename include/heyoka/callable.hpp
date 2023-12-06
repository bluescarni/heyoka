// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

// Fwd declaration of the detector
// for any callable object.
template <typename>
struct is_any_callable;

namespace detail
{

// Declaration of the callable interface template.
template <typename, typename, typename, typename...>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface {
};

// Declaration of the callable interface.
template <typename R, typename... Args>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface<void, void, R, Args...> {
    virtual ~callable_iface() = default;
    virtual R operator()(Args... args) = 0;
    virtual explicit operator bool() const noexcept = 0;
};

// Implementation of the callable interface for
// invocable objects.
template <typename Holder, typename T, typename R, typename... Args>
    requires std::is_invocable_r_v<R, std::remove_reference_t<std::unwrap_reference_t<T>> &, Args...>
                 // NOTE: also require copy constructability like
                 // std::function does.
                 && std::copy_constructible<T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface<Holder, T, R, Args...>
    : virtual callable_iface<void, void, R, Args...>, tanuki::iface_impl_helper<Holder, T, callable_iface, R, Args...> {
    R operator()(Args... args) final
    {
        using unrefT = std::remove_reference_t<std::unwrap_reference_t<T>>;

        if constexpr (std::is_pointer_v<unrefT> || std::is_member_pointer_v<unrefT>) {
            if (this->value() == nullptr) {
                throw std::bad_function_call{};
            }
        }

        // NOTE: if this->value() is an empty std::function or callable,
        // the std::bad_function_call exception will be raised
        // by the invocation.

        if constexpr (std::is_same_v<R, void>) {
            static_cast<void>(std::invoke(this->value(), std::forward<Args>(args)...));
        } else {
            return std::invoke(this->value(), std::forward<Args>(args)...);
        }
    }
    explicit operator bool() const noexcept final
    {
        using unrefT = std::remove_reference_t<std::unwrap_reference_t<T>>;

        if constexpr (std::is_pointer_v<unrefT> || std::is_member_pointer_v<unrefT>) {
            return this->value() != nullptr;
        } else if constexpr (is_any_callable<unrefT>::value || is_any_std_func_v<unrefT>) {
            return static_cast<bool>(this->value());
        } else {
            return true;
        }
    }
};

// Implementation of the reference interface.
template <typename Wrap, typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_ref_iface_impl {
    using result_type = R;

    template <typename JustWrap = Wrap, typename... FArgs>
    auto operator()(FArgs &&...fargs)
        -> decltype(iface_ptr(*static_cast<JustWrap *>(this))->operator()(std::forward<FArgs>(fargs)...))
    {
        if (is_invalid(*static_cast<Wrap *>(this))) {
            throw std::bad_function_call{};
        }

        return iface_ptr(*static_cast<Wrap *>(this))->operator()(std::forward<FArgs>(fargs)...);
    }

    explicit operator bool() const noexcept
    {
        if (is_invalid(*static_cast<const Wrap *>(this))) {
            return false;
        } else {
            return static_cast<bool>(*iface_ptr(*static_cast<const Wrap *>(this)));
        }
    }

    // NOTE: these are part of the old callable interface, and they are not
    // strictly needed as there are equivalent functions in tanuki. Consider removing
    // them in the future.
    auto get_type_index() const noexcept
    {
        return value_type_index(*static_cast<const Wrap *>(this));
    }
    template <typename T>
    T *extract() noexcept
    {
        return value_ptr<T>(*static_cast<Wrap *>(this));
    }
    template <typename T>
    const T *extract() const noexcept
    {
        return value_ptr<T>(*static_cast<const Wrap *>(this));
    }
};

template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_ref_iface {
    template <typename Wrap>
    using type = callable_ref_iface_impl<Wrap, R, Args...>;
};

// Definition of the callable wrap.
template <typename R, typename... Args>
using callable_wrap_t = tanuki::wrap<callable_iface,
                                     tanuki::config<R (*)(Args...), callable_ref_iface<R, Args...>::template type>{
                                         // Similarly to std::function, ensure that callable can store
                                         // in static storage pointers and reference wrappers.
                                         // NOTE: reference wrappers are not guaranteed to have the size
                                         // of a pointer, but in practice that should always be the case.
                                         // In case this is a concern, static asserts can be added
                                         // in the callable interface implementation.
                                         .static_size = tanuki::holder_size<R (*)(Args...), callable_iface, R, Args...>,
                                         .pointer_interface = false,
                                         .explicit_generic_ctor = false},
                                     R, Args...>;

template <typename T>
struct callable_impl {
    static_assert(always_false_v<T>);
};

template <typename R, typename... Args>
struct callable_impl<R(Args...)> {
    using type = callable_wrap_t<R, Args...>;
};

} // namespace detail

template <typename T>
using callable = typename detail::callable_impl<T>::type;

// Detect callable instances.
template <typename>
struct is_any_callable : std::false_type {
};

template <typename R, typename... Args>
struct is_any_callable<detail::callable_wrap_t<R, Args...>> : std::true_type {
};

HEYOKA_END_NAMESPACE

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

// Serialisation macros.
#define HEYOKA_S11N_CALLABLE_EXPORT_KEY(udc, ...)                                                                      \
    TANUKI_S11N_WRAP_EXPORT_KEY(udc, heyoka::detail::callable_iface, __VA_ARGS__)

#define HEYOKA_S11N_CALLABLE_EXPORT_KEY2(udc, gid, ...)                                                                \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, gid, heyoka::detail::callable_iface, __VA_ARGS__)

#define HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(udc, ...)                                                                \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(udc, heyoka::detail::callable_iface, __VA_ARGS__)

#define HEYOKA_S11N_CALLABLE_EXPORT(udc, ...)                                                                          \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(udc, __VA_ARGS__)                                                                  \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(udc, __VA_ARGS__)

#define HEYOKA_S11N_CALLABLE_EXPORT2(udc, gid, ...)                                                                    \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY2(udc, gid, __VA_ARGS__)                                                            \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(udc, __VA_ARGS__)

#endif
