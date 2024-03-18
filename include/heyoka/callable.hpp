// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
struct is_any_std_func : std::false_type {
};

template <typename R, typename... Args>
struct is_any_std_func<std::function<R(Args...)>> : std::true_type {
};

template <typename T>
inline constexpr bool is_any_std_func_v = is_any_std_func<T>::value;

// Detect callable instances.
template <typename>
struct is_any_callable : std::false_type {
};

// An empty struct used in the default initialisation
// of callable objects.
// NOTE: we use this rather than, e.g., a null function
// pointer so that we can enable serialisation of
// default-constructed callables.
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS empty_callable {
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

// Default (empty) implementation of the callable interface.
template <typename, typename, typename, typename, typename...>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface_impl {
};

// Implementation of the callable interface for invocable objects.
template <typename Base, typename Holder, typename T, typename R, typename... Args>
    requires std::is_invocable_r_v<R, std::remove_reference_t<std::unwrap_reference_t<T>> &, Args...>
                 // NOTE: also require copy constructibility like
                 // std::function does.
                 && std::copy_constructible<T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface_impl<Base, Holder, T, R, Args...>
    : public Base, tanuki::iface_impl_helper<Base, Holder> {
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
    R operator()(Args... args) final
    {
        using unrefT = std::remove_reference_t<std::unwrap_reference_t<T>>;

        // Check if this is empty before invoking the call operator.
        // NOTE: no check needed here for std::function or callable: in case
        // of an empty object, the std::bad_function_call exception will be
        // thrown by the call operator of the object.
        if constexpr (std::is_pointer_v<unrefT> || std::is_member_pointer_v<unrefT>) {
            if (this->value() == nullptr) {
                throw std::bad_function_call{};
            }
        }

        if constexpr (std::is_same_v<R, void>) {
            static_cast<void>(std::invoke(this->value(), std::forward<Args>(args)...));
        } else {
            return std::invoke(this->value(), std::forward<Args>(args)...);
        }
    }
};

// Implementation of the callable interface for the empty callable.
template <typename Base, typename Holder, typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface_impl<Base, Holder, empty_callable, R, Args...> : public Base {
    // NOTE: the empty callable is always empty and always results
    // in an exception being thrown if called.
    explicit operator bool() const noexcept final
    {
        return false;
    }
    [[noreturn]] R operator()(Args...) final
    {
        throw std::bad_function_call{};
    }
};

// Definition of the callable interface.
template <typename R, typename... Args>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_iface {
    virtual ~callable_iface() = default;
    virtual R operator()(Args... args) = 0;
    virtual explicit operator bool() const noexcept = 0;

    template <typename Base, typename Holder, typename T>
    using impl = callable_iface_impl<Base, Holder, T, R, Args...>;
};

// Implementation of the reference interface.
template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_ref_iface {
    template <typename Wrap>
    struct impl {
        using result_type = R;

        template <typename JustWrap = Wrap, typename... FArgs>
        auto operator()(FArgs &&...fargs)
            -> decltype(iface_ptr(*static_cast<JustWrap *>(this))->operator()(std::forward<FArgs>(fargs)...))
        {
            // NOTE: a wrap in invalid state is considered empty.
            if (is_invalid(*static_cast<Wrap *>(this))) {
                throw std::bad_function_call{};
            }

            return iface_ptr(*static_cast<Wrap *>(this))->operator()(std::forward<FArgs>(fargs)...);
        }

        explicit operator bool() const noexcept
        {
            // NOTE: a wrap in invalid state is considered empty.
            if (is_invalid(*static_cast<const Wrap *>(this))) {
                return false;
            } else {
                return static_cast<bool>(*iface_ptr(*static_cast<const Wrap *>(this)));
            }
        }
    };
};

// Configuration of the callable wrap.
template <typename R, typename... Args>
inline constexpr auto callable_wrap_config = tanuki::config<empty_callable, callable_ref_iface<R, Args...>>{
    // Similarly to std::function, ensure that callable can store
    // in static storage pointers and reference wrappers.
    // NOTE: reference wrappers are not guaranteed to have the size
    // of a pointer, but in practice that should always be the case.
    // In case this is a concern, static asserts can be added
    // in the callable interface implementation.
    .static_size = tanuki::holder_size<R (*)(Args...), callable_iface<R, Args...>>,
    .pointer_interface = false,
    .explicit_ctor = tanuki::wrap_ctor::always_implicit};

// Definition of the callable wrap.
template <typename R, typename... Args>
using callable_wrap_t = tanuki::wrap<callable_iface<R, Args...>, callable_wrap_config<R, Args...>>;

// Specialise is_any_callable to detect callables.
template <typename R, typename... Args>
struct is_any_callable<detail::callable_wrap_t<R, Args...>> : std::true_type {
};

template <typename T>
struct callable_impl {
};

template <typename R, typename... Args>
struct callable_impl<R(Args...)> {
    using type = callable_wrap_t<R, Args...>;
};

} // namespace detail

template <typename T>
    requires(requires() { typename detail::callable_impl<T>::type; })
using callable = typename detail::callable_impl<T>::type;

HEYOKA_END_NAMESPACE

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

// Serialisation macros.
// NOTE: by default, we build a custom name and pass it to TANUKI_S11N_WRAP_EXPORT_KEY2.
// This allows us to reduce the size of the final guid wrt to what TANUKI_S11N_WRAP_EXPORT_KEY
// would synthesise, and thus to ameliorate the "class name too long" issue.
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
