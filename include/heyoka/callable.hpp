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
#include <typeindex>
#include <typeinfo>
#include <utility>

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

// NOTE: here we implement a stripped-down version of std::function
// on top of classic OOP polymorphism.
//
// The reason for this (as opposed to using std::function directly)
// is principally because we can make our implementation serialisable
// via Boost.Serialization, whereas std::function cannot be supported
// by Boost. We need a serialisable function wrapper in order to be
// able to serialise the callbacks of the events, whose serialisation,
// in turn, is needed in the serialisation of the integrator objects.

namespace heyoka
{

namespace detail
{

template <typename R, typename... Args>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS callable_inner_base {
    virtual ~callable_inner_base() {}

    virtual std::unique_ptr<callable_inner_base> clone() const = 0;

    virtual R operator()(Args...) const = 0;

    virtual std::type_index get_type_index() const = 0;

private:
    // Serialization (empty, no data members).
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
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

    std::type_index get_type_index() const final
    {
        return typeid(T);
    }

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<callable_inner_base<R, Args...>>(*this);
        ar &m_value;
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

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_ptr;
    }

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

    std::type_index get_type_index() const
    {
        if (m_ptr) {
            return m_ptr->get_type_index();
        } else {
            return typeid(void);
        }
    }

    // Extraction.
    template <typename T>
    const T *extract() const noexcept
    {
        if (!m_ptr) {
            return nullptr;
        }

        auto p = dynamic_cast<const detail::callable_inner<T, R, Args...> *>(m_ptr.get());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        if (!m_ptr) {
            return nullptr;
        }

        auto p = dynamic_cast<detail::callable_inner<T, R, Args...> *>(m_ptr.get());
        return p == nullptr ? nullptr : &(p->m_value);
    }
};

template <typename R, typename... Args>
inline void swap(callable<R(Args...)> &c0, callable<R(Args...)> &c1) noexcept
{
    c0.swap(c1);
}

} // namespace heyoka

// Disable Boost.Serialization tracking for the implementation details of callable.
// NOTE: these bits are taken verbatim from the BOOST_CLASS_TRACKING macro, which does not support
// class templates.

namespace boost
{

namespace serialization
{

template <typename R, typename... Args>
struct tracking_level<heyoka::detail::callable_inner_base<R, Args...>> {
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
    BOOST_STATIC_ASSERT((mpl::greater<implementation_level<heyoka::detail::callable_inner_base<R, Args...>>,
                                      mpl::int_<primitive_type>>::value));
};

template <typename T, typename R, typename... Args>
struct tracking_level<heyoka::detail::callable_inner<T, R, Args...>> {
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
    BOOST_STATIC_ASSERT((mpl::greater<implementation_level<heyoka::detail::callable_inner<T, R, Args...>>,
                                      mpl::int_<primitive_type>>::value));
};

} // namespace serialization

} // namespace boost

// NOTE: these are verbatim re-implementations of the BOOST_CLASS_EXPORT_KEY
// and BOOST_CLASS_EXPORT_IMPLEMENT macros, which do not work well with class templates.
#define HEYOKA_S11N_CALLABLE_EXPORT_KEY(...)                                                                           \
    namespace boost                                                                                                    \
    {                                                                                                                  \
    namespace serialization                                                                                            \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct guid_defined<heyoka::detail::callable_inner<__VA_ARGS__>> : boost::mpl::true_ {                             \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<heyoka::detail::callable_inner<__VA_ARGS__>>()                                             \
    {                                                                                                                  \
        /* NOTE: the stringize here will produce a name enclosed by brackets. */                                       \
        return BOOST_PP_STRINGIZE((heyoka::detail::callable_inner<__VA_ARGS__>));                                    \
    }                                                                                                                  \
    }                                                                                                                  \
    }

#define HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(...)                                                                     \
    namespace boost                                                                                                    \
    {                                                                                                                  \
    namespace archive                                                                                                  \
    {                                                                                                                  \
    namespace detail                                                                                                   \
    {                                                                                                                  \
    namespace extra_detail                                                                                             \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct init_guid<heyoka::detail::callable_inner<__VA_ARGS__>> {                                                    \
        static guid_initializer<heyoka::detail::callable_inner<__VA_ARGS__>> const &g;                                 \
    };                                                                                                                 \
    guid_initializer<heyoka::detail::callable_inner<__VA_ARGS__>> const                                                \
        &init_guid<heyoka::detail::callable_inner<__VA_ARGS__>>::g                                                     \
        = ::boost::serialization::singleton<                                                                           \
              guid_initializer<heyoka::detail::callable_inner<__VA_ARGS__>>>::get_mutable_instance()                   \
              .export_guid();                                                                                          \
    }                                                                                                                  \
    }                                                                                                                  \
    }                                                                                                                  \
    }

#define HEYOKA_S11N_CALLABLE_EXPORT(...)                                                                               \
    HEYOKA_S11N_CALLABLE_EXPORT_KEY(__VA_ARGS__)                                                                       \
    HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(__VA_ARGS__)

#endif
