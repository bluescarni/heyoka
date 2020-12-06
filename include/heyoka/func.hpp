// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNC_HPP
#define HEYOKA_FUNC_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC func_base
{
    std::string m_display_name;
    std::unique_ptr<std::vector<expression>> m_args;

public:
    explicit func_base(std::string, std::vector<expression>);

    func_base(const func_base &);
    func_base(func_base &&) noexcept;

    func_base &operator=(const func_base &);
    func_base &operator=(func_base &&) noexcept;

    ~func_base();

    const std::string &get_display_name() const;
    const std::vector<expression> &get_args() const;
};

namespace detail
{

struct HEYOKA_DLL_PUBLIC func_inner_base {
    virtual ~func_inner_base();
    virtual std::unique_ptr<func_inner_base> clone() const = 0;

    virtual std::type_index get_type_index() const = 0;
    virtual const void *get_ptr() const = 0;
    virtual void *get_ptr() = 0;

    virtual const std::string &get_display_name() const = 0;
    virtual const std::vector<expression> &get_args() const = 0;

    virtual llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const = 0;
    virtual llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const = 0;
#if defined(HEYOKA_HAVE_REAL128)
    virtual llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const = 0;
#endif
};

template <typename T>
using func_codegen_dbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().codegen_dbl(
    std::declval<llvm_state &>(), std::declval<const std::vector<llvm::Value *> &>()));

template <typename T>
inline constexpr bool func_has_codegen_dbl_v = std::is_same_v<detected_t<func_codegen_dbl_t, T>, llvm::Value *>;

template <typename T>
using func_codegen_ldbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().codegen_ldbl(
    std::declval<llvm_state &>(), std::declval<const std::vector<llvm::Value *> &>()));

template <typename T>
inline constexpr bool func_has_codegen_ldbl_v = std::is_same_v<detected_t<func_codegen_ldbl_t, T>, llvm::Value *>;

#if defined(HEYOKA_HAVE_REAL128)

template <typename T>
using func_codegen_f128_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().codegen_f128(
    std::declval<llvm_state &>(), std::declval<const std::vector<llvm::Value *> &>()));

template <typename T>
inline constexpr bool func_has_codegen_f128_v = std::is_same_v<detected_t<func_codegen_f128_t, T>, llvm::Value *>;

#endif

template <typename T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_inner final : func_inner_base {
    T m_value;

    // We just need the def ctor, delete everything else.
    func_inner() = default;
    func_inner(const func_inner &) = delete;
    func_inner(func_inner &&) = delete;
    func_inner &operator=(const func_inner &) = delete;
    func_inner &operator=(func_inner &&) = delete;

    // Constructors from T (copy and move variants).
    explicit func_inner(const T &x) : m_value(x) {}
    explicit func_inner(T &&x) : m_value(std::move(x)) {}

    // The clone function.
    std::unique_ptr<func_inner_base> clone() const final
    {
        return std::make_unique<func_inner>(m_value);
    }

    // Get the type at runtime.
    std::type_index get_type_index() const final
    {
        return typeid(T);
    }
    // Raw getters for the internal instance.
    const void *get_ptr() const final
    {
        return &m_value;
    }
    void *get_ptr() final
    {
        return &m_value;
    }

    const std::string &get_display_name() const final
    {
        return static_cast<const func_base *>(&m_value)->get_display_name();
    }
    const std::vector<expression> &get_args() const final
    {
        return static_cast<const func_base *>(&m_value)->get_args();
    }

    // codegen.
    llvm::Value *codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &v) const final
    {
        if constexpr (func_has_codegen_dbl_v<T>) {
            return m_value.codegen_dbl(s, v);
        } else {
            throw not_implemented_error("double codegen is not implemented for the function '" + get_display_name()
                                        + "'");
        }
    }
    llvm::Value *codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &v) const final
    {
        if constexpr (func_has_codegen_ldbl_v<T>) {
            return m_value.codegen_ldbl(s, v);
        } else {
            throw not_implemented_error("long double codegen is not implemented for the function '" + get_display_name()
                                        + "'");
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &v) const final
    {
        if constexpr (func_has_codegen_f128_v<T>) {
            return m_value.codegen_f128(s, v);
        } else {
            throw not_implemented_error("real128 codegen is not implemented for the function '" + get_display_name()
                                        + "'");
        }
    }
#endif
};

template <typename T>
using is_func = std::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                 std::is_copy_constructible<T>, std::is_move_constructible<T>, std::is_destructible<T>,
                                 // https://en.cppreference.com/w/cpp/concepts/derived_from
                                 // NOTE: use add_pointer/add_cv in order to avoid
                                 // issues if invoked with problematic types (e.g., void).
                                 std::is_base_of<func_base, T>,
                                 std::is_convertible<std::add_pointer_t<std::add_cv_t<T>>, const volatile func_base *>>;

} // namespace detail

class HEYOKA_DLL_PUBLIC func
{
    // Pointer to the inner base.
    std::unique_ptr<detail::func_inner_base> m_ptr;

    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    const detail::func_inner_base *ptr() const;
    detail::func_inner_base *ptr();

    template <typename T>
    using generic_ctor_enabler
        = std::enable_if_t<std::conjunction_v<std::negation<std::is_same<func, detail::uncvref_t<T>>>,
                                              detail::is_func<detail::uncvref_t<T>>>,
                           int>;

public:
    template <typename T, generic_ctor_enabler<T &&> = 0>
    explicit func(T &&x) : m_ptr(std::make_unique<detail::func_inner<detail::uncvref_t<T>>>(std::forward<T>(x)))
    {
    }

    func(const func &);
    func(func &&) noexcept;

    func &operator=(const func &);
    func &operator=(func &&) noexcept;

    ~func();

    std::type_index get_type_index() const;
    const void *get_ptr() const;
    void *get_ptr();

    const std::string &get_display_name() const;
    const std::vector<expression> &get_args() const;

    llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const;
    llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const;
#endif
};

} // namespace heyoka

#endif
