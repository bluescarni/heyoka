// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNC_HPP
#define HEYOKA_FUNC_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/tanuki.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/s11n.hpp>

// Current archive version is 1.
BOOST_CLASS_VERSION(heyoka::func, 1)

HEYOKA_BEGIN_NAMESPACE

class HEYOKA_DLL_PUBLIC func_base
{
    std::string m_name;
    std::vector<expression> m_args;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_name;
        ar & m_args;
    }

public:
    explicit func_base(std::string, std::vector<expression>);

    func_base(const func_base &);
    func_base(func_base &&) noexcept;

    func_base &operator=(const func_base &);
    func_base &operator=(func_base &&) noexcept;

    ~func_base();

    [[nodiscard]] const std::string &get_name() const noexcept;
    [[nodiscard]] const std::vector<expression> &args() const noexcept;

    // NOTE: this is supposed to be private, but there are issues making friends
    // with concept constraints on clang. Leave it public and undocumented for now.
    std::pair<expression *, expression *> get_mutable_args_range();
};

// UDF concept.
template <typename T>
concept is_udf
    = std::default_initializable<T> && std::movable<T> && std::copyable<T> && std::derived_from<T, func_base>;

namespace detail
{

// Fwd declaration of the function interface implementation.
template <typename, typename, is_udf T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_iface_impl;

HEYOKA_DLL_PUBLIC void func_default_to_stream_impl(std::ostringstream &, const func_base &);

// The function interface.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct HEYOKA_DLL_PUBLIC func_iface {
    virtual ~func_iface();

    [[nodiscard]] virtual const std::string &get_name() const = 0;

    virtual void to_stream(std::ostringstream &) const = 0;

    [[nodiscard]] virtual bool is_time_dependent() const = 0;

    [[nodiscard]] virtual bool has_normalise() const = 0;
    [[nodiscard]] virtual expression normalise() const = 0;

    [[nodiscard]] virtual const std::vector<expression> &args() const = 0;
    virtual std::pair<expression *, expression *> get_mutable_args_range() = 0;

    [[nodiscard]] virtual bool has_diff_var() const = 0;
    virtual expression diff(funcptr_map<expression> &, const std::string &) const = 0;
    [[nodiscard]] virtual bool has_diff_par() const = 0;
    virtual expression diff(funcptr_map<expression> &, const param &) const = 0;
    [[nodiscard]] virtual bool has_gradient() const = 0;
    [[nodiscard]] virtual std::vector<expression> gradient() const = 0;
    [[nodiscard]] std::vector<expression> fetch_gradient(const std::string &) const;

    [[nodiscard]] virtual llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &,
                                                 llvm::Value *, llvm::Value *, llvm::Value *, std::uint32_t, bool) const
        = 0;

    [[nodiscard]] virtual llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const = 0;

    // NOTE: this is the last remaining trace of mutability
    // related to decomposition. Note, however, that this is never
    // exposed in the public API.
    virtual taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) && = 0;
    [[nodiscard]] virtual bool has_taylor_decompose() const = 0;

    virtual llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                     const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t, bool) const
        = 0;

    virtual llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const
        = 0;

    template <typename Base, typename Holder, typename T>
    using impl = func_iface_impl<Base, Holder, T>;
};

// The udf used in the default construction of a func.
struct HEYOKA_DLL_PUBLIC null_func : func_base {
    null_func();
};

} // namespace detail

namespace detail
{

template <typename T>
concept func_has_normalise = requires(const T &x) {
    {
        x.normalise()
    } -> std::same_as<expression>;
};

template <typename T>
concept func_has_diff_var = requires(const T &x, funcptr_map<expression> &m, const std::string &name) {
    {
        x.diff(m, name)
    } -> std::same_as<expression>;
};

template <typename T>
concept func_has_diff_par = requires(const T &x, funcptr_map<expression> &m, const param &p) {
    {
        x.diff(m, p)
    } -> std::same_as<expression>;
};

template <typename T>
concept func_has_gradient = requires(const T &x) {
    {
        x.gradient()
    } -> std::same_as<std::vector<expression>>;
};

template <typename T>
concept func_has_taylor_decompose = requires(T &&x, taylor_dc_t &dc) {
    {
        std::move(x).taylor_decompose(dc)
    } -> std::same_as<taylor_dc_t::size_type>;
};

// Function interface implementation.
template <typename Base, typename Holder, is_udf T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_iface_impl : public Base, tanuki::iface_impl_helper<Base, Holder> {
    [[nodiscard]] const std::string &get_name() const final
    {
        // NOTE: make sure we are invoking the member functions
        // from func_base (these functions could have been overriden
        // in the derived class).
        return static_cast<const func_base &>(this->value()).get_name();
    }

    void to_stream(std::ostringstream &oss) const final
    {
        if constexpr (requires() { static_cast<void>(this->value().to_stream(oss)); }) {
            static_cast<void>(this->value().to_stream(oss));
        } else {
            func_default_to_stream_impl(oss, static_cast<const func_base &>(this->value()));
        }
    }

    [[nodiscard]] bool is_time_dependent() const final
    {
        if constexpr (requires() { static_cast<bool>(this->value().is_time_dependent()); }) {
            return static_cast<bool>(this->value().is_time_dependent());
        } else {
            return false;
        }
    }

    [[nodiscard]] bool has_normalise() const final
    {
        return func_has_normalise<T>;
    }
    [[nodiscard]] expression normalise() const final;

    [[nodiscard]] const std::vector<expression> &args() const final
    {
        return static_cast<const func_base &>(this->value()).args();
    }
    std::pair<expression *, expression *> get_mutable_args_range() final
    {
        return static_cast<func_base &>(this->value()).get_mutable_args_range();
    }

    // diff.
    [[nodiscard]] bool has_diff_var() const final
    {
        return func_has_diff_var<T>;
    }
    expression diff(funcptr_map<expression> &, const std::string &) const final;
    [[nodiscard]] bool has_diff_par() const final
    {
        return func_has_diff_par<T>;
    }
    expression diff(funcptr_map<expression> &, const param &) const final;

    // gradient.
    [[nodiscard]] bool has_gradient() const final
    {
        return func_has_gradient<T>;
    }
    [[nodiscard]] std::vector<expression> gradient() const final
    {
        if constexpr (func_has_gradient<T>) {
            return this->value().gradient();
        }

        // LCOV_EXCL_START
        assert(false);
        throw;
        // LCOV_EXCL_STOP
    }

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                         llvm::Value *par_ptr, llvm::Value *time_ptr, llvm::Value *stride,
                                         std::uint32_t batch_size, bool high_accuracy) const final
    {
        if constexpr (requires() {
                          {
                              this->value().llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size,
                                                      high_accuracy)
                          } -> std::same_as<llvm::Value *>;
                      }) {
            return this->value().llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size, high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("llvm_eval() is not implemented for the function '{}'", get_name()));
        }
    }

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                                   bool high_accuracy) const final
    {
        if constexpr (requires() {
                          {
                              this->value().llvm_c_eval_func(s, fp_t, batch_size, high_accuracy)
                          } -> std::same_as<llvm::Function *>;
                      }) {
            return this->value().llvm_c_eval_func(s, fp_t, batch_size, high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("llvm_c_eval_func() is not implemented for the function '{}'", get_name()));
        }
    }

    // Taylor.
    taylor_dc_t::size_type taylor_decompose(taylor_dc_t &dc) && final
    {
        if constexpr (func_has_taylor_decompose<T>) {
            return std::move(this->value()).taylor_decompose(dc);
        }

        // LCOV_EXCL_START
        assert(false);
        throw;
        // LCOV_EXCL_STOP
    }
    [[nodiscard]] bool has_taylor_decompose() const final
    {
        return func_has_taylor_decompose<T>;
    }
    llvm::Value *taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size,
                             bool high_accuracy) const final
    {
        if constexpr (requires() {
                          {
                              this->value().taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx,
                                                        batch_size, high_accuracy)
                          } -> std::same_as<llvm::Value *>;
                      }) {
            return this->value().taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                             high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("Taylor diff is not implemented for the function '{}'", get_name()));
        }
    }
    llvm::Function *taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars, std::uint32_t batch_size,
                                       bool high_accuracy) const final
    {
        if constexpr (requires() {
                          {
                              this->value().taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy)
                          } -> std::same_as<llvm::Function *>;
                      }) {
            return this->value().taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("Taylor diff in compact mode is not implemented for the function '{}'", get_name()));
        }
    }
};

} // namespace detail

namespace detail
{

struct HEYOKA_DLL_PUBLIC func_inner_base {
    func_inner_base();
    func_inner_base(const func_inner_base &) = delete;
    func_inner_base(func_inner_base &&) noexcept = delete;
    func_inner_base &operator=(const func_inner_base &) = delete;
    func_inner_base &operator=(func_inner_base &&) noexcept = delete;
    virtual ~func_inner_base();

    [[nodiscard]] virtual std::unique_ptr<func_inner_base> clone() const = 0;

    [[nodiscard]] virtual std::type_index get_type_index() const = 0;
    [[nodiscard]] virtual const void *get_ptr() const = 0;

    [[nodiscard]] virtual const std::string &get_name() const = 0;

    virtual void to_stream(std::ostringstream &) const = 0;

    [[nodiscard]] virtual bool is_time_dependent() const = 0;

    [[nodiscard]] virtual bool has_normalise() const = 0;
    [[nodiscard]] virtual expression normalise() const = 0;

    [[nodiscard]] virtual const std::vector<expression> &args() const = 0;
    virtual std::pair<expression *, expression *> get_mutable_args_range() = 0;

    [[nodiscard]] virtual bool has_diff_var() const = 0;
    virtual expression diff(funcptr_map<expression> &, const std::string &) const = 0;
    [[nodiscard]] virtual bool has_diff_par() const = 0;
    virtual expression diff(funcptr_map<expression> &, const param &) const = 0;
    [[nodiscard]] virtual bool has_gradient() const = 0;
    [[nodiscard]] virtual std::vector<expression> gradient() const = 0;

    [[nodiscard]] virtual llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &,
                                                 llvm::Value *, llvm::Value *, llvm::Value *, std::uint32_t, bool) const
        = 0;

    [[nodiscard]] virtual llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const = 0;

    // NOTE: this is the last remaining trace of mutability
    // related to decomposition. Note, however, that this is never
    // exposed in the public API.
    virtual taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) && = 0;
    [[nodiscard]] virtual bool has_taylor_decompose() const = 0;

    virtual llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                     const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t, bool) const
        = 0;

    virtual llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const
        = 0;

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
using func_to_stream_t
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().to_stream(std::declval<std::ostringstream &>()));

template <typename T>
inline constexpr bool func_has_to_stream_v = std::is_same_v<detected_t<func_to_stream_t, T>, void>;

template <typename T>
using func_is_time_dependent_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().is_time_dependent());

template <typename T>
inline constexpr bool func_has_is_time_dependent_v = std::is_same_v<detected_t<func_is_time_dependent_t, T>, bool>;

template <typename T>
using func_normalise_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().normalise());

template <typename T>
inline constexpr bool func_has_normalise_v = std::is_same_v<detected_t<func_normalise_t, T>, expression>;

template <typename T>
using func_diff_var_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().diff(
    std::declval<funcptr_map<expression> &>(), std::declval<const std::string &>()));

template <typename T>
inline constexpr bool func_has_diff_var_v = std::is_same_v<detected_t<func_diff_var_t, T>, expression>;

template <typename T>
using func_diff_par_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().diff(
    std::declval<funcptr_map<expression> &>(), std::declval<const param &>()));

template <typename T>
inline constexpr bool func_has_diff_par_v = std::is_same_v<detected_t<func_diff_par_t, T>, expression>;

template <typename T>
using func_gradient_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().gradient());

template <typename T>
inline constexpr bool func_has_gradient_v = std::is_same_v<detected_t<func_gradient_t, T>, std::vector<expression>>;

template <typename T>
using func_llvm_eval_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().llvm_eval(
    std::declval<llvm_state &>(), std::declval<llvm::Type *>(), std::declval<const std::vector<llvm::Value *> &>(),
    std::declval<llvm::Value *>(), std::declval<llvm::Value *>(), std::declval<llvm::Value *>(),
    std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_llvm_eval_v = std::is_same_v<detected_t<func_llvm_eval_t, T>, llvm::Value *>;

template <typename T>
using func_llvm_c_eval_func_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().llvm_c_eval_func(
    std::declval<llvm_state &>(), std::declval<llvm::Type *>(), std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_llvm_c_eval_func_v
    = std::is_same_v<detected_t<func_llvm_c_eval_func_t, T>, llvm::Function *>;

template <typename T>
using func_taylor_decompose_t
    = decltype(std::declval<std::add_rvalue_reference_t<T>>().taylor_decompose(std::declval<taylor_dc_t &>()));

template <typename T>
inline constexpr bool func_has_taylor_decompose_v
    = std::is_same_v<detected_t<func_taylor_decompose_t, T>, taylor_dc_t::size_type>;

template <typename T>
using func_taylor_diff_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_diff(
    std::declval<llvm_state &>(), std::declval<llvm::Type *>(), std::declval<const std::vector<std::uint32_t> &>(),
    std::declval<const std::vector<llvm::Value *> &>(), std::declval<llvm::Value *>(), std::declval<llvm::Value *>(),
    std::declval<std::uint32_t>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>(),
    std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_taylor_diff_v = std::is_same_v<detected_t<func_taylor_diff_t, T>, llvm::Value *>;

template <typename T>
using func_taylor_c_diff_func_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_c_diff_func(
    std::declval<llvm_state &>(), std::declval<llvm::Type *>(), std::declval<std::uint32_t>(),
    std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_taylor_c_diff_func_v
    = std::is_same_v<detected_t<func_taylor_c_diff_func_t, T>, llvm::Function *>;

template <typename T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_inner final : func_inner_base {
    T m_value;

    // We just need the def ctor, delete everything else.
    func_inner() = default;
    func_inner(const func_inner &) = delete;
    func_inner(func_inner &&) = delete;
    func_inner &operator=(const func_inner &) = delete;
    func_inner &operator=(func_inner &&) = delete;
    ~func_inner() final = default;

    // Constructors from T (copy and move variants).
    explicit func_inner(const T &x) : m_value(x) {}
    explicit func_inner(T &&x) : m_value(std::move(x)) {}

    // The clone function.
    [[nodiscard]] std::unique_ptr<func_inner_base> clone() const final
    {
        return std::make_unique<func_inner>(m_value);
    }

    // Get the type at runtime.
    [[nodiscard]] std::type_index get_type_index() const final
    {
        return typeid(T);
    }
    // Raw value getters.
    [[nodiscard]] const void *get_ptr() const final
    {
        return std::addressof(m_value);
    }

    [[nodiscard]] const std::string &get_name() const final
    {
        // NOTE: make sure we are invoking the member functions
        // from func_base (these functions could have been overriden
        // in the derived class).
        return static_cast<const func_base &>(m_value).get_name();
    }

    void to_stream(std::ostringstream &oss) const final
    {
        if constexpr (func_has_to_stream_v<T>) {
            m_value.to_stream(oss);
        } else {
            func_default_to_stream_impl(oss, static_cast<const func_base &>(m_value));
        }
    }

    [[nodiscard]] bool is_time_dependent() const final
    {
        if constexpr (func_has_is_time_dependent_v<T>) {
            return m_value.is_time_dependent();
        } else {
            return false;
        }
    }

    [[nodiscard]] bool has_normalise() const final
    {
        return func_has_normalise_v<T>;
    }
    [[nodiscard]] expression normalise() const final;

    [[nodiscard]] const std::vector<expression> &args() const final
    {
        return static_cast<const func_base &>(m_value).args();
    }
    std::pair<expression *, expression *> get_mutable_args_range() final
    {
        return static_cast<func_base &>(m_value).get_mutable_args_range();
    }

    // diff.
    [[nodiscard]] bool has_diff_var() const final
    {
        return func_has_diff_var_v<T>;
    }
    expression diff(funcptr_map<expression> &, const std::string &) const final;
    [[nodiscard]] bool has_diff_par() const final
    {
        return func_has_diff_par_v<T>;
    }
    expression diff(funcptr_map<expression> &, const param &) const final;

    // gradient.
    [[nodiscard]] bool has_gradient() const final
    {
        return func_has_gradient_v<T>;
    }
    [[nodiscard]] std::vector<expression> gradient() const final
    {
        if constexpr (func_has_gradient_v<T>) {
            return m_value.gradient();
        }

        // LCOV_EXCL_START
        assert(false);
        throw;
        // LCOV_EXCL_STOP
    }

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                         llvm::Value *par_ptr, llvm::Value *time_ptr, llvm::Value *stride,
                                         std::uint32_t batch_size, bool high_accuracy) const final
    {
        if constexpr (func_has_llvm_eval_v<T>) {
            return m_value.llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size, high_accuracy);
        } else {
            throw not_implemented_error("llvm_eval() is not implemented for the function '" + get_name() + "'");
        }
    }

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                                   bool high_accuracy) const final
    {
        if constexpr (func_has_llvm_c_eval_func_v<T>) {
            return m_value.llvm_c_eval_func(s, fp_t, batch_size, high_accuracy);
        } else {
            throw not_implemented_error("llvm_c_eval_func() is not implemented for the function '" + get_name() + "'");
        }
    }

    // Taylor.
    taylor_dc_t::size_type taylor_decompose(taylor_dc_t &dc) && final
    {
        if constexpr (func_has_taylor_decompose_v<T>) {
            return std::move(m_value).taylor_decompose(dc);
        }

        // LCOV_EXCL_START
        assert(false);
        throw;
        // LCOV_EXCL_STOP
    }
    [[nodiscard]] bool has_taylor_decompose() const final
    {
        return func_has_taylor_decompose_v<T>;
    }
    llvm::Value *taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size,
                             bool high_accuracy) const final
    {
        if constexpr (func_has_taylor_diff_v<T>) {
            return m_value.taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                       high_accuracy);
        } else {
            throw not_implemented_error("Taylor diff is not implemented for the function '" + get_name() + "'");
        }
    }
    llvm::Function *taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars, std::uint32_t batch_size,
                                       bool high_accuracy) const final
    {
        if constexpr (func_has_taylor_c_diff_func_v<T>) {
            return m_value.taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy);
        } else {
            throw not_implemented_error("Taylor diff in compact mode is not implemented for the function '" + get_name()
                                        + "'");
        }
    }

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_inner_base>(*this);
        ar & m_value;
    }
};

} // namespace detail

HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;

HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &);
HEYOKA_DLL_PUBLIC bool operator!=(const func &, const func &);
HEYOKA_DLL_PUBLIC bool operator<(const func &, const func &);

class HEYOKA_DLL_PUBLIC func
{
    friend HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;
    friend HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &);
    friend HEYOKA_DLL_PUBLIC bool operator<(const func &, const func &);

    // Pointer to the inner base.
    std::shared_ptr<detail::func_inner_base> m_ptr;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned version)
    {
        // LCOV_EXCL_START
        if (version < static_cast<unsigned>(boost::serialization::version<func>::type::value)) {
            throw std::invalid_argument("Cannot load a function instance from an older archive");
        }
        // LCOV_EXCL_STOP

        ar & m_ptr;
    }

    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    [[nodiscard]] const detail::func_inner_base *ptr() const;
    detail::func_inner_base *ptr();

    // Private constructor used in the copy() function.
    HEYOKA_DLL_LOCAL explicit func(std::unique_ptr<detail::func_inner_base>);

    // Private helper to extract and check the gradient in the
    // diff() implementations.
    [[nodiscard]] HEYOKA_DLL_LOCAL std::vector<expression> fetch_gradient(const std::string &) const;

public:
    func();

    template <typename T>
        requires(!std::same_as<func, std::remove_cvref_t<T>>) && is_udf<std::remove_cvref_t<T>>
    explicit func(T &&x) : m_ptr(std::make_shared<detail::func_inner<detail::uncvref_t<T>>>(std::forward<T>(x)))
    {
    }

    func(const func &) noexcept;
    func(func &&) noexcept;

    func &operator=(const func &) noexcept;
    func &operator=(func &&) noexcept;

    ~func();

    // NOTE: this creates a new func containing
    // a copy of the inner object in which the original
    // function arguments have been replaced by the
    // provided vector of arguments.
    [[nodiscard]] func copy(const std::vector<expression> &) const;

    // NOTE: like in pagmo, this may fail if invoked
    // from different DLLs in certain situations (e.g.,
    // Python bindings on OSX). I don't
    // think this is currently an interesting use case
    // for heyoka (as we don't provide a way of implementing
    // new functions in Python), but, if it becomes a problem
    // in the future, we can solve this in the same way as
    // in pagmo.
    template <typename T>
    [[nodiscard]] const T *extract() const noexcept
    {
        const auto *p = dynamic_cast<const detail::func_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    [[nodiscard]] std::type_index get_type_index() const;
    [[nodiscard]] const void *get_ptr() const;

    [[nodiscard]] bool is_time_dependent() const;

    [[nodiscard]] expression normalise() const;

    [[nodiscard]] const std::string &get_name() const;

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::size_t hash(detail::funcptr_map<std::size_t> &) const;

    [[nodiscard]] const std::vector<expression> &args() const;

    expression diff(detail::funcptr_map<expression> &, const std::string &) const;
    expression diff(detail::funcptr_map<expression> &, const param &) const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    std::vector<expression>::size_type decompose(detail::funcptr_map<std::vector<expression>::size_type> &,
                                                 std::vector<expression> &) const;

    taylor_dc_t::size_type taylor_decompose(detail::funcptr_map<taylor_dc_t::size_type> &, taylor_dc_t &) const;
    llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint32_t, bool) const;
    llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const;
};

namespace detail
{

[[nodiscard]] llvm::Value *cfunc_nc_param_codegen(llvm_state &, const param &, std::uint32_t, llvm::Type *,
                                                  llvm::Value *, llvm::Value *);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *
llvm_eval_helper(const std::function<llvm::Value *(const std::vector<llvm::Value *> &, bool)> &, const func_base &,
                 llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                 std::uint32_t, bool);

std::pair<std::string, std::vector<llvm::Type *>> llvm_c_eval_func_name_args(llvm::LLVMContext &, llvm::Type *,
                                                                             const std::string &, std::uint32_t,
                                                                             const std::vector<expression> &);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *
llvm_c_eval_func_helper(const std::string &,
                        const std::function<llvm::Value *(const std::vector<llvm::Value *> &, bool)> &,
                        const func_base &, llvm_state &, llvm::Type *, std::uint32_t, bool);

} // namespace detail

HEYOKA_END_NAMESPACE

// Macros for the registration of s11n for concrete functions.
#define HEYOKA_S11N_FUNC_EXPORT_KEY(f) BOOST_CLASS_EXPORT_KEY(heyoka::detail::func_inner<f>)

#define HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f) BOOST_CLASS_EXPORT_IMPLEMENT(heyoka::detail::func_inner<f>)

#define HEYOKA_S11N_FUNC_EXPORT(f)                                                                                     \
    HEYOKA_S11N_FUNC_EXPORT_KEY(f)                                                                                     \
    HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f)

// Export the key for null_func.
HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::null_func)

#endif
