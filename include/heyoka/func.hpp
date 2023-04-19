// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNC_HPP
#define HEYOKA_FUNC_HPP

#include <heyoka/config.hpp>

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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/s11n.hpp>

// Current archive version is 1.
BOOST_CLASS_VERSION(heyoka::func, 1)

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Fwd declaration.
template <typename>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_inner;

} // namespace detail

class HEYOKA_DLL_PUBLIC func_base
{
    std::string m_name;
    std::vector<expression> m_args;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_name;
        ar &m_args;
    }

    // NOTE: func_inner needs access to get_mutable_args_range().
    template <typename>
    friend struct HEYOKA_DLL_PUBLIC_INLINE_CLASS detail::func_inner;

    std::pair<expression *, expression *> get_mutable_args_range();

public:
    explicit func_base(std::string, std::vector<expression>);

    func_base(const func_base &);
    func_base(func_base &&) noexcept;

    func_base &operator=(const func_base &);
    func_base &operator=(func_base &&) noexcept;

    ~func_base();

    [[nodiscard]] const std::string &get_name() const;
    [[nodiscard]] const std::vector<expression> &args() const;
};

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
    virtual void *get_ptr() = 0;

    [[nodiscard]] virtual const std::string &get_name() const = 0;

    virtual void to_stream(std::ostringstream &) const = 0;

    [[nodiscard]] virtual bool extra_equal_to(const func &) const = 0;

    [[nodiscard]] virtual bool is_time_dependent() const = 0;
    [[nodiscard]] virtual bool is_commutative() const = 0;

    [[nodiscard]] virtual std::size_t extra_hash() const = 0;

    [[nodiscard]] virtual const std::vector<expression> &args() const = 0;
    virtual std::pair<expression *, expression *> get_mutable_args_range() = 0;

    [[nodiscard]] virtual bool has_diff_var() const = 0;
    virtual expression diff(funcptr_map<expression> &, const std::string &) const = 0;
    [[nodiscard]] virtual bool has_diff_par() const = 0;
    virtual expression diff(funcptr_map<expression> &, const param &) const = 0;
    [[nodiscard]] virtual bool has_gradient() const = 0;
    [[nodiscard]] virtual std::vector<expression> gradient() const = 0;

    [[nodiscard]] virtual double eval_dbl(const std::unordered_map<std::string, double> &,
                                          const std::vector<double> &) const
        = 0;
    [[nodiscard]] virtual long double eval_ldbl(const std::unordered_map<std::string, long double> &,
                                                const std::vector<long double> &) const
        = 0;
#if defined(HEYOKA_HAVE_REAL128)
    [[nodiscard]] virtual mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                                                  const std::vector<mppp::real128> &) const
        = 0;
#endif

    virtual void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                                const std::vector<double> &) const
        = 0;
    [[nodiscard]] virtual double eval_num_dbl(const std::vector<double> &) const = 0;
    [[nodiscard]] virtual double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const = 0;

    [[nodiscard]] virtual llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &,
                                                 llvm::Value *, llvm::Value *, llvm::Value *, std::uint32_t, bool) const
        = 0;

    [[nodiscard]] virtual llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const = 0;

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
using func_extra_equal_to_t
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().extra_equal_to(std::declval<const func &>()));

template <typename T>
inline constexpr bool func_has_extra_equal_to_v = std::is_same_v<detected_t<func_extra_equal_to_t, T>, bool>;

template <typename T>
using func_is_time_dependent_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().is_time_dependent());

template <typename T>
inline constexpr bool func_has_is_time_dependent_v = std::is_same_v<detected_t<func_is_time_dependent_t, T>, bool>;

template <typename T>
using func_is_commutative_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().is_commutative());

template <typename T>
inline constexpr bool func_has_is_commutative_v = std::is_same_v<detected_t<func_is_commutative_t, T>, bool>;

template <typename T>
using func_extra_hash_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().extra_hash());

template <typename T>
inline constexpr bool func_has_extra_hash_v = std::is_same_v<detected_t<func_extra_hash_t, T>, std::size_t>;

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
using func_eval_dbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().eval_dbl(
    std::declval<const std::unordered_map<std::string, double> &>(), std::declval<const std::vector<double> &>()));

template <typename T>
inline constexpr bool func_has_eval_dbl_v = std::is_same_v<detected_t<func_eval_dbl_t, T>, double>;

template <typename T>
using func_eval_ldbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().eval_ldbl(
    std::declval<const std::unordered_map<std::string, long double> &>(),
    std::declval<const std::vector<long double> &>()));

template <typename T>
inline constexpr bool func_has_eval_ldbl_v = std::is_same_v<detected_t<func_eval_ldbl_t, T>, long double>;

#if defined(HEYOKA_HAVE_REAL128)

template <typename T>
using func_eval_f128_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().eval_f128(
    std::declval<const std::unordered_map<std::string, mppp::real128> &>(),
    std::declval<const std::vector<mppp::real128> &>()));

template <typename T>
inline constexpr bool func_has_eval_f128_v = std::is_same_v<detected_t<func_eval_f128_t, T>, mppp::real128>;

#endif

template <typename T>
using func_eval_batch_dbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().eval_batch_dbl(
    std::declval<std::vector<double> &>(), std::declval<const std::unordered_map<std::string, std::vector<double>> &>(),
    std::declval<const std::vector<double> &>()));

template <typename T>
inline constexpr bool func_has_eval_batch_dbl_v = std::is_same_v<detected_t<func_eval_batch_dbl_t, T>, void>;

template <typename T>
using func_eval_num_dbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().eval_num_dbl(
    std::declval<const std::vector<double> &>()));

template <typename T>
inline constexpr bool func_has_eval_num_dbl_v = std::is_same_v<detected_t<func_eval_num_dbl_t, T>, double>;

template <typename T>
using func_deval_num_dbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().deval_num_dbl(
    std::declval<const std::vector<double> &>(), std::declval<std::vector<double>::size_type>()));

template <typename T>
inline constexpr bool func_has_deval_num_dbl_v = std::is_same_v<detected_t<func_deval_num_dbl_t, T>, double>;

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

HEYOKA_DLL_PUBLIC void func_default_to_stream_impl(std::ostringstream &, const func_base &);

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
    // Raw getters for the internal instance.
    [[nodiscard]] const void *get_ptr() const final
    {
        return &m_value;
    }
    void *get_ptr() final
    {
        return &m_value;
    }

    [[nodiscard]] const std::string &get_name() const final
    {
        // NOTE: make sure we are invoking the member functions
        // from func_base (these functions could have been overriden
        // in the derived class).
        return static_cast<const func_base *>(&m_value)->get_name();
    }

    void to_stream(std::ostringstream &oss) const final
    {
        if constexpr (func_has_to_stream_v<T>) {
            m_value.to_stream(oss);
        } else {
            func_default_to_stream_impl(oss, static_cast<const func_base &>(m_value));
        }
    }

    [[nodiscard]] bool extra_equal_to(const func &f) const final
    {
        if constexpr (func_has_extra_equal_to_v<T>) {
            return m_value.extra_equal_to(f);
        } else {
            return true;
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

    [[nodiscard]] bool is_commutative() const final
    {
        if constexpr (func_has_is_commutative_v<T>) {
            return m_value.is_commutative();
        } else {
            return false;
        }
    }

    [[nodiscard]] std::size_t extra_hash() const final
    {
        if constexpr (func_has_extra_hash_v<T>) {
            return m_value.extra_hash();
        } else {
            return 0;
        }
    }

    [[nodiscard]] const std::vector<expression> &args() const final
    {
        return static_cast<const func_base *>(&m_value)->args();
    }
    std::pair<expression *, expression *> get_mutable_args_range() final
    {
        return static_cast<func_base *>(&m_value)->get_mutable_args_range();
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

    // eval.
    [[nodiscard]] double eval_dbl(const std::unordered_map<std::string, double> &m,
                                  const std::vector<double> &pars) const final
    {
        if constexpr (func_has_eval_dbl_v<T>) {
            return m_value.eval_dbl(m, pars);
        } else {
            throw not_implemented_error("double eval is not implemented for the function '" + get_name() + "'");
        }
    }
    [[nodiscard]] long double eval_ldbl(const std::unordered_map<std::string, long double> &m,
                                        const std::vector<long double> &pars) const final
    {
        if constexpr (func_has_eval_ldbl_v<T>) {
            return m_value.eval_ldbl(m, pars);
        } else {
            throw not_implemented_error("long double eval is not implemented for the function '" + get_name() + "'");
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    [[nodiscard]] mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &m,
                                          const std::vector<mppp::real128> &pars) const final
    {
        if constexpr (func_has_eval_f128_v<T>) {
            return m_value.eval_f128(m, pars);
        } else {
            throw not_implemented_error("mppp::real128 eval is not implemented for the function '" + get_name() + "'");
        }
    }
#endif
    void eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &m,
                        const std::vector<double> &pars) const final
    {
        if constexpr (func_has_eval_batch_dbl_v<T>) {
            m_value.eval_batch_dbl(out, m, pars);
        } else {
            throw not_implemented_error("double batch eval is not implemented for the function '" + get_name() + "'");
        }
    }
    [[nodiscard]] double eval_num_dbl(const std::vector<double> &v) const final
    {
        if constexpr (func_has_eval_num_dbl_v<T>) {
            return m_value.eval_num_dbl(v);
        } else {
            throw not_implemented_error("double numerical eval is not implemented for the function '" + get_name()
                                        + "'");
        }
    }
    [[nodiscard]] double deval_num_dbl(const std::vector<double> &v, std::vector<double>::size_type i) const final
    {
        if constexpr (func_has_deval_num_dbl_v<T>) {
            return m_value.deval_num_dbl(v, i);
        } else {
            throw not_implemented_error("double numerical eval of the derivative is not implemented for the function '"
                                        + get_name() + "'");
        }
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
        ar &m_value;
    }
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

HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;

HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &);
HEYOKA_DLL_PUBLIC bool operator!=(const func &, const func &);

class HEYOKA_DLL_PUBLIC func
{
    friend HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;
    friend HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &);

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

        ar &m_ptr;
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

    template <typename T>
    using generic_ctor_enabler
        = std::enable_if_t<std::conjunction_v<std::negation<std::is_same<func, detail::uncvref_t<T>>>,
                                              detail::is_func<detail::uncvref_t<T>>>,
                           int>;

public:
    func();

    template <typename T, generic_ctor_enabler<T &&> = 0>
    explicit func(T &&x) : m_ptr(std::make_unique<detail::func_inner<detail::uncvref_t<T>>>(std::forward<T>(x)))
    {
    }

    func(const func &);
    func(func &&) noexcept;

    func &operator=(const func &);
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
        auto p = dynamic_cast<const detail::func_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    [[nodiscard]] std::type_index get_type_index() const;
    [[nodiscard]] const void *get_ptr() const;

    [[nodiscard]] bool is_time_dependent() const;
    [[nodiscard]] bool is_commutative() const;

    [[nodiscard]] const std::string &get_name() const;

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::size_t hash(detail::funcptr_map<std::size_t> &) const;

    [[nodiscard]] const std::vector<expression> &args() const;

    expression diff(detail::funcptr_map<expression> &, const std::string &) const;
    expression diff(detail::funcptr_map<expression> &, const param &) const;

    [[nodiscard]] double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    [[nodiscard]] long double eval_ldbl(const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    [[nodiscard]] mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &) const;
#endif

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const;
    [[nodiscard]] double eval_num_dbl(const std::vector<double> &) const;
    [[nodiscard]] double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const;

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

HEYOKA_DLL_PUBLIC double eval_dbl(const func &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> &);
HEYOKA_DLL_PUBLIC long double eval_ldbl(const func &, const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC mppp::real128 eval_f128(const func &, const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &);
#endif

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const func &,
                                      const std::unordered_map<std::string, std::vector<double>> &,
                                      const std::vector<double> &);
HEYOKA_DLL_PUBLIC double eval_num_dbl(const func &, const std::vector<double> &);
HEYOKA_DLL_PUBLIC double deval_num_dbl(const func &, const std::vector<double> &, std::vector<double>::size_type);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const func &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const func &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const func &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

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

#endif
