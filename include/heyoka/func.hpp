// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <memory>
#include <ostream>
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

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/s11n.hpp>

namespace heyoka
{

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

public:
    explicit func_base(std::string, std::vector<expression>);

    func_base(const func_base &);
    func_base(func_base &&) noexcept;

    func_base &operator=(const func_base &);
    func_base &operator=(func_base &&) noexcept;

    ~func_base();

    const std::string &get_name() const;
    const std::vector<expression> &args() const;
    std::pair<std::vector<expression>::iterator, std::vector<expression>::iterator> get_mutable_args_it();
};

namespace detail
{

struct HEYOKA_DLL_PUBLIC func_inner_base {
    virtual ~func_inner_base();
    virtual std::unique_ptr<func_inner_base> clone() const = 0;

    virtual std::type_index get_type_index() const = 0;
    virtual const void *get_ptr() const = 0;
    virtual void *get_ptr() = 0;

    virtual const std::string &get_name() const = 0;

    virtual void to_stream(std::ostream &) const = 0;

    virtual bool extra_equal_to(const func &) const = 0;

    virtual std::size_t extra_hash() const = 0;

    virtual const std::vector<expression> &args() const = 0;
    virtual std::pair<std::vector<expression>::iterator, std::vector<expression>::iterator> get_mutable_args_it() = 0;

    virtual llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const = 0;
    virtual llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const = 0;
#if defined(HEYOKA_HAVE_REAL128)
    virtual llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const = 0;
#endif

    virtual bool has_diff_var() const = 0;
    virtual expression diff(std::unordered_map<const void *, expression> &, const std::string &) const = 0;
    virtual bool has_diff_par() const = 0;
    virtual expression diff(std::unordered_map<const void *, expression> &, const param &) const = 0;
    virtual bool has_gradient() const = 0;
    virtual std::vector<expression> gradient() const = 0;

    virtual double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const = 0;
    virtual long double eval_ldbl(const std::unordered_map<std::string, long double> &,
                                  const std::vector<long double> &) const = 0;
#if defined(HEYOKA_HAVE_REAL128)
    virtual mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                                    const std::vector<mppp::real128> &) const = 0;
#endif

    virtual void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                                const std::vector<double> &) const = 0;
    virtual double eval_num_dbl(const std::vector<double> &) const = 0;
    virtual double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const = 0;

    virtual taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) && = 0;
    virtual bool has_taylor_decompose() const = 0;
    virtual llvm::Value *taylor_diff_dbl(llvm_state &, const std::vector<std::uint32_t> &,
                                         const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                         std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const = 0;
    virtual llvm::Value *taylor_diff_ldbl(llvm_state &, const std::vector<std::uint32_t> &,
                                          const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                          std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const = 0;
#if defined(HEYOKA_HAVE_REAL128)
    virtual llvm::Value *taylor_diff_f128(llvm_state &, const std::vector<std::uint32_t> &,
                                          const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                          std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const = 0;
#endif
    virtual llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t) const = 0;
    virtual llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t) const = 0;
#if defined(HEYOKA_HAVE_REAL128)
    virtual llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t) const = 0;
#endif

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
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().to_stream(std::declval<std::ostream &>()));

template <typename T>
inline constexpr bool func_has_to_stream_v = std::is_same_v<detected_t<func_to_stream_t, T>, void>;

template <typename T>
using func_extra_equal_to_t
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().extra_equal_to(std::declval<const func &>()));

template <typename T>
inline constexpr bool func_has_extra_equal_to_v = std::is_same_v<detected_t<func_extra_equal_to_t, T>, bool>;

template <typename T>
using func_extra_hash_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().extra_hash());

template <typename T>
inline constexpr bool func_has_extra_hash_v = std::is_same_v<detected_t<func_extra_hash_t, T>, std::size_t>;

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
using func_diff_var_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().diff(
    std::declval<std::unordered_map<const void *, expression> &>(), std::declval<const std::string &>()));

template <typename T>
inline constexpr bool func_has_diff_var_v = std::is_same_v<detected_t<func_diff_var_t, T>, expression>;

template <typename T>
using func_diff_par_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().diff(
    std::declval<std::unordered_map<const void *, expression> &>(), std::declval<const param &>()));

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
using func_taylor_decompose_t
    = decltype(std::declval<std::add_rvalue_reference_t<T>>().taylor_decompose(std::declval<taylor_dc_t &>()));

template <typename T>
inline constexpr bool func_has_taylor_decompose_v
    = std::is_same_v<detected_t<func_taylor_decompose_t, T>, taylor_dc_t::size_type>;

template <typename T>
using func_taylor_diff_dbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_diff_dbl(
    std::declval<llvm_state &>(), std::declval<const std::vector<std::uint32_t> &>(),
    std::declval<const std::vector<llvm::Value *> &>(), std::declval<llvm::Value *>(), std::declval<llvm::Value *>(),
    std::declval<std::uint32_t>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>(),
    std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_taylor_diff_dbl_v = std::is_same_v<detected_t<func_taylor_diff_dbl_t, T>, llvm::Value *>;

template <typename T>
using func_taylor_diff_ldbl_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_diff_ldbl(
    std::declval<llvm_state &>(), std::declval<const std::vector<std::uint32_t> &>(),
    std::declval<const std::vector<llvm::Value *> &>(), std::declval<llvm::Value *>(), std::declval<llvm::Value *>(),
    std::declval<std::uint32_t>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>(),
    std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_taylor_diff_ldbl_v
    = std::is_same_v<detected_t<func_taylor_diff_ldbl_t, T>, llvm::Value *>;

#if defined(HEYOKA_HAVE_REAL128)

template <typename T>
using func_taylor_diff_f128_t = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_diff_f128(
    std::declval<llvm_state &>(), std::declval<const std::vector<std::uint32_t> &>(),
    std::declval<const std::vector<llvm::Value *> &>(), std::declval<llvm::Value *>(), std::declval<llvm::Value *>(),
    std::declval<std::uint32_t>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>(),
    std::declval<std::uint32_t>(), std::declval<bool>()));

template <typename T>
inline constexpr bool func_has_taylor_diff_f128_v
    = std::is_same_v<detected_t<func_taylor_diff_f128_t, T>, llvm::Value *>;

#endif

template <typename T>
using func_taylor_c_diff_func_dbl_t
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_c_diff_func_dbl(
        std::declval<llvm_state &>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>()));

template <typename T>
inline constexpr bool func_has_taylor_c_diff_func_dbl_v
    = std::is_same_v<detected_t<func_taylor_c_diff_func_dbl_t, T>, llvm::Function *>;

template <typename T>
using func_taylor_c_diff_func_ldbl_t
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_c_diff_func_ldbl(
        std::declval<llvm_state &>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>()));

template <typename T>
inline constexpr bool func_has_taylor_c_diff_func_ldbl_v
    = std::is_same_v<detected_t<func_taylor_c_diff_func_ldbl_t, T>, llvm::Function *>;

#if defined(HEYOKA_HAVE_REAL128)

template <typename T>
using func_taylor_c_diff_func_f128_t
    = decltype(std::declval<std::add_lvalue_reference_t<const T>>().taylor_c_diff_func_f128(
        std::declval<llvm_state &>(), std::declval<std::uint32_t>(), std::declval<std::uint32_t>()));

template <typename T>
inline constexpr bool func_has_taylor_c_diff_func_f128_v
    = std::is_same_v<detected_t<func_taylor_c_diff_func_f128_t, T>, llvm::Function *>;

#endif

HEYOKA_DLL_PUBLIC void func_default_to_stream_impl(std::ostream &, const func_base &);

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

    const std::string &get_name() const final
    {
        // NOTE: make sure we are invoking the member functions
        // from func_base (these functions could have been overriden
        // in the derived class).
        return static_cast<const func_base *>(&m_value)->get_name();
    }

    void to_stream(std::ostream &os) const final
    {
        if constexpr (func_has_to_stream_v<T>) {
            m_value.to_stream(os);
        } else {
            func_default_to_stream_impl(os, static_cast<const func_base &>(m_value));
        }
    }

    bool extra_equal_to(const func &f) const final
    {
        if constexpr (func_has_extra_equal_to_v<T>) {
            return m_value.extra_equal_to(f);
        } else {
            return true;
        }
    }

    std::size_t extra_hash() const final
    {
        if constexpr (func_has_extra_hash_v<T>) {
            return m_value.extra_hash();
        } else {
            return 0;
        }
    }

    const std::vector<expression> &args() const final
    {
        return static_cast<const func_base *>(&m_value)->args();
    }
    std::pair<std::vector<expression>::iterator, std::vector<expression>::iterator> get_mutable_args_it() final
    {
        return static_cast<func_base *>(&m_value)->get_mutable_args_it();
    }

    // codegen.
    llvm::Value *codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &v) const final
    {
        if constexpr (func_has_codegen_dbl_v<T>) {
            return m_value.codegen_dbl(s, v);
        } else {
            throw not_implemented_error("double codegen is not implemented for the function '" + get_name() + "'");
        }
    }
    llvm::Value *codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &v) const final
    {
        if constexpr (func_has_codegen_ldbl_v<T>) {
            return m_value.codegen_ldbl(s, v);
        } else {
            throw not_implemented_error("long double codegen is not implemented for the function '" + get_name() + "'");
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &v) const final
    {
        if constexpr (func_has_codegen_f128_v<T>) {
            return m_value.codegen_f128(s, v);
        } else {
            throw not_implemented_error("float128 codegen is not implemented for the function '" + get_name() + "'");
        }
    }
#endif

    // diff.
    bool has_diff_var() const final
    {
        return func_has_diff_var_v<T>;
    }
    expression diff(std::unordered_map<const void *, expression> &, const std::string &) const final;
    bool has_diff_par() const final
    {
        return func_has_diff_par_v<T>;
    }
    expression diff(std::unordered_map<const void *, expression> &, const param &) const final;

    // gradient.
    bool has_gradient() const final
    {
        return func_has_gradient_v<T>;
    }
    std::vector<expression> gradient() const final
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
    double eval_dbl(const std::unordered_map<std::string, double> &m, const std::vector<double> &pars) const final
    {
        if constexpr (func_has_eval_dbl_v<T>) {
            return m_value.eval_dbl(m, pars);
        } else {
            throw not_implemented_error("double eval is not implemented for the function '" + get_name() + "'");
        }
    }
    long double eval_ldbl(const std::unordered_map<std::string, long double> &m,
                          const std::vector<long double> &pars) const final
    {
        if constexpr (func_has_eval_ldbl_v<T>) {
            return m_value.eval_ldbl(m, pars);
        } else {
            throw not_implemented_error("long double eval is not implemented for the function '" + get_name() + "'");
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &m,
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
    double eval_num_dbl(const std::vector<double> &v) const final
    {
        if constexpr (func_has_eval_num_dbl_v<T>) {
            return m_value.eval_num_dbl(v);
        } else {
            throw not_implemented_error("double numerical eval is not implemented for the function '" + get_name()
                                        + "'");
        }
    }
    double deval_num_dbl(const std::vector<double> &v, std::vector<double>::size_type i) const final
    {
        if constexpr (func_has_deval_num_dbl_v<T>) {
            return m_value.deval_num_dbl(v, i);
        } else {
            throw not_implemented_error("double numerical eval of the derivative is not implemented for the function '"
                                        + get_name() + "'");
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
    bool has_taylor_decompose() const final
    {
        return func_has_taylor_decompose_v<T>;
    }
    llvm::Value *taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                 const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                 std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                 std::uint32_t batch_size, bool high_accuracy) const final
    {
        if constexpr (func_has_taylor_diff_dbl_v<T>) {
            return m_value.taylor_diff_dbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                           high_accuracy);
        } else {
            throw not_implemented_error("double Taylor diff is not implemented for the function '" + get_name() + "'");
        }
    }
    llvm::Value *taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                  std::uint32_t batch_size, bool high_accuracy) const final
    {
        if constexpr (func_has_taylor_diff_ldbl_v<T>) {
            return m_value.taylor_diff_ldbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                            high_accuracy);
        } else {
            throw not_implemented_error("long double Taylor diff is not implemented for the function '" + get_name()
                                        + "'");
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                  std::uint32_t batch_size, bool high_accuracy) const final
    {
        if constexpr (func_has_taylor_diff_f128_v<T>) {
            return m_value.taylor_diff_f128(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                            high_accuracy);
        } else {
            throw not_implemented_error("float128 Taylor diff is not implemented for the function '" + get_name()
                                        + "'");
        }
    }
#endif
    llvm::Function *taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const final
    {
        if constexpr (func_has_taylor_c_diff_func_dbl_v<T>) {
            return m_value.taylor_c_diff_func_dbl(s, n_uvars, batch_size);
        } else {
            throw not_implemented_error("double Taylor diff in compact mode is not implemented for the function '"
                                        + get_name() + "'");
        }
    }
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const final
    {
        if constexpr (func_has_taylor_c_diff_func_ldbl_v<T>) {
            return m_value.taylor_c_diff_func_ldbl(s, n_uvars, batch_size);
        } else {
            throw not_implemented_error("long double Taylor diff in compact mode is not implemented for the function '"
                                        + get_name() + "'");
        }
    }
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const final
    {
        if constexpr (func_has_taylor_c_diff_func_f128_v<T>) {
            return m_value.taylor_c_diff_func_f128(s, n_uvars, batch_size);
        } else {
            throw not_implemented_error("float128 Taylor diff in compact mode is not implemented for the function '"
                                        + get_name() + "'");
        }
    }
#endif

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

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const func &);

HEYOKA_DLL_PUBLIC std::size_t hash(const func &);

HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &);
HEYOKA_DLL_PUBLIC bool operator!=(const func &, const func &);

class HEYOKA_DLL_PUBLIC func
{
    friend HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;
    friend HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const func &);
    friend HEYOKA_DLL_PUBLIC std::size_t hash(const func &);
    friend HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &);

    // Pointer to the inner base.
    std::shared_ptr<detail::func_inner_base> m_ptr;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned version)
    {
        // LCOV_EXCL_START
        if (version == 0u) {
            throw std::invalid_argument("Cannot load a function instance from an older archive");
        }
        // LCOV_EXCL_STOP

        ar &m_ptr;
    }

    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    const detail::func_inner_base *ptr() const;
    detail::func_inner_base *ptr();

    // Private constructor used in the copy() function.
    HEYOKA_DLL_LOCAL explicit func(std::unique_ptr<detail::func_inner_base>);

    // Private helper to extract and check the gradient in the
    // diff() implementations.
    HEYOKA_DLL_LOCAL std::vector<expression> fetch_gradient(const std::string &) const;

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
    // a copy of the inner object: this means that
    // the function arguments are shallow-copied and
    // NOT deep-copied.
    func copy() const;

    // NOTE: like in pagmo, this may fail if invoked
    // from different DLLs in certain situations (e.g.,
    // Python bindings on OSX). I don't
    // think this is currently an interesting use case
    // for heyoka (as we don't provide a way of implementing
    // new functions in Python), but, if it becomes a problem
    // in the future, we can solve this in the same way as
    // in pagmo.
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::func_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::func_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    std::type_index get_type_index() const;
    const void *get_ptr() const;
    void *get_ptr();

    const std::string &get_name() const;

    const std::vector<expression> &args() const;
    std::pair<std::vector<expression>::iterator, std::vector<expression>::iterator> get_mutable_args_it();

    llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const;
    llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const;
#endif

    expression diff(std::unordered_map<const void *, expression> &, const std::string &) const;
    expression diff(std::unordered_map<const void *, expression> &, const param &) const;

    double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    long double eval_ldbl(const std::unordered_map<std::string, long double> &, const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                            const std::vector<mppp::real128> &) const;
#endif

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const;
    double eval_num_dbl(const std::vector<double> &) const;
    double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const;

    taylor_dc_t::size_type taylor_decompose(std::unordered_map<const void *, taylor_dc_t::size_type> &,
                                            taylor_dc_t &) const;
    llvm::Value *taylor_diff_dbl(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                 llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                 std::uint32_t, bool) const;
    llvm::Value *taylor_diff_ldbl(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                  llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t, bool) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *taylor_diff_f128(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                  llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t, bool) const;
#endif
    llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t) const;
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t) const;
#endif
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

// Helper to run the codegen of a function-like object with the arguments
// represented as a vector of LLVM values.
template <typename T, typename F>
inline llvm::Value *codegen_from_values(llvm_state &s, const F &f, const std::vector<llvm::Value *> &args_v)
{
    if constexpr (std::is_same_v<T, double>) {
        return f.codegen_dbl(s, args_v);
    } else if constexpr (std::is_same_v<T, long double>) {
        return f.codegen_ldbl(s, args_v);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return f.codegen_f128(s, args_v);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace detail

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_dbl(llvm_state &, const func &, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, const func &, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_f128(llvm_state &, const func &, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Function *taylor_c_diff_func(llvm_state &s, const func &f, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_c_diff_func_dbl(s, f, n_uvars, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_c_diff_func_ldbl(s, f, n_uvars, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_c_diff_func_f128(s, f, n_uvars, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

// Current archive version is 1.
BOOST_CLASS_VERSION(heyoka::func, 1)

// Macros for the registration of s11n for concrete functions.
#define HEYOKA_S11N_FUNC_EXPORT_KEY(f) BOOST_CLASS_EXPORT_KEY(heyoka::detail::func_inner<f>)

#define HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f) BOOST_CLASS_EXPORT_IMPLEMENT(heyoka::detail::func_inner<f>)

#define HEYOKA_S11N_FUNC_EXPORT(f)                                                                                     \
    HEYOKA_S11N_FUNC_EXPORT_KEY(f)                                                                                     \
    HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f)

#endif
