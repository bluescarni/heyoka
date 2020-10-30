// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNCTION_HPP
#define HEYOKA_FUNCTION_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC function
{
public:
    using codegen_t = std::function<llvm::Value *(llvm_state &, const std::vector<llvm::Value *> &)>;

    using diff_t = std::function<expression(const std::vector<expression> &, const std::string &)>;

    // Evaluation function types.
    using eval_dbl_t
        = std::function<double(const std::vector<expression> &, const std::unordered_map<std::string, double> &)>;
    using eval_batch_dbl_t = std::function<void(std::vector<double> &, const std::vector<expression> &,
                                                const std::unordered_map<std::string, std::vector<double>> &)>;
    using eval_num_dbl_t = std::function<double(const std::vector<double> &)>;
    using deval_num_dbl_t = std::function<double(const std::vector<double> &, std::vector<double>::size_type)>;

    // Taylor integration function types.
    using taylor_decompose_t
        = std::function<std::vector<expression>::size_type(function &&, std::vector<expression> &)>;
    using taylor_u_init_t = std::function<llvm::Value *(llvm_state &, const function &,
                                                        const std::vector<llvm::Value *> &, std::uint32_t)>;
    using taylor_diff_t
        = std::function<llvm::Value *(llvm_state &, const function &, const std::vector<llvm::Value *> &, std::uint32_t,
                                      std::uint32_t, std::uint32_t, std::uint32_t)>;
    using taylor_c_u_init_t
        = std::function<llvm::Value *(llvm_state &, const function &, llvm::Value *, std::uint32_t)>;
    using taylor_c_diff_func_t
        = std::function<llvm::Function *(llvm_state &, const function &, std::uint32_t, std::uint32_t)>;

private:
    codegen_t m_codegen_dbl_f, m_codegen_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_codegen_f128_f
#endif
        ;

    std::string m_display_name;

    std::unique_ptr<std::vector<expression>> m_args;

    diff_t m_diff_f;

    eval_dbl_t m_eval_dbl_f;
    eval_batch_dbl_t m_eval_batch_dbl_f;
    eval_num_dbl_t m_eval_num_dbl_f;
    deval_num_dbl_t m_deval_num_dbl_f;

    taylor_decompose_t m_taylor_decompose_f;
    taylor_u_init_t m_taylor_u_init_dbl_f, m_taylor_u_init_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_taylor_u_init_f128_f
#endif
        ;
    taylor_diff_t m_taylor_diff_dbl_f, m_taylor_diff_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_taylor_diff_f128_f
#endif
        ;
    taylor_c_u_init_t m_taylor_c_u_init_dbl_f, m_taylor_c_u_init_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_taylor_c_u_init_f128_f
#endif
        ;
    taylor_c_diff_func_t m_taylor_c_diff_func_dbl_f, m_taylor_c_diff_func_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_taylor_c_diff_func_f128_f
#endif
        ;

public:
    explicit function(std::vector<expression>);
    function(const function &);
    function(function &&) noexcept;
    ~function();

    function &operator=(const function &);
    function &operator=(function &&) noexcept;

    codegen_t &codegen_dbl_f();
    codegen_t &codegen_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    codegen_t &codegen_f128_f();
#endif
    std::string &display_name();
    std::vector<expression> &args();
    diff_t &diff_f();
    eval_dbl_t &eval_dbl_f();
    eval_batch_dbl_t &eval_batch_dbl_f();
    eval_num_dbl_t &eval_num_dbl_f();
    deval_num_dbl_t &deval_num_dbl_f();
    taylor_decompose_t &taylor_decompose_f();
    taylor_u_init_t &taylor_u_init_dbl_f();
    taylor_u_init_t &taylor_u_init_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    taylor_u_init_t &taylor_u_init_f128_f();
#endif
    taylor_diff_t &taylor_diff_dbl_f();
    taylor_diff_t &taylor_diff_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    taylor_diff_t &taylor_diff_f128_f();
#endif
    taylor_c_u_init_t &taylor_c_u_init_dbl_f();
    taylor_c_u_init_t &taylor_c_u_init_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    taylor_c_u_init_t &taylor_c_u_init_f128_f();
#endif
    taylor_c_diff_func_t &taylor_c_diff_func_dbl_f();
    taylor_c_diff_func_t &taylor_c_diff_func_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    taylor_c_diff_func_t &taylor_c_diff_func_f128_f();
#endif

    const codegen_t &codegen_dbl_f() const;
    const codegen_t &codegen_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const codegen_t &codegen_f128_f() const;
#endif
    const std::string &display_name() const;
    const std::vector<expression> &args() const;
    const diff_t &diff_f() const;
    const eval_dbl_t &eval_dbl_f() const;
    const eval_batch_dbl_t &eval_batch_dbl_f() const;
    const eval_num_dbl_t &eval_num_dbl_f() const;
    const deval_num_dbl_t &deval_num_dbl_f() const;
    const taylor_decompose_t &taylor_decompose_f() const;
    const taylor_u_init_t &taylor_u_init_dbl_f() const;
    const taylor_u_init_t &taylor_u_init_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const taylor_u_init_t &taylor_u_init_f128_f() const;
#endif
    const taylor_diff_t &taylor_diff_dbl_f() const;
    const taylor_diff_t &taylor_diff_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const taylor_diff_t &taylor_diff_f128_f() const;
#endif
    const taylor_c_u_init_t &taylor_c_u_init_dbl_f() const;
    const taylor_c_u_init_t &taylor_c_u_init_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const taylor_c_u_init_t &taylor_c_u_init_f128_f() const;
#endif
    const taylor_c_diff_func_t &taylor_c_diff_func_dbl_f() const;
    const taylor_c_diff_func_t &taylor_c_diff_func_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const taylor_c_diff_func_t &taylor_c_diff_func_f128_f() const;
#endif
};

HEYOKA_DLL_PUBLIC void swap(function &, function &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const function &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const function &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const function &);
HEYOKA_DLL_PUBLIC void rename_variables(function &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const function &, const function &);
HEYOKA_DLL_PUBLIC bool operator!=(const function &, const function &);

HEYOKA_DLL_PUBLIC expression subs(const function &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const function &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const function &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const function &,
                                      const std::unordered_map<std::string, std::vector<double>> &);

HEYOKA_DLL_PUBLIC double eval_num_dbl_f(const function &, const std::vector<double> &);
HEYOKA_DLL_PUBLIC double deval_num_dbl_f(const function &, const std::vector<double> &, std::vector<double>::size_type);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const function &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const function &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const function &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const function &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const function &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *codegen_f128(llvm_state &, const function &);

#endif

template <typename T>
inline llvm::Value *codegen(llvm_state &s, const function &f)
{
    if constexpr (std::is_same_v<T, double>) {
        return codegen_dbl(s, f);
    } else if constexpr (std::is_same_v<T, long double>) {
        return codegen_ldbl(s, f);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return codegen_f128(s, f);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(function &&, std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_dbl(llvm_state &, const function &, const std::vector<llvm::Value *> &,
                                                 std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_ldbl(llvm_state &, const function &, const std::vector<llvm::Value *> &,
                                                  std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_f128(llvm_state &, const function &, const std::vector<llvm::Value *> &,
                                                  std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_u_init(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_u_init_dbl(s, f, arr, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_u_init_ldbl(s, f, arr, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_u_init_f128(s, f, arr, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_dbl(llvm_state &, const function &, const std::vector<llvm::Value *> &,
                                               std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_ldbl(llvm_state &, const function &, const std::vector<llvm::Value *> &,
                                                std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_f128(llvm_state &, const function &, const std::vector<llvm::Value *> &,
                                                std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_diff(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_diff_dbl(s, f, arr, n_uvars, order, idx, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_diff_ldbl(s, f, arr, n_uvars, order, idx, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_diff_f128(s, f, arr, n_uvars, order, idx, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_u_init_dbl(llvm_state &, const function &, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_u_init_ldbl(llvm_state &, const function &, llvm::Value *, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_u_init_f128(llvm_state &, const function &, llvm::Value *, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_c_u_init(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_c_u_init_dbl(s, f, arr, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_c_u_init_ldbl(s, f, arr, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_c_u_init_f128(s, f, arr, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_dbl(llvm_state &, const function &, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, const function &, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_f128(llvm_state &, const function &, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Function *taylor_c_diff_func(llvm_state &s, const function &func, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_c_diff_func_dbl(s, func, n_uvars, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_c_diff_func_ldbl(s, func, n_uvars, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_c_diff_func_f128(s, func, n_uvars, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

#endif
