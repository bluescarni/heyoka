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

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/tfp.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC function
{
public:
    enum class type { internal, external, builtin };

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
    using taylor_init_batch_t = std::function<llvm::Value *(llvm_state &, const function &, llvm::Value *,
                                                            std::uint32_t, std::uint32_t, std::uint32_t)>;
    using taylor_diff_batch_t = std::function<llvm::Value *(
        llvm_state &, const function &, std::uint32_t, std::uint32_t, std::uint32_t, llvm::Value *, std::uint32_t,
        std::uint32_t, std::uint32_t, const std::unordered_map<std::uint32_t, number> &)>;

private:
    std::string m_name_dbl, m_name_ldbl,
#if defined(HEYOKA_HAVE_REAL128)
        m_name_f128,
#endif
        m_display_name;
    std::unique_ptr<std::vector<expression>> m_args;
    std::vector<llvm::Attribute::AttrKind> m_attributes_dbl, m_attributes_ldbl
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_attributes_f128
#endif
        ;
    type m_ty_dbl = type::internal;
    type m_ty_ldbl = type::internal;
#if defined(HEYOKA_HAVE_REAL128)
    type m_ty_f128 = type::internal;
#endif

    diff_t m_diff_f;

    eval_dbl_t m_eval_dbl_f;
    eval_batch_dbl_t m_eval_batch_dbl_f;
    eval_num_dbl_t m_eval_num_dbl_f;
    deval_num_dbl_t m_deval_num_dbl_f;

    taylor_decompose_t m_taylor_decompose_f;
    taylor_init_batch_t m_taylor_init_batch_dbl_f, m_taylor_init_batch_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_taylor_init_batch_f128_f
#endif
        ;
    taylor_diff_batch_t m_taylor_diff_batch_dbl_f, m_taylor_diff_batch_ldbl_f
#if defined(HEYOKA_HAVE_REAL128)
        ,
        m_taylor_diff_batch_f128_f
#endif
        ;

public:
    explicit function(std::vector<expression>);
    function(const function &);
    function(function &&) noexcept;
    ~function();

    function &operator=(const function &);
    function &operator=(function &&) noexcept;

    std::string &name_dbl();
    std::string &name_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
    std::string &name_f128();
#endif
    std::string &display_name();
    std::vector<expression> &args();
    std::vector<llvm::Attribute::AttrKind> &attributes_dbl();
    std::vector<llvm::Attribute::AttrKind> &attributes_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
    std::vector<llvm::Attribute::AttrKind> &attributes_f128();
#endif
    type &ty_dbl();
    type &ty_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
    type &ty_f128();
#endif
    diff_t &diff_f();
    eval_dbl_t &eval_dbl_f();
    eval_batch_dbl_t &eval_batch_dbl_f();
    eval_num_dbl_t &eval_num_dbl_f();
    deval_num_dbl_t &deval_num_dbl_f();
    taylor_decompose_t &taylor_decompose_f();
    taylor_init_batch_t &taylor_init_batch_dbl_f();
    taylor_init_batch_t &taylor_init_batch_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    taylor_init_batch_t &taylor_init_batch_f128_f();
#endif
    taylor_diff_batch_t &taylor_diff_batch_dbl_f();
    taylor_diff_batch_t &taylor_diff_batch_ldbl_f();
#if defined(HEYOKA_HAVE_REAL128)
    taylor_diff_batch_t &taylor_diff_batch_f128_f();
#endif

    const std::string &name_dbl() const;
    const std::string &name_ldbl() const;
#if defined(HEYOKA_HAVE_REAL128)
    const std::string &name_f128() const;
#endif
    const std::string &display_name() const;
    const std::vector<expression> &args() const;
    const std::vector<llvm::Attribute::AttrKind> &attributes_dbl() const;
    const std::vector<llvm::Attribute::AttrKind> &attributes_ldbl() const;
#if defined(HEYOKA_HAVE_REAL128)
    const std::vector<llvm::Attribute::AttrKind> &attributes_f128() const;
#endif
    const type &ty_dbl() const;
    const type &ty_ldbl() const;
#if defined(HEYOKA_HAVE_REAL128)
    const type &ty_f128() const;
#endif
    const diff_t &diff_f() const;
    const eval_dbl_t &eval_dbl_f() const;
    const eval_batch_dbl_t &eval_batch_dbl_f() const;
    const eval_num_dbl_t &eval_num_dbl_f() const;
    const deval_num_dbl_t &deval_num_dbl_f() const;
    const taylor_decompose_t &taylor_decompose_f() const;
    const taylor_init_batch_t &taylor_init_batch_dbl_f() const;
    const taylor_init_batch_t &taylor_init_batch_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const taylor_init_batch_t &taylor_init_batch_f128_f() const;
#endif
    const taylor_diff_batch_t &taylor_diff_batch_dbl_f() const;
    const taylor_diff_batch_t &taylor_diff_batch_ldbl_f() const;
#if defined(HEYOKA_HAVE_REAL128)
    const taylor_diff_batch_t &taylor_diff_batch_f128_f() const;
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

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_batch_dbl(llvm_state &, const function &, llvm::Value *, std::uint32_t,
                                                     std::uint32_t, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_batch_ldbl(llvm_state &, const function &, llvm::Value *, std::uint32_t,
                                                      std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_batch_f128(llvm_state &, const function &, llvm::Value *, std::uint32_t,
                                                      std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_init_batch(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                      std::uint32_t batch_size, std::uint32_t vector_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_init_batch_dbl(s, f, arr, batch_idx, batch_size, vector_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_init_batch_ldbl(s, f, arr, batch_idx, batch_size, vector_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_init_batch_f128(s, f, arr, batch_idx, batch_size, vector_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_batch_dbl(llvm_state &, const function &, std::uint32_t, std::uint32_t,
                                                     std::uint32_t, llvm::Value *, std::uint32_t, std::uint32_t,
                                                     std::uint32_t, const std::unordered_map<std::uint32_t, number> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_batch_ldbl(llvm_state &, const function &, std::uint32_t, std::uint32_t,
                                                      std::uint32_t, llvm::Value *, std::uint32_t, std::uint32_t,
                                                      std::uint32_t, const std::unordered_map<std::uint32_t, number> &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_batch_f128(llvm_state &, const function &, std::uint32_t, std::uint32_t,
                                                      std::uint32_t, llvm::Value *, std::uint32_t, std::uint32_t,
                                                      std::uint32_t, const std::unordered_map<std::uint32_t, number> &);

#endif

template <typename T>
inline llvm::Value *taylor_diff_batch(llvm_state &s, const function &f, std::uint32_t idx, std::uint32_t order,
                                      std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                      std::uint32_t batch_size, std::uint32_t vector_size,
                                      const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_diff_batch_dbl(s, f, idx, order, n_uvars, diff_arr, batch_idx, batch_size, vector_size, cd_uvars);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_diff_batch_ldbl(s, f, idx, order, n_uvars, diff_arr, batch_idx, batch_size, vector_size,
                                      cd_uvars);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_diff_batch_f128(s, f, idx, order, n_uvars, diff_arr, batch_idx, batch_size, vector_size,
                                      cd_uvars);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

namespace detail
{

template <typename T>
HEYOKA_DLL_PUBLIC llvm::Value *function_codegen_from_values(llvm_state &, const function &,
                                                            const std::vector<llvm::Value *> &);

}

HEYOKA_DLL_PUBLIC tfp taylor_u_init_dbl(llvm_state &, const function &, const std::vector<tfp> &, std::uint32_t, bool);
HEYOKA_DLL_PUBLIC tfp taylor_u_init_ldbl(llvm_state &, const function &, const std::vector<tfp> &, std::uint32_t, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC tfp taylor_u_init_f128(llvm_state &, const function &, const std::vector<tfp> &, std::uint32_t, bool);

#endif

template <typename T>
inline tfp taylor_u_init(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                         bool high_accuracy)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_u_init_dbl(s, f, arr, batch_size, high_accuracy);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_u_init_ldbl(s, f, arr, batch_size, high_accuracy);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_u_init_f128(s, f, arr, batch_size, high_accuracy);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC tfp taylor_diff_dbl(llvm_state &, const function &, const std::vector<tfp> &, std::uint32_t,
                                      std::uint32_t, std::uint32_t, std::uint32_t, bool);

HEYOKA_DLL_PUBLIC tfp taylor_diff_ldbl(llvm_state &, const function &, const std::vector<tfp> &, std::uint32_t,
                                       std::uint32_t, std::uint32_t, std::uint32_t, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC tfp taylor_diff_f128(llvm_state &, const function &, const std::vector<tfp> &, std::uint32_t,
                                       std::uint32_t, std::uint32_t, std::uint32_t, bool);

#endif

template <typename T>
inline tfp taylor_diff(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                       std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_diff_dbl(s, f, arr, n_uvars, order, idx, batch_size, high_accuracy);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_diff_ldbl(s, f, arr, n_uvars, order, idx, batch_size, high_accuracy);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_diff_f128(s, f, arr, n_uvars, order, idx, batch_size, high_accuracy);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

#endif
