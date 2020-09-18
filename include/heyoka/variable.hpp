// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_VARIABLE_HPP
#define HEYOKA_VARIABLE_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/tfp.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC variable
{
    std::string m_name;

public:
    explicit variable(std::string);
    variable(const variable &);
    variable(variable &&) noexcept;
    ~variable();

    variable &operator=(const variable &);
    variable &operator=(variable &&) noexcept;

    std::string &name();
    const std::string &name() const;
};

HEYOKA_DLL_PUBLIC void swap(variable &, variable &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const variable &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const variable &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const variable &);
HEYOKA_DLL_PUBLIC void rename_variables(variable &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const variable &, const variable &);
HEYOKA_DLL_PUBLIC bool operator!=(const variable &, const variable &);

HEYOKA_DLL_PUBLIC expression subs(const variable &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const variable &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const variable &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const variable &,
                                      const std::unordered_map<std::string, std::vector<double>> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const variable &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const variable &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const variable &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const variable &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const variable &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *codegen_f128(llvm_state &, const variable &);

#endif

template <typename T>
inline llvm::Value *codegen(llvm_state &s, const variable &var)
{
    if constexpr (std::is_same_v<T, double>) {
        return codegen_dbl(s, var);
    } else if constexpr (std::is_same_v<T, long double>) {
        return codegen_ldbl(s, var);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return codegen_f128(s, var);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(variable &&, std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_batch_dbl(llvm_state &, const variable &, llvm::Value *, std::uint32_t,
                                                     std::uint32_t, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_batch_ldbl(llvm_state &, const variable &, llvm::Value *, std::uint32_t,
                                                      std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_batch_f128(llvm_state &, const variable &, llvm::Value *, std::uint32_t,
                                                      std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_init_batch(llvm_state &s, const variable &var, llvm::Value *arr, std::uint32_t batch_idx,
                                      std::uint32_t batch_size, std::uint32_t vector_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_init_batch_dbl(s, var, arr, batch_idx, batch_size, vector_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_init_batch_ldbl(s, var, arr, batch_idx, batch_size, vector_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_init_batch_f128(s, var, arr, batch_idx, batch_size, vector_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC tfp taylor_u_init_dbl(llvm_state &, const variable &, const std::vector<tfp> &, std::uint32_t, bool);
HEYOKA_DLL_PUBLIC tfp taylor_u_init_ldbl(llvm_state &, const variable &, const std::vector<tfp> &, std::uint32_t, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC tfp taylor_u_init_f128(llvm_state &, const variable &, const std::vector<tfp> &, std::uint32_t, bool);

#endif

template <typename T>
inline tfp taylor_u_init(llvm_state &s, const variable &var, const std::vector<tfp> &arr, std::uint32_t batch_size,
                         bool high_accuracy)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_u_init_dbl(s, var, arr, batch_size, high_accuracy);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_u_init_ldbl(s, var, arr, batch_size, high_accuracy);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_u_init_f128(s, var, arr, batch_size, high_accuracy);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const variable &, llvm::Value *, std::uint32_t, bool);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const variable &, llvm::Value *, std::uint32_t, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_f128(llvm_state &, const variable &, llvm::Value *, std::uint32_t, bool);

#endif

} // namespace heyoka

#endif
