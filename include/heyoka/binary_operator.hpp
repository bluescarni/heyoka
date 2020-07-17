// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_BINARY_OPERATOR_HPP
#define HEYOKA_BINARY_OPERATOR_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

class binary_operator;

HEYOKA_DLL_PUBLIC void swap(binary_operator &, binary_operator &) noexcept;

class HEYOKA_DLL_PUBLIC binary_operator
{
    friend void swap(binary_operator &, binary_operator &) noexcept;

public:
    enum class type { add, sub, mul, div };

private:
    type m_type;
    std::unique_ptr<std::array<expression, 2>> m_ops;

public:
    explicit binary_operator(type, expression, expression);
    binary_operator(const binary_operator &);
    binary_operator(binary_operator &&) noexcept;
    ~binary_operator();

    binary_operator &operator=(const binary_operator &);
    binary_operator &operator=(binary_operator &&) noexcept;

    expression &lhs();
    expression &rhs();
    type &op();
    const expression &lhs() const;
    const expression &rhs() const;
    const type &op() const;
};

HEYOKA_DLL_PUBLIC std::size_t hash(const binary_operator &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const binary_operator &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const binary_operator &);
HEYOKA_DLL_PUBLIC void rename_variables(binary_operator &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const binary_operator &, const binary_operator &);
HEYOKA_DLL_PUBLIC bool operator!=(const binary_operator &, const binary_operator &);

HEYOKA_DLL_PUBLIC expression subs(const binary_operator &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const binary_operator &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const binary_operator &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const binary_operator &,
                                      const std::unordered_map<std::string, std::vector<double>> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const binary_operator &,
                                          std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const binary_operator &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const binary_operator &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const binary_operator &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const binary_operator &);

template <typename T>
inline llvm::Value *codegen(llvm_state &s, const binary_operator &bo)
{
    if constexpr (std::is_same_v<T, double>) {
        return codegen_dbl(s, bo);
    } else if constexpr (std::is_same_v<T, long double>) {
        return codegen_ldbl(s, bo);
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(binary_operator &&,
                                                                               std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const binary_operator &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const binary_operator &, llvm::Value *);

template <typename T>
inline llvm::Value *taylor_init(llvm_state &s, const binary_operator &bo, llvm::Value *arr)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_init_dbl(s, bo, arr);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_init_ldbl(s, bo, arr);
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_dbl(llvm_state &, const binary_operator &, std::uint32_t,
                                                  const std::string &, std::uint32_t,
                                                  const std::unordered_map<std::uint32_t, number> &);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_ldbl(llvm_state &, const binary_operator &, std::uint32_t,
                                                   const std::string &, std::uint32_t,
                                                   const std::unordered_map<std::uint32_t, number> &);

template <typename T>
inline llvm::Function *taylor_diff(llvm_state &s, const binary_operator &bo, std::uint32_t idx, const std::string &name,
                                   std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_diff_dbl(s, bo, idx, name, n_uvars, cd_uvars);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_diff_ldbl(s, bo, idx, name, n_uvars, cd_uvars);
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

#endif
