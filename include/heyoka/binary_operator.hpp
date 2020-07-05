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
#include <unordered_map>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

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

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const binary_operator &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const binary_operator &);
HEYOKA_DLL_PUBLIC void rename_variables(binary_operator &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const binary_operator &, const binary_operator &);
HEYOKA_DLL_PUBLIC bool operator!=(const binary_operator &, const binary_operator &);

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

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(binary_operator &&,
                                                                               std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const binary_operator &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const binary_operator &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_dbl(llvm_state &, const binary_operator &, std::uint32_t,
                                                  const std::string &, std::uint32_t,
                                                  const std::unordered_map<std::uint32_t, number> &);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_ldbl(llvm_state &, const binary_operator &, std::uint32_t,
                                                   const std::string &, std::uint32_t,
                                                   const std::unordered_map<std::uint32_t, number> &);

} // namespace heyoka

#endif
