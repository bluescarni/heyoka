// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_VARIABLE_HPP
#define HEYOKA_VARIABLE_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Value.h>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

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

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(variable &&, std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const variable &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const variable &, llvm::Value *);

} // namespace heyoka

#endif
