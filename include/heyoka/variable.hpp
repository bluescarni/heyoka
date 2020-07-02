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

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

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

inline namespace literals
{

HEYOKA_DLL_PUBLIC expression operator""_var(const char *, std::size_t);

}

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const variable &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const variable &);
HEYOKA_DLL_PUBLIC void rename_variables(variable &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const variable &, const variable &);
HEYOKA_DLL_PUBLIC bool operator!=(const variable &, const variable &);

HEYOKA_DLL_PUBLIC expression diff(const variable &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const variable &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(const variable &, const std::unordered_map<std::string, std::vector<double>> &,
                                      std::vector<double> &);

HEYOKA_DLL_PUBLIC void update_connections(const variable &, std::vector<std::vector<unsigned>> &, unsigned &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(const variable &, const std::unordered_map<std::string, double> &,
                                              std::vector<double> &node_values,
                                              const std::vector<std::vector<unsigned>> &, unsigned &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(const variable &, const std::unordered_map<std::string, double> &,
                                       std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<unsigned>> &, unsigned &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const variable &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const variable &);

} // namespace heyoka

#endif
