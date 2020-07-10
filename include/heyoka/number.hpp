// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_NUMBER_HPP
#define HEYOKA_NUMBER_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <llvm/IR/Value.h>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC number
{
public:
    using value_type = std::variant<double, long double>;

private:
    value_type m_value;

public:
    explicit number(double);
    explicit number(long double);
    number(const number &);
    number(number &&) noexcept;
    ~number();

    number &operator=(const number &);
    number &operator=(number &&) noexcept;

    value_type &value();
    const value_type &value() const;
};

HEYOKA_DLL_PUBLIC void swap(number &, number &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const number &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const number &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const number &);
HEYOKA_DLL_PUBLIC void rename_variables(number &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool is_zero(const number &);
HEYOKA_DLL_PUBLIC bool is_one(const number &);
HEYOKA_DLL_PUBLIC bool is_negative_one(const number &);

HEYOKA_DLL_PUBLIC number operator+(number, number);
HEYOKA_DLL_PUBLIC number operator-(number, number);
HEYOKA_DLL_PUBLIC number operator*(number, number);
HEYOKA_DLL_PUBLIC number operator/(number, number);

HEYOKA_DLL_PUBLIC bool operator==(const number &, const number &);
HEYOKA_DLL_PUBLIC bool operator!=(const number &, const number &);

HEYOKA_DLL_PUBLIC expression diff(const number &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const number &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const number &,
                                      const std::unordered_map<std::string, std::vector<double>> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const number &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const number &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const number &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const number &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const number &);

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(number &&, std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const number &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const number &, llvm::Value *);

} // namespace heyoka

#endif
