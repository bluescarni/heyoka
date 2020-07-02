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
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC binary_operator
{
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

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const binary_operator &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const binary_operator &);

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(binary_operator &&,
                                                                               std::vector<expression> &);

} // namespace heyoka

#endif
