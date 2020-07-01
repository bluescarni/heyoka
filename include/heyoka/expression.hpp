// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EXPRESSION_HPP
#define HEYOKA_EXPRESSION_HPP

#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/function.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC expression
{
public:
    using value_type = std::variant<number, variable, binary_operator, function>;

private:
    value_type m_value;

public:
    explicit expression(number);
    explicit expression(variable);
    explicit expression(binary_operator);
    explicit expression(function);
    expression(const expression &);
    expression(expression &&) noexcept;
    ~expression();

    value_type &value();
    const value_type &value() const;
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const expression &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const expression &);

HEYOKA_DLL_PUBLIC expression operator+(expression);
HEYOKA_DLL_PUBLIC expression operator-(expression);

HEYOKA_DLL_PUBLIC expression operator+(expression, expression);
HEYOKA_DLL_PUBLIC expression operator-(expression, expression);
HEYOKA_DLL_PUBLIC expression operator*(expression, expression);
HEYOKA_DLL_PUBLIC expression operator/(expression, expression);

HEYOKA_DLL_PUBLIC bool operator==(const expression &, const expression &);
HEYOKA_DLL_PUBLIC bool operator!=(const expression &, const expression &);

HEYOKA_DLL_PUBLIC expression diff(const expression &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const expression &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(const expression &, const std::unordered_map<std::string, std::vector<double>> &,
                                      std::vector<double> &);

HEYOKA_DLL_PUBLIC std::vector<std::vector<unsigned>> compute_connections(const expression &);

HEYOKA_DLL_PUBLIC void update_connections(const expression &, std::vector<std::vector<unsigned>> &, unsigned &);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const expression &);

} // namespace heyoka

#endif
