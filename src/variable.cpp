// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/Value.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

variable::variable(std::string s) : m_name(std::move(s)) {}

variable::variable(const variable &) = default;

variable::variable(variable &&) noexcept = default;

variable::~variable() = default;

std::string &variable::name()
{
    return m_name;
}

const std::string &variable::name() const
{
    return m_name;
}

inline namespace literals
{

expression operator""_var(const char *s, std::size_t n)
{
    return expression{variable{std::string{s, n}}};
}

} // namespace literals

std::ostream &operator<<(std::ostream &os, const variable &var)
{
    return os << var.name();
}

std::vector<std::string> get_variables(const variable &var)
{
    return {var.name()};
}

bool operator==(const variable &v1, const variable &v2)
{
    return v1.name() == v2.name();
}

bool operator!=(const variable &v1, const variable &v2)
{
    return !(v1 == v2);
}

expression diff(const variable &var, const std::string &s)
{
    if (s == var.name()) {
        return expression{number{1.}};
    } else {
        return expression{number{0.}};
    }
}

double eval_dbl(const variable &var, const std::unordered_map<std::string, double> &map)
{
    if (auto it = map.find(var.name()); it != map.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Cannot evaluate the variable '" + var.name()
                                    + "' because it is missing from the evaluation map");
    }
}

llvm::Value *codegen_dbl(llvm_state &s, const variable &var)
{
    const auto &nv = s.named_values();

    auto it = nv.find(var.name());
    if (it == nv.end()) {
        throw std::invalid_argument("Unknown variable name: " + var.name());
    }

    return it->second;
}

llvm::Value *codegen_ldbl(llvm_state &s, const variable &var)
{
    return codegen_dbl(s, var);
}

} // namespace heyoka
