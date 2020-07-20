// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/assert_nonnull_ret.hpp>
#include <heyoka/detail/string_conv.hpp>
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

variable &variable::operator=(const variable &) = default;

variable &variable::operator=(variable &&) noexcept = default;

std::string &variable::name()
{
    return m_name;
}

const std::string &variable::name() const
{
    return m_name;
}

void swap(variable &v0, variable &v1) noexcept
{
    std::swap(v0.name(), v1.name());
}

std::size_t hash(const variable &v)
{
    return std::hash<std::string>{}(v.name());
}

std::ostream &operator<<(std::ostream &os, const variable &var)
{
    return os << var.name();
}

std::vector<std::string> get_variables(const variable &var)
{
    return {var.name()};
}

void rename_variables(variable &var, const std::unordered_map<std::string, std::string> &repl_map)
{
    if (auto it = repl_map.find(var.name()); it != repl_map.end()) {
        var.name() = it->second;
    }
}

bool operator==(const variable &v1, const variable &v2)
{
    return v1.name() == v2.name();
}

bool operator!=(const variable &v1, const variable &v2)
{
    return !(v1 == v2);
}

expression subs(const variable &var, const std::unordered_map<std::string, expression> &smap)
{
    if (auto it = smap.find(var.name()); it == smap.end()) {
        return expression{var};
    } else {
        return it->second;
    }
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

void eval_batch_dbl(std::vector<double> &out_values, const variable &var,
                    const std::unordered_map<std::string, std::vector<double>> &map)
{
    if (auto it = map.find(var.name()); it != map.end()) {
        out_values = it->second;
    } else {
        throw std::invalid_argument("Cannot evaluate the variable '" + var.name()
                                    + "' because it is missing from the evaluation map");
    }
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const variable &,
                        std::size_t &node_counter)
{
    node_connections.push_back(std::vector<std::size_t>());
    node_counter++;
}

void update_node_values_dbl(std::vector<double> &node_values, const variable &var,
                            const std::unordered_map<std::string, double> &map,
                            const std::vector<std::vector<std::size_t>> &, std::size_t &node_counter)
{
    if (auto it = map.find(var.name()); it != map.end()) {
        node_values[node_counter] = it->second;
    } else {
        throw std::invalid_argument("Cannot update the node output for the variable '" + var.name()
                                    + "' because it is missing from the evaluation map");
    }
    node_counter++;
}

void update_grad_dbl(std::unordered_map<std::string, double> &grad, const variable &var,
                     const std::unordered_map<std::string, double> &, const std::vector<double> &,
                     const std::vector<std::vector<std::size_t>> &, std::size_t &node_counter, double acc)
{
    grad[var.name()] = grad[var.name()] + acc;
    node_counter++;
}

llvm::Value *codegen_dbl(llvm_state &s, const variable &var)
{
    const auto &nv = s.named_values();

    auto it = nv.find(var.name());
    if (it == nv.end()) {
        throw std::invalid_argument("Unknown variable name: " + var.name());
    }

    assert(it->second != nullptr);
    return it->second;
}

llvm::Value *codegen_ldbl(llvm_state &s, const variable &var)
{
    return codegen_dbl(s, var);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *codegen_f128(llvm_state &s, const variable &var)
{
    return codegen_dbl(s, var);
}

#endif

std::vector<expression>::size_type taylor_decompose_in_place(variable &&, std::vector<expression> &)
{
    // NOTE: variables do not require decomposition.
    return 0;
}

llvm::Value *taylor_init_dbl(llvm_state &s, const variable &var, llvm::Value *arr)
{
    auto &builder = s.builder();

    // Check that var is a u variable and extract its index.
    const auto &var_name = var.name();
    if (var_name.rfind("u_", 0) != 0) {
        throw std::invalid_argument("Invalid variable name '" + var_name
                                    + "' encountered in the Taylor initialization phase (the name "
                                      "must be in the form 'u_n', where n is a non-negative integer)");
    }
    const auto idx = detail::uname_to_index(var_name);

    // Index into the array of derivatives.
    auto ptr = builder.CreateInBoundsGEP(arr, {builder.getInt32(0), builder.getInt32(idx)}, "diff_ptr");
    assert(ptr != nullptr);

    // Return a load instruction from the array of derivatives.
    heyoka_assert_nonnull_ret(builder.CreateLoad(ptr, "diff_load"));
}

llvm::Value *taylor_init_ldbl(llvm_state &s, const variable &var, llvm::Value *arr)
{
    // NOTE: no codegen differences between dbl and ldbl in this case.
    return taylor_init_dbl(s, var, arr);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_init_f128(llvm_state &s, const variable &var, llvm::Value *arr)
{
    return taylor_init_dbl(s, var, arr);
}

#endif

} // namespace heyoka
