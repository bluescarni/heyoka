// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

variable::variable() : variable("") {}

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

namespace detail
{

std::vector<std::string> get_variables(const std::unordered_set<const void *> &, const variable &var)
{
    return {var.name()};
}

void rename_variables(const std::unordered_set<const void *> &, variable &var,
                      const std::unordered_map<std::string, std::string> &repl_map)
{
    if (auto it = repl_map.find(var.name()); it != repl_map.end()) {
        var.name() = it->second;
    }
}

} // namespace detail

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

double eval_dbl(const variable &var, const std::unordered_map<std::string, double> &map, const std::vector<double> &)
{
    using namespace fmt::literals;
    if (auto it = map.find(var.name()); it != map.end()) {
        return it->second;
    } else {
        throw std::invalid_argument(
            "Cannot evaluate the variable '{}' because it is missing from the evaluation map"_format(var.name()));
    }
}

long double eval_ldbl(const variable &var, const std::unordered_map<std::string, long double> &map,
                      const std::vector<long double> &)
{
    using namespace fmt::literals;
    if (auto it = map.find(var.name()); it != map.end()) {
        return it->second;
    } else {
        throw std::invalid_argument(
            "Cannot evaluate the variable '{}' because it is missing from the evaluation map"_format(var.name()));
    }
}

#if defined(HEYOKA_HAVE_REAL128)

mppp::real128 eval_f128(const variable &var, const std::unordered_map<std::string, mppp::real128> &map,
                        const std::vector<mppp::real128> &)
{
    using namespace fmt::literals;
    if (auto it = map.find(var.name()); it != map.end()) {
        return it->second;
    } else {
        throw std::invalid_argument(
            "Cannot evaluate the variable '{}' because it is missing from the evaluation map"_format(var.name()));
    }
}

#endif

void eval_batch_dbl(std::vector<double> &out_values, const variable &var,
                    const std::unordered_map<std::string, std::vector<double>> &map, const std::vector<double> &)
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

} // namespace heyoka
