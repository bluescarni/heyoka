// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

number::number(double x) : m_value(x) {}

number::number(long double x) : m_value(x) {}

number::number(const number &) = default;

number::number(number &&) noexcept = default;

number::~number() = default;

number::value_type &number::value()
{
    return m_value;
}

const number::value_type &number::value() const
{
    return m_value;
}

inline namespace literals
{

expression operator""_dbl(long double x)
{
    return expression{number{static_cast<double>(x)}};
}

expression operator""_dbl(unsigned long long n)
{
    return expression{number{static_cast<double>(n)}};
}

expression operator""_ldbl(long double x)
{
    return expression{number{x}};
}

expression operator""_ldbl(unsigned long long n)
{
    return expression{number{static_cast<long double>(n)}};
}

} // namespace literals

std::ostream &operator<<(std::ostream &os, const number &n)
{
    return std::visit([&os](const auto &arg) -> std::ostream & { return os << arg; }, n.value());
}

std::vector<std::string> get_variables(const number &)
{
    return {};
}

bool is_zero(const number &n)
{
    return std::visit([](const auto &arg) { return arg == 0; }, n.value());
}

bool is_one(const number &n)
{
    return std::visit([](const auto &arg) { return arg == 1; }, n.value());
}

bool is_negative_one(const number &n)
{
    return std::visit([](const auto &arg) { return arg == -1; }, n.value());
}

number operator+(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) {
            return number{std::forward<decltype(arg1)>(arg1) + std::forward<decltype(arg2)>(arg2)};
        },
        std::move(n1.value()), std::move(n2.value()));
}

number operator-(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) {
            return number{std::forward<decltype(arg1)>(arg1) - std::forward<decltype(arg2)>(arg2)};
        },
        std::move(n1.value()), std::move(n2.value()));
}

number operator*(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) {
            return number{std::forward<decltype(arg1)>(arg1) * std::forward<decltype(arg2)>(arg2)};
        },
        std::move(n1.value()), std::move(n2.value()));
}

number operator/(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) {
            return number{std::forward<decltype(arg1)>(arg1) / std::forward<decltype(arg2)>(arg2)};
        },
        std::move(n1.value()), std::move(n2.value()));
}

bool operator==(const number &n1, const number &n2)
{
    auto visitor = [](const auto &v1, const auto &v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, type2>) {
            return v1 == v2;
        } else {
            return false;
        }
    };

    return std::visit(visitor, n1.value(), n2.value());
}

bool operator!=(const number &n1, const number &n2)
{
    return !(n1 == n2);
}

expression diff(const number &n, const std::string &)
{
    return std::visit(
        [](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            return expression{number{type(0)}};
        },
        n.value());
}

double eval_dbl(const number &n, const std::unordered_map<std::string, double> &)
{
    return std::visit([](const auto &v) { return static_cast<double>(v); }, n.value());
}

void eval_batch_dbl(const number &n, const std::unordered_map<std::string, std::vector<double>> &,
                    std::vector<double> &out_values)
{
    return std::visit(
        [&out_values](const auto &v) {
            for (auto &el : out_values) {
                el = static_cast<double>(v);
            }
        },
        n.value());
}

void update_connections(const number &, std::vector<std::vector<unsigned>> &node_connections, unsigned &node_counter)
{
    node_connections.push_back(std::vector<unsigned>());
    node_counter++;
}

void update_node_values_dbl(const number &n, const std::unordered_map<std::string, double> &map,
                            std::vector<double> &node_values,
                            const std::vector<std::vector<unsigned>> &node_connections, unsigned &node_counter)
{

    std::visit(
        [&node_values, &node_counter](const auto &v) {
            node_values[node_counter] = static_cast<double>(v);
        },
        n.value());
    node_counter++;
}

// NOTE: for the generation of constants of other floating-point types
// a possible pattern seems to be:
//
// const auto &sem = detail::to_llvm_type<type>(s.context())->getFltSemantics();
// return llvm::ConstantFP::get(s.context(), llvm::APFloat(sem, detail::li_to_string(v)));
//
// That is, we fetch the floating-point semantics of whatever LLVM type
// corresponds to the C++ type, and then we construct a constant
// from the string representation.
llvm::Value *codegen_dbl(llvm_state &s, const number &n)
{
    return std::visit(
        [&s](const auto &v) { return llvm::ConstantFP::get(s.context(), llvm::APFloat(static_cast<double>(v))); },
        n.value());
}

llvm::Value *codegen_ldbl(llvm_state &s, const number &n)
{
    return std::visit(
        [&s](const auto &v) {
            // NOTE: the idea here is that we first fetch the FP
            // semantics of the LLVM type long double corresponds
            // to. Then we use them to construct a FP constant from
            // the string representation of v.
            // NOTE: v must be cast to long double so that we ensure
            // that li_to_string() produces a string representation
            // of v in long double precision accurate to the
            // last digit.
            const auto &sem = detail::to_llvm_type<long double>(s.context())->getFltSemantics();
            return llvm::ConstantFP::get(s.context(),
                                         llvm::APFloat(sem, detail::li_to_string(static_cast<long double>(v))));
        },
        n.value());
}

} // namespace heyoka
