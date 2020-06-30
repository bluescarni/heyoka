// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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

} // namespace heyoka
