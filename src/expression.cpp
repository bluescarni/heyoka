// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{
void compute_connections(const expression &e, std::vector<std::vector<unsigned>> &node_connections,
                         unsigned &node_counter)
{
    return std::visit([&node_connections,
                       &node_counter](const auto &arg) { compute_connections(arg, node_connections, node_counter); },
                      e.value());
}
} // namespace detail

expression::expression(number n) : m_value(std::move(n)) {}

expression::expression(variable var) : m_value(std::move(var)) {}

expression::expression(binary_operator bo) : m_value(std::move(bo)) {}

expression::expression(function f) : m_value(std::move(f)) {}

expression::expression(const expression &) = default;

expression::expression(expression &&) noexcept = default;

expression::~expression() = default;

expression::value_type &expression::value()
{
    return m_value;
}

const expression::value_type &expression::value() const
{
    return m_value;
}

std::vector<std::string> get_variables(const expression &e)
{
    return std::visit([](const auto &arg) { return get_variables(arg); }, e.value());
}

std::ostream &operator<<(std::ostream &os, const expression &e)
{
    return std::visit([&os](const auto &arg) -> std::ostream & { return os << arg; }, e.value());
}

expression operator+(expression e)
{
    return e;
}

expression operator-(expression e)
{
    return expression{number{-1.}} * std::move(e);
}

expression operator+(expression e1, expression e2)
{
    auto visitor = [](auto &&v1, auto &&v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, number> && std::is_same_v<type2, number>) {
            // Both are numbers, add them.
            return expression{std::forward<decltype(v1)>(v1) + std::forward<decltype(v2)>(v2)};
        } else if constexpr (std::is_same_v<type1, number>) {
            // e1 number, e2 symbolic.
            if (is_zero(v1)) {
                // 0 + e2 = e2.
                return expression{std::forward<decltype(v2)>(v2)};
            }
            // NOTE: fall through the standard case if e1 is not zero.
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 symbolic, e2 number.
            if (is_zero(v2)) {
                // e1 + 0 = e1.
                return expression{std::forward<decltype(v1)>(v1)};
            }
            // NOTE: fall through the standard case if e2 is not zero.
        }

        // The standard case.
        return expression{binary_operator{binary_operator::type::add, expression{std::forward<decltype(v1)>(v1)},
                                          expression{std::forward<decltype(v2)>(v2)}}};
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator-(expression e1, expression e2)
{
    auto visitor = [](auto &&v1, auto &&v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, number> && std::is_same_v<type2, number>) {
            // Both are numbers, subtract them.
            return expression{std::forward<decltype(v1)>(v1) - std::forward<decltype(v2)>(v2)};
        } else if constexpr (std::is_same_v<type1, number>) {
            // e1 number, e2 symbolic.
            if (is_zero(v1)) {
                // 0 - e2 = -e2.
                return -expression{std::forward<decltype(v2)>(v2)};
            }
            // NOTE: fall through the standard case if e1 is not zero.
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 symbolic, e2 number.
            if (is_zero(v2)) {
                // e1 - 0 = e1.
                return expression{std::forward<decltype(v1)>(v1)};
            }
            // NOTE: fall through the standard case if e2 is not zero.
        }

        // The standard case.
        return expression{binary_operator{binary_operator::type::sub, expression{std::forward<decltype(v1)>(v1)},
                                          expression{std::forward<decltype(v2)>(v2)}}};
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator*(expression e1, expression e2)
{
    auto visitor = [](auto &&v1, auto &&v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, number> && std::is_same_v<type2, number>) {
            // Both are numbers, multiply them.
            return expression{std::forward<decltype(v1)>(v1) * std::forward<decltype(v2)>(v2)};
        } else if constexpr (std::is_same_v<type1, number>) {
            // e1 number, e2 symbolic.
            if (is_zero(v1)) {
                // 0 * e2 = 0.
                return expression{number{0.}};
            } else if (is_one(v1)) {
                // 1 * e2 = e2.
                return expression{std::forward<decltype(v2)>(v2)};
            }
            // NOTE: fall through the standard case if e1 is not zero or one.
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 symbolic, e2 number.
            if (is_zero(v2)) {
                // e1 * 0 = 0.
                return expression{number{0.}};
            } else if (is_one(v2)) {
                // e1 * 1 = e1.
                return expression{std::forward<decltype(v1)>(v1)};
            }
            // NOTE: fall through the standard case if e1 is not zero or one.
        }

        // The standard case.
        return expression{binary_operator{binary_operator::type::mul, expression{std::forward<decltype(v1)>(v1)},
                                          expression{std::forward<decltype(v2)>(v2)}}};
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator/(expression e1, expression e2)
{
    auto visitor = [](auto &&v1, auto &&v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, number> && std::is_same_v<type2, number>) {
            // Both are numbers, divide them.
            return expression{std::forward<decltype(v1)>(v1) / std::forward<decltype(v2)>(v2)};
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 is symbolic, e2 a number.
            if (is_one(v2) == 1) {
                // e1 / 1 = e1.
                return expression{std::forward<decltype(v1)>(v1)};
            } else if (is_negative_one(v2)) {
                // e1 / -1 = -e1.
                return -expression{std::forward<decltype(v1)>(v1)};
            } else {
                // e1 / x = e1 * 1/x.
                return expression{std::forward<decltype(v1)>(v1)}
                       * expression{number{1.} / std::forward<decltype(v2)>(v2)};
            }
        } else {
            // The standard case.
            return expression{binary_operator{binary_operator::type::div, expression{std::forward<decltype(v1)>(v1)},
                                              expression{std::forward<decltype(v2)>(v2)}}};
        }
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

bool operator==(const expression &e1, const expression &e2)
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

    return std::visit(visitor, e1.value(), e2.value());
}

bool operator!=(const expression &e1, const expression &e2)
{
    return !(e1 == e2);
}

expression diff(const expression &e, const std::string &s)
{
    return std::visit([&s](const auto &arg) { return diff(arg, s); }, e.value());
}

double eval_dbl(const expression &e, const std::unordered_map<std::string, double> &map)
{
    return std::visit([&map](const auto &arg) { return eval_dbl(arg, map); }, e.value());
}

std::vector<std::vector<unsigned>> compute_connections(const expression &e)
{
    std::vector<std::vector<unsigned>> node_connections;
    unsigned node_counter = 0u;
    detail::compute_connections(e, node_connections, node_counter);
    return node_connections;
}

} // namespace heyoka
