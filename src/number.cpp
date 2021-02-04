// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_fwd.hpp>
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

#if defined(HEYOKA_HAVE_REAL128)

number::number(mppp::real128 x) : m_value(x) {}

#endif

number::number(const number &) = default;

number::number(number &&) noexcept = default;

number::~number() = default;

number &number::operator=(const number &) = default;

number &number::operator=(number &&) noexcept = default;

number::value_type &number::value()
{
    return m_value;
}

const number::value_type &number::value() const
{
    return m_value;
}

void swap(number &n0, number &n1) noexcept
{
    std::swap(n0.value(), n1.value());
}

// NOTE: for consistency with the equality operator,
// we want to ensure that:
// - all nan values hash to the same value,
// - two numbers with the same value hash to the same value,
//   even if they are of different types.
// The strategy is then to cast the value to the largest
// floating-point type (which ensures that the original
// value is preserved exactly) and then hash on that.
std::size_t hash(const number &n)
{
    return std::visit(
        [](const auto &v) -> std::size_t {
#if defined(HEYOKA_HAVE_REAL128)
            // NOTE: mppp::hash() already ensures that
            // all nans hash to the same value.
            return mppp::hash(static_cast<mppp::real128>(v));
#else
            if (std::isnan(v)) {
                // Make sure all nan values
                // have the same hash.
                return 0;
            } else {
                return std::hash<long double>{}(static_cast<long double>(v));
            }
#endif
        },
        n.value());
}

std::ostream &operator<<(std::ostream &os, const number &n)
{
    // NOTE: we make sure to print all digits
    // necessary for short-circuiting. Make also
    // sure to always print the decimal point and to
    // use the C locale.
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;

    std::visit(
        [&oss](const auto &arg) {
            oss.precision(std::numeric_limits<detail::uncvref_t<decltype(arg)>>::max_digits10);
            oss << arg;
        },
        n.value());

    return os << oss.str();
}

std::vector<std::string> get_variables(const number &)
{
    return {};
}

void rename_variables(number &, const std::unordered_map<std::string, std::string> &) {}

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

number operator-(number n)
{
    return std::visit([](auto &&arg) { return number{-std::forward<decltype(arg)>(arg)}; }, std::move(n.value()));
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
    return std::visit(
        [](const auto &v1, const auto &v2) {
            using std::isnan;

            if (isnan(v1) && isnan(v2)) {
                // NOTE: make nan compare equal, for consistency
                // with hashing.
                return true;
            } else {
                // NOTE: this covers the following cases:
                // - neither v1 nor v2 is nan,
                // - v1 is nan and v2 is not,
                // - v2 is nan and v1 is not.
                return v1 == v2;
            }
        },
        n1.value(), n2.value());
}

bool operator!=(const number &n1, const number &n2)
{
    return !(n1 == n2);
}

expression subs(const number &n, const std::unordered_map<std::string, expression> &)
{
    return expression{n};
}

expression diff(const number &n, const std::string &)
{
    return std::visit([](const auto &v) { return expression{number{detail::uncvref_t<decltype(v)>(0)}}; }, n.value());
}

double eval_dbl(const number &n, const std::unordered_map<std::string, double> &, const std::vector<double> &)
{
    return std::visit([](const auto &v) { return static_cast<double>(v); }, n.value());
}

void eval_batch_dbl(std::vector<double> &out_values, const number &n,
                    const std::unordered_map<std::string, std::vector<double>> &, const std::vector<double> &)
{
    std::visit(
        [&out_values](const auto &v) {
            for (auto &el : out_values) {
                el = static_cast<double>(v);
            }
        },
        n.value());
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const number &,
                        std::size_t &node_counter)
{
    node_connections.emplace_back();
    node_counter++;
}

void update_node_values_dbl(std::vector<double> &node_values, const number &n,
                            const std::unordered_map<std::string, double> &,
                            const std::vector<std::vector<std::size_t>> &, std::size_t &node_counter)
{
    std::visit([&node_values, &node_counter](const auto &v) { node_values[node_counter] = static_cast<double>(v); },
               n.value());
    node_counter++;
}

void update_grad_dbl(std::unordered_map<std::string, double> &, const number &,
                     const std::unordered_map<std::string, double> &, const std::vector<double> &,
                     const std::vector<std::vector<std::size_t>> &, std::size_t &node_counter, double)
{
    node_counter++;
}

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

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *codegen_f128(llvm_state &s, const number &n)
{
    return std::visit(
        [&s](const auto &v) {
            const auto &sem = detail::to_llvm_type<mppp::real128>(s.context())->getFltSemantics();
            return llvm::ConstantFP::get(s.context(),
                                         llvm::APFloat(sem, detail::li_to_string(static_cast<mppp::real128>(v))));
        },
        n.value());
}

#endif

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
taylor_decompose_in_place(number &&, std::vector<std::pair<expression, std::vector<std::uint32_t>>> &)
{
    // NOTE: numbers do not require decomposition.
    return 0;
}

} // namespace heyoka
