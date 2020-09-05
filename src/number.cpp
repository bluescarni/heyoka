// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/tfp.hpp>

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

std::size_t hash(const number &n)
{
    return std::visit(
        [](const auto &v) {
#if defined(HEYOKA_HAVE_REAL128)
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, mppp::real128>) {
                // NOTE: the real128 hash already guarantees
                // that all nan values return the same hash.
                return mppp::hash(v);
            } else {
#endif
                if (std::isnan(v)) {
                    // Make all nan return the same hash value.
                    return std::size_t(0);
                } else {
                    return std::hash<detail::uncvref_t<decltype(v)>>{}(v);
                }
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif
        },
        n.value());
}

std::ostream &operator<<(std::ostream &os, const number &n)
{
    return std::visit([&os](const auto &arg) -> std::ostream & { return os << arg; }, n.value());
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
#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (std::is_same_v<type1, mppp::real128>) {
                // NOTE: the real128_equal_to() function considers
                // all nan equal.
                return mppp::real128_equal_to(v1, v2);
            } else {
#endif
                // NOTE: make nan compare equal, for consistency
                // with hashing.
                if (std::isnan(v1) && std::isnan(v2)) {
                    return true;
                } else {
                    return v1 == v2;
                }
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif
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

expression subs(const number &n, const std::unordered_map<std::string, expression> &)
{
    return expression{n};
}

expression diff(const number &n, const std::string &)
{
    return std::visit([](const auto &v) { return expression{number{detail::uncvref_t<decltype(v)>(0)}}; }, n.value());
}

double eval_dbl(const number &n, const std::unordered_map<std::string, double> &)
{
    return std::visit([](const auto &v) { return static_cast<double>(v); }, n.value());
}

void eval_batch_dbl(std::vector<double> &out_values, const number &n,
                    const std::unordered_map<std::string, std::vector<double>> &)
{
    return std::visit(
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
    node_connections.push_back(std::vector<std::size_t>());
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

std::vector<expression>::size_type taylor_decompose_in_place(number &&, std::vector<expression> &)
{
    // NOTE: numbers do not require decomposition.
    return 0;
}

// NOTE: for numbers, the Taylor init phase is
// just the codegen.
llvm::Value *taylor_init_batch_dbl(llvm_state &s, const number &n, llvm::Value *, std::uint32_t, std::uint32_t,
                                   std::uint32_t vector_size)
{
    auto ret = codegen_dbl(s, n);

    if (vector_size > 0u) {
        ret = detail::create_constant_vector(s.builder(), ret, vector_size);
    }

    return ret;
}

llvm::Value *taylor_init_batch_ldbl(llvm_state &s, const number &n, llvm::Value *, std::uint32_t, std::uint32_t,
                                    std::uint32_t vector_size)
{
    auto ret = codegen_ldbl(s, n);

    if (vector_size > 0u) {
        ret = detail::create_constant_vector(s.builder(), ret, vector_size);
    }

    return ret;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_init_batch_f128(llvm_state &s, const number &n, llvm::Value *, std::uint32_t, std::uint32_t,
                                    std::uint32_t vector_size)
{
    auto ret = codegen_f128(s, n);

    if (vector_size > 0u) {
        ret = detail::create_constant_vector(s.builder(), ret, vector_size);
    }

    return ret;
}

#endif

namespace detail
{

namespace
{

// NOTE: for numbers, the Taylor init phase is
// just the codegen.
template <typename T>
tfp taylor_u_init_number_impl(llvm_state &s, const number &n, const std::vector<tfp> &, std::uint32_t batch_size,
                              bool high_accuracy)
{
    auto ret = create_constant_vector(s.builder(), codegen<T>(s, n), batch_size);

    if (high_accuracy) {
        return std::pair{ret, create_constant_vector(s.builder(), codegen<T>(s, number(0.)), batch_size)};
    } else {
        return ret;
    }
}

} // namespace

} // namespace detail

tfp taylor_u_init_dbl(llvm_state &s, const number &n, const std::vector<tfp> &arr, std::uint32_t batch_size,
                      bool high_accuracy)
{
    return detail::taylor_u_init_number_impl<double>(s, n, arr, batch_size, high_accuracy);
}

tfp taylor_u_init_ldbl(llvm_state &s, const number &n, const std::vector<tfp> &arr, std::uint32_t batch_size,
                       bool high_accuracy)
{
    return detail::taylor_u_init_number_impl<long double>(s, n, arr, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

tfp taylor_u_init_f128(llvm_state &s, const number &n, const std::vector<tfp> &arr, std::uint32_t batch_size,
                       bool high_accuracy)
{
    return detail::taylor_u_init_number_impl<mppp::real128>(s, n, arr, batch_size, high_accuracy);
}

#endif

} // namespace heyoka
