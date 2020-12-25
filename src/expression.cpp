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
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/math_wrappers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

expression::expression() : expression(number{0.}) {}

expression::expression(double x) : expression(number{x}) {}

expression::expression(long double x) : expression(number{x}) {}

#if defined(HEYOKA_HAVE_REAL128)

expression::expression(mppp::real128 x) : expression(number{x}) {}

#endif

expression::expression(std::string s) : expression(variable{std::move(s)}) {}

expression::expression(number n) : m_value(std::move(n)) {}

expression::expression(variable var) : m_value(std::move(var)) {}

expression::expression(binary_operator bo) : m_value(std::move(bo)) {}

expression::expression(func f) : m_value(std::move(f)) {}

expression::expression(const expression &) = default;

expression::expression(expression &&) noexcept = default;

expression::~expression() = default;

expression &expression::operator=(const expression &) = default;

expression &expression::operator=(expression &&) noexcept = default;

expression::value_type &expression::value()
{
    return m_value;
}

const expression::value_type &expression::value() const
{
    return m_value;
}

inline namespace literals
{

expression operator""_dbl(long double x)
{
    return expression{static_cast<double>(x)};
}

expression operator""_dbl(unsigned long long n)
{
    return expression{static_cast<double>(n)};
}

expression operator""_ldbl(long double x)
{
    return expression{x};
}

expression operator""_ldbl(unsigned long long n)
{
    return expression{static_cast<long double>(n)};
}

expression operator""_var(const char *s, std::size_t n)
{
    return expression{variable{std::string(s, n)}};
}

} // namespace literals

namespace detail
{

prime_wrapper::prime_wrapper(std::string s) : m_str(std::move(s)) {}

prime_wrapper::prime_wrapper(const prime_wrapper &) = default;

prime_wrapper::prime_wrapper(prime_wrapper &&) noexcept = default;

prime_wrapper &prime_wrapper::operator=(const prime_wrapper &) = default;

prime_wrapper &prime_wrapper::operator=(prime_wrapper &&) noexcept = default;

prime_wrapper::~prime_wrapper() = default;

std::pair<expression, expression> prime_wrapper::operator=(expression e) &&
{
    return std::pair{expression{variable{std::move(m_str)}}, std::move(e)};
}

} // namespace detail

detail::prime_wrapper prime(expression e)
{
    return std::visit(
        [&e](auto &v) -> detail::prime_wrapper {
            if constexpr (std::is_same_v<variable, detail::uncvref_t<decltype(v)>>) {
                return detail::prime_wrapper{std::move(v.name())};
            } else {
                std::ostringstream oss;
                oss << e;

                throw std::invalid_argument("Cannot apply the prime() operator to the non-variable expression '"
                                            + oss.str() + "'");
            }
        },
        e.value());
}

inline namespace literals
{

detail::prime_wrapper operator""_p(const char *s, std::size_t n)
{
    return detail::prime_wrapper{std::string(s, n)};
}

} // namespace literals

std::vector<std::string> get_variables(const expression &e)
{
    return std::visit([](const auto &arg) { return get_variables(arg); }, e.value());
}

void rename_variables(expression &e, const std::unordered_map<std::string, std::string> &repl_map)
{
    std::visit([&repl_map](auto &arg) { rename_variables(arg, repl_map); }, e.value());
}

void swap(expression &ex0, expression &ex1) noexcept
{
    std::swap(ex0.value(), ex1.value());
}

std::size_t hash(const expression &ex)
{
    return std::visit([](const auto &v) { return hash(v); }, ex.value());
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
            if (is_one(v2)) {
                // e1 / 1 = e1.
                return expression{std::forward<decltype(v1)>(v1)};
            } else if (is_negative_one(v2)) {
                // e1 / -1 = -e1.
                return -expression{std::forward<decltype(v1)>(v1)};
            }
            // NOTE: fall through to the standard case.
        } else if constexpr (std::is_same_v<type1, number>) {
            // e1 is a number, e2 is symbolic.
            if (is_zero(v1)) {
                // 0 / e2 == 0.
                return expression{number{0.}};
            }
            // NOTE: fall through to the standard case.
        }
        // The standard case.
        return expression{binary_operator{binary_operator::type::div, expression{std::forward<decltype(v1)>(v1)},
                                          expression{std::forward<decltype(v2)>(v2)}}};
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator+(expression ex, double x)
{
    return std::move(ex) + expression{x};
}

expression operator+(expression ex, long double x)
{
    return std::move(ex) + expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator+(expression ex, mppp::real128 x)
{
    return std::move(ex) + expression{x};
}

#endif

expression operator+(double x, expression ex)
{
    return expression{x} + std::move(ex);
}

expression operator+(long double x, expression ex)
{
    return expression{x} + std::move(ex);
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator+(mppp::real128 x, expression ex)
{
    return expression{x} + std::move(ex);
}

#endif

expression operator-(expression ex, double x)
{
    return std::move(ex) - expression{x};
}

expression operator-(expression ex, long double x)
{
    return std::move(ex) - expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator-(expression ex, mppp::real128 x)
{
    return std::move(ex) - expression{x};
}

#endif

expression operator-(double x, expression ex)
{
    return expression{x} - std::move(ex);
}

expression operator-(long double x, expression ex)
{
    return expression{x} - std::move(ex);
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator-(mppp::real128 x, expression ex)
{
    return expression{x} - std::move(ex);
}

#endif

expression operator*(expression ex, double x)
{
    return std::move(ex) * expression{x};
}

expression operator*(expression ex, long double x)
{
    return std::move(ex) * expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator*(expression ex, mppp::real128 x)
{
    return std::move(ex) * expression{x};
}

#endif

expression operator*(double x, expression ex)
{
    return expression{x} * std::move(ex);
}

expression operator*(long double x, expression ex)
{
    return expression{x} * std::move(ex);
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator*(mppp::real128 x, expression ex)
{
    return expression{x} * std::move(ex);
}

#endif

expression operator/(expression ex, double x)
{
    return std::move(ex) / expression{x};
}

expression operator/(expression ex, long double x)
{
    return std::move(ex) / expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator/(expression ex, mppp::real128 x)
{
    return std::move(ex) / expression{x};
}

#endif

expression operator/(double x, expression ex)
{
    return expression{x} / std::move(ex);
}

expression operator/(long double x, expression ex)
{
    return expression{x} / std::move(ex);
}

#if defined(HEYOKA_HAVE_REAL128)

expression operator/(mppp::real128 x, expression ex)
{
    return expression{x} / std::move(ex);
}

#endif

expression &operator+=(expression &x, expression e)
{
    return x = std::move(x) + std::move(e);
}

expression &operator-=(expression &x, expression e)
{
    return x = std::move(x) - std::move(e);
}

expression &operator*=(expression &x, expression e)
{
    return x = std::move(x) * std::move(e);
}

expression &operator/=(expression &x, expression e)
{
    return x = std::move(x) / std::move(e);
}

expression &operator+=(expression &ex, double x)
{
    return ex += expression{x};
}

expression &operator+=(expression &ex, long double x)
{
    return ex += expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator+=(expression &ex, mppp::real128 x)
{
    return ex += expression{x};
}

#endif

expression &operator-=(expression &ex, double x)
{
    return ex -= expression{x};
}

expression &operator-=(expression &ex, long double x)
{
    return ex -= expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator-=(expression &ex, mppp::real128 x)
{
    return ex -= expression{x};
}

#endif

expression &operator*=(expression &ex, double x)
{
    return ex *= expression{x};
}

expression &operator*=(expression &ex, long double x)
{
    return ex *= expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator*=(expression &ex, mppp::real128 x)
{
    return ex *= expression{x};
}

#endif

expression &operator/=(expression &ex, double x)
{
    return ex /= expression{x};
}

expression &operator/=(expression &ex, long double x)
{
    return ex /= expression{x};
}

#if defined(HEYOKA_HAVE_REAL128)

expression &operator/=(expression &ex, mppp::real128 x)
{
    return ex /= expression{x};
}

#endif

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

expression diff(const expression &e, const expression &x)
{
    return std::visit(
        [&e](const auto &v) -> expression {
            if constexpr (std::is_same_v<detail::uncvref_t<decltype(v)>, variable>) {
                return diff(e, v.name());
            } else {
                std::ostringstream oss;
                oss << e;

                throw std::invalid_argument(
                    "Cannot differentiate an expression with respect to the non-variable expression '" + oss.str()
                    + "'");
            }
        },
        x.value());
}

expression subs(const expression &e, const std::unordered_map<std::string, expression> &smap)
{
    return std::visit([&smap](const auto &arg) { return subs(arg, smap); }, e.value());
}

// Pairwise summation of a vector of expressions.
// https://en.wikipedia.org/wiki/Pairwise_summation
expression pairwise_sum(std::vector<expression> sum)
{
    if (sum.size() == std::numeric_limits<decltype(sum.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_sum()");
    }

    if (sum.empty()) {
        return expression{0.};
    }

    while (sum.size() != 1u) {
        std::vector<expression> new_sum;

        for (decltype(sum.size()) i = 0; i < sum.size(); i += 2u) {
            if (i + 1u == sum.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_sum.push_back(std::move(sum[i]));
            } else {
                new_sum.push_back(std::move(sum[i]) + std::move(sum[i + 1u]));
            }
        }

        new_sum.swap(sum);
    }

    return sum[0];
}

double eval_dbl(const expression &e, const std::unordered_map<std::string, double> &map)
{
    return std::visit([&map](const auto &arg) { return eval_dbl(arg, map); }, e.value());
}

void eval_batch_dbl(std::vector<double> &retval, const expression &e,
                    const std::unordered_map<std::string, std::vector<double>> &map)
{
    std::visit([&map, &retval](const auto &arg) { eval_batch_dbl(retval, arg, map); }, e.value());
}

std::vector<std::vector<std::size_t>> compute_connections(const expression &e)
{
    std::vector<std::vector<std::size_t>> node_connections;
    std::size_t node_counter = 0u;
    update_connections(node_connections, e, node_counter);
    return node_connections;
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const expression &e,
                        std::size_t &node_counter)
{
    std::visit([&node_connections,
                &node_counter](const auto &arg) { update_connections(node_connections, arg, node_counter); },
               e.value());
}

std::vector<double> compute_node_values_dbl(const expression &e, const std::unordered_map<std::string, double> &map,
                                            const std::vector<std::vector<std::size_t>> &node_connections)
{
    std::vector<double> node_values(node_connections.size());
    std::size_t node_counter = 0u;
    update_node_values_dbl(node_values, e, map, node_connections, node_counter);
    return node_values;
}

void update_node_values_dbl(std::vector<double> &node_values, const expression &e,
                            const std::unordered_map<std::string, double> &map,
                            const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter)
{
    std::visit([&map, &node_values, &node_connections, &node_counter](
                   const auto &arg) { update_node_values_dbl(node_values, arg, map, node_connections, node_counter); },
               e.value());
}

std::unordered_map<std::string, double> compute_grad_dbl(const expression &e,
                                                         const std::unordered_map<std::string, double> &map,
                                                         const std::vector<std::vector<std::size_t>> &node_connections)
{
    std::unordered_map<std::string, double> grad;
    auto node_values = compute_node_values_dbl(e, map, node_connections);
    std::size_t node_counter = 0u;
    update_grad_dbl(grad, e, map, node_values, node_connections, node_counter);
    return grad;
}

void update_grad_dbl(std::unordered_map<std::string, double> &grad, const expression &e,
                     const std::unordered_map<std::string, double> &map, const std::vector<double> &node_values,
                     const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter,
                     double acc)
{
    std::visit(
        [&map, &grad, &node_values, &node_connections, &node_counter, &acc](const auto &arg) {
            update_grad_dbl(grad, arg, map, node_values, node_connections, node_counter, acc);
        },
        e.value());
}

// Transform in-place ex by decomposition, appending the
// result of the decomposition to u_vars_defs.
// The return value is the index, in u_vars_defs,
// which corresponds to the decomposed version of ex.
// If the return value is zero, ex was not decomposed.
// NOTE: this will render ex unusable.
std::vector<expression>::size_type taylor_decompose_in_place(expression &&ex, std::vector<expression> &u_vars_defs)
{
    return std::visit(
        [&u_vars_defs](auto &&v) { return taylor_decompose_in_place(std::forward<decltype(v)>(v), u_vars_defs); },
        std::move(ex.value()));
}

llvm::Value *taylor_u_init_dbl(llvm_state &s, const expression &e, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    return std::visit([&](const auto &arg) { return taylor_u_init_dbl(s, arg, arr, batch_size); }, e.value());
}

llvm::Value *taylor_u_init_ldbl(llvm_state &s, const expression &e, const std::vector<llvm::Value *> &arr,
                                std::uint32_t batch_size)
{
    return std::visit([&](const auto &arg) { return taylor_u_init_ldbl(s, arg, arr, batch_size); }, e.value());
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_u_init_f128(llvm_state &s, const expression &e, const std::vector<llvm::Value *> &arr,
                                std::uint32_t batch_size)
{
    return std::visit([&](const auto &arg) { return taylor_u_init_f128(s, arg, arr, batch_size); }, e.value());
}

#endif

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_diff_impl(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v) -> llvm::Value * {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, binary_operator> || std::is_same_v<type, func>) {
                return taylor_diff<T>(s, v, arr, n_uvars, order, idx, batch_size);
            } else {
                throw std::invalid_argument(
                    "Taylor derivatives can be computed only for binary operators or functions");
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

llvm::Value *taylor_diff_dbl(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)

{
    return detail::taylor_diff_impl<double>(s, ex, arr, n_uvars, order, idx, batch_size);
}

llvm::Value *taylor_diff_ldbl(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return detail::taylor_diff_impl<long double>(s, ex, arr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_diff_f128(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return detail::taylor_diff_impl<mppp::real128>(s, ex, arr, n_uvars, order, idx, batch_size);
}

#endif

namespace detail
{

namespace
{

template <typename T>
llvm::Function *taylor_c_diff_func_impl(llvm_state &s, const expression &ex, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v) -> llvm::Function * {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, binary_operator> || std::is_same_v<type, func>) {
                return taylor_c_diff_func<T>(s, v, n_uvars, batch_size);
            } else {
                throw std::invalid_argument(
                    "Taylor derivatives in compact mode can be computed only for binary operators or functions");
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

llvm::Function *taylor_c_diff_func_dbl(llvm_state &s, const expression &ex, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    return detail::taylor_c_diff_func_impl<double>(s, ex, n_uvars, batch_size);
}

llvm::Function *taylor_c_diff_func_ldbl(llvm_state &s, const expression &ex, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    return detail::taylor_c_diff_func_impl<long double>(s, ex, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *taylor_c_diff_func_f128(llvm_state &s, const expression &ex, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    return detail::taylor_c_diff_func_impl<mppp::real128>(s, ex, n_uvars, batch_size);
}

#endif

namespace detail
{

// Helper to detect if ex is an integral number.
bool is_integral(const expression &ex)
{
    return std::visit(
        [](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, number>) {
                return std::visit(
                    [](const auto &x) {
                        using std::trunc;

                        return isfinite(x) && x == trunc(x);
                    },
                    v.value());
            } else {
                // Not a number.
                return false;
            }
        },
        ex.value());
}

// Helper to detect if ex is a number in the form n / 2,
// where n is an odd integral value.
bool is_odd_integral_half(const expression &ex)
{
    return std::visit(
        [](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, number>) {
                return std::visit(
                    [](const auto &x) {
                        using std::trunc;

                        if (!isfinite(x) || x == trunc(x)) {
                            // x is not finite, or it is already
                            // an integral value.
                            return false;
                        }

                        const auto y = 2 * x;

                        return isfinite(y) && y == trunc(y);
                    },
                    v.value());
            } else {
                // Not a number.
                return false;
            }
        },
        ex.value());
}

} // namespace detail

} // namespace heyoka
