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
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/math/neg.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/math/tpoly.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
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

expression::expression(func f) : m_value(std::move(f)) {}

expression::expression(param p) : m_value(std::move(p)) {}

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
                using namespace fmt::literals;

                throw std::invalid_argument(
                    "Cannot apply the prime() operator to the non-variable expression '{}'"_format(e));
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
    if (auto num_ptr = std::get_if<number>(&e.value())) {
        return expression{-std::move(*num_ptr)};
    } else {
        if (auto fptr = detail::is_neg(e)) {
            // Simplify -(-x) to x.
            auto rng = fptr->get_mutable_args_it();
            assert(rng.first != rng.second);
            return std::move(*rng.first);
        } else {
            return neg(std::move(e));
        }
    }
}

expression operator+(expression e1, expression e2)
{
    // Simplify x + neg(y) to x - y.
    if (auto fptr = detail::is_neg(e2)) {
        auto rng = fptr->get_mutable_args_it();
        assert(rng.first != rng.second);
        return std::move(e1) - std::move(*rng.first);
    }

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
            if constexpr (std::is_same_v<func, type2>) {
                if (auto pbop = v2.template extract<detail::binary_op>();
                    pbop != nullptr && pbop->op() == detail::binary_op::type::add
                    && std::holds_alternative<number>(pbop->args()[0].value())) {
                    // e2 = a + x, where a is a number. Simplify e1 + (a + x) -> c + x, where c = e1 + a.
                    auto rng2 = pbop->get_mutable_args_it();

                    return expression{std::forward<decltype(v1)>(v1)} + std::move(*rng2.first)
                           + std::move(*(rng2.first + 1));
                }
            }

            // NOTE: fall through the standard case.
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 symbolic, e2 number. Swap the operands so that the number comes first.
            return expression{std::forward<decltype(v2)>(v2)} + expression{std::forward<decltype(v1)>(v1)};
        }

        // The standard case.
        return add(expression{std::forward<decltype(v1)>(v1)}, expression{std::forward<decltype(v2)>(v2)});
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator-(expression e1, expression e2)
{
    // Simplify x - (-y) to x + y.
    if (auto fptr = detail::is_neg(e2)) {
        auto rng = fptr->get_mutable_args_it();
        assert(rng.first != rng.second);
        return std::move(e1) + std::move(*rng.first);
    }

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
            // e1 symbolic, e2 number. Turn e1 - e2 into e1 + (-e2),
            // because addition provides more simplification capabilities.
            return expression{std::forward<decltype(v1)>(v1)} + expression{-std::forward<decltype(v2)>(v2)};
        }

        // The standard case.
        return sub(expression{std::forward<decltype(v1)>(v1)}, expression{std::forward<decltype(v2)>(v2)});
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator*(expression e1, expression e2)
{
    auto fptr1 = detail::is_neg(e1);
    auto fptr2 = detail::is_neg(e2);

    if (fptr1 != nullptr && fptr2 != nullptr) {
        // Simplify (-x) * (-y) into x*y.
        auto rng1 = fptr1->get_mutable_args_it();
        auto rng2 = fptr2->get_mutable_args_it();

        assert(rng1.first != rng1.second);
        assert(rng2.first != rng2.second);

        return std::move(*rng1.first) * std::move(*rng2.first);
    }

    // Simplify x*x -> square(x) if x is not a number (otherwise,
    // we will numerically compute the result below).
    if (e1 == e2 && !std::holds_alternative<number>(e1.value())) {
        return square(std::move(e1));
    }

    auto visitor = [fptr2](auto &&v1, auto &&v2) {
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
            }
            if (is_one(v1)) {
                // 1 * e2 = e2.
                return expression{std::forward<decltype(v2)>(v2)};
            }
            if (fptr2 != nullptr) {
                // a * (-x) = (-a) * x.
                auto rng2 = fptr2->get_mutable_args_it();
                assert(rng2.first != rng2.second);

                return expression{-std::forward<decltype(v1)>(v1)} * std::move(*rng2.first);
            }
            if constexpr (std::is_same_v<func, type2>) {
                if (auto pbop = v2.template extract<detail::binary_op>();
                    pbop != nullptr && pbop->op() == detail::binary_op::type::mul
                    && std::holds_alternative<number>(pbop->args()[0].value())) {
                    // e2 = a * x, where a is a number. Simplify e1 * (a * x) -> c * x, where c = e1 * a.
                    auto rng2 = pbop->get_mutable_args_it();

                    return expression{std::forward<decltype(v1)>(v1)} * std::move(*rng2.first)
                           * std::move(*(rng2.first + 1));
                }
            }

            // NOTE: fall through the standard case.
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 symbolic, e2 number. Swap the operands so that the number comes first.
            return expression{std::forward<decltype(v2)>(v2)} * expression{std::forward<decltype(v1)>(v1)};
        }

        // The standard case.
        return mul(expression{std::forward<decltype(v1)>(v1)}, expression{std::forward<decltype(v2)>(v2)});
    };

    return std::visit(visitor, std::move(e1.value()), std::move(e2.value()));
}

expression operator/(expression e1, expression e2)
{
    auto fptr1 = detail::is_neg(e1);
    auto fptr2 = detail::is_neg(e2);

    if (fptr1 != nullptr && fptr2 != nullptr) {
        // Simplify (-x) / (-y) into x/y.
        auto rng1 = fptr1->get_mutable_args_it();
        auto rng2 = fptr2->get_mutable_args_it();

        assert(rng1.first != rng1.second);
        assert(rng2.first != rng2.second);

        return std::move(*rng1.first) / std::move(*rng2.first);
    }

    auto visitor = [fptr1, fptr2](auto &&v1, auto &&v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type2, number>) {
            // If the divisor is zero, always raise an error.
            if (is_zero(v2)) {
                throw zero_division_error("Division by zero");
            }
        }

        if constexpr (std::is_same_v<type1, number> && std::is_same_v<type2, number>) {
            // Both are numbers, divide them.
            return expression{std::forward<decltype(v1)>(v1) / std::forward<decltype(v2)>(v2)};
        } else if constexpr (std::is_same_v<type2, number>) {
            // e1 is symbolic, e2 a number.
            if (is_one(v2)) {
                // e1 / 1 = e1.
                return expression{std::forward<decltype(v1)>(v1)};
            }
            if (is_negative_one(v2)) {
                // e1 / -1 = -e1.
                return -expression{std::forward<decltype(v1)>(v1)};
            }
            if (fptr1 != nullptr) {
                // (-e1) / a = e1 / (-a).
                auto rng1 = fptr1->get_mutable_args_it();
                assert(rng1.first != rng1.second);

                return std::move(*rng1.first) / expression{-std::forward<decltype(v2)>(v2)};
            }
            if constexpr (std::is_same_v<func, type1>) {
                if (auto pbop = v1.template extract<detail::binary_op>();
                    pbop != nullptr && pbop->op() == detail::binary_op::type::div
                    && std::holds_alternative<number>(pbop->args()[1].value())) {
                    // e1 = x / a, where a is a number. Simplify (x / a) / b -> x / (a * b).
                    auto rng1 = pbop->get_mutable_args_it();

                    return std::move(*rng1.first)
                           / (std::move(*(rng1.first + 1)) * expression{std::forward<decltype(v2)>(v2)});
                }
            }

            // NOTE: fall through to the standard case.
        } else if constexpr (std::is_same_v<type1, number>) {
            // e1 is a number, e2 is symbolic.
            if (is_zero(v1)) {
                // 0 / e2 == 0.
                return expression{number{0.}};
            }
            if (fptr2 != nullptr) {
                // a / (-e2) = (-a) / e2.
                auto rng2 = fptr2->get_mutable_args_it();
                assert(rng2.first != rng2.second);

                return expression{-std::forward<decltype(v1)>(v1)} / std::move(*rng2.first);
            }

            // NOTE: fall through to the standard case.
        }

        // The standard case.
        return div(expression{std::forward<decltype(v1)>(v1)}, expression{std::forward<decltype(v2)>(v2)});
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

std::size_t get_n_nodes(const expression &e)
{
    return std::visit(
        [](const auto &arg) -> std::size_t {
            if constexpr (std::is_same_v<func, detail::uncvref_t<decltype(arg)>>) {
                std::size_t retval = 1;

                for (const auto &ex : arg.args()) {
                    retval += get_n_nodes(ex);
                }

                return retval;
            } else {
                return 1;
            }
        },
        e.value());
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
                using namespace fmt::literals;

                throw std::invalid_argument(
                    "Cannot differentiate an expression with respect to the non-variable expression '{}'"_format(e));
            }
        },
        x.value());
}

expression subs(const expression &e, const std::unordered_map<std::string, expression> &smap)
{
    return std::visit([&smap](const auto &arg) { return subs(arg, smap); }, e.value());
}

namespace detail
{

namespace
{

// Pairwise reduction of a vector of expressions.
template <typename F>
expression pairwise_reduce(const F &func, std::vector<expression> list)
{
    assert(!list.empty());

    // LCOV_EXCL_START
    if (list.size() == std::numeric_limits<decltype(list.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_reduce()");
    }
    // LCOV_EXCL_STOP

    while (list.size() != 1u) {
        const auto cur_size = list.size();

        // Init the new list. The size will be halved, +1 if the
        // current size is odd.
        const auto next_size = cur_size / 2u + cur_size % 2u;
        std::vector<expression> new_list(next_size);

        tbb::parallel_for(tbb::blocked_range<decltype(new_list.size())>(0, new_list.size()),
                          [&list, &new_list, cur_size, &func](const auto &r) {
                              for (auto i = r.begin(); i != r.end(); ++i) {
                                  if (i * 2u == cur_size - 1u) {
                                      // list has an odd size, and we are at the last element of list.
                                      // Just move it to new_list.
                                      new_list[i] = std::move(list.back());
                                  } else {
                                      new_list[i] = func(std::move(list[i * 2u]), std::move(list[i * 2u + 1u]));
                                  }
                              }
                          });

        new_list.swap(list);
    }

    return std::move(list[0]);
}

} // namespace

} // namespace detail

// Pairwise summation of a vector of expressions.
// https://en.wikipedia.org/wiki/Pairwise_summation
expression pairwise_sum(std::vector<expression> sum)
{
    if (sum.empty()) {
        return expression{0.};
    }

    return detail::pairwise_reduce([](expression &&a, expression &&b) { return std::move(a) + std::move(b); },
                                   std::move(sum));
}

// Pairwise product.
expression pairwise_prod(std::vector<expression> prod)
{
    if (prod.empty()) {
        return expression{1.};
    }

    return detail::pairwise_reduce([](expression &&a, expression &&b) { return std::move(a) * std::move(b); },
                                   std::move(prod));
}

double eval_dbl(const expression &e, const std::unordered_map<std::string, double> &map,
                const std::vector<double> &pars)
{
    return std::visit([&](const auto &arg) { return eval_dbl(arg, map, pars); }, e.value());
}

long double eval_ldbl(const expression &e, const std::unordered_map<std::string, long double> &map,
                      const std::vector<long double> &pars)
{
    return std::visit([&](const auto &arg) { return eval_ldbl(arg, map, pars); }, e.value());
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 eval_f128(const expression &e, const std::unordered_map<std::string, mppp::real128> &map,
                        const std::vector<mppp::real128> &pars)
{
    return std::visit([&](const auto &arg) { return eval_f128(arg, map, pars); }, e.value());
}
#endif

void eval_batch_dbl(std::vector<double> &retval, const expression &e,
                    const std::unordered_map<std::string, std::vector<double>> &map, const std::vector<double> &pars)
{
    std::visit([&](const auto &arg) { eval_batch_dbl(retval, arg, map, pars); }, e.value());
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
taylor_dc_t::size_type taylor_decompose_in_place(expression &&ex, taylor_dc_t &u_vars_defs)
{
    if (auto fptr = std::get_if<func>(&ex.value())) {
        return taylor_decompose_in_place(std::move(*fptr), u_vars_defs);
    } else {
        return 0;
    }
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_diff_impl(llvm_state &s, const expression &ex, const std::vector<std::uint32_t> &deps,
                              const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return std::visit(
        [&](const auto &v) -> llvm::Value * {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                return taylor_diff<T>(s, v, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
            } else {
                throw std::invalid_argument("Taylor derivatives can be computed only for functions");
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

llvm::Value *taylor_diff_dbl(llvm_state &s, const expression &ex, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)

{
    return detail::taylor_diff_impl<double>(s, ex, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *taylor_diff_ldbl(llvm_state &s, const expression &ex, const std::vector<std::uint32_t> &deps,
                              const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return detail::taylor_diff_impl<long double>(s, ex, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_diff_f128(llvm_state &s, const expression &ex, const std::vector<std::uint32_t> &deps,
                              const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return detail::taylor_diff_impl<mppp::real128>(s, ex, deps, arr, par_ptr, time_ptr, n_uvars, order, idx,
                                                   batch_size);
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

            if constexpr (std::is_same_v<type, func>) {
                return taylor_c_diff_func<T>(s, v, n_uvars, batch_size);
            } else {
                throw std::invalid_argument("Taylor derivatives in compact mode can be computed only for functions");
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
                        using std::isfinite;

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
                        using std::isfinite;

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

expression par_impl::operator[](std::uint32_t idx) const
{
    return expression{param{idx}};
}

} // namespace detail

// Determine the size of the parameter vector from the highest
// param index appearing in an expression. If the return value
// is zero, no params appear in the expression.
std::uint32_t get_param_size(const expression &ex)
{
    std::uint32_t retval = 0;

    std::visit(
        [&retval](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, param>) {
                if (v.idx() == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("Overflow dected in get_n_param()");
                }

                retval = std::max(static_cast<std::uint32_t>(v.idx() + 1u), retval);
            } else if constexpr (std::is_same_v<type, func>) {
                for (const auto &a : v.args()) {
                    retval = std::max(get_param_size(a), retval);
                }
            }
        },
        ex.value());

    return retval;
}

// Determine if an expression is time-dependent.
bool has_time(const expression &ex)
{
    // If the expression itself is a time function or a tpoly,
    // return true.
    if (detail::is_time(ex) || detail::is_tpoly(ex)) {
        return true;
    }

    // Otherwise:
    // - if ex is a function, check if any of its arguments
    //   is time-dependent,
    // - otherwise, return false.
    return std::visit(
        [](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                for (const auto &a : v.args()) {
                    if (has_time(a)) {
                        return true;
                    }
                }
            }

            return false;
        },
        ex.value());
}

} // namespace heyoka
