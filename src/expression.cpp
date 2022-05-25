// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/math/neg.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/sum.hpp>
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

namespace detail
{

namespace
{

expression copy_impl(std::unordered_map<const void *, expression> &func_map, const expression &e)
{
    return std::visit(
        [&func_map](const auto &v) {
            if constexpr (std::is_same_v<detail::uncvref_t<decltype(v)>, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already copied the current function, fetch the copy
                    // from the cache.
                    return it->second;
                }

                // Create a copy of v. Note that this will copy
                // the arguments of v via their copy constructor,
                // and thus any argument which is itself a function
                // will be shallow-copied.
                auto f_copy = v.copy();

                // Perform a copy of the arguments of v which are functions.
                assert(v.args().size() == f_copy.args().size()); // LCOV_EXCL_LINE
                auto b1 = v.args().begin();
                for (auto [b2, e2] = f_copy.get_mutable_args_it(); b2 != e2; ++b1, ++b2) {
                    // NOTE: the argument needs to be copied via a recursive
                    // call to copy_impl() only if it is a func. Otherwise, the copy
                    // we made earlier via the copy constructor is already a deep copy.
                    if (std::holds_alternative<func>(b1->value())) {
                        *b2 = copy_impl(func_map, *b1);
                    }
                }

                // Construct the return value and put it into the cache.
                auto ex = expression{std::move(f_copy)};
                [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, ex});
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return ex;
            } else {
                return expression{v};
            }
        },
        e.value());
}

} // namespace

} // namespace detail

expression copy(const expression &e)
{
    std::unordered_map<const void *, expression> func_map;

    return detail::copy_impl(func_map, e);
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
                throw std::invalid_argument(
                    fmt::format("Cannot apply the prime() operator to the non-variable expression '{}'", e));
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

namespace detail
{

namespace
{

void get_variables(std::unordered_set<const void *> &func_set, std::set<std::string> &s_set, const expression &e)
{
    std::visit(
        [&func_set, &s_set](const auto &arg) {
            using type = detail::uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = arg.get_ptr();

                if (func_set.find(f_id) != func_set.end()) {
                    // We already determined the list of variables for the
                    // current function, exit.
                    return;
                }

                // Determine the list of variables for each
                // function argument.
                for (const auto &farg : arg.args()) {
                    get_variables(func_set, s_set, farg);
                }

                // Add the id of f to the set.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            } else if constexpr (std::is_same_v<type, variable>) {
                s_set.insert(arg.name());
            }
        },
        e.value());
}

void rename_variables(std::unordered_set<const void *> &func_set, expression &e,
                      const std::unordered_map<std::string, std::string> &repl_map)
{
    std::visit(
        [&func_set, &repl_map](auto &arg) {
            using type = detail::uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = arg.get_ptr();

                if (func_set.find(f_id) != func_set.end()) {
                    // We already renamed variables for the current function,
                    // just return.
                    return;
                }

                for (auto [beg, end] = arg.get_mutable_args_it(); beg != end; ++beg) {
                    rename_variables(func_set, *beg, repl_map);
                }

                // Add the id of f to the set.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            } else if constexpr (std::is_same_v<type, variable>) {
                if (auto it = repl_map.find(arg.name()); it != repl_map.end()) {
                    arg.name() = it->second;
                }
            }
        },
        e.value());
}

} // namespace

} // namespace detail

std::vector<std::string> get_variables(const expression &e)
{
    std::unordered_set<const void *> func_set;
    std::set<std::string> s_set;

    detail::get_variables(func_set, s_set, e);

    return std::vector<std::string>(s_set.begin(), s_set.end());
}

void rename_variables(expression &e, const std::unordered_map<std::string, std::string> &repl_map)
{
    std::unordered_set<const void *> func_set;

    detail::rename_variables(func_set, e, repl_map);
}

void swap(expression &ex0, expression &ex1) noexcept
{
    std::swap(ex0.value(), ex1.value());
}

// NOTE: this implementation does not take advantage of potentially
// repeating subexpressions. This is not currently a problem because
// hashing is needed only in the CSE for the decomposition, which involves
// only trivial expressions. However, this would likely be needed by a to_sympy()
// implementation in heyoka.py which allows for a dictionary of custom
// substitutions to be provided by the user.
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
        // Simplify -number to its numerical value.
        return expression{-std::move(*num_ptr)};
    } else {
        if (auto fptr = detail::is_neg(e)) {
            // Simplify -(-x) to x.
            assert(!fptr->args().empty()); // LCOV_EXCL_LINE
            return fptr->args()[0];
        } else {
            return neg(std::move(e));
        }
    }
}

expression operator+(expression e1, expression e2)
{
    // Simplify x + neg(y) to x - y.
    if (auto fptr = detail::is_neg(e2)) {
        assert(!fptr->args().empty()); // LCOV_EXCL_LINE
        return std::move(e1) - fptr->args()[0];
    }

    auto visitor = [](auto &&v1, auto &&v2) {
        using type1 = detail::uncvref_t<decltype(v1)>;
        using type2 = detail::uncvref_t<decltype(v2)>;

        if constexpr (std::is_same_v<type1, number> && std::is_same_v<type2, number>) {
            // Both are numbers, add them and return the result.
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
                    return expression{std::forward<decltype(v1)>(v1)} + pbop->args()[0] + pbop->args()[1];
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
        assert(!fptr->args().empty()); // LCOV_EXCL_LINE
        return std::move(e1) + fptr->args()[0];
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
        assert(!fptr1->args().empty()); // LCOV_EXCL_LINE
        assert(!fptr2->args().empty()); // LCOV_EXCL_LINE
        return fptr1->args()[0] * fptr2->args()[0];
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
            if (is_negative_one(v1)) {
                // -1 * e2 = -e2.
                return -expression{std::forward<decltype(v2)>(v2)};
            }
            if (fptr2 != nullptr) {
                // a * (-x) = (-a) * x.
                assert(!fptr2->args().empty()); // LCOV_EXCL_LINE
                return expression{-std::forward<decltype(v1)>(v1)} * fptr2->args()[0];
            }
            if constexpr (std::is_same_v<func, type2>) {
                if (auto pbop = v2.template extract<detail::binary_op>();
                    pbop != nullptr && pbop->op() == detail::binary_op::type::mul
                    && std::holds_alternative<number>(pbop->args()[0].value())) {
                    // e2 = a * x, where a is a number. Simplify e1 * (a * x) -> c * x, where c = e1 * a.
                    return expression{std::forward<decltype(v1)>(v1)} * pbop->args()[0] * pbop->args()[1];
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
        assert(!fptr1->args().empty()); // LCOV_EXCL_LINE
        assert(!fptr2->args().empty()); // LCOV_EXCL_LINE
        return fptr1->args()[0] / fptr2->args()[0];
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
                assert(!fptr1->args().empty()); // LCOV_EXCL_LINE
                return fptr1->args()[0] / expression{-std::forward<decltype(v2)>(v2)};
            }
            if constexpr (std::is_same_v<func, type1>) {
                if (auto pbop = v1.template extract<detail::binary_op>();
                    pbop != nullptr && pbop->op() == detail::binary_op::type::div
                    && std::holds_alternative<number>(pbop->args()[1].value())) {
                    // e1 = x / a, where a is a number. Simplify (x / a) / b -> x / (a * b).
                    return pbop->args()[0] / (pbop->args()[1] * expression{std::forward<decltype(v2)>(v2)});
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
                assert(!fptr2->args().empty()); // LCOV_EXCL_LINE
                return expression{-std::forward<decltype(v1)>(v1)} / fptr2->args()[0];
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

namespace detail
{

namespace
{

std::size_t get_n_nodes(std::unordered_map<const void *, std::size_t> &func_map, const expression &e)
{
    return std::visit(
        [&func_map](const auto &arg) -> std::size_t {
            if constexpr (std::is_same_v<func, detail::uncvref_t<decltype(arg)>>) {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already computed the number of nodes for the current
                    // function, return it.
                    return it->second;
                }

                std::size_t retval = 1;
                for (const auto &ex : arg.args()) {
                    retval += get_n_nodes(func_map, ex);
                }

                // Store the number of nodes for the current function
                // in the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, retval});
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return retval;
            } else {
                return 1;
            }
        },
        e.value());
}

} // namespace

} // namespace detail

std::size_t get_n_nodes(const expression &e)
{
    std::unordered_map<const void *, std::size_t> func_map;

    return detail::get_n_nodes(func_map, e);
}

namespace detail
{

expression diff(std::unordered_map<const void *, expression> &func_map, const expression &e, const std::string &s)
{
    return std::visit(
        [&func_map, &s](const auto &arg) {
            using type = detail::uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, number>) {
                return std::visit([](const auto &v) { return expression{number{detail::uncvref_t<decltype(v)>(0)}}; },
                                  arg.value());
            } else if constexpr (std::is_same_v<type, param>) {
                // NOTE: if we ever implement single-precision support,
                // this should be probably changed into 0_flt (i.e., the lowest
                // precision numerical type), so that it does not trigger
                // type promotions in numerical constants. Other similar
                // occurrences as well (e.g., diff for variable).
                return 0_dbl;
            } else if constexpr (std::is_same_v<type, variable>) {
                if (s == arg.name()) {
                    return 1_dbl;
                } else {
                    return 0_dbl;
                }
            } else {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already performed diff on the current function,
                    // fetch the result from the cache.
                    return it->second;
                }

                auto ret = arg.diff(func_map, s);

                // Put the return value in the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, ret});
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            }
        },
        e.value());
}

expression diff(std::unordered_map<const void *, expression> &func_map, const expression &e, const param &p)
{
    return std::visit(
        [&func_map, &p](const auto &arg) {
            using type = detail::uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, number>) {
                return std::visit([](const auto &v) { return expression{number{detail::uncvref_t<decltype(v)>(0)}}; },
                                  arg.value());
            } else if constexpr (std::is_same_v<type, param>) {
                if (p.idx() == arg.idx()) {
                    return 1_dbl;
                } else {
                    return 0_dbl;
                }
            } else if constexpr (std::is_same_v<type, variable>) {
                return 0_dbl;
            } else {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already performed diff on the current function,
                    // fetch the result from the cache.
                    return it->second;
                }

                auto ret = arg.diff(func_map, p);

                // Put the return value in the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, ret});
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            }
        },
        e.value());
}

} // namespace detail

expression diff(const expression &e, const std::string &s)
{
    std::unordered_map<const void *, expression> func_map;

    return detail::diff(func_map, e, s);
}

expression diff(const expression &e, const param &p)
{
    std::unordered_map<const void *, expression> func_map;

    return detail::diff(func_map, e, p);
}

expression diff(const expression &e, const expression &x)
{
    return std::visit(
        [&e](const auto &v) -> expression {
            if constexpr (std::is_same_v<detail::uncvref_t<decltype(v)>, variable>) {
                return diff(e, v.name());
            } else if constexpr (std::is_same_v<detail::uncvref_t<decltype(v)>, param>) {
                return diff(e, v);
            } else {
                throw std::invalid_argument(
                    "Derivatives are currently supported only with respect to variables and parameters");
            }
        },
        x.value());
}

namespace detail
{

namespace
{

// NOTE: an in-place API would perform better.
expression subs(std::unordered_map<const void *, expression> &func_map, const expression &ex,
                const std::unordered_map<std::string, expression> &smap)
{
    return std::visit(
        [&func_map, &smap](const auto &arg) {
            using type = detail::uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, number> || std::is_same_v<type, param>) {
                return expression{arg};
            } else if constexpr (std::is_same_v<type, variable>) {
                if (auto it = smap.find(arg.name()); it == smap.end()) {
                    return expression{arg};
                } else {
                    return it->second;
                }
            } else {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already performed substitution on the current function,
                    // fetch the result from the cache.
                    return it->second;
                }

                // NOTE: this creates a separate instance of arg, but its
                // arguments are shallow-copied.
                auto tmp = arg.copy();

                for (auto [b, e] = tmp.get_mutable_args_it(); b != e; ++b) {
                    *b = subs(func_map, *b, smap);
                }

                // Put the return value in the cache.
                auto ret = expression{std::move(tmp)};
                [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, ret});
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

expression subs(const expression &e, const std::unordered_map<std::string, expression> &smap)
{
    std::unordered_map<const void *, expression> func_map;

    return detail::subs(func_map, e, smap);
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

// Pairwise product.
expression pairwise_prod(std::vector<expression> prod)
{
    if (prod.empty()) {
        return 1_dbl;
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

namespace detail
{

taylor_dc_t::size_type taylor_decompose(std::unordered_map<const void *, taylor_dc_t::size_type> &func_map,
                                        const expression &ex, taylor_dc_t &dc)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        return fptr->taylor_decompose(func_map, dc);
    } else {
        return 0;
    }
}

} // namespace detail

// Decompose ex into dc. The return value is the index, in dc,
// which corresponds to the decomposed version of ex.
// If the return value is zero, ex was not decomposed.
taylor_dc_t::size_type taylor_decompose(const expression &ex, taylor_dc_t &dc)
{
    std::unordered_map<const void *, taylor_dc_t::size_type> func_map;

    return detail::taylor_decompose(func_map, ex, dc);
}

template <typename T>
llvm::Value *taylor_diff(llvm_state &s, const expression &ex, const std::vector<std::uint32_t> &deps,
                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size,
                         bool high_accuracy)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        if constexpr (std::is_same_v<T, double>) {
            return fptr->taylor_diff_dbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                         high_accuracy);
        } else if constexpr (std::is_same_v<T, long double>) {
            return fptr->taylor_diff_ldbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                          high_accuracy);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            return fptr->taylor_diff_f128(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                          high_accuracy);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument("Taylor derivatives can be computed only for functions");
        // LCOV_EXCL_STOP
    }
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff<double>(llvm_state &, const expression &,
                                                            const std::vector<std::uint32_t> &,
                                                            const std::vector<llvm::Value *> &, llvm::Value *,
                                                            llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                                            std::uint32_t, bool);

template HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff<long double>(llvm_state &, const expression &,
                                                                 const std::vector<std::uint32_t> &,
                                                                 const std::vector<llvm::Value *> &, llvm::Value *,
                                                                 llvm::Value *, std::uint32_t, std::uint32_t,
                                                                 std::uint32_t, std::uint32_t, bool);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff<mppp::real128>(llvm_state &, const expression &,
                                                                   const std::vector<std::uint32_t> &,
                                                                   const std::vector<llvm::Value *> &, llvm::Value *,
                                                                   llvm::Value *, std::uint32_t, std::uint32_t,
                                                                   std::uint32_t, std::uint32_t, bool);
#endif

template <typename T>
llvm::Function *taylor_c_diff_func(llvm_state &s, const expression &ex, std::uint32_t n_uvars, std::uint32_t batch_size,
                                   bool high_accuracy)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        if constexpr (std::is_same_v<T, double>) {
            return fptr->taylor_c_diff_func_dbl(s, n_uvars, batch_size, high_accuracy);
        } else if constexpr (std::is_same_v<T, long double>) {
            return fptr->taylor_c_diff_func_ldbl(s, n_uvars, batch_size, high_accuracy);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            return fptr->taylor_c_diff_func_f128(s, n_uvars, batch_size, high_accuracy);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument("Taylor derivatives in compact mode can be computed only for functions");
        // LCOV_EXCL_STOP
    }
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func<double>(llvm_state &, const expression &, std::uint32_t,
                                                                      std::uint32_t, bool);

template HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func<long double>(llvm_state &, const expression &,
                                                                           std::uint32_t, std::uint32_t, bool);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func<mppp::real128>(llvm_state &, const expression &,
                                                                             std::uint32_t, std::uint32_t, bool);

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

                        // NOTE: here we will be assuming that, for all supported
                        // float types, multiplication by 2 is exact.
                        // Since we are assuming IEEE floats anyway, we should be
                        // safe here.
                        // NOTE: y should never become infinity here, because this would mean
                        // that x is integral (since large float values are all integrals anyway).
                        const auto y = 2 * x;
                        return y == trunc(y);
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

namespace detail
{

namespace
{

std::uint32_t get_param_size(std::unordered_set<const void *> &func_set, const expression &ex)
{
    std::uint32_t retval = 0;

    std::visit(
        [&retval, &func_set](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, param>) {
                if (v.idx() == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("Overflow dected in get_n_param()");
                }

                retval = std::max(static_cast<std::uint32_t>(v.idx() + 1u), retval);
            } else if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_set.find(f_id); it != func_set.end()) {
                    // We already computed the number of params for the current
                    // function, exit.
                    return;
                }

                for (const auto &a : v.args()) {
                    retval = std::max(get_param_size(func_set, a), retval);
                }

                // Update the cache.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            }
        },
        ex.value());

    return retval;
}

} // namespace

} // namespace detail

// Determine the size of the parameter vector from the highest
// param index appearing in an expression. If the return value
// is zero, no params appear in the expression.
std::uint32_t get_param_size(const expression &ex)
{
    std::unordered_set<const void *> func_set;

    return detail::get_param_size(func_set, ex);
}

namespace detail
{

namespace
{

bool has_time(std::unordered_set<const void *> &func_set, const expression &ex)
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
        [&func_set](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_set.find(f_id); it != func_set.end()) {
                    // We already determined if this function contains time,
                    // return false (if the function does contain time, the first
                    // time it was encountered we returned true and we could not
                    // possibly end up here).
                    return false;
                }

                // Update the cache.
                // NOTE: do it earlier than usual in order to avoid having
                // to repeat this code twice for the two paths below.
                func_set.insert(f_id);

                for (const auto &a : v.args()) {
                    if (has_time(func_set, a)) {
                        return true;
                    }
                }
            }

            return false;
        },
        ex.value());
}

} // namespace

// A couple of helpers for deep-copying containers of expressions.
std::vector<expression> copy(const std::vector<expression> &v_ex)
{
    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    std::transform(v_ex.begin(), v_ex.end(), std::back_inserter(ret), [](const expression &e) { return copy(e); });

    return ret;
}

std::vector<std::pair<expression, expression>> copy(const std::vector<std::pair<expression, expression>> &v)
{
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(v.size());

    std::transform(v.begin(), v.end(), std::back_inserter(ret), [](const auto &p) {
        return std::pair{copy(p.first), copy(p.second)};
    });

    return ret;
}

std::optional<std::vector<expression>::size_type>
decompose(std::unordered_map<const void *, std::vector<expression>::size_type> &func_map, const expression &ex,
          std::vector<expression> &dc)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        return fptr->decompose(func_map, dc);
    } else {
        return {};
    }
}

namespace
{

// LCOV_EXCL_START

#if !defined(NDEBUG)

// Helper to verify a function decomposition.
void verify_function_dec(const std::vector<expression> &orig, const std::vector<expression> &dc,
                         std::vector<expression>::size_type nvars)
{
    using idx_t = std::vector<expression>::size_type;

    // Cache the number of outputs.
    const auto nouts = orig.size();

    assert(dc.size() >= nouts);

    // The first nvars expressions of u variables
    // must be just variables.
    for (idx_t i = 0; i < nvars; ++i) {
        assert(std::holds_alternative<variable>(dc[i].value()));
    }

    // From nvars to dc.size() - nouts, the expressions
    // must be functions whose arguments
    // are either variables in the u_n form,
    // where n < i, or numbers/params.
    for (auto i = nvars; i < dc.size() - nouts; ++i) {
        std::visit(
            [i](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    for (const auto &arg : v.args()) {
                        if (auto p_var = std::get_if<variable>(&arg.value())) {
                            assert(p_var->name().rfind("u_", 0) == 0);
                            assert(uname_to_index(p_var->name()) < i);
                        } else if (std::get_if<number>(&arg.value()) == nullptr
                                   && std::get_if<param>(&arg.value()) == nullptr) {
                            assert(false);
                        }
                    }
                } else {
                    assert(false);
                }
            },
            dc[i].value());
    }

    // From dc.size() - nouts to dc.size(), the expressions
    // must be either variables in the u_n form, where n < i,
    // or numbers/params.
    for (auto i = dc.size() - nouts; i < dc.size(); ++i) {
        std::visit(
            [i](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    assert(v.name().rfind("u_", 0) == 0);
                    assert(uname_to_index(v.name()) < i);
                } else if constexpr (!std::is_same_v<type, number> && !std::is_same_v<type, param>) {
                    assert(false);
                }
            },
            dc[i].value());
    }

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of the original variables or other u variables,
    // and store it in subs_map.
    for (idx_t i = 0; i < dc.size() - nouts; ++i) {
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map));
    }

    // Reconstruct the function components
    // and compare them to the original ones.
    for (auto i = dc.size() - nouts; i < dc.size(); ++i) {
        assert(subs(dc[i], subs_map) == orig[i - (dc.size() - nouts)]);
    }
}

#endif

// LCOV_EXCL_STOP

// Simplify a function decomposition by removing
// common subexpressions.
std::vector<expression> function_decompose_cse(std::vector<expression> &v_ex, std::vector<expression>::size_type nvars,
                                               std::vector<expression>::size_type nouts)
{
    using idx_t = std::vector<expression>::size_type;

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Cache the original size for logging later.
    const auto orig_size = v_ex.size();

    // A function decomposition is supposed
    // to have nvars variables at the beginning,
    // nouts variables at the end and possibly
    // extra variables in the middle.
    assert(v_ex.size() >= nouts + nvars);

    // Init the return value.
    std::vector<expression> retval;

    // expression -> idx map. This will end up containing
    // all the unique expressions from v_ex, and it will
    // map them to their indices in retval (which will
    // in general differ from their indices in v_ex).
    std::unordered_map<expression, idx_t> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // The first nvars definitions are just renaming
    // of the original variables into u variables.
    for (idx_t i = 0; i < nvars; ++i) {
        assert(std::holds_alternative<variable>(v_ex[i].value()));
        retval.push_back(std::move(v_ex[i]));

        // NOTE: the u vars that correspond to the original
        // variables are never simplified,
        // thus map them onto themselves.
        [[maybe_unused]] const auto res = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Handle the u variables which do not correspond to the original variables.
    for (auto i = nvars; i < v_ex.size() - nouts; ++i) {
        auto &ex = v_ex[i];

        // Rename the u variables in ex.
        rename_variables(ex, uvars_rename);

        if (auto it = ex_map.find(ex); it == ex_map.end()) {
            // This is the first occurrence of ex in the
            // decomposition. Add it to retval.
            retval.push_back(ex);

            // Add ex to ex_map, mapping it to
            // the index it corresponds to in retval
            // (let's call it j).
            ex_map.emplace(std::move(ex), retval.size() - 1u);

            // Update uvars_rename. This will ensure that
            // occurrences of the variable 'u_i' in the next
            // elements of v_ex will be renamed to 'u_j'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", retval.size() - 1u));
            assert(res.second);
        } else {
            // ex is redundant. This means
            // that it already appears in retval at index
            // it->second. Don't add anything to retval,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", it->second));
            assert(res.second);
        }
    }

    // Handle the definitions of the outputs at the end of the decomposition.
    // We just need to ensure that
    // the u variables in their definitions are renamed with
    // the new indices.
    for (auto i = v_ex.size() - nouts; i < v_ex.size(); ++i) {
        auto &ex = v_ex[i];

        // NOTE: here we expect only vars, numbers or params.
        assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
               || std::holds_alternative<param>(ex.value()));

        rename_variables(ex, uvars_rename);

        retval.push_back(std::move(ex));
    }

    get_logger()->debug("function CSE reduced decomposition size from {} to {}", orig_size, retval.size());
    get_logger()->trace("function CSE runtime: {}", sw);

    return retval;
}

// Perform a topological sort on a graph representation
// of a function decomposition. This can improve performance
// by grouping together operations that can be performed in parallel,
// and it also makes compact mode much more effective by creating
// clusters of subexpressions whose derivatives can be computed in
// parallel.
// NOTE: the original decomposition dc is already topologically sorted,
// in the sense that the definitions of the u variables are already
// ordered according to dependency. However, because the original decomposition
// comes from a depth-first search, it has the tendency to group together
// expressions which are dependent on each other. By doing another topological
// sort, this time based on breadth-first search, we determine another valid
// sorting in which independent operations tend to be clustered together.
std::vector<expression> function_sort_dc(std::vector<expression> &dc, std::vector<expression>::size_type nvars,
                                         std::vector<expression>::size_type nouts)
{
    // A function decomposition is supposed
    // to have nvars variables at the beginning,
    // nouts variables at the end and possibly
    // extra variables in the middle.
    assert(dc.size() >= nouts + nvars);

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // The graph type that we will use for the topological sorting.
    using graph_t = boost::adjacency_list<boost::vecS,           // std::vector for list of adjacent vertices
                                          boost::vecS,           // std::vector for the list of vertices
                                          boost::bidirectionalS, // directed graph with efficient access
                                                                 // to in-edges
                                          boost::no_property,    // no vertex properties
                                          boost::no_property,    // no edge properties
                                          boost::no_property,    // no graph properties
                                          boost::listS           // std::list for of the graph's edge list
                                          >;

    graph_t g;

    // Add the root node.
    const auto root_v = boost::add_vertex(g);

    // Add the nodes corresponding to the original variables.
    for (decltype(nvars) i = 0; i < nvars; ++i) {
        auto v = boost::add_vertex(g);

        // Add a dependency on the root node.
        boost::add_edge(root_v, v, g);
    }

    // Add the rest of the u variables.
    for (decltype(nvars) i = nvars; i < dc.size() - nouts; ++i) {
        auto v = boost::add_vertex(g);

        // Fetch the list of variables in the current expression.
        const auto vars = get_variables(dc[i]);

        if (vars.empty()) {
            // The current expression does not contain
            // any variable: make it depend on the root
            // node. This means that in the topological
            // sort below, the current u var will appear
            // immediately after the original variables.
            boost::add_edge(root_v, v, g);
        } else {
            // Mark the current u variable as depending on all the
            // variables in the current expression.
            for (const auto &var : vars) {
                // Extract the index.
                const auto idx = uname_to_index(var);

                // Add the dependency.
                // NOTE: add +1 because the i-th vertex
                // corresponds to the (i-1)-th u variable
                // due to the presence of the root node.
                boost::add_edge(boost::vertex(idx + 1u, g), v, g);
            }
        }
    }

    assert(boost::num_vertices(g) - 1u == dc.size() - nouts);

    // Run the BF topological sort on the graph. This is Kahn's algorithm:
    // https://en.wikipedia.org/wiki/Topological_sorting

    // The result of the sort.
    std::vector<decltype(dc.size())> v_idx;

    // Temp variable used to sort a list of edges in the loop below.
    std::vector<boost::graph_traits<graph_t>::edge_descriptor> tmp_edges;

    // The set of all nodes with no incoming edge.
    std::deque<decltype(dc.size())> tmp;
    // The root node has no incoming edge.
    tmp.push_back(0);

    // Main loop.
    while (!tmp.empty()) {
        // Pop the first element from tmp
        // and append it to the result.
        const auto v = tmp.front();
        tmp.pop_front();
        v_idx.push_back(v);

        // Fetch all the out edges of v and sort them according
        // to the target vertex.
        // NOTE: the sorting is important to ensure that all the original
        // variables are insered into v_idx in the correct order.
        const auto e_range = boost::out_edges(v, g);
        tmp_edges.assign(e_range.first, e_range.second);
        std::sort(tmp_edges.begin(), tmp_edges.end(),
                  [&g](const auto &e1, const auto &e2) { return boost::target(e1, g) < boost::target(e2, g); });

        // For each out edge of v:
        // - eliminate it;
        // - check if the target vertex of the edge
        //   has other incoming edges;
        // - if it does not, insert it into tmp.
        for (auto &e : tmp_edges) {
            // Fetch the target of the edge.
            const auto t = boost::target(e, g);

            // Remove the edge.
            boost::remove_edge(e, g);

            // Get the range of vertices connecting to t.
            const auto iav = boost::inv_adjacent_vertices(t, g);

            if (iav.first == iav.second) {
                // t does not have any incoming edges, add it to tmp.
                tmp.push_back(t);
            }
        }
    }

    assert(v_idx.size() == boost::num_vertices(g));
    assert(boost::num_edges(g) == 0u);

    // Adjust v_idx: remove the index of the root node,
    // decrease by one all other indices, insert the final
    // nouts indices.
    for (decltype(v_idx.size()) i = 0; i < v_idx.size() - 1u; ++i) {
        v_idx[i] = v_idx[i + 1u] - 1u;
    }
    v_idx.resize(boost::numeric_cast<decltype(v_idx.size())>(dc.size()));
    std::iota(v_idx.data() + dc.size() - nouts, v_idx.data() + dc.size(), dc.size() - nouts);

    // Create the remapping dictionary.
    std::unordered_map<std::string, std::string> remap;
    // NOTE: the u vars that correspond to the original
    // variables were inserted into v_idx in the original
    // order, thus they are not re-sorted and they do not
    // need renaming.
    for (decltype(v_idx.size()) i = 0; i < nvars; ++i) {
        assert(v_idx[i] == i);
        [[maybe_unused]] const auto res = remap.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }
    // Establish the remapping for the u variables that are not
    // original variables.
    for (decltype(v_idx.size()) i = nvars; i < v_idx.size() - nouts; ++i) {
        [[maybe_unused]] const auto res = remap.emplace(fmt::format("u_{}", v_idx[i]), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Do the remap for the definitions of the u variables and of the components.
    for (auto *it = dc.data() + nvars; it != dc.data() + dc.size(); ++it) {
        // Remap the expression.
        rename_variables(*it, remap);
    }

    // Reorder the decomposition.
    std::vector<expression> retval;
    retval.reserve(v_idx.size());
    for (auto idx : v_idx) {
        retval.push_back(std::move(dc[idx]));
    }

    get_logger()->trace("function topological sort runtime: {}", sw);

    return retval;
}

} // namespace

} // namespace detail

std::optional<std::vector<expression>::size_type> decompose(const expression &ex, std::vector<expression> &dc)
{
    std::unordered_map<const void *, std::vector<expression>::size_type> func_map;

    return detail::decompose(func_map, ex, dc);
}

// Decomposition with automatic deduction of variables.
std::pair<std::vector<expression>, std::vector<expression>::size_type>
function_decompose(const std::vector<expression> &v_ex_)
{
    // Need to operate on a copy due to in-place mutation
    // via rename_variables() and decompose().
    // NOTE: this is suboptimal, as expressions which are shared
    // across different elements of v_ex will be not shared any more
    // after the copy.
    auto v_ex = detail::copy(v_ex_);

    if (v_ex.empty()) {
        throw std::invalid_argument("Cannot decompose a function with no outputs");
    }

    // Determine the variables.
    std::set<std::string> vars;
    for (const auto &ex : v_ex) {
        for (const auto &var : get_variables(ex)) {
            vars.emplace(var);
        }
    }

    // Cache the number of variables.
    const auto nvars = vars.size();

    // Cache the number of outputs.
    const auto nouts = v_ex.size();

    // Create the map for renaming the variables to u_i.
    // The renaming will be done in alphabetical order.
    std::unordered_map<std::string, std::string> repl_map;
    {
        decltype(vars.size()) var_idx = 0;
        for (const auto &var : vars) {
            [[maybe_unused]] const auto eres = repl_map.emplace(var, fmt::format("u_{}", var_idx++));
            assert(eres.second);
        }
    }

#if !defined(NDEBUG)

    // Store a copy of the original function for checking later.
    auto orig_v_ex = detail::copy(v_ex);

#endif

    // Rename the variables in the original function.
    for (auto &ex : v_ex) {
        rename_variables(ex, repl_map);
    }

    // Init the decomposition. It begins with a list
    // of the original variables of the system.
    std::vector<expression> ret;
    ret.reserve(vars.size());
    for (const auto &var : vars) {
        ret.emplace_back(var);
    }

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition on each component of the function.
    for (auto &ex : v_ex) {
        // Decompose the current equation.
        if (const auto dres = decompose(ex, ret)) {
            // NOTE: if the component was decomposed
            // (that is, it is not constant or a single variable),
            // we have to update the original definition
            // of the equation in v_ex
            // so that it points to the u variable
            // that now represents it.
            // NOTE: all functions are forced to return
            // a non-empty dres
            // in the func API, so the only entities that
            // can return dres == 0 are const/params or
            // variables.
            ex = expression{fmt::format("u_{}", *dres)};
        } else {
            assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
                   || std::holds_alternative<param>(ex.value()));
        }
    }

    // Append the definitions of the outputs
    // in terms of u variables.
    for (auto &ex : v_ex) {
        ret.emplace_back(std::move(ex));
    }

    detail::get_logger()->trace("function decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    detail::verify_function_dec(orig_v_ex, ret, nvars);

#endif

    // Simplify the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    ret = detail::function_decompose_cse(ret, nvars, nouts);

#if !defined(NDEBUG)

    // Verify the simplified decomposition.
    detail::verify_function_dec(orig_v_ex, ret, nvars);

#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    ret = detail::function_sort_dc(ret, nvars, nouts);

#if !defined(NDEBUG)

    // Verify the reordered decomposition.
    detail::verify_function_dec(orig_v_ex, ret, nvars);

#endif

    // NOTE: static_cast is fine, as we know that ret contains at least nvars elements.
    return std::make_pair(std::move(ret), static_cast<std::vector<expression>::size_type>(nvars));
}

// Determine if an expression is time-dependent.
bool has_time(const expression &ex)
{
    std::unordered_set<const void *> func_set;

    return detail::has_time(func_set, ex);
}

} // namespace heyoka
