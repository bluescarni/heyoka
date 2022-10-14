// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <ios>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <llvm/ADT/APFloat.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/integer.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/binomial.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

number::number() : number(0.) {}

number::number(float x) : m_value(x) {}

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
    return std::visit([](const auto &v) { return std::hash<detail::uncvref_t<decltype(v)>>{}(v); }, n.value());
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

namespace detail
{

namespace
{

// Type-traits to detect arithmetic and comparison capabilities
// in a type. Used in the implementation of the corresponding operations
// for the number class.
template <typename T, typename U>
using add_t = decltype(std::declval<T>() + std::declval<U>());

template <typename T, typename U = T>
using is_addable = std::conjunction<is_detected<add_t, T, U>, is_detected<add_t, U, T>,
                                    std::is_same<detected_t<add_t, T, U>, detected_t<add_t, U, T>>>;

template <typename T, typename U>
using sub_t = decltype(std::declval<T>() - std::declval<U>());

template <typename T, typename U = T>
using is_subtractable = std::conjunction<is_detected<sub_t, T, U>, is_detected<sub_t, U, T>,
                                         std::is_same<detected_t<sub_t, T, U>, detected_t<sub_t, U, T>>>;

template <typename T, typename U>
using mul_t = decltype(std::declval<T>() * std::declval<U>());

template <typename T, typename U = T>
using is_multipliable = std::conjunction<is_detected<mul_t, T, U>, is_detected<mul_t, U, T>,
                                         std::is_same<detected_t<mul_t, T, U>, detected_t<mul_t, U, T>>>;

template <typename T, typename U>
using div_t = decltype(std::declval<T>() / std::declval<U>());

template <typename T, typename U = T>
using is_divisible = std::conjunction<is_detected<div_t, T, U>, is_detected<div_t, U, T>,
                                      std::is_same<detected_t<div_t, T, U>, detected_t<div_t, U, T>>>;

} // namespace

} // namespace detail

number operator+(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) -> number {
            if constexpr (detail::is_addable<decltype(arg1), decltype(arg2)>::value) {
                return number{std::forward<decltype(arg1)>(arg1) + std::forward<decltype(arg2)>(arg2)};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("Cannot add an object of type '{}' to an object of type '{}'",
                                                        boost::core::demangle(typeid(arg1).name()),
                                                        boost::core::demangle(typeid(arg2).name())));
                // LCOV_EXCL_STOP
            }
        },
        std::move(n1.value()), std::move(n2.value()));
}

number operator-(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) -> number {
            if constexpr (detail::is_subtractable<decltype(arg1), decltype(arg2)>::value) {
                return number{std::forward<decltype(arg1)>(arg1) - std::forward<decltype(arg2)>(arg2)};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format(
                    "Cannot subtract an object of type '{}' from an object of type '{}'",
                    boost::core::demangle(typeid(arg2).name()), boost::core::demangle(typeid(arg1).name())));
                // LCOV_EXCL_STOP
            }
        },
        std::move(n1.value()), std::move(n2.value()));
}

number operator*(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) -> number {
            if constexpr (detail::is_multipliable<decltype(arg1), decltype(arg2)>::value) {
                return number{std::forward<decltype(arg1)>(arg1) * std::forward<decltype(arg2)>(arg2)};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format(
                    "Cannot multiply an object of type '{}' by an object of type '{}'",
                    boost::core::demangle(typeid(arg1).name()), boost::core::demangle(typeid(arg2).name())));
                // LCOV_EXCL_STOP
            }
        },
        std::move(n1.value()), std::move(n2.value()));
}

number operator/(number n1, number n2)
{
    return std::visit(
        [](auto &&arg1, auto &&arg2) -> number {
            if constexpr (detail::is_divisible<decltype(arg1), decltype(arg2)>::value) {
                return number{std::forward<decltype(arg1)>(arg1) / std::forward<decltype(arg2)>(arg2)};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format(
                    "Cannot divide an object of type '{}' by an object of type '{}'",
                    boost::core::demangle(typeid(arg1).name()), boost::core::demangle(typeid(arg2).name())));
                // LCOV_EXCL_STOP
            }
        },
        std::move(n1.value()), std::move(n2.value()));
}

// NOTE: in order for equality to be consistent with hashing,
// we want to make sure that two numbers of different type
// are always considered different (were they considered equal,
// we would have then to ensure that they both hash to the same
// value, which would be quite hard to do).
bool operator==(const number &n1, const number &n2)
{
    return std::visit(
        [](const auto &v1, const auto &v2) -> bool {
            using type1 = detail::uncvref_t<decltype(v1)>;
            using type2 = detail::uncvref_t<decltype(v2)>;

            if constexpr (std::is_same_v<type1, type2>) {
                return v1 == v2;
            } else {
                return false;
            }
        },
        n1.value(), n2.value());
}

bool operator!=(const number &n1, const number &n2)
{
    return !(n1 == n2);
}

number exp(number n)
{
    return std::visit(
        [](auto &&arg) {
            using std::exp;

            return number{exp(std::forward<decltype(arg)>(arg))};
        },
        std::move(n.value()));
}

number binomial(const number &i, const number &j)
{
    return std::visit(
        [](const auto &v1, const auto &v2) -> number {
            using type1 = detail::uncvref_t<decltype(v1)>;
            using type2 = detail::uncvref_t<decltype(v2)>;

            if constexpr (!std::is_same_v<type1, type2>) {
                throw std::invalid_argument("Cannot compute the binomial coefficient of two numbers of different type");
            } else {
                using std::isfinite;
                using std::trunc;

                if (!isfinite(v1) || !isfinite(v2)) {
                    throw std::invalid_argument("Cannot compute the binomial coefficient of non-finite values");
                }

                if (trunc(v1) != v1 || trunc(v2) != v2) {
                    throw std::invalid_argument("Cannot compute the binomial coefficient non-integral values");
                }

                if constexpr (std::is_floating_point_v<type1>) {
                    // For C++ FP types, we can use directly the binomial
                    // implementation in detail, after casting the
                    // arguments back to std::uint32_t.
                    return number{detail::binomial<type1>(boost::numeric_cast<std::uint32_t>(v1),
                                                          boost::numeric_cast<std::uint32_t>(v2))};
#if defined(HEYOKA_HAVE_REAL128)
                } else if constexpr (std::is_same_v<type1, mppp::real128>) {
                    // For real128, we cannot use boost::numeric_cast, so we go through
                    // a checked conversion via mppp::integer.
                    const auto n1 = static_cast<mppp::integer<1>>(v1);
                    const auto n2 = static_cast<mppp::integer<1>>(v2);

                    return number{
                        detail::binomial<type1>(static_cast<std::uint32_t>(n1), static_cast<std::uint32_t>(n2))};
#endif
                    // LCOV_EXCL_START
                } else {
                    throw std::invalid_argument(fmt::format("Arguments of type '{}' are not supported by binomial()",
                                                            boost::core::demangle(typeid(type1).name())));
                }
                // LCOV_EXCL_STOP
            }
        },
        i.value(), j.value());
}

number nextafter(const number &from, const number &to)
{
    return std::visit(
        [](const auto &v1, const auto &v2) -> number {
            using type1 = detail::uncvref_t<decltype(v1)>;
            using type2 = detail::uncvref_t<decltype(v2)>;

            if constexpr (!std::is_same_v<type1, type2>) {
                throw std::invalid_argument("Cannot invoke nextafter() on two numbers of different type");
            } else {
                using std::nextafter;

                return number{nextafter(v1, v2)};
            }
        },
        from.value(), to.value());
}

namespace detail
{

namespace
{

template <typename To>
To number_eval_impl(const number &n)
{
    return std::visit(
        [](const auto &v) -> To {
            if constexpr (std::is_constructible_v<To, decltype(v)>) {
                return static_cast<To>(v);
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(
                    fmt::format("Cannot convert an object of type '{}' to an object of type '{}'",
                                boost::core::demangle(typeid(v).name()), boost::core::demangle(typeid(To).name())));
                // LCOV_EXCL_STOP
            }
        },
        n.value());
}

} // namespace

} // namespace detail

double eval_dbl(const number &n, const std::unordered_map<std::string, double> &, const std::vector<double> &)
{
    return detail::number_eval_impl<double>(n);
}

long double eval_ldbl(const number &n, const std::unordered_map<std::string, long double> &,
                      const std::vector<long double> &)
{
    return detail::number_eval_impl<long double>(n);
}

#if defined(HEYOKA_HAVE_REAL128)

mppp::real128 eval_f128(const number &n, const std::unordered_map<std::string, mppp::real128> &,
                        const std::vector<mppp::real128> &)
{
    return detail::number_eval_impl<mppp::real128>(n);
}

#endif

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

// Generate an LLVM constant of type tp representing the number n.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
llvm::Value *llvm_codegen(llvm_state &s, llvm::Type *tp, const number &n)
{
    assert(tp != nullptr);

    // NOTE: isIEEE() is only available since LLVM 13.
    // For earlier versions of LLVM, we check that
    // tp is not a double-double, all the other available
    // FP types should be IEEE.
    if (tp->isFloatingPointTy() &&
#if LLVM_VERSION_MAJOR >= 13
        tp->isIEEE()
#else
        !tp->isPPC_FP128Ty()
#endif
    ) {
        // NOTE: for float and double we can construct
        // directly an APFloat.
        if (tp->isFloatTy() || tp->isDoubleTy()) {
            const auto apf
                = tp->isFloatTy()
                      ? llvm::APFloat(std::visit([](const auto &v) { return static_cast<float>(v); }, n.value()))
                      : llvm::APFloat(std::visit([](const auto &v) { return static_cast<double>(v); }, n.value()));

            return llvm::ConstantFP::get(s.context(), apf);
        }

        // Fetch the FP semantics and precision.
        const auto &sem = tp->getFltSemantics();
        const auto prec = llvm::APFloatBase::semanticsPrecision(sem);

        // Compute the number of base-10 digits that are necessary to uniquely represent
        // all distinct values of the type tp. See:
        // https://en.cppreference.com/w/cpp/types/numeric_limits/max_digits10
        const auto max_d10 = boost::numeric_cast<std::streamsize>(std::ceil(prec * std::log10(2.) + 1));

#if !defined(NDEBUG) && defined(HEYOKA_HAVE_REAL128)

        if (tp == llvm::Type::getFP128Ty(s.context())) {
            assert(max_d10 == std::numeric_limits<mppp::real128>::max_digits10);
            assert(prec == static_cast<unsigned>(std::numeric_limits<mppp::real128>::digits));
        }

#endif

        // Fetch a string representation of n via the stream operator.
        // Ensure that we use max_d10 digits in the representation, so that
        // we get the closest approximation possible of n for the type tp.
        std::ostringstream ss;
        ss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        ss.imbue(std::locale::classic());
        ss.precision(max_d10);

        const auto str_rep = std::visit(
            [&ss](const auto &v) {
                ss << v;

                return ss.str();
            },
            n.value());

        // Construct the FP constant.
        // NOTE: llvm will deduce the correct type for the codegen from the supplied
        // floating-point semantics.
        return llvm::ConstantFP::get(s.context(), llvm::APFloat(sem, str_rep));
    } else {
        throw std::invalid_argument(
            fmt::format("Cannot generate an LLVM constant of type '{}'", detail::llvm_type_name(tp)));
    }
}

namespace detail
{

// A small helper to create a number instance containing the value val
// cast to the C++ type corresponding to the LLVM type tp.
number number_like(llvm_state &s, llvm::Type *tp, double val)
{
    assert(tp != nullptr);

    auto &context = s.context();

    if (tp == to_llvm_type<float>(context, false)) {
        return number{static_cast<float>(val)};
    } else if (tp == to_llvm_type<double>(context, false)) {
        return number{val};
    } else if (tp == to_llvm_type<long double>(context, false)) {
        return number{static_cast<long double>(val)};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (tp == to_llvm_type<mppp::real128>(context, false)) {
        return number{static_cast<mppp::real128>(val)};
#endif
    }

    throw std::invalid_argument(
        fmt::format("Unable to create a number of type '{}' from the input value {}", llvm_type_name(tp), val));
}

} // namespace detail

} // namespace heyoka
