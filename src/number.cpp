// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
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
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/integer.hpp>
#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/binomial.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

number::number() noexcept : number(0.) {}

number::number(float x) noexcept : m_value(x) {}

number::number(double x) noexcept : m_value(x) {}

number::number(long double x) noexcept : m_value(x) {}

#if defined(HEYOKA_HAVE_REAL128)

number::number(mppp::real128 x) noexcept : m_value(x) {}

#endif

#if defined(HEYOKA_HAVE_REAL)

number::number(mppp::real x) : m_value(std::move(x)) {}

#endif

number::number(const number &) = default;

// NOLINTNEXTLINE(bugprone-exception-escape)
number::number(number &&other) noexcept : m_value(std::move(other.m_value))
{
    // NOTE: ensure other is equivalent to a
    // default-constructed number.
    other.m_value.emplace<double>(0.);
}

number::~number() = default;

number &number::operator=(const number &other)
{
    if (this != &other) {
        *this = number(other);
    }

    return *this;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
number &number::operator=(number &&other) noexcept
{
    if (this != &other) {
        m_value = std::move(other.m_value);
        // NOTE: ensure other is equivalent to a
        // default-constructed number.
        other.m_value.emplace<double>(0.);
    }

    return *this;
}

void number::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_value;
}

void number::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_value;
}

const number::value_type &number::value() const noexcept
{
    return m_value;
}

void swap(number &n0, number &n1) noexcept
{
    std::swap(n0.m_value, n1.m_value);
}

namespace detail
{

// NOLINTNEXTLINE(bugprone-exception-escape)
std::size_t hash(const number &n) noexcept
{
    return std::visit(
        [&n](const auto &v) -> std::size_t {
            using std::isnan;

            if (isnan(v)) {
                // NOTE: enforce that all NaN values of a given type
                // have the same hash, because NaNs are considered equal to each other
                // by the comparison operator.
                return std::hash<std::size_t>{}(n.value().index());
            } else {
                return std::hash<detail::uncvref_t<decltype(v)>>{}(v);
            }
        },
        n.value());
}

} // namespace detail

std::ostream &operator<<(std::ostream &os, const number &n)
{
    std::visit(
        [&os](const auto &arg) {
            using type = detail::uncvref_t<decltype(arg)>;

#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<type, mppp::real>) {
                os << arg.to_string();
            } else {
#endif
                // NOTE: we make sure to print all digits
                // necessary for short-circuiting. Make also
                // sure to always print the decimal point and to
                // use the C locale.
                std::ostringstream oss;
                oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);

                oss.imbue(std::locale::classic());
                oss << std::showpoint;

                oss.precision(std::numeric_limits<type>::max_digits10);
                oss << arg;

                os << oss.str();
#if defined(HEYOKA_HAVE_REAL)
            }
#endif
        },
        n.value());

    return os;
}

bool is_zero(const number &n)
{
    return std::visit(
        [](const auto &arg) {
            using type [[maybe_unused]] = detail::uncvref_t<decltype(arg)>;

#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<type, mppp::real>) {
                return arg.zero_p();
            } else {
#endif
                return arg == 0;
#if defined(HEYOKA_HAVE_REAL)
            }
#endif
        },
        n.value());
}

bool is_one(const number &n)
{
    return std::visit(
        [](const auto &arg) {
            using type [[maybe_unused]] = detail::uncvref_t<decltype(arg)>;

#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<type, mppp::real>) {
                return arg.is_one();
            } else {
#endif
                return arg == 1;
#if defined(HEYOKA_HAVE_REAL)
            }
#endif
        },
        n.value());
}

bool is_negative_one(const number &n)
{
    return std::visit([](const auto &arg) { return arg == -1; }, n.value());
}

bool is_negative(const number &n)
{
    return std::visit([](const auto &arg) { return arg < 0; }, n.value());
}

number operator+(number n)
{
    return n;
}

number operator-(const number &n)
{
    return std::visit([](const auto &arg) { return number{-arg}; }, n.value());
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

number operator+(const number &n1, const number &n2)
{
    return std::visit(
        [](const auto &arg1, const auto &arg2) -> number {
            if constexpr (detail::is_addable<decltype(arg1), decltype(arg2)>::value) {
                return number{arg1 + arg2};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("Cannot add an object of type '{}' to an object of type '{}'",
                                                        boost::core::demangle(typeid(arg1).name()),
                                                        boost::core::demangle(typeid(arg2).name())));
                // LCOV_EXCL_STOP
            }
        },
        n1.value(), n2.value());
}

number operator-(const number &n1, const number &n2)
{
    return std::visit(
        [](const auto &arg1, const auto &arg2) -> number {
            if constexpr (detail::is_subtractable<decltype(arg1), decltype(arg2)>::value) {
                return number{arg1 - arg2};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format(
                    "Cannot subtract an object of type '{}' from an object of type '{}'",
                    boost::core::demangle(typeid(arg2).name()), boost::core::demangle(typeid(arg1).name())));
                // LCOV_EXCL_STOP
            }
        },
        n1.value(), n2.value());
}

number operator*(const number &n1, const number &n2)
{
    return std::visit(
        [](const auto &arg1, const auto &arg2) -> number {
            if constexpr (detail::is_multipliable<decltype(arg1), decltype(arg2)>::value) {
                return number{arg1 * arg2};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format(
                    "Cannot multiply an object of type '{}' by an object of type '{}'",
                    boost::core::demangle(typeid(arg1).name()), boost::core::demangle(typeid(arg2).name())));
                // LCOV_EXCL_STOP
            }
        },
        n1.value(), n2.value());
}

number operator/(const number &n1, const number &n2)
{
    return std::visit(
        [](const auto &arg1, const auto &arg2) -> number {
            if constexpr (detail::is_divisible<decltype(arg1), decltype(arg2)>::value) {
                return number{arg1 / arg2};
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format(
                    "Cannot divide an object of type '{}' by an object of type '{}'",
                    boost::core::demangle(typeid(arg1).name()), boost::core::demangle(typeid(arg2).name())));
                // LCOV_EXCL_STOP
            }
        },
        n1.value(), n2.value());
}

// NOTE: in order for equality to be consistent with hashing,
// we want to make sure that two numbers of different type
// are always considered different (were they considered equal,
// we would have then to ensure that they both hash to the same
// value, which would be quite hard to do).
// NOTE: it is convenient to consider NaNs equal to each other.
// This allows to avoid ugly special-casing, e.g., when
// verifying decompositions: when the original expression is
// reconstructed from the subexpressions and we compare, the
// check would fail due to NaN != NaN.
// NOLINTNEXTLINE(bugprone-exception-escape)
bool operator==(const number &n1, const number &n2) noexcept
{
    return std::visit(
        [](const auto &v1, const auto &v2) -> bool {
            if constexpr (std::is_same_v<decltype(v1), decltype(v2)>) {
                using std::isnan;

                if (isnan(v1) && isnan(v2)) {
                    return true;
                } else {
                    return v1 == v2;
                }
            } else {
                return false;
            }
        },
        n1.value(), n2.value());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
bool operator!=(const number &n1, const number &n2) noexcept
{
    return !(n1 == n2);
}

// NOTE: consistently with operator==(), n1 and n2 must be of the
// same type in order to be considered equivalent. Also, NaNs are considered
// greater than any other value, so that they will be placed at the
// end of a sorted range.
// NOLINTNEXTLINE(bugprone-exception-escape)
bool operator<(const number &n1, const number &n2) noexcept
{
    return std::visit(
        [&](const auto &v1, const auto &v2) -> bool {
            if constexpr (std::is_same_v<decltype(v1), decltype(v2)>) {
                using std::isnan;

                if (isnan(v1)) {
                    // NaN cannot be less than anything,
                    // including NaN.
                    return false;
                }

                if (isnan(v2)) {
                    // v1 < NaN, with v1 non-NaN.
                    return true;
                }

                return v1 < v2;
            } else {
                return n1.value().index() < n2.value().index();
            }
        },
        n1.value(), n2.value());
}

number exp(const number &n)
{
    return std::visit(
        [](const auto &arg) {
            using std::exp;

            return number{exp(arg)};
        },
        n.value());
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
#if defined(HEYOKA_HAVE_REAL)
                } else if constexpr (std::is_same_v<type1, mppp::real>) {
                    // For real, we transform the input arguments into mppp::integer,
                    // invoke binomial and then convert back to real.
                    const auto n1 = static_cast<mppp::integer<1>>(v1);
                    const auto n2 = static_cast<mppp::integer<1>>(v2);

                    // NOTE: do the conversion using the maximum precision among the operands, as usual.
                    return number{mppp::real{binomial(n1, n2), std::max(v1.get_prec(), v2.get_prec())}};
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

number sqrt(const number &n)
{
    return std::visit(
        [](auto &&arg) {
            using std::sqrt;

            return number{sqrt(arg)};
        },
        n.value());
}

// Generate an LLVM constant of type tp representing the number n.
// NOLINTNEXTLINE(misc-no-recursion)
llvm::Value *llvm_codegen(llvm_state &s, llvm::Type *tp, const number &n)
{
    assert(tp != nullptr);

    // If tp is an llvm_vector_type, codegen the scalar value and splat it.
    if (const auto *vector_t = llvm::dyn_cast<detail::llvm_vector_type>(tp)) {
        const auto vec_size = boost::numeric_cast<std::uint32_t>(vector_t->getNumElements());

        return detail::vector_splat(s.builder(), llvm_codegen(s, vector_t->getScalarType(), n), vec_size);
    }

    if (tp->isFloatingPointTy() && tp->isIEEE()) {
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
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto real_prec = detail::llvm_is_real(tp)) {
        // From the number, generate a real with the desired precision.
        const auto r = std::visit([real_prec](const auto &v) { return mppp::real{v, real_prec}; }, n.value());

        // Generate the limb array in LLVM.
        auto *struct_tp = llvm::cast<llvm::StructType>(tp);
        auto *limb_array_t = llvm::cast<llvm::ArrayType>(struct_tp->elements()[2]);

        std::vector<llvm::Constant *> limbs;
        for (std::size_t i = 0; i < r.get_nlimbs(); ++i) {
            limbs.push_back(llvm::ConstantInt::get(limb_array_t->getElementType(),
                                                   boost::numeric_cast<std::uint64_t>(r.get_mpfr_t()->_mpfr_d[i])));
        }

        auto *limb_arr = llvm::ConstantArray::get(limb_array_t, limbs);

        // Generate sign and exponent.
        auto *sign = llvm::ConstantInt::getSigned(struct_tp->elements()[0],
                                                  boost::numeric_cast<std::int64_t>(r.get_mpfr_t()->_mpfr_sign));
        auto *exp = llvm::ConstantInt::getSigned(struct_tp->elements()[1],
                                                 boost::numeric_cast<std::int64_t>(r.get_mpfr_t()->_mpfr_exp));

        // Generate the struct.
        return llvm::ConstantStruct::get(struct_tp, {sign, exp, limb_arr});
#endif
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Cannot generate an LLVM constant of type '{}'", detail::llvm_type_name(tp)));
    // LCOV_EXCL_STOP
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
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto real_prec = llvm_is_real(tp)) {
        return number{mppp::real{val, real_prec}};
#endif
    }

    throw std::invalid_argument(
        fmt::format("Unable to create a number of type '{}' from the input value {}", llvm_type_name(tp), val));
}

} // namespace detail

HEYOKA_END_NAMESPACE
