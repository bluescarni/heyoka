// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EXPRESSION_HPP
#define HEYOKA_EXPRESSION_HPP

#include <heyoka/config.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC expression
{
public:
    using value_type = std::variant<number, variable, binary_operator, function>;

private:
    value_type m_value;

public:
    explicit expression(number);
    explicit expression(variable);
    explicit expression(binary_operator);
    explicit expression(function);
    expression(const expression &);
    expression(expression &&) noexcept;
    ~expression();

    expression &operator=(const expression &);
    expression &operator=(expression &&) noexcept;

    value_type &value();
    const value_type &value() const;
};

inline namespace literals
{

HEYOKA_DLL_PUBLIC expression operator""_dbl(long double);
HEYOKA_DLL_PUBLIC expression operator""_dbl(unsigned long long);

HEYOKA_DLL_PUBLIC expression operator""_ldbl(long double);
HEYOKA_DLL_PUBLIC expression operator""_ldbl(unsigned long long);

#if defined(HEYOKA_HAVE_REAL128)

template <char... Chars>
inline expression operator"" _f128()
{
    return expression{number{mppp::literals::operator"" _rq<Chars...>()}};
}

#endif

HEYOKA_DLL_PUBLIC expression operator""_var(const char *, std::size_t);

} // namespace literals

namespace detail
{

struct HEYOKA_DLL_PUBLIC prime_wrapper {
    std::string m_str;

    explicit prime_wrapper(std::string);
    prime_wrapper(const prime_wrapper &);
    prime_wrapper(prime_wrapper &&) noexcept;
    prime_wrapper &operator=(const prime_wrapper &);
    prime_wrapper &operator=(prime_wrapper &&) noexcept;
    ~prime_wrapper();

    std::pair<expression, expression> operator=(expression) &&;
};

} // namespace detail

HEYOKA_DLL_PUBLIC detail::prime_wrapper prime(expression);

inline namespace literals
{

HEYOKA_DLL_PUBLIC detail::prime_wrapper operator""_p(const char *, std::size_t);

} // namespace literals

HEYOKA_DLL_PUBLIC void swap(expression &, expression &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const expression &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const expression &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const expression &);
HEYOKA_DLL_PUBLIC void rename_variables(expression &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC expression operator+(expression);
HEYOKA_DLL_PUBLIC expression operator-(expression);

HEYOKA_DLL_PUBLIC expression operator+(expression, expression);
HEYOKA_DLL_PUBLIC expression operator-(expression, expression);
HEYOKA_DLL_PUBLIC expression operator*(expression, expression);
HEYOKA_DLL_PUBLIC expression operator/(expression, expression);

HEYOKA_DLL_PUBLIC expression &operator+=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, expression);

HEYOKA_DLL_PUBLIC bool operator==(const expression &, const expression &);
HEYOKA_DLL_PUBLIC bool operator!=(const expression &, const expression &);

HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const expression &, const std::string &);
HEYOKA_DLL_PUBLIC expression diff(const expression &, const expression &);

HEYOKA_DLL_PUBLIC double eval_dbl(const expression &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const expression &,
                                      const std::unordered_map<std::string, std::vector<double>> &);

// When traversing the expression tree with some recursive algorithm we may have to do some book-keeping and use
// preallocated memory to store the result, in which case the corresponding function is called update_*. A corresponding
// method, more friendly to use, takes care of allocating memory and initializing the book-keeping variables, its called
// compute_*.
HEYOKA_DLL_PUBLIC std::vector<std::vector<std::size_t>> compute_connections(const expression &);
HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const expression &, std::size_t &);
HEYOKA_DLL_PUBLIC std::vector<double> compute_node_values_dbl(const expression &,
                                                              const std::unordered_map<std::string, double> &,
                                                              const std::vector<std::vector<std::size_t>> &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const expression &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);

HEYOKA_DLL_PUBLIC std::unordered_map<std::string, double>
compute_grad_dbl(const expression &, const std::unordered_map<std::string, double> &,
                 const std::vector<std::vector<std::size_t>> &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const expression &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double = 1.);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const expression &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const expression &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *codegen_f128(llvm_state &, const expression &);

#endif

template <typename T>
inline llvm::Value *codegen(llvm_state &s, const expression &ex)
{
    if constexpr (std::is_same_v<T, double>) {
        return codegen_dbl(s, ex);
    } else if constexpr (std::is_same_v<T, long double>) {
        return codegen_ldbl(s, ex);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return codegen_f128(s, ex);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(expression &&,
                                                                               std::vector<expression> &);

template <typename... Args>
inline std::array<expression, sizeof...(Args)> make_vars(const Args &... strs)
{
    return std::array{expression{variable{strs}}...};
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_dbl(llvm_state &, const expression &, const std::vector<llvm::Value *> &,
                                                 std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_ldbl(llvm_state &, const expression &, const std::vector<llvm::Value *> &,
                                                  std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_f128(llvm_state &, const expression &, const std::vector<llvm::Value *> &,
                                                  std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_u_init(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_u_init_dbl(s, ex, arr, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_u_init_ldbl(s, ex, arr, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_u_init_f128(s, ex, arr, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_dbl(llvm_state &, const expression &, const std::vector<llvm::Value *> &,
                                               std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_ldbl(llvm_state &, const expression &, const std::vector<llvm::Value *> &,
                                                std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_f128(llvm_state &, const expression &, const std::vector<llvm::Value *> &,
                                                std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_diff(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                                std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_diff_dbl(s, ex, arr, n_uvars, order, idx, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_diff_ldbl(s, ex, arr, n_uvars, order, idx, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_diff_f128(s, ex, arr, n_uvars, order, idx, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_u_init_dbl(llvm_state &, const expression &, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_u_init_ldbl(llvm_state &, const expression &, llvm::Value *, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_u_init_f128(llvm_state &, const expression &, llvm::Value *, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_c_u_init(llvm_state &s, const expression &ex, llvm::Value *arr, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_c_u_init_dbl(s, ex, arr, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_c_u_init_ldbl(s, ex, arr, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_c_u_init_f128(s, ex, arr, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_dbl(llvm_state &, const expression &, llvm::Value *, std::uint32_t,
                                                 llvm::Value *, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_ldbl(llvm_state &, const expression &, llvm::Value *, std::uint32_t,
                                                  llvm::Value *, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_f128(llvm_state &, const expression &, llvm::Value *, std::uint32_t,
                                                  llvm::Value *, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_c_diff(llvm_state &s, const expression &ex, llvm::Value *arr, std::uint32_t n_uvars,
                                  llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_c_diff_dbl(s, ex, arr, n_uvars, order, idx, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_c_diff_ldbl(s, ex, arr, n_uvars, order, idx, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_c_diff_f128(s, ex, arr, n_uvars, order, idx, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

namespace std
{

// Specialisation of std::hash for expression.
template <>
struct hash<heyoka::expression> {
    size_t operator()(const heyoka::expression &ex) const
    {
        return heyoka::hash(ex);
    }
};

} // namespace std

#endif
