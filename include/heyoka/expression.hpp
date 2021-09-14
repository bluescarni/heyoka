// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/func.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC expression
{
public:
    using value_type = std::variant<number, variable, func, param>;

private:
    value_type m_value;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_value;
    }

public:
    expression();

    explicit expression(double);
    explicit expression(long double);
#if defined(HEYOKA_HAVE_REAL128)
    explicit expression(mppp::real128);
#endif
    explicit expression(std::string);

    explicit expression(number);
    explicit expression(variable);
    explicit expression(func);
    explicit expression(param);

    expression(const expression &);
    expression(expression &&) noexcept;

    ~expression();

    expression &operator=(const expression &);
    expression &operator=(expression &&) noexcept;

    value_type &value();
    const value_type &value() const;
};

HEYOKA_DLL_PUBLIC expression copy(const expression &);

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
    return expression{mppp::literals::operator"" _rq<Chars...>()};
}

#endif

HEYOKA_DLL_PUBLIC expression operator""_var(const char *, std::size_t);

} // namespace literals

namespace detail
{

// NOTE: these need to go here because
// the definition of expression must be visible
// in order for these to be well-formed.
template <typename T>
inline expression func_inner<T>::diff(const std::string &s) const
{
    if constexpr (func_has_diff_v<T>) {
        return m_value.diff(s);
    } else {
        throw not_implemented_error("The derivative is not implemented for the function '" + get_name() + "'");
    }
}

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

namespace detail
{

std::vector<std::string> get_variables(std::unordered_set<const void *> &, const expression &);
void rename_variables(std::unordered_set<const void *> &, expression &,
                      const std::unordered_map<std::string, std::string> &);

} // namespace detail

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const expression &);
HEYOKA_DLL_PUBLIC void rename_variables(expression &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC expression operator+(expression);
HEYOKA_DLL_PUBLIC expression operator-(expression);

HEYOKA_DLL_PUBLIC expression operator+(expression, expression);
HEYOKA_DLL_PUBLIC expression operator+(expression, double);
HEYOKA_DLL_PUBLIC expression operator+(expression, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator+(expression, mppp::real128);
#endif
HEYOKA_DLL_PUBLIC expression operator+(double, expression);
HEYOKA_DLL_PUBLIC expression operator+(long double, expression);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator+(mppp::real128, expression);
#endif

HEYOKA_DLL_PUBLIC expression operator-(expression, expression);
HEYOKA_DLL_PUBLIC expression operator-(expression, double);
HEYOKA_DLL_PUBLIC expression operator-(expression, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator-(expression, mppp::real128);
#endif
HEYOKA_DLL_PUBLIC expression operator-(double, expression);
HEYOKA_DLL_PUBLIC expression operator-(long double, expression);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator-(mppp::real128, expression);
#endif

HEYOKA_DLL_PUBLIC expression operator*(expression, expression);
HEYOKA_DLL_PUBLIC expression operator*(expression, double);
HEYOKA_DLL_PUBLIC expression operator*(expression, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator*(expression, mppp::real128);
#endif
HEYOKA_DLL_PUBLIC expression operator*(double, expression);
HEYOKA_DLL_PUBLIC expression operator*(long double, expression);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator*(mppp::real128, expression);
#endif

HEYOKA_DLL_PUBLIC expression operator/(expression, expression);
HEYOKA_DLL_PUBLIC expression operator/(expression, double);
HEYOKA_DLL_PUBLIC expression operator/(expression, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator/(expression, mppp::real128);
#endif
HEYOKA_DLL_PUBLIC expression operator/(double, expression);
HEYOKA_DLL_PUBLIC expression operator/(long double, expression);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator/(mppp::real128, expression);
#endif

HEYOKA_DLL_PUBLIC expression &operator+=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, mppp::real128);
#endif

HEYOKA_DLL_PUBLIC expression &operator-=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, mppp::real128);
#endif

HEYOKA_DLL_PUBLIC expression &operator*=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, mppp::real128);
#endif

HEYOKA_DLL_PUBLIC expression &operator/=(expression &, expression);
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, mppp::real128);
#endif

HEYOKA_DLL_PUBLIC bool operator==(const expression &, const expression &);
HEYOKA_DLL_PUBLIC bool operator!=(const expression &, const expression &);

HEYOKA_DLL_PUBLIC std::size_t get_n_nodes(const expression &);

HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const expression &, const std::string &);
HEYOKA_DLL_PUBLIC expression diff(const expression &, const expression &);

HEYOKA_DLL_PUBLIC expression pairwise_sum(std::vector<expression>);
HEYOKA_DLL_PUBLIC expression pairwise_prod(std::vector<expression>);

HEYOKA_DLL_PUBLIC double eval_dbl(const expression &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> & = {});
HEYOKA_DLL_PUBLIC long double eval_ldbl(const expression &, const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> & = {});
#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC mppp::real128 eval_f128(const expression &, const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> & = {});

#endif

template <typename T>
inline T eval(const expression &e, const std::unordered_map<std::string, T> &map, const std::vector<T> &pars = {})
{
    if constexpr (std::is_same_v<T, double>) {
        return eval_dbl(e, map, pars);
    } else if constexpr (std::is_same_v<T, long double>) {
        return eval_ldbl(e, map, pars);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return eval_f128(e, map, pars);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const expression &,
                                      const std::unordered_map<std::string, std::vector<double>> &,
                                      const std::vector<double> & = {});

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

namespace detail
{

taylor_dc_t::size_type taylor_decompose_in_place(std::unordered_map<const void *, taylor_dc_t::size_type> &,
                                                 const expression &, taylor_dc_t &);

}

HEYOKA_DLL_PUBLIC taylor_dc_t::size_type taylor_decompose_in_place(const expression &, taylor_dc_t &);

template <typename... Args>
inline std::array<expression, sizeof...(Args)> make_vars(const Args &...strs)
{
    return std::array{expression{variable{strs}}...};
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_dbl(llvm_state &, const expression &, const std::vector<std::uint32_t> &,
                                               const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                               std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_ldbl(llvm_state &, const expression &, const std::vector<std::uint32_t> &,
                                                const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                                std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff_f128(llvm_state &, const expression &, const std::vector<std::uint32_t> &,
                                                const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                                std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_diff(llvm_state &s, const expression &ex, const std::vector<std::uint32_t> &deps,
                                const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_diff_dbl(s, ex, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_diff_ldbl(s, ex, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_diff_f128(s, ex, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_dbl(llvm_state &, const expression &, std::uint32_t,
                                                         std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, const expression &, std::uint32_t,
                                                          std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func_f128(llvm_state &, const expression &, std::uint32_t,
                                                          std::uint32_t);

#endif

template <typename T>
inline llvm::Function *taylor_c_diff_func(llvm_state &s, const expression &ex, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_c_diff_func_dbl(s, ex, n_uvars, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_c_diff_func_ldbl(s, ex, n_uvars, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_c_diff_func_f128(s, ex, n_uvars, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::uint32_t get_param_size(const expression &);

HEYOKA_DLL_PUBLIC bool has_time(const expression &);

namespace detail
{

class HEYOKA_DLL_PUBLIC par_impl
{
public:
    expression operator[](std::uint32_t) const;
};

} // namespace detail

inline constexpr detail::par_impl par;

namespace detail
{

HEYOKA_DLL_PUBLIC bool is_integral(const expression &);
HEYOKA_DLL_PUBLIC bool is_odd_integral_half(const expression &);

} // namespace detail

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
