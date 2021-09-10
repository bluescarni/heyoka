// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_NUMBER_HPP
#define HEYOKA_NUMBER_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC number
{
public:
    using value_type = std::variant<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                    ,
                                    mppp::real128
#endif
                                    >;

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
    number();
    explicit number(double);
    explicit number(long double);
#if defined(HEYOKA_HAVE_REAL128)
    explicit number(mppp::real128);
#endif
    number(const number &);
    number(number &&) noexcept;
    ~number();

    number &operator=(const number &);
    number &operator=(number &&) noexcept;

    value_type &value();
    const value_type &value() const;
};

HEYOKA_DLL_PUBLIC void swap(number &, number &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const number &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const number &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const number &);
HEYOKA_DLL_PUBLIC void rename_variables(number &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool is_zero(const number &);
HEYOKA_DLL_PUBLIC bool is_one(const number &);
HEYOKA_DLL_PUBLIC bool is_negative_one(const number &);

HEYOKA_DLL_PUBLIC number operator-(number);

HEYOKA_DLL_PUBLIC number operator+(number, number);
HEYOKA_DLL_PUBLIC number operator-(number, number);
HEYOKA_DLL_PUBLIC number operator*(number, number);
HEYOKA_DLL_PUBLIC number operator/(number, number);

HEYOKA_DLL_PUBLIC bool operator==(const number &, const number &);
HEYOKA_DLL_PUBLIC bool operator!=(const number &, const number &);

HEYOKA_DLL_PUBLIC expression subs(const number &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const number &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const number &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> &);
HEYOKA_DLL_PUBLIC long double eval_ldbl(const number &, const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &);

#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC mppp::real128 eval_f128(const number &, const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &);
#endif

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const number &,
                                      const std::unordered_map<std::string, std::vector<double>> &,
                                      const std::vector<double> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const number &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const number &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const number &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const number &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const number &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *codegen_f128(llvm_state &, const number &);

#endif

template <typename T>
inline llvm::Value *codegen(llvm_state &s, const number &n)
{
    if constexpr (std::is_same_v<T, double>) {
        return codegen_dbl(s, n);
    } else if constexpr (std::is_same_v<T, long double>) {
        return codegen_ldbl(s, n);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return codegen_f128(s, n);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

#endif
