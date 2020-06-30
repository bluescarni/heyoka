// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_NUMBER_HPP
#define HEYOKA_NUMBER_HPP

#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC number
{
public:
    using value_type = std::variant<double, long double>;

private:
    value_type m_value;

public:
    explicit number(double);
    explicit number(long double);
    number(const number &);
    number(number &&) noexcept;
    ~number();

    value_type &value();
    const value_type &value() const;
};

inline namespace literals
{

HEYOKA_DLL_PUBLIC expression operator""_dbl(long double);
HEYOKA_DLL_PUBLIC expression operator""_dbl(unsigned long long);

HEYOKA_DLL_PUBLIC expression operator""_ldbl(long double);
HEYOKA_DLL_PUBLIC expression operator""_ldbl(unsigned long long);

} // namespace literals

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const number &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const number &);

HEYOKA_DLL_PUBLIC bool is_zero(const number &);
HEYOKA_DLL_PUBLIC bool is_one(const number &);
HEYOKA_DLL_PUBLIC bool is_negative_one(const number &);

HEYOKA_DLL_PUBLIC number operator+(number, number);
HEYOKA_DLL_PUBLIC number operator-(number, number);
HEYOKA_DLL_PUBLIC number operator*(number, number);
HEYOKA_DLL_PUBLIC number operator/(number, number);

HEYOKA_DLL_PUBLIC bool operator==(const number &, const number &);
HEYOKA_DLL_PUBLIC bool operator!=(const number &, const number &);

HEYOKA_DLL_PUBLIC expression diff(const number &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const number &, const std::unordered_map<std::string, double> &);

} // namespace heyoka

#endif
