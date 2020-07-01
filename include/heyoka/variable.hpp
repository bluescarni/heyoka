// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_VARIABLE_HPP
#define HEYOKA_VARIABLE_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC variable
{
    std::string m_name;

public:
    explicit variable(std::string);
    variable(const variable &);
    variable(variable &&) noexcept;
    ~variable();

    std::string &name();
    const std::string &name() const;
};

inline namespace literals
{

HEYOKA_DLL_PUBLIC expression operator""_var(const char *, std::size_t);

}

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const variable &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const variable &);

HEYOKA_DLL_PUBLIC bool operator==(const variable &, const variable &);
HEYOKA_DLL_PUBLIC bool operator!=(const variable &, const variable &);

HEYOKA_DLL_PUBLIC expression diff(const variable &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const variable &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void compute_connections(const variable &, std::vector<std::vector<unsigned>> &, unsigned &);


} // namespace heyoka

#endif
