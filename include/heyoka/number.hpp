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
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC number
{
    double m_value;

public:
    explicit number(double);
    number(const number &);
    number(number &&) noexcept;
    ~number();

    double &value();
    const double &value() const;
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const number &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const number &);

} // namespace heyoka

#endif
