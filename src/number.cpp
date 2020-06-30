// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

#include <heyoka/number.hpp>

namespace heyoka
{

number::number(double x) : m_value(x) {}

number::number(const number &) = default;

number::number(number &&) noexcept = default;

number::~number() = default;

double &number::value()
{
    return m_value;
}

const double &number::value() const
{
    return m_value;
}

std::ostream &operator<<(std::ostream &os, const number &n)
{
    return os << n.value();
}

std::vector<std::string> get_variables(const number &)
{
    return {};
}

} // namespace heyoka
