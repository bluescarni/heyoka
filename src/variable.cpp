// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <utility>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

void variable::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_name;
}

void variable::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_name;
}

variable::variable() : variable("") {}

variable::variable(std::string s) : m_name(std::move(s)) {}

variable::variable(const variable &) = default;

variable::variable(variable &&) noexcept = default;

variable::~variable() = default;

variable &variable::operator=(const variable &) = default;

variable &variable::operator=(variable &&) noexcept = default;

const std::string &variable::name() const noexcept
{
    return m_name;
}

void swap(variable &v0, variable &v1) noexcept
{
    std::swap(v0.m_name, v1.m_name);
}

namespace detail
{

std::size_t hash(const variable &v) noexcept
{
    return std::hash<std::string>{}(v.name());
}

} // namespace detail

std::ostream &operator<<(std::ostream &os, const variable &var)
{
    return os << var.name();
}

bool operator==(const variable &v1, const variable &v2) noexcept
{
    return v1.name() == v2.name();
}

bool operator!=(const variable &v1, const variable &v2) noexcept
{
    return !(v1 == v2);
}

HEYOKA_END_NAMESPACE
