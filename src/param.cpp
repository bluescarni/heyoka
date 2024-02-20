// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <utility>

#include <fmt/format.h>

#include <heyoka/config.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/param.hpp>

HEYOKA_BEGIN_NAMESPACE

param::param() noexcept : param(0) {}

param::param(std::uint32_t idx) noexcept : m_index(idx) {}

param::param(const param &) noexcept = default;

param::param(param &&) noexcept = default;

param &param::operator=(const param &) noexcept = default;

param &param::operator=(param &&) noexcept = default;

// NOLINTNEXTLINE(performance-trivially-destructible)
param::~param() = default;

std::uint32_t param::idx() const noexcept
{
    return m_index;
}

void swap(param &p0, param &p1) noexcept
{
    std::swap(p0.m_index, p1.m_index);
}

namespace detail
{

std::size_t hash(const param &p) noexcept
{
    return std::hash<std::uint32_t>{}(p.idx());
}

} // namespace detail

std::ostream &operator<<(std::ostream &os, const param &p)
{
    return os << fmt::format("p{}", p.idx());
}

bool operator==(const param &p0, const param &p1) noexcept
{
    return p0.idx() == p1.idx();
}

bool operator!=(const param &p0, const param &p1) noexcept
{
    return !(p0 == p1);
}

HEYOKA_END_NAMESPACE
