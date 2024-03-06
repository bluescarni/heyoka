// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TSERIES_HPP
#define HEYOKA_TSERIES_HPP

#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

class HEYOKA_DLL_PUBLIC tseries
{
    struct impl;
    std::unique_ptr<impl> m_impl;

    struct ptag;
    template <typename T>
    HEYOKA_DLL_LOCAL explicit tseries(ptag, T, std::uint32_t);

public:
    tseries() = delete;
    explicit tseries(std::vector<expression>);
    explicit tseries(const variable &, std::uint32_t);
    explicit tseries(number, std::uint32_t);
    explicit tseries(param, std::uint32_t);
    tseries(const tseries &);
    tseries(tseries &&) noexcept;
    tseries &operator=(const tseries &);
    tseries &operator=(tseries &&) noexcept;
    ~tseries();

    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] const std::vector<expression> &get_cfs() const;
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const tseries &);

namespace detail
{

HEYOKA_DLL_PUBLIC tseries to_tseries_impl(funcptr_map<tseries> &, const expression &, std::uint32_t);

} // namespace detail

HEYOKA_DLL_PUBLIC std::vector<tseries> to_tseries(const std::vector<expression> &, std::uint32_t);

HEYOKA_END_NAMESPACE

// fmt formatter for tseries, implemented
// on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::tseries> : fmt::ostream_formatter {
};

} // namespace fmt

#endif
