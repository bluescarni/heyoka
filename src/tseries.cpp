// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/tseries.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

struct tseries::impl {
    std::vector<expression> m_cf;
};

struct tseries::ptag {
};

tseries::tseries(std::vector<expression> vex) : m_impl(std::make_unique<impl>(impl{std::move(vex)}))
{
    if (m_impl->m_cf.empty()) [[unlikely]] {
        throw std::invalid_argument("Cannot construct a tseries from an empty list of coefficients");
    }

    // LCOV_EXCL_START
    if (m_impl->m_cf.size() > std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        throw std::overflow_error("Overflow detected during the construction of a tseries: the order is too large");
    }
    // LCOV_EXCL_STOP
}

tseries::tseries(const variable &v, std::uint32_t order)
{
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;

    std::vector<expression> cf_vec;
    cf_vec.reserve(su32_t(order) + 1);
    for (su32_t i = 0; i <= order; ++i) {
        cf_vec.emplace_back(fmt::format("cf_{}_{}", static_cast<std::uint32_t>(i), v.name()));
    }

    m_impl = std::make_unique<impl>(impl{std::move(cf_vec)});
}

template <typename T>
tseries::tseries(ptag, T x, std::uint32_t order)
{
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;

    std::vector<expression> cf_vec;
    cf_vec.reserve(su32_t(order) + 1);
    cf_vec.emplace_back(std::move(x));
    for (std::uint32_t i = 0; i < order; ++i) {
        cf_vec.emplace_back(0.);
    }

    m_impl = std::make_unique<impl>(impl{std::move(cf_vec)});
}

tseries::tseries(number n, std::uint32_t order) : tseries(ptag{}, std::move(n), order) {}

tseries::tseries(param p, std::uint32_t order) : tseries(ptag{}, std::move(p), order) {}

tseries::tseries(const tseries &other) : m_impl(std::make_unique<impl>(*other.m_impl)) {}

tseries::tseries(tseries &&) noexcept = default;

tseries &tseries::operator=(const tseries &other)
{
    if (this != &other) {
        // NOTE: this will correctly revive a moved-from object.
        *this = tseries(other);
    }

    return *this;
}

tseries &tseries::operator=(tseries &&) noexcept = default;

tseries::~tseries() = default;

std::uint32_t tseries::get_order() const
{
    assert(!m_impl->m_cf.empty());

    return static_cast<std::uint32_t>(m_impl->m_cf.size() - 1u);
}

const std::vector<expression> &tseries::get_cfs() const
{
    return m_impl->m_cf;
}

std::ostream &operator<<(std::ostream &os, const tseries &ts)
{
    os << fmt::format("{}", ts.get_cfs());
    return os;
}

namespace detail
{

tseries to_tseries_impl(funcptr_map<tseries> &func_map, const expression &ex, std::uint32_t order)
{
    return std::visit(
        [order, &func_map](const auto &arg) {
            using type = std::remove_cvref_t<decltype(arg)>;

            if constexpr (std::same_as<type, func>) {
                const auto *f_id = arg.get_ptr();

                // Check if we already converted ex to tseries.
                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                auto ret = arg.to_tseries(func_map, order);

                // Put the return value in the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            } else {
                return tseries(arg, order);
            }
        },
        ex.value());
}

} // namespace detail

std::vector<tseries> to_tseries(const std::vector<expression> &v_ex_, std::uint32_t order)
{
    // Pre-process v_ex with several transformations.

    // Transform sums into subs.
    auto v_ex = detail::sum_to_sub(v_ex_);

    // Split sums.
    v_ex = detail::split_sums_for_decompose(v_ex);

    // Transform prods into divs.
    v_ex = detail::prod_to_div_taylor_diff(v_ex);

    // Split prods.
    // NOTE: split must be 2 here as we implement only
    // binary multiplication for tseries.
    v_ex = detail::split_prods_for_decompose(v_ex, 2);

    detail::funcptr_map<tseries> func_map;

    std::vector<tseries> ret;
    ret.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        ret.push_back(detail::to_tseries_impl(func_map, ex, order));
    }

    return ret;
}

HEYOKA_END_NAMESPACE
