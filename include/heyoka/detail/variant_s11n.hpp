// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VARIANT_S11N_HPP
#define HEYOKA_DETAIL_VARIANT_S11N_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <heyoka/config.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Implementation detail for loading a std::variant.
template <typename Archive, typename... Args, std::size_t... Is>
void s11n_variant_load_impl(Archive &ar, std::variant<Args...> &var, std::size_t idx, std::index_sequence<Is...>)
{
    auto loader = [&ar, &var, idx](auto val) {
        constexpr auto N = decltype(val)::value;

        if (N == idx) {
            // NOTE: deserialise into a temporary, then move
            // it into the variant.
            std::variant_alternative_t<N, std::variant<Args...>> x;
            ar >> x;
            var = std::move(x);

            // Inform the archive of the new address
            // of the object we just deserialised.
            ar.reset_object_address(&std::get<N>(var), &x);

            return true;
        } else {
            assert(N < idx);

            return false;
        }
    };

    // LCOV_EXCL_START
    [[maybe_unused]] auto ret = (loader(std::integral_constant<std::size_t, Is>{}) || ...);

    assert(ret);
    assert(var.index() == idx);
    // LCOV_EXCL_STOP
}

} // namespace detail

HEYOKA_END_NAMESPACE

namespace boost::serialization
{

// NOTE: inspired by the boost::variant implementation.
template <typename Archive, typename... Args>
void save(Archive &ar, const std::variant<Args...> &var, unsigned)
{
    const auto idx = var.index();

    ar << idx;

    std::visit([&ar](const auto &x) { ar << x; }, var);
}

template <typename Archive, typename... Args>
void load(Archive &ar, std::variant<Args...> &var, unsigned)
{
    // Recover the variant index.
    std::size_t idx{};
    ar >> idx;

    // LCOV_EXCL_START
    if (idx >= sizeof...(Args)) {
        throw std::invalid_argument("Invalid index loaded during the deserialisation of a variant");
    }
    // LCOV_EXCL_STOP

    heyoka::detail::s11n_variant_load_impl(ar, var, idx, std::make_index_sequence<sizeof...(Args)>{});
}

template <typename Archive, typename... Args>
void serialize(Archive &ar, std::variant<Args...> &var, unsigned v)
{
    split_free(ar, var, v);
}

} // namespace boost::serialization

#endif
