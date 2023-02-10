// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_S11N_HPP
#define HEYOKA_S11N_HPP

#include <cassert>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

// NOTE: workaround for a GCC bug when including the Boost.Serialization
// support for std::shared_ptr:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84075

#if defined(__GNUC__) && __GNUC__ >= 7

namespace boost::serialization
{

struct U {
};

} // namespace boost::serialization

#endif

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

// NOTE: we used to have polymorphic
// archives here instead, but apparently
// those do not support long double and thus
// lead to compilation errors when trying
// to (de)serialize numbers.
// NOTE: we also had text archives here, but
// they have issues with the infinite time
// values in the batch integrator.
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Implementation detail for loading a std::variant.
template <typename Archive, typename... Args, std::size_t... Is>
inline void s11n_variant_load_impl(Archive &ar, std::variant<Args...> &var, std::size_t idx, std::index_sequence<Is...>)
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

// Implement serialization for a few std types currently missing
// from Boost. We should drop these if/when they become available
// in Boost.

namespace boost::serialization
{

// NOTE: inspired by the boost::variant implementation.
template <typename Archive, typename... Args>
inline void save(Archive &ar, const std::variant<Args...> &var, unsigned)
{
    const auto idx = var.index();

    ar << idx;

    std::visit([&ar](const auto &x) { ar << x; }, var);
}

template <typename Archive, typename... Args>
inline void load(Archive &ar, std::variant<Args...> &var, unsigned)
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
inline void serialize(Archive &ar, std::variant<Args...> &var, unsigned v)
{
    split_free(ar, var, v);
}

template <typename Archive, typename... Args>
inline void serialize(Archive &ar, std::tuple<Args...> &tup, unsigned)
{
    // NOTE: this is a right fold, which, in conjunction with the
    // builtin comma operator, ensures that the serialisation of
    // the tuple elements proceeds in the correct order and with
    // the correct sequencing.
    std::apply([&ar](auto &...x) { (void(ar & x), ...); }, tup);
}

// NOTE: inspired by the boost::optional implementation.
template <typename Archive, typename T>
inline void save(Archive &ar, const std::optional<T> &opt, unsigned)
{
    // First check if opt contains something.
    const auto flag = static_cast<bool>(opt);
    ar << flag;

    if (flag) {
        // Save the contained value.
        ar << *opt;
    }
}

template <typename Archive, typename T>
inline void load(Archive &ar, std::optional<T> &opt, unsigned)
{
    // Recover the flag.
    bool flag{};
    ar >> flag;

    if (!flag) {
        // No value in the archive,
        // reset opt and return.
        opt.reset();

        return;
    }

    if (!opt) {
        // opt is currently empty, reset it
        // to the def-cted value.
        opt = T{};
    }

    // Deserialise the value.
    ar >> *opt;
}

template <typename Archive, typename T>
inline void serialize(Archive &ar, std::optional<T> &opt, unsigned v)
{
    split_free(ar, opt, v);
}

} // namespace boost::serialization

#endif
