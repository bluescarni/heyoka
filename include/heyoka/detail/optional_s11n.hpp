// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_OPTIONAL_S11N_HPP
#define HEYOKA_DETAIL_OPTIONAL_S11N_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <optional>

#include <heyoka/s11n.hpp>

namespace boost::serialization
{

// NOTE: inspired by the boost::optional implementation.
template <typename Archive, typename T>
void save(Archive &ar, const std::optional<T> &opt, unsigned)
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
void load(Archive &ar, std::optional<T> &opt, unsigned)
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
void serialize(Archive &ar, std::optional<T> &opt, unsigned v)
{
    split_free(ar, opt, v);
}

} // namespace boost::serialization

#endif
