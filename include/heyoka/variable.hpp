// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_VARIABLE_HPP
#define HEYOKA_VARIABLE_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <functional>
#include <ostream>
#include <string>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC void swap(variable &, variable &) noexcept;

class HEYOKA_DLL_PUBLIC variable
{
    friend HEYOKA_DLL_PUBLIC void swap(variable &, variable &) noexcept;

    std::string m_name;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    variable();
    explicit variable(std::string);
    variable(const variable &);
    variable(variable &&) noexcept;
    ~variable();

    variable &operator=(const variable &);
    variable &operator=(variable &&) noexcept;

    [[nodiscard]] const std::string &name() const noexcept;
};

namespace detail
{

HEYOKA_DLL_PUBLIC std::size_t hash(const variable &) noexcept;

} // namespace detail

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const variable &);

HEYOKA_DLL_PUBLIC bool operator==(const variable &, const variable &) noexcept;
HEYOKA_DLL_PUBLIC bool operator!=(const variable &, const variable &) noexcept;

HEYOKA_END_NAMESPACE

namespace std
{

template <>
struct hash<heyoka::variable> {
    size_t operator()(const heyoka::variable &v) const noexcept
    {
        return heyoka::detail::hash(v);
    }
};

} // namespace std

#endif
