// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PARAM_HPP
#define HEYOKA_PARAM_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC void swap(param &, param &) noexcept;

class HEYOKA_DLL_PUBLIC param
{
    friend HEYOKA_DLL_PUBLIC void swap(param &, param &) noexcept;

    std::uint32_t m_index;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_index;
    }

public:
    param() noexcept;

    explicit param(std::uint32_t) noexcept;

    param(const param &) noexcept;
    param(param &&) noexcept;

    param &operator=(const param &) noexcept;
    param &operator=(param &&) noexcept;

    ~param();

    [[nodiscard]] std::uint32_t idx() const noexcept;
};

namespace detail
{

HEYOKA_DLL_PUBLIC std::size_t hash(const param &) noexcept;

} // namespace detail

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const param &);

HEYOKA_DLL_PUBLIC bool operator==(const param &, const param &) noexcept;
HEYOKA_DLL_PUBLIC bool operator!=(const param &, const param &) noexcept;

HEYOKA_END_NAMESPACE

namespace std
{

template <>
struct hash<heyoka::param> {
    size_t operator()(const heyoka::param &p) const noexcept
    {
        return heyoka::detail::hash(p);
    }
};

} // namespace std

#endif
