// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_DFUN_HPP
#define HEYOKA_MATH_DFUN_HPP

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

class HEYOKA_DLL_PUBLIC dfun_impl : public func_base
{
    std::string m_v_name;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> m_didx;

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
        ar & m_v_name;
        ar & m_didx;
    }

public:
    dfun_impl();
    explicit dfun_impl(const expression &, std::vector<expression>,
                       std::vector<std::pair<std::uint32_t, std::uint32_t>>);

    void to_stream(std::ostringstream &) const;

    //[[nodiscard]] std::vector<expression> gradient() const;
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression dfun(const expression &, std::vector<expression>,
                                  std::vector<std::pair<std::uint32_t, std::uint32_t>> = {});

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::dfun_impl)

#endif
