// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/func_args.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

class HEYOKA_DLL_PUBLIC dfun_impl : public func_base
{
    std::string m_id_name;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> m_didx;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // A private ctor used only in the implementation of gradient().
    explicit dfun_impl(std::string, std::string, func_args::shared_args_t,
                       std::vector<std::pair<std::uint32_t, std::uint32_t>>);

public:
    dfun_impl();
    explicit dfun_impl(std::string, std::vector<expression>, std::vector<std::pair<std::uint32_t, std::uint32_t>>);
    explicit dfun_impl(std::string, func_args::shared_args_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>);
    explicit dfun_impl(std::string, func_args, std::vector<std::pair<std::uint32_t, std::uint32_t>>);

    [[nodiscard]] const std::string &get_id_name() const;
    [[nodiscard]] const std::vector<std::pair<std::uint32_t, std::uint32_t>> &get_didx() const;

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::vector<expression> gradient() const;
};

HEYOKA_DLL_PUBLIC bool contains_dfun(const std::vector<expression> &);

} // namespace detail

HEYOKA_DLL_PUBLIC expression dfun(std::string, std::vector<expression>,
                                  std::vector<std::pair<std::uint32_t, std::uint32_t>> = {});

HEYOKA_DLL_PUBLIC expression dfun(std::string, func_args::shared_args_t,
                                  std::vector<std::pair<std::uint32_t, std::uint32_t>> = {});

HEYOKA_DLL_PUBLIC expression dfun(std::string, func_args, std::vector<std::pair<std::uint32_t, std::uint32_t>> = {});

HEYOKA_END_NAMESPACE

// Current archive version is 2.
//
// Changelog:
// - version 2: changed class base to func_base (from the now-deleted
//   shared_func_base class).
BOOST_CLASS_VERSION(heyoka::detail::dfun_impl, 2)

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::dfun_impl)

#endif
