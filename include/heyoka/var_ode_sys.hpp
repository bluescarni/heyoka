// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_VAR_ODE_SYS_HPP
#define HEYOKA_VAR_ODE_SYS_HPP

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

enum class var_args : unsigned { vars = 0b001, params = 0b010, time = 0b100, all = 0b111 };

[[nodiscard]] HEYOKA_DLL_PUBLIC var_args operator|(var_args, var_args) noexcept;
[[nodiscard]] HEYOKA_DLL_PUBLIC bool operator&(var_args, var_args) noexcept;

class HEYOKA_DLL_PUBLIC var_ode_sys
{
    struct impl;
    std::unique_ptr<impl> m_impl;

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    explicit var_ode_sys(const std::vector<std::pair<expression, expression>> &,
                         const std::variant<var_args, std::vector<expression>> &, std::uint32_t = 1);
    var_ode_sys(const var_ode_sys &);
    var_ode_sys(var_ode_sys &&) noexcept;
    var_ode_sys &operator=(const var_ode_sys &);
    var_ode_sys &operator=(var_ode_sys &&) noexcept;
    ~var_ode_sys();

    [[nodiscard]] const std::vector<std::pair<expression, expression>> &get_sys() const noexcept;
    [[nodiscard]] const std::vector<expression> &get_vargs() const noexcept;
};

HEYOKA_END_NAMESPACE

#endif
