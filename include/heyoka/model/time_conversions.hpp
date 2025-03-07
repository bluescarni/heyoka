// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_TIME_CONVERSIONS_HPP
#define HEYOKA_MODEL_TIME_CONVERSIONS_HPP

#include <string>

#include <heyoka/callable.hpp>
#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Implementation of the delta_tt_tai constant.
class HEYOKA_DLL_PUBLIC delta_tt_tai_func
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }

public:
    [[nodiscard]] std::string operator()(unsigned) const;
};

} // namespace detail

// A constant representing the difference TT - TAI (exactly 32.184s).
HEYOKA_DLL_PUBLIC extern const expression delta_tt_tai;

[[nodiscard]] HEYOKA_DLL_PUBLIC expression delta_tdb_tt(const expression & = heyoka::time);

} // namespace model

HEYOKA_END_NAMESPACE

HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::model::detail::delta_tt_tai_func, std::string, unsigned)

#endif
