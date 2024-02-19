// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
HEYOKA_BEGIN_NAMESPACE

expression sqrt(expression e)
{
    return pow(std::move(e), .5_dbl);
}

HEYOKA_END_NAMESPACE
