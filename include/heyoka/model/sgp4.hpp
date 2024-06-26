// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_SGP4_HPP
#define HEYOKA_MODEL_SGP4_HPP

#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

HEYOKA_DLL_PUBLIC std::pair<std::vector<expression>, std::vector<expression>> sgp4();

} // namespace model

HEYOKA_END_NAMESPACE

#endif
