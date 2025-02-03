// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_HAMILTONIAN_HPP
#define HEYOKA_HAMILTONIAN_HPP

#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>>
hamiltonian(const expression &, const std::vector<expression> &, const std::vector<expression> &);

HEYOKA_END_NAMESPACE

#endif
