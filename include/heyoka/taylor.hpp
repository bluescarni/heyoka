// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TAYLOR_HPP
#define HEYOKA_TAYLOR_HPP

#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_decompose(std::vector<expression>);

}

#endif
