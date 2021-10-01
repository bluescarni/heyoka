// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_KW_HPP
#define HEYOKA_KW_HPP

#include <heyoka/detail/igor.hpp>

namespace heyoka
{

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(masses);
IGOR_MAKE_NAMED_ARGUMENT(points);
IGOR_MAKE_NAMED_ARGUMENT(omega);
IGOR_MAKE_NAMED_ARGUMENT(state);
IGOR_MAKE_NAMED_ARGUMENT(Gconst);
IGOR_MAKE_NAMED_ARGUMENT(time);

} // namespace kw

} // namespace heyoka

#endif
