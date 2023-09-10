// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_KW_HPP
#define HEYOKA_KW_HPP

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>

// NOTE: these are keyword arguments that are
// shared among several files.

HEYOKA_BEGIN_NAMESPACE

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(masses);
IGOR_MAKE_NAMED_ARGUMENT(omega);
IGOR_MAKE_NAMED_ARGUMENT(Gconst);
IGOR_MAKE_NAMED_ARGUMENT(time);
IGOR_MAKE_NAMED_ARGUMENT(compact_mode);
IGOR_MAKE_NAMED_ARGUMENT(high_accuracy);
IGOR_MAKE_NAMED_ARGUMENT(parallel_mode);
IGOR_MAKE_NAMED_ARGUMENT(prec);
IGOR_MAKE_NAMED_ARGUMENT(mu);

} // namespace kw

HEYOKA_END_NAMESPACE

#endif
