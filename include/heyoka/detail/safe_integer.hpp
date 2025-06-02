// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_SAFE_INTEGER_HPP
#define HEYOKA_DETAIL_SAFE_INTEGER_HPP

// NOTE: this is a small header that wraps the inclusion of Boost's safe numerics library while enforcing the inclusion
// of <exception> *beforehand*. This is necessary because of an issue in the safe numerics library where
// std::terminate() is used without including the <exception> header, which leads to compilation errors on some
// platforms. See:
//
// https://github.com/boostorg/safe_numerics/issues/139

#include <exception>

#include <boost/safe_numerics/safe_integer.hpp>

#endif
