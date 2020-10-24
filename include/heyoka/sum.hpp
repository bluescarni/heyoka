// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_SUM_HPP
#define HEYOKA_SUM_HPP

#include <initializer_list>
#include <utility>
#include <vector>

#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

namespace heyoka
{

HEYOKA_DLL_PUBLIC expression sum(std::vector<expression>);

template <typename... Args>
inline auto sum(Args... args) -> decltype(sum(std::vector<expression>{std::move(args)...}))
{
    return sum(std::vector<expression>{std::move(args)...});
}

} // namespace heyoka

#endif
