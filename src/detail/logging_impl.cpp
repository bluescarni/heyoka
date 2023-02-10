// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this needs to go first because of the
// SPDLOG_ACTIVE_LEVEL definition.
#include <heyoka/detail/logging_impl.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

auto make_logger()
{
    auto ret = spdlog::stdout_color_mt("heyoka");
#if !defined(NDEBUG)
    ret->info("heyoka logger initialised");
#endif

    return ret;
}

} // namespace

spdlog::logger *get_logger()
{
    static auto ret = make_logger();

    return ret.get();
}

} // namespace detail

HEYOKA_END_NAMESPACE
