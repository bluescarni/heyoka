// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <memory>
#include <mutex>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <heyoka/logging.hpp>

#include <iostream>

namespace heyoka
{

namespace detail
{

namespace
{

std::once_flag logger_inited;

} // namespace

} // namespace detail

std::shared_ptr<spdlog::logger> get_logger()
{
    std::call_once(detail::logger_inited, []() {
        auto logger = spdlog::create<spdlog::sinks::stdout_color_sink_mt>("heyoka");
#if !defined(NDEBUG)
        logger->set_level(spdlog::level::debug);
#endif
    });

    auto retval = spdlog::get("heyoka");
    assert(static_cast<bool>(retval));

    return retval;
}

} // namespace heyoka
