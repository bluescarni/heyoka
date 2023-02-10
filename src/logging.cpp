// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/logging.hpp>

HEYOKA_BEGIN_NAMESPACE

void *create_logger()
{
    return detail::get_logger();
}

void set_logger_level_trace()
{
    detail::get_logger()->set_level(spdlog::level::trace);
}

void set_logger_level_debug()
{
    detail::get_logger()->set_level(spdlog::level::debug);
}

void set_logger_level_info()
{
    detail::get_logger()->set_level(spdlog::level::info);
}

void set_logger_level_warn()
{
    detail::get_logger()->set_level(spdlog::level::warn);
}

void set_logger_level_err()
{
    detail::get_logger()->set_level(spdlog::level::err);
}

void set_logger_level_critical()
{
    detail::get_logger()->set_level(spdlog::level::critical);
}

HEYOKA_END_NAMESPACE
