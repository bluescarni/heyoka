// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/logging.hpp>

namespace heyoka
{

void set_logger_level_debug()
{
    detail::get_logger()->set_level(spdlog::level::debug);
}

} // namespace heyoka
