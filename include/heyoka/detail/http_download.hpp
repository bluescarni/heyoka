// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_HTTP_DOWNLOAD_HPP
#define HEYOKA_DETAIL_HTTP_DOWNLOAD_HPP

#include <string>
#include <utility>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

std::pair<std::string, std::string> https_download(const std::string &, unsigned, const std::string &);
std::pair<std::string, std::string> http_download(const std::string &, unsigned, const std::string &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
