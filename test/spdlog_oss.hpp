// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TEST_UTILS_SPDLOG_OSS_HPP
#define HEYOKA_TEST_UTILS_SPDLOG_OSS_HPP

#include <memory>
#include <sstream>

namespace heyoka_test
{

class spdlog_oss
{
    struct impl;
    std::unique_ptr<impl> m_impl;

public:
    spdlog_oss();
    ~spdlog_oss();

    std::ostringstream &oss();

    void flush();
};

} // namespace heyoka_test

#endif
