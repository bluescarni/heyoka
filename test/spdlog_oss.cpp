// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include "spdlog_oss.hpp"

namespace heyoka_test
{

struct spdlog_oss::impl {
    explicit impl(spdlog::sink_ptr &&log) : orig_logger(std::move(log)) {}

    spdlog::sink_ptr orig_logger;
    std::ostringstream oss;
};

spdlog_oss::spdlog_oss()
{
    auto logger = spdlog::get("heyoka");

    if (!logger) {
        throw std::runtime_error("Logger not inited yet!");
    }

    auto orig_logger = std::move(logger->sinks()[0]);
    logger->sinks().pop_back();

    m_impl = std::make_unique<impl>(std::move(orig_logger));

    auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(m_impl->oss);

    logger->sinks().push_back(std::move(ostream_sink));
}

spdlog_oss::~spdlog_oss()
{
    auto logger = spdlog::get("heyoka");
    logger->sinks().pop_back();
    logger->sinks().push_back(std::move(m_impl->orig_logger));
}

std::ostringstream &spdlog_oss::oss()
{
    return m_impl->oss;
}

void spdlog_oss::flush()
{
    spdlog::get("heyoka")->flush();
}

} // namespace heyoka_test
