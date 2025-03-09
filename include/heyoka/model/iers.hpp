// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_IERS_HPP
#define HEYOKA_MODEL_IERS_HPP

#include <memory>
#include <string>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

struct iers_data_row {
    // UTC modified Julian date.
    double mjd = 0;
    // UT1-UTC (seconds).
    double delta_ut1_utc = 0;
};

using iers_data_t = std::vector<iers_data_row>;

[[nodiscard]] HEYOKA_DLL_PUBLIC iers_data_t parse_iers_data(const std::string &);

[[nodiscard]] HEYOKA_DLL_PUBLIC std::shared_ptr<const iers_data_t> get_iers_data();
HEYOKA_DLL_PUBLIC void set_iers_data(iers_data_t);

} // namespace model

HEYOKA_END_NAMESPACE

#endif
