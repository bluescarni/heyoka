// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_IERS_HPP
#define HEYOKA_MODEL_IERS_HPP

#include <string>
#include <vector>

#include <boost/smart_ptr/shared_ptr.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

struct HEYOKA_DLL_PUBLIC iers_data_row {
    // UTC modified Julian date.
    double mjd = 0;
    // UT1-UTC (seconds).
    double delta_ut1_utc = 0;

    bool operator==(const iers_data_row &) const noexcept;
};

using iers_data_t = std::vector<iers_data_row>;

[[nodiscard]] HEYOKA_DLL_PUBLIC iers_data_t parse_iers_data(const std::string &);

// NOTE: here we have to use boost::shared_ptr instead of std::shared_ptr due to the
// lack of std::atomic<std::shared_ptr> on OSX.
[[nodiscard]] HEYOKA_DLL_PUBLIC boost::shared_ptr<const iers_data_t> get_iers_data();
HEYOKA_DLL_PUBLIC void set_iers_data(iers_data_t);

} // namespace model

HEYOKA_END_NAMESPACE

#endif
