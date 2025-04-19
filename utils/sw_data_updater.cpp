// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <fstream>

#include <fmt/core.h>

#include <heyoka/sw_data.hpp>

int main()
{
    // Download the latest sw data.
    const auto idata = heyoka::sw_data::fetch_latest_celestrak(true);

    // Create the header file first.
    std::ofstream oheader("builtin_sw_data.hpp");
    oheader << fmt::format(R"(#ifndef HEYOKA_DETAIL_SW_DATA_BUILTIN_SW_DATA_HPP
#define HEYOKA_DETAIL_SW_DATA_BUILTIN_SW_DATA_HPP

#include <heyoka/config.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{{

extern const char *const builtin_sw_data_ts;

extern const sw_data_row builtin_sw_data[{}];

}} // namespace detail

HEYOKA_END_NAMESPACE

#endif
)",
                           idata.get_table().size());

    // Now the cpp file.
    std::ofstream ocpp("builtin_sw_data.cpp");
    ocpp << fmt::format(R"(#include <heyoka/config.hpp>
#include <heyoka/detail/sw_data/builtin_sw_data.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{{

const char *const builtin_sw_data_ts = "{}";

const sw_data_row builtin_sw_data[{}] = {{)",
                        idata.get_timestamp(), idata.get_table().size());

    for (const auto &[mjd, Ap_avg, f107, f107a_center81] : idata.get_table()) {
        ocpp << fmt::format("{{.mjd={}, .Ap_avg={}, .f107={}, .f107a_center81={}}},\n", mjd, Ap_avg, f107,
                            f107a_center81);
    }

    ocpp << R"(};

} // namespace detail

HEYOKA_END_NAMESPACE
)";
}
