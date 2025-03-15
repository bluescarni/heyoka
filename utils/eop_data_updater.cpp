// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <fstream>

#include <fmt/core.h>

#include <heyoka/eop_data.hpp>

int main()
{
    // Download the latest eop data.
    const auto idata = heyoka::eop_data::fetch_latest_iers_rapid();

    // Create the header file first.
    std::ofstream oheader("builtin_eop_data.hpp");
    oheader << fmt::format(R"(#ifndef HEYOKA_DETAIL_EOP_DATA_BUILTIN_EOP_DATA_HPP
#define HEYOKA_DETAIL_EOP_DATA_BUILTIN_EOP_DATA_HPP

#include <heyoka/config.hpp>
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{{

extern const char *const builtin_eop_data_ts;

extern const eop_data_row builtin_eop_data[{}];

}} // namespace detail

HEYOKA_END_NAMESPACE

#endif
)",
                           idata.get_table().size());

    // Now the cpp file.
    std::ofstream ocpp("builtin_eop_data.cpp");
    ocpp << fmt::format(R"(#include <heyoka/config.hpp>
#include <heyoka/detail/eop_data/builtin_eop_data.hpp>
#include <heyoka/eop_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{{

const char *const builtin_eop_data_ts = "{}";

const eop_data_row builtin_eop_data[{}] = {{)",
                        idata.get_timestamp(), idata.get_table().size());

    for (const auto &[mjd, cur_delta_ut1_utc] : idata.get_table()) {
        ocpp << fmt::format("{{.mjd={}, .delta_ut1_utc={}}},\n", mjd, cur_delta_ut1_utc);
    }

    ocpp << R"(};

} // namespace detail

HEYOKA_END_NAMESPACE
)";
}
