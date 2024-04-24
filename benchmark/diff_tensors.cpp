// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>

using namespace heyoka;

int main()
{
    set_logger_level_trace();

    auto x = make_vars("x", "y", "z", "px", "py", "pz");

    auto f = sin(pow(x[0] * x[1] - x[2], 5.) * (x[0] + x[1]) * 3.
                 - 0.3 * pow(x[3] * x[4] - x[1], 7.) * pow(x[5] + x[3], 3.));

    auto dt = diff_tensors({f}, std::vector(x.begin(), x.end()), kw::diff_order = 6);
}
