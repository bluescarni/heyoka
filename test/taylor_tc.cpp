// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

template <typename Out, typename P, typename T>
auto &horner_eval(Out &ret, const P &p, int order, const T &eval)
{
    ret = xt::view(p, xt::all(), order);

    for (--order; order >= 0; --order) {
        ret = xt::view(p, xt::all(), order) + ret * eval;
    }

    return ret;
}

TEST_CASE("taylor tc basic")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    auto tca = xt::adapt(ta.get_tc_data(), {2u, ta.get_order() + 1u});

    auto [_, h] = ta.step(write_tc::yes);

    auto ret = xt::eval(xt::zeros<double>({2}));

    std::cout << horner_eval(ret, tca, 20, 0) << '\n';
    std::cout << horner_eval(ret, tca, 20, h / 4) << '\n';
    std::cout << horner_eval(ret, tca, 20, h / 2) << '\n';
    std::cout << horner_eval(ret, tca, 20, (3 * h) / 4) << '\n';
    std::cout << horner_eval(ret, tca, 20, h) << '\n';

    std::cout << ta << '\n';
}