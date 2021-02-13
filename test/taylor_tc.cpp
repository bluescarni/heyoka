// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <tuple>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

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
    // Scalar test.
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto [x, v] = make_vars("x", "v");

                auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                  {0.05, 0.025},
                                                  kw::high_accuracy = ha,
                                                  kw::compact_mode = cm,
                                                  kw::opt_level = opt_level};

                REQUIRE(ta.get_tc().size() == 2u * (ta.get_order() + 1u));

                auto tca = xt::adapt(ta.get_tc_data(), {2u, ta.get_order() + 1u});

                auto [oc, h] = ta.step(true);

                auto ret = xt::eval(xt::zeros<double>({2}));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), 0);
                REQUIRE(ret[0] == approximately(0.05, 10.));
                REQUIRE(ret[1] == approximately(0.025, 10.));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
                REQUIRE(ret[0] == approximately(ta.get_state()[0], 10.));
                REQUIRE(ret[1] == approximately(ta.get_state()[1], 10.));

                auto old_state = ta.get_state();

                std::tie(oc, h) = ta.step_backward(true);

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), 0);
                REQUIRE(ret[0] == approximately(old_state[0], 10.));
                REQUIRE(ret[1] == approximately(old_state[1], 10.));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
                REQUIRE(ret[0] == approximately(ta.get_state()[0], 10.));
                REQUIRE(ret[1] == approximately(ta.get_state()[1], 10.));

                old_state = ta.get_state();

                std::tie(oc, h) = ta.step(1e-3, true);

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), 0);
                REQUIRE(ret[0] == approximately(old_state[0], 10.));
                REQUIRE(ret[1] == approximately(old_state[1], 10.));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
                REQUIRE(ret[0] == approximately(ta.get_state()[0], 10.));
                REQUIRE(ret[1] == approximately(ta.get_state()[1], 10.));
            }
        }
    }

    // Batch test.
    for (auto batch_size : {1u, 4u, 23u}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            for (auto cm : {false, true}) {
                for (auto ha : {false, true}) {
                    auto [x, v] = make_vars("x", "v");

                    std::vector<double> init_state;
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.05 + i / 100.);
                    }
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.025 + i / 1000.);
                    }

                    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                            init_state,
                                                            batch_size,
                                                            kw::high_accuracy = ha,
                                                            kw::compact_mode = cm,
                                                            kw::opt_level = opt_level};

                    REQUIRE(ta.get_tc().size() == 2u * (ta.get_order() + 1u) * batch_size);

                    auto tca = xt::adapt(ta.get_tc_data(), {2u, ta.get_order() + 1u, batch_size});
                    auto sa = xt::adapt(ta.get_state_data(), {2u, batch_size});
                    auto isa = xt::adapt(init_state.data(), {2u, batch_size});

                    {
                        auto &oc = ta.step(true);

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto ret = xt::eval(xt::zeros<double>({2u}));

                            horner_eval(ret, xt::view(tca, xt::all(), xt::all(), i), static_cast<int>(ta.get_order()),
                                        0);
                            REQUIRE(ret[0] == approximately(xt::view(isa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(isa, 1, i)[0], 10.));

                            horner_eval(ret, xt::view(tca, xt::all(), xt::all(), i), static_cast<int>(ta.get_order()),
                                        std::get<1>(oc[i]));
                            REQUIRE(ret[0] == approximately(xt::view(sa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(sa, 1, i)[0], 10.));
                        }
                    }

                    init_state = ta.get_state();

                    {
                        auto &oc = ta.step_backward(true);

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto ret = xt::eval(xt::zeros<double>({2u}));

                            horner_eval(ret, xt::view(tca, xt::all(), xt::all(), i), static_cast<int>(ta.get_order()),
                                        0);
                            REQUIRE(ret[0] == approximately(xt::view(isa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(isa, 1, i)[0], 10.));

                            horner_eval(ret, xt::view(tca, xt::all(), xt::all(), i), static_cast<int>(ta.get_order()),
                                        std::get<1>(oc[i]));
                            REQUIRE(ret[0] == approximately(xt::view(sa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(sa, 1, i)[0], 10.));
                        }
                    }

                    init_state = ta.get_state();

                    {
                        std::vector<double> max_delta_t;
                        for (auto i = 0u; i < batch_size; ++i) {
                            max_delta_t.push_back(1e-5 + i * 1e-5);
                        }

                        auto &oc = ta.step(max_delta_t, true);

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto ret = xt::eval(xt::zeros<double>({2u}));

                            horner_eval(ret, xt::view(tca, xt::all(), xt::all(), i), static_cast<int>(ta.get_order()),
                                        0);
                            REQUIRE(ret[0] == approximately(xt::view(isa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(isa, 1, i)[0], 10.));

                            horner_eval(ret, xt::view(tca, xt::all(), xt::all(), i), static_cast<int>(ta.get_order()),
                                        std::get<1>(oc[i]));
                            REQUIRE(ret[0] == approximately(xt::view(sa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(sa, 1, i)[0], 10.));
                        }
                    }
                }
            }
        }
    }
}
