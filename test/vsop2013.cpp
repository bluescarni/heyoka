// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <heyoka/celmec/vsop2013.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

const std::vector dates = {2411545.0, 2415545.0, 2419545.0, 2423545.0, 2427545.0, 2431545.0,
                           2435545.0, 2439545.0, 2443545.0, 2447545.0, 2451545.0};

TEST_CASE("error handling")
{
    using Catch::Matchers::Message;

    REQUIRE_THROWS_MATCHES(vsop2013_elliptic(0, 1), std::invalid_argument,
                           Message("Invalid planet index passed to vsop2013_elliptic(): "
                                   "the index must be in the [1, 9] range, but it is 0 instead"));
    REQUIRE_THROWS_MATCHES(vsop2013_elliptic(10, 1), std::invalid_argument,
                           Message("Invalid planet index passed to vsop2013_elliptic(): "
                                   "the index must be in the [1, 9] range, but it is 10 instead"));
    REQUIRE_THROWS_MATCHES(vsop2013_elliptic(1, 0), std::invalid_argument,
                           Message("Invalid variable index passed to vsop2013_elliptic(): "
                                   "the index must be in the [1, 6] range, but it is 0 instead"));
    REQUIRE_THROWS_MATCHES(vsop2013_elliptic(1, 7), std::invalid_argument,
                           Message("Invalid variable index passed to vsop2013_elliptic(): "
                                   "the index must be in the [1, 6] range, but it is 7 instead"));
    REQUIRE_THROWS_MATCHES(vsop2013_elliptic(1, 1, kw::vsop2013_thresh = -1.), std::invalid_argument,
                           Message("Invalid threshold value passed to vsop2013_elliptic(): "
                                   "the value must be finite and non-negative, but it is -1 instead"));
    REQUIRE_THROWS_AS(vsop2013_elliptic(1, 1, kw::vsop2013_thresh = std::numeric_limits<double>::infinity()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsop2013_elliptic(1, 1, kw::vsop2013_thresh = std::numeric_limits<double>::quiet_NaN()),
                      std::invalid_argument);
}

TEST_CASE("mercury")
{
    std::cout << "Testing Mercury...\n";

    auto x = "x"_var;

    {
        auto merc_a = vsop2013_elliptic(1, 1, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = merc_a}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.3870979635, 0.3870966235, 0.3870965607, 0.3870975307, 0.3870971271, 0.3870990120,
                                    0.3870991050, 0.3870986764, 0.3870984073, 0.3870985734, 0.3870982122};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 1e-8);
        }
    }

    {
        auto merc_lam = vsop2013_elliptic(1, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = merc_lam}, {0.}, kw::compact_mode = true};

        const std::vector values = {6.2605163414, 2.9331298264, 5.8889006181, 2.5615070697, 5.5172901512, 2.1899138863,
                                    5.1457263304, 1.8183546988, 4.7741673767, 1.4467914533, 4.4026055470};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 1e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 1e-8);
        }
    }

    {
        auto merc_k = vsop2013_elliptic(1, 3, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = merc_k}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0452614144, 0.0452099977, 0.0451485382, 0.0450934263, 0.0450275900, 0.0449601649,
                                    0.0448988996, 0.0448363569, 0.0447776649, 0.0447224543, 0.0446647836};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 1e-8);
        }
    }
}

TEST_CASE("venus")
{
    std::cout << "Testing Venus...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(2, 1, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.7233268460, 0.7233324174, 0.7233307847, 0.7233242646, 0.7233283654, 0.7233426547,
                                    0.7233248700, 0.7233262220, 0.7233314949, 0.7233310596, 0.7233269276};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 1e-8);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(2, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {3.0850544129, 1.8375355480, 0.5899962012, 5.6256196213, 4.3780843283, 3.1306248680,
                                    1.8830820759, 0.6355264420, 5.6711894725, 4.4236836479, 3.1761349270};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 2e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 2e-8);
        }
    }

    {
        auto h_sol = vsop2013_elliptic(2, 4, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = h_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0051297811, 0.0050926797, 0.0050804051, 0.0050923304, 0.0051193425, 0.0050791039,
                                    0.0050664598, 0.0050968270, 0.0050966509, 0.0050523635, 0.0050312156};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 2e-8);
        }
    }
}

TEST_CASE("emb")
{
    std::cout << "Testing EMB...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(3, 1, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {1.0000096358, 1.0000051435, 1.0000073760, 1.0000152008, 1.0000198307, 1.0000114829,
                                    1.0000163384, 1.0000033982, 0.9999915689, 0.9999918217, 0.9999964273};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 2e-8);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(3, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {4.8188777642, 4.5123365638, 4.2058195207, 3.8992626620, 3.5927069396, 3.2861200052,
                                    2.9795779728, 2.6730276965, 2.3664845522, 2.0599374998, 1.7534127341};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 2e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 2e-8);
        }
    }

    {
        auto q_sol = vsop2013_elliptic(3, 5, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = q_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0001248730, 0.0001127412, 0.0000987436, 0.0000866254, 0.0000744490, 0.0000620847,
                                    0.0000500081, 0.0000379132, 0.0000244774, 0.0000120455, -0.0000006055};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 2e-8);
        }
    }
}

TEST_CASE("mars")
{
    std::cout << "Testing Mars...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(4, 1, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {1.5236841626, 1.5236124046, 1.5236050441, 1.5236700442, 1.5236699766, 1.5236472115,
                                    1.5236402785, 1.5236249712, 1.5237113425, 1.5237954208, 1.5236789921};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 2e-8);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(4, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {4.7846953863, 3.6698641019, 2.5549157614, 1.4402987476, 0.3256326851, 5.4940145108,
                                    4.3792616969, 3.2645748509, 2.1497402787, 1.0351666881, 6.2038755297};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 2e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 2e-8);
        }
    }

    {
        auto p_sol = vsop2013_elliptic(4, 6, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = p_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0124027403, 0.0123912341, 0.0123796421, 0.0123681256, 0.0123558641, 0.0123428097,
                                    0.0123306142, 0.0123188895, 0.0123067748, 0.0122967829, 0.0122862564};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 2e-8);
        }
    }
}

TEST_CASE("jupiter")
{
    std::cout << "Testing Jupiter...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(5, 1, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {5.2027787025, 5.2038176826, 5.2031571728, 5.2036777563, 5.2028498329, 5.2019289799,
                                    5.2022852460, 5.2027385182, 5.2023828005, 5.2031355531, 5.2042666358};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 3e-8);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(5, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {5.4273729856, 4.9451607529, 4.4623167354, 3.9785888270, 3.4960804302, 3.0133520898,
                                    2.5315367461, 2.0491122123, 1.5666703819, 1.0827082985, 0.5999763772};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 3e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 3e-8);
        }
    }

    {
        auto k_sol = vsop2013_elliptic(5, 3, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = k_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0474307493, 0.0471559767, 0.0466623835, 0.0463956959, 0.0467350940, 0.0474068314,
                                    0.0473940279, 0.0467765874, 0.0464356668, 0.0463907506, 0.0469878209};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 2e-8);
        }
    }
}

TEST_CASE("saturn")
{
    std::cout << "Testing Saturn...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(6, 1, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {9.5223420509, 9.5804354890, 9.5168564262, 9.5761016066, 9.5333198276, 9.5707010600,
                                    9.5556956862, 9.5373716711, 9.5773036171, 9.5206673886, 9.5820171858};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 4e-8);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(6, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {2.6372558976, 4.9717748287, 1.0254146612, 3.3661620553, 5.7000637562, 1.7571326506,
                                    4.0881041894, 0.1451158582, 2.4779176849, 4.8205638410, 0.8727423149};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 3e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 3e-8);
        }
    }

    {
        auto h_sol = vsop2013_elliptic(6, 4, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = h_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0534078734, 0.0511811007, 0.0532426486, 0.0519988373, 0.0533175060, 0.0564567139,
                                    0.0534997059, 0.0551206798, 0.0574331445, 0.0549642683, 0.0557223948};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 3e-8);
        }
    }
}

TEST_CASE("uranus")
{
    std::cout << "Testing Uranus...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(7, 1, kw::vsop2013_time = par[0], kw::vsop2013_thresh = 1e-8);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values
            = {19.2176223319, 19.3078158741, 19.2609787024, 19.1641280095, 19.1513346057, 19.2256858388,
               19.2860044093, 19.2488403482, 19.1729005374, 19.1614227772, 19.2294114915};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 5e-7);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(7, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {3.5628257887, 4.3814332273, 5.1965156941, 6.0140255405, 0.5528437584, 1.3757063112,
                                    2.1928483940, 3.0072821232, 3.8258250751, 4.6497814972, 5.4713784606};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 3e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 3e-8);
        }
    }

    {
        auto q_sol = vsop2013_elliptic(7, 5, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = q_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0018734886, 0.0018655719, 0.0018626230, 0.0018640142, 0.0018604021, 0.0018638623,
                                    0.0018579532, 0.0018524703, 0.0018468533, 0.0018557187, 0.0018595555};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 3e-8);
        }
    }
}

TEST_CASE("neptune")
{
    std::cout << "Testing Neptune...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(8, 1, kw::vsop2013_time = par[0], kw::vsop2013_thresh = 1e-8);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values
            = {30.0385847963, 29.9279835646, 30.0170916189, 30.1299497658, 30.1872688174, 30.2554951881,
               30.2115050254, 30.0346199466, 29.9603948982, 30.0263849856, 30.1036378528};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 5e-7);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(8, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {1.1425611229, 1.5621276309, 1.9826291995, 2.4032479642, 2.8186551969, 3.2313775475,
                                    3.6473388606, 4.0649027402, 4.4837211336, 4.9061335825, 5.3268940339};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 3e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 3e-8);
        }
    }

    {
        auto p_sol = vsop2013_elliptic(8, 6, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = p_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {0.0115258164, 0.0115145041, 0.0115137290, 0.0115053819, 0.0115078692, 0.0115165839,
                                    0.0115253141, 0.0115221121, 0.0115153100, 0.0115124106, 0.0115020499};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 3e-8);
        }
    }
}

TEST_CASE("pluto")
{
    std::cout << "Testing Pluto...\n";

    auto x = "x"_var;

    {
        auto a_sol = vsop2013_elliptic(9, 1, kw::vsop2013_time = par[0], kw::vsop2013_thresh = 1e-8);
        auto ta = taylor_adaptive<double>{{prime(x) = a_sol}, {0.}, kw::compact_mode = true};

        const std::vector values
            = {39.4227219159, 39.3129146524, 39.3965378947, 39.4657239282, 39.5593694964, 39.7847670022,
               39.8105744754, 39.6397923115, 39.5355974284, 39.3577302458, 39.2648542648};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 5e-7);
        }
    }

    {
        auto lam_sol = vsop2013_elliptic(9, 2, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = lam_sol}, {0.}, kw::compact_mode = true};

        const std::vector values = {1.3910260961, 1.6701997041, 1.9513031849, 2.2334191944, 2.5096847076, 2.7848046342,
                                    3.0604019239, 3.3340683196, 3.6105864836, 3.8918549636, 4.1726045776};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(std::sin(ta.get_state()[0]) - std::sin(values[i])) < 3e-8);
            REQUIRE(std::abs(std::cos(ta.get_state()[0]) - std::cos(values[i])) < 3e-8);
        }
    }

    {
        auto k_sol = vsop2013_elliptic(9, 3, kw::vsop2013_time = par[0]);
        auto ta = taylor_adaptive<double>{{prime(x) = k_sol}, {0.}, kw::compact_mode = true};

        const std::vector values
            = {-0.1777508423, -0.1783612429, -0.1790052702, -0.1807103287, -0.1812211321, -0.1845941835,
               -0.1852877487, -0.1822720892, -0.1808345328, -0.1773615624, -0.1758641167};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - values[i]) < 3e-8);
        }
    }
}

// Test the conversion to Cartesian coordinates.
TEST_CASE("cartesian")
{
    std::cout << "Checking cartesian...\n";

    {
        auto [x, y, z] = make_vars("x", "y", "z");

        auto cart_merc = vsop2013_cartesian(1, kw::vsop2013_time = par[0], kw::vsop2013_thresh = 1e-8);

        auto ta = taylor_adaptive<double>{{prime(x) = cart_merc[0], prime(y) = cart_merc[1], prime(z) = cart_merc[2]},
                                          {0., 0., 0.},
                                          kw::compact_mode = true};

        const std::vector x_values
            = {0.3493879042, -0.3953232516, 0.2950960732,  -0.3676232510, 0.2077238852, -0.2846205582,
               0.1004921192, -0.1477141027, -0.0153851723, 0.0231248482,  -0.1300935038};
        const std::vector y_values
            = {-0.1615770401, -0.0777332457, -0.2880996654, 0.0614568422, -0.3828848527, 0.1905474525,
               -0.4415450234, 0.2820933984,  -0.4628510334, 0.3063446927, -0.4472876448};
        const std::vector z_values
            = {-0.0453430160, 0.0300460134, -0.0506541931, 0.0388297583, -0.0503462084, 0.0417160738,
               -0.0452790468, 0.0366021553, -0.0363833853, 0.0228975392, -0.0245983783};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = 0;
            ta.get_state_data()[2] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - x_values[i]) < 1e-7);
            REQUIRE(std::abs(ta.get_state()[1] - y_values[i]) < 1e-7);
            REQUIRE(std::abs(ta.get_state()[2] - z_values[i]) < 1e-7);
        }
    }

    {
        auto [vx, vy, vz] = make_vars("vx", "vy", "vz");

        auto cart_merc = vsop2013_cartesian(1, kw::vsop2013_time = par[0], kw::vsop2013_thresh = 1e-8);

        auto ta
            = taylor_adaptive<double>{{prime(vx) = cart_merc[3], prime(vy) = cart_merc[4], prime(vz) = cart_merc[5]},
                                      {0., 0., 0.},
                                      kw::compact_mode = true};

        const std::vector vx_values
            = {0.0063187162, -0.0004137456, 0.0140821085, -0.0104752394, 0.0191072963, -0.0214017916,
               0.0217935195, -0.0305814190, 0.0224772850, -0.0336968841, 0.0213663969};
        const std::vector vy_values
            = {0.0268317850, -0.0263778552, 0.0214796788, -0.0265268619, 0.0148263475, -0.0221969199,
               0.0076915765, -0.0119754462, 0.0005128660, 0.0031518512,  -0.0064479843};
        const std::vector vz_values
            = {0.0016062487,  -0.0021132769, 0.0004564492,  -0.0012007540, -0.0005471246, 0.0001557197,
               -0.0013748907, 0.0018317537,  -0.0020224742, 0.0033512757,  -0.0024878666};

        for (auto i = 0u; i < 11u; ++i) {
            ta.set_time(0);
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = 0;
            ta.get_state_data()[2] = 0;

            ta.get_pars_data()[0] = (dates[i] - 2451545.0) / 365250;
            ta.propagate_until(1);

            REQUIRE(std::abs(ta.get_state()[0] - vx_values[i]) < 1e-7);
            REQUIRE(std::abs(ta.get_state()[1] - vy_values[i]) < 1e-7);
            REQUIRE(std::abs(ta.get_state()[2] - vz_values[i]) < 1e-7);
        }
    }
}
