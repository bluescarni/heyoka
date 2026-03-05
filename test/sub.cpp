// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/sub.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

constexpr bool skip_batch_ld =
#if LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sub_impl ss;

        REQUIRE(ss.args().size() == 2u);
        REQUIRE(ss.args()[0] == 0_dbl);
        REQUIRE(ss.args()[1] == 0_dbl);
        REQUIRE(ss.get_name() == "sub");
    }

    {
        detail::sub_impl ss(x, y);

        REQUIRE(ss.args() == std::vector{x, y});
    }
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = detail::sub(x, y);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == detail::sub(x, y));
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-1., 1.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 4u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            cfunc<fp_t> cf({detail::sub(expression{fp_t{-1}}, x), detail::sub(x, par[0]),
                            detail::sub(1_dbl, par[0]), detail::sub(x, y)},
                           {x, y}, kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                           kw::compact_mode = compact_mode, kw::opt_level = opt_level);

            if (opt_level == 0u && compact_mode) {
                const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
                REQUIRE(std::ranges::any_of(irs, [](const auto &ir) {
                    return boost::contains(ir, "heyoka.llvm_c_eval.sub.");
                }));
            }

            cf(mdspan<fp_t, dextents<std::size_t, 2>>(outs.data(), 4u, batch_size),
               mdspan<const fp_t, dextents<std::size_t, 2>>(ins.data(), 2u, batch_size),
               kw::pars = mdspan<const fp_t, dextents<std::size_t, 2>>(pars.data(), 1u, batch_size));

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(-fp_t(1) - ins[i], fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(ins[i] - pars[i], fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(fp_t(1) - pars[i], fp_t(100)));
                REQUIRE(outs[i + 3u * batch_size] == approximately(ins[i] - ins[i + batch_size], fp_t(100)));
            }
        }
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc_mp")
{
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            cfunc<mppp::real> cf({detail::sub(expression{mppp::real{-1, prec}}, x), detail::sub(x, par[0]),
                                  detail::sub(expression{mppp::real{1, prec}}, par[0]), detail::sub(x, y)},
                                 {x, y}, kw::compact_mode = compact_mode, kw::prec = prec,
                                 kw::opt_level = opt_level);

            const std::vector ins{mppp::real{".7", prec}, mppp::real{"-.3", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            std::vector<mppp::real> outs(4u, mppp::real{0, prec});

            cf(outs, ins, kw::pars = pars);

            auto i = 0u;
            REQUIRE(outs[i] == approximately(-1 - ins[i]));
            REQUIRE(outs[i + 1u] == approximately(ins[i] - pars[i]));
            REQUIRE(outs[i + 2u] == approximately(1 - pars[i]));
            REQUIRE(outs[i + 3u] == approximately(ins[i] - ins[i + 1]));
        }
    }
}

#endif
