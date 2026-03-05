// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "heyoka/kw.hpp"
#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
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

#include <heyoka/detail/sum_sq.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

// Wrapper to ease the transition of old test code
// after the removal of sum_sq() from the public API.
auto sum_sq(const std::vector<expression> &args)
{
    std::vector<expression> new_args;
    new_args.reserve(args.size());

    for (const auto &arg : args) {
        new_args.push_back(pow(arg, 2_dbl));
    }

    return sum(new_args);
}

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
        detail::sum_sq_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "sum_sq");
    }

    {
        detail::sum_sq_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::sum_sq_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }

    {
        auto ss = sum_sq({0_dbl, 0_dbl, x, 0_dbl});

        REQUIRE(ss == pow(x, 2_dbl));
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == "()");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x**2");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x**2 + y**2)");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x**2 + y**2 + z**2)");
    }
}

TEST_CASE("sum_sq function")
{
    using Catch::Matchers::Message;

    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum_sq({}) == 0_dbl);
    REQUIRE(sum_sq({x}) == pow(x, 2_dbl));
}

TEST_CASE("sum_sq s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = expression{func{detail::sum_sq_impl({x, y, z})}};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == expression{func{detail::sum_sq_impl({x, y, z})}});
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-10., 10.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 3u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            cfunc<fp_t> cf({sum_sq({x, y}), sum_sq({x, expression{fp_t(.5)}}), sum_sq({par[0], y})},
                           {x, y}, kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                           kw::compact_mode = compact_mode, kw::opt_level = opt_level);

            if (opt_level == 0u && compact_mode) {
                const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
                REQUIRE(std::ranges::any_of(irs, [](const auto &ir) {
                    return boost::contains(ir, "heyoka.llvm_c_eval.sum_sq.");
                }));
            }

            cf(mdspan<fp_t, dextents<std::size_t, 2>>(outs.data(), 3u, batch_size),
               mdspan<const fp_t, dextents<std::size_t, 2>>(ins.data(), 2u, batch_size),
               kw::pars = mdspan<const fp_t, dextents<std::size_t, 2>>(pars.data(), 1u, batch_size));

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i]
                        == approximately(ins[i] * ins[i] + ins[i + batch_size] * ins[i + batch_size], fp_t(100)));
                REQUIRE(outs[i + batch_size]
                        == approximately(ins[i] * ins[i] + static_cast<fp_t>(.5) * fp_t(0.5), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size]
                        == approximately(pars[i] * pars[i] + ins[i + batch_size] * ins[i + batch_size], fp_t(100)));
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

    // Small test to check nested sum square replacements.
    auto [x, y] = make_vars("x", "y");

    {
        cfunc<double> cf({sum_sq({x, y, cos(sum_sq({x, y}))})}, {x, y});

        std::vector<double> inputs = {1, 2};
        std::vector<double> output(1u);

        cf(output, inputs);

        REQUIRE(output[0] == approximately(1. + 4 + std::cos(1. + 4) * std::cos(1. + 4)));
    }

    // Check sum_to_sum_sq() failure due to non-numeric exponent.
    {
        cfunc<double> cf({sum({pow(x, 2_dbl), pow(x, y)})}, {x, y},
                         kw::compact_mode = true, kw::opt_level = 0u);

        const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
        REQUIRE(std::ranges::none_of(irs, [](const auto &ir) {
            return boost::contains(ir, "sum_sq");
        }));
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc_mp")
{
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            cfunc<mppp::real> cf({sum_sq({x, y}), sum_sq({x, expression{.5}}), sum_sq({par[0], y})},
                                 {x, y}, kw::compact_mode = compact_mode, kw::prec = prec,
                                 kw::opt_level = opt_level);

            const std::vector ins{mppp::real{".7", prec}, mppp::real{".1", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf(outs, ins, kw::pars = pars);

            auto i = 0u;
            REQUIRE(outs[i] == ins[i] * ins[i] + ins[i + 1u] * ins[i + 1u]);
            REQUIRE(outs[i + 1u] == ins[i] * ins[i] + .5 * .5);
            REQUIRE(outs[i + 2u * 1u] == pars[i] * pars[i] + ins[i + 1u] * ins[i + 1u]);
        }
    }
}

#endif
