// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <initializer_list>
#include <random>
#include <sstream>
#include <tuple>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

TEST_CASE("basic")
{
    using binary_op = detail::binary_op;

    REQUIRE(binary_op{}.op() == binary_op::type::add);
    REQUIRE(binary_op{}.lhs() == 0_dbl);
    REQUIRE(binary_op{}.rhs() == 0_dbl);

    REQUIRE(binary_op{binary_op::type::div, 1_dbl, 2_dbl}.op() == binary_op::type::div);

    REQUIRE(binary_op{binary_op::type::div, 1_dbl, 2_dbl}.lhs() == 1_dbl);
    REQUIRE(binary_op{binary_op::type::div, 1_dbl, 2_dbl}.rhs() == 2_dbl);

    {
        const binary_op op{binary_op::type::div, "x"_var, 2_dbl};

        REQUIRE(op.lhs() == "x"_var);
        REQUIRE(op.rhs() == 2_dbl);
    }
}

TEST_CASE("stream")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::ostringstream oss;

        oss << x + y;

        REQUIRE(oss.str() == "(x + y)");
    }

    {
        std::ostringstream oss;

        oss << x - y;

        REQUIRE(oss.str() == "(x - y)");
    }

    {
        std::ostringstream oss;

        oss << x * y;

        REQUIRE(oss.str() == "(x * y)");
    }

    {
        std::ostringstream oss;

        oss << x / y;

        REQUIRE(oss.str() == "(x / y)");
    }
}

TEST_CASE("equality")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(add(x, y) == add(x, y));
    REQUIRE(add(x, y) != sub(x, y));
    REQUIRE(add(x, y) != mul(x, y));
    REQUIRE(add(x, y) != div(x, y));
}

TEST_CASE("hashing")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(hash(add(x, y)) == hash(add(x, y)));
    REQUIRE(hash(add(x, y)) != hash(sub(x, y)));
    REQUIRE(hash(add(x, y)) != hash(mul(x, y)));
    REQUIRE(hash(add(x, y)) != hash(div(x, y)));
}

TEST_CASE("diff var")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(x + y, "x") == 1_dbl);
    REQUIRE(diff(x - y, "y") == -1_dbl);
    REQUIRE(diff(x * y, "x") == y);
    REQUIRE(diff(x / y, "x") == y / (y * y));
}

TEST_CASE("diff par")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(par[0] + y, par[0]) == 1_dbl);
    REQUIRE(diff(par[0] + y, par[1]) == 0_dbl);

    REQUIRE(diff(x - par[0], par[0]) == -1_dbl);
    REQUIRE(diff(x - par[0], par[1]) == 0_dbl);

    REQUIRE(diff(par[2] * y, par[2]) == y);
    REQUIRE(diff(par[2] * y, par[1]) == 0_dbl);

    REQUIRE(diff(par[3] / y, par[3]) == y / (y * y));
    REQUIRE(diff(par[3] / y, par[4]) == 0_dbl);
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = x + y;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = x - y;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == x + y);
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
            outs.resize(batch_size * 8u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {x + y, x - y, x * y, x / y, x + par[0], x - 3_dbl, par[0] * y, 1_dbl / y},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.add."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.mul."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sub."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.div."));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == ins[i] + ins[i + batch_size]);
                REQUIRE(outs[i + batch_size] == ins[i] - ins[i + batch_size]);
                REQUIRE(outs[i + 2u * batch_size] == ins[i] * ins[i + batch_size]);
                REQUIRE(outs[i + 3u * batch_size] == ins[i] / ins[i + batch_size]);
                REQUIRE(outs[i + 4u * batch_size] == ins[i] + pars[i]);
                REQUIRE(outs[i + 5u * batch_size] == ins[i] - 3.);
                REQUIRE(outs[i + 6u * batch_size] == pars[i] * ins[i + batch_size]);
                REQUIRE(outs[i + 7u * batch_size] == 1. / ins[i + batch_size]);
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
