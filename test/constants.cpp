// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/math/constants/constants.hpp>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

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

constexpr bool skip_batch_ld =
#if LLVM_VERSION_MAJOR == 13 || LLVM_VERSION_MAJOR == 14 || LLVM_VERSION_MAJOR == 15
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

// Variable template for the constant pi at different levels of precision.
template <typename T>
const auto pi_const = boost::math::constants::pi<T>();

#if defined(HEYOKA_HAVE_REAL128)

template <>
const mppp::real128 pi_const<mppp::real128> = mppp::pi_128;

#endif

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    constant c0;

    REQUIRE(c0.get_str_func_t() == typeid(detail::null_constant_func));
    REQUIRE(std::get<func>(pi.value()).extract<constant>()->get_str_func_t() == typeid(detail::pi_constant_func));

    std::ostringstream oss;
    oss << expression{func{c0}};
    REQUIRE(oss.str() == "null_constant");

    REQUIRE(c0(1u) == "0");
    REQUIRE(c0(237u) == "0");

    REQUIRE(c0.gradient().empty());

    const constant c1(
        "foo", [](unsigned) { return std::string{"0"}; }, "pippo");
    oss.str("");
    oss << expression{func{c1}};
    REQUIRE(oss.str() == "pippo");

    // Error modes.
    REQUIRE_THROWS_MATCHES(constant("foo", {}), std::invalid_argument,
                           Message("Cannot construct a constant with an empty string function"));

    REQUIRE_THROWS_MATCHES(c0(0u), std::invalid_argument,
                           Message("Cannot generate a constant with a precision of zero bits"));

    REQUIRE_THROWS_MATCHES(constant("foo", [](unsigned) { return std::string{"sadsadasd"}; })(1u),
                           std::invalid_argument,
                           Message("The string 'sadsadasd' returned by the implementation of a constant is not a "
                                   "valid representation of a floating-point number in base 10"));
}

TEST_CASE("pi stream")
{
    std::ostringstream oss;

    oss << heyoka::pi;

    REQUIRE(oss.str() == u8"π");
}

TEST_CASE("pi diff")
{
    REQUIRE(diff(heyoka::pi, "x") == 0_dbl);
    REQUIRE(diff(heyoka::pi, par[0]) == 0_dbl);

    auto x = "x"_var;

    REQUIRE(diff(heyoka::pi * cos(2. * x + 2. * heyoka::pi), "x")
            == heyoka::pi * (-2. * sin(2. * x + 2. * heyoka::pi)));

    REQUIRE(diff(heyoka::pi * cos(2. * par[0] + 2. * heyoka::pi), par[0])
            == heyoka::pi * (-2. * sin(2. * par[0] + 2. * heyoka::pi)));
}

TEST_CASE("pi s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = heyoka::pi + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == heyoka::pi + x);
}

TEST_CASE("default s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = expression{func{constant{}}} + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == expression{func{constant{}}} + x);
}

TEST_CASE("pi cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        std::vector<fp_t> outs;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {heyoka::pi}, kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pi."));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), nullptr, nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == pi_const<fp_t>);
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

TEST_CASE("pi cfunc mp")
{
    using fp_t = mppp::real;

    const auto prec = 237u;

    std::vector<fp_t> outs{mppp::real{0, prec}};

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            outs[0] = mppp::real{0, prec};

            add_cfunc<fp_t>(s, "cfunc", {heyoka::pi}, kw::prec = prec, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pi."));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), nullptr, nullptr);

            REQUIRE(outs[0] == mppp::real_pi(prec));
        }
    }
}

#endif
