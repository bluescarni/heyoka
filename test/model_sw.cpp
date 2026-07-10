// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <ranges>
#include <sstream>
#include <tuple>
#include <variant>
#include <vector>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/sw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

static std::mt19937 rng;

// NOTE: no point here in testing with precision higher than double, since the sw data accuracy
// is in general nowhere near double precision.
const auto fp_types = std::tuple<float, double>{};

TEST_CASE("basics")
{
    REQUIRE(model::Ap_avg() == model::Ap_avg(kw::time_expr = heyoka::time, kw::sw_data = sw_data{}));
    REQUIRE(model::f107() == model::f107(kw::time_expr = heyoka::time, kw::sw_data = sw_data{}));
    REQUIRE(model::f107a_center81() == model::f107a_center81(kw::time_expr = heyoka::time, kw::sw_data = sw_data{}));

    REQUIRE(std::get<func>(model::Ap_avg().value()).get_name().starts_with("sw_Ap_avg_"));
    REQUIRE(std::get<func>(model::f107().value()).get_name().starts_with("sw_f107_"));
    REQUIRE(std::get<func>(model::f107a_center81().value()).get_name().starts_with("sw_f107a_center81_"));

    auto x = make_vars("x");

    REQUIRE(model::Ap_avg() != model::Ap_avg(kw::time_expr = x, kw::sw_data = sw_data{}));
    REQUIRE(model::f107() != model::f107(kw::time_expr = x, kw::sw_data = sw_data{}));
    REQUIRE(model::f107a_center81() != model::f107a_center81(kw::time_expr = x, kw::sw_data = sw_data{}));

    REQUIRE(model::Ap_avg() != model::f107());
    REQUIRE(model::f107a_center81() != model::f107());
    REQUIRE(model::Ap_avg() != model::f107a_center81());

    // Derivatives: default-construction equivalence.
    REQUIRE(model::Ap_avgp() == model::Ap_avgp(kw::time_expr = heyoka::time, kw::sw_data = sw_data{}));
    REQUIRE(model::f107p() == model::f107p(kw::time_expr = heyoka::time, kw::sw_data = sw_data{}));
    REQUIRE(model::f107a_center81p() == model::f107a_center81p(kw::time_expr = heyoka::time, kw::sw_data = sw_data{}));

    // Derivatives: name mangling.
    REQUIRE(std::get<func>(model::Ap_avgp().value()).get_name().starts_with("sw_Ap_avgp_"));
    REQUIRE(std::get<func>(model::f107p().value()).get_name().starts_with("sw_f107p_"));
    REQUIRE(std::get<func>(model::f107a_center81p().value()).get_name().starts_with("sw_f107a_center81p_"));

    // Derivatives: distinct from their default when given a variable arg.
    REQUIRE(model::Ap_avgp() != model::Ap_avgp(kw::time_expr = x, kw::sw_data = sw_data{}));
    REQUIRE(model::f107p() != model::f107p(kw::time_expr = x, kw::sw_data = sw_data{}));
    REQUIRE(model::f107a_center81p() != model::f107a_center81p(kw::time_expr = x, kw::sw_data = sw_data{}));

    // Derivatives distinct from each other.
    REQUIRE(model::Ap_avgp() != model::f107p());
    REQUIRE(model::f107a_center81p() != model::f107p());
    REQUIRE(model::Ap_avgp() != model::f107a_center81p());

    // A quantity is distinct from its own derivative (verifies the sw_X_ vs sw_Xp_ mangling).
    REQUIRE(model::Ap_avg() != model::Ap_avgp());
    REQUIRE(model::f107() != model::f107p());
    REQUIRE(model::f107a_center81() != model::f107a_center81p());
}

TEST_CASE("sw s11n")
{
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::Ap_avg(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::Ap_avg(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::f107(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::f107(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::f107a_center81(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::f107a_center81(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::Ap_avgp(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::Ap_avgp(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::f107p(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::f107p(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::f107a_center81p(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::f107a_center81p(kw::time_expr = x));
    }
}

TEST_CASE("sw diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::Ap_avg(kw::time_expr = 2. * x), x) == 2. * model::Ap_avgp(kw::time_expr = 2. * x));
    REQUIRE(diff(model::f107(kw::time_expr = 2. * x), x) == 2. * model::f107p(kw::time_expr = 2. * x));
    REQUIRE(diff(model::f107a_center81(kw::time_expr = 2. * x), x)
            == 2. * model::f107a_center81p(kw::time_expr = 2. * x));
}

TEST_CASE("swp diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::Ap_avgp(kw::time_expr = 2. * x), x) == 0_dbl);
    REQUIRE(diff(model::f107p(kw::time_expr = 2. * x), x) == 0_dbl);
    REQUIRE(diff(model::f107a_center81p(kw::time_expr = 2. * x), x) == 0_dbl);
}

TEST_CASE("sw cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using std::isnan;

        auto x = make_vars("x");

        std::vector<fp_t> outs, ins;

        for (auto batch_size : {1u}) {
            outs.resize(batch_size * 3u);
            ins.resize(batch_size);

            std::ranges::fill(ins, fp_t(0));

            // NOTE: here we create one output per sw quantity.
            cfunc<fp_t> cf({model::Ap_avg(kw::time_expr = x), model::f107(kw::time_expr = x),
                            model::f107a_center81(kw::time_expr = x)},
                           {x}, kw::batch_size = batch_size, kw::compact_mode = compact_mode,
                           kw::opt_level = opt_level);

            if (opt_level == 0u && compact_mode) {
                const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
                REQUIRE(std::ranges::any_of(
                    irs, [](const auto &ir) { return boost::contains(ir, "heyoka.llvm_c_eval.sw_Ap_avg_"); }));
                REQUIRE(std::ranges::any_of(
                    irs, [](const auto &ir) { return boost::contains(ir, "heyoka.llvm_c_eval.sw_f107_"); }));
                REQUIRE(std::ranges::any_of(
                    irs, [](const auto &ir) { return boost::contains(ir, "heyoka.llvm_c_eval.sw_f107a_center81_"); }));
            }

            cf(mdspan<fp_t, dextents<std::size_t, 2>>(outs.data(), 3u, batch_size),
               mdspan<const fp_t, dextents<std::size_t, 2>>(ins.data(), 1u, batch_size));

            // NOTE: the numerical correctness of the linear interpolation is exercised by the EOP tests, which use the
            // same underlying machinery. Here we just check that we get finite (non-nan) values out.
            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(!isnan(outs[i]));
                REQUIRE(!isnan(outs[i + batch_size]));
                REQUIRE(!isnan(outs[i + 2u * batch_size]));
            }
        }
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 0, cm); });
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 3, cm); });
    }
}

// End-to-end check that the expression-system f107 function correctly linearly interpolates the underlying data table.
// Unlike the low-level interpolation tests in the EOP suite (which invoke the codegen helper directly), this exercises
// the full path model::f107() -> cfunc, and compares against a linear interpolation performed by hand in the test.
TEST_CASE("f107 interpolation")
{
    const sw_data data;
    const auto &tab = data.get_table();

    // Convert a UTC mjd to TT centuries since J2000.
    auto mjd_to_tt_cy = [](double mjd) {
        double tai1{}, tai2{}, tt1{}, tt2{};
        ::eraUtctai(2400000.5, mjd, &tai1, &tai2);
        ::eraTaitt(tai1, tai2, &tt1, &tt2);
        return ((tt1 - 2451545.) + tt2) / 36525.;
    };

    // Evaluate f107 as a function of time via a compiled function.
    auto x = make_vars("x");
    cfunc<double> cf{{model::f107(kw::time_expr = x)}, {x}};
    auto eval = [&cf](double t) {
        double out{};
        cf(std::ranges::subrange(&out, &out + 1), std::ranges::subrange(&t, &t + 1));
        return out;
    };

    const auto close = [](double a, double b) { return std::abs(a - b) <= 1e-12 * std::abs(b); };

    // Pick an interval well inside the table, advancing to the first one with a non-constant f107 so the interpolation
    // is non-degenerate.
    auto idx = tab.size() / 2u;
    while (idx + 1u < tab.size() && tab[idx].f107 == tab[idx + 1u].f107) {
        ++idx;
    }
    REQUIRE(idx + 1u < tab.size());

    const double t0 = mjd_to_tt_cy(tab[idx].mjd);
    const double t1 = mjd_to_tt_cy(tab[idx + 1u].mjd);
    const double f0 = tab[idx].f107;
    const double f1 = tab[idx + 1u].f107;

    // Interval endpoints.
    REQUIRE(close(eval(t0), f0));
    REQUIRE(close(eval(t1), f1));

    // Interval midpoint.
    REQUIRE(close(eval((t0 + t1) / 2.), (f0 + f1) / 2.));

    // A generic interior point (30% into the interval) - catches weight/sign mistakes the symmetric midpoint would not.
    const double frac = 0.3;
    REQUIRE(close(eval(t0 + (frac * (t1 - t0))), f0 + (frac * (f1 - f0))));
}

#if defined(HEYOKA_HAVE_REAL)

// NOTE: the point of the multiprecision test is just to check we used the correct
// llvm primitives in the implementation.
TEST_CASE("sw cfunc_mp")
{
    auto x = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 3u}) {
            cfunc<mppp::real> cf({model::Ap_avg(kw::time_expr = x), model::f107(kw::time_expr = par[0]),
                                  model::f107a_center81(kw::time_expr = expression{0.})},
                                 {x}, kw::compact_mode = compact_mode, kw::prec = prec, kw::opt_level = opt_level);

            const std::vector ins{mppp::real{0, prec}};
            const std::vector pars{mppp::real{0, prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf(outs, ins, kw::pars = pars);

            auto i = 0u;
            REQUIRE(!isnan(outs[i]));
            REQUIRE(!isnan(outs[i + 1u]));
            REQUIRE(!isnan(outs[i + 2u]));
        }
    }
}

#endif

TEST_CASE("taylor scalar")
{
    using model::Ap_avg;
    using model::f107;
    using model::f107a_center81;
    using model::f107p;

    auto x = "x"_var, y = "y"_var;

    // NOTE: use as base time coordinate 6 hours after J2000.0.
    const auto tm_coord = 0.25 / 36525;

    const auto dyn = {prime(x) = Ap_avg(kw::time_expr = par[0]) + f107(kw::time_expr = 3. * y) * y,
                      prime(y) = f107a_center81(kw::time_expr = 0_dbl)};

    auto scalar_tester = [&dyn, tm_coord, x, y](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        // Convert tm_coord to fp_t.
        const auto tm = static_cast<fp_t>(tm_coord);

        // Create compiled function wrappers for the evaluation of eop/eopp.
        auto Ap_avg_wrapper = [cf = cfunc<fp_t>{{Ap_avg(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto f107_wrapper = [cf = cfunc<fp_t>{{f107(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto f107p_wrapper = [cf = cfunc<fp_t>{{f107p(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto f107a_center81_wrapper
            = [cf = cfunc<fp_t>{{f107a_center81(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
                  fp_t retval{};
                  cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
                  return retval;
              };

        const std::vector<fp_t> pars = {tm};

        auto ta = taylor_adaptive<fp_t>{
            dyn, {tm, -tm}, kw::tol = .1, kw::compact_mode = compact_mode, kw::opt_level = opt_level, kw::pars = pars};

        ta.step(true);

        const auto jet = tc_to_jet(ta);

        REQUIRE(jet[0] == tm);
        REQUIRE(jet[1] == -tm);

        REQUIRE(jet[2] == approximately(Ap_avg_wrapper(pars[0]) + f107_wrapper(3 * jet[1]) * jet[1]));
        REQUIRE(jet[3] == approximately(f107a_center81_wrapper(0.)));

        REQUIRE(jet[4]
                == approximately((3 * f107p_wrapper(3 * jet[1]) * jet[3] * jet[1] + f107_wrapper(3 * jet[1]) * jet[3])
                                 / 2));
        REQUIRE(jet[5] == 0.);

        REQUIRE(jet[6] == approximately(f107p_wrapper(3 * jet[1]) * jet[3] * jet[3]));
        REQUIRE(jet[7] == 0.);
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 0, cm); });
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 3, cm); });
    }
}

// Check that a custom SW dataset built from a contiguous slice of the builtin dataset produces, within the slice's
// covered time range, exactly the same values as the builtin dataset. Outside that range the custom dataset must
// produce NaNs: this proves that the restricted dataset is actually being used, rather than the codegen silently
// falling back to the builtin one.
TEST_CASE("custom dataset equivalence")
{
    const sw_data builtin;
    const auto &full = builtin.get_table();
    REQUIRE(full.size() > 100u);

    // A small contiguous interior slice of the builtin table, reused as a custom dataset. Same rows but a different
    // identifier, hence a distinct symbolic identity/name mangling and a separate set of global arrays in codegen.
    const auto p = full.size() / 2u;
    const std::vector<sw_data_row> slice(full.data() + p, full.data() + p + 8);
    const sw_data custom(slice, "ts", "id");

    auto x = make_vars("x");

    // Approximate tt-cy-since-J2000 bounds of the slice.
    const auto to_cy = [](double mjd) { return (mjd - 51544.5) / 36525.; };
    const auto lo = to_cy(slice.front().mjd);
    const auto hi = to_cy(slice.back().mjd);

    for (const auto compact_mode : {false, true}) {
        // Paired builtin/custom outputs for the SW quantities (all step-wise interpolation).
        cfunc<double> cf({model::Ap_avg(kw::time_expr = x, kw::sw_data = builtin),
                          model::Ap_avg(kw::time_expr = x, kw::sw_data = custom),
                          model::f107(kw::time_expr = x, kw::sw_data = builtin),
                          model::f107(kw::time_expr = x, kw::sw_data = custom),
                          model::f107a_center81(kw::time_expr = x, kw::sw_data = builtin),
                          model::f107a_center81(kw::time_expr = x, kw::sw_data = custom)},
                         {x}, kw::compact_mode = compact_mode);

        std::vector<double> outs(6), ins(1);

        const auto eval = [&cf, &outs, &ins](double t) {
            ins[0] = t;
            cf(mdspan<double, dextents<std::size_t, 2>>(outs.data(), 6u, 1u),
               mdspan<const double, dextents<std::size_t, 2>>(ins.data(), 1u, 1u));
        };

        // Inside the slice: builtin and custom outputs must be identical for every quantity.
        for (auto i = 1; i < 10; ++i) {
            eval(lo + (hi - lo) * (i / 10.));

            REQUIRE(outs[0] == outs[1]);
            REQUIRE(outs[2] == outs[3]);
            REQUIRE(outs[4] == outs[5]);
        }

        // Outside the slice: the custom dataset is out of range (NaN) for every quantity, while the builtin dataset -
        // which the slice was taken from the middle of - is still in range and finite.
        for (const auto t : {lo - (hi - lo), hi + (hi - lo)}) {
            eval(t);

            REQUIRE(std::isnan(outs[1]));
            REQUIRE(std::isnan(outs[3]));
            REQUIRE(std::isnan(outs[5]));

            REQUIRE(!std::isnan(outs[0]));
            REQUIRE(!std::isnan(outs[2]));
            REQUIRE(!std::isnan(outs[4]));
        }
    }
}
