// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
// #include <ranges>
// #include <regex>
#include <sstream>
#include <tuple>
#include <variant>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/sw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

static std::mt19937 rng;

constexpr auto ntrials = 10000;

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
}

TEST_CASE("sw diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::Ap_avg(kw::time_expr = 2. * x), x) == 0_dbl);
    REQUIRE(diff(model::f107(kw::time_expr = 2. * x), x) == 0_dbl);
    REQUIRE(diff(model::f107a_center81(kw::time_expr = 2. * x), x) == 0_dbl);
}

// NOTE: we will be testing only Ap_avg here.
TEST_CASE("get_sw_func")
{
    const sw_data data;

    auto tester = [&data]<typename T>() {
        // The function for the computation of sw.
        using fptr1_t = void (*)(T *, const T *) noexcept;
        // The function to fetch the date/sw data.
        using fptr2_t = void (*)(const T **, const T **) noexcept;

        auto add_test_funcs = [&data](llvm_state &s, std::uint32_t batch_size) {
            auto &ctx = s.context();
            auto &bld = s.builder();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);
            auto *ptr_t = llvm::PointerType::getUnqual(ctx);

            // Add the function for the computation of sw.
            auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

            auto *sw_ptr = f->getArg(0);
            auto *time_ptr = f->getArg(1);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *tm_val = detail::ext_load_vector_from_memory(s, scal_t, time_ptr, batch_size);

            auto *sw_f = model::detail::llvm_get_sw_func(s, scal_t, batch_size, data, "Ap_avg",
                                                         &detail::llvm_get_sw_data_Ap_avg);

            auto *sw = bld.CreateCall(sw_f, tm_val);

            detail::ext_store_vector_to_memory(s, sw_ptr, sw);

            bld.CreateRetVoid();

            // Add the function to fetch the date/sw data.
            ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t}, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fetch_data", &md);

            auto *date_ptr_ptr = f->getArg(0);
            auto *sw_ptr_ptr = f->getArg(1);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *date_data_ptr = detail::llvm_get_eop_sw_data_date_tt_cy_j2000(s, data, scal_t, "sw");
            auto *sw_data_ptr = detail::llvm_get_sw_data_Ap_avg(s, data, scal_t);

            bld.CreateStore(date_data_ptr, date_ptr_ptr);
            bld.CreateStore(sw_data_ptr, sw_ptr_ptr);

            bld.CreateRetVoid();
        };

        const auto data_size = data.get_table().size();

        // NOTE: here we search sw on the C++ side to check the results coming out from LLVM.
        auto sw_comp = [data_size](const T *date_ptr, const T *sw_ptr, T tm) -> T {
            // Locate the first date *greater than* tm.
            const auto *const date_it = std::ranges::upper_bound(date_ptr, date_ptr + data_size, tm);
            if (date_it == date_ptr || date_it == date_ptr + data_size) {
                return std::numeric_limits<T>::quiet_NaN();
            }

            // Establish the index of the time interval.
            const auto idx = (date_it - 1) - date_ptr;

            // Fetch and return the sw value.
            return sw_ptr[idx];
        };

        for (const auto batch_size : {1u, 2u, 4u, 5u, 8u}) {
            // Setup the compiled functions.
            llvm_state s;
            add_test_funcs(s, batch_size);
            s.compile();
            auto *fptr1 = reinterpret_cast<fptr1_t>(s.jit_lookup("test"));
            auto *fptr2 = reinterpret_cast<fptr2_t>(s.jit_lookup("fetch_data"));

            // Fetch the date/sw pointers.
            const T *date_ptr{};
            const T *sw_ptr{};
            fptr2(&date_ptr, &sw_ptr);

            // Prepare the input/output vectors.
            std::vector<T> sw_vec(batch_size), tm_vec(batch_size);

            // Randomised testing.
            const auto first_date = date_ptr[0];
            const auto last_date = date_ptr[data_size - 1u];
            std::uniform_real_distribution<T> date_dist(first_date, last_date);
            std::uniform_int_distribution<int> lp_dist(0, 100);
            for (auto i = 0; i < ntrials; ++i) {
                std::ranges::generate(tm_vec, [&date_dist, &lp_dist, first_date, last_date]() {
                    // With low probability insert a date outside the date bounds.
                    if (lp_dist(rng) == 0) {
                        return lp_dist(rng) < 50 ? first_date - T(0.1) : last_date + T(0.1);
                    }

                    return date_dist(rng);
                });

                fptr1(sw_vec.data(), tm_vec.data());

                for (auto j = 0u; j < batch_size; ++j) {
                    const auto sw_cmp = sw_comp(date_ptr, sw_ptr, tm_vec[j]);

                    if (std::isnan(sw_cmp)) {
                        REQUIRE(std::isnan(sw_vec[j]));
                    } else {
                        REQUIRE(sw_vec[j] == approximately(sw_cmp, T(1000)));
                    }
                }
            }
        }
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

TEST_CASE("sw cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = make_vars("x");

        std::vector<fp_t> outs, ins;

        for (auto batch_size : {1u}) {
            outs.resize(batch_size * 3u);
            ins.resize(batch_size);

            std::ranges::fill(ins, fp_t(0));

            llvm_state s{kw::opt_level = opt_level};

            // NOTE: here we create one output per sw quantity.
            add_cfunc<fp_t>(s, "cfunc",
                            {model::Ap_avg(kw::time_expr = x), model::f107(kw::time_expr = x),
                             model::f107a_center81(kw::time_expr = x)},
                            {x}, kw::batch_size = batch_size, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sw_Ap_avg_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sw_f107_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sw_f107a_center81_"));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(static_cast<fp_t>(30)));
                REQUIRE(outs[i + batch_size] == approximately(static_cast<fp_t>(129.9)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(static_cast<fp_t>(166.2)));
            }
        }
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 0, cm); });
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 3, cm); });
    }
}

#if 0
#if defined(HEYOKA_HAVE_REAL)

// NOTE: the point of the multiprecision test is just to check we used the correct
// llvm primitives in the implementation.
TEST_CASE("eop eopp cfunc_mp")
{
    auto x = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc",
                                  {model::pm_x(kw::time_expr = x), model::pm_x(kw::time_expr = par[0]),
                                   model::dX(kw::time_expr = expression{0.}), model::dYp()},
                                  {x}, kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{0, prec}};
            const std::vector pars{mppp::real{0, prec}};
            const std::vector tm{mppp::real{0, prec}};
            std::vector<mppp::real> outs(4u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), tm.data());

            auto i = 0u;
            REQUIRE(!isnan(outs[i]));
            REQUIRE(!isnan(outs[i + 1u]));
            REQUIRE(!isnan(outs[i + 2u]));
            REQUIRE(!isnan(outs[i + 3u]));

            REQUIRE(outs[i] == outs[i + 1u]);
        }
    }
}

#endif

TEST_CASE("taylor scalar")
{
    using model::dX;
    using model::dXp;
    using model::dY;
    using model::dYp;
    using model::pm_x;
    using model::pm_xp;
    using model::pm_y;
    using model::pm_yp;

    auto x = "x"_var, y = "y"_var;

    // NOTE: use as base time coordinate 6 hours after J2000.0.
    const auto tm_coord = 0.25 / 36525;

    const auto dyn = {prime(x) = pm_x(kw::time_expr = 2. * y) + pm_xp(kw::time_expr = 3. * y) * y
                                 + pm_y(kw::time_expr = number{tm_coord}) + pm_yp(kw::time_expr = -3. * y),
                      prime(y) = dX(kw::time_expr = 4. * x) + dXp(kw::time_expr = par[0]) * x
                                 + dY(kw::time_expr = -4. * x) + dYp(kw::time_expr = -5. * x)};

    auto scalar_tester = [&dyn, tm_coord, x, y](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        // Convert tm_coord to fp_t.
        const auto tm = static_cast<fp_t>(tm_coord);

        // Create compiled function wrappers for the evaluation of eop/eopp.
        auto dX_wrapper = [cf = cfunc<fp_t>{{dX(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto dXp_wrapper = [cf = cfunc<fp_t>{{dXp(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto dY_wrapper = [cf = cfunc<fp_t>{{dY(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto dYp_wrapper = [cf = cfunc<fp_t>{{dYp(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto pm_x_wrapper = [cf = cfunc<fp_t>{{pm_x(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto pm_xp_wrapper = [cf = cfunc<fp_t>{{pm_xp(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto pm_y_wrapper = [cf = cfunc<fp_t>{{pm_y(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };
        auto pm_yp_wrapper = [cf = cfunc<fp_t>{{pm_yp(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
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

        REQUIRE(jet[2]
                == approximately(pm_x_wrapper(2 * jet[1]) + pm_xp_wrapper(3 * jet[1]) * jet[1] + pm_y_wrapper(tm)
                                 + pm_yp_wrapper(-3 * jet[1])));
        REQUIRE(jet[3]
                == approximately(dX_wrapper(4 * jet[0]) + dXp_wrapper(pars[0]) * jet[0] + dY_wrapper(-4 * jet[0])
                                 + dYp_wrapper(-5 * jet[0])));

        REQUIRE(jet[4]
                == approximately((pm_xp_wrapper(2 * jet[1]) * 2 * jet[3] + pm_xp_wrapper(3 * jet[1]) * jet[3]) / 2));
        REQUIRE(jet[5]
                == approximately((dXp_wrapper(4 * jet[0]) * 4 * jet[2] + dXp_wrapper(pars[0]) * jet[2]
                                  + dYp_wrapper(-4 * jet[0]) * -4 * jet[2])
                                 / 2));

        REQUIRE(jet[6]
                == approximately((pm_xp_wrapper(2 * jet[1]) * 2 * 2 * jet[5] + pm_xp_wrapper(3 * jet[1]) * 2 * jet[5])
                                 / 6));
        REQUIRE(jet[7]
                == approximately((dXp_wrapper(4 * jet[0]) * 4 * 2 * jet[4] + dXp_wrapper(pars[0]) * 2 * jet[4]
                                  + dYp_wrapper(-4 * jet[0]) * 2 * -4 * jet[4])
                                 / 6));
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 0, cm); });
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 3, cm); });
    }
}

#endif
