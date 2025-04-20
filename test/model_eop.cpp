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
#include <ranges>
#include <regex>
#include <sstream>
#include <tuple>
#include <utility>
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
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/eop.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

static std::mt19937 rng;

constexpr auto ntrials = 10000;

// NOTE: no point here in testing with precision higher than double. Since the eop data accuracy
// is in general nowhere near double precision, in several places we are at the moment hard-coding calculations
// in double precision (e.g., the computation of the ERA in llvm_get_eop_data_era()).
const auto fp_types = std::tuple<float, double>{};

TEST_CASE("basics")
{
    REQUIRE(model::pm_x() == model::pm_x(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::pm_xp() == model::pm_xp(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::pm_y() == model::pm_y(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::pm_yp() == model::pm_yp(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::dX() == model::dX(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::dXp() == model::dXp(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::dY() == model::dY(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::dYp() == model::dYp(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));

    REQUIRE(std::get<func>(model::pm_x().value()).get_name().starts_with("eop_pm_x_"));
    REQUIRE(std::get<func>(model::pm_xp().value()).get_name().starts_with("eop_pm_xp_"));
    REQUIRE(std::get<func>(model::pm_y().value()).get_name().starts_with("eop_pm_y_"));
    REQUIRE(std::get<func>(model::pm_yp().value()).get_name().starts_with("eop_pm_yp_"));
    REQUIRE(std::get<func>(model::dX().value()).get_name().starts_with("eop_dX_"));
    REQUIRE(std::get<func>(model::dXp().value()).get_name().starts_with("eop_dXp_"));
    REQUIRE(std::get<func>(model::dY().value()).get_name().starts_with("eop_dY_"));
    REQUIRE(std::get<func>(model::dYp().value()).get_name().starts_with("eop_dYp_"));

    auto x = make_vars("x");

    REQUIRE(model::pm_x() != model::pm_x(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::pm_xp() != model::pm_xp(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::pm_y() != model::pm_y(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::pm_yp() != model::pm_yp(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::dX() != model::dX(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::dXp() != model::dXp(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::dY() != model::dY(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::dYp() != model::dYp(kw::time_expr = x, kw::eop_data = eop_data{}));

    REQUIRE(model::pm_x() != model::pm_y());
    REQUIRE(model::pm_x() != model::pm_xp());
    REQUIRE(model::pm_y() != model::pm_yp());
    REQUIRE(model::pm_xp() != model::pm_yp());
}

TEST_CASE("eop s11n")
{
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::pm_x(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::pm_x(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::pm_xp(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::pm_xp(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::pm_y(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::pm_y(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::pm_yp(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::pm_yp(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::dX(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::dX(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::dXp(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::dXp(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::dY(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::dY(kw::time_expr = x));
    }
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::dYp(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::dYp(kw::time_expr = x));
    }
}

TEST_CASE("eop diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::pm_x(kw::time_expr = 2. * x), x) == 2. * model::pm_xp(kw::time_expr = 2. * x));
    REQUIRE(diff(model::pm_y(kw::time_expr = 2. * x), x) == 2. * model::pm_yp(kw::time_expr = 2. * x));
    REQUIRE(diff(model::dX(kw::time_expr = 2. * x), x) == 2. * model::dXp(kw::time_expr = 2. * x));
    REQUIRE(diff(model::dY(kw::time_expr = 2. * x), x) == 2. * model::dYp(kw::time_expr = 2. * x));
}

// NOTE: we will be testing only pm_x here.
TEST_CASE("get_eop_eop_func")
{
    const eop_data data;

    auto tester = [&data]<typename T>() {
        // The function for the computation of eop/eopp.
        using fptr1_t = void (*)(T *, T *, const T *) noexcept;
        // The function to fetch the date/eop data.
        using fptr2_t = void (*)(const T **, const T **) noexcept;

        auto add_test_funcs = [&data](llvm_state &s, std::uint32_t batch_size) {
            auto &ctx = s.context();
            auto &bld = s.builder();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);
            auto *ptr_t = llvm::PointerType::getUnqual(ctx);

            // Add the function for the computation of pm_x/pm_xp.
            auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

            auto *pm_x_ptr = f->getArg(0);
            auto *pm_xp_ptr = f->getArg(1);
            auto *time_ptr = f->getArg(2);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *tm_val = detail::ext_load_vector_from_memory(s, scal_t, time_ptr, batch_size);

            auto *pm_x_pm_xp_f = model::detail::llvm_get_eop_func(s, scal_t, batch_size, data, "pm_x",
                                                                  &detail::llvm_get_eop_data_pm_x);

            auto *pm_x_pm_xp = bld.CreateCall(pm_x_pm_xp_f, tm_val);
            auto *pm_x = bld.CreateExtractValue(pm_x_pm_xp, 0);
            auto *pm_xp = bld.CreateExtractValue(pm_x_pm_xp, 1);

            detail::ext_store_vector_to_memory(s, pm_x_ptr, pm_x);
            detail::ext_store_vector_to_memory(s, pm_xp_ptr, pm_xp);

            bld.CreateRetVoid();

            // Add the function to fetch the date/pm_x data.
            ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t}, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fetch_data", &md);

            auto *date_ptr_ptr = f->getArg(0);
            auto *pm_x_ptr_ptr = f->getArg(1);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *date_data_ptr = detail::llvm_get_eop_sw_data_date_tt_cy_j2000(s, data, scal_t, "eop");
            auto *pm_x_data_ptr = detail::llvm_get_eop_data_pm_x(s, data, scal_t);

            bld.CreateStore(date_data_ptr, date_ptr_ptr);
            bld.CreateStore(pm_x_data_ptr, pm_x_ptr_ptr);

            bld.CreateRetVoid();
        };

        const auto data_size = data.get_table().size();

        // NOTE: here we do a separate linear interpolation to check the results coming out from LLVM.
        auto pm_x_comp = [data_size](const T *date_ptr, const T *pm_x_ptr, T tm) -> std::pair<T, T> {
            // Locate the first date *greater than* tm.
            const auto *const date_it = std::ranges::upper_bound(date_ptr, date_ptr + data_size, tm);
            if (date_it == date_ptr || date_it == date_ptr + data_size) {
                return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
            }

            // Establish the index of the time interval.
            const auto idx = (date_it - 1) - date_ptr;

            // Fetch the time and pm_x values.
            const auto t0 = date_ptr[idx];
            const auto t1 = date_ptr[idx + 1];
            const auto pm_x0 = pm_x_ptr[idx];
            const auto pm_x1 = pm_x_ptr[idx + 1];

            // Run the linear interpolation.
            auto pm_x = (pm_x0 * (t1 - tm) + pm_x1 * (tm - t0)) / (t1 - t0);
            const auto pm_xp = (pm_x1 - pm_x0) / (t1 - t0);

            return std::make_pair(pm_x, pm_xp);
        };

        for (const auto batch_size : {1u, 2u, 4u, 5u, 8u}) {
            // Setup the compiled functions.
            llvm_state s;
            add_test_funcs(s, batch_size);
            s.compile();
            auto *fptr1 = reinterpret_cast<fptr1_t>(s.jit_lookup("test"));
            auto *fptr2 = reinterpret_cast<fptr2_t>(s.jit_lookup("fetch_data"));

            // Fetch the date/pm_x pointers.
            const T *date_ptr{};
            const T *pm_x_ptr{};
            fptr2(&date_ptr, &pm_x_ptr);

            // Prepare the input/output vectors.
            std::vector<T> pm_x_vec(batch_size), pm_xp_vec(batch_size), tm_vec(batch_size);

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

                fptr1(pm_x_vec.data(), pm_xp_vec.data(), tm_vec.data());

                for (auto j = 0u; j < batch_size; ++j) {
                    const auto [pm_x_cmp, pm_xp_cmp] = pm_x_comp(date_ptr, pm_x_ptr, tm_vec[j]);

                    if (std::isnan(pm_x_cmp)) {
                        REQUIRE(std::isnan(pm_x_vec[j]));
                    } else {
                        REQUIRE(pm_x_vec[j] == approximately(pm_x_cmp, T(1000)));
                    }

                    if (std::isnan(pm_xp_cmp)) {
                        REQUIRE(std::isnan(pm_xp_vec[j]));
                    } else {
                        REQUIRE(pm_xp_vec[j] == approximately(pm_xp_cmp, T(1000)));
                    }
                }
            }
        }
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

TEST_CASE("eop cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = make_vars("x");

        std::vector<fp_t> outs, ins;

        for (auto batch_size : {1u}) {
            outs.resize(batch_size * 8u);
            ins.resize(batch_size);

            std::ranges::fill(ins, fp_t(0));

            llvm_state s{kw::opt_level = opt_level};

            // NOTE: here we create one output per eop quantity.
            add_cfunc<fp_t>(s, "cfunc",
                            {model::pm_x(kw::time_expr = x), model::pm_xp(kw::time_expr = x),
                             model::pm_y(kw::time_expr = x), model::pm_yp(kw::time_expr = x),
                             model::dX(kw::time_expr = x), model::dXp(kw::time_expr = x), model::dY(kw::time_expr = x),
                             model::dYp(kw::time_expr = x)},
                            {x}, kw::batch_size = batch_size, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_pm_x_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_pm_xp_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_pm_y_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_pm_yp_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_dX_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_dXp_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_dY_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_dYp_"));
            }

            s.compile();

            if (opt_level == 3u) {
                // NOTE: in the compiled function, we are evaluating eop quantities and their derivatives. The purpose
                // of this check is to make sure that the computation is done with a single call to the combined
                // function. That is, we want to make sure that LLVM understood it does not need to call the same
                // function twice.
                const auto get_eop_eopp_call_regex = std::regex(R"(.*call.*heyoka\.get_.*_.*p\..*)");
                auto count = 0u;
                const auto ir = s.get_ir();
                for (const auto line : ir | std::ranges::views::split('\n')) {
                    // NOTE: libstdc++ bug on large strings:
                    // https://stackoverflow.com/questions/36304204/c-regex-segfault-on-long-sequences
                    if (std::ranges::size(line) > 200u) {
                        continue;
                    }

                    std::cmatch matches;
                    if (std::regex_match(std::ranges::data(line), std::ranges::data(line) + std::ranges::size(line),
                                         matches, get_eop_eopp_call_regex)) {
                        ++count;
                    }
                }
                REQUIRE(count <= 4u);
            }

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(static_cast<fp_t>(2.100929642630355e-07)));
                REQUIRE(outs[i + batch_size] == approximately(static_cast<fp_t>(5.135267713732468e-05)));

                REQUIRE(outs[i + 2u * batch_size] == approximately(static_cast<fp_t>(1.8306813848457274e-06)));
                REQUIRE(outs[i + 3u * batch_size]
                        == approximately(static_cast<fp_t>(-3.364485743478496e-05), fp_t(10000)));

                REQUIRE(outs[i + 4u * batch_size]
                        == approximately(static_cast<fp_t>(-8.804446479573813e-10), fp_t(10000)));
                REQUIRE(outs[i + 5u * batch_size]
                        == approximately(static_cast<fp_t>(2.4968025780561377e-05), fp_t(10000)));

                REQUIRE(outs[i + 6u * batch_size]
                        == approximately(static_cast<fp_t>(-4.801096057859792e-10), fp_t(10000)));
                REQUIRE(outs[i + 7u * batch_size]
                        == approximately(static_cast<fp_t>(7.083127881010322e-06), fp_t(10000)));
            }
        }
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 0, cm); });
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 3, cm); });
    }
}

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
