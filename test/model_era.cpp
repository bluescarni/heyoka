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
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/era.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

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
    REQUIRE(model::era() == model::era(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::erap() == model::erap(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));

    auto x = make_vars("x");

    REQUIRE(model::era() != model::era(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::erap() != model::erap(kw::time_expr = x, kw::eop_data = eop_data{}));
}

TEST_CASE("get_era_erap_func")
{
    const eop_data data;

    auto tester = [&data]<typename T>() {
        using arr_t = T[2];

        // The function for the computation of era/erap.
        using fptr1_t = void (*)(T *, T *, const T *) noexcept;
        // The function to fetch the date/era data.
        using fptr2_t = void (*)(const T **, const arr_t **) noexcept;

        auto add_test_funcs = [&data](llvm_state &s, std::uint32_t batch_size) {
            auto &ctx = s.context();
            auto &bld = s.builder();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);
            auto *ptr_t = llvm::PointerType::getUnqual(ctx);

            // Add the function for the computation of era/erap.
            auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

            auto *era_ptr = f->getArg(0);
            auto *erap_ptr = f->getArg(1);
            auto *time_ptr = f->getArg(2);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *tm_val = detail::ext_load_vector_from_memory(s, scal_t, time_ptr, batch_size);

            auto *era_erap_f = model::detail::llvm_get_era_erap_func(s, scal_t, batch_size, data);

            auto *era_erap = bld.CreateCall(era_erap_f, tm_val);
            auto *era = bld.CreateExtractValue(era_erap, 0);
            auto *erap = bld.CreateExtractValue(era_erap, 1);

            detail::ext_store_vector_to_memory(s, era_ptr, era);
            detail::ext_store_vector_to_memory(s, erap_ptr, erap);

            bld.CreateRetVoid();

            // Add the function to fetch the date/era data.
            ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t}, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fetch_data", &md);

            auto *date_ptr_ptr = f->getArg(0);
            auto *era_ptr_ptr = f->getArg(1);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *date_data_ptr = detail::llvm_get_eop_data_date_tt_cy_j2000(s, data, scal_t);
            auto *era_data_ptr = detail::llvm_get_eop_data_era(s, data, scal_t);

            bld.CreateStore(date_data_ptr, date_ptr_ptr);
            bld.CreateStore(era_data_ptr, era_ptr_ptr);

            bld.CreateRetVoid();
        };

        const auto data_size = data.get_table().size();

        // NOTE: the idea for testing is that we compute the ERA in multiprecision and compare the results.
        // Here we implement the helper for the multiprecision computation.
        auto mp_era_comp = [data_size](const T *date_ptr, const arr_t *era_ptr, T tm) -> std::pair<T, T> {
            using oct_t = boost::multiprecision::cpp_bin_float_oct;

            // Locate the first date *greater than* tm.
            const auto *const date_it = std::ranges::upper_bound(date_ptr, date_ptr + data_size, tm);
            if (date_it == date_ptr || date_it == date_ptr + data_size) {
                return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
            }

            // Establish the index of the time interval.
            const auto idx = (date_it - 1) - date_ptr;

            // Fetch the time and era values.
            const auto t0 = oct_t{date_ptr[idx]};
            const auto t1 = oct_t{date_ptr[idx + 1]};
            const auto [era0_hi, era0_lo] = era_ptr[idx];
            const auto [era1_hi, era1_lo] = era_ptr[idx + 1];
            const auto era0 = oct_t{era0_hi} + era0_lo;
            const auto era1 = oct_t{era1_hi} + era1_lo;

            // Run the linear interpolation.
            auto era = (era0 * (t1 - tm) + era1 * (tm - t0)) / (t1 - t0);
            const auto erap = (era1 - era0) / (t1 - t0);

            // Reduce era modulo 2pi.
            const auto twopi = 2 * boost::math::constants::pi<oct_t>();
            era = era - twopi * floor(era / twopi);

            return {static_cast<T>(era), static_cast<T>(erap)};
        };

        for (const auto batch_size : {1u, 2u, 4u, 5u, 8u}) {
            // Setup the compiled functions.
            llvm_state s;
            add_test_funcs(s, batch_size);
            s.compile();
            auto *fptr1 = reinterpret_cast<fptr1_t>(s.jit_lookup("test"));
            auto *fptr2 = reinterpret_cast<fptr2_t>(s.jit_lookup("fetch_data"));

            // Fetch the date/era pointers.
            const T *date_ptr{};
            const arr_t *era_ptr{};
            fptr2(&date_ptr, &era_ptr);

            // Prepare the input/output vectors.
            std::vector<T> era_vec(batch_size), erap_vec(batch_size), tm_vec(batch_size);

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

                fptr1(era_vec.data(), erap_vec.data(), tm_vec.data());

                for (auto j = 0u; j < batch_size; ++j) {
                    const auto [era_cmp, erap_cmp] = mp_era_comp(date_ptr, era_ptr, tm_vec[j]);

                    if (std::isnan(era_cmp)) {
                        REQUIRE(std::isnan(era_vec[j]));
                    } else {
                        REQUIRE(era_vec[j] == approximately(era_cmp));
                    }

                    if (std::isnan(erap_cmp)) {
                        REQUIRE(std::isnan(erap_vec[j]));
                    } else {
                        REQUIRE(erap_vec[j] == approximately(erap_cmp));
                    }
                }
            }
        }
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

TEST_CASE("era s11n")
{
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = model::era(kw::time_expr = x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == model::era(kw::time_expr = x));
}

TEST_CASE("erap s11n")
{
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = model::erap(kw::time_expr = x, kw::eop_data = eop_data());

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == model::erap(kw::time_expr = x));
}

TEST_CASE("era diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::era(kw::time_expr = 2. * x), x) == 2. * model::erap(kw::time_expr = 2. * x));
}

TEST_CASE("erap diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::erap(kw::time_expr = 2. * x), x) == 0_dbl);
}

// NOTE: the functional testing for the era/erap is done in the get_era_erap_func test. Here we
// just basically checks that the cfunc compiles and that we get a non-nan value out of it.
TEST_CASE("era erap cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool compact_mode) {
        using std::isnan;

        using fp_t = decltype(fp_x);

        auto x = make_vars("x");

        std::vector<fp_t> outs, ins, pars, tm;

        for (auto batch_size : {1u}) {
            outs.resize(batch_size * 5u);
            ins.resize(batch_size);
            pars.resize(batch_size);
            tm.resize(batch_size);

            std::ranges::fill(ins, fp_t(0));
            std::ranges::fill(pars, fp_t(0));
            std::ranges::fill(tm, fp_t(0));

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc",
                            {model::era(kw::time_expr = x), model::era(kw::time_expr = par[0]),
                             model::erap(kw::time_expr = expression{fp_t(0.)}), model::erap(),
                             model::erap(kw::time_expr = x)},
                            {x}, kw::batch_size = batch_size, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.era_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.erap_"));
            }

            s.compile();

            if (opt_level == 3u) {
                // NOTE: in the compiled function, we are evaluating both era(x) and erap(x). The purpose
                // of this check is to make sure that the computation of era/erap is done with a single call
                // to the "get_era_erap()" function. That is, we want to make sure that LLVM understood it does not
                // need to call the same function twice.
                const auto get_era_erap_call_regex = std::regex(R"(.*call.*heyoka\.get_era_erap\..*)");
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
                                         matches, get_era_erap_call_regex)) {
                        ++count;
                    }
                }
                REQUIRE(count == 4u);
            }

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), tm.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(!isnan(outs[i]));
                REQUIRE(!isnan(outs[i + batch_size]));
                REQUIRE(!isnan(outs[i + 2u * batch_size]));
                REQUIRE(!isnan(outs[i + 3u * batch_size]));
                REQUIRE(!isnan(outs[i + 4u * batch_size]));

                REQUIRE(outs[i] == outs[i + batch_size]);
                REQUIRE(outs[i + 2u * batch_size] == outs[i + 3u * batch_size]);
                REQUIRE(outs[i + 2u * batch_size] == outs[i + 4u * batch_size]);
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
TEST_CASE("era erap cfunc_mp")
{
    auto x = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc",
                                  {model::era(kw::time_expr = x), model::era(kw::time_expr = par[0]),
                                   model::erap(kw::time_expr = expression{0.}), model::erap()},
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
            REQUIRE(outs[i + 2u] == outs[i + 3u]);
        }
    }
}

#endif

TEST_CASE("taylor era_erap")
{
    using model::era;
    using model::erap;

    auto x = "x"_var, y = "y"_var;

    // NOTE: use as base time coordinate 6 hours after J2000.0.
    const auto tm_coord = 0.25 / 36525;

    const auto dyn = {prime(x) = era(kw::time_expr = 2. * y) + erap(kw::time_expr = 3. * y) * y,
                      prime(y) = era(kw::time_expr = 4. * x) + erap(kw::time_expr = 5. * par[0]) * x};

    auto scalar_tester = [&dyn, tm_coord, x, y](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        // Convert tm_coord to fp_t.
        const auto tm = static_cast<fp_t>(tm_coord);

        // Create a compiled function for the evaluation of era/erap.
        auto cf = cfunc<fp_t>{{era(kw::time_expr = x), erap(kw::time_expr = y)}, {x, y}};

        // Buffer for the output of cf.
        // NOTE: size 4 because we need 2 evals for era and 2 evals for erap.
        std::vector<fp_t> cf_in(4u), cf_out(4u);
        typename cfunc<fp_t>::out_2d out_span(cf_out.data(), 2, 2);
        typename cfunc<fp_t>::out_2d in_span(cf_in.data(), 2, 2);
        typename cfunc<fp_t>::in_2d in_span_ro(cf_in.data(), 2, 2);

        const std::vector<fp_t> pars = {tm};

        auto ta = taylor_adaptive<fp_t>{
            dyn, {tm, -tm}, kw::tol = .1, kw::compact_mode = compact_mode, kw::opt_level = opt_level, kw::pars = pars};

        ta.step(true);

        const auto jet = tc_to_jet(ta);

        REQUIRE(jet[0] == tm);
        REQUIRE(jet[1] == -tm);

        in_span(0, 0) = 2 * jet[1];
        in_span(0, 1) = 4 * jet[0];
        in_span(1, 0) = 3 * jet[1];
        in_span(1, 1) = 5 * pars[0];
        cf(out_span, in_span_ro);

        REQUIRE(jet[2] == approximately(out_span(0, 0) + out_span(1, 0) * jet[1]));
        REQUIRE(jet[3] == approximately(out_span(0, 1) + out_span(1, 1) * jet[0]));

        // NOTE: at orders higher than 1 we have only erap evaluations.
        in_span(1, 0) = 2 * jet[1];
        in_span(1, 1) = 3 * jet[1];
        cf(out_span, in_span_ro);
        REQUIRE(jet[4] == approximately((out_span(1, 0) * 2 * jet[3] + out_span(1, 1) * jet[3]) / 2));

        in_span(1, 0) = 4 * jet[0];
        in_span(1, 1) = 5 * pars[0];
        cf(out_span, in_span_ro);
        REQUIRE(jet[5] == approximately((out_span(1, 0) * 4 * jet[2] + out_span(1, 1) * jet[2]) / 2));

        in_span(1, 0) = 2 * jet[1];
        in_span(1, 1) = 3 * jet[1];
        cf(out_span, in_span_ro);
        REQUIRE(jet[6] == approximately((out_span(1, 0) * 2 * 2 * jet[5] + out_span(1, 1) * 2 * jet[5]) / 6));

        in_span(1, 0) = 4 * jet[0];
        in_span(1, 1) = 5 * pars[0];
        cf(out_span, in_span_ro);
        REQUIRE(jet[7] == approximately((out_span(1, 0) * 4 * 2 * jet[4] + out_span(1, 1) * 2 * jet[4]) / 6));
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 0, cm); });
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 3, cm); });
    }
}

// Test expression simplification with era/erap.
TEST_CASE("taylor era_erap cse")
{
    using model::era;
    using model::erap;

    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{{prime(x) = (era(kw::time_expr = x) + erap(kw::time_expr = y))
                                                  + (era(kw::time_expr = y) + erap(kw::time_expr = x)),
                                       prime(y) = (era(kw::time_expr = x) + erap(kw::time_expr = y))
                                                  + (era(kw::time_expr = y) + erap(kw::time_expr = x))},
                                      {2., 3.},
                                      kw::opt_level = 0,
                                      kw::tol = 1.};

    REQUIRE(ta.get_decomposition().size() == 11u);
}
