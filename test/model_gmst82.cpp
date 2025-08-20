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
#include <sstream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/regex.hpp>

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
    REQUIRE(model::gmst82() == model::gmst82(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));
    REQUIRE(model::gmst82p() == model::gmst82p(kw::time_expr = heyoka::time, kw::eop_data = eop_data{}));

    REQUIRE(std::get<func>(model::gmst82().value()).get_name().starts_with("eop_gmst82_"));
    REQUIRE(std::get<func>(model::gmst82p().value()).get_name().starts_with("eop_gmst82p_"));

    auto x = make_vars("x");

    REQUIRE(model::gmst82() != model::gmst82(kw::time_expr = x, kw::eop_data = eop_data{}));
    REQUIRE(model::gmst82p() != model::gmst82p(kw::time_expr = x, kw::eop_data = eop_data{}));
}

TEST_CASE("get_gmst82_gmst82p_func")
{
    const eop_data data;

    auto tester = [&data]<typename T>() {
        using arr_t = T[2];

        // The function for the computation of gmst82/gmst82p.
        using fptr1_t = void (*)(T *, T *, const T *) noexcept;
        // The function to fetch the date/gmst82 data.
        using fptr2_t = void (*)(const T **, const arr_t **) noexcept;

        auto add_test_funcs = [&data](llvm_state &s, std::uint32_t batch_size) {
            auto &ctx = s.context();
            auto &bld = s.builder();
            auto &md = s.module();

            auto *scal_t = detail::to_external_llvm_type<T>(ctx);
            auto *ptr_t = llvm::PointerType::getUnqual(ctx);

            // Add the function for the computation of gmst82/gmst82p.
            auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

            auto *gmst82_ptr = f->getArg(0);
            auto *gmst82p_ptr = f->getArg(1);
            auto *time_ptr = f->getArg(2);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *tm_val = detail::ext_load_vector_from_memory(s, scal_t, time_ptr, batch_size);

            auto *gmst82_gmst82p_f = model::detail::llvm_get_gmst82_gmst82p_func(s, scal_t, batch_size, data);

            auto *gmst82_gmst82p = bld.CreateCall(gmst82_gmst82p_f, tm_val);
            auto *gmst82 = bld.CreateExtractValue(gmst82_gmst82p, 0);
            auto *gmst82p = bld.CreateExtractValue(gmst82_gmst82p, 1);

            detail::ext_store_vector_to_memory(s, gmst82_ptr, gmst82);
            detail::ext_store_vector_to_memory(s, gmst82p_ptr, gmst82p);

            bld.CreateRetVoid();

            // Add the function to fetch the date/gmst82 data.
            ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t}, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fetch_data", &md);

            auto *date_ptr_ptr = f->getArg(0);
            auto *gmst82_ptr_ptr = f->getArg(1);

            bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

            auto *date_data_ptr = detail::llvm_get_eop_sw_data_date_tt_cy_j2000(s, data, scal_t, "eop");
            auto *gmst82_data_ptr = detail::llvm_get_eop_data_gmst82(s, data, scal_t);

            bld.CreateStore(date_data_ptr, date_ptr_ptr);
            bld.CreateStore(gmst82_data_ptr, gmst82_ptr_ptr);

            bld.CreateRetVoid();
        };

        const auto data_size = data.get_table().size();

        // NOTE: the idea for testing is that we compute the gmst82 in multiprecision and compare the results.
        // Here we implement the helper for the multiprecision computation.
        auto mp_gmst82_comp = [data_size](const T *date_ptr, const arr_t *gmst82_ptr, T tm) -> std::pair<T, T> {
            using oct_t = boost::multiprecision::cpp_bin_float_oct;

            // Locate the first date *greater than* tm.
            const auto *const date_it = std::ranges::upper_bound(date_ptr, date_ptr + data_size, tm);
            if (date_it == date_ptr || date_it == date_ptr + data_size) {
                return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
            }

            // Establish the index of the time interval.
            const auto idx = (date_it - 1) - date_ptr;

            // Fetch the time and gmst82 values.
            const auto t0 = oct_t{date_ptr[idx]};
            const auto t1 = oct_t{date_ptr[idx + 1]};
            const auto [gmst820_hi, gmst820_lo] = gmst82_ptr[idx];
            const auto [gmst821_hi, gmst821_lo] = gmst82_ptr[idx + 1];
            const auto gmst820 = oct_t{gmst820_hi} + gmst820_lo;
            const auto gmst821 = oct_t{gmst821_hi} + gmst821_lo;

            // Run the linear interpolation.
            auto gmst82 = (gmst820 * (t1 - tm) + gmst821 * (tm - t0)) / (t1 - t0);
            const auto gmst82p = (gmst821 - gmst820) / (t1 - t0);

            // Reduce gmst82 modulo 2pi.
            const auto twopi = 2 * boost::math::constants::pi<oct_t>();
            gmst82 = gmst82 - twopi * floor(gmst82 / twopi);

            return {static_cast<T>(gmst82), static_cast<T>(gmst82p)};
        };

        for (const auto batch_size : {1u, 2u, 4u, 5u, 8u}) {
            // Setup the compiled functions.
            llvm_state s;
            add_test_funcs(s, batch_size);
            s.compile();
            auto *fptr1 = reinterpret_cast<fptr1_t>(s.jit_lookup("test"));
            auto *fptr2 = reinterpret_cast<fptr2_t>(s.jit_lookup("fetch_data"));

            // Fetch the date/gmst82 pointers.
            const T *date_ptr{};
            const arr_t *gmst82_ptr{};
            fptr2(&date_ptr, &gmst82_ptr);

            // Prepare the input/output vectors.
            std::vector<T> gmst82_vec(batch_size), gmst82p_vec(batch_size), tm_vec(batch_size);

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

                fptr1(gmst82_vec.data(), gmst82p_vec.data(), tm_vec.data());

                for (auto j = 0u; j < batch_size; ++j) {
                    const auto [gmst82_cmp, gmst82p_cmp] = mp_gmst82_comp(date_ptr, gmst82_ptr, tm_vec[j]);

                    if (std::isnan(gmst82_cmp)) {
                        REQUIRE(std::isnan(gmst82_vec[j]));
                    } else {
                        REQUIRE(gmst82_vec[j] == approximately(gmst82_cmp));
                    }

                    if (std::isnan(gmst82p_cmp)) {
                        REQUIRE(std::isnan(gmst82p_vec[j]));
                    } else {
                        REQUIRE(gmst82p_vec[j] == approximately(gmst82p_cmp));
                    }
                }
            }
        }
    };

    tester.operator()<float>();
    tester.operator()<double>();
}

TEST_CASE("gmst82 s11n")
{
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = model::gmst82(kw::time_expr = x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == model::gmst82(kw::time_expr = x));
}

TEST_CASE("gmst82p s11n")
{
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = model::gmst82p(kw::time_expr = x, kw::eop_data = eop_data());

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == model::gmst82p(kw::time_expr = x));
}

TEST_CASE("gmst82 diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::gmst82(kw::time_expr = 2. * x), x) == 2. * model::gmst82p(kw::time_expr = 2. * x));
}

TEST_CASE("gmst82p diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::gmst82p(kw::time_expr = 2. * x), x) == 0_dbl);
}

// NOTE: the functional testing for the gmst82/gmst82p is done in the get_gmst82_gmst82p_func test. Here we
// just basically check that the cfunc compiles and that we get a non-nan value out of it.
TEST_CASE("gmst82 gmst82p cfunc")
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
                            {model::gmst82(kw::time_expr = x), model::gmst82(kw::time_expr = par[0]),
                             model::gmst82p(kw::time_expr = expression{fp_t(0.)}), model::gmst82p(),
                             model::gmst82p(kw::time_expr = x)},
                            {x}, kw::batch_size = batch_size, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_gmst82_"));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.eop_gmst82p_"));
            }

            s.compile();

            if (opt_level == 3u) {
                // NOTE: in the compiled function, we are evaluating both gmst82(x) and gmst82p(x). The purpose
                // of this check is to make sure that the computation of gmst82/gmst82p is done with a single call
                // to the "get_gmst82_gmst82p()" function. That is, we want to make sure that LLVM understood it does
                // not need to call the same function twice.
                const auto get_gmst82_gmst82p_call_regex = boost::regex(R"(.*call.*heyoka\.get_gmst82_gmst82p\..*)");
                auto count = 0u;
                const auto ir = s.get_ir();
                for (const auto line : ir | std::ranges::views::split('\n')) {
                    boost::cmatch matches;
                    if (boost::regex_match(std::ranges::data(line), std::ranges::data(line) + std::ranges::size(line),
                                           matches, get_gmst82_gmst82p_call_regex)) {
                        ++count;
                    }
                }
                // NOTE: check <= since inlining could remove function calls altogether.
                REQUIRE(count <= 4u);
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
TEST_CASE("gmst82 gmst82p cfunc_mp")
{
    auto x = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc",
                                  {model::gmst82(kw::time_expr = x), model::gmst82(kw::time_expr = par[0]),
                                   model::gmst82p(kw::time_expr = expression{0.}), model::gmst82p()},
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

TEST_CASE("taylor scalar gmst82_gmst82p")
{
    using model::gmst82;
    using model::gmst82p;

    auto x = "x"_var, y = "y"_var;

    // NOTE: use as base time coordinate 6 hours after J2000.0.
    const auto tm_coord = 0.25 / 36525;

    const auto dyn = {prime(x) = gmst82(kw::time_expr = 2. * y) + gmst82p(kw::time_expr = 3. * y) * y
                                 + gmst82(kw::time_expr = number{tm_coord}),
                      prime(y) = gmst82(kw::time_expr = 4. * x) + gmst82p(kw::time_expr = par[0]) * x};

    auto scalar_tester = [&dyn, tm_coord, x, y](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        // Convert tm_coord to fp_t.
        const auto tm = static_cast<fp_t>(tm_coord);

        // Create compiled function wrappers for the evaluation of gmst82/gmst82p.
        auto gmst82_wrapper = [cf = cfunc<fp_t>{{gmst82(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };

        auto gmst82p_wrapper = [cf = cfunc<fp_t>{{gmst82p(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };

        const std::vector<fp_t> pars = {tm};

        auto ta = taylor_adaptive<fp_t>{
            dyn, {tm, -tm}, kw::tol = .1, kw::compact_mode = compact_mode, kw::opt_level = opt_level, kw::pars = pars};

        // NOTE: we want to check here that the gmst82/gmst82p calculation is performed no more than 5 times.
        if (opt_level == 3u) {
            const auto get_gmst82_gmst82p_call_regex = boost::regex(R"(.*call.*heyoka\.get_gmst82_gmst82p\..*)");
            auto count = 0u;

            if (compact_mode) {
                for (const auto &ir : std::get<1>(ta.get_llvm_state()).get_ir()) {
                    for (const auto line : ir | std::ranges::views::split('\n')) {
                        boost::cmatch matches;
                        if (boost::regex_match(std::ranges::data(line),
                                               std::ranges::data(line) + std::ranges::size(line), matches,
                                               get_gmst82_gmst82p_call_regex)) {
                            ++count;
                        }
                    }
                }
            } else {
                const auto ir = std::get<0>(ta.get_llvm_state()).get_ir();
                for (const auto line : ir | std::ranges::views::split('\n')) {
                    boost::cmatch matches;
                    if (boost::regex_match(std::ranges::data(line), std::ranges::data(line) + std::ranges::size(line),
                                           matches, get_gmst82_gmst82p_call_regex)) {
                        ++count;
                    }
                }
            }

            REQUIRE(count <= 5u);
        }

        ta.step(true);

        const auto jet = tc_to_jet(ta);

        REQUIRE(jet[0] == tm);
        REQUIRE(jet[1] == -tm);

        REQUIRE(
            jet[2]
            == approximately(gmst82_wrapper(2 * jet[1]) + gmst82p_wrapper(3 * jet[1]) * jet[1] + gmst82_wrapper(tm)));
        REQUIRE(jet[3] == approximately(gmst82_wrapper(4 * jet[0]) + gmst82p_wrapper(pars[0]) * jet[0]));

        REQUIRE(
            jet[4]
            == approximately((gmst82p_wrapper(2 * jet[1]) * 2 * jet[3] + gmst82p_wrapper(3 * jet[1]) * jet[3]) / 2));
        REQUIRE(jet[5]
                == approximately((gmst82p_wrapper(4 * jet[0]) * 4 * jet[2] + gmst82p_wrapper(pars[0]) * jet[2]) / 2));

        REQUIRE(jet[6]
                == approximately(
                    (gmst82p_wrapper(2 * jet[1]) * 2 * 2 * jet[5] + gmst82p_wrapper(3 * jet[1]) * 2 * jet[5]) / 6));
        REQUIRE(jet[7]
                == approximately((gmst82p_wrapper(4 * jet[0]) * 4 * 2 * jet[4] + gmst82p_wrapper(pars[0]) * 2 * jet[4])
                                 / 6));
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 0, cm); });
        tuple_for_each(fp_types, [&scalar_tester, cm](auto x) { scalar_tester(x, 3, cm); });
    }
}

TEST_CASE("taylor batch gmst82_gmst82p")
{
    using model::gmst82;
    using model::gmst82p;

    auto x = "x"_var, y = "y"_var;

    // NOTE: use as base time coordinate 6 hours after J2000.0.
    const auto tm_coord = 0.25 / 36525;

    const auto dyn = {prime(x) = gmst82(kw::time_expr = 2. * y) + gmst82p(kw::time_expr = 3. * y) * y
                                 + gmst82(kw::time_expr = number{tm_coord}),
                      prime(y) = gmst82(kw::time_expr = 4. * x) + gmst82p(kw::time_expr = par[0]) * x};

    auto batch_tester = [&dyn, tm_coord, x, y](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        // Convert tm_coord to fp_t.
        const auto tm = static_cast<fp_t>(tm_coord);

        // Create compiled function wrappers for the evaluation of gmst82/gmst82p.
        auto gmst82_wrapper = [cf = cfunc<fp_t>{{gmst82(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };

        auto gmst82p_wrapper = [cf = cfunc<fp_t>{{gmst82p(kw::time_expr = x)}, {x}}](const fp_t v) mutable {
            fp_t retval{};
            cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
            return retval;
        };

        const std::vector<fp_t> pars = {tm, tm, tm};

        auto ta = taylor_adaptive_batch<fp_t>{dyn,
                                              {tm, tm, tm, -tm, -tm, -tm},
                                              3,
                                              kw::tol = .1,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = pars};

        // NOTE: we want to check here that the gmst82/gmst82p calculation is performed no more than 5 times.
        if (opt_level == 3u) {
            const auto get_gmst82_gmst82p_call_regex = boost::regex(R"(.*call.*heyoka\.get_gmst82_gmst82p\..*)");
            auto count = 0u;

            if (compact_mode) {
                for (const auto &ir : std::get<1>(ta.get_llvm_state()).get_ir()) {
                    for (const auto line : ir | std::ranges::views::split('\n')) {
                        boost::cmatch matches;
                        if (boost::regex_match(std::ranges::data(line),
                                               std::ranges::data(line) + std::ranges::size(line), matches,
                                               get_gmst82_gmst82p_call_regex)) {
                            ++count;
                        }
                    }
                }
            } else {
                const auto ir = std::get<0>(ta.get_llvm_state()).get_ir();
                for (const auto line : ir | std::ranges::views::split('\n')) {
                    boost::cmatch matches;
                    if (boost::regex_match(std::ranges::data(line), std::ranges::data(line) + std::ranges::size(line),
                                           matches, get_gmst82_gmst82p_call_regex)) {
                        ++count;
                    }
                }
            }

            REQUIRE(count <= 5u);
        }

        ta.step(true);

        const auto jet = tc_to_jet(ta);

        REQUIRE(jet[0] == tm);
        REQUIRE(jet[1] == tm);
        REQUIRE(jet[2] == tm);
        REQUIRE(jet[3] == -tm);
        REQUIRE(jet[4] == -tm);
        REQUIRE(jet[5] == -tm);

        REQUIRE(
            jet[6]
            == approximately(gmst82_wrapper(2 * jet[3]) + gmst82p_wrapper(3 * jet[3]) * jet[3] + gmst82_wrapper(tm)));
        REQUIRE(
            jet[7]
            == approximately(gmst82_wrapper(2 * jet[3]) + gmst82p_wrapper(3 * jet[3]) * jet[3] + gmst82_wrapper(tm)));
        REQUIRE(
            jet[8]
            == approximately(gmst82_wrapper(2 * jet[3]) + gmst82p_wrapper(3 * jet[3]) * jet[3] + gmst82_wrapper(tm)));
        REQUIRE(jet[9] == approximately(gmst82_wrapper(4 * jet[0]) + gmst82p_wrapper(pars[0]) * jet[0]));
        REQUIRE(jet[10] == approximately(gmst82_wrapper(4 * jet[0]) + gmst82p_wrapper(pars[0]) * jet[0]));
        REQUIRE(jet[11] == approximately(gmst82_wrapper(4 * jet[0]) + gmst82p_wrapper(pars[0]) * jet[0]));

        REQUIRE(
            jet[12]
            == approximately((gmst82p_wrapper(2 * jet[3]) * 2 * jet[9] + gmst82p_wrapper(3 * jet[3]) * jet[9]) / 2));
        REQUIRE(
            jet[13]
            == approximately((gmst82p_wrapper(2 * jet[3]) * 2 * jet[9] + gmst82p_wrapper(3 * jet[3]) * jet[9]) / 2));
        REQUIRE(
            jet[14]
            == approximately((gmst82p_wrapper(2 * jet[3]) * 2 * jet[9] + gmst82p_wrapper(3 * jet[3]) * jet[9]) / 2));
        REQUIRE(jet[15]
                == approximately((gmst82p_wrapper(4 * jet[0]) * 4 * jet[6] + gmst82p_wrapper(pars[0]) * jet[6]) / 2));
        REQUIRE(jet[16]
                == approximately((gmst82p_wrapper(4 * jet[0]) * 4 * jet[6] + gmst82p_wrapper(pars[0]) * jet[6]) / 2));
        REQUIRE(jet[17]
                == approximately((gmst82p_wrapper(4 * jet[0]) * 4 * jet[6] + gmst82p_wrapper(pars[0]) * jet[6]) / 2));

        REQUIRE(jet[18]
                == approximately(
                    (gmst82p_wrapper(2 * jet[3]) * 2 * 2 * jet[15] + gmst82p_wrapper(3 * jet[3]) * 2 * jet[15]) / 6));
        REQUIRE(jet[19]
                == approximately(
                    (gmst82p_wrapper(2 * jet[3]) * 2 * 2 * jet[15] + gmst82p_wrapper(3 * jet[3]) * 2 * jet[15]) / 6));
        REQUIRE(jet[20]
                == approximately(
                    (gmst82p_wrapper(2 * jet[3]) * 2 * 2 * jet[15] + gmst82p_wrapper(3 * jet[3]) * 2 * jet[15]) / 6));
        REQUIRE(jet[21]
                == approximately(
                    (gmst82p_wrapper(4 * jet[0]) * 4 * 2 * jet[12] + gmst82p_wrapper(pars[0]) * 2 * jet[12]) / 6));
        REQUIRE(jet[22]
                == approximately(
                    (gmst82p_wrapper(4 * jet[0]) * 4 * 2 * jet[12] + gmst82p_wrapper(pars[0]) * 2 * jet[12]) / 6));
        REQUIRE(jet[23]
                == approximately(
                    (gmst82p_wrapper(4 * jet[0]) * 4 * 2 * jet[12] + gmst82p_wrapper(pars[0]) * 2 * jet[12]) / 6));
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&batch_tester, cm](auto x) { batch_tester(x, 0, cm); });
        tuple_for_each(fp_types, [&batch_tester, cm](auto x) { batch_tester(x, 3, cm); });
    }
}

#if defined(HEYOKA_HAVE_REAL)

// NOTE: the point of the multiprecision test is just to check we used the correct
// llvm primitives in the implementation.
TEST_CASE("taylor mp gmst82_gmst82p")
{
    using model::gmst82;
    using model::gmst82p;

    auto x = "x"_var, y = "y"_var;

    // NOTE: use as base time coordinate 6 hours after J2000.0.
    const auto tm_coord = 0.25 / 36525;

    const auto dyn = {prime(x) = gmst82(kw::time_expr = 2. * y) + gmst82p(kw::time_expr = 3. * y) * y
                                 + gmst82(kw::time_expr = number{tm_coord}),
                      prime(y) = gmst82(kw::time_expr = 4. * x) + gmst82p(kw::time_expr = par[0]) * x};

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto prec : {30, 123}) {
                // Convert tm_coord to real.
                const auto tm = mppp::real(tm_coord, prec);

                // Create compiled function wrappers for the evaluation of gmst82/gmst82p.
                auto gmst82_wrapper = [prec, cf = cfunc<mppp::real>{{gmst82(kw::time_expr = x)}, {x}, kw::prec = prec}](
                                          const mppp::real &v) mutable {
                    mppp::real retval{0, prec};
                    cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
                    return retval;
                };

                auto gmst82p_wrapper
                    = [prec, cf = cfunc<mppp::real>{{gmst82p(kw::time_expr = x)}, {x}, kw::prec = prec}](
                          const mppp::real &v) mutable {
                          mppp::real retval{0, prec};
                          cf(std::ranges::subrange(&retval, &retval + 1), std::ranges::subrange(&v, &v + 1));
                          return retval;
                      };

                const std::vector pars = {tm};

                auto ta = taylor_adaptive<mppp::real>{dyn,
                                                      {tm, -tm},
                                                      kw::tol = .1,
                                                      kw::compact_mode = cm,
                                                      kw::opt_level = opt_level,
                                                      kw::pars = pars,
                                                      kw::prec = prec};

                ta.step(true);

                const auto jet = tc_to_jet(ta);

                REQUIRE(jet[0] == tm);
                REQUIRE(jet[1] == -tm);

                REQUIRE(jet[2]
                        == approximately(gmst82_wrapper(mppp::real{2, prec} * jet[1])
                                         + gmst82p_wrapper(mppp::real{3, prec} * jet[1]) * jet[1]
                                         + gmst82_wrapper(tm)));
                REQUIRE(
                    jet[3]
                    == approximately(gmst82_wrapper(mppp::real{4, prec} * jet[0]) + gmst82p_wrapper(pars[0]) * jet[0]));

                REQUIRE(jet[4]
                        == approximately((gmst82p_wrapper(mppp::real{2, prec} * jet[1]) * mppp::real{2, prec} * jet[3]
                                          + gmst82p_wrapper(mppp::real{3, prec} * jet[1]) * jet[3])
                                         / mppp::real{2, prec}));
                REQUIRE(jet[5]
                        == approximately((gmst82p_wrapper(mppp::real{4, prec} * jet[0]) * mppp::real{4, prec} * jet[2]
                                          + gmst82p_wrapper(pars[0]) * jet[2])
                                         / mppp::real{2, prec}));

                REQUIRE(
                    jet[6]
                    == approximately((gmst82p_wrapper(mppp::real{2, prec} * jet[1]) * mppp::real{2, prec}
                                          * mppp::real{2, prec} * jet[5]
                                      + gmst82p_wrapper(mppp::real{3, prec} * jet[1]) * mppp::real{2, prec} * jet[5])
                                     / mppp::real{6, prec}));
                REQUIRE(jet[7]
                        == approximately((gmst82p_wrapper(mppp::real{4, prec} * jet[0]) * mppp::real{4, prec}
                                              * mppp::real{2, prec} * jet[4]
                                          + gmst82p_wrapper(pars[0]) * mppp::real{2, prec} * jet[4])
                                         / mppp::real{6, prec}));
            }
        }
    }
}

#endif

// Test expression simplification with gmst82/gmst82p.
TEST_CASE("taylor gmst82_gmst82p cse")
{
    using model::gmst82;
    using model::gmst82p;

    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{{prime(x) = (gmst82(kw::time_expr = x) + gmst82p(kw::time_expr = y))
                                                  + (gmst82(kw::time_expr = y) + gmst82p(kw::time_expr = x)),
                                       prime(y) = (gmst82(kw::time_expr = x) + gmst82p(kw::time_expr = y))
                                                  + (gmst82(kw::time_expr = y) + gmst82p(kw::time_expr = x))},
                                      {2., 3.},
                                      kw::opt_level = 0,
                                      kw::tol = 1.};

    REQUIRE(ta.get_decomposition().size() == 11u);
}
