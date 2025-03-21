// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/era.hpp>
#include <heyoka/s11n.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

static std::mt19937 rng;

constexpr auto ntrials = 10000;

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

TEST_CASE("era diff")
{
    auto x = make_vars("x");

    REQUIRE(diff(model::era(kw::time_expr = 2. * x), x) == 2. * model::erap(kw::time_expr = 2. * x));
}
