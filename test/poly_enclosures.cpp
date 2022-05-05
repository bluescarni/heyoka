// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <initializer_list>
#include <random>
#include <tuple>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

static const int ntrials = 100;

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

TEST_CASE("polynomial enclosures")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using fp_t = decltype(fp_x);

        std::uniform_real_distribution<double> rdist(-10., 10.);
        std::vector<fp_t> poly, h, h_lo, h_hi, res_lo1, res_hi1, res_lo2, res_hi2;

        for (auto batch_size : {1u, 2u, 4u}) {
            for (auto order : {1u, 2u, 13u, 20u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = detail::to_llvm_type<fp_t>(context);
                auto ptr_val_t = llvm::PointerType::getUnqual(val_t);

                // Fetch the current insertion block.
                auto *orig_bb = builder.GetInsertBlock();

                // Add the interval-arithmetic function.
                auto *ft
                    = llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type *>(5u, ptr_val_t), false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "penc_interval", &md);

                auto *out_lo_ptr = f->args().begin();
                auto *out_hi_ptr = f->args().begin() + 1;
                auto *cf_ptr = f->args().begin() + 2;
                auto *h_lo_ptr = f->args().begin() + 3;
                auto *h_hi_ptr = f->args().begin() + 4;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                // Load the h values.
                auto *h_lo_val = detail::load_vector_from_memory(builder, h_lo_ptr, batch_size);
                auto *h_hi_val = detail::load_vector_from_memory(builder, h_hi_ptr, batch_size);

                {
                    auto [res_lo, res_hi]
                        = detail::llvm_penc_interval<fp_t>(s, cf_ptr, order, h_lo_val, h_hi_val, batch_size);

                    // Store the result.
                    detail::store_vector_to_memory(builder, out_lo_ptr, res_lo);
                    detail::store_vector_to_memory(builder, out_hi_ptr, res_hi);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Restore the original insertion block.
                builder.SetInsertPoint(orig_bb);

                // Add the Cargo-Shisha function.
                ft = llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type *>(4u, ptr_val_t), false);
                f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "penc_cargo_shisha", &md);

                out_lo_ptr = f->args().begin();
                out_hi_ptr = f->args().begin() + 1;
                cf_ptr = f->args().begin() + 2;
                auto *h_ptr = f->args().begin() + 3;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                // Load the h values.
                auto *h_val = detail::load_vector_from_memory(builder, h_ptr, batch_size);

                {
                    auto [res_lo, res_hi] = detail::llvm_penc_cargo_shisha<fp_t>(s, cf_ptr, order, h_val, batch_size);

                    // Store the result.
                    detail::store_vector_to_memory(builder, out_lo_ptr, res_lo);
                    detail::store_vector_to_memory(builder, out_hi_ptr, res_hi);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the functions.
                auto *penc_int_f = reinterpret_cast<void (*)(fp_t *, fp_t *, const fp_t *, const fp_t *, const fp_t *)>(
                    s.jit_lookup("penc_interval"));
                auto *penc_cs_f = reinterpret_cast<void (*)(fp_t *, fp_t *, const fp_t *, const fp_t *)>(
                    s.jit_lookup("penc_cargo_shisha"));

                // Prepare the buffers.
                poly.resize((order + 1u) * batch_size);
                h.resize(batch_size);
                h_lo.resize(batch_size);
                h_hi.resize(batch_size);
                res_lo1.resize(batch_size);
                res_hi1.resize(batch_size);
                res_lo2.resize(batch_size);
                res_hi2.resize(batch_size);

                for (auto _ = 0; _ < ntrials; ++_) {
                    // Generate the polynomial.
                    for (auto &cf : poly) {
                        cf = rdist(rng);
                    }

                    // Generate the h values.
                    for (auto i = 0u; i < batch_size; ++i) {
                        const auto tmp = rdist(rng);

                        h[i] = tmp;
                        h_lo[i] = tmp >= 0 ? 0. : tmp;
                        h_hi[i] = tmp >= 0 ? tmp : 0.;
                    }

                    penc_int_f(res_lo1.data(), res_hi1.data(), poly.data(), h_lo.data(), h_hi.data());
                    penc_cs_f(res_lo2.data(), res_hi2.data(), poly.data(), h.data());

                    for (auto i = 0u; i < batch_size; ++i) {
                        // Test that the intervals are sane.
                        REQUIRE(res_hi1[i] >= res_lo1[i]);
                        REQUIRE(res_hi2[i] >= res_lo2[i]);

                        // Test that the intervals overlap.
                        REQUIRE((res_lo1[i] <= res_hi2[i] && res_hi1[i] >= res_lo2[i]));
                    }
                }
            }
        }
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        tuple_for_each(fp_types, [&tester, opt_level](auto x) { tester(x, opt_level); });
    }
}
