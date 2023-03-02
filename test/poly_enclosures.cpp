// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

static const int ntrials = 100;

using namespace heyoka;
using namespace heyoka_test;

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
                auto *h_lo_val = detail::load_vector_from_memory(builder, val_t, h_lo_ptr, batch_size);
                auto *h_hi_val = detail::load_vector_from_memory(builder, val_t, h_hi_ptr, batch_size);

                {
                    auto [res_lo, res_hi]
                        = detail::llvm_penc_interval(s, val_t, cf_ptr, order, h_lo_val, h_hi_val, batch_size);

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
                auto *h_val = detail::load_vector_from_memory(builder, val_t, h_ptr, batch_size);

                {
                    auto [res_lo, res_hi] = detail::llvm_penc_cargo_shisha(s, val_t, cf_ptr, order, h_val, batch_size);

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
                        cf = static_cast<fp_t>(rdist(rng));
                    }

                    // Generate the h values.
                    for (auto i = 0u; i < batch_size; ++i) {
                        const auto tmp = static_cast<fp_t>(rdist(rng));

                        h[i] = tmp;
                        h_lo[i] = tmp >= 0 ? 0 : tmp;
                        h_hi[i] = tmp >= 0 ? tmp : 0;
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

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("polynomial enclosures mp")
{
    using fp_t = mppp::real;

    std::uniform_real_distribution<double> rdist(-10., 10.);

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            std::vector<fp_t> poly, h, h_lo, h_hi, res_lo1, res_hi1, res_lo2, res_hi2;

            for (auto order : {1u, 2u, 13u, 20u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto *val_t = detail::llvm_type_like(s, fp_t{0, prec});
                auto *ext_val_t = detail::llvm_ext_type(val_t);
                auto *ext_ptr_val_t = llvm::PointerType::getUnqual(ext_val_t);

                // Fetch the current insertion block.
                auto *orig_bb = builder.GetInsertBlock();

                // Add the interval-arithmetic function.
                auto *ft
                    = llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type *>(5u, ext_ptr_val_t), false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "penc_interval", &md);

                auto *out_lo_ptr = f->args().begin();
                auto *out_hi_ptr = f->args().begin() + 1;
                auto *cf_ptr = f->args().begin() + 2;
                auto *h_lo_ptr = f->args().begin() + 3;
                auto *h_hi_ptr = f->args().begin() + 4;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                // Load the h values.
                auto *h_lo_val = detail::ext_load_vector_from_memory(s, val_t, h_lo_ptr, 1);
                auto *h_hi_val = detail::ext_load_vector_from_memory(s, val_t, h_hi_ptr, 1);

                {
                    auto [res_lo, res_hi] = detail::llvm_penc_interval(s, val_t, cf_ptr, order, h_lo_val, h_hi_val, 1);

                    // Store the result.
                    detail::ext_store_vector_to_memory(s, out_lo_ptr, res_lo);
                    detail::ext_store_vector_to_memory(s, out_hi_ptr, res_hi);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Restore the original insertion block.
                builder.SetInsertPoint(orig_bb);

                // Add the Cargo-Shisha function.
                ft = llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type *>(4u, ext_ptr_val_t), false);
                f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "penc_cargo_shisha", &md);

                out_lo_ptr = f->args().begin();
                out_hi_ptr = f->args().begin() + 1;
                cf_ptr = f->args().begin() + 2;
                auto *h_ptr = f->args().begin() + 3;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                // Load the h values.
                auto *h_val = detail::ext_load_vector_from_memory(s, val_t, h_ptr, 1);

                {
                    auto [res_lo, res_hi] = detail::llvm_penc_cargo_shisha(s, val_t, cf_ptr, order, h_val, 1);

                    // Store the result.
                    detail::ext_store_vector_to_memory(s, out_lo_ptr, res_lo);
                    detail::ext_store_vector_to_memory(s, out_hi_ptr, res_hi);
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
                poly.resize(order + 1u, fp_t{0, prec});
                h.resize(1, fp_t{0, prec});
                h_lo.resize(1, fp_t{0, prec});
                h_hi.resize(1, fp_t{0, prec});
                res_lo1.resize(1, fp_t{0, prec});
                res_hi1.resize(1, fp_t{0, prec});
                res_lo2.resize(1, fp_t{0, prec});
                res_hi2.resize(1, fp_t{0, prec});

                for (auto _ = 0; _ < ntrials; ++_) {
                    // Generate the polynomial.
                    for (auto &cf : poly) {
                        mppp::set(cf, rdist(rng));
                    }

                    // Generate the h values.
                    const auto tmp = rdist(rng);

                    mppp::set(h[0], tmp);
                    mppp::set(h_lo[0], tmp >= 0 ? 0 : tmp);
                    mppp::set(h_hi[0], tmp >= 0 ? tmp : 0);

                    penc_int_f(res_lo1.data(), res_hi1.data(), poly.data(), h_lo.data(), h_hi.data());
                    penc_cs_f(res_lo2.data(), res_hi2.data(), poly.data(), h.data());

                    // Test that the intervals are sane.
                    REQUIRE(res_hi1[0] >= res_lo1[0]);
                    REQUIRE(res_hi2[0] >= res_lo2[0]);

                    // Test that the intervals overlap.
                    REQUIRE((res_lo1[0] <= res_hi2[0] && res_hi1[0] >= res_lo2[0]));
                }
            }
        }
    }
}

#endif
