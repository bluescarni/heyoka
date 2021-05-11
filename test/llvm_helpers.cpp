// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/roots.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

std::mt19937 rng;

TEST_CASE("sincos scalar")
{
    using detail::llvm_sincos;
    using detail::to_llvm_type;
    using std::cos;
    using std::sin;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{val_t, llvm::PointerType::getUnqual(val_t),
                                            llvm::PointerType::getUnqual(val_t)};
            auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "sc", &md);

            auto x = f->args().begin();
            auto sptr = f->args().begin() + 1;
            auto cptr = f->args().begin() + 2;

            builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

            auto ret = llvm_sincos(s, x);
            builder.CreateStore(ret.first, sptr);
            builder.CreateStore(ret.second, cptr);

            // Create the return value.
            builder.CreateRetVoid();

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<void (*)(fp_t, fp_t *, fp_t *)>(s.jit_lookup("sc"));

            fp_t sn, cs;
            f_ptr(fp_t(2), &sn, &cs);
            REQUIRE(sn == approximately(sin(fp_t(2))));
            REQUIRE(cs == approximately(cos(fp_t(2))));

            f_ptr(fp_t(-123.45), &sn, &cs);
            REQUIRE(sn == approximately(sin(fp_t(-123.45))));
            REQUIRE(cs == approximately(cos(fp_t(-123.45))));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("sincos batch")
{
    using detail::llvm_sincos;
    using detail::to_llvm_type;
    using std::cos;
    using std::sin;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(val_t),
                                                llvm::PointerType::getUnqual(val_t),
                                                llvm::PointerType::getUnqual(val_t)};
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "sc", &md);

                auto xptr = f->args().begin();
                auto sptr = f->args().begin() + 1;
                auto cptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto x = detail::load_vector_from_memory(builder, xptr, batch_size);

                auto ret = llvm_sincos(s, x);
                detail::store_vector_to_memory(builder, sptr, ret.first);
                detail::store_vector_to_memory(builder, cptr, ret.second);

                // Create the return value.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *)>(s.jit_lookup("sc"));

                // Setup the argument and the output values.
                std::vector<fp_t> x_vec(batch_size), s_vec(x_vec), c_vec(x_vec);
                for (auto i = 0u; i < batch_size; ++i) {
                    x_vec[i] = i + 1u;
                }

                f_ptr(x_vec.data(), s_vec.data(), c_vec.data());

                for (auto i = 0u; i < batch_size; ++i) {
                    REQUIRE(s_vec[i] == approximately(sin(x_vec[i])));
                    REQUIRE(c_vec[i] == approximately(cos(x_vec[i])));
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("modulus scalar")
{
    using detail::llvm_modulus;
    using detail::to_llvm_type;
    using std::floor;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{val_t, val_t};
            auto *ft = llvm::FunctionType::get(val_t, fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_rem", &md);

            auto x = f->args().begin();
            auto y = f->args().begin() + 1;

            builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

            builder.CreateRet(llvm_modulus(s, x, y));

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<fp_t (*)(fp_t, fp_t)>(s.jit_lookup("hey_rem"));

            auto a = fp_t(123);
            auto b = fp_t(2) / fp_t(7);

            REQUIRE(f_ptr(a, b) == approximately(a - b * floor(a / b), fp_t(1000)));

            a = fp_t(-4);
            b = fp_t(314) / fp_t(100);

            REQUIRE(f_ptr(a, b) == approximately(a - b * floor(a / b), fp_t(1000)));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("modulus batch")
{
    using detail::llvm_modulus;
    using detail::to_llvm_type;
    using std::floor;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(val_t),
                                                llvm::PointerType::getUnqual(val_t),
                                                llvm::PointerType::getUnqual(val_t)};
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_rem", &md);

                auto ret_ptr = f->args().begin();
                auto x_ptr = f->args().begin() + 1;
                auto y_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto ret = llvm_modulus(s, detail::load_vector_from_memory(builder, x_ptr, batch_size),
                                        detail::load_vector_from_memory(builder, y_ptr, batch_size));

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *)>(s.jit_lookup("hey_rem"));

                // Setup the arguments and the output value.
                std::vector<fp_t> ret_vec(batch_size), a_vec(ret_vec), b_vec(ret_vec);
                for (auto i = 0u; i < batch_size; ++i) {
                    a_vec[i] = i + 1u;
                    b_vec[i] = a_vec[i] * 10 * (i + 1u);
                }

                f_ptr(ret_vec.data(), a_vec.data(), b_vec.data());

                for (auto i = 0u; i < batch_size; ++i) {
                    auto a = a_vec[i];
                    auto b = b_vec[i];

                    REQUIRE(ret_vec[i] == approximately(a - b * floor(a / b), fp_t(1000)));
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("inv_kep_scalar")
{
    using detail::llvm_add_inv_kep;
    using detail::to_llvm_type;
    namespace bmt = boost::math::tools;
    using std::cos;
    using std::sin;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto fkep = llvm_add_inv_kep<fp_t>(s, 1);

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            {
                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs{val_t, val_t};
                auto *ft = llvm::FunctionType::get(val_t, fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_kep", &md);

                auto e = f->args().begin();
                auto M = f->args().begin() + 1;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                builder.CreateRet(builder.CreateCall(fkep, {e, M}));

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();
            }

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<fp_t (*)(fp_t, fp_t)>(s.jit_lookup("hey_kep"));

            auto bmt_inv_kep = [](fp_t ecc, fp_t M) {
                // Initial guess.
                auto ig = ecc < 0.8 ? M : boost::math::constants::pi<double>();

                auto func = [ecc, M](fp_t E) { return std::make_pair(E - ecc * sin(E) - M, 1 - ecc * cos(E)); };

                return bmt::newton_raphson_iterate(func, ig, fp_t(0), fp_t(2 * boost::math::constants::pi<double>()),
                                                   std::numeric_limits<fp_t>::digits - 2);
            };

            std::uniform_real_distribution<double> e_dist(0., 1.), M_dist(0., 2 * boost::math::constants::pi<double>());

            const auto ntrials = 100;

            // First set of tests with zero eccentricity.
            for (auto i = 0; i < ntrials; ++i) {
                const auto M = M_dist(rng);
                REQUIRE(f_ptr(0, M) == approximately(bmt_inv_kep(0, M)));
            }

            // Non-zero eccentricities.
            for (auto i = 0; i < ntrials * 10; ++i) {
                const auto M = M_dist(rng);
                const auto e = e_dist(rng);
                REQUIRE(f_ptr(e, M) == approximately(bmt_inv_kep(e, M), fp_t(10000)));
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("inv_kep_batch")
{
    using detail::llvm_add_inv_kep;
    using detail::to_llvm_type;
    namespace bmt = boost::math::tools;
    using std::cos;
    using std::sin;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto fkep = llvm_add_inv_kep<fp_t>(s, batch_size);

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(val_t),
                                                llvm::PointerType::getUnqual(val_t),
                                                llvm::PointerType::getUnqual(val_t)};
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_kep", &md);

                auto ret_ptr = f->args().begin();
                auto e_ptr = f->args().begin() + 1;
                auto M_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto ret = builder.CreateCall(fkep, {detail::load_vector_from_memory(builder, e_ptr, batch_size),
                                                     detail::load_vector_from_memory(builder, M_ptr, batch_size)});

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *)>(s.jit_lookup("hey_kep"));

                auto bmt_inv_kep = [](fp_t ecc, fp_t M) {
                    // Initial guess.
                    auto ig = ecc < 0.8 ? M : boost::math::constants::pi<double>();

                    auto func = [ecc, M](fp_t E) { return std::make_pair(E - ecc * sin(E) - M, 1 - ecc * cos(E)); };

                    return bmt::newton_raphson_iterate(func, ig, fp_t(0),
                                                       fp_t(2 * boost::math::constants::pi<double>()),
                                                       std::numeric_limits<fp_t>::digits - 2);
                };

                std::uniform_real_distribution<double> e_dist(0., 1.),
                    M_dist(0., 2 * boost::math::constants::pi<double>());

                const auto ntrials = 100;

                std::vector<fp_t> ret_vec(batch_size), e_vec(ret_vec), M_vec(ret_vec);

                // First set of tests with zero eccentricity.
                for (auto i = 0; i < ntrials; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);
                    }
                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        REQUIRE(ret_vec[j] == approximately(bmt_inv_kep(0, M_vec[j])));
                    }
                }

                // Non-zero eccentricities.
                for (auto i = 0; i < ntrials * 10; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);
                        e_vec[j] = e_dist(rng);
                    }
                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        REQUIRE(ret_vec[j] == approximately(bmt_inv_kep(e_vec[j], M_vec[j]), fp_t(10000)));
                    }
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("while_loop")
{
    using detail::llvm_while_loop;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state s{kw::opt_level = opt_level};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto val_t = builder.getInt32Ty();

        std::vector<llvm::Type *> fargs{val_t};
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "count_n", &md);

        auto final_n = f->args().begin();

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto retval = builder.CreateAlloca(val_t);
        builder.CreateStore(builder.getInt32(0), retval);

        llvm_while_loop(
            s, [&]() -> llvm::Value * { return builder.CreateICmpULT(builder.CreateLoad(retval), final_n); },
            [&]() { builder.CreateStore(builder.CreateAdd(builder.CreateLoad(retval), builder.getInt32(1)), retval); });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

        // Verify.
        s.verify_function(f);

        // Run the optimisation pass.
        s.optimise();

        // Compile.
        s.compile();

        // Fetch the function pointer.
        auto f_ptr = reinterpret_cast<std::uint32_t (*)(std::uint32_t)>(s.jit_lookup("count_n"));

        REQUIRE(f_ptr(0) == 0u);
        REQUIRE(f_ptr(1) == 1u);
        REQUIRE(f_ptr(2) == 2u);
        REQUIRE(f_ptr(3) == 3u);
        REQUIRE(f_ptr(4) == 4u);
    }

    // Error handling.
    {
        llvm_state s;

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto val_t = builder.getInt32Ty();

        std::vector<llvm::Type *> fargs{val_t};
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "count_n", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto retval = builder.CreateAlloca(val_t);
        builder.CreateStore(builder.getInt32(0), retval);

        // NOTE: if we don't do the cleanup of f before re-throwing,
        // on the OSX CI the test will hang on shutdown (i.e., all the tests
        // run correctly but the test program hangs on exit). Not sure what is
        // going on with that, perhaps another bad interaction between LLVM and
        // exceptions?
        auto thrower = [&]() {
            try {
                llvm_while_loop(
                    s, [&]() -> llvm::Value * { throw std::runtime_error{"aa"}; }, [&]() {});
            } catch (...) {
                f->eraseFromParent();

                throw;
            }
        };

        REQUIRE_THROWS_AS(thrower(), std::runtime_error);
    }

    {
        llvm_state s;

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto val_t = builder.getInt32Ty();

        std::vector<llvm::Type *> fargs{val_t};
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "count_n", &md);

        auto final_n = f->args().begin();

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto retval = builder.CreateAlloca(val_t);
        builder.CreateStore(builder.getInt32(0), retval);

        auto thrower = [&]() {
            try {
                llvm_while_loop(
                    s, [&]() -> llvm::Value * { return builder.CreateICmpULT(builder.CreateLoad(retval), final_n); },
                    [&]() { throw std::runtime_error{"aa"}; });
            } catch (...) {
                f->eraseFromParent();

                throw;
            }
        };

        REQUIRE_THROWS_AS(thrower(), std::runtime_error);
    }
}
