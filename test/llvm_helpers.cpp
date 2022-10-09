// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <fmt/format.h>

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

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

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

std::mt19937 rng;

constexpr auto ntrials = 100;

TEST_CASE("sgn scalar")
{
    using detail::llvm_sgn;
    using detail::to_llvm_type;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            auto *ft = llvm::FunctionType::get(builder.getInt32Ty(), {val_t}, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "sgn", &md);

            auto x = f->args().begin();

            builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

            // Create the return value.
            builder.CreateRet(llvm_sgn(s, x));

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<std::int32_t (*)(fp_t)>(s.jit_lookup("sgn"));

            REQUIRE(f_ptr(0) == 0);
            REQUIRE(f_ptr(-42) == -1);
            REQUIRE(f_ptr(123) == 1);
        }
    };

    tuple_for_each(fp_types, tester);
}

// Generic branchless sign function.
template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

TEST_CASE("sgn batch")
{
    using detail::llvm_sgn;
    using detail::to_llvm_type;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                auto *ft = llvm::FunctionType::get(
                    builder.getVoidTy(),
                    {llvm::PointerType::getUnqual(builder.getInt32Ty()), llvm::PointerType::getUnqual(val_t)}, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "sgn", &md);

                auto out = f->args().begin();
                auto x = f->args().begin() + 1;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                // Load the vector from memory.
                auto v = detail::load_vector_from_memory(builder, val_t, x, batch_size);

                // Create and store the return value.
                detail::store_vector_to_memory(builder, out, llvm_sgn(s, v));

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(std::int32_t *, const fp_t *)>(s.jit_lookup("sgn"));

                std::uniform_real_distribution<double> rdist(-10., 10.);
                std::vector<fp_t> values(batch_size);
                std::generate(values.begin(), values.end(), [&rdist]() { return rdist(rng); });
                std::vector<std::int32_t> signs(batch_size);

                f_ptr(signs.data(), values.data());

                for (auto i = 0u; i < batch_size; ++i) {
                    REQUIRE(signs[i] == sgn(values[i]));
                }

                values[0] = 0;

                f_ptr(signs.data(), values.data());
                REQUIRE(signs[0] == 0);
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

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

                auto *xptr = f->args().begin();
                auto *sptr = f->args().begin() + 1;
                auto *cptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto *x = detail::load_vector_from_memory(builder, val_t, xptr, batch_size);

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

                auto *ret_ptr = f->args().begin();
                auto *x_ptr = f->args().begin() + 1;
                auto *y_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto *ret = llvm_modulus(s, detail::load_vector_from_memory(builder, val_t, x_ptr, batch_size),
                                         detail::load_vector_from_memory(builder, val_t, y_ptr, batch_size));

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

TEST_CASE("inv_kep_E_scalar")
{
    using detail::llvm_add_inv_kep_E_wrapper;
    namespace bmt = boost::math::tools;
    using std::cos;
    using std::isnan;
    using std::sin;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            // Add the function.
            llvm_add_inv_kep_E_wrapper<fp_t>(s, 1, "hey_kep");

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("hey_kep"));

            std::uniform_real_distribution<double> e_dist(0., 1.), M_dist(0., 2 * boost::math::constants::pi<double>());

            // First set of tests with zero eccentricity.
            for (auto i = 0; i < ntrials; ++i) {
                const fp_t M = M_dist(rng);
                const fp_t e = 0;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(fp_t(M) == approximately(E));
            }

            // Non-zero eccentricities.
            for (auto i = 0; i < ntrials * 10; ++i) {
                const fp_t M = M_dist(rng);
                const fp_t e = e_dist(rng);
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(fp_t(M) == approximately(E - e * sin(E), fp_t(10000)));
            }

            // Try a very high eccentricity.
            {
                fp_t M = 0;
                fp_t e = 1 - std::numeric_limits<fp_t>::epsilon() * 4;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(M == approximately(E - e * sin(E), fp_t(10000)));
            }

            // Test invalid inputs.
            {
                fp_t M = 1.23;
                fp_t e = -.1;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = 1.23;
                fp_t e = 1.;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = 1.23;
                fp_t e = std::numeric_limits<fp_t>::infinity();
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = 1.23;
                fp_t e = -std::numeric_limits<fp_t>::infinity();
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = 1.23;
                fp_t e = std::numeric_limits<fp_t>::quiet_NaN();
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = std::numeric_limits<fp_t>::infinity();
                fp_t e = .1;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = -std::numeric_limits<fp_t>::infinity();
                fp_t e = .2;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }

            {
                fp_t M = std::numeric_limits<fp_t>::quiet_NaN();
                fp_t e = .1;
                fp_t E;

                f_ptr(&E, &e, &M);

                REQUIRE(isnan(E));
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("inv_kep_E_batch")
{
    using detail::llvm_add_inv_kep_E_wrapper;
    namespace bmt = boost::math::tools;
    using std::cos;
    using std::isnan;
    using std::sin;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                // Add the function.
                llvm_add_inv_kep_E_wrapper<fp_t>(s, batch_size, "hey_kep");

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("hey_kep"));

                std::uniform_real_distribution<double> e_dist(0., 1.),
                    M_dist(0., 2 * boost::math::constants::pi<double>());

                std::vector<fp_t> ret_vec(batch_size), e_vec(ret_vec), M_vec(ret_vec);

                // First set of tests with zero eccentricity.
                for (auto i = 0; i < ntrials; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);
                    }
                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        REQUIRE(M_vec[j] == approximately(ret_vec[j]));
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
                        REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                    }
                }

                // Test invalid inputs.
                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);

                        if (j == 1u) {
                            e_vec[j] = -.1;
                        } else {
                            e_vec[j] = e_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);

                        if (j == 1u) {
                            e_vec[j] = 1;
                        } else {
                            e_vec[j] = e_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);

                        if (j == 1u) {
                            e_vec[j] = std::numeric_limits<fp_t>::infinity();
                        } else {
                            e_vec[j] = e_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);

                        if (j == 1u) {
                            e_vec[j] = -std::numeric_limits<fp_t>::infinity();
                        } else {
                            e_vec[j] = e_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        M_vec[j] = M_dist(rng);

                        if (j == 1u) {
                            e_vec[j] = std::numeric_limits<fp_t>::quiet_NaN();
                        } else {
                            e_vec[j] = e_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        e_vec[j] = e_dist(rng);

                        if (j == 1u) {
                            M_vec[j] = std::numeric_limits<fp_t>::infinity();
                        } else {
                            M_vec[j] = M_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        e_vec[j] = e_dist(rng);

                        if (j == 1u) {
                            M_vec[j] = -std::numeric_limits<fp_t>::infinity();
                        } else {
                            M_vec[j] = M_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
                    }
                }

                {
                    for (auto j = 0u; j < batch_size; ++j) {
                        e_vec[j] = e_dist(rng);

                        if (j == 1u) {
                            M_vec[j] = std::numeric_limits<fp_t>::quiet_NaN();
                        } else {
                            M_vec[j] = M_dist(rng);
                        }
                    }

                    f_ptr(ret_vec.data(), e_vec.data(), M_vec.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (j == 1u) {
                            REQUIRE(isnan(ret_vec[j]));
                        } else {
                            REQUIRE(M_vec[j] == approximately(ret_vec[j] - e_vec[j] * sin(ret_vec[j]), fp_t(10000)));
                        }
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
            s, [&]() -> llvm::Value * { return builder.CreateICmpULT(builder.CreateLoad(val_t, retval), final_n); },
            [&]() {
                builder.CreateStore(builder.CreateAdd(builder.CreateLoad(val_t, retval), builder.getInt32(1)), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

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

    // NOTE: don't run the error handling test on OSX, as
    // we occasionally experience hangs/errors when
    // catching and re-throwing exceptions. Not sure whether
    // this is an LLVM issue or some compiler/toolchain bug.
    // Perhaps re-check this with later LLVM versions, different
    // build types (e.g., Release) or different compiler flags.
#if !defined(__APPLE__)

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
                    s,
                    [&]() -> llvm::Value * {
                        return builder.CreateICmpULT(builder.CreateLoad(val_t, retval), final_n);
                    },
                    [&]() { throw std::runtime_error{"aa"}; });
            } catch (...) {
                f->eraseFromParent();

                throw;
            }
        };

        REQUIRE_THROWS_AS(thrower(), std::runtime_error);
    }

#endif
}

TEST_CASE("csc_scalar")
{
    using detail::llvm_add_csc;
    using detail::llvm_mangle_type;
    using detail::to_llvm_type;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            const auto degree = 4u;

            llvm_add_csc(s, to_llvm_type<fp_t>(s.context()), degree, 1);

            s.optimise();

            s.compile();

            auto f_ptr = reinterpret_cast<void (*)(std::uint32_t *, const fp_t *)>(s.jit_lookup(
                fmt::format("heyoka_csc_degree_{}_{}", degree, llvm_mangle_type(to_llvm_type<fp_t>(s.context())))));

            // Random testing.
            std::uniform_real_distribution<double> rdist(-10., 10.);
            std::uniform_int_distribution<int> idist(0, 9);
            std::uint32_t out = 0;
            std::vector<fp_t> cfs(degree + 1u), nz_values;

            for (auto i = 0; i < ntrials * 10; ++i) {
                nz_values.clear();

                // Generate random coefficients, putting
                // in a zero every once in a while.
                std::generate(cfs.begin(), cfs.end(), [&idist, &rdist, &nz_values]() {
                    auto ret = idist(rng) == 0 ? fp_t(0) : fp_t(rdist(rng));
                    if (ret != 0) {
                        nz_values.push_back(ret);
                    }

                    return ret;
                });

                // Determine the number of sign changes.
                auto n_sc = 0u;
                for (decltype(nz_values.size()) j = 1; j < nz_values.size(); ++j) {
                    n_sc += sgn(nz_values[j]) != sgn(nz_values[j - 1u]);
                }

                // Check it.
                f_ptr(&out, cfs.data());
                REQUIRE(out == n_sc);
            }

            // A full zero test.
            std::fill(cfs.begin(), cfs.end(), fp_t(0));
            f_ptr(&out, cfs.data());
            REQUIRE(out == 0u);

            // Full 1.
            std::fill(cfs.begin(), cfs.end(), fp_t(1));
            f_ptr(&out, cfs.data());
            REQUIRE(out == 0u);

            // Full -1.
            std::fill(cfs.begin(), cfs.end(), fp_t(-1));
            f_ptr(&out, cfs.data());
            REQUIRE(out == 0u);
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("csc_batch")
{
    using detail::llvm_add_csc;
    using detail::llvm_mangle_type;
    using detail::make_vector_type;
    using detail::to_llvm_type;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                const auto degree = 4u;

                llvm_add_csc(s, to_llvm_type<fp_t>(s.context()), degree, batch_size);

                s.optimise();

                s.compile();

                auto f_ptr = reinterpret_cast<void (*)(std::uint32_t *, const fp_t *)>(s.jit_lookup(
                    fmt::format("heyoka_csc_degree_{}_{}", degree,
                                llvm_mangle_type(make_vector_type(to_llvm_type<fp_t>(s.context()), batch_size)))));

                // Random testing.
                std::uniform_real_distribution<double> rdist(-10., 10.);
                std::uniform_int_distribution<int> idist(0, 9);
                std::vector<std::uint32_t> out(batch_size), n_sc(batch_size);
                std::vector<fp_t> cfs((degree + 1u) * batch_size), nz_values;

                for (auto i = 0; i < ntrials * 10; ++i) {
                    // Generate random coefficients, putting
                    // in a zero every once in a while.
                    std::generate(cfs.begin(), cfs.end(),
                                  [&idist, &rdist]() { return idist(rng) == 0 ? fp_t(0) : fp_t(rdist(rng)); });

                    // Determine the number of sign changes for each batch element.
                    for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
                        nz_values.clear();

                        for (auto j = 0u; j <= degree; ++j) {
                            if (cfs[batch_size * j + batch_idx] != 0) {
                                nz_values.push_back(cfs[batch_size * j + batch_idx]);
                            }
                        }

                        n_sc[batch_idx] = 0;
                        for (decltype(nz_values.size()) j = 1; j < nz_values.size(); ++j) {
                            n_sc[batch_idx] += sgn(nz_values[j]) != sgn(nz_values[j - 1u]);
                        }
                    }

                    // Check the result.
                    f_ptr(out.data(), cfs.data());
                    REQUIRE(std::equal(out.begin(), out.end(), n_sc.begin()));
                }

                // A full zero test.
                std::fill(cfs.begin(), cfs.end(), fp_t(0));
                f_ptr(out.data(), cfs.data());
                REQUIRE(std::all_of(out.begin(), out.end(), [](auto x) { return x == 0; }));

                // Full 1.
                std::fill(cfs.begin(), cfs.end(), fp_t(1));
                f_ptr(out.data(), cfs.data());
                REQUIRE(std::all_of(out.begin(), out.end(), [](auto x) { return x == 0; }));

                // Full -1.
                std::fill(cfs.begin(), cfs.end(), fp_t(-1));
                f_ptr(out.data(), cfs.data());
                REQUIRE(std::all_of(out.begin(), out.end(), [](auto x) { return x == 0; }));
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("minmax")
{
    using detail::to_llvm_type;
    using std::isnan;

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

                // llvm_min.
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "min", &md);

                auto ret_ptr = f->args().begin();
                auto a_ptr = f->args().begin() + 1;
                auto b_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto a = detail::load_vector_from_memory(builder, val_t, a_ptr, batch_size);
                auto b = detail::load_vector_from_memory(builder, val_t, b_ptr, batch_size);

                auto ret = detail::llvm_min(s, a, b);

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                // Create the return value.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // llvm_max.
                f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "max", &md);

                ret_ptr = f->args().begin();
                a_ptr = f->args().begin() + 1;
                b_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                a = detail::load_vector_from_memory(builder, val_t, a_ptr, batch_size);
                b = detail::load_vector_from_memory(builder, val_t, b_ptr, batch_size);

                ret = detail::llvm_max(s, a, b);

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                // Create the return value.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // llvm_min_nan.
                f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "min_nan", &md);

                ret_ptr = f->args().begin();
                a_ptr = f->args().begin() + 1;
                b_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                a = detail::load_vector_from_memory(builder, val_t, a_ptr, batch_size);
                b = detail::load_vector_from_memory(builder, val_t, b_ptr, batch_size);

                ret = detail::llvm_min_nan(s, a, b);

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                // Create the return value.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // llvm_max_nan.
                f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "max_nan", &md);

                ret_ptr = f->args().begin();
                a_ptr = f->args().begin() + 1;
                b_ptr = f->args().begin() + 2;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                a = detail::load_vector_from_memory(builder, val_t, a_ptr, batch_size);
                b = detail::load_vector_from_memory(builder, val_t, b_ptr, batch_size);

                ret = detail::llvm_max_nan(s, a, b);

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                // Create the return value.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the pointers.
                auto llvm_min = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("min"));
                auto llvm_max = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("max"));
                auto llvm_min_nan
                    = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("min_nan"));
                auto llvm_max_nan
                    = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("max_nan"));

                // The C++ implementations.
                auto cpp_min = [batch_size](fp_t *ret_ptr_, const fp_t *a_ptr_, const fp_t *b_ptr_) {
                    for (auto i = 0u; i < batch_size; ++i) {
                        ret_ptr_[i] = (b_ptr_[i] < a_ptr_[i]) ? b_ptr_[i] : a_ptr_[i];
                    }
                };
                auto cpp_max = [batch_size](fp_t *ret_ptr_, const fp_t *a_ptr_, const fp_t *b_ptr_) {
                    for (auto i = 0u; i < batch_size; ++i) {
                        ret_ptr_[i] = (a_ptr_[i] < b_ptr_[i]) ? b_ptr_[i] : a_ptr_[i];
                    }
                };
                auto cpp_min_nan = [batch_size](fp_t *ret_ptr_, const fp_t *a_ptr_, const fp_t *b_ptr_) {
                    for (auto i = 0u; i < batch_size; ++i) {
                        if (isnan(a_ptr_[i]) || isnan(b_ptr_[i])) {
                            ret_ptr_[i] = std::numeric_limits<fp_t>::quiet_NaN();
                        } else {
                            ret_ptr_[i] = (b_ptr_[i] < a_ptr_[i]) ? b_ptr_[i] : a_ptr_[i];
                        }
                    }
                };
                auto cpp_max_nan = [batch_size](fp_t *ret_ptr_, const fp_t *a_ptr_, const fp_t *b_ptr_) {
                    for (auto i = 0u; i < batch_size; ++i) {
                        if (isnan(a_ptr_[i]) || isnan(b_ptr_[i])) {
                            ret_ptr_[i] = std::numeric_limits<fp_t>::quiet_NaN();
                        } else {
                            ret_ptr_[i] = (a_ptr_[i] < b_ptr_[i]) ? b_ptr_[i] : a_ptr_[i];
                        }
                    }
                };

                std::vector<fp_t> av(batch_size), bv(batch_size), retv1(batch_size), retv2(batch_size);
                std::uniform_real_distribution<double> rdist(-10, 10);
                std::uniform_int_distribution<int> idist(0, 1);

                for (auto i = 0; i < ntrials; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        if (idist(rng) && idist(rng) && idist(rng)) {
                            av[j] = std::numeric_limits<fp_t>::quiet_NaN();
                        } else {
                            av[j] = rdist(rng);
                        }

                        if (idist(rng) && idist(rng) && idist(rng)) {
                            bv[j] = std::numeric_limits<fp_t>::quiet_NaN();
                        } else {
                            bv[j] = rdist(rng);
                        }
                    }

                    llvm_min(retv1.data(), av.data(), bv.data());
                    cpp_min(retv2.data(), av.data(), bv.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (isnan(retv1[j])) {
                            REQUIRE(isnan(retv2[j]));
                        } else {
                            REQUIRE(retv1[j] == retv2[j]);
                        }
                    }

                    llvm_max(retv1.data(), av.data(), bv.data());
                    cpp_max(retv2.data(), av.data(), bv.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (isnan(retv1[j])) {
                            REQUIRE(isnan(retv2[j]));
                        } else {
                            REQUIRE(retv1[j] == retv2[j]);
                        }
                    }

                    llvm_min_nan(retv1.data(), av.data(), bv.data());
                    cpp_min_nan(retv2.data(), av.data(), bv.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (isnan(retv1[j])) {
                            REQUIRE(isnan(retv2[j]));
                        } else {
                            REQUIRE(retv1[j] == retv2[j]);
                        }
                    }

                    llvm_max_nan(retv1.data(), av.data(), bv.data());
                    cpp_max_nan(retv2.data(), av.data(), bv.data());

                    for (auto j = 0u; j < batch_size; ++j) {
                        if (isnan(retv1[j])) {
                            REQUIRE(isnan(retv2[j]));
                        } else {
                            REQUIRE(retv1[j] == retv2[j]);
                        }
                    }
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("fma scalar")
{
    using detail::llvm_fma;
    using detail::to_llvm_type;
    using std::fma;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{val_t, val_t, val_t};
            auto *ft = llvm::FunctionType::get(val_t, fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_fma", &md);

            auto x = f->args().begin();
            auto y = f->args().begin() + 1;
            auto z = f->args().begin() + 2;

            builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

            builder.CreateRet(llvm_fma(s, x, y, z));

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<fp_t (*)(fp_t, fp_t, fp_t)>(s.jit_lookup("hey_fma"));

            auto a = fp_t(123);
            auto b = fp_t(2) / fp_t(7);
            auto c = fp_t(-3) / fp_t(4);

            REQUIRE(f_ptr(a, b, c) == approximately(fma(a, b, c), fp_t(10)));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("fma batch")
{
    using detail::llvm_fma;
    using detail::to_llvm_type;
    using std::fma;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs(4u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_fma", &md);

                auto ret_ptr = f->args().begin();
                auto x_ptr = f->args().begin() + 1;
                auto y_ptr = f->args().begin() + 2;
                auto z_ptr = f->args().begin() + 3;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto ret = llvm_fma(s, detail::load_vector_from_memory(builder, val_t, x_ptr, batch_size),
                                    detail::load_vector_from_memory(builder, val_t, y_ptr, batch_size),
                                    detail::load_vector_from_memory(builder, val_t, z_ptr, batch_size));

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *, fp_t *)>(s.jit_lookup("hey_fma"));

                // Setup the arguments and the output value.
                std::vector<fp_t> ret_vec(batch_size), a_vec(ret_vec), b_vec(ret_vec), c_vec(ret_vec);
                for (auto i = 0u; i < batch_size; ++i) {
                    a_vec[i] = i + 1u;
                    b_vec[i] = a_vec[i] * 10 * (i + 1u);
                    c_vec[i] = b_vec[i] * 10 * (i + 1u);
                }

                f_ptr(ret_vec.data(), a_vec.data(), b_vec.data(), c_vec.data());

                for (auto i = 0u; i < batch_size; ++i) {
                    auto a = a_vec[i];
                    auto b = b_vec[i];
                    auto c = c_vec[i];

                    REQUIRE(ret_vec[i] == approximately(fma(a, b, c), fp_t(10)));
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("eft_product scalar")
{
    using detail::llvm_eft_product;
    using detail::to_llvm_type;
    using std::abs;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(val_t), llvm::PointerType::getUnqual(val_t),
                                            val_t, val_t};
            auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_eft_prod", &md);

            {
                auto x_ptr = f->args().begin();
                auto y_ptr = f->args().begin() + 1;
                auto a = f->args().begin() + 2;
                auto b = f->args().begin() + 3;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto [x, y] = llvm_eft_product(s, a, b);

                builder.CreateStore(x, x_ptr);
                builder.CreateStore(y, y_ptr);
            }

            builder.CreateRetVoid();

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t, fp_t)>(s.jit_lookup("hey_eft_prod"));

            std::uniform_int_distribution<int> idist(-10000, 10000);

            for (auto i = 0; i < ntrials; ++i) {
                fp_t x, y;

                auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                den1 += (den1 == 0);
                den2 += (den2 == 0);

                auto a = fp_t(num1) / fp_t(den1);
                auto b = fp_t(num2) / fp_t(den2);

                f_ptr(&x, &y, a, b);

                REQUIRE(x == a * b);

#if defined(HEYOKA_HAVE_REAL128)
                if constexpr (!std::is_same_v<fp_t, mppp::real128>) {
#endif
                    namespace bmp = boost::multiprecision;
                    using mp_fp_t
                        = bmp::number<bmp::cpp_bin_float<std::numeric_limits<fp_t>::digits * 2, bmp::digit_base_2>>;

                    REQUIRE(mp_fp_t(x) + mp_fp_t(y) == mp_fp_t(a) * mp_fp_t(b));
#if defined(HEYOKA_HAVE_REAL128)
                }
#endif
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("eft_product batch")
{
    using detail::llvm_eft_product;
    using detail::to_llvm_type;
    using std::abs;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs(4u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_eft_prod", &md);

                auto x_ptr = f->args().begin();
                auto y_ptr = f->args().begin() + 1;
                auto a_ptr = f->args().begin() + 2;
                auto b_ptr = f->args().begin() + 3;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto [x, y] = llvm_eft_product(s, detail::load_vector_from_memory(builder, val_t, a_ptr, batch_size),
                                               detail::load_vector_from_memory(builder, val_t, b_ptr, batch_size));

                detail::store_vector_to_memory(builder, x_ptr, x);
                detail::store_vector_to_memory(builder, y_ptr, y);

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *, fp_t *)>(s.jit_lookup("hey_eft_prod"));

                std::uniform_int_distribution<int> idist(-10000, 10000);

                std::vector<fp_t> x_vec(batch_size), y_vec(x_vec), a_vec(x_vec), b_vec(x_vec);

                for (auto j = 0; j < ntrials; ++j) {
                    // Setup the arguments and the output value.
                    for (auto i = 0u; i < batch_size; ++i) {
                        auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                        den1 += (den1 == 0);
                        den2 += (den2 == 0);

                        a_vec[i] = fp_t(num1) / fp_t(den1);
                        b_vec[i] = fp_t(num2) / fp_t(den2);
                    }

                    f_ptr(x_vec.data(), y_vec.data(), a_vec.data(), b_vec.data());

                    for (auto i = 0u; i < batch_size; ++i) {
                        auto a = a_vec[i];
                        auto b = b_vec[i];
                        auto xv = x_vec[i];
                        auto yv = y_vec[i];

                        REQUIRE(xv == a * b);

#if defined(HEYOKA_HAVE_REAL128)
                        if constexpr (!std::is_same_v<fp_t, mppp::real128>) {
#endif
                            namespace bmp = boost::multiprecision;
                            using mp_fp_t = bmp::number<
                                bmp::cpp_bin_float<std::numeric_limits<fp_t>::digits * 2, bmp::digit_base_2>>;

                            REQUIRE(mp_fp_t(xv) + mp_fp_t(yv) == mp_fp_t(a) * mp_fp_t(b));
#if defined(HEYOKA_HAVE_REAL128)
                        }
#endif
                    }
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl mul scalar")
{
    using detail::llvm_dl_mul;
    using detail::to_llvm_type;
    using std::abs;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{
                llvm::PointerType::getUnqual(val_t), llvm::PointerType::getUnqual(val_t), val_t, val_t, val_t, val_t};
            auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_mul", &md);

            {
                auto x_ptr = f->args().begin();
                auto y_ptr = f->args().begin() + 1;
                auto a_hi = f->args().begin() + 2;
                auto a_lo = f->args().begin() + 3;
                auto b_hi = f->args().begin() + 4;
                auto b_lo = f->args().begin() + 5;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto [x, y] = llvm_dl_mul(s, a_hi, a_lo, b_hi, b_lo);

                builder.CreateStore(x, x_ptr);
                builder.CreateStore(y, y_ptr);
            }

            builder.CreateRetVoid();

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t, fp_t, fp_t, fp_t)>(s.jit_lookup("hey_dl_mul"));

            std::uniform_int_distribution<int> idist(-10000, 10000);

            for (auto i = 0; i < ntrials; ++i) {
                fp_t x, y;

                auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                den1 += (den1 == 0);
                den2 += (den2 == 0);

                auto a_hi = fp_t(num1) / fp_t(den1);
                auto a_lo = fp_t(num2) / fp_t(den2);
                if (abs(a_hi) < abs(a_lo)) {
                    std::swap(a_hi, a_lo);
                }
                std::tie(a_hi, a_lo) = detail::eft_add_dekker(a_hi, a_lo);

                num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                den1 += (den1 == 0);
                den2 += (den2 == 0);

                auto b_hi = fp_t(num1) / fp_t(den1);
                auto b_lo = fp_t(num2) / fp_t(den2);
                if (abs(b_hi) < abs(b_lo)) {
                    std::swap(b_hi, b_lo);
                }
                std::tie(b_hi, b_lo) = detail::eft_add_dekker(b_hi, b_lo);

                f_ptr(&x, &y, a_hi, a_lo, b_hi, b_lo);

                auto ret1_hi = x;
                auto ret1_lo = y;

                f_ptr(&x, &y, b_hi, b_lo, a_hi, a_lo);

                auto ret2_hi = x;
                auto ret2_lo = y;

                // Check commutativity.
                REQUIRE(ret1_hi == ret2_hi);
                REQUIRE(ret1_lo == ret2_lo);

                // Check smallness.
                REQUIRE(ret1_hi + ret1_lo == ret1_hi);
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl mul batch")
{
    using detail::llvm_dl_mul;
    using detail::to_llvm_type;
    using std::abs;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs(6u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_mul", &md);

                {
                    auto x_ptr = f->args().begin();
                    auto y_ptr = f->args().begin() + 1;
                    auto a_hi_ptr = f->args().begin() + 2;
                    auto a_lo_ptr = f->args().begin() + 3;
                    auto b_hi_ptr = f->args().begin() + 4;
                    auto b_lo_ptr = f->args().begin() + 5;

                    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                    auto [x, y] = llvm_dl_mul(s, detail::load_vector_from_memory(builder, val_t, a_hi_ptr, batch_size),
                                              detail::load_vector_from_memory(builder, val_t, a_lo_ptr, batch_size),
                                              detail::load_vector_from_memory(builder, val_t, b_hi_ptr, batch_size),
                                              detail::load_vector_from_memory(builder, val_t, b_lo_ptr, batch_size));

                    detail::store_vector_to_memory(builder, x_ptr, x);
                    detail::store_vector_to_memory(builder, y_ptr, y);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();
                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *, fp_t *, fp_t *, fp_t *)>(
                    s.jit_lookup("hey_dl_mul"));

                std::uniform_int_distribution<int> idist(-10000, 10000);

                std::vector<fp_t> x_vec(batch_size), y_vec(x_vec), a_hi_vec(x_vec), a_lo_vec(x_vec), b_hi_vec(x_vec),
                    b_lo_vec(x_vec);

                for (auto j = 0; j < ntrials; ++j) {
                    // Setup the arguments and the output value.
                    for (auto i = 0u; i < batch_size; ++i) {
                        auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                        den1 += (den1 == 0);
                        den2 += (den2 == 0);

                        auto a_hi = fp_t(num1) / fp_t(den1);
                        auto a_lo = fp_t(num2) / fp_t(den2);
                        if (abs(a_hi) < abs(a_lo)) {
                            std::swap(a_hi, a_lo);
                        }
                        std::tie(a_hi, a_lo) = detail::eft_add_dekker(a_hi, a_lo);

                        num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                        den1 += (den1 == 0);
                        den2 += (den2 == 0);

                        auto b_hi = fp_t(num1) / fp_t(den1);
                        auto b_lo = fp_t(num2) / fp_t(den2);
                        if (abs(b_hi) < abs(b_lo)) {
                            std::swap(b_hi, b_lo);
                        }
                        std::tie(b_hi, b_lo) = detail::eft_add_dekker(b_hi, b_lo);

                        a_hi_vec[i] = a_hi;
                        a_lo_vec[i] = a_lo;
                        b_hi_vec[i] = b_hi;
                        b_lo_vec[i] = b_lo;
                    }

                    f_ptr(x_vec.data(), y_vec.data(), a_hi_vec.data(), a_lo_vec.data(), b_hi_vec.data(),
                          b_lo_vec.data());

                    auto ret1_hi = x_vec;
                    auto ret1_lo = y_vec;

                    f_ptr(x_vec.data(), y_vec.data(), b_hi_vec.data(), b_lo_vec.data(), a_hi_vec.data(),
                          a_lo_vec.data());

                    auto ret2_hi = x_vec;
                    auto ret2_lo = y_vec;

                    for (auto i = 0u; i < batch_size; ++i) {
                        // Check commutativity.
                        REQUIRE(ret1_hi[i] == ret2_hi[i]);
                        REQUIRE(ret1_lo[i] == ret2_lo[i]);

                        // Check smallness.
                        REQUIRE(ret1_hi[i] == ret1_hi[i] + ret1_lo[i]);
                    }
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl div scalar")
{
    using detail::llvm_dl_div;
    using detail::to_llvm_type;
    using std::abs;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{
                llvm::PointerType::getUnqual(val_t), llvm::PointerType::getUnqual(val_t), val_t, val_t, val_t, val_t};
            auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_div", &md);

            {
                auto x_ptr = f->args().begin();
                auto y_ptr = f->args().begin() + 1;
                auto a_hi = f->args().begin() + 2;
                auto a_lo = f->args().begin() + 3;
                auto b_hi = f->args().begin() + 4;
                auto b_lo = f->args().begin() + 5;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto [x, y] = llvm_dl_div(s, a_hi, a_lo, b_hi, b_lo);

                builder.CreateStore(x, x_ptr);
                builder.CreateStore(y, y_ptr);
            }

            builder.CreateRetVoid();

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t, fp_t, fp_t, fp_t)>(s.jit_lookup("hey_dl_div"));

            std::uniform_int_distribution<int> idist(-10000, 10000);

            for (auto i = 0; i < ntrials; ++i) {
                fp_t x, y;

                auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                den1 += (den1 == 0);
                den2 += (den2 == 0);

                auto a_hi = fp_t(num1) / fp_t(den1);
                auto a_lo = fp_t(num2) / fp_t(den2);
                if (abs(a_hi) < abs(a_lo)) {
                    std::swap(a_hi, a_lo);
                }
                std::tie(a_hi, a_lo) = detail::eft_add_dekker(a_hi, a_lo);

                num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                den1 += (den1 == 0);
                den2 += (den2 == 0);

                auto b_hi = fp_t(num1) / fp_t(den1);
                auto b_lo = fp_t(num2) / fp_t(den2);
                if (abs(b_hi) < abs(b_lo)) {
                    std::swap(b_hi, b_lo);
                }
                std::tie(b_hi, b_lo) = detail::eft_add_dekker(b_hi, b_lo);

                // NOTE: avoid (unlikely) division by zero.
                if (b_hi == 0) {
                    b_hi = 1;
                }

                f_ptr(&x, &y, a_hi, a_lo, b_hi, b_lo);

                auto ret1_hi = x;
                auto ret1_lo = y;

                // Check smallness.
                REQUIRE(ret1_hi + ret1_lo == ret1_hi);
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl div batch")
{
    using detail::llvm_dl_div;
    using detail::to_llvm_type;
    using std::abs;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs(6u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_div", &md);

                {
                    auto x_ptr = f->args().begin();
                    auto y_ptr = f->args().begin() + 1;
                    auto a_hi_ptr = f->args().begin() + 2;
                    auto a_lo_ptr = f->args().begin() + 3;
                    auto b_hi_ptr = f->args().begin() + 4;
                    auto b_lo_ptr = f->args().begin() + 5;

                    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                    auto [x, y] = llvm_dl_div(s, detail::load_vector_from_memory(builder, val_t, a_hi_ptr, batch_size),
                                              detail::load_vector_from_memory(builder, val_t, a_lo_ptr, batch_size),
                                              detail::load_vector_from_memory(builder, val_t, b_hi_ptr, batch_size),
                                              detail::load_vector_from_memory(builder, val_t, b_lo_ptr, batch_size));

                    detail::store_vector_to_memory(builder, x_ptr, x);
                    detail::store_vector_to_memory(builder, y_ptr, y);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();
                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *, fp_t *, fp_t *, fp_t *)>(
                    s.jit_lookup("hey_dl_div"));

                std::uniform_int_distribution<int> idist(-10000, 10000);

                std::vector<fp_t> x_vec(batch_size), y_vec(x_vec), a_hi_vec(x_vec), a_lo_vec(x_vec), b_hi_vec(x_vec),
                    b_lo_vec(x_vec);

                for (auto j = 0; j < ntrials; ++j) {
                    // Setup the arguments and the output value.
                    for (auto i = 0u; i < batch_size; ++i) {
                        auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                        den1 += (den1 == 0);
                        den2 += (den2 == 0);

                        auto a_hi = fp_t(num1) / fp_t(den1);
                        auto a_lo = fp_t(num2) / fp_t(den2);
                        if (abs(a_hi) < abs(a_lo)) {
                            std::swap(a_hi, a_lo);
                        }
                        std::tie(a_hi, a_lo) = detail::eft_add_dekker(a_hi, a_lo);

                        num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                        den1 += (den1 == 0);
                        den2 += (den2 == 0);

                        auto b_hi = fp_t(num1) / fp_t(den1);
                        auto b_lo = fp_t(num2) / fp_t(den2);
                        if (abs(b_hi) < abs(b_lo)) {
                            std::swap(b_hi, b_lo);
                        }
                        std::tie(b_hi, b_lo) = detail::eft_add_dekker(b_hi, b_lo);

                        // NOTE: avoid (unlikely) division by zero.
                        if (b_hi == 0) {
                            b_hi = 1;
                        }

                        a_hi_vec[i] = a_hi;
                        a_lo_vec[i] = a_lo;
                        b_hi_vec[i] = b_hi;
                        b_lo_vec[i] = b_lo;
                    }

                    f_ptr(x_vec.data(), y_vec.data(), a_hi_vec.data(), a_lo_vec.data(), b_hi_vec.data(),
                          b_lo_vec.data());

                    auto ret1_hi = x_vec;
                    auto ret1_lo = y_vec;

                    for (auto i = 0u; i < batch_size; ++i) {
                        // Check smallness.
                        REQUIRE(ret1_hi[i] == ret1_hi[i] + ret1_lo[i]);
                    }
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("floor scalar")
{
    using detail::llvm_floor;
    using detail::to_llvm_type;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{val_t};
            auto *ft = llvm::FunctionType::get(val_t, fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_floor", &md);

            auto x = f->args().begin();

            builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

            builder.CreateRet(llvm_floor(s, x));

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<fp_t (*)(fp_t)>(s.jit_lookup("hey_floor"));

            REQUIRE(f_ptr(fp_t(2) / 7) == 0);
            REQUIRE(f_ptr(fp_t(8) / 7) == 1);
            REQUIRE(f_ptr(fp_t(-2) / 7) == -1);
            REQUIRE(f_ptr(fp_t(-8) / 7) == -2);
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("floor batch")
{
    using detail::llvm_floor;
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

                std::vector<llvm::Type *> fargs(2u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_floor", &md);

                auto ret_ptr = f->args().begin();
                auto x_ptr = f->args().begin() + 1;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto ret = llvm_floor(s, detail::load_vector_from_memory(builder, val_t, x_ptr, batch_size));

                detail::store_vector_to_memory(builder, ret_ptr, ret);

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *)>(s.jit_lookup("hey_floor"));

                // Setup the argument and the output value.
                std::vector<fp_t> ret_vec(batch_size), a_vec(ret_vec);
                for (auto i = 0u; i < batch_size; ++i) {
                    a_vec[i] = fp_t(i + 1u) / 3 * (i % 2u == 0 ? 1 : -1);
                }

                f_ptr(ret_vec.data(), a_vec.data());

                for (auto i = 0u; i < batch_size; ++i) {
                    REQUIRE(ret_vec[i] == floor(a_vec[i]));
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl floor scalar")
{
    using detail::llvm_dl_floor;
    using detail::to_llvm_type;
    using std::abs;
    using std::floor;
    using std::trunc;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            auto &md = s.module();
            auto &builder = s.builder();
            auto &context = s.context();

            auto val_t = to_llvm_type<fp_t>(context);

            std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(val_t), llvm::PointerType::getUnqual(val_t),
                                            val_t, val_t};
            auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_floor", &md);

            {
                auto x_ptr = f->args().begin();
                auto y_ptr = f->args().begin() + 1;
                auto a_hi = f->args().begin() + 2;
                auto a_lo = f->args().begin() + 3;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto [x, y] = llvm_dl_floor(s, a_hi, a_lo);

                builder.CreateStore(x, x_ptr);
                builder.CreateStore(y, y_ptr);
            }

            builder.CreateRetVoid();

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t, fp_t)>(s.jit_lookup("hey_dl_floor"));

            std::uniform_int_distribution<int> idist(-10000, 10000);

            for (auto i = 0; i < ntrials; ++i) {
                fp_t x, y;

                auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                den1 += (den1 == 0);
                den2 += (den2 == 0);

                auto a_hi = fp_t(num1) / fp_t(den1);
                auto a_lo = fp_t(num2) / fp_t(den2);
                if (abs(a_hi) < abs(a_lo)) {
                    std::swap(a_hi, a_lo);
                }
                std::tie(a_hi, a_lo) = detail::eft_add_dekker(a_hi, a_lo);

                f_ptr(&x, &y, a_hi, a_lo);

                REQUIRE(trunc(x) == x);
                REQUIRE(y == 0);

                a_hi = floor(a_hi);

                f_ptr(&x, &y, a_hi, a_lo);

                REQUIRE(trunc(x) == x);
                REQUIRE(y == 0);
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl floor batch")
{
    using detail::llvm_dl_floor;
    using detail::to_llvm_type;
    using std::abs;
    using std::floor;
    using std::trunc;

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        for (auto batch_size : {1u, 2u, 4u, 13u}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                auto &md = s.module();
                auto &builder = s.builder();
                auto &context = s.context();

                auto val_t = to_llvm_type<fp_t>(context);

                std::vector<llvm::Type *> fargs(4u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_floor", &md);

                {
                    auto x_ptr = f->args().begin();
                    auto y_ptr = f->args().begin() + 1;
                    auto a_hi_ptr = f->args().begin() + 2;
                    auto a_lo_ptr = f->args().begin() + 3;

                    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                    auto [x, y]
                        = llvm_dl_floor(s, detail::load_vector_from_memory(builder, val_t, a_hi_ptr, batch_size),
                                        detail::load_vector_from_memory(builder, val_t, a_lo_ptr, batch_size));

                    detail::store_vector_to_memory(builder, x_ptr, x);
                    detail::store_vector_to_memory(builder, y_ptr, y);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *, fp_t *)>(s.jit_lookup("hey_dl_floor"));

                std::uniform_int_distribution<int> idist(-10000, 10000);

                std::vector<fp_t> x_vec(batch_size), y_vec(x_vec), a_hi_vec(x_vec), a_lo_vec(x_vec);

                for (auto j = 0; j < ntrials; ++j) {
                    // Setup the argument and the output value.
                    for (auto i = 0u; i < batch_size; ++i) {
                        auto num1 = idist(rng), num2 = idist(rng), den1 = idist(rng), den2 = idist(rng);

                        den1 += (den1 == 0);
                        den2 += (den2 == 0);

                        auto a_hi = fp_t(num1) / fp_t(den1);
                        auto a_lo = fp_t(num2) / fp_t(den2);
                        if (abs(a_hi) < abs(a_lo)) {
                            std::swap(a_hi, a_lo);
                        }
                        std::tie(a_hi, a_lo) = detail::eft_add_dekker(a_hi, a_lo);

                        a_hi_vec[i] = a_hi;
                        a_lo_vec[i] = a_lo;
                    }

                    f_ptr(x_vec.data(), y_vec.data(), a_hi_vec.data(), a_lo_vec.data());

                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(trunc(x_vec[i]) == x_vec[i]);
                        REQUIRE(y_vec[i] == 0);

                        a_hi_vec[i] = floor(a_hi_vec[i]);
                    }

                    f_ptr(x_vec.data(), y_vec.data(), a_hi_vec.data(), a_lo_vec.data());

                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(trunc(x_vec[i]) == x_vec[i]);
                        REQUIRE(y_vec[i] == 0);
                    }
                }
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl modulus scalar")
{
    using detail::llvm_dl_modulus;
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

            std::vector<llvm::Type *> fargs{
                llvm::PointerType::getUnqual(val_t), llvm::PointerType::getUnqual(val_t), val_t, val_t, val_t, val_t};
            auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
            auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_modulus", &md);

            {
                auto x_ptr = f->args().begin();
                auto y_ptr = f->args().begin() + 1;
                auto a_hi = f->args().begin() + 2;
                auto a_lo = f->args().begin() + 3;
                auto b_hi = f->args().begin() + 4;
                auto b_lo = f->args().begin() + 5;

                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                auto [x, y] = llvm_dl_modulus(s, a_hi, a_lo, b_hi, b_lo);

                builder.CreateStore(x, x_ptr);
                builder.CreateStore(y, y_ptr);
            }

            builder.CreateRetVoid();

            // Verify.
            s.verify_function(f);

            // Run the optimisation pass.
            s.optimise();

            // Compile.
            s.compile();

            // Fetch the function pointer.
            auto f_ptr
                = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t, fp_t, fp_t, fp_t)>(s.jit_lookup("hey_dl_modulus"));

#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (!std::is_same_v<fp_t, mppp::real128>) {
#endif
                namespace bmp = boost::multiprecision;
                using mp_fp_t
                    = bmp::number<bmp::cpp_bin_float<std::numeric_limits<fp_t>::digits * 2, bmp::digit_base_2>>;

                std::uniform_real_distribution<fp_t> op_dist(-1e6, 1e6), quo_dist(.1, 10.);

                for (auto i = 0; i < ntrials; ++i) {
                    auto x = fp_t(op_dist(rng)), y = fp_t(quo_dist(rng));

                    fp_t res_hi, res_lo;

                    f_ptr(&res_hi, &res_lo, x, 0, y, 0);

                    auto res_mp = mp_fp_t(x) - mp_fp_t(y) * floor(mp_fp_t(x) / mp_fp_t(y));

                    REQUIRE(res_hi == approximately(static_cast<fp_t>(res_mp), fp_t(10)));
                }

#if defined(HEYOKA_HAVE_REAL128)
            }
#endif
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("dl modulus batch")
{
    using detail::llvm_dl_modulus;
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

                std::vector<llvm::Type *> fargs(6u, llvm::PointerType::getUnqual(val_t));
                auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
                auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "hey_dl_modulus", &md);

                {
                    auto x_ptr = f->args().begin();
                    auto y_ptr = f->args().begin() + 1;
                    auto a_hi_ptr = f->args().begin() + 2;
                    auto a_lo_ptr = f->args().begin() + 3;
                    auto b_hi_ptr = f->args().begin() + 4;
                    auto b_lo_ptr = f->args().begin() + 5;

                    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

                    auto [x, y]
                        = llvm_dl_modulus(s, detail::load_vector_from_memory(builder, val_t, a_hi_ptr, batch_size),
                                          detail::load_vector_from_memory(builder, val_t, a_lo_ptr, batch_size),
                                          detail::load_vector_from_memory(builder, val_t, b_hi_ptr, batch_size),
                                          detail::load_vector_from_memory(builder, val_t, b_lo_ptr, batch_size));

                    detail::store_vector_to_memory(builder, x_ptr, x);
                    detail::store_vector_to_memory(builder, y_ptr, y);
                }

                builder.CreateRetVoid();

                // Verify.
                s.verify_function(f);

                // Run the optimisation pass.
                s.optimise();

                // Compile.
                s.compile();

                // Fetch the function pointer.
                auto f_ptr = reinterpret_cast<void (*)(fp_t *, fp_t *, fp_t *, fp_t *, fp_t *, fp_t *)>(
                    s.jit_lookup("hey_dl_modulus"));

#if defined(HEYOKA_HAVE_REAL128)
                if constexpr (!std::is_same_v<fp_t, mppp::real128>) {
#endif
                    namespace bmp = boost::multiprecision;
                    using mp_fp_t
                        = bmp::number<bmp::cpp_bin_float<std::numeric_limits<fp_t>::digits * 2, bmp::digit_base_2>>;

                    std::uniform_real_distribution<fp_t> op_dist(-1e6, 1e6), quo_dist(.1, 10.);

                    std::vector<fp_t> x_vec(batch_size), y_vec(x_vec), a_hi_vec(x_vec), a_lo_vec(x_vec),
                        b_hi_vec(x_vec), b_lo_vec(x_vec);

                    for (auto j = 0; j < ntrials; ++j) {
                        // Setup the arguments.
                        for (auto i = 0u; i < batch_size; ++i) {
                            a_hi_vec[i] = fp_t(op_dist(rng));
                            a_lo_vec[i] = 0;

                            b_hi_vec[i] = fp_t(quo_dist(rng));
                            b_lo_vec[i] = 0;
                        }

                        f_ptr(x_vec.data(), y_vec.data(), a_hi_vec.data(), a_lo_vec.data(), b_hi_vec.data(),
                              b_lo_vec.data());

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto res_mp = mp_fp_t(a_hi_vec[i])
                                          - mp_fp_t(b_hi_vec[i]) * floor(mp_fp_t(a_hi_vec[i]) / mp_fp_t(b_hi_vec[i]));

                            REQUIRE(x_vec[i] == approximately(static_cast<fp_t>(res_mp), fp_t(10)));
                        }
                    }

#if defined(HEYOKA_HAVE_REAL128)
                }
#endif
            }
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("get_alignment")
{
    llvm_state s;

    auto &md = s.module();
    auto &context = s.context();
    auto &builder = s.builder();

    auto *tp = detail::to_llvm_type<double>(context);
    REQUIRE(detail::get_alignment(md, tp) == alignof(double));

#if !defined(HEYOKA_ARCH_PPC)
    tp = detail::to_llvm_type<long double>(context);
    REQUIRE(detail::get_alignment(md, tp) == alignof(long double));
#endif

#if defined(HEYOKA_HAVE_REAL128)
    tp = detail::to_llvm_type<mppp::real128>(context);
    REQUIRE(detail::get_alignment(md, tp) == alignof(mppp::real128));
#endif

    REQUIRE(detail::get_alignment(md, builder.getInt32Ty()) == alignof(std::uint32_t));
    REQUIRE(detail::get_alignment(md, builder.getInt32Ty()) == alignof(std::int32_t));
}

TEST_CASE("to_size_t")
{
    using namespace heyoka::detail;

    {
        llvm_state s;

        auto &builder = s.builder();
        auto &context = s.context();

        std::vector<llvm::Type *> fargs(1, builder.getInt32Ty());
        auto *lst = to_llvm_type<std::size_t>(context);
        auto *ft = llvm::FunctionType::get(lst, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &s.module());

        auto *in_val = f->args().begin();

        auto *bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        builder.CreateRet(to_size_t(s, in_val));

        s.optimise();

        s.compile();

        auto f_ptr = reinterpret_cast<std::size_t (*)(std::uint32_t)>(s.jit_lookup("test"));

        REQUIRE(f_ptr(std::numeric_limits<std::uint32_t>::max())
                == static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
    }

    {
        llvm_state s;

        auto &builder = s.builder();
        auto &context = s.context();

        auto *lst = to_llvm_type<std::size_t>(context);
        std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(lst),
                                        llvm::PointerType::getUnqual(builder.getInt32Ty())};
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &s.module());

        auto *out_val = f->args().begin();
        auto *in_val = f->args().begin() + 1;

        auto *bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        auto *ret = to_size_t(s, load_vector_from_memory(builder, builder.getInt32Ty(), in_val, 4));
        store_vector_to_memory(builder, out_val, ret);

        builder.CreateRetVoid();

        s.optimise();

        s.compile();

        auto f_ptr = reinterpret_cast<void (*)(std::size_t *, std::uint32_t *)>(s.jit_lookup("test"));

        std::uint32_t in[] = {std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(),
                              std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max()};
        std::size_t out[4];

        f_ptr(out, in);

        REQUIRE(out[0] == static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
        REQUIRE(out[1] == static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
        REQUIRE(out[2] == static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
        REQUIRE(out[3] == static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
    }

    {
        llvm_state s;

        auto &builder = s.builder();
        auto &context = s.context();

        std::vector<llvm::Type *> fargs(1, builder.getInt64Ty());
        auto *lst = to_llvm_type<std::size_t>(context);
        auto *ft = llvm::FunctionType::get(lst, fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &s.module());

        auto *in_val = f->args().begin();

        auto *bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        builder.CreateRet(to_size_t(s, in_val));

        s.optimise();

        s.compile();

        auto f_ptr = reinterpret_cast<std::size_t (*)(std::uint64_t)>(s.jit_lookup("test"));

        REQUIRE(f_ptr(std::numeric_limits<std::uint64_t>::max())
                == static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()));
    }

    {
        llvm_state s;

        auto &builder = s.builder();
        auto &context = s.context();

        auto *lst = to_llvm_type<std::size_t>(context);
        std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(lst),
                                        llvm::PointerType::getUnqual(builder.getInt64Ty())};
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &s.module());

        auto *out_val = f->args().begin();
        auto *in_val = f->args().begin() + 1;

        auto *bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        auto *ret = to_size_t(s, load_vector_from_memory(builder, builder.getInt64Ty(), in_val, 4));
        store_vector_to_memory(builder, out_val, ret);

        builder.CreateRetVoid();

        s.optimise();

        s.compile();

        auto f_ptr = reinterpret_cast<void (*)(std::size_t *, std::uint64_t *)>(s.jit_lookup("test"));

        std::uint64_t in[] = {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max(),
                              std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
        std::size_t out[4];

        f_ptr(out, in);

        REQUIRE(out[0] == static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()));
        REQUIRE(out[1] == static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()));
        REQUIRE(out[2] == static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()));
        REQUIRE(out[3] == static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()));
    }
}
