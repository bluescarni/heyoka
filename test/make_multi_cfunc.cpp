// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/debug.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/model/sgp4.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

struct al_array_deleter {
    std::align_val_t al = std::align_val_t{0};
    void operator()(void *ptr) const
    {
        ::operator delete[](ptr, al);
    }
};

std::unique_ptr<char[], al_array_deleter> make_aligned_storage(std::size_t sz, std::size_t al)
{
    if (sz == 0u) {
        return {};
    } else {
#if defined(_MSC_VER)
        // MSVC workaround for this issue:
        // https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320

        // Allocate the raw memory.
        auto *buf = ::operator new[](sz, std::align_val_t{al});

        // Formally construct the bytes array.
        auto *ptr = ::new (buf) char[sz];

        // Constrcut and return the unique ptr.
        return std::unique_ptr<char[], al_array_deleter>{ptr, {.al = std::align_val_t{al}}};
#else
        return std::unique_ptr<char[], al_array_deleter>{new (std::align_val_t{al}) char[sz],
                                                         {.al = std::align_val_t{al}}};
#endif
    }
}

TEST_CASE("basic")
{
    auto [x, y] = make_vars("x", "y");

    // Scalar test.
    for (auto opt_level : {0u, 1u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        auto [ms, dc, sa] = detail::make_multi_cfunc<double>(tplt, "test", {x + y + heyoka::time, x - y - par[0]},
                                                             {x, y}, 1, false, false, 0);

        REQUIRE(sa.size() == 1u);

        ms.compile();

        {
            // Scalar unstrided.
            auto *cf_s_u = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *)>(
                ms.jit_lookup("test.unstrided.batch_size_1"));

            std::vector<double> ins{1, 2}, outs(2u), pars = {-0.25}, time{0.5};

            const auto [sz, al] = sa[0];
            REQUIRE(al == alignof(double));

            auto ext_storage = make_aligned_storage(sz, al);

            cf_s_u(outs.data(), ins.data(), pars.data(), time.data(), ext_storage.get());

            REQUIRE(outs[0] == ins[0] + ins[1] + time[0]);
            REQUIRE(outs[1] == ins[0] - ins[1] - pars[0]);
        }

        {
            // Scalar strided.
            auto *cf_s_s = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *,
                                                     std::size_t)>(ms.jit_lookup("test.strided.batch_size_1"));

            // Stride value of 3.
            std::vector<double> ins{1, 0, 0, 2, 0, 0}, outs(6u), pars = {-0.25, 0, 0}, time{0.5, 0, 0};

            const auto [sz, al] = sa[0];
            REQUIRE(al == alignof(double));

            auto ext_storage = make_aligned_storage(sz, al);

            cf_s_s(outs.data(), ins.data(), pars.data(), time.data(), ext_storage.get(), 3);

            REQUIRE(outs[0] == ins[0] + ins[3] + time[0]);
            REQUIRE(outs[3] == ins[0] - ins[3] - pars[0]);
        }
    }

    // Batch test.
    for (auto opt_level : {0u, 1u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        auto [ms, dc, sa] = detail::make_multi_cfunc<double>(tplt, "test", {x + y + heyoka::time, x - y - par[0]},
                                                             {x, y}, 2, false, false, 0);

        REQUIRE(sa.size() == 2u);

        ms.compile();

        {
            // Scalar unstrided.
            auto *cf_s_u = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *)>(
                ms.jit_lookup("test.unstrided.batch_size_1"));

            std::vector<double> ins{1, 2}, outs(2u), pars = {-0.25}, time{0.5};

            const auto [sz, al] = sa[0];
            REQUIRE(al == alignof(double));

            auto ext_storage = make_aligned_storage(sz, al);

            cf_s_u(outs.data(), ins.data(), pars.data(), time.data(), ext_storage.get());

            REQUIRE(outs[0] == ins[0] + ins[1] + time[0]);
            REQUIRE(outs[1] == ins[0] - ins[1] - pars[0]);
        }

        {
            // Scalar strided.
            auto *cf_s_s = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *,
                                                     std::size_t)>(ms.jit_lookup("test.strided.batch_size_1"));

            // Stride value of 3.
            std::vector<double> ins{1, 0, 0, 2, 0, 0}, outs(6u), pars = {-0.25, 0, 0}, time{0.5, 0, 0};

            const auto [sz, al] = sa[0];
            REQUIRE(al == alignof(double));

            auto ext_storage = make_aligned_storage(sz, al);

            cf_s_s(outs.data(), ins.data(), pars.data(), time.data(), ext_storage.get(), 3);

            REQUIRE(outs[0] == ins[0] + ins[3] + time[0]);
            REQUIRE(outs[3] == ins[0] - ins[3] - pars[0]);
        }

        {
            // Batch strided.
            auto *cf_b_s = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *,
                                                     std::size_t)>(ms.jit_lookup("test.strided.batch_size_2"));

            // Stride value of 3.
            std::vector<double> ins{1, 1.1, 0, 2, 2.1, 0}, outs(6u), pars = {-0.25, -0.26, 0}, time{0.5, 0.51, 0};

            const auto [sz, al] = sa[1];
            REQUIRE(al >= alignof(double));

            auto ext_storage = make_aligned_storage(sz, al);

            cf_b_s(outs.data(), ins.data(), pars.data(), time.data(), ext_storage.get(), 3);

            REQUIRE(outs[0] == ins[0] + ins[3] + time[0]);
            REQUIRE(outs[1] == ins[1] + ins[4] + time[1]);
            REQUIRE(outs[3] == ins[0] - ins[3] - pars[0]);
            REQUIRE(outs[4] == ins[1] - ins[4] - pars[1]);
        }

        REQUIRE_THROWS_AS(ms.jit_lookup("test.unstrided.batch_size_2"), std::invalid_argument);
    }
}

TEST_CASE("sgp4")
{
    detail::edb_disabler ed;

    const auto inputs = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");

    auto cf = cfunc<double>(model::sgp4(), inputs);

    llvm_state tplt;

    auto [ms, dc, sa] = detail::make_multi_cfunc<double>(tplt, "test", model::sgp4(),
                                                         std::vector(inputs.begin(), inputs.end()), 1, false, false, 0);

    REQUIRE(sa.size() == 1u);

    ms.compile();

    auto *cf_s_u = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *)>(
        ms.jit_lookup("test.unstrided.batch_size_1"));

    const auto revday2radmin = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 1440.; };
    const auto deg2rad = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 360.; };

    std::vector<double> ins = {revday2radmin(15.50103472202482),
                               0.0007417,
                               deg2rad(51.6439),
                               deg2rad(211.2001),
                               deg2rad(17.6667),
                               deg2rad(85.6398),
                               .38792e-4,
                               0.},
                        outs1(7u), outs2(7u);

    const auto [sz, al] = sa[0];
    REQUIRE(al == alignof(double));

    auto ext_storage = make_aligned_storage(sz, al);

    cf_s_u(outs1.data(), ins.data(), nullptr, nullptr, ext_storage.get());
    cf(outs2, ins);

    for (auto i = 0u; i < 7u; ++i) {
        REQUIRE(outs1[i] == approximately(outs2[i]));
    }
}

// N-body with fixed masses.
TEST_CASE("nbody")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<double> outs, ins;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            llvm_state tplt{kw::opt_level = opt_level};

            outs.resize(36u * batch_size);
            ins.resize(36u * batch_size);

            std::generate(ins.begin(), ins.end(), gen);

            std::vector<expression> vars;
            std::ranges::transform(sys, std::back_inserter(vars), [](const auto &p) { return p.first; });
            std::ranges::sort(vars, std::less<expression>{});

            auto [ms, dc, sa] = detail::make_multi_cfunc<double>(tplt, "test", exs, vars, batch_size, false, false, 0);

            ms.compile();

            if (batch_size == 1u) {
                REQUIRE(sa.size() == 1u);

                const auto [sz, al] = sa[0];
                REQUIRE(al == alignof(double));

                auto *cf_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *)>(
                        ms.jit_lookup("test.unstrided.batch_size_1"));

                auto ext_storage = make_aligned_storage(sz, al);

                cf_ptr(outs.data(), ins.data(), nullptr, nullptr, ext_storage.get());
            } else {
                REQUIRE(sa.size() == 2u);

                const auto [sz, al] = sa[1];
                REQUIRE(al >= alignof(double));

                auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *,
                                                         void *, std::size_t)>(
                    ms.jit_lookup(fmt::format("test.strided.batch_size_{}", batch_size)));

                auto ext_storage = make_aligned_storage(sz, al);

                cf_ptr(outs.data(), ins.data(), nullptr, nullptr, ext_storage.get(), batch_size);
            }

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    // x_i' == vx_i.
                    REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], 100.));
                    // y_i' == vy_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                            == approximately(ins[i * batch_size + batch_size * 6u + j], 100.));
                    // z_i' == vz_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                            == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], 100.));

                    // Accelerations.
                    auto acc_x = 0., acc_y = 0., acc_z = 0.;

                    const auto xi = ins[18u * batch_size + i * batch_size + j];
                    const auto yi = ins[24u * batch_size + i * batch_size + j];
                    const auto zi = ins[30u * batch_size + i * batch_size + j];

                    for (auto k = 0u; k < 6u; ++k) {
                        if (k == i) {
                            continue;
                        }

                        const auto xk = ins[18u * batch_size + k * batch_size + j];
                        const auto dx = xk - xi;

                        const auto yk = ins[24u * batch_size + k * batch_size + j];
                        const auto dy = yk - yi;

                        const auto zk = ins[30u * batch_size + k * batch_size + j];
                        const auto dz = zk - zi;

                        const auto rm3 = std::pow(dx * dx + dy * dy + dz * dz, -3 / 2.);

                        acc_x += dx * G * masses[k] * rm3;
                        acc_y += dy * G * masses[k] * rm3;
                        acc_z += dz * G * masses[k] * rm3;
                    }

                    REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, 100.));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, 100.));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, 100.));
                }
            }
        }
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("nbody mp")
{
    using std::pow;

    const auto prec = 237u;

    auto masses
        = std::vector{mppp::real{1.00000597682, prec}, mppp::real{1 / 1047.355, prec}, mppp::real{1 / 3501.6, prec},
                      mppp::real{1 / 22869., prec},    mppp::real{1 / 19314., prec},   mppp::real{7.4074074e-09, prec}};

    const auto G = mppp::real{0.01720209895 * 0.01720209895 * 365 * 365, prec};

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<mppp::real> outs, ins;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&]() { return mppp::real{rdist(rng), static_cast<int>(prec)}; };

    const auto batch_size = 1u;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        outs.resize(36u * batch_size);
        ins.resize(36u * batch_size);

        std::generate(ins.begin(), ins.end(), gen);
        std::generate(outs.begin(), outs.end(), gen);

        std::vector<expression> vars;
        std::ranges::transform(sys, std::back_inserter(vars), [](const auto &p) { return p.first; });
        std::ranges::sort(vars, std::less<expression>{});

        auto [ms, dc, sa] = detail::make_multi_cfunc<mppp::real>(tplt, "test", exs, vars, 1, false, false, prec);

        ms.compile();

        REQUIRE(sa.size() == 1u);

        const auto [sz, al] = sa[0];
        REQUIRE(al == alignof(mppp::real));

        auto *cf_ptr
            = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *,
                                        void *)>(ms.jit_lookup("test.unstrided.batch_size_1"));

        auto ext_storage = make_aligned_storage(sz, al);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr, ext_storage.get());

        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                // x_i' == vx_i.
                REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], mppp::real{100.}));
                // y_i' == vy_i.
                REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                        == approximately(ins[i * batch_size + batch_size * 6u + j], mppp::real{100.}));
                // z_i' == vz_i.
                REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                        == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], mppp::real{100.}));

                // Accelerations.
                mppp::real acc_x{0., prec}, acc_y{0., prec}, acc_z{0., prec};

                const auto xi = ins[18u * batch_size + i * batch_size + j];
                const auto yi = ins[24u * batch_size + i * batch_size + j];
                const auto zi = ins[30u * batch_size + i * batch_size + j];

                for (auto k = 0u; k < 6u; ++k) {
                    if (k == i) {
                        continue;
                    }

                    const auto xk = ins[18u * batch_size + k * batch_size + j];
                    const auto dx = xk - xi;

                    const auto yk = ins[24u * batch_size + k * batch_size + j];
                    const auto dy = yk - yi;

                    const auto zk = ins[30u * batch_size + k * batch_size + j];
                    const auto dz = zk - zi;

                    const auto rm3 = pow(dx * dx + dy * dy + dz * dz, mppp::real{-3 / 2., prec});

                    acc_x += dx * G * masses[k] * rm3;
                    acc_y += dy * G * masses[k] * rm3;
                    acc_z += dz * G * masses[k] * rm3;
                }

                REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, mppp::real{100.}));
                REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, mppp::real{100.}));
                REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, mppp::real{100.}));
            }
        }
    }
}

#endif

// N-body with parametric masses.
TEST_CASE("nbody par")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = model::nbody(6, kw::Gconst = G, kw::masses = {par[0], par[1], par[2], par[3], par[4], par[5]});
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<double> outs, ins, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            llvm_state tplt{kw::opt_level = opt_level};

            outs.resize(36u * batch_size);
            ins.resize(36u * batch_size);
            pars.resize(6u * batch_size);

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    pars[i * batch_size + j] = masses[i];
                }
            }

            std::generate(ins.begin(), ins.end(), gen);

            std::vector<expression> vars;
            std::ranges::transform(sys, std::back_inserter(vars), [](const auto &p) { return p.first; });
            std::ranges::sort(vars, std::less<expression>{});

            auto [ms, dc, sa] = detail::make_multi_cfunc<double>(tplt, "test", exs, vars, batch_size, false, false, 0);

            ms.compile();

            REQUIRE(((batch_size == 1u && sa.size() == 1u) || (batch_size > 1u && sa.size() == 2u)));

            // Fetch the strided function for the current batch size.
            auto *cfs_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *,
                                                      std::size_t)>(
                ms.jit_lookup(fmt::format("test.strided.batch_size_{}", batch_size)));

            const auto [sz_b, al_b] = sa[batch_size == 1u ? 0 : 1];
            auto ext_storage_sb = make_aligned_storage(sz_b, al_b);

            if (batch_size == 1u) {
                REQUIRE(sa.size() == 1u);

                auto *cf_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *)>(
                        ms.jit_lookup("test.unstrided.batch_size_1"));

                const auto [sz, al] = sa[0];

                auto ext_storage = make_aligned_storage(sz, al);

                cf_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage.get());
            } else {
                REQUIRE(sa.size() == 2u);

                cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage_sb.get(), batch_size);
            }

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    // x_i' == vx_i.
                    REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], 100.));
                    // y_i' == vy_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                            == approximately(ins[i * batch_size + batch_size * 6u + j], 100.));
                    // z_i' == vz_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                            == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], 100.));

                    // Accelerations.
                    auto acc_x = 0., acc_y = 0., acc_z = 0.;

                    const auto xi = ins[18u * batch_size + i * batch_size + j];
                    const auto yi = ins[24u * batch_size + i * batch_size + j];
                    const auto zi = ins[30u * batch_size + i * batch_size + j];

                    for (auto k = 0u; k < 6u; ++k) {
                        if (k == i) {
                            continue;
                        }

                        const auto xk = ins[18u * batch_size + k * batch_size + j];
                        const auto dx = xk - xi;

                        const auto yk = ins[24u * batch_size + k * batch_size + j];
                        const auto dy = yk - yi;

                        const auto zk = ins[30u * batch_size + k * batch_size + j];
                        const auto dz = zk - zi;

                        const auto rm3 = std::pow(dx * dx + dy * dy + dz * dz, -3 / 2.);

                        acc_x += dx * G * masses[k] * rm3;
                        acc_y += dy * G * masses[k] * rm3;
                        acc_z += dz * G * masses[k] * rm3;
                    }

                    REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, 1000.));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, 1000.));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, 1000.));
                }
            }

            // Run the test on the strided function too.
            const std::size_t extra_stride = 3;
            outs.resize(36u * (batch_size + extra_stride));
            ins.resize(36u * (batch_size + extra_stride));
            pars.resize(6u * (batch_size + extra_stride));

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    pars[i * (batch_size + extra_stride) + j] = masses[i];
                }
            }

            std::generate(ins.begin(), ins.end(), gen);

            cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage_sb.get(), batch_size + extra_stride);

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    // x_i' == vx_i.
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + j]
                            == approximately(ins[i * (batch_size + extra_stride) + j], 100.));
                    // y_i' == vy_i.
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) + j]
                            == approximately(
                                ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u + j], 100.));
                    // z_i' == vz_i.
                    REQUIRE(
                        outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 2u + j]
                        == approximately(
                            ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u * 2u + j], 100.));

                    // Accelerations.
                    auto acc_x = 0., acc_y = 0., acc_z = 0.;

                    const auto xi = ins[18u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                    const auto yi = ins[24u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                    const auto zi = ins[30u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];

                    for (auto k = 0u; k < 6u; ++k) {
                        if (k == i) {
                            continue;
                        }

                        const auto xk = ins[18u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                        const auto dx = xk - xi;

                        const auto yk = ins[24u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                        const auto dy = yk - yi;

                        const auto zk = ins[30u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                        const auto dz = zk - zi;

                        const auto rm3 = std::pow(dx * dx + dy * dy + dz * dz, -3 / 2.);

                        acc_x += dx * G * masses[k] * rm3;
                        acc_y += dy * G * masses[k] * rm3;
                        acc_z += dz * G * masses[k] * rm3;
                    }

                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 3u + j]
                            == approximately(acc_x, 1000.));
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 4u + j]
                            == approximately(acc_y, 1000.));
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 5u + j]
                            == approximately(acc_z, 1000.));
                }
            }
        }
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("nbody par mp")
{
    using std::pow;

    const auto prec = 237u;

    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = mppp::real{0.01720209895 * 0.01720209895 * 365 * 365, prec};

    auto sys = model::nbody(6, kw::Gconst = G, kw::masses = {par[0], par[1], par[2], par[3], par[4], par[5]});
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<mppp::real> outs, ins, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&]() { return mppp::real{rdist(rng), static_cast<int>(prec)}; };

    const auto batch_size = 1u;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        outs.resize(36u * batch_size);
        ins.resize(36u * batch_size);
        pars.resize(6u * batch_size);

        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                pars[i * batch_size + j] = mppp::real{masses[i], prec};
            }
        }

        std::generate(ins.begin(), ins.end(), gen);
        std::generate(outs.begin(), outs.end(), gen);

        std::vector<expression> vars;
        std::ranges::transform(sys, std::back_inserter(vars), [](const auto &p) { return p.first; });
        std::ranges::sort(vars, std::less<expression>{});

        auto [ms, dc, sa] = detail::make_multi_cfunc<mppp::real>(tplt, "test", exs, vars, 1, false, false, prec);

        ms.compile();

        REQUIRE(sa.size() == 1u);

        const auto [sz, al] = sa[0];
        REQUIRE(al == alignof(mppp::real));

        auto *cf_ptr
            = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *,
                                        void *)>(ms.jit_lookup("test.unstrided.batch_size_1"));

        auto *cfs_ptr
            = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *,
                                        void *, std::size_t)>(ms.jit_lookup("test.strided.batch_size_1"));

        auto ext_storage = make_aligned_storage(sz, al);

        cf_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage.get());

        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                // x_i' == vx_i.
                REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], mppp::real{100.}));
                // y_i' == vy_i.
                REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                        == approximately(ins[i * batch_size + batch_size * 6u + j], mppp::real{100.}));
                // z_i' == vz_i.
                REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                        == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], mppp::real{100.}));

                // Accelerations.
                mppp::real acc_x{0., prec}, acc_y{0., prec}, acc_z{0., prec};

                const auto xi = ins[18u * batch_size + i * batch_size + j];
                const auto yi = ins[24u * batch_size + i * batch_size + j];
                const auto zi = ins[30u * batch_size + i * batch_size + j];

                for (auto k = 0u; k < 6u; ++k) {
                    if (k == i) {
                        continue;
                    }

                    const auto xk = ins[18u * batch_size + k * batch_size + j];
                    const auto dx = xk - xi;

                    const auto yk = ins[24u * batch_size + k * batch_size + j];
                    const auto dy = yk - yi;

                    const auto zk = ins[30u * batch_size + k * batch_size + j];
                    const auto dz = zk - zi;

                    const auto rm3 = pow(dx * dx + dy * dy + dz * dz, mppp::real{-3 / 2., prec});

                    acc_x += dx * G * masses[k] * rm3;
                    acc_y += dy * G * masses[k] * rm3;
                    acc_z += dz * G * masses[k] * rm3;
                }

                REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, mppp::real{1000.}));
                REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, mppp::real{1000.}));
                REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, mppp::real{1000.}));
            }
        }

        // Run the test on the strided function too.
        const std::size_t extra_stride = 3;
        outs.resize(36u * (batch_size + extra_stride));
        ins.resize(36u * (batch_size + extra_stride));
        pars.resize(6u * (batch_size + extra_stride));

        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                pars[i * (batch_size + extra_stride) + j] = mppp::real{masses[i], prec};
            }
        }

        std::generate(ins.begin(), ins.end(), gen);
        std::generate(outs.begin(), outs.end(), gen);

        cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage.get(), batch_size + extra_stride);

        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = 0u; j < batch_size; ++j) {
                // x_i' == vx_i.
                REQUIRE(outs[i * (batch_size + extra_stride) * 6u + j]
                        == approximately(ins[i * (batch_size + extra_stride) + j], mppp::real{100.}));
                // y_i' == vy_i.
                REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) + j]
                        == approximately(ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u + j],
                                         mppp::real{100.}));
                // z_i' == vz_i.
                REQUIRE(
                    outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 2u + j]
                    == approximately(ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u * 2u + j],
                                     mppp::real{100.}));

                // Accelerations.
                mppp::real acc_x{0., prec}, acc_y{0., prec}, acc_z{0., prec};

                const auto xi = ins[18u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                const auto yi = ins[24u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                const auto zi = ins[30u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];

                for (auto k = 0u; k < 6u; ++k) {
                    if (k == i) {
                        continue;
                    }

                    const auto xk = ins[18u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                    const auto dx = xk - xi;

                    const auto yk = ins[24u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                    const auto dy = yk - yi;

                    const auto zk = ins[30u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                    const auto dz = zk - zi;

                    const auto rm3 = pow(dx * dx + dy * dy + dz * dz, mppp::real{-3 / 2., prec});

                    acc_x += dx * G * masses[k] * rm3;
                    acc_y += dy * G * masses[k] * rm3;
                    acc_z += dz * G * masses[k] * rm3;
                }

                REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 3u + j]
                        == approximately(acc_x, mppp::real{1000.}));
                REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 4u + j]
                        == approximately(acc_y, mppp::real{1000.}));
                REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 5u + j]
                        == approximately(acc_z, mppp::real{1000.}));
            }
        }
    }
}

#endif

// A test in which all outputs are equal to numbers or params.
TEST_CASE("numparams")
{
    std::vector<double> outs, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            llvm_state tplt{kw::opt_level = opt_level};

            outs.resize(2u * batch_size);
            pars.resize(batch_size);

            std::generate(pars.begin(), pars.end(), gen);

            auto [ms, dc, sa]
                = detail::make_multi_cfunc<double>(tplt, "test", {1_dbl, par[0]}, {}, batch_size, false, false, 0);

            REQUIRE(((batch_size == 1u && sa.size() == 1u) || (batch_size > 1u && sa.size() == 2u)));

            ms.compile();

            auto *cfs_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *,
                                                      std::size_t)>(
                ms.jit_lookup(fmt::format("test.strided.batch_size_{}", batch_size)));

            const auto [sz_b, al_b] = sa[batch_size == 1u ? 0 : 1];
            auto ext_storage_sb = make_aligned_storage(sz_b, al_b);

            if (batch_size == 1u) {
                REQUIRE(sa.size() == 1u);

                auto *cf_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *)>(
                        ms.jit_lookup("test.unstrided.batch_size_1"));

                const auto [sz, al] = sa[0];

                auto ext_storage = make_aligned_storage(sz, al);

                cf_ptr(outs.data(), nullptr, pars.data(), nullptr, ext_storage.get());
            } else {
                REQUIRE(sa.size() == 2u);

                cfs_ptr(outs.data(), nullptr, pars.data(), nullptr, ext_storage_sb.get(), batch_size);
            }

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(outs[j] == 1);
                REQUIRE(outs[j + batch_size] == pars[j]);
            }

            // Run the test on the strided function too.
            const std::size_t extra_stride = 3;
            outs.resize(2u * (batch_size + extra_stride));
            pars.resize(batch_size + extra_stride);

            std::generate(pars.begin(), pars.end(), gen);

            cfs_ptr(outs.data(), nullptr, pars.data(), nullptr, ext_storage_sb.get(), batch_size + extra_stride);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(outs[j] == 1);
                REQUIRE(outs[j + batch_size + extra_stride] == pars[j]);
            }
        }
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("numparams mp")
{
    const auto prec = 237u;

    const auto batch_size = 1u;

    std::vector<mppp::real> outs, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&]() { return mppp::real{rdist(rng), static_cast<int>(prec)}; };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        outs.resize(4u * batch_size);
        pars.resize(2u * batch_size);

        std::generate(pars.begin(), pars.end(), gen);
        std::generate(outs.begin(), outs.end(), gen);

        auto [ms, dc, sa] = detail::make_multi_cfunc<mppp::real>(tplt, "test", {1_dbl, par[0], par[1], -2_dbl}, {}, 1,
                                                                 false, false, prec);

        ms.compile();

        REQUIRE(sa.size() == 1u);

        const auto [sz, al] = sa[0];
        REQUIRE(al == alignof(mppp::real));

        auto *cf_ptr
            = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *,
                                        void *)>(ms.jit_lookup("test.unstrided.batch_size_1"));

        auto *cfs_ptr
            = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *,
                                        void *, std::size_t)>(ms.jit_lookup("test.strided.batch_size_1"));

        auto ext_storage = make_aligned_storage(sz, al);

        cf_ptr(outs.data(), nullptr, pars.data(), nullptr, ext_storage.get());

        for (auto j = 0u; j < batch_size; ++j) {
            REQUIRE(outs[j] == 1);
            REQUIRE(outs[j + batch_size] == pars[j]);
            REQUIRE(outs[j + 2u * batch_size] == pars[j + 1u]);
            REQUIRE(outs[j + 3u * batch_size] == -2);
        }

        // Run the test on the strided function too.
        const std::size_t extra_stride = 3;
        outs.resize(4u * (batch_size + extra_stride));
        pars.resize(2u * (batch_size + extra_stride));

        std::generate(pars.begin(), pars.end(), gen);
        std::generate(outs.begin(), outs.end(), gen);

        cfs_ptr(outs.data(), nullptr, pars.data(), nullptr, ext_storage.get(), batch_size + extra_stride);

        for (auto j = 0u; j < batch_size; ++j) {
            REQUIRE(outs[j] == 1);
            REQUIRE(outs[j + batch_size + extra_stride] == pars[j]);
            REQUIRE(outs[j + 2u * (batch_size + extra_stride)] == pars[j + batch_size + extra_stride]);
            REQUIRE(outs[j + 3u * (batch_size + extra_stride)] == -2);
        }
    }
}

#endif

// Test for stride values under the batch size.
TEST_CASE("bogus stride")
{
    std::vector<double> outs, ins, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    auto [x, y, z] = make_vars("x", "y", "z");

    for (auto batch_size : {1u, 2u, 4u, 5u}) {
        llvm_state tplt;

        outs.resize(2u * batch_size);
        ins.resize(3u * batch_size);
        pars.resize(2u * batch_size);

        std::generate(ins.begin(), ins.end(), gen);
        std::generate(pars.begin(), pars.end(), gen);

        auto [ms, dc, sa] = detail::make_multi_cfunc<double>(tplt, "test", {x + 2_dbl * y + par[0] * z, par[1] - x * y},
                                                             {x, y, z}, batch_size, false, false, 0);

        ms.compile();

        REQUIRE(((batch_size == 1u && sa.size() == 1u) || (batch_size > 1u && sa.size() == 2u)));

        auto *cfs_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, void *, std::size_t)>(
                ms.jit_lookup(fmt::format("test.strided.batch_size_{}", batch_size)));

        const auto [sz_b, al_b] = sa[batch_size == 1u ? 0 : 1];
        auto ext_storage_sb = make_aligned_storage(sz_b, al_b);

        cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage_sb.get(), batch_size - 1u);

        if (batch_size > 1u) {
            for (auto j = 0u; j < batch_size - 1u; ++j) {
                REQUIRE(outs[j]
                        == approximately(ins[j] + 2. * ins[(batch_size - 1u) + j]
                                             + pars[j] * ins[(batch_size - 1u) * 2u + j],
                                         100.));
                REQUIRE(outs[(batch_size - 1u) + j]
                        == approximately(pars[(batch_size - 1u) + j] - ins[j] * ins[(batch_size - 1u) + j], 100.));
            }

            cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, ext_storage_sb.get(), 0);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(outs[j] == approximately(pars[j] - ins[j] * ins[j], 100.));
            }
        } else {
            REQUIRE(outs[0] == approximately(pars[0] - ins[0] * ins[0], 100.));
        }
    }
}

TEST_CASE("failure modes")
{
    using Catch::Matchers::Message;

    REQUIRE_THROWS_MATCHES(
        detail::make_multi_cfunc<double>(llvm_state{}, "cfunc", {1_dbl, par[0]}, {}, 0, false, false, 0),
        std::invalid_argument, Message("The batch size of a compiled function cannot be zero"));

    REQUIRE_THROWS_MATCHES(
        detail::make_multi_cfunc<double>(llvm_state{}, "cfunc", {1_dbl, par[0]}, {}, 1, false, true, 0),
        std::invalid_argument, Message("Parallel mode has not been implemented yet"));

#if defined(HEYOKA_ARCH_PPC)

    REQUIRE_THROWS_MATCHES(
        detail::make_multi_cfunc<long double>(llvm_state{}, "cfunc", {1_dbl, par[0]}, {}, 1, false, false, 0),
        not_implemented_error, Message("'long double' computations are not supported on PowerPC"));

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE_THROWS_MATCHES(
        detail::make_multi_cfunc<mppp::real>(llvm_state{}, "cfunc", {1_dbl, par[0]}, {}, 1, false, false, 0),
        std::invalid_argument,
        Message(fmt::format("An invalid precision value of 0 was passed to make_multi_cfunc() (the "
                            "value must be in the [{}, {}] range)",
                            mppp::real_prec_min(), mppp::real_prec_max())));

#endif

    REQUIRE_THROWS_MATCHES(detail::make_multi_cfunc<double>(llvm_state{}, "", {1_dbl, par[0]}, {}, 1, false, false, 0),
                           std::invalid_argument,
                           Message("A non-empty function name is required when invoking make_multi_cfunc()"));
}
