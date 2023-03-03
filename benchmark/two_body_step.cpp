// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <chrono>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

template <typename T>
void run_bench(T tol, bool high_accuracy, bool compact_mode, bool fast_math, long long prec)
{
    warmup();

    // NOTE: this setup mimics the 'simplest' test from rebound.
    auto sys = model::nbody(2, kw::masses = {1., 0.});

    std::vector init_state{T(0), T(0), T(0), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1), T(0)};

    auto start = std::chrono::high_resolution_clock::now();

    auto tad = taylor_adaptive<T>{std::move(sys), std::move(init_state),           kw::high_accuracy = high_accuracy,
                                  kw::tol = tol,  kw::compact_mode = compact_mode, kw::fast_math = fast_math,
                                  kw::prec = prec};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "ms\n";

    start = std::chrono::high_resolution_clock::now();

    tad.propagate_until(T(10000));

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Integration time: " << elapsed << "μs\n";
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type;
    long long prec;
    bool high_accuracy = false;
    double tol;
    bool compact_mode = false;
    bool fast_math = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type")(
        "tol", po::value<double>(&tol)->default_value(0.), "tolerance (if 0, it will be the type's epsilon)")(
        "high_accuracy", "enable high-accuracy mode")("compact_mode", "enable compact mode")(
        "fast_math", "enable fast math flags")("prec", po::value<long long>(&prec)->default_value(0),
                                               "precision (used only in multiprecision mode)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("high_accuracy")) {
        high_accuracy = true;
    }

    if (vm.count("compact_mode")) {
        compact_mode = true;
    }

    if (vm.count("fast_math")) {
        fast_math = true;
    }

    if (fp_type == "double") {
        run_bench<double>(tol, high_accuracy, compact_mode, fast_math, prec);
    } else if (fp_type == "long double") {
        run_bench<long double>(tol, high_accuracy, compact_mode, fast_math, prec);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_bench<mppp::real128>(mppp::real128(tol), high_accuracy, compact_mode, fast_math, prec);
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (fp_type == "real") {
        run_bench<mppp::real>(mppp::real(tol), high_accuracy, compact_mode, fast_math, prec);
#endif
    } else {
        throw std::invalid_argument("Invalid floating-point type: '" + fp_type + "'");
    }
}
