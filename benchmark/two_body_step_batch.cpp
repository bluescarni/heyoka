// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <chrono>
#include <cstdint>
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

#include <heyoka/kw.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

template <typename T>
void run_bench(T tol, bool high_accuracy, std::uint32_t batch_size, bool compact_mode, bool fast_math)
{
    // NOTE: this setup mimics the 'simplest' test from rebound.
    auto sys = model::nbody(2, kw::masses = {1., 0.});

    // Generate the initial state/time vector for the batch integrator.
    std::vector<T> init_states(batch_size * 12u);
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        init_states[0u * batch_size + i] = 0;
        init_states[1u * batch_size + i] = 0;
        init_states[2u * batch_size + i] = 0;
        init_states[3u * batch_size + i] = 0;
        init_states[4u * batch_size + i] = 0;
        init_states[5u * batch_size + i] = 0;
        init_states[6u * batch_size + i] = 1;
        init_states[7u * batch_size + i] = 0;
        init_states[8u * batch_size + i] = 0;
        init_states[9u * batch_size + i] = 0;
        init_states[10u * batch_size + i] = 1;
        init_states[11u * batch_size + i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Init the batch integrator.
    auto tad = taylor_adaptive_batch<T>{sys,
                                        std::move(init_states),
                                        batch_size,
                                        kw::high_accuracy = high_accuracy,
                                        kw::tol = tol,
                                        kw::compact_mode = compact_mode,
                                        kw::fast_math = fast_math};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "ms\n";

    const std::vector<T> final_times(batch_size, T(10000));

    start = std::chrono::high_resolution_clock::now();

    tad.propagate_until(final_times);

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Integration time: " << elapsed << "μs\n";
    std::cout << "Integration time per batch element: " << elapsed / batch_size << "μs\n";
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type;
    bool high_accuracy = false;
    double tol;
    std::uint32_t batch_size;
    bool compact_mode = false;
    bool fast_math = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type")(
        "tol", po::value<double>(&tol)->default_value(0.), "tolerance (if 0, it will be the type's epsilon)")(
        "batch_size", po::value<std::uint32_t>(&batch_size)->default_value(1u), "batch size")(
        "high_accuracy", "enable high-accuracy mode")("compact_mode", "enable compact mode")("fast_math",
                                                                                             "enable fast math flags");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    // Validate the command-line arguments.
    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size cannot be zero");
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
        run_bench<double>(tol, high_accuracy, batch_size, compact_mode, fast_math);
    } else if (fp_type == "long double") {
        run_bench<long double>(tol, high_accuracy, batch_size, compact_mode, fast_math);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_bench<mppp::real128>(mppp::real128(tol), high_accuracy, batch_size, compact_mode, fast_math);
#endif
    } else {
        throw std::invalid_argument("Invalid floating-point type: '" + fp_type + "'");
    }
}
