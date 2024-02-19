// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <fmt/core.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <oneapi/tbb/global_control.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/mascon.hpp>

using namespace heyoka;

template <typename T>
void run_benchmark(unsigned long long nevals, std::uint32_t batch_size, bool compact_mode)
{
    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // Buffers.
    std::vector<T> out_buffer(nevals), in_buffer(nevals * 6u, static_cast<T>(0.1));

    // Total number of particles.
    const std::uint32_t N = 200ull;

    // Generate random positions and mass values for the particles.
    std::mt19937 rng;
    std::uniform_real_distribution<double> rdist;

    auto gen = [&rng, &rdist]() { return rdist(rng); };

    std::vector<double> pos_vals, masses_vals;

    std::generate_n(std::back_inserter(pos_vals), N * 3u, gen);
    std::generate_n(std::back_inserter(masses_vals), N, gen);

    // Create the mascon energy expression.
    auto en = model::fixed_centres_energy(kw::positions = pos_vals, kw::masses = masses_vals);

    // Build the compiled function.
    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    auto cf0 = cfunc<T>({en}, {x, y, z, vx, vy, vz}, kw::compact_mode = compact_mode, kw::batch_size = batch_size);

    // Prepare the buffer views.
    typename cfunc<T>::out_2d out{out_buffer.data(), 1, nevals};
    typename cfunc<T>::in_2d in{in_buffer.data(), 6, nevals};

    spdlog::stopwatch sw;

    // Evaluate.
    cf0(out, in);

    logger->trace("Total elapsed time: {}", sw);
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    unsigned long long nevals{};
    std::string fp_type;
    unsigned nthreads{};
    std::uint32_t batch_size{};
    bool compact_mode = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "nevals", po::value<unsigned long long>(&nevals)->default_value(10'000'000ull), "number of evaluations")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"),
        "floating-point type")("nthreads", po::value<unsigned>(&nthreads)->default_value(0u), "number of threads")(
        "batch_size", po::value<std::uint32_t>(&batch_size)->default_value(0u), "batch size")("compact_mode",
                                                                                              "compact mode");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (nevals == 0u) {
        throw std::invalid_argument("nevals cannot be zero");
    }

    if (vm.count("compact_mode") != 0u) {
        compact_mode = true;
    }

    // Setup the number of threads to use. Zero means auto-deduced.
    std::optional<oneapi::tbb::global_control> tbb_gc;
    if (nthreads != 0u) {
        tbb_gc.emplace(oneapi::tbb::global_control::max_allowed_parallelism, nthreads);
    }

    if (fp_type == "double") {
        run_benchmark<double>(nevals, batch_size, compact_mode);
    } else if (fp_type == "float") {
        run_benchmark<float>(nevals, batch_size, compact_mode);
    } else if (fp_type == "long double") {
        run_benchmark<long double>(nevals, batch_size, compact_mode);
    } else {
        throw std::invalid_argument(fmt::format("Invalid fp type '{}'", fp_type));
    }
}
