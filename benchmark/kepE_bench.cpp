// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <ios>
#include <iostream>
#include <random>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>

#include <fmt/core.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/kepE.hpp>
#include <stdexcept>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double ecc{};
    unsigned seed{};
    bool fast_math{};

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("ecc", po::value<double>(&ecc)->default_value(0.1),
                                                       "eccentricity")(
        "seed", po::value<unsigned>(&seed)->default_value(42u),
        "random seed")("fast-math", po::value<bool>(&fast_math)->default_value(true), "fast math mode");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (!std::isfinite(ecc) || ecc < 0 || ecc >= 1) {
        throw std::invalid_argument(fmt::format("Invalid eccentricity value: {}", ecc));
    }

    constexpr auto N = 1'000'000ul;

    std::cout << std::boolalpha;
    std::cout << "Eccentricity: " << ecc << '\n';
    std::cout << "fast_math   : " << fast_math << '\n';
    std::cout << "N           : " << N << "\n\n";

    // RNG setup.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> Mdist(0, 2 * boost::math::constants::pi<double>());

    // Data setup.
    std::vector<double> e_vec, M_vec, out_vec, out_vec_batch;
    e_vec.resize(N, ecc);
    M_vec.resize(N);
    out_vec.resize(N);
    out_vec_batch.resize(N);
    std::generate(M_vec.begin(), M_vec.end(), [&rng, &Mdist]() { return Mdist(rng); });

    // cfunc setup.
    auto [e, M] = make_vars("e", "M");

    llvm_state s{kw::fast_math = fast_math};
    const auto batch_size = recommended_simd_size<double>();
    add_cfunc<double>(s, "f_scalar", {kepE(e, M)}, {e, M});
    add_cfunc<double>(s, "f_batch", {kepE(e, M)}, {e, M}, kw::batch_size = batch_size);
    s.compile();

    auto *f_sc = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
        s.jit_lookup("f_scalar"));
    auto *f_ba
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("f_batch"));

    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // Scalar runtime.
    spdlog::stopwatch sw;

    for (auto i = 0ull; i < N; ++i) {
        double ins[] = {e_vec[i], M_vec[i]};
        f_sc(out_vec.data() + i, ins, nullptr, nullptr);
    }

    logger->trace("Scalar run took: {}s", sw);

    std::vector<double> batch_buffer(batch_size * 2ul);
    auto *batch_b_ptr = batch_buffer.data();

    sw.reset();

    for (auto i = 0ull; i < N - N % batch_size; i += batch_size) {
        std::copy(e_vec.data() + i, e_vec.data() + i + batch_size, batch_b_ptr);
        std::copy(M_vec.data() + i, M_vec.data() + i + batch_size, batch_b_ptr + batch_size);
        f_ba(out_vec_batch.data() + i, batch_b_ptr, nullptr, nullptr);
    }

    logger->trace("Batch run took: {}s", sw);

    std::cout.precision(16);
    for (auto i = 0u; i < 20u; ++i) {
        std::cout << out_vec[i] << " vs " << out_vec_batch[i] << '\n';
    }
}
