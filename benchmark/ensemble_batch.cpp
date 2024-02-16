// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::size_t niter{};
    std::uint32_t batch_size{};
    double final_time{};

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("nevals", po::value<std::size_t>(&niter)->default_value(8),
                                                       "number of iterations")(
        "batch_size", po::value<std::uint32_t>(&batch_size)->default_value(recommended_simd_size<double>()),
        "batch size")("final_time", po::value<double>(&final_time)->default_value(3e5), "final integration time");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (niter == 0u) {
        throw std::invalid_argument("niter cannot be zero");
    }

    if (batch_size == 0u) {
        throw std::invalid_argument("batch_size cannot be zero");
    }

    if (!std::isfinite(final_time) || final_time <= 0) {
        throw std::invalid_argument("final_time must be finite and positive");
    }

    create_logger();
    auto logger = spdlog::get("heyoka");

    set_logger_level_trace();

    const std::vector masses = {1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    const std::vector ic = {-4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6,
                            +6.69048890636161e-6 * 365, -6.33922479583593e-6 * 365, -3.13202145590767e-9 * 365,
                            // Jupiter.
                            +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2,
                            -5.59797969310664e-3 * 365, +5.51815399480116e-3 * 365, -2.66711392865591e-6 * 365,
                            // Saturn.
                            +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1,
                            -4.17354020307064e-3 * 365, +3.99723751748116e-3 * 365, +1.67206320571441e-5 * 365,
                            // Uranus.
                            +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1,
                            -3.25884806151064e-3 * 365, +2.06438412905916e-3 * 365, -2.17699042180559e-5 * 365,
                            // Neptune.
                            -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1,
                            -2.17471785045538e-4 * 365, -3.11361111025884e-3 * 365, +3.58344705491441e-5 * 365,
                            // Pluto.
                            -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0,
                            -1.76936577252484e-3 * 365, -2.06720938381724e-3 * 365, +6.58091931493844e-4 * 365};

    const auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);

    auto ta = taylor_adaptive(sys, ic, kw::high_accuracy = true, kw::tol = 1e-18);

    spdlog::stopwatch sw;

    ta.propagate_until(final_time);

    logger->trace("Serial scalar baseline: {}", sw);

    ta = taylor_adaptive(sys, ic, kw::high_accuracy = true, kw::tol = 1e-18);

    sw.reset();

    ensemble_propagate_until<double>(
        ta, final_time, niter, [](auto t, auto) { return t; }, kw::high_accuracy = true, kw::tol = 1e-18);

    logger->trace("Ensemble: {}", sw);

    std::vector<double> ic_batch(36u * batch_size);
    for (auto i = 0u; i < 36u; ++i) {
        std::ranges::fill(ic_batch.data() + i * batch_size, ic_batch.data() + (i + 1u) * batch_size, ic[i]);
    }

    auto tab = taylor_adaptive_batch(sys, ic_batch, batch_size, kw::high_accuracy = true, kw::tol = 1e-18);

    sw.reset();

    ensemble_propagate_until_batch<double>(
        tab, final_time, niter, [](auto t, auto) { return t; }, kw::high_accuracy = true, kw::tol = 1e-18);

    logger->trace("Ensemble+batch: {}", sw);
}
