// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/tanh.hpp>
#include <heyoka/model/ffnn.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    create_logger();
    set_logger_level_trace();

    auto logger = spdlog::get("heyoka");

    unsigned nn_layer{};

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("nn_layer", po::value<unsigned>(&nn_layer)->default_value(10u),
                                                       "number of neurons per layer");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (nn_layer == 0u) {
        throw std::invalid_argument("The number of neurons per layer cannot be zero");
    }

    auto [x, y] = make_vars("x", "y");
    auto ffnn = model::ffnn(kw::inputs = {x, y}, kw::nn_hidden = {nn_layer, nn_layer, nn_layer}, kw::n_out = 2,
                            kw::activations = {heyoka::tanh, heyoka::tanh, heyoka::tanh, heyoka::tanh});

    auto dt = diff_tensors(ffnn, kw::diff_args = diff_args::params);
    auto dNdtheta = dt.get_jacobian();
}
