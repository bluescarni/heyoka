// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <iostream>
#include <random>

#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

using namespace heyoka;

// The seed is fixed as to get always the same expression
static std::mt19937 gen(12345u);

std::vector<std::vector<double>> random_args_vv(unsigned N, unsigned n)
{
    std::uniform_real_distribution<double> rngm11(-1, 1.);
    std::vector<std::vector<double>> retval(N, std::vector<double>(n, 0u));
    for (auto &vec : retval) {
        for (auto &el : vec) {
            el = rngm11(gen);
        }
    }
    return retval;
}

std::unordered_map<std::string, std::vector<double>> vv_to_dv(const std::vector<std::vector<double>> &in)
{
    std::unordered_map<std::string, std::vector<double>> retval;
    std::vector<double> x_vec(in.size()), y_vec(in.size());
    for (decltype(in.size()) i = 0u; i < in.size(); ++i) {
        x_vec[i] = in[i][0];
        y_vec[i] = in[i][1];
    }
    retval["x"] = x_vec;
    retval["y"] = y_vec;
    return retval;
}

std::vector<std::unordered_map<std::string, double>> vv_to_vd(const std::vector<std::vector<double>> &in)
{
    std::vector<std::unordered_map<std::string, double>> retval(in.size());
    for (decltype(in.size()) i = 0u; i < in.size(); ++i) {
        retval[i]["x"] = in[i][0];
        retval[i]["y"] = in[i][1];
    }
    return retval;
}

// ------------------------------------ Main --------------------------------------
using namespace std::chrono;
int main()
{
    std::random_device rd;
    detail::random_engine_type engine(rd());
    // Here we define the type of expression (two variables, default choices for the operators)
    expression_generator generator({"x", "y"}, engine());
    auto ex = generator(2, 4);
    // uncomment for this expression
    // auto ex = "x"_var * "x"_var + "y"_var + "y"_var * "y"_var - "y"_var * "x"_var;
    std::cout << "ex: " << ex << "\n";

    unsigned N = 10000u;
    unsigned N_batches = 50u;
    unsigned batch_size = N / N_batches;

    auto args_vv = random_args_vv(N, 3u);

    //// 1 - we time the function call from evaluate
    auto args_vd = vv_to_vd(args_vv);
    auto start = high_resolution_clock::now();
    for (auto &args : args_vd) {
        auto res = eval_dbl(ex, args);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of evaluations per second (tree): " << 1. / (static_cast<double>(duration.count()) / N)
              << "M\n";

    //// 2 - we time the function call from evaluate_batch
    auto args_dv = vv_to_dv(args_vv);
    std::vector<double> out(10000, 0.12345);
    start = high_resolution_clock::now();
    eval_batch_dbl(out, ex, args_dv);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of evaluations per second (tree in one batch): "
              << 1. / (static_cast<double>(duration.count()) / N) << "M\n";

    //// 5 - we time the function call from evaluate_batch (size 200)
    std::vector<std::unordered_map<std::string, std::vector<double>>> args_dv_batches(N_batches);
    for (decltype(args_dv_batches.size()) i = 0u; i < args_dv_batches.size(); ++i) {
        std::vector<std::vector<double>> tmp(args_vv.begin() + batch_size * i, args_vv.begin() + batch_size * (i + 1));
        args_dv_batches[i] = vv_to_dv(tmp);
    }
    out = std::vector<double>(batch_size, 0.123);
    start = high_resolution_clock::now();
    for (decltype(args_dv_batches.size()) i = 0u; i < args_dv_batches.size(); ++i) {
        eval_batch_dbl(out, ex, args_dv_batches[i]);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of evaluations per second (tree in batches of 200): "
              << 1. / (static_cast<double>(duration.count()) / N) << "M\n";
    //
    // Init the LLVM machinery.
    // llvm_state s{"optimized"};
    //
    //// 6 - we time the function call from llvm batch
    // auto func_batch = s.fetch_batch("f");
    // out = std::vector<double>(10000, 0.12345);
    // auto llvm_batch_args = random_args_batch(2, 10000);
    // start = high_resolution_clock::now();
    // func_batch(out.data(), llvm_batch_args.data());
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Millions of evaluations per second (llvm batch 10000): "
    //          << 1. / (static_cast<double>(duration.count()) / N) << "M\n";
    //
    //// 7 - we time the function call from llvm batch (20).
    // auto g_batch = s.fetch_batch("g");
    // start = high_resolution_clock::now();
    // for (auto i = 0u; i < 500u; ++i) {
    //    g_batch(out.data() + i * 20, llvm_batch_args.data() + 2 * i * 20);
    //}
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Millions of evaluations per second (llvm batch 20): "
    //          << 1. / (static_cast<double>(duration.count()) / N) << "M\n";
    //
    return 0;
}
