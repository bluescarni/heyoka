// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <random>
#include <ratio>
#include <stdexcept>
#include <string>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <boost/program_options.hpp>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

using namespace heyoka;

static const long ntrials = 100000;

template <typename T>
void run_benchmark(unsigned order)
{
    std::mt19937 rng(std::random_device{}());

    std::uniform_real_distribution<double> rdist(-1., 1.);
    std::vector<T> poly, h, h_lo, h_hi, res_lo1, res_hi1, res_lo2, res_hi2;

    const auto batch_size = 1u;

    poly.resize((order + 1u) * batch_size);
    h.resize(batch_size);
    h_lo.resize(batch_size);
    h_hi.resize(batch_size);
    res_lo1.resize(batch_size);
    res_hi1.resize(batch_size);
    res_lo2.resize(batch_size);
    res_hi2.resize(batch_size);

    llvm_state s;

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    auto val_t = detail::to_llvm_type<T>(context);
    auto ptr_val_t = llvm::PointerType::getUnqual(val_t);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // Add the interval-arithmetic function.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type *>(5u, ptr_val_t), false);
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "penc_interval", &md);

    auto *out_lo_ptr = f->args().begin();
    auto *out_hi_ptr = f->args().begin() + 1;
    auto *cf_ptr = f->args().begin() + 2;
    auto *h_lo_ptr = f->args().begin() + 3;
    auto *h_hi_ptr = f->args().begin() + 4;

    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Load the h values.
    auto *h_lo_val = detail::load_vector_from_memory(builder, h_lo_ptr, batch_size);
    auto *h_hi_val = detail::load_vector_from_memory(builder, h_hi_ptr, batch_size);

    {
        auto [res_lo, res_hi] = detail::llvm_penc_interval<T>(s, cf_ptr, order, h_lo_val, h_hi_val, batch_size);

        // Store the result.
        detail::store_vector_to_memory(builder, out_lo_ptr, res_lo);
        detail::store_vector_to_memory(builder, out_hi_ptr, res_hi);
    }

    builder.CreateRetVoid();

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // Add the Cargo-Shisha function.
    ft = llvm::FunctionType::get(builder.getVoidTy(), std::vector<llvm::Type *>(4u, ptr_val_t), false);
    f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "penc_cargo_shisha", &md);

    out_lo_ptr = f->args().begin();
    out_hi_ptr = f->args().begin() + 1;
    cf_ptr = f->args().begin() + 2;
    auto *h_ptr = f->args().begin() + 3;

    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Load the h values.
    auto *h_val = detail::load_vector_from_memory(builder, h_ptr, batch_size);

    {
        auto [res_lo, res_hi] = detail::llvm_penc_cargo_shisha<T>(s, cf_ptr, order, h_val, batch_size);

        // Store the result.
        detail::store_vector_to_memory(builder, out_lo_ptr, res_lo);
        detail::store_vector_to_memory(builder, out_hi_ptr, res_hi);
    }

    builder.CreateRetVoid();

    // Verify.
    s.verify_function(f);

    // Run the optimisation pass.
    s.optimise();

    // Compile.
    s.compile();

    // Fetch the functions.
    auto *penc_int_f
        = reinterpret_cast<void (*)(T *, T *, const T *, const T *, const T *)>(s.jit_lookup("penc_interval"));
    auto *penc_cs_f = reinterpret_cast<void (*)(T *, T *, const T *, const T *)>(s.jit_lookup("penc_cargo_shisha"));

    double tot_int_time = 0, tot_cs_time = 0;
    T tot_int_width = 0, tot_cs_width = 0;

    for (auto _ = 0l; _ < ntrials; ++_) {
        // Generate the polynomial.
        for (auto &cf : poly) {
            cf = rdist(rng);
        }

        // Generate the h values.
        for (auto i = 0u; i < batch_size; ++i) {
            const auto tmp = rdist(rng);

            h[i] = tmp;
            h_lo[i] = tmp >= 0 ? 0. : tmp;
            h_hi[i] = tmp >= 0 ? tmp : 0.;
        }

        auto start = std::chrono::steady_clock::now();

        penc_int_f(res_lo1.data(), res_hi1.data(), poly.data(), h_lo.data(), h_hi.data());

        tot_int_time
            += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();

        tot_int_width += res_hi1[0] - res_lo1[0];

        start = std::chrono::steady_clock::now();

        penc_cs_f(res_lo2.data(), res_hi2.data(), poly.data(), h.data());

        tot_cs_time
            += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();

        tot_cs_width += res_hi2[0] - res_lo2[0];
    }

    std::cout << fmt::format("Runtime (interval vs CS): {}μs vs {}μs\n", tot_int_time / 1000., tot_cs_time / 1000.);
    std::cout << fmt::format("Average width (interval vs CS): {} vs {}\n", tot_int_width / ntrials,
                             tot_cs_width / ntrials);
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type;
    unsigned order = 0;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"),
        "floating-point type")("order", po::value<unsigned>(&order)->default_value(20u), "polynomial order");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (order == 0u) {
        throw std::invalid_argument("The polynomial order cannot be zero");
    }

    if (fp_type == "double") {
        run_benchmark<double>(order);
    } else if (fp_type == "long double") {
        run_benchmark<long double>(order);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_benchmark<mppp::real128>(order);
#endif
    } else {
        throw std::invalid_argument(fmt::format("Unsupported floating-point type '{}'", fp_type));
    }
}
