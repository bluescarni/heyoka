// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// NOTE: this is a stress test for parallel JIT compilation in llvm_multi_state.
//
// LLVM's ORC (LLVM >= 18) contains a race which, when several modules are compiled in parallel, can result in spurious
// "duplicate definition" errors for the external symbols resolved via the dynamic library search generators. heyoka
// works around this in define_abs_symbols_tolerant() (see llvm_state.cpp). The race is timing-dependent and rare. This
// test maximises the chances of triggering it by creating many small modules which all invoke the same set of functions
// from the C math library, and by compiling them in parallel over and over again. A single run is not guaranteed to hit
// the race, but over many CI runs the test acts as a canary: a failure here means that the workaround is not effective.

namespace
{

// The functions from the C math library invoked by every module.
//
// NOTE: we avoid functions such as sqrt() or fabs(), which LLVM may lower to machine instructions instead of calls.
//
// NOTE: all these functions are well-defined for the input value used in the test (see below).
constexpr std::array libm_funcs
    = {"sin", "cos", "tan", "exp", "log", "atan", "asin", "acos", "sinh", "cosh", "tanh", "cbrt"};

// Reference implementation of the function added to each module.
double ref_func(double x)
{
    return std::sin(x) + std::cos(x) + std::tan(x) + std::exp(x) + std::log(x) + std::atan(x) + std::asin(x)
           + std::acos(x) + std::sinh(x) + std::cosh(x) + std::tanh(x) + std::cbrt(x);
}

// Number of modules in each llvm_multi_state.
constexpr unsigned n_modules = 200;

// Number of iterations.
//
// NOTE: this has been chosen heuristically so that the test runs for a few seconds on a modern desktop machine.
constexpr unsigned n_iters = 35;

// Add to s a function with the given name which takes a double-precision argument x and returns the sum of all the
// functions in libm_funcs evaluated at x.
void add_test_func(llvm_state &s, const std::string &name)
{
    auto &bld = s.builder();
    auto &ctx = s.context();
    auto &md = s.module();

    auto *dbl_t = bld.getDoubleTy();
    auto *ft = llvm::FunctionType::get(dbl_t, {dbl_t}, false);
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &md);
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    auto *x = f->getArg(0);
    llvm::Value *ret = nullptr;
    for (const auto *fname : libm_funcs) {
        auto *val = detail::llvm_invoke_external(s, fname, dbl_t, {x});
        ret = (ret == nullptr) ? val : bld.CreateFAdd(ret, val);
    }

    bld.CreateRet(ret);
}

} // namespace

TEST_CASE("parjit stress")
{
    // NOTE: valid input for all the functions in libm_funcs.
    const auto x = 0.5;
    const auto ref = ref_func(x);

    for (unsigned iter = 0; iter < n_iters; ++iter) {
        INFO("Iteration: " << iter);

        // NOTE: clear the in-memory cache at every iteration. Otherwise, all iterations but the first one would fetch
        // the object code from the cache, thus skipping the parallel compilation.
        llvm_state::clear_memcache();

        // Create the modules.
        std::vector<llvm_state> states;
        states.reserve(n_modules);
        for (unsigned i = 0; i < n_modules; ++i) {
            llvm_state s{kw::opt_level = 0u};
            add_test_func(s, fmt::format("test_func_{}", i));
            states.push_back(std::move(s));
        }

        // Compile them in parallel.
        llvm_multi_state ms{std::move(states), true};
        REQUIRE(ms.get_parjit());
        ms.compile();

        // Sanity check a few of the compiled functions.
        for (const auto i : {0u, n_modules / 2u, n_modules - 1u}) {
            auto *fptr = reinterpret_cast<double (*)(double)>(ms.jit_lookup(fmt::format("test_func_{}", i)));
            REQUIRE(fptr(x) == approximately(ref));
        }
    }
}
