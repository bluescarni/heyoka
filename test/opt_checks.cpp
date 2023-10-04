// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>

#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#include <heyoka/config.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/model/pendulum.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("function inlining")
{
    auto sys = model::nbody(6);

    auto ta = taylor_adaptive<double>{sys, std::vector<double>(36u, 0.), kw::compact_mode = true};

    auto md_ir = ta.get_llvm_state().get_ir();

    using string_find_iterator = boost::find_iterator<std::string::iterator>;

    auto count = 0u;
    for (auto it = boost::make_find_iterator(md_ir, boost::first_finder("define ", boost::is_iequal()));
         it != string_find_iterator(); ++it) {
        ++count;
    }

    // NOTE: in general we expect 3 functions definitions, but auto-vectorization
    // could bump up this number. I think 6 is the maximum right now (3 possible
    // vector width on x86 - 2, 4, 8).
    REQUIRE(count <= 6u);
}

// Vectorization of the pow() function when determining
// the timestep size in an integrator.
TEST_CASE("pow vect")
{
    auto ta = taylor_adaptive<double>{model::pendulum(), std::vector<double>(2u, 0.), kw::slp_vectorize = true};

#if defined(HEYOKA_WITH_SLEEF)

    auto md_ir = ta.get_llvm_state().get_ir();

    const auto &tf = detail::get_target_features();

    // NOTE: run the check only on some archs/LLVM versions.
    if (tf.sse2) {
        REQUIRE(!boost::algorithm::contains(md_ir, "llvm.pow"));
    }

#if LLVM_VERSION_MAJOR >= 16

    std::cout << md_ir << '\n';

    // NOTE: LLVM16 is currently the version tested in the CI on arm64.
    if (tf.aarch64) {
        REQUIRE(!boost::algorithm::contains(md_ir, "llvm.pow"));
    }

#endif

#endif
}
