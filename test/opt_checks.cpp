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

#include <heyoka/model/nbody.hpp>
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

    REQUIRE(count == 3u);
}
