// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <variant>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>

#include "catch.hpp"

using namespace heyoka;

using heyoka::detail::combined_cos;
using heyoka::detail::combined_sin;

TEST_CASE("sincos fp")
{
    {
        auto x = "x"_var;

        cfunc<double> cf({combined_sin(x) + combined_cos(x)}, {x});
    }

    {
        auto [a, b, c, d] = make_vars("a", "b", "c", "d");

        cfunc<double> cf({combined_sin(a), combined_sin(b), combined_sin(c), combined_sin(d), combined_cos(a),
                          combined_cos(b), combined_cos(c), combined_cos(d)},
                         {a, b, c, d}, kw::slp_vectorize = true, kw::opt_level = 3);

        std::cout << std::get<0>(cf.get_llvm_states())[0].get_ir() << "\n\n\n";
    }
}
