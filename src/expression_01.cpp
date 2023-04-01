// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

std::vector<expression> reverse_diff(const expression &e, const std::vector<expression> &args)
{
    assert(std::all_of(args.begin(), args.end(), [](const auto &arg) {
        return std::holds_alternative<variable>(arg.value()) || std::holds_alternative<param>(arg.value());
    }));
    assert(std::unordered_set(args.begin(), args.end()).size() == args.size());

    // Need to operate on a copy due to in-place mutation
    // via rename_variables().
    auto ex = copy(e);

    // Determine the list of variables and params.
    const auto vars = get_variables(ex);
    const auto params = get_params(ex);

    // Create the map for renaming variables and params to u_i.
    // The variables will precede tha params, the renaming will be done in alphabetical order
    // for the variables and in index order for the params.
    std::unordered_map<expression, expression> repl_map;
    {
        boost::safe_numerics::safe<std::size_t> u_idx = 0;

        for (const auto &var : vars) {
            [[maybe_unused]] const auto eres
                = repl_map.emplace(var, fmt::format("u_{}", static_cast<std::size_t>(u_idx++)));
            assert(eres.second);
        }

        for (const auto &p : params) {
            [[maybe_unused]] const auto eres
                = repl_map.emplace(p, fmt::format("u_{}", static_cast<std::size_t>(u_idx++)));
            assert(eres.second);
        }
    }

#if !defined(NDEBUG)

    // Store a copy of the original expression for checking later.
    auto orig_ex = copy(ex);

#endif
}

} // namespace detail

std::vector<expression> grad(const expression &e, const std::vector<expression> &args, diff_mode dm)
{
    // Ensure that every expression in args is either a variable
    // or a param.
    if (std::any_of(args.begin(), args.end(), [](const auto &arg) {
            return !std::holds_alternative<variable>(arg.value()) && !std::holds_alternative<param>(arg.value());
        })) {
        throw std::invalid_argument("The list of expressions with respect to which the "
                                    "gradient is to be computed can contain only variables and parameters");
    }

    // Check if there are repeated entries in args.
    std::unordered_set args_set(args.begin(), args.end());
    if (args_set.size() != args.size()) {
        throw std::invalid_argument("Duplicate entries detected in the list of variables with respect to which the "
                                    "gradient is to be computed");
    }

    if (dm == diff_mode::forward) {
        std::vector<expression> retval;
        retval.reserve(args.size());

        // NOTE: this can clearly be easily parallelised,
        // if needed.
        for (const auto &arg : args) {
            retval.push_back(diff(e, arg));
        }

        return retval;
    } else {
        return detail::reverse_diff(e, args);
    }
}

HEYOKA_END_NAMESPACE
