// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Transform in-place ex by decomposition, appending the
// result of the decomposition to u_vars_defs.
// NOTE: this will render ex unusable.
void taylor_decompose_ex(expression &ex, std::vector<expression> &u_vars_defs)
{
    auto visitor = [&u_vars_defs](auto &v) {
        using type = detail::uncvref_t<decltype(v)>;

        if constexpr (std::is_same_v<type, variable> || std::is_same_v<type, number>) {
            // NOTE: an expression does *not* require decomposition
            // if it is a variable or a number.
        } else if constexpr (std::is_same_v<type, binary_operator>) {
            // Variables to track how the size
            // of u_vars_defs changes after the decomposition
            // of lhs and rhs.
            auto old_size = u_vars_defs.size(), new_size = old_size;

            // We decompose the lhs, and we check if the
            // decomposition added new elements to u_vars_defs.
            // If it did, then it means that the lhs required
            // further decompositions and the creation of new
            // u vars: the new lhs will become the last added
            // u variable. If it did not, it means that the lhs
            // was a variable or a number (see above), and thus
            // we can use it as-is.
            taylor_decompose_ex(v.lhs(), u_vars_defs);
            new_size = u_vars_defs.size();
            if (new_size > old_size) {
                v.lhs() = expression{variable{"u_" + detail::li_to_string(new_size - 1u)}};
            }
            old_size = new_size;

            // Same for the rhs.
            taylor_decompose_ex(v.rhs(), u_vars_defs);
            new_size = u_vars_defs.size();
            if (new_size > old_size) {
                v.rhs() = expression{variable{"u_" + detail::li_to_string(new_size - 1u)}};
            }

            u_vars_defs.emplace_back(std::move(v));
        } else if constexpr (std::is_same_v<type, function>) {
            // The function call treatment is a generalization
            // of the binary operator.
            auto old_size = u_vars_defs.size(), new_size = old_size;

            for (auto &arg : v.args()) {
                taylor_decompose_ex(arg, u_vars_defs);
                new_size = u_vars_defs.size();
                if (new_size > old_size) {
                    arg = expression{variable{"u_" + detail::li_to_string(new_size - 1u)}};
                }
                old_size = new_size;
            }

            u_vars_defs.emplace_back(std::move(v));
        } else {
            static_assert(always_false_v<type>, "Unhandled expression type");
        }
    };

    std::visit(visitor, ex.value());
}

} // namespace

} // namespace detail

std::vector<expression> taylor_decompose(std::vector<expression> v_ex)
{
    // Determine the variables in the system of equations.
    std::vector<std::string> vars;
    for (const auto &ex : v_ex) {
        auto ex_vars = get_variables(ex);
        vars.insert(vars.end(), std::make_move_iterator(ex_vars.begin()), std::make_move_iterator(ex_vars.end()));
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
    }

    if (vars.size() != v_ex.size()) {
        throw std::invalid_argument("The number of variables (" + std::to_string(vars.size())
                                    + ") differs from the number of equations (" + std::to_string(v_ex.size()) + ")");
    }

    // Create the map for renaming the variables to u_i.
    // The renaming will be done in alphabetical order.
    std::unordered_map<std::string, std::string> repl_map;
    for (decltype(vars.size()) i = 0; i < vars.size(); ++i) {
        [[maybe_unused]] const auto eres = repl_map.emplace(vars[i], "u_" + detail::li_to_string(i));
        assert(eres.second);
    }

    // Rename the variables in the original equations.
    for (auto &ex : v_ex) {
        rename_variables(ex, repl_map);
    }

    // Init the vector containing the definitions
    // of the u variables. It begins with a list
    // of the original variables of the system.
    std::vector<expression> u_vars_defs;
    for (const auto &var : vars) {
        u_vars_defs.emplace_back(variable{var});
    }

    // Create a copy of the original equations in terms of u variables.
    // We will be reusing this below.
    auto v_ex_copy = v_ex;

    // Run the decomposition on each equation.
    for (decltype(v_ex.size()) i = 0; i < v_ex.size(); ++i) {
        // Record the current size of u_vars_defs.
        const auto orig_size = u_vars_defs.size();

        // Decompose the current equation.
        detail::taylor_decompose_ex(v_ex[i], u_vars_defs);

        if (u_vars_defs.size() != orig_size) {
            // NOTE: if the size of u_vars_defs changes,
            // it means v_ex[i] was decomposed in multiple
            // expressions. In such
            // case, we replace the original definition of the
            // equation with its definition in terms of the
            // last u variable added in the decomposition.
            // In the other case, v_ex_copy will keep on containing
            // the original definition of v_ex[i] in terms
            // of u variables.
            v_ex_copy[i] = expression{variable{"u_" + detail::li_to_string(u_vars_defs.size() - 1u)}};
        }
    }

    // Append the (possibly updated) definitions of the diff equations
    // in terms of u variables.
    for (auto &ex : v_ex_copy) {
        u_vars_defs.emplace_back(std::move(ex));
    }

    return u_vars_defs;
}

} // namespace heyoka
