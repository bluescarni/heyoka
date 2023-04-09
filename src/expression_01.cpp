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
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

#if !defined(NDEBUG)

// Helper to verify a revdiff decomposition.
// NOTE: here nvars refers to the total number of variables
// *and* params in orig.
void verify_revdiff_dec(const expression &orig, const std::vector<expression> &dc,
                        std::vector<expression>::size_type nvars)
{
    assert(!dc.empty());

    using idx_t = std::vector<expression>::size_type;

    // The first nvars expressions must be just variables or params.
    for (idx_t i = 0; i < nvars; ++i) {
        assert(std::holds_alternative<variable>(dc[i].value()) || std::holds_alternative<param>(dc[i].value()));
    }

    // From nvars to dc.size() - 1, the expressions
    // must be functions whose arguments
    // are either variables in the u_n form,
    // where n < i, or numbers.
    // NOTE: arguments cannot be params because these
    // were turned into variables previously.
    for (auto i = nvars; i < dc.size() - 1u; ++i) {
        std::visit(
            [i](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    for (const auto &arg : v.args()) {
                        if (auto *p_var = std::get_if<variable>(&arg.value())) {
                            assert(p_var->name().rfind("u_", 0) == 0);
                            assert(uname_to_index(p_var->name()) < i);
                        } else {
                            assert(std::holds_alternative<number>(arg.value()));
                        }
                    }
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].value());
    }

    // The last element of the decomposition must be
    // a variable in the u_n form.
    // NOTE: the last element cannot be a number because
    // this special case has been dealt with previously.
    std::visit(
        [&dc](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, variable>) {
                assert(v.name().rfind("u_", 0) == 0);
                assert(uname_to_index(v.name()) < dc.size() - 1u);
            } else {
                assert(false); // LCOV_EXCL_LINE
            }
        },
        dc.back().value());

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of the original variables or other u variables,
    // and store it in subs_map.
    for (idx_t i = 0; i < dc.size() - 1u; ++i) {
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map));
    }

    // Reconstruct the expression and compare it to the original one.
    assert(subs(dc.back(), subs_map) == orig);
}

#endif

// Simplify a revdiff decomposition by removing
// common subexpressions.
// NOTE: here nvars refers to the total number of variables
// *and* params.
std::vector<expression> revdiff_decompose_cse(std::vector<expression> &v_ex, std::vector<expression>::size_type nvars)
{
    using idx_t = std::vector<expression>::size_type;

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Cache the original size for logging later.
    const auto orig_size = v_ex.size();

    // A function decomposition is supposed
    // to have nvars variables + params at the beginning,
    // 1 variable at the end and possibly
    // extra variables in the middle.
    assert(v_ex.size() >= nvars + 1u);

    // Init the return value.
    std::vector<expression> retval;

    // expression -> idx map. This will end up containing
    // all the unique expressions from v_ex, and it will
    // map them to their indices in retval (which will
    // in general differ from their indices in v_ex).
    std::unordered_map<expression, idx_t> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // The first nvars definitions are just renaming
    // of the original variables/params into u variables.
    for (idx_t i = 0; i < nvars; ++i) {
        assert(std::holds_alternative<variable>(v_ex[i].value()) || std::holds_alternative<param>(v_ex[i].value()));
        retval.push_back(std::move(v_ex[i]));

        // NOTE: the u vars that correspond to the original
        // variables/params are never simplified,
        // thus map them onto themselves.
        [[maybe_unused]] const auto res = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Handle the u variables which do not correspond to the original variables/params.
    for (auto i = nvars; i < v_ex.size() - 1u; ++i) {
        auto &ex = v_ex[i];

        // Rename the u variables in ex.
        rename_variables(ex, uvars_rename);

        if (auto it = ex_map.find(ex); it == ex_map.end()) {
            // This is the first occurrence of ex in the
            // decomposition. Add it to retval.
            retval.push_back(ex);

            // Add ex to ex_map, mapping it to
            // the index it corresponds to in retval
            // (let's call it j).
            ex_map.emplace(std::move(ex), retval.size() - 1u);

            // Update uvars_rename. This will ensure that
            // occurrences of the variable 'u_i' in the next
            // elements of v_ex will be renamed to 'u_j'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", retval.size() - 1u));
            assert(res.second);
        } else {
            // ex is redundant. This means
            // that it already appears in retval at index
            // it->second. Don't add anything to retval,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", it->second));
            assert(res.second); // LCOV_EXCL_LINE
        }
    }

    // Handle the last element of the decomposition. We just need to
    // adjust the u index.
    // NOTE: the last element can be only a variable - params were turned
    // into variables and the special case of a number expression was dealt
    // with previously.
    assert(std::holds_alternative<variable>(v_ex.back().value()));
    rename_variables(v_ex.back(), uvars_rename);

    retval.push_back(std::move(v_ex.back()));

    get_logger()->debug("revdiff CSE reduced decomposition size from {} to {}", orig_size, retval.size());
    get_logger()->trace("revdiff CSE runtime: {}", sw);

    return retval;
}

} // namespace

std::pair<std::vector<expression>, std::vector<expression>::size_type> revdiff_decompose(const expression &e)
{
    // Determine the list of variables and params.
    const auto vars = get_variables(e);
    const auto nvars = vars.size();

    const auto params = get_params(e);
    const auto npars = params.size();

    // Create the map for renaming variables and params to u_i.
    // The variables will precede the params. The renaming will be
    // done in alphabetical order for the variables and in index order
    // for the params.
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

    // Store a copy of the original function for checking later.
    auto orig_ex = copy(e);

#endif

    // Rename variables and params.
    // NOTE: this creates a new deep copy of e.
    auto ex = subs(e, repl_map);

    // Init the decomposition. It begins with a list
    // of the original variables and params of the function.
    std::vector<expression> ret;
    ret.reserve(boost::safe_numerics::safe<decltype(ret.size())>(nvars) + npars);
    for (const auto &var : vars) {
        ret.emplace_back(var);
    }
    for (const auto &par : params) {
        ret.emplace_back(par);
    }

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition.
    if (const auto dres = decompose(ex, ret)) {
        // NOTE: if the expression was decomposed
        // we have to update the original definition of ex
        // so that it points to the u variable
        // that now represents it.
        // NOTE: all functions are forced to return
        // a non-empty dres in the func API.
        ex = expression{fmt::format("u_{}", *dres)};
    } else {
        // NOTE: ex can only be a variable because the case in which
        // ex is a number was handled at the beginning of this function,
        // and if e was originally a param, it was turned into
        // a variable when invoking subs() earlier.
        assert(std::holds_alternative<variable>(ex.value()));
    }

    // Append the definition of ex
    ret.emplace_back(std::move(ex));

    get_logger()->trace("revdiff decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    verify_revdiff_dec(orig_ex, ret, nvars + npars);

#endif

    // Simplify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = revdiff_decompose_cse(ret, nvars + npars);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_revdiff_dec(orig_ex, ret, nvars + npars);

#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = function_sort_dc(ret, nvars + npars, 1);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_revdiff_dec(orig_ex, ret, nvars + npars);

#endif

    return {std::move(ret), nvars + npars};
}

namespace
{

auto revdiff_make_adj_revdep(const std::vector<expression> &dc, std::vector<expression>::size_type nvars)
{
    assert(dc.size() >= 2u);
    assert(nvars < dc.size());

    using idx_t = std::vector<expression>::size_type;

    // Do an initial pass to create the adjoints, the (unsorted)
    // vectors of reverse dependencies and the substitution map.
    std::vector<std::unordered_map<std::uint32_t, expression>> adj;
    adj.resize(boost::numeric_cast<decltype(adj.size())>(dc.size()));

    std::vector<std::vector<std::uint32_t>> revdep;
    revdep.resize(boost::numeric_cast<decltype(revdep.size())>(dc.size()));

    std::unordered_map<std::string, expression> subs_map;

    for (idx_t i = 0; i < nvars; ++i) {
        // NOTE: no adjoints or reverse dependencies needed for the initial definitions.
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map));
    }

    for (idx_t i = nvars; i < dc.size(); ++i) {
        auto &cur_adj_dict = adj[i];

        for (const auto &var : get_variables(dc[i])) {
            const auto idx = uname_to_index(var);

            assert(cur_adj_dict.count(idx) == 0u);
            cur_adj_dict[idx] = diff(dc[i], var);

            assert(idx < revdep.size());
            revdep[idx].push_back(boost::numeric_cast<std::uint32_t>(i));
        }

        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map));
    }

    // Sort the vectors of reverse dependencies.
    // NOTE: this is not strictly necessary for the correctness
    // of the algorithm. It will just ensure that when we eventually
    // compute the derivative of the output wrt a subexpression, the
    // summation over the reverse dependencies happens in index order.
    for (auto &rvec : revdep) {
        std::sort(rvec.begin(), rvec.end());

        // Check that there are no duplicates.
        assert(std::adjacent_find(rvec.begin(), rvec.end()) == rvec.end());
    }

#if !defined(NDEBUG)

    // Sanity checks in debug mode.
    for (idx_t i = 0; i < nvars; ++i) {
        // No adjoints for the vars/params definitions.
        assert(adj[i].empty());

        // Each var/param must be a dependency for some
        // other subexpression.
        assert(!revdep[i].empty());
    }

    for (idx_t i = nvars; i < dc.size() - 1u; ++i) {
        // The only possibility for an adjoint dict to be empty
        // is if all the subexpression arguments are numbers.
        assert(!adj[i].empty() || get_variables(dc[i]).empty());

        // Every subexpression apart from the last (i.e., the output)
        // must be a dependency for some other subexpression.
        assert(!revdep[i].empty());
    }

    // The last subexpression is the output: it must have only
    // 1 element in the adjoints dict, that element must be the constant 1,
    // and it cannot be the dependency for any other subexpression.
    assert(adj.back().size() == 1u);
    assert(adj.back().count(boost::numeric_cast<std::uint32_t>(dc.size() - 2u)) == 1u);
    assert(adj.back().at(boost::numeric_cast<std::uint32_t>(dc.size() - 2u)) == 1_dbl);
    assert(revdep.back().empty());

#endif

    return std::tuple{std::move(adj), std::move(revdep), std::move(subs_map)};
}

} // namespace

// Implementation of reverse-mode differentiation.
std::vector<expression> reverse_diff(const expression &e, const std::vector<expression> &args)
{
    assert(!args.empty());
    assert(std::all_of(args.begin(), args.end(), [](const auto &arg) {
        return std::holds_alternative<variable>(arg.value()) || std::holds_alternative<param>(arg.value());
    }));
    assert(std::unordered_set(args.begin(), args.end()).size() == args.size());

    // Handle the trivial case first.
    if (std::holds_alternative<number>(e.value())) {
        return {args.size(), 0_dbl};
    }

    // Run the decomposition.
    const auto [dc, nvars] = revdiff_decompose(e);

    // Create the adjoints, the reverse dependencies and the substitution map.
    const auto [adj, revdep, subs_map] = revdiff_make_adj_revdep(dc, nvars);

    // Create the list of derivatives of the output wrt every
    // subexpression. Init the last element with the constant 1.
    std::vector<expression> diffs(dc.size());
    diffs.back() = 1_dbl;

    // Fill in the remaining derivatives, including those wrt the
    // original vars and params.
    for (std::vector<expression>::size_type i = 1; i < dc.size(); ++i) {
        // NOTE: iterate backwards (this is the reverse pass).
        const auto cur_idx = dc.size() - 1u - i;

        std::vector<expression> tmp_sum;
        for (const auto rd_idx : revdep[cur_idx]) {
            assert(rd_idx < diffs.size());
            assert(rd_idx < adj.size());
            // NOTE: the reverse dependency must point
            // to a subexpression *after* the current one.
            assert(rd_idx > cur_idx);

            tmp_sum.push_back(diffs[rd_idx] * adj[rd_idx].at(boost::numeric_cast<std::uint32_t>(cur_idx)));
        }

        assert(!tmp_sum.empty());

        diffs[cur_idx] = sum(std::move(tmp_sum));
    }

    // Create a dict mapping the vars/params in e to the derivatives
    // wrt them.
    std::unordered_map<expression, expression> diff_map;
    for (std::vector<expression>::size_type i = 0; i < nvars; ++i) {
        [[maybe_unused]] const auto [_, flag] = diff_map.try_emplace(dc[i], diffs[i]);
        assert(flag);
    }

    // Assemble the return value.
    std::vector<expression> retval;
    retval.reserve(args.size());
    for (const auto &diff_arg : args) {
        if (auto it = diff_map.find(diff_arg); it != diff_map.end()) {
            retval.push_back(subs(it->second, subs_map));
        } else {
            retval.push_back(0_dbl);
        }
    }

    return retval;
}

std::vector<expression> grad_impl(const expression &e, diff_mode dm,
                                  const std::variant<diff_args, std::vector<expression>> &d_args)
{
    // Extract/build the arguments.
    std::vector<expression> args;

    if (std::holds_alternative<std::vector<expression>>(d_args)) {
        args = std::get<std::vector<expression>>(d_args);
    } else {
        switch (std::get<diff_args>(d_args)) {
            case diff_args::all:
                for (const auto &var : get_variables(e)) {
                    args.emplace_back(var);
                }

                for (const auto &par : get_params(e)) {
                    args.push_back(par);
                }

                break;
            case diff_args::vars:
                for (const auto &var : get_variables(e)) {
                    args.emplace_back(var);
                }

                break;
            case diff_args::params:
                for (const auto &par : get_params(e)) {
                    args.push_back(par);
                }

                break;
            default:
                throw std::invalid_argument("Invalid diff_args enumerator");
        }
    }

    // Handle empty args.
    if (args.empty()) {
        return {};
    }

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

} // namespace detail

HEYOKA_END_NAMESPACE
