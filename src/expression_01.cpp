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
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

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

std::pair<std::vector<expression>, std::vector<expression>::size_type>
revdiff_decompose(const std::vector<expression> &v_ex_)
{
    // Determine the list of variables and params.
    const auto vars = get_variables(v_ex_);
    const auto nvars = vars.size();

    const auto params = get_params(v_ex_);
    const auto npars = params.size();

    // Cache the number of outputs.
    const auto nouts = v_ex_.size();
    assert(nouts > 0u);

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
    auto orig_v_ex = copy(v_ex_);

#endif

    // Rename variables and params.
    // NOTE: this creates a new deep copy of v_ex_.
    auto v_ex = subs(v_ex_, repl_map);

    // Init the decomposition. It begins with a list
    // of the original variables and params of the function.
    std::vector<expression> ret;
    ret.reserve(boost::safe_numerics::safe<decltype(ret.size())>(nvars) + npars);
    for (const auto &var : vars) {
        // NOTE: transform into push_back() once get_variables() returns
        // expressions rather than strings.
        ret.emplace_back(var);
    }
    for (const auto &par : params) {
        ret.push_back(par);
    }

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition.
    decompose(v_ex, ret);

    // Append the definitions of the outputs
    // in terms of u variables or numbers.
    for (auto &ex : v_ex) {
        ret.emplace_back(std::move(ex));
    }

    get_logger()->trace("revdiff decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    verify_function_dec(orig_v_ex, ret, nvars + npars, true);

#endif

    // Simplify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = function_decompose_cse(ret, nvars + npars, nouts);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_function_dec(orig_v_ex, ret, nvars + npars, true);

#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = function_sort_dc(ret, nvars + npars, nouts);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_function_dec(orig_v_ex, ret, nvars + npars, true);

#endif

    return {std::move(ret), nvars + npars};
}

namespace
{

auto revdiff_make_adj_revdep(const std::vector<expression> &dc, std::vector<expression>::size_type nvars,
                             [[maybe_unused]] std::vector<expression>::size_type nouts)
{
    // NOTE: the shortest possible dc is for a scalar
    // function identically equal to a number. In this case,
    // dc will have a size of 1.
    assert(!dc.empty());
    assert(nvars < dc.size());
    assert(nouts >= 1u);
    assert(nouts <= dc.size());

    using idx_t = std::vector<expression>::size_type;

    // Do an initial pass to create the adjoints, the
    // vectors of direct and reverse dependencies,
    // and the substitution map.
    std::vector<std::unordered_map<std::uint32_t, expression>> adj;
    adj.resize(boost::numeric_cast<decltype(adj.size())>(dc.size()));

    std::vector<std::vector<std::uint32_t>> dep;
    dep.resize(boost::numeric_cast<decltype(dep.size())>(dc.size()));

    std::vector<std::vector<std::uint32_t>> revdep;
    revdep.resize(boost::numeric_cast<decltype(revdep.size())>(dc.size()));

    std::unordered_map<std::string, expression> subs_map;

    for (idx_t i = 0; i < nvars; ++i) {
        // NOTE: no adjoints or direct/reverse dependencies needed for the initial definitions.
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map));
    }

    for (idx_t i = nvars; i < dc.size(); ++i) {
        auto &cur_adj_dict = adj[i];
        auto &cur_dep = dep[i];

        for (const auto &var : get_variables(dc[i])) {
            const auto idx = uname_to_index(var);

            assert(cur_adj_dict.count(idx) == 0u);
            cur_adj_dict[idx] = diff(dc[i], var);

            assert(idx < revdep.size());
            revdep[idx].push_back(boost::numeric_cast<std::uint32_t>(i));

            cur_dep.push_back(idx);
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

        // No direct dependencies in the
        // initial definitions.
        assert(dep[i].empty());
    }

    for (idx_t i = nvars; i < dc.size() - nouts; ++i) {
        // The only possibility for an adjoint dict to be empty
        // is if all the subexpression arguments are numbers.
        // NOTE: params have been turned into variables, so that
        // get_variables() will also list param args.
        assert(!adj[i].empty() || get_variables(dc[i]).empty());

        // Every subexpression must be a dependency for some other subexpression.
        assert(!revdep[i].empty());

        // Every subexpression must depend on another subexpression,
        // unless all the subexpression arguments are numbers.
        assert(!dep[i].empty() || get_variables(dc[i]).empty());
    }

    // Each output:
    // - cannot be the dependency for any subexpression,
    // - must depend on 1 subexpression, unless the output
    //   itself is a number,
    // - must either be a number or have only 1 element in the adjoints dict, and:
    //   - the key of such element cannot be another output,
    //   - the value of such element must be the constant 1_dbl
    //     (this comes from the derivative of a variables wrt itself
    //     returning 1_dbl).
    for (idx_t i = dc.size() - nouts; i < dc.size(); ++i) {
        if (adj[i].empty()) {
            assert(std::holds_alternative<number>(dc[i].value()));
        } else {
            assert(adj[i].size() == 1u);
            assert(adj[i].begin()->first < dc.size() - nouts);
            assert(adj[i].begin()->second == 1_dbl);
        }

        assert(revdep[i].empty());

        if (std::holds_alternative<number>(dc[i].value())) {
            assert(dep[i].empty());
        } else {
            assert(dep[i].size() == 1u);
        }
    }

#endif

    return std::tuple{std::move(adj), std::move(dep), std::move(revdep), std::move(subs_map)};
}

// Hasher for a vector of 32-bit unsigned integers.
struct vec32_hasher {
    std::size_t operator()(const std::vector<std::uint32_t> &v) const noexcept
    {
        // Initial seed taken from the size of v.
        std::size_t seed = std::hash<decltype(v.size())>{}(v.size());

        for (const auto idx : v) {
            boost::hash_combine(seed, idx);
        }

        return seed;
    }
};

using dtens_diff_map_t = std::unordered_map<std::vector<std::uint32_t>, expression, vec32_hasher>;

using dtens_list_t = std::vector<std::vector<expression>>;

} // namespace

} // namespace detail

struct dtens::impl {
    detail::dtens_diff_map_t m_map;
    detail::dtens_list_t m_list;
};

namespace detail
{

namespace
{

auto diff_tensors_impl(const std::vector<expression> &v_ex, const std::vector<expression> &args, std::uint32_t order)
{
    spdlog::stopwatch sw;

    assert(!args.empty());
    assert(std::all_of(args.begin(), args.end(), [](const auto &arg) {
        return std::holds_alternative<variable>(arg.value()) || std::holds_alternative<param>(arg.value());
    }));
    assert(std::unordered_set(args.begin(), args.end()).size() == args.size());

    // Cache the original number of outputs and the diff arguments.
    const auto orig_nouts = v_ex.size();
    const auto nargs = args.size();

    assert(orig_nouts > 0u);
    assert(nargs > 0u);

    // Map to associate a vector of indices to a derivative.
    // The first element in each vector is the component index,
    // the rest of the indices are the derivative orders for
    // each diff args. E.g., with args = [x, y, z],
    // then [0, 1, 2, 1] means d4f0/(dx dy**2 dz) (where f0 is the
    // first component of the vector function f).
    dtens_diff_map_t diff_map;

    // List of tensors of derivatives.
    dtens_list_t tensor_list;

    // List of indices vectors. This is used to facilitate the
    // iterative construction of the indices vectors order
    // by order.
    std::vector<std::vector<std::uint32_t>> v_indices;

    // An indices vector with preallocated storage,
    // used as temporary variable in several places below.
    std::vector<std::uint32_t> tmp_v_idx;
    tmp_v_idx.resize(1 + boost::safe_numerics::safe<decltype(tmp_v_idx.size())>(nargs));

    // Init tensor_list and diff_map with the order 0 derivatives
    // (i.e., the original function components).
    tensor_list.emplace_back();

    for (decltype(v_ex.size()) i = 0; i < orig_nouts; ++i) {
        tmp_v_idx[0] = boost::numeric_cast<std::uint32_t>(i);
        v_indices.push_back(tmp_v_idx);

        assert(diff_map.count(tmp_v_idx) == 0u);
        diff_map[tmp_v_idx] = v_ex[i];

        tensor_list.back().push_back(v_ex[i]);
    }

    // Fetch the range in v_indices corresponding to the
    // current order (i.e., order-0) indices.
    decltype(v_indices.size()) cur_idx_begin = 0, cur_idx_end = v_indices.size();

    // Iterate over the orders.
    for (std::uint32_t cur_order = 0; cur_order < order; ++cur_order) {
        // Prepare the new tensor.
        tensor_list.emplace_back();
        auto &new_tensor = tensor_list.back();
        const auto &prev_tensor = *(tensor_list.end() - 2);

        // The current number of outputs is the number of components
        // in the previous order tensor.
        const auto cur_nouts = prev_tensor.size();

        // Run the decomposition on the tensor of the previous order.
        const auto [dc, nvars] = revdiff_decompose(prev_tensor);

        // Create the adjoints, the direct/reverse dependencies and the substitution map.
        const auto [adj, dep, revdep, subs_map] = revdiff_make_adj_revdep(dc, nvars, cur_nouts);

        // These two containers will be used to store the list of subexpressions
        // on which an output depends. They are used in the reverse pass
        // to avoid iterating over those subexpressions on which the output
        // does not depend (recall that the decomposition contains the subexpressions
        // for ALL outputs). We need two containers (with identical content)
        // because we need both ordered iteration AND fast lookup.
        std::unordered_set<std::uint32_t> out_deps;
        std::vector<std::uint32_t> sorted_out_deps;

        // A stack to be used when filling up out_deps/sorted_out_deps.
        std::deque<std::uint32_t> stack;

        spdlog::stopwatch sw_inner;

        // Run the reverse pass for each output. The derivatives
        // wrt the output will be stored into diffs.
        std::vector<expression> diffs(dc.size());
        for (decltype(v_ex.size()) i = 0; i < cur_nouts; ++i) {
            // Compute the index of the current output in the decomposition.
            const auto out_idx = boost::numeric_cast<std::uint32_t>(diffs.size() - cur_nouts + i);

            // Seed the stack and out_deps/sorted_out_deps with the
            // current output's dependency.
            stack.assign(dep[out_idx].begin(), dep[out_idx].end());
            sorted_out_deps.assign(dep[out_idx].begin(), dep[out_idx].end());
            out_deps.clear();
            out_deps.insert(dep[out_idx].begin(), dep[out_idx].end());

#if !defined(NDEBUG)

            // NOTE: an output can only have 0 or 1 dependencies.
            if (stack.empty()) {
                assert(std::holds_alternative<number>(dc[out_idx].value()));
            } else {
                assert(stack.size() == 1u);
            }

#endif

            // Build out_deps/sorted_out_deps by traversing
            // the decomposition backwards.
            while (!stack.empty()) {
                // Pop the first element from the stack.
                const auto cur_idx = stack.front();
                stack.pop_front();

                // Push into the stack and out_deps/sorted_out_deps
                // the dependencies of cur_idx.
                for (const auto next_idx : dep[cur_idx]) {
                    // NOTE: if next_idx is already in out_deps,
                    // it means that it was visited already and thus
                    // it does not need to be put in the stack.
                    if (out_deps.count(next_idx) == 0u) {
                        stack.push_back(next_idx);
                        sorted_out_deps.push_back(next_idx);
                        out_deps.insert(next_idx);
                    }
                }
            }

            // Sort sorted_out_deps in decreasing order.
            std::sort(sorted_out_deps.begin(), sorted_out_deps.end(), std::greater{});

            // sorted_out_deps cannot have duplicate values.
            assert(std::adjacent_find(sorted_out_deps.begin(), sorted_out_deps.end()) == sorted_out_deps.end());
            // sorted_out_deps either must be empty, or its last index
            // must refer to a variable/param (i.e., the current output
            // must have a var/param as last element in the chain of dependencies).
            assert(sorted_out_deps.empty() || *sorted_out_deps.rbegin() < nvars);
            assert(sorted_out_deps.size() == out_deps.size());

            // Set the seed value for the current output.
            diffs[out_idx] = 1_dbl;

            // Set the derivatives wrt all vars/params for the current output
            // to zero, so that if the current output does not depend on a
            // var/param then the derivative wrt that var/param is pre-emptively
            // set to zero.
            std::fill(diffs.data(), diffs.data() + nvars, 0_dbl);

            // Run the reverse pass on all subexpressions which
            // the current output depends on.
            for (const auto cur_idx : sorted_out_deps) {
                std::vector<expression> tmp_sum;

                for (const auto rd_idx : revdep[cur_idx]) {
                    assert(rd_idx < diffs.size());
                    assert(rd_idx < adj.size());
                    // NOTE: the reverse dependency must point
                    // to a subexpression *after* the current one.
                    assert(rd_idx > cur_idx);
                    assert(adj[rd_idx].count(cur_idx) == 1u);

                    // NOTE: if the current subexpression is a dependency
                    // for another subexpression which is neither the current output
                    // nor one of its dependencies, then the derivative is zero.
                    if (rd_idx != out_idx && out_deps.count(rd_idx) == 0u) {
                        tmp_sum.push_back(0_dbl);
                    } else {
                        tmp_sum.push_back(diffs[rd_idx] * adj[rd_idx].find(cur_idx)->second);
                    }
                }

                assert(!tmp_sum.empty());

                diffs[cur_idx] = sum(std::move(tmp_sum));
            }

            // Create a dict mapping the vars/params in the decomposition
            // to the derivatives of the current output wrt them. This is used
            // to fetch from diffs only the derivatives we are interested in
            // (since there may be vars/params in the decomposition wrt which
            // the derivatives are not requested).
            std::unordered_map<expression, expression> dmap;
            for (std::vector<expression>::size_type j = 0; j < nvars; ++j) {
                [[maybe_unused]] const auto [_, flag] = dmap.try_emplace(dc[j], diffs[j]);
                assert(flag);
            }

            // Add the derivatives to the new tensor and to diff_map.
            for (decltype(args.size()) j = 0; j < nargs; ++j) {
                // Compute the indices vector for the current derivative.
                tmp_v_idx = v_indices[cur_idx_begin];
                assert(j + 1u < tmp_v_idx.size());
                tmp_v_idx[j + 1u] = boost::safe_numerics::safe<std::uint32_t>(tmp_v_idx[j + 1u]) + 1;

                // Check if we already computed this derivative.
                // NOTE: if the derivative has NOT been computed before and
                // the diff argument is NOT present in the decomposition, then
                // cur_der will remain zero.
                expression cur_der = 0_dbl;
                bool already_computed = false;
                if (auto it = diff_map.find(tmp_v_idx); it != diff_map.end()) {
                    cur_der = it->second;
                    already_computed = true;
                } else if (auto it_dmap = dmap.find(args[j]); it_dmap != dmap.end()) {
                    cur_der = subs(it_dmap->second, subs_map);
                }

                // Add the derivative to the current tensor and diff_map, if needed.
                new_tensor.push_back(cur_der);

                if (!already_computed) {
                    [[maybe_unused]] const auto [_, flag] = diff_map.try_emplace(tmp_v_idx, cur_der);
                    assert(flag);
                }

                // Add the new indices vector
                v_indices.push_back(tmp_v_idx);
            }

            // Update cur_idx_begin as we move to the next output.
            ++cur_idx_begin;
        }

        get_logger()->trace("dtens reverse passes runtime for order {}: {}", cur_order + 1u, sw_inner);

        assert(cur_idx_begin == cur_idx_end);

        // Update cur_idx_begin and cur_idx_end for the next order.
        cur_idx_begin = cur_idx_end;
        cur_idx_end = v_indices.size();
    }

    get_logger()->trace("dtens creation runtime: {}", sw);

    // Assemble and return the result.
    return std::tuple{std::move(diff_map), std::move(tensor_list)};
}

} // namespace

dtens diff_tensors(const std::vector<expression> &v_ex, const std::variant<diff_args, std::vector<expression>> &d_args,
                   std::uint32_t order)
{
    if (v_ex.empty()) {
        throw std::invalid_argument("Cannot compute the derivatives of a function with zero components");
    }

    // Extract/build the diff arguments.
    std::vector<expression> args;

    if (std::holds_alternative<std::vector<expression>>(d_args)) {
        args = std::get<std::vector<expression>>(d_args);
    } else {
        switch (std::get<diff_args>(d_args)) {
            case diff_args::all: {
                // NOTE: this can be simplified once get_variables() returns
                // a list of expressions, rather than strings.
                for (const auto &var : get_variables(v_ex)) {
                    args.emplace_back(var);
                }

                const auto params = get_params(v_ex);
                args.insert(args.end(), params.begin(), params.end());

                break;
            }
            case diff_args::vars:
                for (const auto &var : get_variables(v_ex)) {
                    args.emplace_back(var);
                }

                break;
            case diff_args::params:
                args = get_params(v_ex);

                break;
            default:
                throw std::invalid_argument("An invalid diff_args enumerator was passed to diff_tensors()");
        }
    }

    // Handle empty args.
    if (args.empty()) {
        throw std::invalid_argument("Cannot compute derivatives with respect to an empty set of arguments");
    }

    // Ensure that every expression in args is either a variable
    // or a param.
    if (std::any_of(args.begin(), args.end(), [](const auto &arg) {
            return !std::holds_alternative<variable>(arg.value()) && !std::holds_alternative<param>(arg.value());
        })) {
        throw std::invalid_argument("Derivatives can be computed only with respect to variables and/or parameters");
    }

    // Check if there are repeated entries in args.
    std::unordered_set args_set(args.begin(), args.end());
    if (args_set.size() != args.size()) {
        throw std::invalid_argument(
            "Duplicate entries detected in the list of variables/parameters with respect to which the "
            "derivatives are to be computed");
    }

    auto [diff_map, tensor_list] = diff_tensors_impl(v_ex, args, order);

    return dtens{dtens::impl{std::move(diff_map), std::move(tensor_list)}};
}

} // namespace detail

dtens::dtens(impl x) : p_impl(std::make_unique<impl>(std::move(x))) {}

dtens::dtens() : dtens(impl{}) {}

dtens::dtens(const dtens &other) : dtens(*other.p_impl) {}

dtens::dtens(dtens &&) noexcept = default;

dtens &dtens::operator=(const dtens &other)
{
    if (&other != this) {
        *this = dtens(other);
    }

    return *this;
}

dtens &dtens::operator=(dtens &&) noexcept = default;

dtens::~dtens() = default;

const std::vector<std::vector<expression>> &dtens::get_tensors() const
{
    return p_impl->m_list;
}

const expression &dtens::operator[](const std::vector<std::uint32_t> &vidx) const
{
    const auto it = p_impl->m_map.find(vidx);

    if (it == p_impl->m_map.end()) {
        throw std::invalid_argument(
            fmt::format("Cannot locate the derivative corresponding to the index vector {}", vidx));
    }

    return it->second;
}

HEYOKA_END_NAMESPACE
