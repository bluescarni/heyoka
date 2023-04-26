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
#include <map>
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

#include <boost/container/container_fwd.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
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

    // Rename variables and params.
    const auto v_ex = subs(v_ex_, repl_map);

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

    // Prepare the outputs vector.
    std::vector<expression> outs;
    outs.reserve(nouts);

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition.
    detail::funcptr_map<std::vector<expression>::size_type> func_map;
    for (const auto &ex : v_ex) {
        // Decompose the current component.
        if (const auto dres = detail::decompose(func_map, ex, ret)) {
            // NOTE: if the component was decomposed
            // (that is, it is not constant or a single variable),
            // then the output is a u variable.
            // NOTE: all functions are forced to return
            // a non-empty dres
            // in the func API, so the only entities that
            // can return an empty dres are consts or
            // variables.
            outs.emplace_back(fmt::format("u_{}", *dres));
        } else {
            // NOTE: params have been turned into variables,
            // thus here the only 2 possibilities are variable
            // and number.
            assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value()));

            outs.push_back(ex);
        }
    }

    assert(outs.size() == nouts);

    // Append the definitions of the outputs.
    ret.insert(ret.end(), outs.begin(), outs.end());

    get_logger()->trace("revdiff decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    verify_function_dec(v_ex_, ret, nvars + npars, true);

#endif

    // Simplify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = function_decompose_cse(ret, nvars + npars, nouts);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_function_dec(v_ex_, ret, nvars + npars, true);

#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = function_sort_dc(ret, nvars + npars, nouts);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_function_dec(v_ex_, ret, nvars + npars, true);

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
        assert(subs(dc[i], subs_map) == dc[i]);

        // NOTE: no adjoints or direct/reverse dependencies needed for the initial definitions,
        // we only need to fill in subs_map.
        [[maybe_unused]] const auto flag = subs_map.emplace(fmt::format("u_{}", i), dc[i]).second;

        assert(flag);
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

        // NOTE: when building the substitution map, ensure that
        // subs() canonicalises commutative operators, so that ultimately
        // the result of reverse-mode differentiation will also be canonicalised.
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map, true));
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

} // namespace

bool dtens_v_idx_cmp::operator()(const dtens_v_idx_t &v1, const dtens_v_idx_t &v2) const
{
    assert(v1.size() == v2.size());
    assert(v1.size() > 1u);

    // Compute the total derivative order for both
    // vectors.
    // NOTE: here we have to use safe_numerics because this comparison operator
    // might end up being invoked on a user-supplied v_idx_t, whose total degree
    // may overflow. The v_idx_t in dtens, by contrast, are guaranteed to never
    // overflow when computing the total degree.
    boost::safe_numerics::safe<std::uint32_t> deg1 = 0, deg2 = 0;
    const auto size = v1.size();
    for (decltype(v1.size()) i = 1; i < size; ++i) {
        deg1 += v1[i];
        deg2 += v2[i];
    }

    if (deg1 < deg2) {
        return true;
    }

    if (deg1 > deg2) {
        return false;
    }

    // The total derivative order is the same, look at
    // the component index next.
    if (v1[0] < v2[0]) {
        return true;
    }

    if (v1[0] > v2[0]) {
        return false;
    }

    // Component and total derivative order are the same,
    // resort to reverse lexicographical compare on the
    // derivative orders.
    return std::lexicographical_compare(v1.begin() + 1, v1.end(), v2.begin() + 1, v2.end(), std::greater{});
}

} // namespace detail

struct dtens::impl {
    detail::dtens_map_t m_map;

    // Serialisation.
    // NOTE: this is essentially a manual implementation of serialisation
    // for flat_map, which is currently missing. See:
    // https://stackoverflow.com/questions/69492511/boost-serialize-writing-a-general-map-serialization-function
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        // Serialise the size.
        const auto size = m_map.size();
        ar << size;

        // Serialise the elements.
        for (const auto &p : m_map) {
            ar << p;
        }
    }

    // NOTE: as usual, we assume here that the archive contains
    // a correctly-serialised instance. In particular, we are assuming
    // that the elements in ar are sorted correctly.
    void load(boost::archive::binary_iarchive &ar, unsigned)
    {
        try {
            // Reset m_map.
            m_map.clear();

            // Read the size.
            size_type size = 0;
            ar >> size;

            // Reserve space.
            // NOTE: this is important as it ensures that
            // the addresses of the inserted elements
            // do not change as we insert more elements.
            m_map.reserve(size);

            // Read the elements.
            for (size_type i = 0; i < size; ++i) {
                detail::dtens_map_t::value_type tmp_val;
                ar >> tmp_val;
                const auto it = m_map.insert(m_map.end(), std::move(tmp_val));
                assert(it == m_map.end() - 1);

                // Reset the object address.
                ar.reset_object_address(std::addressof(*it), &tmp_val);
            }

            assert(m_map.size() == size);

            // LCOV_EXCL_START
        } catch (...) {
            *this = impl{};
            throw;
        }
        // LCOV_EXCL_STOP
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

namespace detail
{

namespace
{

auto diff_tensors_impl(const std::vector<expression> &v_ex, const std::vector<expression> &args, std::uint32_t order)
{
    spdlog::stopwatch sw;

    assert(std::all_of(args.begin(), args.end(), [](const auto &arg) {
        return std::holds_alternative<variable>(arg.value()) || std::holds_alternative<param>(arg.value());
    }));
    assert(std::unordered_set(args.begin(), args.end()).size() == args.size());

    // Cache the original number of outputs and the diff arguments.
    const auto orig_nouts = v_ex.size();
    const auto nargs = args.size();

    assert(orig_nouts > 0u);
    assert(nargs > 0u);

    // NOTE: check that nargs fits in a 32-bit int, so that
    // in the dtens API get_nvars() can safely return std::uint32_t.
    (void)(boost::numeric_cast<std::uint32_t>(nargs));

    // Map to associate a vector of indices to a derivative.
    // The first index in each vector is the component index,
    // the rest of the indices are the derivative orders for
    // each diff args. E.g., with diff args = [x, y, z],
    // then [0, 1, 2, 1] means d4f0/(dx dy**2 dz) (where f0 is the
    // first component of the vector function f).
    // Using a std::map is handy for iterative construction, it will
    // be turned into an equivalent flat_map at the end.
    std::map<dtens_v_idx_t, expression, dtens_v_idx_cmp> diff_map;

    // An indices vector with preallocated storage,
    // used as temporary variable in several places below.
    std::vector<std::uint32_t> tmp_v_idx;
    tmp_v_idx.resize(1 + boost::safe_numerics::safe<decltype(tmp_v_idx.size())>(nargs));

    // Init diff_map with the order 0 derivatives
    // (i.e., the original function components).
    for (decltype(v_ex.size()) i = 0; i < orig_nouts; ++i) {
        tmp_v_idx[0] = boost::numeric_cast<std::uint32_t>(i);

        assert(diff_map.count(tmp_v_idx) == 0u);
        diff_map[tmp_v_idx] = v_ex[i];
    }

    // Iterate over the derivative orders.
    for (std::uint32_t cur_order = 0; cur_order < order; ++cur_order) {
        // Locate the iterator in diff_map corresponding to the beginning
        // of the previous-order derivatives.
        tmp_v_idx[0] = 0;
        tmp_v_idx[1] = cur_order;
        std::fill(tmp_v_idx.begin() + 2, tmp_v_idx.end(), static_cast<std::uint32_t>(0));
        auto prev_begin = diff_map.find(tmp_v_idx);
        assert(prev_begin != diff_map.end());

        // Store the previous-order derivatives into a separate
        // vector so that we can construct the decomposition.
        std::vector<expression> prev_diffs;
        std::transform(prev_begin, diff_map.end(), std::back_inserter(prev_diffs),
                       [](const auto &p) { return p.second; });

        // For the purposes of the diff decomposition, the number of outputs
        // is the number of previous derivatives.
        const auto cur_nouts = prev_diffs.size();

        // Run the decomposition on the derivatives of the previous order.
        const auto [dc, nvars] = revdiff_decompose(prev_diffs);

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

            // Add the derivatives to diff_map.
            for (decltype(args.size()) j = 0; j < nargs; ++j) {
                // Compute the indices vector for the current derivative.
                tmp_v_idx = prev_begin->first;
                assert(j + 1u < tmp_v_idx.size());
                // NOTE: no need to overflow check here, because no derivative
                // order can end up being larger than the total diff order which
                // is representable by std::uint32_t.
                tmp_v_idx[j + 1u] += 1u;

                // Check if we already computed this derivative.
                if (const auto it = diff_map.find(tmp_v_idx); it == diff_map.end()) {
                    // The derivative is new. If the diff arguent is present in the
                    // decomposition, then we will calculate the derivative and add it.
                    // Otherwise, we set the derivative to zero and add it.
                    expression cur_der = 0_dbl;

                    if (const auto it_dmap = dmap.find(args[j]); it_dmap != dmap.end()) {
                        // NOTE: when substituting the original variables in the derivative, ensure that
                        // subs() canonicalises commutative operators, so that ultimately
                        // the result of reverse-mode differentiation will also be canonicalised.
                        cur_der = subs(it_dmap->second, subs_map, true);
                    }

                    [[maybe_unused]] const auto [_, flag] = diff_map.try_emplace(tmp_v_idx, cur_der);
                    assert(flag);
                }
            }

            // Update prev_begin as we move to the next output.
            // NOTE; the iterator here stays valid even if we keep on modifying
            // diff_map thanks to the iterator invalidation rules for associative
            // containers (i.e., we never call erase, we just keep on inserting new
            // elements).
            // NOTE: prev_begin is iterating over the previous-order derivatives
            // without interference from the new derivatives we are adding, since
            // they are all higher-order derivatives (and thus appended past the end
            // of the previous order derivative range).
            ++prev_begin;

            assert(prev_begin != diff_map.end());
        }

        get_logger()->trace("dtens reverse passes runtime for order {}: {}", cur_order + 1u, sw_inner);
    }

    get_logger()->trace("dtens creation runtime: {}", sw);

    // Assemble and return the result.
    auto retval = dtens_map_t(boost::container::ordered_unique_range_t{}, diff_map.begin(), diff_map.end());

    // Check sorting.
    assert(std::is_sorted(retval.begin(), retval.end(),
                          [](const auto &p1, const auto &p2) { return dtens_v_idx_cmp{}(p1.first, p2.first); }));
    // Check the number of elements in the indices vectors.
    assert(std::all_of(retval.begin(), retval.end(),
                       [&nargs](const auto &p) { return p.first.size() >= 2u && p.first.size() - 1u == nargs; }));
    // No duplicates in the indices vectors.
    assert(std::adjacent_find(retval.begin(), retval.end(),
                              [](const auto &p1, const auto &p2) { return p1.first == p2.first; })
           == retval.end());

    return retval;
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
            fmt::format("Duplicate entries detected in the list of variables/parameters with respect to which the "
                        "derivatives are to be computed: {}",
                        args));
    }

    return dtens{dtens::impl{diff_tensors_impl(v_ex, args, order)}};
}

} // namespace detail

dtens::subrange::subrange(const iterator &begin, const iterator &end) : m_begin(begin), m_end(end) {}

dtens::subrange::subrange(const subrange &) = default;

dtens::subrange::subrange(subrange &&) noexcept = default;

dtens::subrange &dtens::subrange::operator=(const subrange &) = default;

dtens::subrange &dtens::subrange::operator=(subrange &&) noexcept = default;

dtens::iterator dtens::subrange::begin() const
{
    return m_begin;
}

dtens::iterator dtens::subrange::end() const
{
    return m_end;
}

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

dtens::iterator dtens::begin() const
{
    return p_impl->m_map.begin();
}

dtens::iterator dtens::end() const
{
    return p_impl->m_map.end();
}

std::uint32_t dtens::get_order() const
{
    if (p_impl->m_map.empty()) {
        return 0;
    }

    // We can fetch the total derivative
    // order from the last derivative
    // in the map (specifically, it is
    // the last element in the indices
    // vector of the last derivative).
    return (end() - 1)->first.back();
}

dtens::iterator dtens::find(const v_idx_t &vidx) const
{
    // NOTE: we need to sanity check vidx as the
    // custom comparison operator for the internal map
    // has preconditions.

    // First we handle the empty case.
    if (p_impl->m_map.empty()) {
        return end();
    }

    // Second, we check that the size of vidx is correct.
    if (begin()->first.size() != vidx.size()) {
        return end();
    }

    assert(vidx.size() > 1u);

    // vidx is ok, look it up.
    return p_impl->m_map.find(vidx);
}

const expression &dtens::operator[](const v_idx_t &vidx) const
{
    const auto it = find(vidx);

    if (it == end()) {
        throw std::out_of_range(
            fmt::format("Cannot locate the derivative corresponding to the indices vector {}", vidx));
    }

    return it->second;
}

// Get a range containing all derivatives of the given order for all components.
dtens::subrange dtens::get_derivatives(std::uint32_t order) const
{
    // First we handle the empty case. This will return
    // an empty range.
    if (p_impl->m_map.empty()) {
        return subrange{begin(), end()};
    }

    // Create the indices vector corresponding to the first derivative
    // of component 0 for the given order in the map.
    auto vidx = begin()->first;
    assert(std::all_of(vidx.begin(), vidx.end(), [](auto x) { return x == 0u; }));
    vidx[1] = order;

    // Locate the corresponding derivative in the map.
    // NOTE: this could be end() for invalid order.
    const auto b = p_impl->m_map.find(vidx);

#if !defined(NDEBUG)

    if (order <= get_order()) {
        assert(b != end());
    } else {
        assert(b == end());
    }

#endif

    // Modify vidx so that it now refers to the last derivative
    // for the last component at the given order in the map.
    // NOTE: get_nouts() can return zero only if the internal
    // map is empty, and we handled this corner case earlier.
    assert(get_nouts() > 0u);
    vidx[0] = get_nouts() - 1u;
    vidx[1] = 0;
    vidx.back() = order;
    // NOTE: this could be end() for invalid order.
    auto e = p_impl->m_map.find(vidx);

#if !defined(NDEBUG)

    if (order <= get_order()) {
        assert(e != end());
    } else {
        assert(e == end());
    }

#endif

    // Need to move 1 past, if possible,
    // to produce a half-open range.
    if (e != end()) {
        ++e;
    }

    return subrange{b, e};
}

// Get a range containing all derivatives of the given order for a component.
dtens::subrange dtens::get_derivatives(std::uint32_t component, std::uint32_t order) const
{
    // First we handle the empty case. This will return
    // an empty range.
    if (p_impl->m_map.empty()) {
        return subrange{begin(), end()};
    }

    // Create the indices vector corresponding to the first derivative
    // for the given order and component in the map.
    auto vidx = begin()->first;
    assert(std::all_of(vidx.begin(), vidx.end(), [](auto x) { return x == 0u; }));
    vidx[0] = component;
    vidx[1] = order;

    // Locate the corresponding derivative in the map.
    // NOTE: this could be end() for invalid component/order.
    const auto b = p_impl->m_map.find(vidx);

#if !defined(NDEBUG)

    if (component < get_nouts() && order <= get_order()) {
        assert(b != end());
    } else {
        assert(b == end());
    }

#endif

    // Modify vidx so that it now refers to the last derivative
    // for the given order and component in the map.
    vidx[1] = 0;
    vidx.back() = order;
    // NOTE: this could be end() for invalid component/order.
    auto e = p_impl->m_map.find(vidx);

#if !defined(NDEBUG)

    if (component < get_nouts() && order <= get_order()) {
        assert(e != end());
    } else {
        assert(e == end());
    }

#endif

    // Need to move 1 past, if possible,
    // to produce a half-open range.
    if (e != end()) {
        ++e;
    }

    return subrange{b, e};
}

std::uint32_t dtens::get_nvars() const
{
    if (p_impl->m_map.empty()) {
        return 0;
    }

    // NOTE: we ensure in the diff_tensors() implementation
    // that the number of diff variables is representable
    // by std::uint32_t.
    return static_cast<std::uint32_t>(begin()->first.size() - 1u);
}

std::uint32_t dtens::get_nouts() const
{
    if (p_impl->m_map.empty()) {
        return 0;
    }

    // Construct the indices vector corresponding
    // to the first derivative of order 1 of the first component.
    auto vidx = begin()->first;
    assert(std::all_of(vidx.begin(), vidx.end(), [](auto x) { return x == 0u; }));
    vidx[1] = 1;

    // Try to find it in the map.
    const auto it = p_impl->m_map.find(vidx);

    // NOTE: the number of outputs is always representable by
    // std::uint32_t, otherwise we could not index the function
    // components via std::uint32_t.
    if (it == end()) {
        // There are no derivatives in the map, which
        // means that the order must be zero and that the
        // size of the map gives directly the number of components.
        assert(get_order() == 0u);
        return static_cast<std::uint32_t>(p_impl->m_map.size());
    } else {
        assert(get_order() > 0u);
        return static_cast<std::uint32_t>(p_impl->m_map.index_of(it));
    }
}

dtens::size_type dtens::size() const
{
    return p_impl->m_map.size();
}

void dtens::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << p_impl;
}

void dtens::load(boost::archive::binary_iarchive &ar, unsigned)
{
    try {
        ar >> p_impl;
        // LCOV_EXCL_START
    } catch (...) {
        *this = dtens{};
        throw;
    }
    // LCOV_EXCL_STOP
}

HEYOKA_END_NAMESPACE
