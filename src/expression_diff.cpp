// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
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
#include <boost/container_hash/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dtens_impl.hpp>
#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/tbb_isolated.hpp>
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

namespace
{

// Implementation of expression diff wrt a variable or a parameter.
//
// NOTE: this is quite similar to ex_traverse_transform_nodes(), but unfortunately we need access to the original
// arguments of the function in order to compute derivatives via the chain rule, and ex_traverse_transform_nodes()
// instead applies the branch node transformation function *after* the arguments have already been replaced with their
// derivatives. Hence, we have a custom implementation of this traversal. Perhaps in the future we can try to generalise
// ex_traverse_transform_nodes() for this use case.
template <typename T>
    requires std::same_as<T, std::string> || std::same_as<T, param>
expression diff_impl(void_ptr_map<const expression> &func_map, sargs_ptr_map<const func_args::shared_args_t> &sargs_map,
                     const expression &e, const T &diff_arg)
{
    // The leaf transformation function.
    const auto leaf_tfunc = [&diff_arg](const expression &e) {
        if (const auto *num_ptr = std::get_if<number>(&e.value())) {
            // The leaf is a number, its derivative is always zero. Make sure to create the zero with the correct type.
            return std::visit([]<typename U>(const U &) { return expression{number{static_cast<U>(0)}}; },
                              num_ptr->value());
        } else if (const auto *par_ptr = std::get_if<param>(&e.value())) {
            // The leaf is a param.
            if constexpr (std::same_as<T, param>) {
                // We are differentiating wrt a param.
                if (par_ptr->idx() == diff_arg.idx()) {
                    return 1_dbl;
                } else {
                    return 0_dbl;
                }
            } else {
                // We are differentiating wrt a variable.
                return 0_dbl;
            }
        } else {
            // The leaf is a variable.
            const auto &var = std::get<variable>(e.value());

            if constexpr (std::same_as<T, param>) {
                // We are differentiating wrt a param.
                return 0_dbl;
            } else {
                // We are differentiating wrt a variable.
                if (var.name() == diff_arg) {
                    return 1_dbl;
                } else {
                    return 0_dbl;
                }
            }
        }
    };

    traverse_stack stack;
    return_stack<expression> out_stack;

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the traversal stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        if (const auto *f_ptr = std::get_if<func>(&cur_ex->value())) {
            // Function (i.e., internal) node.
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            if (visited) {
                // NOTE: if this is the second visit, we know that the the function cannot possibly be in the cache,
                // and thus we can avoid an unnecessary lookup.
                assert(!func_map.contains(f_id));
            } else if (const auto it = func_map.find(f_id); it != func_map.end()) {
                // We already differentiated the current function. Fetch the result from the cache and add it to
                // out_stack.
                out_stack.emplace_back(it->second);
                continue;
            }

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now visited and differentiated all the children of the function node (i.e., the function
                // arguments). The results are at the tail end of out_stack. We will be popping them from out_stack and
                // use them to differentiate the function.

                // Build the set of differentiated arguments.
                std::vector<expression> diff_args;
                const auto n_args = f.args().size();
                diff_args.reserve(n_args);
                for (decltype(diff_args.size()) i = 0; i < n_args; ++i) {
                    // NOTE: out_stack must not be empty and its last element also cannot be empty.
                    assert(!out_stack.empty());
                    assert(out_stack.back());

                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    diff_args.push_back(std::move(*out_stack.back()));
                    out_stack.pop_back();
                }

                // Compute the gradient of the function wrt the original arguments.
                const auto grad = f.gradient();

                // Build the arguments for the sum of derivatives (i.e., the total derivative).
                std::vector<expression> sum_args;
                sum_args.reserve(n_args);
                for (decltype(sum_args.size()) i = 0; i < n_args; ++i) {
                    sum_args.push_back(grad[i] * diff_args[i]);
                }

                if (shared_args) {
                    // NOTE: if the function manages its arguments via a shared reference, we must make
                    // sure to record the differentiated arguments in sargs_map, so that when we run again into the
                    // same shared reference we re-use the cached result.
                    auto new_sargs = std::make_shared<const std::vector<expression>>(std::move(diff_args));

                    assert(!sargs_map.contains(&*shared_args)); // LCOV_EXCL_LINE
                    sargs_map.emplace(&*shared_args, new_sargs);
                }

                // Create the derivative of the function.
                auto ex_diff = sum(std::move(sum_args));

                // Add it to the cache.
                func_map.emplace(f_id, ex_diff);

                // Add it to out_stack.
                // NOTE: out_stack must not be empty and its last element must be empty (it is supposed to be
                // the empty function we pushed the first time we visited).
                assert(!out_stack.empty());
                assert(!out_stack.back());
                out_stack.back().emplace(std::move(ex_diff));
            } else {
                // It is the first time we visit this function.
                if (shared_args) {
                    // The function manages its arguments via a shared reference. Check if we already differentiated the
                    // arguments before.
                    if (const auto it = sargs_map.find(&*shared_args); it != sargs_map.end()) {
                        // The arguments have been differentiated before. Fetch them from the cache and
                        // use them to compute the derivative of the function.
                        const auto &diff_args = *it->second;
                        const auto n_args = diff_args.size();

                        // Compute the gradient of the function wrt the original arguments.
                        const auto grad = f.gradient();
                        assert(n_args == grad.size()); // LCOV_EXCL_LINE

                        // Build the arguments for the sum of derivatives (i.e., the total derivative).
                        std::vector<expression> sum_args;
                        sum_args.reserve(n_args);
                        for (decltype(sum_args.size()) i = 0; i < n_args; ++i) {
                            sum_args.push_back(grad[i] * diff_args[i]);
                        }

                        // Create the derivative of the function.
                        auto ex_diff = sum(std::move(sum_args));

                        // Add the new function to the cache and to out_stack.
                        func_map.emplace(f_id, ex_diff);
                        out_stack.emplace_back(std::move(ex_diff));

                        continue;
                    }

                    // NOTE: if we arrive here, it means that the shared arguments of the function have never
                    // been differentiated before. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its
                // arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }

                // Add an empty expression to out_stack. We will create its derivative once we have differentiated all
                // arguments.
                out_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            // Differentiate the leaf node and add the result to out_stack.
            out_stack.emplace_back(leaf_tfunc(*cur_ex));
        }
    }

    assert(out_stack.size() == 1u);
    assert(out_stack.back());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return std::move(*out_stack.back());
}

} // namespace

} // namespace detail

expression diff(const expression &e, const std::string &s)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    return detail::diff_impl(func_map, sargs_map, e, s);
}

expression diff(const expression &e, const param &p)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    return detail::diff_impl(func_map, sargs_map, e, p);
}

namespace detail
{

namespace
{

std::vector<expression> diff_vec_impl(const std::vector<expression> &v_ex, const auto &x)
{
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        retval.push_back(diff_impl(func_map, sargs_map, ex, x));
    }

    return retval;
}

template <typename T>
T diff_ex_impl(const T &input, const expression &x)
{
    return std::visit(
        [&input](const auto &v) -> T {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(v)>, variable>) {
                return diff(input, v.name());
            } else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(v)>, param>) {
                return diff(input, v);
            } else {
                throw std::invalid_argument(
                    "Derivatives are currently supported only with respect to variables and parameters");
            }
        },
        x.value());
}

} // namespace

} // namespace detail

std::vector<expression> diff(const std::vector<expression> &v_ex, const std::string &s)
{
    return detail::diff_vec_impl(v_ex, s);
}

std::vector<expression> diff(const std::vector<expression> &v_ex, const param &p)
{
    return detail::diff_vec_impl(v_ex, p);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
expression diff(const expression &e, const expression &x)
{
    return detail::diff_ex_impl(e, x);
}

std::vector<expression> diff(const std::vector<expression> &v_ex, const expression &x)
{
    return detail::diff_ex_impl(v_ex, x);
}

namespace detail
{

// Function decomposition for symbolic differentiation.
std::pair<std::vector<expression>, std::vector<expression>::size_type>
diff_decompose(const std::vector<expression> &v_ex_)
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
    std::map<expression, expression> repl_map;
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

    // NOTE: split prods into binary mults. In reverse-mode AD,
    // n-ary products slow down performance because the complexity
    // of each adjoint increases linearly with the number of arguments,
    // leading to quadratic complexity in the reverse pass. By contrast,
    // the adjoints of binary products have fixed complexity and the number
    // of binary multiplications necessary to reconstruct an n-ary product
    // increases only linearly with the number of arguments.
    auto v_ex = detail::split_prods_for_decompose(v_ex_, 2u);

#if !defined(NDEBUG)

    // Save copy for checking in debug mode.
    const auto v_ex_verify = v_ex;

#endif

    // Rename variables and params.
    v_ex = subs(v_ex, repl_map);

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
    detail::void_ptr_map<const std::vector<expression>::size_type> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;
    for (const auto &ex : v_ex) {
        // Decompose the current component.
        if (const auto dres = detail::decompose(func_map, sargs_map, ex, ret)) {
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

    get_logger()->trace("diff decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    verify_function_dec(v_ex_verify, ret, nvars + npars, true);

#endif

    // Simplify the decomposition.
    // NOTE: nvars + npars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars + npars items.
    ret = function_decompose_cse(ret, nvars + npars, nouts);

#if !defined(NDEBUG)

    // Verify the decomposition.
    verify_function_dec(v_ex_verify, ret, nvars + npars, true);

#endif

    // NOTE: we do not run here the topological sorting like we do in cfuncs and Taylor integrators, as I do not see at
    // this time any benefit in doing so.
    return {std::move(ret), nvars + npars};
}

namespace
{

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto diff_make_adj_dep(const std::vector<expression> &dc, std::vector<expression>::size_type nvars,
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
    std::vector<boost::unordered_flat_map<std::uint32_t, expression>> adj;
    adj.resize(boost::numeric_cast<decltype(adj.size())>(dc.size()));

    std::vector<std::vector<std::uint32_t>> dep;
    dep.resize(boost::numeric_cast<decltype(dep.size())>(dc.size()));

    std::vector<std::vector<std::uint32_t>> revdep;
    revdep.resize(boost::numeric_cast<decltype(revdep.size())>(dc.size()));

    std::unordered_map<std::string, expression> subs_map;

    // Initial definitions.
    for (idx_t i = 0; i < nvars; ++i) {
        // The initial definitions must all consist of
        // variables or parameters.
        assert(std::holds_alternative<variable>(dc[i].value()) || std::holds_alternative<param>(dc[i].value()));

        // NOTE: no adjoints or direct/reverse dependencies needed for the initial definitions,
        // we only need to fill in subs_map.
        assert(!subs_map.contains(fmt::format("u_{}", i)));
        subs_map.emplace(fmt::format("u_{}", i), dc[i]);
    }

    // NOTE: this map is used in the next loop (see comments there).
    // NOTE: as usual, use a sorted map (instead of a hash map) in order
    // to avoid non-deterministic ordering of operations.
    // NOTE: if this turns out to be a bottleneck, we can always
    // re-implement in terms of boost::unordered_flat_map and manually sort after construction.
    std::map<std::string, std::vector<expression>> grad_map;

    // NOTE: these are caches used in the construction of subs_map.
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    // Elementary subexpressions.
    for (idx_t i = nvars; i < dc.size() - nouts; ++i) {
        // Reset grad_map.
        grad_map.clear();

        // Fetch references to the current dict of adjoints
        // and the direct deps vector.
        auto &cur_adj_dict = adj[i];
        auto &cur_dep = dep[i];

        // The elementary subexpressions must all be functions.
        assert(std::holds_alternative<func>(dc[i].value()));
        const auto &fn = std::get<func>(dc[i].value());

        // Fetch the gradient of the subexpression wrt its arguments.
        auto grad = fn.gradient();
        assert(grad.size() == fn.args().size());

        // Fill in grad_map: for each variable in the arguments of the subexpression,
        // compute the list of associated partial derivatives. For instance, if the subexpression
        // is something like f(u_0, 1.0, u_0, u_1), there will be two partial derivatives
        // for the variable u_0: one containing the derivative of f wrt the first argument,
        // the other containing the derivative wrt the third argument.
        // NOTE: we are doing all this in order to compute (in the next loop) the total
        // derivatives of the subexpression wrt its variables. However, instead of just using
        // diff() for this, we are re-implementing manually the total derivative calculation
        // in an effort to avoid quadratic complexity in case of functions with many arguments.
        for (decltype(grad.size()) arg_idx = 0; arg_idx < grad.size(); ++arg_idx) {
            const auto &arg = fn.args()[arg_idx];

            if (const auto *var_ptr = std::get_if<variable>(&arg.value())) {
                // The argument is a variable.
                grad_map[var_ptr->name()].push_back(std::move(grad[arg_idx]));
            } else {
                // The only other possibility is that the argument is a number.
                assert(std::holds_alternative<number>(arg.value()));
            }
        }

        // Fill in the adjoints and the direct/reverse dependencies.
        for (auto &[var, pdiffs] : grad_map) {
            const auto idx = uname_to_index(var);

            // NOTE: this is the computation of the total derivative.
            assert(!cur_adj_dict.contains(idx));
            cur_adj_dict[idx] = sum(std::move(pdiffs));

            assert(idx < revdep.size());
            revdep[idx].push_back(boost::numeric_cast<std::uint32_t>(i));

            cur_dep.push_back(idx);
        }

        assert(!subs_map.contains(fmt::format("u_{}", i)));
        subs_map.emplace(fmt::format("u_{}", i),
                         // NOTE: the point of using subs_impl() with the caches, rather than subs(), is that we want
                         // to avoid creating multiple copies of shared arguments.
                         subs_impl(func_map, sargs_map, dc[i], subs_map));
    }

    // Outputs.
    for (idx_t i = dc.size() - nouts; i < dc.size(); ++i) {
        if (const auto *var_ptr = std::get_if<variable>(&dc[i].value())) {
            // The current output is a variable.
            auto &cur_adj_dict = adj[i];
            auto &cur_dep = dep[i];

            // The variable index must preceed the index of the first output.
            const auto idx = uname_to_index(var_ptr->name());
            assert(idx < dc.size() - nouts);

            assert(cur_adj_dict.empty());
            cur_adj_dict[idx] = 1_dbl;

            assert(idx < revdep.size());
            revdep[idx].push_back(boost::numeric_cast<std::uint32_t>(i));

            assert(cur_dep.empty());
            cur_dep.push_back(idx);

            assert(subs_map.contains(var_ptr->name()));
            assert(!subs_map.contains(fmt::format("u_{}", i)));
            subs_map.emplace(fmt::format("u_{}", i), subs_map.find(var_ptr->name())->second);
        } else {
            // The outputs must all be variables or numbers.
            assert(std::holds_alternative<number>(dc[i].value()));

            // The current output is a number.
            assert(!subs_map.contains(fmt::format("u_{}", i)));
            subs_map.emplace(fmt::format("u_{}", i), dc[i]);
        }
    }

    // Sort the vectors of reverse dependencies.
    // NOTE: this is not strictly necessary for the correctness
    // of the algorithm. It will just ensure that when we eventually
    // compute the derivative of the output wrt a subexpression, the
    // summation over the reverse dependencies happens in index order.
    // NOTE: should we do this for the direct deps as well?
    // NOTE: this can be easily parallelised if needed.
    for (auto &rvec : revdep) {
        std::ranges::sort(rvec);

        // Check that there are no duplicates.
        assert(std::ranges::adjacent_find(rvec) == rvec.end());
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

// This is an alternative version of dtens_sv_idx_t that uses a dictionary
// for storing the index/order pairs instead of a sorted vector. Using a dictionary
// allows for faster/easier manipulation.
using dtens_ss_idx_t = std::pair<std::uint32_t, boost::unordered_flat_map<std::uint32_t, std::uint32_t>>;

// Helper to turn a dtens_sv_idx_t into a dtens_ss_idx_t.
void vidx_v2s(dtens_ss_idx_t &output, const dtens_sv_idx_t &input)
{
    // Assign the component.
    output.first = input.first;

    // Assign the index/order pairs.
    output.second.clear();
    for (const auto &p : input.second) {
        [[maybe_unused]] auto [_, flag] = output.second.insert(p);
        assert(flag);
    }
}

// Helper to turn a dtens_ss_idx_t into a dtens_sv_idx_t.
dtens_sv_idx_t vidx_s2v(const dtens_ss_idx_t &input)
{
    // Init retval.
    dtens_sv_idx_t retval{input.first, {input.second.begin(), input.second.end()}};

    // Sort the index/order pairs.
    std::ranges::sort(retval.second, [](const auto &p1, const auto &p2) { return p1.first < p2.first; });

    return retval;
} // LCOV_EXCL_LINE

// Hasher for the local maps of derivatives used in the
// forward/reverse mode implementations.
struct diff_map_hasher {
    std::size_t operator()(const dtens_ss_idx_t &s) const noexcept
    {
        // Use as seed the component index.
        std::size_t seed = std::hash<std::uint32_t>{}(s.first);

        // Compose via additions the hashes of the index/order pairs.
        // NOTE: it is important that we use here a commutative operation
        // for the composition so that the final hash is independent of the order
        // in which the pairs are stored in the dictionary.
        for (const auto &p : s.second) {
            std::size_t p_hash = std::hash<std::uint32_t>{}(p.first);
            boost::hash_combine(p_hash, std::hash<std::uint32_t>{}(p.second));

            // NOTE: make sure there's not funny promotion business going
            // on with std::size_t.
            static_assert(std::is_same_v<std::size_t, std::common_type_t<std::size_t, std::size_t>>);
            seed += p_hash;
        }

        return seed;
    }
};

// Forward-mode implementation of diff_tensors().
// NOTE: the parallelisation of the forward/reverse mode implementations does not seem
// too difficult, as the iterations over the inputs/outputs are independent from each other.
// The real question is how to handle the duplicates that can arise for order > 1:
// either we accept them and we remove them *after* running the impls via
// sort + std::unique + remove (but what are the consequences
// in terms of memory utilisation?), or we try to accumulate the derivatives
// into a global thread-safe map. In the latter case, bulk insertion from local
// maps could perhaps perform well.
// NOTE: also, in parallel mode fetch const versions of diff_map with std::as_const()
// during parallel operations.
template <typename DiffMap, typename Dep, typename Adj>
void diff_tensors_forward_impl(
    // The map of derivatives. It will be updated after all the
    // derivatives have been computed.
    DiffMap &diff_map,
    // The number of derivatives in the previous-order tensor.
    const std::vector<expression>::size_type cur_nouts,
    // The decomposition of the previous-order tensor.
    const std::vector<expression> &dc,
    // The direct and reverse dependencies for the
    // subexpressions in dc.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Dep &dep, const Dep &revdep,
    // The adjoints of the subexpressions in dc.
    const Adj &adj,
    // The total number of variables in dc (this accounts also
    // for params, as they are turned into variables during the
    // construction of the decomposition).
    const std::vector<expression>::size_type nvars,
    // The diff arguments.
    const std::vector<expression> &args,
    // Iterator in diff_map pointing to the first
    // derivative for the previous order.
    const typename DiffMap::iterator &prev_begin,
    // The current derivative order.
    const std::uint32_t cur_order)
{
    assert(dc.size() > nvars);
    assert(cur_order > 0u);

    // Create a dictionary mapping an input to its position
    // in the decomposition. This is used to locate diff arguments
    // in the decomposition.
    const auto input_idx_map = [&]() {
        boost::unordered_flat_map<expression, std::vector<expression>::size_type, std::hash<expression>> retval;

        for (std::vector<expression>::size_type i = 0; i < nvars; ++i) {
            const auto &cur_in = dc[i];
            assert(!retval.contains(cur_in));
            retval[cur_in] = i;
        }

        return retval;
    }();

    // Local data structures used to temporarily store the derivatives,
    // which will eventually be added to diff_map.
    // For derivative orders > 1, the algorithm we employ
    // will produce several times the same derivative, and thus
    // we need to store the derivatives in a dictionary in order
    // to prevent duplicates. For order-1 derivatives, no duplicate
    // derivatives will be produced and thus we can use a plain vector,
    // which can be quite a bit faster.
    using diff_map_t = boost::unordered_flat_map<dtens_ss_idx_t, expression, diff_map_hasher>;
    using diff_vec_t = std::vector<std::pair<dtens_ss_idx_t, expression>>;
    using local_diff_t = std::variant<diff_map_t, diff_vec_t>;
    auto local_diff = (cur_order == 1u) ? local_diff_t(diff_vec_t{}) : local_diff_t(diff_map_t{});

    // Helpers to ease the access to the active member of the local_diff variant.
    // NOTE: if used incorrectly, these will throw at runtime.
    auto local_dmap = [&local_diff]() -> diff_map_t & { return std::get<diff_map_t>(local_diff); };
    auto local_dvec = [&local_diff]() -> diff_vec_t & { return std::get<diff_vec_t>(local_diff); };

    // This is used as a temporary variable in several places below.
    dtens_ss_idx_t tmp_v_idx;

    // These two containers will be used to store the list of subexpressions
    // which depend on an input. They are used in the forward pass
    // to avoid iterating over those subexpressions which do not depend on
    // an input. We need two containers (with identical content)
    // because we need both ordered iteration AND fast lookup.
    boost::unordered_flat_set<std::uint32_t> in_deps;
    std::vector<std::uint32_t> sorted_in_deps;

    // A stack to be used when filling up in_deps/sorted_in_deps.
    std::deque<std::uint32_t> stack;

    // Vector of expressions used to accumulate sums during the forward pass.
    std::vector<expression> tmp_sum;

    // Run the forward pass for each diff argument. The derivatives
    // wrt the diff argument will be stored into diffs.
    std::vector<expression> diffs(dc.size());
    for (decltype(args.size()) diff_arg_idx = 0; diff_arg_idx < args.size(); ++diff_arg_idx) {
        const auto &cur_diff_arg = args[diff_arg_idx];

        // Check if the current diff argument is one of the inputs.
        const auto input_it = input_idx_map.find(cur_diff_arg);
        if (input_it == input_idx_map.end()) {
            // The diff argument is not one of the inputs:
            // set the derivatives of all outputs wrt to the
            // diff argument to zero.
            auto out_it = prev_begin;

            for (std::vector<expression>::size_type out_idx = 0; out_idx < cur_nouts; ++out_idx, ++out_it) {
                assert(out_it != diff_map.end());

                vidx_v2s(tmp_v_idx, out_it->first);
                tmp_v_idx.second[static_cast<std::uint32_t>(diff_arg_idx)] += 1u;

                if (cur_order == 1u) {
                    local_dvec().emplace_back(tmp_v_idx, 0_dbl);
                } else {
                    // NOTE: use try_emplace() so that if the derivative
                    // has already been computed, nothing happens.
                    local_dmap().try_emplace(tmp_v_idx, 0_dbl);
                }
            }

            // Move to the next diff argument.
            continue;
        }

        // The diff argument is one of the inputs. Fetch its index.
        const auto input_idx = input_it->second;

        // Seed the stack and in_deps/sorted_in_deps with the
        // dependees of the current input.
        stack.assign(revdep[input_idx].begin(), revdep[input_idx].end());
        sorted_in_deps.assign(revdep[input_idx].begin(), revdep[input_idx].end());
        in_deps.clear();
        in_deps.insert(revdep[input_idx].begin(), revdep[input_idx].end());

        // Build in_deps/sorted_in_deps by traversing
        // the decomposition forward.
        while (!stack.empty()) {
            // Pop the first element from the stack.
            const auto cur_idx = stack.front();
            stack.pop_front();

            // Push into the stack and in_deps/sorted_in_deps
            // the dependees of cur_idx.
            for (const auto next_idx : revdep[cur_idx]) {
                // NOTE: if next_idx is already in in_deps,
                // it means that it was visited already and thus
                // it does not need to be put in the stack.
                if (in_deps.count(next_idx) == 0u) {
                    stack.push_back(next_idx);
                    sorted_in_deps.push_back(next_idx);
                    in_deps.insert(next_idx);
                }
            }
        }

        // Sort sorted_in_deps in ascending order.
        std::ranges::sort(sorted_in_deps);

        // sorted_in_deps cannot have duplicate values.
        assert(std::ranges::adjacent_find(sorted_in_deps) == sorted_in_deps.end());
        // sorted_in_deps either must be empty, or its last index
        // must refer to an output (i.e., the current input must be
        // a dependency for some output).
        assert(sorted_in_deps.empty() || *sorted_in_deps.rbegin() >= diffs.size() - cur_nouts);
        assert(sorted_in_deps.size() == in_deps.size());

        // Set the seed value for the current input.
        diffs[input_idx] = 1_dbl;

        // Set the derivatives of all outputs to zero, so that if
        // an output does not depend on the current input then the
        // derivative of that output wrt the current input is pre-emptively
        // set to zero.
        std::fill(diffs.data() + diffs.size() - cur_nouts, diffs.data() + diffs.size(), 0_dbl);

        // Run the forward pass.
        for (const auto cur_idx : sorted_in_deps) {
            tmp_sum.clear();

            for (const auto d_idx : dep[cur_idx]) {
                assert(d_idx < diffs.size());
                assert(cur_idx < adj.size());
                // NOTE: the dependency must point
                // to a subexpression *before* the current one.
                assert(d_idx < cur_idx);
                assert(adj[cur_idx].count(d_idx) == 1u);

                // NOTE: if the current subexpression depends
                // on another subexpression which neither is
                // the current input nor depends on the current input,
                // then the derivative is zero.
                if (d_idx != input_idx && in_deps.count(d_idx) == 0u) {
                    tmp_sum.push_back(0_dbl);
                } else {
                    auto new_term = diffs[d_idx] * adj[cur_idx].find(d_idx)->second;
                    tmp_sum.push_back(std::move(new_term));
                }
            }

            assert(!tmp_sum.empty());

            diffs[cur_idx] = sum(tmp_sum);
        }

        // Add the derivatives of all outputs wrt the current input
        // to the local map.
        auto out_it = prev_begin;

        for (std::vector<expression>::size_type out_idx = 0; out_idx < cur_nouts; ++out_idx, ++out_it) {
            assert(out_it != diff_map.end());

            vidx_v2s(tmp_v_idx, out_it->first);
            tmp_v_idx.second[static_cast<std::uint32_t>(diff_arg_idx)] += 1u;

            if (cur_order == 1u) {
                auto cur_der = diffs[diffs.size() - cur_nouts + out_idx];
                local_dvec().emplace_back(tmp_v_idx, std::move(cur_der));
            } else {
                // Check if we already computed this derivative.
                if (const auto it = local_dmap().find(tmp_v_idx); it == local_dmap().end()) {
                    // The derivative is new.
                    auto cur_der = diffs[diffs.size() - cur_nouts + out_idx];

                    [[maybe_unused]] const auto [_, flag] = local_dmap().try_emplace(tmp_v_idx, std::move(cur_der));
                    assert(flag);
                }
            }
        }
    }

    // Merge the local map into diff_map.
    if (cur_order == 1u) {
        for (auto &p : local_dvec()) {
            diff_map.emplace_back(vidx_s2v(p.first), std::move(p.second));
        }
    } else {
        for (auto &p : local_dmap()) {
            diff_map.emplace_back(vidx_s2v(p.first), std::move(p.second));
        }
    }
}

// Reverse-mode implementation of diff_tensors().
template <typename DiffMap, typename Dep, typename Adj>
void diff_tensors_reverse_impl(
    // The map of derivatives. It will be updated after all the
    // derivatives have been computed.
    DiffMap &diff_map,
    // The number of derivatives in the previous-order tensor.
    const std::vector<expression>::size_type cur_nouts,
    // The decomposition of the previous-order tensor.
    const std::vector<expression> &dc,
    // The direct and reverse dependencies for the
    // subexpressions in dc.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Dep &dep, const Dep &revdep,
    // The adjoints of the subexpressions in dc.
    const Adj &adj,
    // The total number of variables in dc (this accounts also
    // for params, as they are turned into variables during the
    // construction of the decomposition).
    const std::vector<expression>::size_type nvars,
    // The diff arguments.
    const std::vector<expression> &args,
    // Iterator in diff_map pointing to the first
    // derivative for the previous order.
    // NOTE: this needs to remain mutable, need to understand
    // how to deal with this in case we attempt a parallel implementation.
    typename DiffMap::iterator prev_begin,
    // The current derivative order.
    const std::uint32_t cur_order)
{
    assert(dc.size() > nvars);
    assert(cur_order > 0u);

    // Local data structures used to temporarily store the derivatives,
    // which will eventually be added to diff_map.
    // For derivative orders > 1, the algorithm we employ
    // will produce several times the same derivative, and thus
    // we need to store the derivatives in a dictionary in order
    // to prevent duplicates. For order-1 derivatives, no duplicate
    // derivatives will be produced and thus we can use a plain vector,
    // which can be quite a bit faster.
    using diff_map_t = boost::unordered_flat_map<dtens_ss_idx_t, expression, diff_map_hasher>;
    using diff_vec_t = std::vector<std::pair<dtens_ss_idx_t, expression>>;
    using local_diff_t = std::variant<diff_map_t, diff_vec_t>;
    auto local_diff = (cur_order == 1u) ? local_diff_t(diff_vec_t{}) : local_diff_t(diff_map_t{});

    // Helpers to ease the access to the active member of the local_diff variant.
    // NOTE: if used incorrectly, these will throw at runtime.
    // NOTE: currently local_dmap is never used because the heuristic
    // for deciding between forward and reverse mode prevents reverse mode
    // from being used for order > 1.
    auto local_dmap = [&local_diff]() -> diff_map_t & { return std::get<diff_map_t>(local_diff); }; // LCOV_EXCL_LINE
    auto local_dvec = [&local_diff]() -> diff_vec_t & { return std::get<diff_vec_t>(local_diff); };

    // Cache the number of diff arguments.
    const auto nargs = args.size();

    // This is used as a temporary variable in several places below.
    dtens_ss_idx_t tmp_v_idx;

    // These two containers will be used to store the list of subexpressions
    // on which an output depends. They are used in the reverse pass
    // to avoid iterating over those subexpressions on which the output
    // does not depend (recall that the decomposition contains the subexpressions
    // for ALL outputs). We need two containers (with identical content)
    // because we need both ordered iteration AND fast lookup.
    boost::unordered_flat_set<std::uint32_t> out_deps;
    std::vector<std::uint32_t> sorted_out_deps;

    // A stack to be used when filling up out_deps/sorted_out_deps.
    std::deque<std::uint32_t> stack;

    // Vector of expressions used to accumulate sums during the forward pass.
    std::vector<expression> tmp_sum;

    // Run the reverse pass for each output. The derivatives
    // wrt the output will be stored into diffs.
    std::vector<expression> diffs(dc.size());
    for (std::vector<expression>::size_type i = 0; i < cur_nouts; ++i) {
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
        std::ranges::sort(sorted_out_deps, std::greater{});

        // sorted_out_deps cannot have duplicate values.
        assert(std::ranges::adjacent_find(sorted_out_deps) == sorted_out_deps.end());
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
            tmp_sum.clear();

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
                    auto new_term = diffs[rd_idx] * adj[rd_idx].find(cur_idx)->second;
                    tmp_sum.push_back(std::move(new_term));
                }
            }

            assert(!tmp_sum.empty());

            diffs[cur_idx] = sum(tmp_sum);
        }

        // Create a dict mapping the vars/params in the decomposition
        // to the derivatives of the current output wrt them. This is used
        // to fetch from diffs only the derivatives we are interested in
        // (since there may be vars/params in the decomposition wrt which
        // the derivatives are not requested).
        boost::unordered_flat_map<expression, expression, std::hash<expression>> dmap;
        for (std::vector<expression>::size_type j = 0; j < nvars; ++j) {
            [[maybe_unused]] const auto [_, flag] = dmap.try_emplace(dc[j], diffs[j]);
            assert(flag);
        }

        // Add the derivatives to the local map.
        for (decltype(args.size()) j = 0; j < nargs; ++j) {
            // Compute the indices vector for the current derivative.
            vidx_v2s(tmp_v_idx, prev_begin->first);
            // NOTE: no need to overflow check here, because no derivative
            // order can end up being larger than the total diff order which
            // is representable by std::uint32_t.
            tmp_v_idx.second[static_cast<std::uint32_t>(j)] += 1u;

            if (cur_order == 1u) {
                // Check if the diff argument is present in the
                // decomposition: if it is, we will calculate the derivative and add it.
                // Otherwise, we set the derivative to zero and add it.
                expression cur_der = 0_dbl;

                if (const auto it_dmap = dmap.find(args[j]); it_dmap != dmap.end()) {
                    cur_der = it_dmap->second;
                }

                local_dvec().emplace_back(tmp_v_idx, std::move(cur_der));
            } else {
                // LCOV_EXCL_START
                // Check if we already computed this derivative.
                if (const auto it = local_dmap().find(tmp_v_idx); it == local_dmap().end()) {
                    // The derivative is new. If the diff argument is present in the
                    // decomposition, then we will calculate the derivative and add it.
                    // Otherwise, we set the derivative to zero and add it.
                    expression cur_der = 0_dbl;

                    if (const auto it_dmap = dmap.find(args[j]); it_dmap != dmap.end()) {
                        cur_der = it_dmap->second;
                    }

                    [[maybe_unused]] const auto [_, flag] = local_dmap().try_emplace(tmp_v_idx, std::move(cur_der));
                    assert(flag);
                }
                // LCOV_EXCL_STOP
            }
        }

        // Update prev_begin as we move to the next output.
        ++prev_begin;
        assert(prev_begin != diff_map.end() || i + 1u == cur_nouts);
    }

    // Merge the local map into diff_map.
    if (cur_order == 1u) {
        for (auto &p : local_dvec()) {
            diff_map.emplace_back(vidx_s2v(p.first), std::move(p.second));
        }
    } else {
        // LCOV_EXCL_START
        for (auto &p : local_dmap()) {
            diff_map.emplace_back(vidx_s2v(p.first), std::move(p.second));
        }
        // LCOV_EXCL_STOP
    }
}

} // namespace

// Utility function to check that a dtens_sv_idx_t is well-formed.
bool sv_sanity_check(const dtens_sv_idx_t &v)
{
    if (!v.second.empty()) {
        // Check nonzero order for the first element of v.second.
        if (v.second[0].second == 0u) [[unlikely]] {
            return false;
        }

        // Check the remaining elements.
        for (auto it = v.second.begin() + 1; it != v.second.end(); ++it) {
            // Check that the indices as sorted in strictly ascending order.
            if (!(it->first > (it - 1)->first)) [[unlikely]] {
                return false;
            }

            // Check nonzero order.
            if (it->second == 0u) [[unlikely]] {
                return false;
            }
        }
    }

    return true;
}

} // namespace detail

namespace detail
{

namespace
{

auto diff_tensors_impl(const std::vector<expression> &v_ex, const std::vector<expression> &args, std::uint32_t order)
{
    spdlog::stopwatch sw;

    assert(std::ranges::all_of(args, [](const auto &arg) {
        return std::holds_alternative<variable>(arg.value()) || std::holds_alternative<param>(arg.value());
    }));
    assert(std::unordered_set(args.begin(), args.end()).size() == args.size());

    // Cache the original number of outputs and the diff arguments.
    const auto orig_nouts = v_ex.size();
    const auto nargs = args.size();

    assert(orig_nouts > 0u);
    assert(nargs > 0u);

    // NOTE: check that nargs fits in a 32-bit int, so that
    // in the dtens API get_nargs() can safely return std::uint32_t.
    (void)(boost::numeric_cast<std::uint32_t>(nargs));

    // Map to associate a dtens_sv_idx_t to a derivative.
    // This will be kept manually sorted according to dtens_v_idx_cmp
    // and it will be turned into a flat map at the end.
    dtens_map_t::sequence_type diff_map;

    // Helper to locate a dtens_sv_idx_t in diff_map. If not present,
    // diff_map.end() will be returned.
    auto search_diff_map = [&diff_map](const dtens_sv_idx_t &v) {
        // NOLINTNEXTLINE(modernize-use-ranges)
        auto it = std::lower_bound(diff_map.begin(), diff_map.end(), v, [](const auto &item, const auto &vec) {
            return dtens_sv_idx_cmp{}(item.first, vec);
        });

        if (it != diff_map.end() && it->first == v) {
            return it;
        } else {
            return diff_map.end();
        }
    };

    // This is used as a temporary variable in several places below.
    dtens_sv_idx_t tmp_v_idx;

    // Vector that will store the previous-order derivatives in the loop below.
    // It will be used to construct the decomposition.
    std::vector<expression> prev_diffs;

    // Init diff_map with the order 0 derivatives
    // (i.e., the original function components).
    for (decltype(v_ex.size()) i = 0; i < orig_nouts; ++i) {
        tmp_v_idx.first = boost::numeric_cast<std::uint32_t>(i);

        assert(search_diff_map(tmp_v_idx) == diff_map.end());
        diff_map.emplace_back(tmp_v_idx, v_ex[i]);
    }

    // Iterate over the derivative orders.
    for (std::uint32_t cur_order = 0; cur_order < order; ++cur_order) {
        // Locate the iterator in diff_map corresponding to the beginning
        // of the previous-order derivatives.
        tmp_v_idx.first = 0;
        tmp_v_idx.second.clear();
        if (cur_order != 0u) {
            tmp_v_idx.second.emplace_back(0, cur_order);
        }

        const auto prev_begin = search_diff_map(tmp_v_idx);
        assert(prev_begin != diff_map.end());

        // Store the previous-order derivatives into a separate
        // vector so that we can construct the decomposition.
        prev_diffs.clear();
        std::transform(prev_begin, diff_map.end(), std::back_inserter(prev_diffs),
                       [](const auto &p) { return p.second; });

        // For the purposes of the diff decomposition, the number of outputs
        // is the number of previous derivatives.
        const auto cur_nouts = prev_diffs.size();

        // Run the decomposition on the derivatives of the previous order.
        const auto [dc, nvars] = diff_decompose(prev_diffs);

        // Create the adjoints, the direct/reverse dependencies and the substitution map.
        const auto [adj, dep, revdep, subs_map] = diff_make_adj_dep(dc, nvars, cur_nouts);

        // Store the current diff_map size in order to (later) determine
        // where the set of derivatives for the current order begins.
        const auto orig_diff_map_size = diff_map.size();

        spdlog::stopwatch sw_inner;

        // NOTE: in order to choose between forward and reverse mode, we adopt the standard approach
        // of comparing the number of inputs and outputs. A more accurate (yet more expensive) approach
        // would be to do the computation in both modes (e.g., in parallel) and pick the mode which
        // results in the shortest decomposition. Perhaps we can consider this for a future extension.
        if (cur_nouts >= nargs) {
            diff_tensors_forward_impl(diff_map, cur_nouts, dc, dep, revdep, adj, nvars, args, prev_begin,
                                      cur_order + 1u);
        } else {
            diff_tensors_reverse_impl(diff_map, cur_nouts, dc, dep, revdep, adj, nvars, args, prev_begin,
                                      cur_order + 1u);
        }

        // Determine the range in diff_map for the current-order derivatives.
        auto *cur_begin = diff_map.data() + orig_diff_map_size;
        auto *cur_end = diff_map.data() + diff_map.size();

        // Sort the derivatives for the current order.
        tbb_isolated_parallel_sort(
            cur_begin, cur_end, [](const auto &p1, const auto &p2) { return dtens_sv_idx_cmp{}(p1.first, p2.first); });

        // NOTE: the derivatives we just added to diff_map are still expressed in terms of u variables.
        // We need to apply the substitution map subs_map in order to recover the expressions in terms
        // of the original variables. It is important that we do this now (rather than when constructing
        // the derivatives in diff_tensors_*_impl()) because now we can do the substitution in a vectorised
        // fashion, which greatly reduces the internal redundancy of the resulting expressions.

        // Create the vector of expressions for the substitution.
        std::vector<expression> subs_ret;
        for (auto *it = cur_begin; it != cur_end; ++it) {
            subs_ret.push_back(it->second);
        }

        // Do the substitution.
        subs_ret = subs(subs_ret, subs_map);

        // NOTE: the substitution we just performed results in deep copies of shared arguments. The reason is as
        // follows.
        //
        // subs_map maps "u_i" strings to expressions containing the original variables/parameters. subs_ret, instead,
        // is expressed in terms of the u_i variables, representing the result of the differentiation process. The
        // purpose of the substitution is to express the result of the differentiation process in terms of the original
        // variables/parameters (rather than in terms of u_i variables).
        //
        // If the expressions that were differentiated originally contained shared arguments, these sets of shared
        // arguments are transformed from the original variables/parameters into u_i variables for the purpose of
        // differentiation. Thus, for instance, if originally we had a dfun of (p0, p1, p2), subs_ret will contain
        // a dfun of (u_0, u_1, u_2) instead.
        //
        // The issue is then that, during the substitution, the (u_0, u_1, u_2) set of shared arguments in subs_ret is
        // transformed back into (p0, p1, p2), but we have no way of knowing that this is the exact same set of
        // arguments that shows up in subs_map. Hence, the unnecessary duplication.
        //
        // At this time it does not seem like this duplication is detrimental to performance, at least in tests
        // involving first-order derivatives of neural networks (which is the primary use case for shared arguments). If
        // this becomes a problem, we can consider doing a transformation pass on subs_ret that consolidates identical
        // deep copies of shared arguments in a single shared instance. This would be a bit of a kludge and it should
        // probably be limited to shared arguments containing only non-function expressions (due to the potentially very
        // expensive cost of comparing large expressions trees). This needs to be investigated more.

        // Replace the original expressions in diff_map.
        decltype(subs_ret.size()) i = 0;
        for (auto *it = cur_begin; i < subs_ret.size(); ++i, ++it) {
            it->second = subs_ret[i];
        }

        get_logger()->trace("dtens diff runtime for order {}: {}", cur_order + 1u, sw_inner);
    }

    get_logger()->trace("dtens creation runtime: {}", sw);

    // Assemble and return the result.
    dtens_map_t retval;
    retval.adopt_sequence(boost::container::ordered_unique_range_t{}, std::move(diff_map));

    // Check sorting.
    assert(std::ranges::is_sorted(
        retval, [](const auto &p1, const auto &p2) { return dtens_sv_idx_cmp{}(p1.first, p2.first); }));
    // Check the variable indices.
    assert(std::ranges::all_of(
        retval, [&nargs](const auto &p) { return p.first.second.empty() || p.first.second.back().first < nargs; }));
    // No duplicates in the indices vectors.
    assert(std::ranges::adjacent_find(retval, [](const auto &p1, const auto &p2) { return p1.first == p2.first; })
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
    if (std::ranges::any_of(args, [](const auto &arg) {
            return !std::holds_alternative<variable>(arg.value()) && !std::holds_alternative<param>(arg.value());
        })) {
        throw std::invalid_argument("Derivatives can be computed only with respect to variables and/or parameters");
    }

    // Check if there are repeated entries in args.
    const boost::unordered_flat_set<expression, std::hash<expression>> args_set(args.begin(), args.end());
    if (args_set.size() != args.size()) {
        throw std::invalid_argument(
            fmt::format("Duplicate entries detected in the list of variables/parameters with respect to which the "
                        "derivatives are to be computed: {}",
                        args));
    }

    return dtens{dtens::impl{.m_map = diff_tensors_impl(v_ex, args, order), .m_args = std::move(args)}};
}

} // namespace detail

HEYOKA_END_NAMESPACE
