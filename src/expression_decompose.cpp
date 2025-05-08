// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/func_args.hpp>

// NOTE: this file contains the implementation of expression decomposition, both for Taylor integrators and for
// functions. The two implementations are similar enough that they can be implemented on top of a common generic
// function.

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Implementation of expression decomposition.
//
// This function will decompose the input expression e into the decomposition dc. The return value is an empty optional
// if e is a non-function expression, otherwise it represents the index into dc where the function expression e was
// decomposed. f_dc is the function object to be used to decompose a function expression after all of its arguments have
// been decomposed. func_map and sargs_map are caches used to avoid redundant computations on shared subexpressions and
// shared argument sets.
template <typename T>
std::optional<typename std::vector<T>::size_type>
expression_decompose_impl(void_ptr_map<const typename std::vector<T>::size_type> &func_map,
                          sargs_ptr_map<const func_args::shared_args_t> &sargs_map, const expression &e,
                          std::vector<T> &dc, const auto &f_dc)
{
    traverse_stack stack;
    return_stack<std::optional<typename std::vector<T>::size_type>> out_stack;

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
                // We already decomposed the current function. Fetch the result from the cache
                // and add it to out_stack.
                out_stack.emplace_back(it->second);
                continue;
            }

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now visited and decomposed all the children of the function node (i.e., the function
                // arguments). The results of the decomposition are at the tail end of out_stack. We will be popping
                // them from out_stack and use them to initialise the decomposed version of the function.

                // Build the arguments for the decomposed function.
                std::vector<expression> new_args;
                const auto n_args = f.args().size();
                new_args.reserve(n_args);
                for (decltype(new_args.size()) i = 0; i < n_args; ++i) {
                    // Fetch the original function argument.
                    const auto &orig_arg = f.args()[i];

                    // NOTE: out_stack must not be empty and its last element also cannot be empty.
                    assert(!out_stack.empty());
                    assert(out_stack.back());

                    // Fetch the current decomposition index.
                    const auto opt_idx = *out_stack.back();
                    if (opt_idx) {
                        // The current argument is a decomposed function. It will be replaced
                        // by the corresponding u variable.
                        assert(std::holds_alternative<func>(orig_arg.value()));
                        new_args.emplace_back(fmt::format("u_{}", *opt_idx));
                    } else {
                        // The current argument is a non-function, just copy it.
                        assert(!std::holds_alternative<func>(orig_arg.value()));
                        new_args.push_back(orig_arg);
                    }

                    out_stack.pop_back();
                }

                // Create the decomposed copy of the function.
                auto f_copy = [&]() {
                    if (shared_args) {
                        // NOTE: if the function manages its arguments via a shared reference, we must make
                        // sure to record the new arguments in sargs_map, so that when we run again into the
                        // same shared reference we re-use the cached result.
                        auto new_sargs = std::make_shared<const std::vector<expression>>(std::move(new_args));

                        assert(!sargs_map.contains(&*shared_args));
                        sargs_map.emplace(&*shared_args, new_sargs);

                        return f.make_copy_with_new_args(std::move(new_sargs));
                    } else {
                        return f.make_copy_with_new_args(std::move(new_args));
                    }
                }();

                // Decompose f_copy.
                const auto ret = f_dc(std::move(f_copy), dc);

                // Add the result to the cache.
                func_map.emplace(f_id, ret);

                // Add ret to out_stack.
                // NOTE: out_stack must not be empty and its last element must be empty (it is supposed to be
                // the empty index we pushed the first time we visited).
                assert(!out_stack.empty());
                assert(!out_stack.back());
                out_stack.back().emplace(ret);
            } else {
                // It is the first time we visit this function.
                if (shared_args) {
                    // The function manages its arguments via a shared reference. Check
                    // if we already decomposed the arguments before.
                    if (const auto it = sargs_map.find(&*shared_args); it != sargs_map.end()) {
                        // The arguments have been decomposed before. Fetch them from the cache and
                        // use them to construct a decomposed copy of the function.
                        auto f_copy = f.make_copy_with_new_args(it->second);

                        // Decompose f_copy.
                        const auto ret = f_dc(std::move(f_copy), dc);

                        // Add the decomposed function to the cache.
                        func_map.emplace(f_id, ret);

                        // Add ret to out_stack.
                        out_stack.emplace_back(ret);

                        continue;
                    }

                    // NOTE: if we arrive here, it means that the shared arguments of the function have never
                    // been decomposed before. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its
                // arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }

                // Add an empty return value to out_stack. We will create the real return value once
                // we have decomposed all arguments.
                out_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            // Leaf nodes are not decomposed.
            out_stack.emplace_back(std::optional<typename std::vector<T>::size_type>{});
        }
    }

    assert(out_stack.size() == 1u);
    assert(out_stack.back());

#if !defined(NDEBUG)

    // Check that the final result is consistent with the type of e.
    if (std::holds_alternative<func>(e.value())) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        assert(*out_stack.back());
    } else {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        assert(!*out_stack.back());
    }

#endif

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return *out_stack.back();
}

} // namespace

std::optional<std::vector<expression>::size_type>
decompose(void_ptr_map<const std::vector<expression>::size_type> &func_map,
          sargs_ptr_map<const func_args::shared_args_t> &sargs_map, const expression &e, std::vector<expression> &dc)
{
    const auto f_dc = [](func &&f, std::vector<expression> &dc) {
        assert(std::ranges::none_of(f.args(), [](const auto &ex) { return std::holds_alternative<func>(ex.value()); }));

        // Record the index at which f will be appended to dc.
        const auto ret = dc.size();

        // Add f to dc.
        dc.emplace_back(std::move(f));

        return ret;
    };

    return expression_decompose_impl(func_map, sargs_map, e, dc, f_dc);
}

std::optional<taylor_dc_t::size_type> taylor_decompose(void_ptr_map<const taylor_dc_t::size_type> &func_map,
                                                       sargs_ptr_map<const func_args::shared_args_t> &sargs_map,
                                                       const expression &e, taylor_dc_t &dc)
{
    return expression_decompose_impl(func_map, sargs_map, e, dc, &func_taylor_decompose_impl);
}

} // namespace detail

HEYOKA_END_NAMESPACE
