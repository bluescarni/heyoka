// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <functional>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/func_args.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// This function will return a copy of the input expression e in which the leaf nodes have been transformed
// by the leaf_tfunc function and the branch nodes by the branch_tfunc function.
//
// The transformations are applied in a depth-first post-order fashion. That is, the children of a branch node are
// transformed first, and branch_tfunc is applied afterwards to the new branch node with the transformed
// children. If an input tfunc is empty, no transformation is applied.
//
// func_map and sargs_map are caches used during the traversal in order to avoid repeating redundant computations.
expression ex_traverse_transform_nodes(void_ptr_map<const expression> &func_map,
                                       sargs_ptr_map<const func_args::shared_args_t> &sargs_map, const expression &e,
                                       const std::function<expression(const expression &)> &leaf_tfunc,
                                       const std::function<expression(const expression &)> &branch_tfunc)
{
    traverse_stack stack;
    return_stack<expression> out_stack;

    // Check if we have the transformation functions.
    const auto wih_leaf_tfunc = static_cast<bool>(leaf_tfunc);
    const auto wih_branch_tfunc = static_cast<bool>(branch_tfunc);

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
                // We already created a transformed copy of the current function. Fetch it from the cache
                // and add it to out_stack.
                out_stack.emplace_back(it->second);
                continue;
            }

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now visited and created transformed copies of all the children of the function node
                // (i.e., the function arguments). The copies are at the tail end of out_stack. We will be
                // popping them from out_stack and use them to initialise a new copy of the function.

                // Build the new arguments.
                std::vector<expression> new_args;
                const auto n_args = f.args().size();
                new_args.reserve(n_args);
                for (decltype(new_args.size()) i = 0; i < n_args; ++i) {
                    // NOTE: out_stack must not be empty and its last element also cannot be empty.
                    assert(!out_stack.empty());
                    assert(out_stack.back());

                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    new_args.push_back(std::move(*out_stack.back()));
                    out_stack.pop_back();
                }

                // Create the new copy of the function.
                auto ex_copy = [&]() {
                    if (shared_args) {
                        // NOTE: if the function manages its arguments via a shared reference, we must make
                        // sure to record the new arguments in sargs_map, so that when we run again into the
                        // same shared reference we re-use the cached result.
                        auto new_sargs = std::make_shared<const std::vector<expression>>(std::move(new_args));

                        assert(!sargs_map.contains(&*shared_args));
                        sargs_map.emplace(&*shared_args, new_sargs);

                        return expression{f.make_copy_with_new_args(std::move(new_sargs))};
                    } else {
                        return expression{f.make_copy_with_new_args(std::move(new_args))};
                    }
                }();

                // Transform the new copy, if necessary
                if (wih_branch_tfunc) {
                    ex_copy = branch_tfunc(ex_copy);
                }

                // Add it to the cache.
                func_map.emplace(f_id, ex_copy);

                // Add it to out_stack.
                // NOTE: out_stack must not be empty and its last element must be empty (it is supposed to be
                // the empty function we pushed the first time we visited).
                assert(!out_stack.empty());
                assert(!out_stack.back());
                out_stack.back().emplace(std::move(ex_copy));
            } else {
                // It is the first time we visit this function.
                if (shared_args) {
                    // The function manages its arguments via a shared reference. Check
                    // if we already transformed the arguments before.
                    if (const auto it = sargs_map.find(&*shared_args); it != sargs_map.end()) {
                        // The arguments have been transformed before. Fetch them from the cache and
                        // use them to construct a new copy of the function.
                        auto ex_copy = expression{f.make_copy_with_new_args(it->second)};

                        // Transform the new copy, if necessary
                        if (wih_branch_tfunc) {
                            ex_copy = branch_tfunc(ex_copy);
                        }

                        // Add the new function to the cache and to out_stack.
                        func_map.emplace(f_id, ex_copy);
                        out_stack.emplace_back(std::move(ex_copy));

                        continue;
                    }

                    // NOTE: if we arrive here, it means that the shared arguments of the function have never
                    // been transformed before. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its
                // arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }

                // Add an empty expression to out_stack. We will create the transformed copy
                // once we have transformed all arguments.
                out_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            // Apply the transformation leaf_tfunc to the leaf node, if necessary, and add the
            // result to out_stack.
            out_stack.emplace_back(wih_leaf_tfunc ? leaf_tfunc(*cur_ex) : *cur_ex);
        }
    }

    assert(out_stack.size() == 1u);
    assert(out_stack.back());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return std::move(*out_stack.back());
}

// This function will visit the leaves of the expression e and invoke vfunc on each leaf.
//
// func_map and sargs_map are caches used during the traversal in order to avoid repeating redundant computations.
void ex_traverse_visit_leaves(void_ptr_set &func_set, sargs_ptr_set &sargs_set, const expression &e,
                              const std::function<void(const expression &)> &vfunc)
{
    assert(vfunc);

    traverse_stack stack;

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the stack.
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
                assert(!func_set.contains(f_id));
            } else if (func_set.contains(f_id)) {
                // We already visited the current function, no need to do anything else.
                continue;
            }

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now visited all the children of the function node. We have to add f_id
                // to the cache so that we won't re-visit.
                func_set.emplace(f_id);

                if (shared_args) {
                    // NOTE: if the function manages its arguments via a shared reference,
                    // we must make sure to record in sargs_set that we have visited shared_args,
                    // so that when we run again into the same shared reference we avoid unnecessary
                    // re-visiting.
                    assert(!sargs_set.contains(&*shared_args));
                    sargs_set.emplace(&*shared_args);
                }
            } else {
                // It is the first time we visit this function.
                if (shared_args) {
                    // The function manages its arguments via a shared reference. Check
                    // if we already visited the arguments.
                    if (sargs_set.contains(&*shared_args)) {
                        // We already visited the arguments. Add f_id to the cache and move on.
                        func_set.emplace(f_id);
                        continue;
                    }

                    // NOTE: if we arrive here, it means that we haven't visited the shared arguments of the
                    // function yet. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its arguments
                // to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            // Visit the leaf node.
            vfunc(*cur_ex);
        }
    }
}

// This function will visit the all the nodes of the expression e and it will return true if at least
// one node satisfies the predicate pred.
//
// func_map and sargs_map are caches used during the traversal in order to avoid repeating redundant computations.
bool ex_traverse_test_any(void_ptr_set &func_set, sargs_ptr_set &sargs_set, const expression &e,
                          const std::function<bool(const expression &)> &pred)
{
    assert(pred);

    traverse_stack stack;

    // Init the return value.
    auto retval = false;

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        assert(!visited || std::holds_alternative<func>(cur_ex->value()));

        // If the current expression is a function, check if we already tested
        // that it does *not* satisfy the predicate.
        const auto *f_ptr = std::get_if<func>(&cur_ex->value());
        if (f_ptr != nullptr) {
            if (visited) {
                // NOTE: if this is the second visit, we know that the the function cannot possibly be in the cache,
                // and thus we can avoid an unnecessary lookup.
                assert(!func_set.contains(f_ptr->get_ptr()));
            } else if (func_set.contains(f_ptr->get_ptr())) {
                continue;
            }
        }

        // Check the predicate.
        // NOTE: no need to check the predicate if this is the second visit.
        if (!visited && pred(*cur_ex)) {
            // The expression satisfies the predicate. Set retval
            // to true and break out.
            retval = true;
            break;
        }

        // The current expression does *not* satisfy the predicate. If it is a function,
        // we need to test its arguments, otherwise we don't have do do anything else.
        if (f_ptr != nullptr) {
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now determined that neither the function nor all its children satisfy
                // the predicate. We have to add f_id to the cache so that we won't re-test it.
                func_set.emplace(f_id);

                if (shared_args) {
                    // NOTE: if the function manages its arguments via a shared reference,
                    // we must make sure to record in sargs_set that we have tested shared_args,
                    // so that when we run again into the same shared reference we avoid redundant
                    // computations.
                    assert(!sargs_set.contains(&*shared_args));
                    sargs_set.emplace(&*shared_args);
                }
            } else {
                // It is the first time we test this function.
                if (shared_args) {
                    // The function does not satisfy the predicate and it manages its arguments via a
                    // shared reference. Check if we already tested that all arguments do *not* satisfy
                    // the predicate.
                    if (sargs_set.contains(&*shared_args)) {
                        // We already determined that all arguments do *not* satisfy the predicate. Add f_id
                        // to the cache and move on.
                        func_set.emplace(f_id);
                        continue;
                    }

                    // NOTE: if we arrive here, it means that we haven't tested the shared arguments of the
                    // function yet. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its arguments
                // to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }
            }
        }
    }

    return retval;
}

} // namespace detail

HEYOKA_END_NAMESPACE
