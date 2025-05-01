// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this header contains utilities used when traversing expressions.

#ifndef HEYOKA_DETAIL_EX_TRAVERSAL_HPP
#define HEYOKA_DETAIL_EX_TRAVERSAL_HPP

#include <cstddef>
#include <functional>
#include <optional>
#include <utility>

#include <boost/container/small_vector.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/func_args.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: these are set/map structures used to cache the result of a computation
// during expression traversal.
using void_ptr_set = boost::unordered_flat_set<const void *>;

template <typename T>
using void_ptr_map = boost::unordered_flat_map<const void *, T>;

// NOTE: here we define a couple of stack data structures to be used when traversing
// the nodes of an expression. We use boost::small_vector in order to avoid paying for
// heap allocations on small expressions.
constexpr std::size_t static_ex_traversal_stack_size = 20;

using traverse_stack
    = boost::container::small_vector<std::pair<const expression *, bool>, static_ex_traversal_stack_size>;

template <typename T>
using return_stack = boost::container::small_vector<std::optional<T>, static_ex_traversal_stack_size>;

expression ex_traverse_transform_leaves(void_ptr_map<const expression> &,
                                        void_ptr_map<const func_args::shared_args_t> &, traverse_stack &,
                                        return_stack<expression> &, const expression &,
                                        const std::function<expression(const expression &)> &);

void ex_traverse_visit_leaves(void_ptr_set &, void_ptr_set &, traverse_stack &, const expression &,
                              const std::function<void(const expression &)> &);

bool ex_traverse_test_any(void_ptr_set &, void_ptr_set &, traverse_stack &, const expression &,
                          const std::function<bool(const expression &)> &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
