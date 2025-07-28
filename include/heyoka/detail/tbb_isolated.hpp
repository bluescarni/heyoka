// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TBB_ISOLATED_HPP
#define HEYOKA_DETAIL_TBB_ISOLATED_HPP

// NOTE: this header contains "isolated" versions of several TBB algorithms.
//
// By default in TBB when nested parallel tasks are created an uncaught exception in a task may lead to the outright
// cancellation of another task. This is explained clearly at this page:
//
// https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2022-2/cancellation-and-nested-parallelism.html
//
// Although this leads to efficient and prompt cancellation of tasks logically belonging to the same tree of tasks, it
// also creates a situation in which entire sections of code are "skipped" if an exception is raised in a far away task,
// breaking locality of reasoning about code. E.g., a specific parallel_for() invocation may be skipped because of an
// uncaught exception in another task.
//
// In order to avoid this, the suggestion is to invoke TBB primitives passing an explicit "isolated" context. This
// prevents downwards propagation of cancellation into an algorithm.

#include <utility>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/task_group.h>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename... Args>
auto tbb_isolated_parallel_for(Args &&...args)
    -> decltype(oneapi::tbb::parallel_for(std::forward<Args>(args)...,
                                          std::declval<oneapi::tbb::task_group_context &>()))
{
    oneapi::tbb::task_group_context isolated_ctx(oneapi::tbb::task_group_context::isolated);
    return oneapi::tbb::parallel_for(std::forward<Args>(args)..., isolated_ctx);
}

template <typename... Args>
auto tbb_isolated_parallel_invoke(Args &&...args)
    -> decltype(oneapi::tbb::parallel_invoke(std::forward<Args>(args)...,
                                             std::declval<oneapi::tbb::task_group_context &>()))
{
    oneapi::tbb::task_group_context isolated_ctx(oneapi::tbb::task_group_context::isolated);
    return oneapi::tbb::parallel_invoke(std::forward<Args>(args)..., isolated_ctx);
}

// NOTE: some TBB algorithms do not accept a context in input. In these cases, we wrap the call into an isolated task
// group.
template <typename... Args>
auto tbb_isolated_parallel_sort(Args &&...args) -> decltype(oneapi::tbb::parallel_sort(std::forward<Args>(args)...))
{
    oneapi::tbb::task_group_context isolated_ctx(oneapi::tbb::task_group_context::isolated);
    oneapi::tbb::task_group tg(isolated_ctx);
    tg.run_and_wait([&args...]() { oneapi::tbb::parallel_sort(std::forward<Args>(args)...); });
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
