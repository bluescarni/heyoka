// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/visibility.hpp>
#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

// NOTE: these actions will be performed concurrently from
// multiple threads of execution:
// - invocation of the generator's call operator,
// - copy construction of the events' and step callbacks and
//   invocation of the call operator on the copies.

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_until_impl(const taylor_adaptive<T> &ta, T t, std::size_t n_iter,
                              const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                              std::size_t max_steps, T max_delta_t, step_callback<T> &cb, bool write_tc,
                              bool with_c_out)
{
    // NOTE: store the results into a vector of optionals, so that we avoid
    // having to init a large number of default-constructed integrators
    // that are anyway going to be destroyed.
    std::vector<std::optional<
        std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>>
        opt_retval;
    opt_retval.resize(boost::numeric_cast<decltype(opt_retval.size())>(n_iter));

    // The actual return value, into which we will eventually move the results of the
    // integrations.
    std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
        retval;
    retval.reserve(boost::numeric_cast<decltype(retval.size())>(n_iter));

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n_iter), [&](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Generate the integrator for the current iteration.
            auto local_ta = gen(ta, i);

            // Make a local copy of the callback.
            auto local_cb = cb;

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                           kw::callback = local_cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::tuple_cat(std::make_tuple(std::move(local_ta)), std::move(loc_ret)));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        assert(opt);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        retval.push_back(std::move(*opt));
    }

    return retval;
}

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_for_impl(const taylor_adaptive<T> &ta, T delta_t, std::size_t n_iter,
                            const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                            std::size_t max_steps, T max_delta_t, step_callback<T> &cb, bool write_tc, bool with_c_out)
{
    // NOTE: store the results into a vector of optionals, so that we avoid
    // having to init a large number of default-constructed integrators
    // that are anyway going to be destroyed.
    std::vector<std::optional<
        std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>>
        opt_retval;
    opt_retval.resize(boost::numeric_cast<decltype(opt_retval.size())>(n_iter));

    // The actual return value, into which we will eventually move the results of the
    // integrations.
    std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
        retval;
    retval.reserve(boost::numeric_cast<decltype(retval.size())>(n_iter));

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n_iter), [&](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Generate the integrator for the current iteration.
            auto local_ta = gen(ta, i);

            // Make a local copy of the callback.
            auto local_cb = cb;

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                         kw::callback = local_cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::tuple_cat(std::make_tuple(std::move(local_ta)), std::move(loc_ret)));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        assert(opt); // LCOV_EXCL_LINE
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        retval.push_back(std::move(*opt));
    }

    return retval;
}

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::vector<T>>>
ensemble_propagate_grid_impl(const taylor_adaptive<T> &ta, std::vector<T> grid, std::size_t n_iter,
                             const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                             std::size_t max_steps, T max_delta_t, step_callback<T> &cb)
{
    // NOTE: store the results into a vector of optionals, so that we avoid
    // having to init a large number of default-constructed integrators
    // that are anyway going to be destroyed.
    std::vector<std::optional<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::vector<T>>>>
        opt_retval;
    opt_retval.resize(boost::numeric_cast<decltype(opt_retval.size())>(n_iter));

    // The actual return value, into which we will eventually move the results of the
    // integrations.
    std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::vector<T>>> retval;
    retval.reserve(boost::numeric_cast<decltype(retval.size())>(n_iter));

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n_iter), [&](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Generate the integrator for the current iteration.
            auto local_ta = gen(ta, i);

            // Make a local copy of the callback.
            auto local_cb = cb;

            // Do the propagation.
            auto loc_ret = local_ta.propagate_grid(grid, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                                   kw::callback = local_cb);

            // Assign the results.
            opt_retval[i].emplace(std::tuple_cat(std::make_tuple(std::move(local_ta)), std::move(loc_ret)));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        assert(opt);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        retval.push_back(std::move(*opt));
    }

    return retval;
}

// Explicit instantiations.
// NOLINTBEGIN
#define HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST(T)                                                                       \
    template HEYOKA_DLL_PUBLIC std::vector<                                                                            \
        std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>        \
    ensemble_propagate_until_impl(const taylor_adaptive<T> &, T, std::size_t,                                          \
                                  const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &,          \
                                  std::size_t, T, step_callback<T> &, bool, bool);                                     \
                                                                                                                       \
    template HEYOKA_DLL_PUBLIC std::vector<                                                                            \
        std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>        \
    ensemble_propagate_for_impl(const taylor_adaptive<T> &, T, std::size_t,                                            \
                                const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &,            \
                                std::size_t, T, step_callback<T> &, bool, bool);                                       \
                                                                                                                       \
    template HEYOKA_DLL_PUBLIC                                                                                         \
        std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::vector<T>>>                 \
        ensemble_propagate_grid_impl(const taylor_adaptive<T> &, std::vector<T>, std::size_t,                          \
                                     const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &,       \
                                     std::size_t, T, step_callback<T> &);
// NOLINTEND

HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST(float)
HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST(double)
HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST(mppp::real)

#endif

#undef HEYOKA_ENSEMBLE_PROPAGATE_SCALAR_INST

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_until_batch_impl(
    const taylor_adaptive_batch<T> &ta, T t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, std::size_t max_steps,
    const std::vector<T> &max_delta_ts, step_callback_batch<T> &cb, bool write_tc, bool with_c_out)
{
    // NOTE: store the results into a vector of optionals, so that we avoid
    // having to init a large number of default-constructed integrators
    // that are anyway going to be destroyed.
    std::vector<std::optional<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>>
        opt_retval;
    opt_retval.resize(boost::numeric_cast<decltype(opt_retval.size())>(n_iter));

    // The actual return value, into which we will eventually move the results of the
    // integrations.
    std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>> retval;
    retval.reserve(boost::numeric_cast<decltype(retval.size())>(n_iter));

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n_iter), [&](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Generate the integrator for the current iteration.
            auto local_ta = gen(ta, i);

            // Make a local copy of the callback.
            auto local_cb = cb;

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_ts,
                                           kw::callback = local_cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::move(loc_ret));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        assert(opt);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        retval.push_back(std::move(*opt));
    }

    return retval;
}

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_for_batch_impl(
    const taylor_adaptive_batch<T> &ta, T delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, std::size_t max_steps,
    const std::vector<T> &max_delta_ts, step_callback_batch<T> &cb, bool write_tc, bool with_c_out)
{
    // NOTE: store the results into a vector of optionals, so that we avoid
    // having to init a large number of default-constructed integrators
    // that are anyway going to be destroyed.
    std::vector<std::optional<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>>
        opt_retval;
    opt_retval.resize(boost::numeric_cast<decltype(opt_retval.size())>(n_iter));

    // The actual return value, into which we will eventually move the results of the
    // integrations.
    std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>> retval;
    retval.reserve(boost::numeric_cast<decltype(retval.size())>(n_iter));

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n_iter), [&](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Generate the integrator for the current iteration.
            auto local_ta = gen(ta, i);

            // Make a local copy of the callback.
            auto local_cb = cb;

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_ts,
                                         kw::callback = local_cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::move(loc_ret));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        assert(opt);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        retval.push_back(std::move(*opt));
    }

    return retval;
}

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::vector<T>>> ensemble_propagate_grid_batch_impl(
    const taylor_adaptive_batch<T> &ta, const std::vector<T> &grid_, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, std::size_t max_steps,
    const std::vector<T> &max_delta_ts, step_callback_batch<T> &cb)
{
    const auto batch_size = ta.get_batch_size();
    assert(batch_size != 0u); // LCOV_EXCL_LINE

    // Splat out the time grid.
    std::vector<T> grid;
    grid.reserve(boost::safe_numerics::safe<decltype(grid.size())>(grid_.size()) * batch_size);
    for (auto gval : grid_) {
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            grid.push_back(gval);
        }
    }
    assert(grid.size() == grid_.size() * batch_size); // LCOV_EXCL_LINE

    // NOTE: store the results into a vector of optionals, so that we avoid
    // having to init a large number of default-constructed integrators
    // that are anyway going to be destroyed.
    std::vector<std::optional<std::tuple<taylor_adaptive_batch<T>, std::vector<T>>>> opt_retval;
    opt_retval.resize(boost::numeric_cast<decltype(opt_retval.size())>(n_iter));

    // The actual return value, into which we will eventually move the results of the
    // integrations.
    std::vector<std::tuple<taylor_adaptive_batch<T>, std::vector<T>>> retval;
    retval.reserve(boost::numeric_cast<decltype(retval.size())>(n_iter));

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n_iter), [&](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Generate the integrator for the current iteration.
            auto local_ta = gen(ta, i);

            // Make a local copy of the callback.
            auto local_cb = cb;

            // Do the propagation.
            auto loc_ret = local_ta.propagate_grid(grid, kw::max_steps = max_steps, kw::max_delta_t = max_delta_ts,
                                                   kw::callback = local_cb);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::move(loc_ret));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        assert(opt);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        retval.push_back(std::move(*opt));
    }

    return retval;
}

// Explicit instantiations.
// NOLINTBEGIN
#define HEYOKA_ENSEMBLE_PROPAGATE_BATCH_INST(T)                                                                        \
    template HEYOKA_DLL_PUBLIC                                                                                         \
        std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>                   \
        ensemble_propagate_until_batch_impl(                                                                           \
            const taylor_adaptive_batch<T> &, T, std::size_t,                                                          \
            const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,       \
            const std::vector<T> &, step_callback_batch<T> &, bool, bool);                                             \
                                                                                                                       \
    template HEYOKA_DLL_PUBLIC                                                                                         \
        std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>                   \
        ensemble_propagate_for_batch_impl(                                                                             \
            const taylor_adaptive_batch<T> &, T, std::size_t,                                                          \
            const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,       \
            const std::vector<T> &, step_callback_batch<T> &, bool, bool);                                             \
                                                                                                                       \
    template HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive_batch<T>, std::vector<T>>>                       \
    ensemble_propagate_grid_batch_impl(                                                                                \
        const taylor_adaptive_batch<T> &, const std::vector<T> &, std::size_t,                                         \
        const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,           \
        const std::vector<T> &, step_callback_batch<T> &);
// NOLINTEND

HEYOKA_ENSEMBLE_PROPAGATE_BATCH_INST(float)
HEYOKA_ENSEMBLE_PROPAGATE_BATCH_INST(double)
HEYOKA_ENSEMBLE_PROPAGATE_BATCH_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ENSEMBLE_PROPAGATE_BATCH_INST(mppp::real128)

#endif

#undef HEYOKA_ENSEMBLE_PROPAGATE_BATCH_INST

} // namespace detail

HEYOKA_END_NAMESPACE
