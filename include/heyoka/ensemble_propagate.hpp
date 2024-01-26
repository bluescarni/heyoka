// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_ENSEMBLE_PROPAGATE_HPP
#define HEYOKA_ENSEMBLE_PROPAGATE_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/continuous_output.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>,
                       step_callback<T>>>
ensemble_propagate_until_impl(const taylor_adaptive<T> &, T, std::size_t,
                              const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &, std::size_t,
                              T, step_callback<T>, bool, bool);

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>,
                       step_callback<T>>>
ensemble_propagate_for_impl(const taylor_adaptive<T> &, T, std::size_t,
                            const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &, std::size_t, T,
                            step_callback<T>, bool, bool);

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, step_callback<T>, std::vector<T>>>
ensemble_propagate_grid_impl(const taylor_adaptive<T> &, std::vector<T>, std::size_t,
                             const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &, std::size_t, T,
                             step_callback<T>);

// Prevent implicit instantiations.
// NOLINTBEGIN
#define HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_SCALAR_INST(T)                                                                \
    extern template std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t,                      \
                                           std::optional<continuous_output<T>>, step_callback<T>>>                     \
    ensemble_propagate_until_impl(const taylor_adaptive<T> &, T, std::size_t,                                          \
                                  const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &,          \
                                  std::size_t, T, step_callback<T>, bool, bool);                                       \
                                                                                                                       \
    extern template std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t,                      \
                                           std::optional<continuous_output<T>>, step_callback<T>>>                     \
    ensemble_propagate_for_impl(const taylor_adaptive<T> &, T, std::size_t,                                            \
                                const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &,            \
                                std::size_t, T, step_callback<T>, bool, bool);                                         \
                                                                                                                       \
    extern template std::vector<                                                                                       \
        std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, step_callback<T>, std::vector<T>>>           \
    ensemble_propagate_grid_impl(const taylor_adaptive<T> &, std::vector<T>, std::size_t,                              \
                                 const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &,           \
                                 std::size_t, T, step_callback<T>);
// NOLINTEND

HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_SCALAR_INST(float)
HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_SCALAR_INST(double)
HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_SCALAR_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_SCALAR_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_SCALAR_INST(mppp::real)

#endif

} // namespace detail

template <typename T, typename... KwArgs>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>,
                       step_callback<T>>>
ensemble_propagate_until(const taylor_adaptive<T> &ta, T t, std::size_t n_iter,
                         const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                         const KwArgs &...kw_args)
{
    auto [max_steps, max_delta_t, cb, write_tc, with_c_out] = detail::taylor_propagate_common_ops<T, false>(kw_args...);

    return detail::ensemble_propagate_until_impl(ta, std::move(t), n_iter, gen, max_steps, std::move(max_delta_t),
                                                 std::move(cb), write_tc, with_c_out);
}

template <typename T, typename... KwArgs>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>,
                       step_callback<T>>>
ensemble_propagate_for(const taylor_adaptive<T> &ta, T delta_t, std::size_t n_iter,
                       const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                       const KwArgs &...kw_args)
{
    auto [max_steps, max_delta_t, cb, write_tc, with_c_out] = detail::taylor_propagate_common_ops<T, false>(kw_args...);

    return detail::ensemble_propagate_for_impl(ta, std::move(delta_t), n_iter, gen, max_steps, std::move(max_delta_t),
                                               std::move(cb), write_tc, with_c_out);
}

template <typename T, typename... KwArgs>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, step_callback<T>, std::vector<T>>>
ensemble_propagate_grid(const taylor_adaptive<T> &ta, std::vector<T> grid, std::size_t n_iter,
                        const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                        const KwArgs &...kw_args)
{
    auto [max_steps, max_delta_t, cb] = detail::taylor_propagate_common_ops<T, true>(kw_args...);

    return detail::ensemble_propagate_grid_impl(ta, std::move(grid), n_iter, gen, max_steps, std::move(max_delta_t),
                                                std::move(cb));
}

namespace detail
{

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>, step_callback_batch<T>>>
ensemble_propagate_until_batch_impl(
    const taylor_adaptive_batch<T> &, T, std::size_t,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,
    const std::vector<T> &, step_callback_batch<T>, bool, bool);

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>, step_callback_batch<T>>>
ensemble_propagate_for_batch_impl(
    const taylor_adaptive_batch<T> &, T, std::size_t,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,
    const std::vector<T> &, step_callback_batch<T>, bool, bool);

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, step_callback_batch<T>, std::vector<T>>>
ensemble_propagate_grid_batch_impl(
    const taylor_adaptive_batch<T> &, const std::vector<T> &, std::size_t,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,
    const std::vector<T> &, step_callback_batch<T>);

// Prevent implicit instantiations.
// NOLINTBEGIN
#define HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_BATCH_INST(T)                                                                 \
    extern template std::vector<                                                                                       \
        std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>, step_callback_batch<T>>>       \
    ensemble_propagate_until_batch_impl(                                                                               \
        const taylor_adaptive_batch<T> &, T, std::size_t,                                                              \
        const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,           \
        const std::vector<T> &, step_callback_batch<T>, bool, bool);                                                   \
                                                                                                                       \
    extern template std::vector<                                                                                       \
        std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>, step_callback_batch<T>>>       \
    ensemble_propagate_for_batch_impl(                                                                                 \
        const taylor_adaptive_batch<T> &, T, std::size_t,                                                              \
        const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,           \
        const std::vector<T> &, step_callback_batch<T>, bool, bool);                                                   \
                                                                                                                       \
    extern template std::vector<std::tuple<taylor_adaptive_batch<T>, step_callback_batch<T>, std::vector<T>>>          \
    ensemble_propagate_grid_batch_impl(                                                                                \
        const taylor_adaptive_batch<T> &, const std::vector<T> &, std::size_t,                                         \
        const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,           \
        const std::vector<T> &, step_callback_batch<T>);
// NOLINTEND

HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_BATCH_INST(float)
HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_BATCH_INST(double)
HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_BATCH_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_BATCH_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_ENSEMBLE_PROPAGATE_EXTERN_BATCH_INST(mppp::real)

#endif

} // namespace detail

template <typename T, typename... KwArgs>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>, step_callback_batch<T>>>
ensemble_propagate_until_batch(
    const taylor_adaptive_batch<T> &ta, T t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, const KwArgs &...kw_args)
{
    // NOTE: here and in the other 2 ensemble batch functions, taylor_propagate_common_ops_batch()
    // is guaranteed to return max_delta_ts as a new object (that is, it cannot alias anything else).
    // Hence, we can pass it by reference to the implementation functions.
    auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops_batch<T, false, true>(ta.get_batch_size(), kw_args...);

    return detail::ensemble_propagate_until_batch_impl(ta, t, n_iter, gen, max_steps, max_delta_ts, std::move(cb),
                                                       write_tc, with_c_out);
}

template <typename T, typename... KwArgs>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>, step_callback_batch<T>>>
ensemble_propagate_for_batch(const taylor_adaptive_batch<T> &ta, T delta_t, std::size_t n_iter,
                             const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen,
                             const KwArgs &...kw_args)
{
    auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops_batch<T, false, true>(ta.get_batch_size(), kw_args...);

    return detail::ensemble_propagate_for_batch_impl(ta, delta_t, n_iter, gen, max_steps, max_delta_ts, std::move(cb),
                                                     write_tc, with_c_out);
}

// NOTE: taking grid by reference here is ok, as the implementation needs
// to splat it anyway and thus it always creates a copy.
template <typename T, typename... KwArgs>
std::vector<std::tuple<taylor_adaptive_batch<T>, step_callback_batch<T>, std::vector<T>>>
ensemble_propagate_grid_batch(const taylor_adaptive_batch<T> &ta, const std::vector<T> &grid, std::size_t n_iter,
                              const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen,
                              const KwArgs &...kw_args)
{
    auto [max_steps, max_delta_ts, cb]
        = detail::taylor_propagate_common_ops_batch<T, true, true>(ta.get_batch_size(), kw_args...);

    return detail::ensemble_propagate_grid_batch_impl(ta, grid, n_iter, gen, max_steps, max_delta_ts, std::move(cb));
}

HEYOKA_END_NAMESPACE

#endif
