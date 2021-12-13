// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/taylor.hpp>

namespace heyoka
{

namespace detail
{

template <typename T>
inline std::vector<
    std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_until_impl(const taylor_adaptive<T> &, T, std::size_t,
                              const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &, std::size_t,
                              T, const std::function<bool(taylor_adaptive<T> &)> &, bool, bool)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    throw;
}

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<double>, taylor_outcome, double, double, std::size_t,
                                         std::optional<continuous_output<double>>>>
ensemble_propagate_until_impl<double>(
    const taylor_adaptive<double> &, double, std::size_t,
    const std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)> &, std::size_t, double,
    const std::function<bool(taylor_adaptive<double> &)> &, bool, bool);

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<long double>, taylor_outcome, long double, long double,
                                         std::size_t, std::optional<continuous_output<long double>>>>
ensemble_propagate_until_impl<long double>(
    const taylor_adaptive<long double> &, long double, std::size_t,
    const std::function<taylor_adaptive<long double>(taylor_adaptive<long double>, std::size_t)> &, std::size_t,
    long double, const std::function<bool(taylor_adaptive<long double> &)> &, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<mppp::real128>, taylor_outcome, mppp::real128, mppp::real128,
                                         std::size_t, std::optional<continuous_output<mppp::real128>>>>
ensemble_propagate_until_impl<mppp::real128>(
    const taylor_adaptive<mppp::real128> &, mppp::real128, std::size_t,
    const std::function<taylor_adaptive<mppp::real128>(taylor_adaptive<mppp::real128>, std::size_t)> &, std::size_t,
    mppp::real128, const std::function<bool(taylor_adaptive<mppp::real128> &)> &, bool, bool);

#endif

template <typename T>
inline std::vector<
    std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_for_impl(const taylor_adaptive<T> &, T, std::size_t,
                            const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &, std::size_t, T,
                            const std::function<bool(taylor_adaptive<T> &)> &, bool, bool)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    throw;
}

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<double>, taylor_outcome, double, double, std::size_t,
                                         std::optional<continuous_output<double>>>>
ensemble_propagate_for_impl<double>(
    const taylor_adaptive<double> &, double, std::size_t,
    const std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)> &, std::size_t, double,
    const std::function<bool(taylor_adaptive<double> &)> &, bool, bool);

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<long double>, taylor_outcome, long double, long double,
                                         std::size_t, std::optional<continuous_output<long double>>>>
ensemble_propagate_for_impl<long double>(
    const taylor_adaptive<long double> &, long double, std::size_t,
    const std::function<taylor_adaptive<long double>(taylor_adaptive<long double>, std::size_t)> &, std::size_t,
    long double, const std::function<bool(taylor_adaptive<long double> &)> &, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<mppp::real128>, taylor_outcome, mppp::real128, mppp::real128,
                                         std::size_t, std::optional<continuous_output<mppp::real128>>>>
ensemble_propagate_for_impl<mppp::real128>(
    const taylor_adaptive<mppp::real128> &, mppp::real128, std::size_t,
    const std::function<taylor_adaptive<mppp::real128>(taylor_adaptive<mppp::real128>, std::size_t)> &, std::size_t,
    mppp::real128, const std::function<bool(taylor_adaptive<mppp::real128> &)> &, bool, bool);

#endif

template <typename T>
inline std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::vector<T>>>
ensemble_propagate_grid_impl(const taylor_adaptive<T> &, const std::vector<T> &, std::size_t,
                             const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &, std::size_t, T,
                             const std::function<bool(taylor_adaptive<T> &)> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    throw;
}

template <>
HEYOKA_DLL_PUBLIC
    std::vector<std::tuple<taylor_adaptive<double>, taylor_outcome, double, double, std::size_t, std::vector<double>>>
    ensemble_propagate_grid_impl<double>(
        const taylor_adaptive<double> &, const std::vector<double> &, std::size_t,
        const std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)> &, std::size_t, double,
        const std::function<bool(taylor_adaptive<double> &)> &);

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<long double>, taylor_outcome, long double, long double,
                                         std::size_t, std::vector<long double>>>
ensemble_propagate_grid_impl<long double>(
    const taylor_adaptive<long double> &, const std::vector<long double> &, std::size_t,
    const std::function<taylor_adaptive<long double>(taylor_adaptive<long double>, std::size_t)> &, std::size_t,
    long double, const std::function<bool(taylor_adaptive<long double> &)> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive<mppp::real128>, taylor_outcome, mppp::real128, mppp::real128,
                                         std::size_t, std::vector<mppp::real128>>>
ensemble_propagate_grid_impl<mppp::real128>(
    const taylor_adaptive<mppp::real128> &, const std::vector<mppp::real128> &, std::size_t,
    const std::function<taylor_adaptive<mppp::real128>(taylor_adaptive<mppp::real128>, std::size_t)> &, std::size_t,
    mppp::real128, const std::function<bool(taylor_adaptive<mppp::real128> &)> &);

#endif

} // namespace detail

template <typename T, typename... KwArgs>
inline std::vector<
    std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_until(const taylor_adaptive<T> &ta, T t, std::size_t n_iter,
                         const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                         KwArgs &&...kw_args)
{
    auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops<T, false>(std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_until_impl(ta, t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

template <typename T, typename... KwArgs>
inline std::vector<
    std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_for(const taylor_adaptive<T> &ta, T delta_t, std::size_t n_iter,
                       const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                       KwArgs &&...kw_args)
{
    auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops<T, false>(std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_for_impl(ta, delta_t, n_iter, gen, max_steps, max_delta_t, cb, write_tc,
                                               with_c_out);
}

template <typename T, typename... KwArgs>
inline std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::vector<T>>>
ensemble_propagate_grid(const taylor_adaptive<T> &ta, const std::vector<T> &grid, std::size_t n_iter,
                        const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                        KwArgs &&...kw_args)
{
    auto [max_steps, max_delta_t, cb, _]
        = detail::taylor_propagate_common_ops<T, true>(std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_grid_impl(ta, grid, n_iter, gen, max_steps, max_delta_t, cb);
}

namespace detail
{

template <typename T>
inline std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_until_batch_impl(
    const taylor_adaptive_batch<T> &, T, std::size_t,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,
    const std::vector<T> &, const std::function<bool(taylor_adaptive_batch<T> &)> &, bool, bool)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    throw;
}

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive_batch<double>, std::optional<continuous_output_batch<double>>>>
ensemble_propagate_until_batch_impl<double>(
    const taylor_adaptive_batch<double> &, double, std::size_t,
    const std::function<taylor_adaptive_batch<double>(taylor_adaptive_batch<double>, std::size_t)> &, std::size_t,
    const std::vector<double> &, const std::function<bool(taylor_adaptive_batch<double> &)> &, bool, bool);

template <>
HEYOKA_DLL_PUBLIC
    std::vector<std::tuple<taylor_adaptive_batch<long double>, std::optional<continuous_output_batch<long double>>>>
    ensemble_propagate_until_batch_impl<long double>(
        const taylor_adaptive_batch<long double> &, long double, std::size_t,
        const std::function<taylor_adaptive_batch<long double>(taylor_adaptive_batch<long double>, std::size_t)> &,
        std::size_t, const std::vector<long double> &,
        const std::function<bool(taylor_adaptive_batch<long double> &)> &, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC
    std::vector<std::tuple<taylor_adaptive_batch<mppp::real128>, std::optional<continuous_output_batch<mppp::real128>>>>
    ensemble_propagate_until_batch_impl<mppp::real128>(
        const taylor_adaptive_batch<mppp::real128> &, mppp::real128, std::size_t,
        const std::function<taylor_adaptive_batch<mppp::real128>(taylor_adaptive_batch<mppp::real128>, std::size_t)> &,
        std::size_t, const std::vector<mppp::real128> &,
        const std::function<bool(taylor_adaptive_batch<mppp::real128> &)> &, bool, bool);

#endif

template <typename T>
inline std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_for_batch_impl(
    const taylor_adaptive_batch<T> &, T, std::size_t,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,
    const std::vector<T> &, const std::function<bool(taylor_adaptive_batch<T> &)> &, bool, bool)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    throw;
}

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive_batch<double>, std::optional<continuous_output_batch<double>>>>
ensemble_propagate_for_batch_impl<double>(
    const taylor_adaptive_batch<double> &, double, std::size_t,
    const std::function<taylor_adaptive_batch<double>(taylor_adaptive_batch<double>, std::size_t)> &, std::size_t,
    const std::vector<double> &, const std::function<bool(taylor_adaptive_batch<double> &)> &, bool, bool);

template <>
HEYOKA_DLL_PUBLIC
    std::vector<std::tuple<taylor_adaptive_batch<long double>, std::optional<continuous_output_batch<long double>>>>
    ensemble_propagate_for_batch_impl<long double>(
        const taylor_adaptive_batch<long double> &, long double, std::size_t,
        const std::function<taylor_adaptive_batch<long double>(taylor_adaptive_batch<long double>, std::size_t)> &,
        std::size_t, const std::vector<long double> &,
        const std::function<bool(taylor_adaptive_batch<long double> &)> &, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC
    std::vector<std::tuple<taylor_adaptive_batch<mppp::real128>, std::optional<continuous_output_batch<mppp::real128>>>>
    ensemble_propagate_for_batch_impl<mppp::real128>(
        const taylor_adaptive_batch<mppp::real128> &, mppp::real128, std::size_t,
        const std::function<taylor_adaptive_batch<mppp::real128>(taylor_adaptive_batch<mppp::real128>, std::size_t)> &,
        std::size_t, const std::vector<mppp::real128> &,
        const std::function<bool(taylor_adaptive_batch<mppp::real128> &)> &, bool, bool);

#endif

template <typename T>
inline std::vector<std::tuple<taylor_adaptive_batch<T>, std::vector<T>>> ensemble_propagate_grid_batch_impl(
    const taylor_adaptive_batch<T> &, const std::vector<T> &, std::size_t,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &, std::size_t,
    const std::vector<T> &, const std::function<bool(taylor_adaptive_batch<T> &)> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    throw;
}

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive_batch<double>, std::vector<double>>>
ensemble_propagate_grid_batch_impl<double>(
    const taylor_adaptive_batch<double> &, const std::vector<double> &, std::size_t,
    const std::function<taylor_adaptive_batch<double>(taylor_adaptive_batch<double>, std::size_t)> &, std::size_t,
    const std::vector<double> &, const std::function<bool(taylor_adaptive_batch<double> &)> &);

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive_batch<long double>, std::vector<long double>>>
ensemble_propagate_grid_batch_impl<long double>(
    const taylor_adaptive_batch<long double> &, const std::vector<long double> &, std::size_t,
    const std::function<taylor_adaptive_batch<long double>(taylor_adaptive_batch<long double>, std::size_t)> &,
    std::size_t, const std::vector<long double> &, const std::function<bool(taylor_adaptive_batch<long double> &)> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::vector<std::tuple<taylor_adaptive_batch<mppp::real128>, std::vector<mppp::real128>>>
ensemble_propagate_grid_batch_impl<mppp::real128>(
    const taylor_adaptive_batch<mppp::real128> &, const std::vector<mppp::real128> &, std::size_t,
    const std::function<taylor_adaptive_batch<mppp::real128>(taylor_adaptive_batch<mppp::real128>, std::size_t)> &,
    std::size_t, const std::vector<mppp::real128> &,
    const std::function<bool(taylor_adaptive_batch<mppp::real128> &)> &);

#endif

} // namespace detail

template <typename T, typename... KwArgs>
inline std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_until_batch(
    const taylor_adaptive_batch<T> &ta, T t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, KwArgs &&...kw_args)
{
    auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops_batch<T, false, true>(ta.get_batch_size(),
                                                                    std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_until_batch_impl(ta, t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                       with_c_out);
}

template <typename T, typename... KwArgs>
inline std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_for_batch(const taylor_adaptive_batch<T> &ta, T delta_t, std::size_t n_iter,
                             const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen,
                             KwArgs &&...kw_args)
{
    auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops_batch<T, false, true>(ta.get_batch_size(),
                                                                    std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_for_batch_impl(ta, delta_t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                     with_c_out);
}

template <typename T, typename... KwArgs>
inline std::vector<std::tuple<taylor_adaptive_batch<T>, std::vector<T>>>
ensemble_propagate_grid_batch(const taylor_adaptive_batch<T> &ta, const std::vector<T> &grid, std::size_t n_iter,
                              const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen,
                              KwArgs &&...kw_args)
{
    auto [max_steps, max_delta_ts, cb, _] = detail::taylor_propagate_common_ops_batch<T, true, true>(
        ta.get_batch_size(), std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_grid_batch_impl(ta, grid, n_iter, gen, max_steps, max_delta_ts, cb);
}

} // namespace heyoka

#endif
