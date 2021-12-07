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

} // namespace detail

template <typename T, typename... KwArgs>
inline std::vector<
    std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_until(const taylor_adaptive<T> &ta, T t, std::size_t n_iter,
                         const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                         KwArgs &&...kw_args)
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");

    auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
        = detail::taylor_propagate_common_ops<T, false>(std::forward<KwArgs>(kw_args)...);

    return detail::ensemble_propagate_until_impl(ta, t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

} // namespace heyoka

#endif
