// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/taylor.hpp>

// NOTE: these actions will be performed concurrently from
// multiple threads of exection:
// - invocation of the generator's call operator,
// - copy construction of the events' callbacks and of the propagate callback,
// - invocation of the call operator of the copies of the callbacks
//   (both event & propagate callbacks).

namespace heyoka::detail
{

namespace
{

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_until_generic(const taylor_adaptive<T> &ta, T t, std::size_t n_iter,
                                 const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                                 std::size_t max_steps, T max_delta_t,
                                 const std::function<bool(taylor_adaptive<T> &)> &cb, bool write_tc, bool with_c_out)
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

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                           kw::callback = cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::get<0>(loc_ret), std::get<1>(loc_ret), std::get<2>(loc_ret),
                                  std::get<3>(loc_ret), std::move(std::get<4>(loc_ret)));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        retval.push_back(std::move(*opt));
    }

    return retval;
}

template <typename T>
std::vector<std::tuple<taylor_adaptive<T>, taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>>
ensemble_propagate_for_generic(const taylor_adaptive<T> &ta, T delta_t, std::size_t n_iter,
                               const std::function<taylor_adaptive<T>(taylor_adaptive<T>, std::size_t)> &gen,
                               std::size_t max_steps, T max_delta_t,
                               const std::function<bool(taylor_adaptive<T> &)> &cb, bool write_tc, bool with_c_out)
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

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                         kw::callback = cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::get<0>(loc_ret), std::get<1>(loc_ret), std::get<2>(loc_ret),
                                  std::get<3>(loc_ret), std::move(std::get<4>(loc_ret)));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        retval.push_back(std::move(*opt));
    }

    return retval;
}

} // namespace

template <>
std::vector<std::tuple<taylor_adaptive<double>, taylor_outcome, double, double, std::size_t,
                       std::optional<continuous_output<double>>>>
ensemble_propagate_until_impl<double>(
    const taylor_adaptive<double> &ta, double t, std::size_t n_iter,
    const std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)> &gen, std::size_t max_steps,
    double max_delta_t, const std::function<bool(taylor_adaptive<double> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_until_generic(ta, t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

template <>
std::vector<std::tuple<taylor_adaptive<long double>, taylor_outcome, long double, long double, std::size_t,
                       std::optional<continuous_output<long double>>>>
ensemble_propagate_until_impl<long double>(
    const taylor_adaptive<long double> &ta, long double t, std::size_t n_iter,
    const std::function<taylor_adaptive<long double>(taylor_adaptive<long double>, std::size_t)> &gen,
    std::size_t max_steps, long double max_delta_t, const std::function<bool(taylor_adaptive<long double> &)> &cb,
    bool write_tc, bool with_c_out)
{
    return ensemble_propagate_until_generic(ta, t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::vector<std::tuple<taylor_adaptive<mppp::real128>, taylor_outcome, mppp::real128, mppp::real128, std::size_t,
                       std::optional<continuous_output<mppp::real128>>>>
ensemble_propagate_until_impl<mppp::real128>(
    const taylor_adaptive<mppp::real128> &ta, mppp::real128 t, std::size_t n_iter,
    const std::function<taylor_adaptive<mppp::real128>(taylor_adaptive<mppp::real128>, std::size_t)> &gen,
    std::size_t max_steps, mppp::real128 max_delta_t, const std::function<bool(taylor_adaptive<mppp::real128> &)> &cb,
    bool write_tc, bool with_c_out)
{
    return ensemble_propagate_until_generic(ta, t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

#endif

template <>
std::vector<std::tuple<taylor_adaptive<double>, taylor_outcome, double, double, std::size_t,
                       std::optional<continuous_output<double>>>>
ensemble_propagate_for_impl<double>(
    const taylor_adaptive<double> &ta, double delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)> &gen, std::size_t max_steps,
    double max_delta_t, const std::function<bool(taylor_adaptive<double> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_for_generic(ta, delta_t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

template <>
std::vector<std::tuple<taylor_adaptive<long double>, taylor_outcome, long double, long double, std::size_t,
                       std::optional<continuous_output<long double>>>>
ensemble_propagate_for_impl<long double>(
    const taylor_adaptive<long double> &ta, long double delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive<long double>(taylor_adaptive<long double>, std::size_t)> &gen,
    std::size_t max_steps, long double max_delta_t, const std::function<bool(taylor_adaptive<long double> &)> &cb,
    bool write_tc, bool with_c_out)
{
    return ensemble_propagate_for_generic(ta, delta_t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::vector<std::tuple<taylor_adaptive<mppp::real128>, taylor_outcome, mppp::real128, mppp::real128, std::size_t,
                       std::optional<continuous_output<mppp::real128>>>>
ensemble_propagate_for_impl<mppp::real128>(
    const taylor_adaptive<mppp::real128> &ta, mppp::real128 delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive<mppp::real128>(taylor_adaptive<mppp::real128>, std::size_t)> &gen,
    std::size_t max_steps, mppp::real128 max_delta_t, const std::function<bool(taylor_adaptive<mppp::real128> &)> &cb,
    bool write_tc, bool with_c_out)
{
    return ensemble_propagate_for_generic(ta, delta_t, n_iter, gen, max_steps, max_delta_t, cb, write_tc, with_c_out);
}

#endif

namespace
{

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_until_batch_generic(
    const taylor_adaptive_batch<T> &ta, T t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, std::size_t max_steps,
    const std::vector<T> &max_delta_ts, const std::function<bool(taylor_adaptive_batch<T> &)> &cb, bool write_tc,
    bool with_c_out)
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

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_ts,
                                           kw::callback = cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::move(loc_ret));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        retval.push_back(std::move(*opt));
    }

    return retval;
}

template <typename T>
std::vector<std::tuple<taylor_adaptive_batch<T>, std::optional<continuous_output_batch<T>>>>
ensemble_propagate_for_batch_generic(
    const taylor_adaptive_batch<T> &ta, T delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<T>(taylor_adaptive_batch<T>, std::size_t)> &gen, std::size_t max_steps,
    const std::vector<T> &max_delta_ts, const std::function<bool(taylor_adaptive_batch<T> &)> &cb, bool write_tc,
    bool with_c_out)
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

            // Do the propagation.
            auto loc_ret
                = local_ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_ts,
                                         kw::callback = cb, kw::write_tc = write_tc, kw::c_output = with_c_out);

            // Assign the results.
            opt_retval[i].emplace(std::move(local_ta), std::move(loc_ret));
        }
    });

    // Move the results from opt_retval to retval.
    for (auto &opt : opt_retval) {
        retval.push_back(std::move(*opt));
    }

    return retval;
}

} // namespace

template <>
std::vector<std::tuple<taylor_adaptive_batch<double>, std::optional<continuous_output_batch<double>>>>
ensemble_propagate_until_batch_impl<double>(
    const taylor_adaptive_batch<double> &ta, double t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<double>(taylor_adaptive_batch<double>, std::size_t)> &gen,
    std::size_t max_steps, const std::vector<double> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch<double> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_until_batch_generic(ta, t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                  with_c_out);
}

template <>
std::vector<std::tuple<taylor_adaptive_batch<long double>, std::optional<continuous_output_batch<long double>>>>
ensemble_propagate_until_batch_impl<long double>(
    const taylor_adaptive_batch<long double> &ta, long double t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<long double>(taylor_adaptive_batch<long double>, std::size_t)> &gen,
    std::size_t max_steps, const std::vector<long double> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch<long double> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_until_batch_generic(ta, t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                  with_c_out);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::vector<std::tuple<taylor_adaptive_batch<mppp::real128>, std::optional<continuous_output_batch<mppp::real128>>>>
ensemble_propagate_until_batch_impl<mppp::real128>(
    const taylor_adaptive_batch<mppp::real128> &ta, mppp::real128 t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<mppp::real128>(taylor_adaptive_batch<mppp::real128>, std::size_t)> &gen,
    std::size_t max_steps, const std::vector<mppp::real128> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch<mppp::real128> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_until_batch_generic(ta, t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                  with_c_out);
}

#endif

template <>
std::vector<std::tuple<taylor_adaptive_batch<double>, std::optional<continuous_output_batch<double>>>>
ensemble_propagate_for_batch_impl<double>(
    const taylor_adaptive_batch<double> &ta, double delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<double>(taylor_adaptive_batch<double>, std::size_t)> &gen,
    std::size_t max_steps, const std::vector<double> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch<double> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_for_batch_generic(ta, delta_t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                with_c_out);
}

template <>
std::vector<std::tuple<taylor_adaptive_batch<long double>, std::optional<continuous_output_batch<long double>>>>
ensemble_propagate_for_batch_impl<long double>(
    const taylor_adaptive_batch<long double> &ta, long double delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<long double>(taylor_adaptive_batch<long double>, std::size_t)> &gen,
    std::size_t max_steps, const std::vector<long double> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch<long double> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_for_batch_generic(ta, delta_t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                with_c_out);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::vector<std::tuple<taylor_adaptive_batch<mppp::real128>, std::optional<continuous_output_batch<mppp::real128>>>>
ensemble_propagate_for_batch_impl<mppp::real128>(
    const taylor_adaptive_batch<mppp::real128> &ta, mppp::real128 delta_t, std::size_t n_iter,
    const std::function<taylor_adaptive_batch<mppp::real128>(taylor_adaptive_batch<mppp::real128>, std::size_t)> &gen,
    std::size_t max_steps, const std::vector<mppp::real128> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch<mppp::real128> &)> &cb, bool write_tc, bool with_c_out)
{
    return ensemble_propagate_for_batch_generic(ta, delta_t, n_iter, gen, max_steps, max_delta_ts, cb, write_tc,
                                                with_c_out);
}

#endif

} // namespace heyoka::detail
