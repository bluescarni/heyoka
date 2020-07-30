// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TAYLOR_HPP
#define HEYOKA_TAYLOR_HPP

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_decompose(std::vector<expression>);
HEYOKA_DLL_PUBLIC std::vector<expression> taylor_decompose(std::vector<std::pair<expression, expression>>);

// Enum to represnt the outcome of a Taylor integration
// stepping function.
enum class taylor_outcome {
    success,           // Integration step was successful, no time/step limits were reached.
    step_limit,        // Maximum number of steps reached.
    time_limit,        // Time limit reached.
    interrupted,       // Interrupted by user-provided stopping criterion.
    err_nf_state,      // Non-finite initial state detected.
    err_nf_derivative, // Non-finite derivative detected.
    err_nan_rho        // NaN estimation of the convergence radius.
};

namespace detail
{

template <typename T>
class HEYOKA_DLL_PUBLIC taylor_adaptive_impl
{
    // State vector.
    std::vector<T> m_state;
    // Time.
    T m_time;
    // Relative and absolute tolerances.
    T m_rtol, m_atol;
    // Taylor orders corresponding to the
    // above tolerances.
    std::uint32_t m_order_r, m_order_a;
    // Vector of pre-computed inverse orders
    // (that is, for i >= 1, m_inv_order[i] = 1 / i).
    std::vector<T> m_inv_order;
    // The factor by which rho must
    // be multiplied in order to determine
    // the integration timestep.
    // There are two versions of this
    // factor, one for the relative Taylor
    // order and the other for the absolute
    // Taylor order.
    T m_rhofac_r, m_rhofac_a;
    // The LLVM machinery.
    llvm_state m_llvm;
    // The jet of normalised derivatives.
    std::vector<T> m_jet;
    // The functions to compute the derivatives.
    using jet_f_t = void (*)(T *);
    jet_f_t m_jet_f_r, m_jet_f_a;
    // Taylor decomposition.
    std::vector<expression> m_dc;

    template <bool, bool>
    HEYOKA_DLL_LOCAL std::tuple<taylor_outcome, T, std::uint32_t> step_impl(T);

    // Private implementation-detail constructor machinery.
    struct p_tag {
    };
    template <typename U>
    HEYOKA_DLL_LOCAL explicit taylor_adaptive_impl(p_tag, U, std::vector<T>, T, T, T, unsigned);

public:
    explicit taylor_adaptive_impl(std::vector<expression>, std::vector<T>, T, T, T, unsigned = 3);
    explicit taylor_adaptive_impl(std::vector<std::pair<expression, expression>>, std::vector<T>, T, T, T,
                                  unsigned = 3);

    taylor_adaptive_impl(const taylor_adaptive_impl &);
    taylor_adaptive_impl(taylor_adaptive_impl &&) noexcept;

    taylor_adaptive_impl &operator=(const taylor_adaptive_impl &);
    taylor_adaptive_impl &operator=(taylor_adaptive_impl &&) noexcept;

    ~taylor_adaptive_impl();

    std::string get_ir() const;
    const std::vector<expression> &get_decomposition() const;

    T get_time() const
    {
        return m_time;
    }
    const std::vector<T> &get_state() const
    {
        return m_state;
    }

    void set_state(const std::vector<T> &);
    void set_time(T);

    std::tuple<taylor_outcome, T, std::uint32_t> step();
    std::tuple<taylor_outcome, T, std::uint32_t> step_backward();
    std::tuple<taylor_outcome, T, std::uint32_t> step(T);
    // NOTE: return values:
    // - outcome,
    // - min abs(timestep),
    // - max abs(timestep),
    // - min Taylor order,
    // - max Taylor order,
    // - total number of steps successfully
    //   undertaken.
    // NOTE: the min/max timesteps and orders are well-defined
    // only if at least 1-2 steps were taken successfully.
    std::tuple<taylor_outcome, T, T, std::uint32_t, std::uint32_t, std::size_t> propagate_for(T, std::size_t = 0);
    std::tuple<taylor_outcome, T, T, std::uint32_t, std::uint32_t, std::size_t> propagate_until(T, std::size_t = 0);

private:
    template <bool Direction, typename F>
    auto propagate_pred_impl(const F &f, std::size_t max_steps)
    {
        // Initial values for the counter,
        // the min/max abs of the integration
        // timesteps, and min/max Taylor orders.
        std::size_t step_counter = 0;
        T min_h = std::numeric_limits<T>::infinity(), max_h = 0;
        std::uint32_t min_order = std::numeric_limits<std::uint32_t>::max(), max_order = 0;

        while (true) {
            const auto sres = Direction ? step() : step_backward();
            const auto &[res, h, t_order] = sres;

            if (res != taylor_outcome::success) {
                return std::tuple{res, min_h, max_h, min_order, max_order, step_counter};
            }

            // Update the number of steps
            // completed successfully.
            ++step_counter;

            // Update min/max Taylor orders.
            min_order = std::min(min_order, t_order);
            max_order = std::max(max_order, t_order);

            // Update min_h/max_h.
            assert(!Direction || h >= 0);
            min_h = std::min(min_h, Direction ? h : std::abs(h));
            max_h = std::max(max_h, Direction ? h : std::abs(h));

            // Check the max number of steps stopping criterion.
            if (max_steps != 0u && step_counter == max_steps) {
                return std::tuple{taylor_outcome::step_limit, min_h, max_h, min_order, max_order, step_counter};
            }

            // Check the stopping criterion.
            if (f(sres, *this)) {
                break;
            }
        }

        return std::tuple{taylor_outcome::interrupted, min_h, max_h, min_order, max_order, step_counter};
    }

public:
    template <typename F>
    std::tuple<taylor_outcome, T, T, std::uint32_t, std::uint32_t, std::size_t>
    propagate_pred(const F &f, std::size_t max_steps = 0)
    {
        return propagate_pred_impl<true>(f, max_steps);
    }
    template <typename F>
    std::tuple<taylor_outcome, T, T, std::uint32_t, std::uint32_t, std::size_t>
    propagate_pred_backward(const F &f, std::size_t max_steps = 0)
    {
        return propagate_pred_impl<false>(f, max_steps);
    }
};

} // namespace detail

class HEYOKA_DLL_PUBLIC taylor_adaptive_dbl : public detail::taylor_adaptive_impl<double>
{
public:
    using base = detail::taylor_adaptive_impl<double>;
    using base::base;
};

class HEYOKA_DLL_PUBLIC taylor_adaptive_ldbl : public detail::taylor_adaptive_impl<long double>
{
public:
    using base = detail::taylor_adaptive_impl<long double>;
    using base::base;
};

#if defined(HEYOKA_HAVE_REAL128)

class HEYOKA_DLL_PUBLIC taylor_adaptive_f128 : public detail::taylor_adaptive_impl<mppp::real128>
{
public:
    using base = detail::taylor_adaptive_impl<mppp::real128>;
    using base::base;
};

#endif

namespace detail
{

template <typename T>
struct taylor_adaptive_t_impl {
    static_assert(always_false_v<T>, "Unhandled type.");
};

template <>
struct taylor_adaptive_t_impl<double> {
    using type = taylor_adaptive_dbl;
};

template <>
struct taylor_adaptive_t_impl<long double> {
    using type = taylor_adaptive_ldbl;
};

#if defined(HEYOKA_HAVE_REAL128)

template <>
struct taylor_adaptive_t_impl<mppp::real128> {
    using type = taylor_adaptive_f128;
};

#endif

} // namespace detail

template <typename T>
using taylor_adaptive = typename detail::taylor_adaptive_t_impl<T>::type;

namespace detail
{

template <typename T>
class HEYOKA_DLL_PUBLIC taylor_adaptive_batch_impl
{
    // The batch size.
    std::uint32_t m_batch_size;
    // State vectors.
    std::vector<T> m_states;
    // Times.
    std::vector<T> m_times;
    // Relative and absolute tolerances.
    T m_rtol, m_atol;
    // Taylor orders corresponding to the
    // above tolerances.
    std::uint32_t m_order_r, m_order_a;
    // Vector of pre-computed inverse orders
    // (that is, for i >= 1, m_inv_order[i] = 1 / i).
    std::vector<T> m_inv_order;
    // The factor by which rho must
    // be multiplied in order to determine
    // the integration timestep.
    // There are two versions of this
    // factor, one for the relative Taylor
    // order and the other for the absolute
    // Taylor order.
    T m_rhofac_r, m_rhofac_a;
    // The LLVM machinery.
    llvm_state m_llvm;
    // The jets of normalised derivatives.
    std::vector<T> m_jet;
    // The functions to compute the derivatives.
    using jet_f_t = void (*)(T *);
    jet_f_t m_jet_f_r, m_jet_f_a;
    // LLVM IR.
    std::string m_ir;
    // Taylor decomposition.
    std::vector<expression> m_dc;
    // Temporary vectors for use
    // in the timestepping functions.
    std::vector<T> m_max_abs_states;
    std::vector<char> m_use_abs_tol;
    std::vector<T> m_max_abs_diff_om1;
    std::vector<T> m_max_abs_diff_o;
    std::vector<T> m_rho_om1;
    std::vector<T> m_rho_o;
    std::vector<T> m_h;
    std::vector<T> m_cur_h;

    template <bool, bool>
    HEYOKA_DLL_LOCAL void step_impl(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &,
                                    const std::vector<T> &);

    // Private implementation-detail constructor machinery.
    struct p_tag {
    };
    template <typename U>
    HEYOKA_DLL_LOCAL explicit taylor_adaptive_batch_impl(p_tag, U, std::vector<T>, std::vector<T>, T, T, std::uint32_t,
                                                         unsigned);

public:
    explicit taylor_adaptive_batch_impl(std::vector<expression>, std::vector<T>, std::vector<T>, T, T, std::uint32_t,
                                        unsigned = 3);
    explicit taylor_adaptive_batch_impl(std::vector<std::pair<expression, expression>>, std::vector<T>, std::vector<T>,
                                        T, T, std::uint32_t, unsigned = 3);

    taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &) = delete;
    taylor_adaptive_batch_impl(taylor_adaptive_batch_impl &&) noexcept;

    taylor_adaptive_batch_impl &operator=(const taylor_adaptive_batch_impl &) = delete;
    taylor_adaptive_batch_impl &operator=(taylor_adaptive_batch_impl &&) noexcept;

    ~taylor_adaptive_batch_impl();

    std::string get_ir() const;
    const std::vector<expression> &get_decomposition() const;

    const std::vector<T> &get_times() const
    {
        return m_times;
    }
    const std::vector<T> &get_states() const
    {
        return m_states;
    }

    void set_states(const std::vector<T> &);
    void set_times(const std::vector<T> &);

    void step(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &);
    void step_backward(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &);
    void step(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &, T);
};

} // namespace detail

class HEYOKA_DLL_PUBLIC taylor_adaptive_batch_dbl : public detail::taylor_adaptive_batch_impl<double>
{
public:
    using base = detail::taylor_adaptive_batch_impl<double>;
    using base::base;
};

class HEYOKA_DLL_PUBLIC taylor_adaptive_batch_ldbl : public detail::taylor_adaptive_batch_impl<long double>
{
public:
    using base = detail::taylor_adaptive_batch_impl<long double>;
    using base::base;
};

#if defined(HEYOKA_HAVE_REAL128)

class HEYOKA_DLL_PUBLIC taylor_adaptive_batch_f128 : public detail::taylor_adaptive_batch_impl<mppp::real128>
{
public:
    using base = detail::taylor_adaptive_batch_impl<mppp::real128>;
    using base::base;
};

#endif

namespace detail
{

template <typename T>
struct taylor_adaptive_batch_t_impl {
    static_assert(always_false_v<T>, "Unhandled type.");
};

template <>
struct taylor_adaptive_batch_t_impl<double> {
    using type = taylor_adaptive_batch_dbl;
};

template <>
struct taylor_adaptive_batch_t_impl<long double> {
    using type = taylor_adaptive_batch_ldbl;
};

#if defined(HEYOKA_HAVE_REAL128)

template <>
struct taylor_adaptive_batch_t_impl<mppp::real128> {
    using type = taylor_adaptive_batch_f128;
};

#endif

} // namespace detail

template <typename T>
using taylor_adaptive_batch = typename detail::taylor_adaptive_batch_t_impl<T>::type;

} // namespace heyoka

#endif
