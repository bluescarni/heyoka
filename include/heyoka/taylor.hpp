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
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

namespace detail
{

HEYOKA_DLL_PUBLIC llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &, std::uint32_t, std::uint32_t,
                                                 std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_load_diff(llvm_state &, llvm::Value *, std::uint32_t, llvm::Value *,
                                                  llvm::Value *);

HEYOKA_DLL_PUBLIC std::string taylor_mangle_suffix(llvm::Type *);

} // namespace detail

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_decompose(std::vector<expression>);
HEYOKA_DLL_PUBLIC std::vector<expression> taylor_decompose(std::vector<std::pair<expression, expression>>);

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_jet_dbl(llvm_state &, const std::string &, std::vector<expression>,
                                                             std::uint32_t, std::uint32_t, bool, bool);
HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_jet_ldbl(llvm_state &, const std::string &,
                                                              std::vector<expression>, std::uint32_t, std::uint32_t,
                                                              bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_jet_f128(llvm_state &, const std::string &,
                                                              std::vector<expression>, std::uint32_t, std::uint32_t,
                                                              bool, bool);

#endif

template <typename T>
std::vector<expression> taylor_add_jet(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                       std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                       bool compact_mode)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_jet_dbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_jet_ldbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_jet_f128(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_jet_dbl(llvm_state &, const std::string &,
                                                             std::vector<std::pair<expression, expression>>,
                                                             std::uint32_t, std::uint32_t, bool, bool);
HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_jet_ldbl(llvm_state &, const std::string &,
                                                              std::vector<std::pair<expression, expression>>,
                                                              std::uint32_t, std::uint32_t, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_jet_f128(llvm_state &, const std::string &,
                                                              std::vector<std::pair<expression, expression>>,
                                                              std::uint32_t, std::uint32_t, bool, bool);

#endif

template <typename T>
std::vector<expression> taylor_add_jet(llvm_state &s, const std::string &name,
                                       std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                       std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_jet_dbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_jet_ldbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_jet_f128(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_adaptive_step_dbl(llvm_state &, const std::string &,
                                                                       std::vector<expression>, double, std::uint32_t,
                                                                       bool, bool);
HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_adaptive_step_ldbl(llvm_state &, const std::string &,
                                                                        std::vector<expression>, long double,
                                                                        std::uint32_t, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_adaptive_step_f128(llvm_state &, const std::string &,
                                                                        std::vector<expression>, mppp::real128,
                                                                        std::uint32_t, bool, bool);

#endif

template <typename T>
std::vector<expression> taylor_add_adaptive_step(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                                 T tol, std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_adaptive_step_dbl(s, name, std::move(sys), tol, batch_size, high_accuracy, compact_mode);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_adaptive_step_ldbl(s, name, std::move(sys), tol, batch_size, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_adaptive_step_f128(s, name, std::move(sys), tol, batch_size, high_accuracy, compact_mode);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_adaptive_step_dbl(llvm_state &, const std::string &,
                                                                       std::vector<std::pair<expression, expression>>,
                                                                       double, std::uint32_t, bool, bool);
HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_adaptive_step_ldbl(llvm_state &, const std::string &,
                                                                        std::vector<std::pair<expression, expression>>,
                                                                        long double, std::uint32_t, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_add_adaptive_step_f128(llvm_state &, const std::string &,
                                                                        std::vector<std::pair<expression, expression>>,
                                                                        mppp::real128, std::uint32_t, bool, bool);

#endif

template <typename T>
std::vector<expression> taylor_add_adaptive_step(llvm_state &s, const std::string &name,
                                                 std::vector<std::pair<expression, expression>> sys, T tol,
                                                 std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_adaptive_step_dbl(s, name, std::move(sys), tol, batch_size, high_accuracy, compact_mode);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_adaptive_step_ldbl(s, name, std::move(sys), tol, batch_size, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_adaptive_step_f128(s, name, std::move(sys), tol, batch_size, high_accuracy, compact_mode);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

// Enum to represnt the outcome of a Taylor integration
// stepping function.
enum class taylor_outcome {
    success,     // Integration step was successful, no time/step limits were reached.
    step_limit,  // Maximum number of steps reached.
    time_limit,  // Time limit reached.
    interrupted, // Interrupted by user-provided stopping criterion.
    err_nf_state // Non-finite initial state detected.
};

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(time);
IGOR_MAKE_NAMED_ARGUMENT(tol);
IGOR_MAKE_NAMED_ARGUMENT(high_accuracy);
IGOR_MAKE_NAMED_ARGUMENT(compact_mode);

} // namespace kw

namespace detail
{

// Helper for parsing common options for the Taylor integrators.
template <typename T, typename... KwArgs>
inline auto taylor_adaptive_common_ops(KwArgs &&... kw_args)
{
    igor::parser p{kw_args...};

    // High accuracy mode (defaults to false).
    auto high_accuracy = [&p]() -> bool {
        if constexpr (p.has(kw::high_accuracy)) {
            return std::forward<decltype(p(kw::high_accuracy))>(p(kw::high_accuracy));
        } else {
            return false;
        }
    }();

    // tol (defaults to eps).
    auto tol = [&p, high_accuracy]() -> T {
        if constexpr (p.has(kw::tol)) {
            auto retval = std::forward<decltype(p(kw::tol))>(p(kw::tol));
            if (retval != T(0)) {
                // NOTE: this covers the NaN case as well.
                return retval;
            }
            // NOTE: zero tolerance will be interpreted
            // as automatically-deduced by falling through
            // the code below.
        }
        auto retval = std::numeric_limits<T>::epsilon();
        if (high_accuracy) {
            // Add extra precision in high-accuracy mode.
            retval *= T(1e-4);
        }

        return retval;
    }();

    // Compact mode (defaults to false).
    auto compact_mode = [&p]() -> bool {
        if constexpr (p.has(kw::compact_mode)) {
            return std::forward<decltype(p(kw::compact_mode))>(p(kw::compact_mode));
        } else {
            return false;
        }
    }();

    return std::tuple{high_accuracy, tol, compact_mode};
}

template <typename T>
class HEYOKA_DLL_PUBLIC taylor_adaptive_impl
{
    // State vector.
    std::vector<T> m_state;
    // Time.
    T m_time;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Taylor decomposition.
    std::vector<expression> m_dc;
    // The stepper.
    using step_f_t = void (*)(T *, T *);
    step_f_t m_step_f;

    HEYOKA_DLL_LOCAL std::tuple<taylor_outcome, T> step_impl(T);

    // Private implementation-detail constructor machinery.
    template <typename U>
    void finalise_ctor_impl(U, std::vector<T>, T, T, bool, bool);
    template <typename U, typename... KwArgs>
    void finalise_ctor(U sys, std::vector<T> state, KwArgs &&... kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of an adaptive Taylor integrator contain "
                          "unnamed arguments.");
        } else {
            // Initial time (defaults to zero).
            const auto time = [&p]() -> T {
                if constexpr (p.has(kw::time)) {
                    return std::forward<decltype(p(kw::time))>(p(kw::time));
                } else {
                    return T(0);
                }
            }();

            const auto [high_accuracy, tol, compact_mode]
                = taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

            finalise_ctor_impl(std::move(sys), std::move(state), time, tol, high_accuracy, compact_mode);
        }
    }

public:
    template <typename... KwArgs>
    explicit taylor_adaptive_impl(std::vector<expression> sys, std::vector<T> state, KwArgs &&... kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(state), std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_impl(std::vector<std::pair<expression, expression>> sys, std::vector<T> state,
                                  KwArgs &&... kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(state), std::forward<KwArgs>(kw_args)...);
    }

    taylor_adaptive_impl(const taylor_adaptive_impl &);
    taylor_adaptive_impl(taylor_adaptive_impl &&) noexcept;

    taylor_adaptive_impl &operator=(const taylor_adaptive_impl &);
    taylor_adaptive_impl &operator=(taylor_adaptive_impl &&) noexcept;

    ~taylor_adaptive_impl();

    const llvm_state &get_llvm_state() const;

    const std::vector<expression> &get_decomposition() const;

    T get_time() const
    {
        return m_time;
    }
    const std::vector<T> &get_state() const
    {
        return m_state;
    }
    const T *get_state_data() const
    {
        return m_state.data();
    }
    T *get_state_data()
    {
        return m_state.data();
    }

    void set_state(const std::vector<T> &);
    void set_time(T);

    std::tuple<taylor_outcome, T> step();
    std::tuple<taylor_outcome, T> step_backward();
    std::tuple<taylor_outcome, T> step(T);
    // NOTE: return values:
    // - outcome,
    // - min abs(timestep),
    // - max abs(timestep),
    // - total number of steps successfully
    //   undertaken.
    // NOTE: the min/max timesteps are well-defined
    // only if at least 1-2 steps were taken successfully.
    std::tuple<taylor_outcome, T, T, std::size_t> propagate_for(T, std::size_t = 0);
    std::tuple<taylor_outcome, T, T, std::size_t> propagate_until(T, std::size_t = 0);

private:
    template <bool Direction, typename F>
    auto propagate_pred_impl(const F &f, std::size_t max_steps)
    {
        // Initial values for the counter,
        // the min/max abs of the integration
        // timesteps, and min/max Taylor orders.
        std::size_t step_counter = 0;
        T min_h = std::numeric_limits<T>::infinity(), max_h = 0;

        while (true) {
            const auto sres = Direction ? step() : step_backward();
            const auto &[res, h] = sres;

            if (res != taylor_outcome::success) {
                return std::tuple{res, min_h, max_h, step_counter};
            }

            // Update the number of steps
            // completed successfully.
            ++step_counter;

            // Update min_h/max_h.
            assert(!Direction || h >= 0);
            min_h = std::min(min_h, Direction ? h : -h);
            max_h = std::max(max_h, Direction ? h : -h);

            // Check the max number of steps stopping criterion.
            if (max_steps != 0u && step_counter == max_steps) {
                return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter};
            }

            // Check the stopping criterion.
            if (f(sres, *this)) {
                break;
            }
        }

        return std::tuple{taylor_outcome::interrupted, min_h, max_h, step_counter};
    }

public:
    template <typename F>
    std::tuple<taylor_outcome, T, T, std::size_t> propagate_pred(const F &f, std::size_t max_steps = 0)
    {
        return propagate_pred_impl<true>(f, max_steps);
    }
    template <typename F>
    std::tuple<taylor_outcome, T, T, std::size_t> propagate_pred_backward(const F &f, std::size_t max_steps = 0)
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
    // The LLVM machinery.
    llvm_state m_llvm;
    // Taylor decomposition.
    std::vector<expression> m_dc;
    // The stepper.
    using step_f_t = void (*)(T *, T *);
    step_f_t m_step_f;
    // Temporary vectors for use
    // in the timestepping functions.
    std::vector<T> m_pinf;
    std::vector<T> m_minf;
    std::vector<T> m_delta_ts;

    HEYOKA_DLL_LOCAL void step_impl(std::vector<std::tuple<taylor_outcome, T>> &, const std::vector<T> &);

    // Private implementation-detail constructor machinery.
    template <typename U>
    void finalise_ctor_impl(U, std::vector<T>, std::uint32_t, std::vector<T>, T, bool, bool);
    template <typename U, typename... KwArgs>
    void finalise_ctor(U sys, std::vector<T> states, std::uint32_t batch_size, KwArgs &&... kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of an adaptive batch Taylor integrator contain "
                          "unnamed arguments.");
        } else {
            // Initial times (defaults to a vector of zeroes).
            auto times = [&p, batch_size]() -> std::vector<T> {
                if constexpr (p.has(kw::time)) {
                    return std::forward<decltype(p(kw::time))>(p(kw::time));
                } else {
                    return std::vector<T>(static_cast<typename std::vector<T>::size_type>(batch_size), T(0));
                }
            }();

            const auto [high_accuracy, tol, compact_mode]
                = taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

            finalise_ctor_impl(std::move(sys), std::move(states), batch_size, std::move(times), tol, high_accuracy,
                               compact_mode);
        }
    }

public:
    template <typename... KwArgs>
    explicit taylor_adaptive_batch_impl(std::vector<expression> sys, std::vector<T> states, std::uint32_t batch_size,
                                        KwArgs &&... kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(states), batch_size, std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch_impl(std::vector<std::pair<expression, expression>> sys, std::vector<T> states,
                                        std::uint32_t batch_size, KwArgs &&... kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(states), batch_size, std::forward<KwArgs>(kw_args)...);
    }

    taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &);
    taylor_adaptive_batch_impl(taylor_adaptive_batch_impl &&) noexcept;

    taylor_adaptive_batch_impl &operator=(const taylor_adaptive_batch_impl &);
    taylor_adaptive_batch_impl &operator=(taylor_adaptive_batch_impl &&) noexcept;

    ~taylor_adaptive_batch_impl();

    const llvm_state &get_llvm_state() const;

    const std::vector<expression> &get_decomposition() const;

    const std::vector<T> &get_times() const
    {
        return m_times;
    }
    const T *get_times_data() const
    {
        return m_times.data();
    }
    T *get_times_data()
    {
        return m_times.data();
    }
    const std::vector<T> &get_states() const
    {
        return m_states;
    }
    const T *get_states_data() const
    {
        return m_states.data();
    }
    T *get_states_data()
    {
        return m_states.data();
    }

    void set_states(const std::vector<T> &);
    void set_times(const std::vector<T> &);

    void step(std::vector<std::tuple<taylor_outcome, T>> &);
    void step_backward(std::vector<std::tuple<taylor_outcome, T>> &);
    void step(std::vector<std::tuple<taylor_outcome, T>> &, const std::vector<T> &);
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
