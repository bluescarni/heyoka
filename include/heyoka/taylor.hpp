// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TAYLOR_HPP
#define HEYOKA_TAYLOR_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

namespace detail
{

// NOTE: these are various utilities useful when dealing in a generic
// fashion with numbers/params in Taylor functions.

// Helper to detect if T is a number or a param.
template <typename T>
using is_num_param = std::disjunction<std::is_same<T, number>, std::is_same<T, param>>;

template <typename T>
inline constexpr bool is_num_param_v = is_num_param<T>::value;

HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam_dbl(llvm_state &, const number &, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam_ldbl(llvm_state &, const number &, llvm::Value *, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam_f128(llvm_state &, const number &, llvm::Value *, std::uint32_t);

#endif

HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam_dbl(llvm_state &, const param &, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam_ldbl(llvm_state &, const param &, llvm::Value *, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam_f128(llvm_state &, const param &, llvm::Value *, std::uint32_t);

#endif

template <typename T, typename U>
llvm::Value *taylor_codegen_numparam(llvm_state &s, const U &n, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_codegen_numparam_dbl(s, n, par_ptr, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_codegen_numparam_ldbl(s, n, par_ptr, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_codegen_numparam_f128(s, n, par_ptr, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::string taylor_c_diff_numparam_mangle(const number &);
HEYOKA_DLL_PUBLIC std::string taylor_c_diff_numparam_mangle(const param &);

HEYOKA_DLL_PUBLIC llvm::Type *taylor_c_diff_numparam_argtype(const std::type_info &, llvm_state &, const number &);
HEYOKA_DLL_PUBLIC llvm::Type *taylor_c_diff_numparam_argtype(const std::type_info &, llvm_state &, const param &);

template <typename T, typename U>
inline llvm::Type *taylor_c_diff_numparam_argtype(llvm_state &s, const U &x)
{
    return taylor_c_diff_numparam_argtype(typeid(T), s, x);
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &, const number &, llvm::Value *,
                                                              llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &, const param &, llvm::Value *, llvm::Value *,
                                                              std::uint32_t);

} // namespace detail

namespace detail
{

HEYOKA_DLL_PUBLIC llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &, std::uint32_t, std::uint32_t,
                                                 std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_load_diff(llvm_state &, llvm::Value *, std::uint32_t, llvm::Value *,
                                                  llvm::Value *);

HEYOKA_DLL_PUBLIC std::string taylor_mangle_suffix(llvm::Type *);

} // namespace detail

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
    taylor_decompose(std::vector<expression>);
HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
    taylor_decompose(std::vector<std::pair<expression, expression>>);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet_dbl(llvm_state &, const std::string &, std::vector<expression>, std::uint32_t, std::uint32_t, bool,
                   bool);
HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet_ldbl(llvm_state &, const std::string &, std::vector<expression>, std::uint32_t, std::uint32_t, bool,
                    bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet_f128(llvm_state &, const std::string &, std::vector<expression>, std::uint32_t, std::uint32_t, bool,
                    bool);

#endif

template <typename T>
std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet(llvm_state &s, const std::string &name, std::vector<expression> sys, std::uint32_t order,
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

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet_dbl(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>, std::uint32_t,
                   std::uint32_t, bool, bool);
HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet_ldbl(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>, std::uint32_t,
                    std::uint32_t, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet_f128(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>, std::uint32_t,
                    std::uint32_t, bool, bool);

#endif

template <typename T>
std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_jet(llvm_state &s, const std::string &name, std::vector<std::pair<expression, expression>> sys,
               std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
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

HEYOKA_DLL_PUBLIC std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step_dbl(llvm_state &, const std::string &, std::vector<expression>, double, std::uint32_t, bool,
                             bool);
HEYOKA_DLL_PUBLIC std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step_ldbl(llvm_state &, const std::string &, std::vector<expression>, long double, std::uint32_t,
                              bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step_f128(llvm_state &, const std::string &, std::vector<expression>, mppp::real128, std::uint32_t,
                              bool, bool);

#endif

template <typename T>
std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step(llvm_state &s, const std::string &name, std::vector<expression> sys, T tol,
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

HEYOKA_DLL_PUBLIC std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step_dbl(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>, double,
                             std::uint32_t, bool, bool);
HEYOKA_DLL_PUBLIC std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step_ldbl(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>,
                              long double, std::uint32_t, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step_f128(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>,
                              mppp::real128, std::uint32_t, bool, bool);

#endif

template <typename T>
std::tuple<std::vector<std::pair<expression, std::vector<std::uint32_t>>>, std::uint32_t>
taylor_add_adaptive_step(llvm_state &s, const std::string &name, std::vector<std::pair<expression, expression>> sys,
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

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step_dbl(llvm_state &, const std::string &, std::vector<expression>, std::uint32_t, std::uint32_t,
                           bool, bool);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step_ldbl(llvm_state &, const std::string &, std::vector<expression>, std::uint32_t, std::uint32_t,
                            bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step_f128(llvm_state &, const std::string &, std::vector<expression>, std::uint32_t, std::uint32_t,
                            bool, bool);

#endif

template <typename T>
std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step(llvm_state &s, const std::string &name, std::vector<expression> sys, std::uint32_t order,
                       std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_custom_step_dbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_custom_step_ldbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_custom_step_f128(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step_dbl(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>,
                           std::uint32_t, std::uint32_t, bool, bool);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step_ldbl(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>,
                            std::uint32_t, std::uint32_t, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step_f128(llvm_state &, const std::string &, std::vector<std::pair<expression, expression>>,
                            std::uint32_t, std::uint32_t, bool, bool);

#endif

template <typename T>
std::vector<std::pair<expression, std::vector<std::uint32_t>>>
taylor_add_custom_step(llvm_state &s, const std::string &name, std::vector<std::pair<expression, expression>> sys,
                       std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_custom_step_dbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_custom_step_ldbl(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_custom_step_f128(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

// Enum to represent the outcome of a Taylor integration
// stepping function.
enum class taylor_outcome {
    success,     // Integration step was successful, no time/step limits were reached.
    step_limit,  // Maximum number of steps reached.
    time_limit,  // Time limit reached.
    err_nf_state // Non-finite state detected at the end of the timestep.
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, taylor_outcome);

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(time);
IGOR_MAKE_NAMED_ARGUMENT(tol);
IGOR_MAKE_NAMED_ARGUMENT(high_accuracy);
IGOR_MAKE_NAMED_ARGUMENT(compact_mode);
IGOR_MAKE_NAMED_ARGUMENT(pars);

} // namespace kw

namespace detail
{

// Helper for parsing common options for the Taylor integrators.
template <typename T, typename... KwArgs>
inline auto taylor_adaptive_common_ops(KwArgs &&...kw_args)
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
    auto tol = [&p]() -> T {
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

        return std::numeric_limits<T>::epsilon();
    }();

    // Compact mode (defaults to false).
    auto compact_mode = [&p]() -> bool {
        if constexpr (p.has(kw::compact_mode)) {
            return std::forward<decltype(p(kw::compact_mode))>(p(kw::compact_mode));
        } else {
            return false;
        }
    }();

    // Vector of parameters (defaults to empty vector).
    auto pars = [&p]() -> std::vector<T> {
        if constexpr (p.has(kw::pars)) {
            return std::forward<decltype(p(kw::pars))>(p(kw::pars));
        } else {
            return {};
        }
    }();

    return std::tuple{high_accuracy, tol, compact_mode, std::move(pars)};
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
    // Dimension of the system.
    std::uint32_t m_dim;
    // Taylor decomposition.
    std::vector<std::pair<expression, std::vector<std::uint32_t>>> m_dc;
    // Taylor order.
    std::uint32_t m_order;
    // The stepper.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *);
    step_f_t m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;

    HEYOKA_DLL_LOCAL std::tuple<taylor_outcome, T> step_impl(T, bool);

    // Private implementation-detail constructor machinery.
    // NOTE: apparently on Windows we need to re-iterate
    // here that this is going to be dll-exported.
    template <typename U>
    HEYOKA_DLL_PUBLIC void finalise_ctor_impl(U, std::vector<T>, T, T, bool, bool, std::vector<T>);
    template <typename U, typename... KwArgs>
    void finalise_ctor(U sys, std::vector<T> state, KwArgs &&...kw_args)
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

            auto [high_accuracy, tol, compact_mode, pars]
                = taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

            finalise_ctor_impl(std::move(sys), std::move(state), time, tol, high_accuracy, compact_mode,
                               std::move(pars));
        }
    }

public:
    template <typename... KwArgs>
    explicit taylor_adaptive_impl(std::vector<expression> sys, std::vector<T> state, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(state), std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_impl(std::vector<std::pair<expression, expression>> sys, std::vector<T> state,
                                  KwArgs &&...kw_args)
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

    const std::vector<std::pair<expression, std::vector<std::uint32_t>>> &get_decomposition() const;

    std::uint32_t get_order() const;
    std::uint32_t get_dim() const;

    T get_time() const
    {
        return m_time;
    }
    void set_time(T t)
    {
        m_time = t;
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

    const std::vector<T> &get_pars() const
    {
        return m_pars;
    }
    const T *get_pars_data() const
    {
        return m_pars.data();
    }
    T *get_pars_data()
    {
        return m_pars.data();
    }

    const std::vector<T> &get_tc() const
    {
        return m_tc;
    }
    const T *get_tc_data() const
    {
        return m_tc.data();
    }
    T *get_tc_data()
    {
        return m_tc.data();
    }

    std::tuple<taylor_outcome, T> step(bool = false);
    std::tuple<taylor_outcome, T> step_backward(bool = false);
    std::tuple<taylor_outcome, T> step(T, bool = false);

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
    std::vector<T> m_state;
    // Times.
    std::vector<T> m_time;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Dimension of the system.
    std::uint32_t m_dim;
    // Taylor decomposition.
    std::vector<std::pair<expression, std::vector<std::uint32_t>>> m_dc;
    // Taylor order.
    std::uint32_t m_order;
    // The stepper.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *);
    step_f_t m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // Temporary vectors for use
    // in the timestepping functions.
    // These two are used as default values,
    // they must never be modified.
    std::vector<T> m_pinf, m_minf;
    // This is used as temporary storage in step_impl().
    std::vector<T> m_delta_ts;
    // The vectors used to store the results of the step
    // and propagate functions.
    std::vector<std::tuple<taylor_outcome, T>> m_step_res;
    std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> m_prop_res;
    // Temporary vectors used in the propagate_until() implementation.
    std::vector<std::size_t> m_ts_count;
    std::vector<T> m_min_abs_h, m_max_abs_h;
    std::vector<T> m_cur_max_delta_ts;
    std::vector<T> m_pfor_ts;

    HEYOKA_DLL_LOCAL const std::vector<std::tuple<taylor_outcome, T>> &step_impl(const std::vector<T> &, bool);

    // Private implementation-detail constructor machinery.
    template <typename U>
    HEYOKA_DLL_PUBLIC void finalise_ctor_impl(U, std::vector<T>, std::uint32_t, std::vector<T>, T, bool, bool,
                                              std::vector<T>);
    template <typename U, typename... KwArgs>
    void finalise_ctor(U sys, std::vector<T> state, std::uint32_t batch_size, KwArgs &&...kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of an adaptive batch Taylor integrator contain "
                          "unnamed arguments.");
        } else {
            // Initial times (defaults to a vector of zeroes).
            auto time = [&p, batch_size]() -> std::vector<T> {
                if constexpr (p.has(kw::time)) {
                    return std::forward<decltype(p(kw::time))>(p(kw::time));
                } else {
                    return std::vector<T>(static_cast<typename std::vector<T>::size_type>(batch_size), T(0));
                }
            }();

            auto [high_accuracy, tol, compact_mode, pars]
                = taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

            finalise_ctor_impl(std::move(sys), std::move(state), batch_size, std::move(time), tol, high_accuracy,
                               compact_mode, std::move(pars));
        }
    }

public:
    template <typename... KwArgs>
    explicit taylor_adaptive_batch_impl(std::vector<expression> sys, std::vector<T> state, std::uint32_t batch_size,
                                        KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(state), batch_size, std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch_impl(std::vector<std::pair<expression, expression>> sys, std::vector<T> state,
                                        std::uint32_t batch_size, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(std::move(sys), std::move(state), batch_size, std::forward<KwArgs>(kw_args)...);
    }

    taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &);
    taylor_adaptive_batch_impl(taylor_adaptive_batch_impl &&) noexcept;

    taylor_adaptive_batch_impl &operator=(const taylor_adaptive_batch_impl &);
    taylor_adaptive_batch_impl &operator=(taylor_adaptive_batch_impl &&) noexcept;

    ~taylor_adaptive_batch_impl();

    const llvm_state &get_llvm_state() const;

    const std::vector<std::pair<expression, std::vector<std::uint32_t>>> &get_decomposition() const;

    std::uint32_t get_batch_size() const;
    std::uint32_t get_order() const;
    std::uint32_t get_dim() const;

    const std::vector<T> &get_time() const
    {
        return m_time;
    }
    const T *get_time_data() const
    {
        return m_time.data();
    }
    T *get_time_data()
    {
        return m_time.data();
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

    const std::vector<T> &get_pars() const
    {
        return m_pars;
    }
    const T *get_pars_data() const
    {
        return m_pars.data();
    }
    T *get_pars_data()
    {
        return m_pars.data();
    }

    const std::vector<T> &get_tc() const
    {
        return m_tc;
    }
    const T *get_tc_data() const
    {
        return m_tc.data();
    }
    T *get_tc_data()
    {
        return m_tc.data();
    }

    const std::vector<std::tuple<taylor_outcome, T>> &step(bool = false);
    const std::vector<std::tuple<taylor_outcome, T>> &step_backward(bool = false);
    const std::vector<std::tuple<taylor_outcome, T>> &step(const std::vector<T> &, bool = false);

    const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &propagate_for(const std::vector<T> &,
                                                                                    std::size_t = 0);
    const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &propagate_until(const std::vector<T> &,
                                                                                      std::size_t = 0);
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

namespace detail
{

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const taylor_adaptive_impl<T> &)
{
    static_assert(always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_impl<double> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_impl<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_impl<mppp::real128> &);

#endif

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch_impl<T> &)
{
    static_assert(always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch_impl<double> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch_impl<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch_impl<mppp::real128> &);

#endif

} // namespace detail

} // namespace heyoka

#endif
