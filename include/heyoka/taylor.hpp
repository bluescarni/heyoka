// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TAYLOR_HPP
#define HEYOKA_TAYLOR_HPP

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/continuous_output.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/var_ode_sys.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: these are various utilities useful when dealing in a generic
// fashion with numbers/params in Taylor functions.

// Helper to detect if T is a number or a param.
template <typename T>
using is_num_param = std::disjunction<std::is_same<T, number>, std::is_same<T, param>>;

template <typename T>
inline constexpr bool is_num_param_v = is_num_param<T>::value;

HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam(llvm_state &, llvm::Type *, const number &, llvm::Value *,
                                                       std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_codegen_numparam(llvm_state &, llvm::Type *, const param &, llvm::Value *,
                                                       std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &, llvm::Type *, const number &, llvm::Value *,
                                                              llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &, llvm::Type *, const param &, llvm::Value *,
                                                              llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &, std::uint32_t, std::uint32_t,
                                                 std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_load_diff(llvm_state &, llvm::Type *, llvm::Value *, std::uint32_t,
                                                  llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC void taylor_c_store_diff(llvm_state &, llvm::Type *, llvm::Value *, std::uint32_t, llvm::Value *,
                                           llvm::Value *, llvm::Value *);

std::tuple<taylor_dc_t, std::array<std::size_t, 2>, std::vector<llvm_state>>
taylor_add_adaptive_step_with_events(llvm_state &, llvm::Type *, const std::string &,
                                     const std::vector<std::pair<expression, expression>> &, std::uint32_t, bool,
                                     const std::vector<expression> &, bool, bool, std::uint32_t);

std::tuple<taylor_dc_t, std::array<std::size_t, 2>, std::vector<llvm_state>>
taylor_add_adaptive_step(llvm_state &, llvm::Type *, llvm::Type *, const std::string &,
                         const std::vector<std::pair<expression, expression>> &, std::uint32_t, bool, bool, bool,
                         std::uint32_t);

llvm::Value *taylor_c_make_sv_funcs_arr(llvm_state &, const std::vector<std::uint32_t> &);

std::variant<std::pair<std::array<std::size_t, 2>, std::vector<llvm_state>>, std::vector<llvm::Value *>>
taylor_compute_jet(llvm_state &, llvm::Type *, llvm::Value *, llvm::Value *, llvm::Value *, llvm::Value *,
                   const taylor_dc_t &, const std::vector<std::uint32_t> &, std::uint32_t, std::uint32_t, std::uint32_t,
                   std::uint32_t, bool, bool, bool);

std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args(llvm::LLVMContext &, llvm::Type *, const std::string &, std::uint32_t, std::uint32_t,
                             const std::vector<std::variant<variable, number, param>> &, std::uint32_t = 0);

// Add a function for computing the dense output
// via polynomial evaluation.
void taylor_add_d_out_function(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, std::uint32_t, bool,
                               bool = true);

template <typename TEvent, typename NTEvent>
void validate_ode_sys(const std::vector<std::pair<expression, expression>> &, const std::vector<TEvent> &,
                      const std::vector<NTEvent> &);
void validate_ode_sys(const std::vector<std::pair<expression, expression>> &);

} // namespace detail

HEYOKA_DLL_PUBLIC std::pair<taylor_dc_t, std::vector<std::uint32_t>>
taylor_decompose_sys(const std::vector<std::pair<expression, expression>> &, const std::vector<expression> &);

// Enum to represent the outcome of a stepping/propagate function.
enum class taylor_outcome : std::int64_t {
    // NOTE: we make these enums start at -2**32 - 1,
    // so that we have 2**32 values in the [-2**32, -1]
    // range to use for signalling stopping terminal events.
    // NOTE: the time_limit outcome signals both a clamped
    // timestep and a propagate_*() function that successfully
    // finished. This can be confusing, perhaps we can consider
    // in the future having different outcomes.
    success = -4294967296ll - 1,      // Integration step was successful.
    step_limit = -4294967296ll - 2,   // Maximum number of steps reached.
    time_limit = -4294967296ll - 3,   // Time limit reached.
    err_nf_state = -4294967296ll - 4, // Non-finite state detected at the end of the timestep.
    cb_stop = -4294967296ll - 5       // Propagation stopped by callback.
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, taylor_outcome);

HEYOKA_END_NAMESPACE

// fmt formatter for taylor_outcome, implemented on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::taylor_outcome> : fmt::ostream_formatter {
};

} // namespace fmt

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper for parsing common options when constructing Taylor integrators.
template <typename T, typename... KwArgs>
auto taylor_adaptive_common_ops(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    // High accuracy mode (defaults to false).
    auto high_accuracy = [&p]() -> bool {
        if constexpr (p.has(kw::high_accuracy)) {
            return p(kw::high_accuracy);
        } else {
            return false;
        }
    }();

    // tol (defaults to undefined). Zero tolerance is considered
    // the same as undefined.
    auto tol = [&p]() -> std::optional<T> {
        if constexpr (p.has(kw::tol)) {
            auto retval = static_cast<T>(p(kw::tol));
            if (retval != 0) {
                // NOTE: this covers the NaN case as well.
                return retval;
            }
            // NOTE: zero tolerance will be interpreted
            // as undefined by falling through
            // the code below.
        }

        return {};
    }();

    // Compact mode (defaults to false, except for real where
    // it defaults to true).
    auto compact_mode = [&p]() -> bool {
        if constexpr (p.has(kw::compact_mode)) {
            return p(kw::compact_mode);
        } else {
#if defined(HEYOKA_HAVE_REAL)
            return std::is_same_v<T, mppp::real>;
#else
            return false;

#endif
        }
    }();

    // Vector of parameters (defaults to empty vector).
    auto pars = [&p]() -> std::vector<T> {
        if constexpr (p.has(kw::pars)) {
            return p(kw::pars);
        } else {
            return {};
        }
    }();

    // Parallel mode (defaults to false).
    auto parallel_mode = [&p]() -> bool {
        if constexpr (p.has(kw::parallel_mode)) {
            return p(kw::parallel_mode);
        } else {
            return false;
        }
    }();

    return std::tuple{high_accuracy, std::move(tol), compact_mode, std::move(pars), parallel_mode};
}

// Small helper to construct a default value for the max_delta_t
// keyword argument.
template <typename T>
HEYOKA_DLL_PUBLIC T taylor_default_max_delta_t();

// Concept to detect if R is an input range from whose reference type
// instances of T can be constructed.
template <typename R, typename T>
concept input_rangeT = std::ranges::input_range<R> && std::constructible_from<T, std::ranges::range_reference_t<R>>;

// Logic for parsing a step callback argument in a propagate_*() function.
// The default return value is an empty callback. If an object that can be used to construct
// a Callback is provided, then use it. Otherwise, if a range of objects
// that can be used to construct a Callback is provided, use them to assemble
// a CallbackSet. Otherwise, the call is malformed.
// NOTE: in any case, we end up creating a new object either by copying
// or moving the input argument(s).
template <typename Callback, typename CallbackSet, typename Parser>
Callback parse_propagate_cb(const Parser &p)
{
    if constexpr (Parser::has(kw::callback)) {
        using cb_arg_t = decltype(p(kw::callback));

        if constexpr (std::convertible_to<cb_arg_t, Callback>) {
            return p(kw::callback);
        } else if constexpr (input_rangeT<cb_arg_t, Callback>) {
            std::vector<Callback> cb_vec;
            for (auto &&cb : p(kw::callback)) {
                cb_vec.emplace_back(std::forward<decltype(cb)>(cb));
            }

            return CallbackSet(std::move(cb_vec));
        } else {
            // LCOV_EXCL_START
            static_assert(detail::always_false_v<CallbackSet>,
                          "A 'callback' keyword argument of an invalid type was passed to a propagate_*() function.");

            throw;
            // LCOV_EXCL_STOP
        }
    } else {
        return {};
    }
}

// Parser for the common kwargs options for the propagate_*() functions.
template <typename T, bool Grid, typename... KwArgs>
auto taylor_propagate_common_ops(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    if constexpr (p.has_unnamed_arguments()) {
        // LCOV_EXCL_START
        static_assert(detail::always_false_v<KwArgs...>,
                      "The variadic arguments to a propagate_*() function in an "
                      "adaptive Taylor integrator cannot contain unnamed arguments.");
        throw;
        // LCOV_EXCL_STOP
    } else {
        // Max number of steps (defaults to zero).
        auto max_steps = [&p]() -> std::size_t {
            if constexpr (p.has(kw::max_steps)) {
                static_assert(std::integral<std::remove_cvref_t<decltype(p(kw::max_steps))>>,
                              "The 'max_steps' keyword argument to a propagate_*() function must be of integral type.");

                return boost::numeric_cast<std::size_t>(p(kw::max_steps));
            } else {
                return 0;
            }
        }();

        // Max delta_t (defaults to positive infinity).
        auto max_delta_t = [&p]() -> T {
            if constexpr (p.has(kw::max_delta_t)) {
                static_assert(
                    std::convertible_to<decltype(p(kw::max_delta_t)), T>,
                    "A 'max_delta_t' keyword argument of an invalid type was passed to a propagate_*() function.");

                return p(kw::max_delta_t);
            } else {
                return taylor_default_max_delta_t<T>();
            }
        }();

        // Parse the callback argument.
        auto cb = parse_propagate_cb<step_callback<T>, step_callback_set<T>>(p);

        // Write the Taylor coefficients (defaults to false).
        // NOTE: this won't be used in propagate_grid().
        auto write_tc = [&p]() -> bool {
            if constexpr (p.has(kw::write_tc)) {
                static_assert(
                    std::convertible_to<decltype(p(kw::write_tc)), bool>,
                    "A 'write_tc' keyword argument of an invalid type was passed to a propagate_*() function.");

                return p(kw::write_tc);
            } else {
                return false;
            }
        }();

        if constexpr (Grid) {
            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb));
        } else {
            // Continuous output (defaults to false).
            auto with_c_out = [&p]() -> bool {
                if constexpr (p.has(kw::c_output)) {
                    static_assert(
                        std::convertible_to<decltype(p(kw::c_output)), bool>,
                        "A 'c_output' keyword argument of an invalid type was passed to a propagate_*() function.");

                    return p(kw::c_output);
                } else {
                    return false;
                }
            }();

            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb), write_tc, with_c_out);
        }
    }
}

// Base class to contain data specific to integrators of type
// T. By default this is just an empty class.
template <typename T, typename Derived>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive_base
{
    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

#if defined(HEYOKA_HAVE_REAL)

template <typename Derived>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive_base<mppp::real, Derived>
{
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes)
    mpfr_prec_t m_prec = 0;

    void data_prec_check() const;

public:
    [[nodiscard]] mpfr_prec_t get_prec() const;
};

#endif

template <typename T>
void setup_variational_ics_varpar(std::vector<T> &, const var_ode_sys &, std::uint32_t);

template <typename T>
void setup_variational_ics_t0(const llvm_state &, std::vector<T> &, const std::vector<T> &, const T *,
                              const var_ode_sys &, std::uint32_t, bool, bool);

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive : public detail::taylor_adaptive_base<T, taylor_adaptive<T>>
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");
    friend class HEYOKA_DLL_PUBLIC_INLINE_CLASS detail::taylor_adaptive_base<T, taylor_adaptive<T>>;
    using base_t = detail::taylor_adaptive_base<T, taylor_adaptive<T>>;

public:
    using value_type = T;

    using nt_event_t = nt_event<T>;
    using t_event_t = t_event<T>;

private:
    // Struct storing the integrator data.
    struct HEYOKA_DLL_PUBLIC i_data;

    // Struct implementing the data/logic for event detection.
    struct HEYOKA_DLL_PUBLIC ed_data;

    // Pimpls.
    std::unique_ptr<i_data> m_i_data;
    std::unique_ptr<ed_data> m_ed_data;

    // Serialization.
    template <typename Archive>
    HEYOKA_DLL_LOCAL void save_impl(Archive &, unsigned) const;
    template <typename Archive>
    HEYOKA_DLL_LOCAL void load_impl(Archive &, unsigned);

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    HEYOKA_DLL_LOCAL std::tuple<taylor_outcome, T> step_impl(T, bool);

    // Private implementation-detail constructor machinery.
    using sys_t = std::variant<std::vector<std::pair<expression, expression>>, var_ode_sys>;
    void finalise_ctor_impl(sys_t, std::vector<T>, std::optional<T>, std::optional<T>, bool, bool, std::vector<T>,
                            std::vector<t_event_t>, std::vector<nt_event_t>, bool, std::optional<long long>);
    template <typename... KwArgs>
    void finalise_ctor(sys_t sys, std::vector<T> state, const KwArgs &...kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of an adaptive Taylor integrator contain "
                          "unnamed arguments.");
        } else {
            // Initial time (defaults to undefined).
            auto tm = [&p]() -> std::optional<T> {
                if constexpr (p.has(kw::time)) {
                    return p(kw::time);
                } else {
                    return {};
                }
            }();

            auto [high_accuracy, tol, compact_mode, pars, parallel_mode]
                = detail::taylor_adaptive_common_ops<T>(kw_args...);

            // Extract the terminal events, if any.
            auto tes = [&p]() -> std::vector<t_event_t> {
                if constexpr (p.has(kw::t_events)) {
                    return p(kw::t_events);
                } else {
                    return {};
                }
            }();

            // Extract the non-terminal events, if any.
            auto ntes = [&p]() -> std::vector<nt_event_t> {
                if constexpr (p.has(kw::nt_events)) {
                    return p(kw::nt_events);
                } else {
                    return {};
                }
            }();

            // Fetch the precision, if provided. Zero precision
            // is considered the same as undefined.
            auto prec = [&p]() -> std::optional<long long> {
                if constexpr (p.has(kw::prec)) {
                    auto ret = static_cast<long long>(p(kw::prec));
                    if (ret != 0) {
                        return ret;
                    }
                }

                return {};
            }();

            finalise_ctor_impl(std::move(sys), std::move(state), std::move(tm), std::move(tol), high_accuracy,
                               compact_mode, std::move(pars), std::move(tes), std::move(ntes), parallel_mode,
                               std::move(prec));
        }
    }

    // NOTE: we need to go through a private non-template constructor
    // in order to avoid having to provide definitions for the pimpled data.
    struct private_ctor_t {
    };
    explicit taylor_adaptive(private_ctor_t, llvm_state);

    HEYOKA_DLL_LOCAL void check_variational(const char *) const;
    HEYOKA_DLL_LOCAL void assign_stepper(bool);

    // Input type for Taylor map computation.
    using tm_input_t = mdspan<const T, dextents<std::uint32_t, 1>>;
    const std::vector<T> &eval_taylor_map_impl(tm_input_t);

public:
    taylor_adaptive();

    template <typename... KwArgs>
    explicit taylor_adaptive(std::vector<std::pair<expression, expression>> sys, std::vector<T> state,
                             const KwArgs &...kw_args)
        : taylor_adaptive(private_ctor_t{}, llvm_state(kw_args...))
    {
        finalise_ctor(std::move(sys), std::move(state), kw_args...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive(std::vector<std::pair<expression, expression>> sys, std::initializer_list<T> state,
                             const KwArgs &...kw_args)
        : taylor_adaptive(std::move(sys), std::vector<T>(state), kw_args...)
    {
    }
    template <typename... KwArgs>
    explicit taylor_adaptive(var_ode_sys sys, std::vector<T> state, const KwArgs &...kw_args)
        : taylor_adaptive(private_ctor_t{}, llvm_state(kw_args...))
    {
        finalise_ctor(std::move(sys), std::move(state), kw_args...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive(var_ode_sys sys, std::initializer_list<T> state, const KwArgs &...kw_args)
        : taylor_adaptive(std::move(sys), std::vector<T>(state), kw_args...)
    {
    }

    taylor_adaptive(const taylor_adaptive &);
    taylor_adaptive(taylor_adaptive &&) noexcept;

    taylor_adaptive &operator=(const taylor_adaptive &);
    taylor_adaptive &operator=(taylor_adaptive &&) noexcept;

    ~taylor_adaptive();

    [[nodiscard]] const std::variant<llvm_state, llvm_multi_state> &get_llvm_state() const;

    [[nodiscard]] const taylor_dc_t &get_decomposition() const;

    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] T get_tol() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] bool get_compact_mode() const;
    [[nodiscard]] std::uint32_t get_dim() const;
    [[nodiscard]] std::uint32_t get_n_orig_sv() const noexcept;

    [[nodiscard]] T get_time() const;
    void set_time(T);

    // Time set/get in double-length format.
    [[nodiscard]] std::pair<T, T> get_dtime() const;
    void set_dtime(T, T);

    [[nodiscard]] const std::vector<T> &get_state() const;
    [[nodiscard]] const T *get_state_data() const;
    [[nodiscard]] T *get_state_data();
    [[nodiscard]] std::ranges::subrange<typename std::vector<T>::iterator> get_state_range();

    [[nodiscard]] const std::vector<T> &get_pars() const;
    [[nodiscard]] const T *get_pars_data() const;
    [[nodiscard]] T *get_pars_data();
    [[nodiscard]] std::ranges::subrange<typename std::vector<T>::iterator> get_pars_range();

    [[nodiscard]] const std::vector<T> &get_tc() const;

    [[nodiscard]] T get_last_h() const;

    [[nodiscard]] const std::vector<T> &get_d_output() const;
    const std::vector<T> &update_d_output(T, bool = false);

    [[nodiscard]] bool with_events() const;
    void reset_cooldowns();
    [[nodiscard]] const std::vector<t_event_t> &get_t_events() const;
    [[nodiscard]] const std::vector<std::optional<std::pair<T, T>>> &get_te_cooldowns() const;
    [[nodiscard]] const std::vector<nt_event_t> &get_nt_events() const;

    [[nodiscard]] const std::vector<std::pair<expression, expression>> &get_sys() const noexcept;

    std::tuple<taylor_outcome, T> step(bool = false);
    std::tuple<taylor_outcome, T> step_backward(bool = false);
    std::tuple<taylor_outcome, T> step(T, bool = false);

    [[nodiscard]] bool is_variational() const noexcept;
    [[nodiscard]] const std::vector<expression> &get_vargs() const;
    [[nodiscard]] std::uint32_t get_vorder() const;

    template <typename R>
        requires std::ranges::contiguous_range<R>
                 && std::same_as<T, std::remove_cvref_t<std::ranges::range_reference_t<R>>>
                 && std::integral<std::ranges::range_size_t<R>>
    const std::vector<T> &eval_taylor_map(R &&r)
    {
        // Turn r into a span.
        tm_input_t s(std::ranges::data(r), boost::numeric_cast<std::uint32_t>(std::ranges::size(r)));

        return eval_taylor_map_impl(s);
    }
    const std::vector<T> &eval_taylor_map(std::initializer_list<T>);
    [[nodiscard]] const std::vector<T> &get_tstate() const;

    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> get_vslice(std::uint32_t) const;
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> get_vslice(std::uint32_t, std::uint32_t) const;
    [[nodiscard]] const dtens::sv_idx_t &get_mindex(std::uint32_t) const;

private:
    // Implementations of the propagate_*() functions.
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
    propagate_until_impl(detail::dfloat<T>, std::size_t, T, step_callback<T>, bool, bool);
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
    propagate_for_impl(T, std::size_t, T, step_callback<T>, bool, bool);
    std::tuple<taylor_outcome, T, T, std::size_t, step_callback<T>, std::vector<T>>
        propagate_grid_impl(std::vector<T>, std::size_t, T, step_callback<T>);

public:
    // NOTE: return values:
    // - outcome,
    // - min abs(timestep),
    // - max abs(timestep),
    // - total number of nonzero steps
    //   successfully undertaken,
    // - continuous output, if requested (only for propagate_for/until()),
    // - step callback, if provided,
    // - grid of state vectors (only for propagate_grid()).
    // NOTE: the min/max timesteps are well-defined
    // only if at least 1-2 steps were taken successfully.
    // NOTE: the propagate_*() functions are not guaranteed to bring
    // the integrator time *exactly* to the requested final time. This
    // ultimately stems from the fact that in floating-point arithmetics
    // in general a + (b - a) != b, and this happens regardless of the
    // use of a double-length time representation. This occurrence however
    // seems to be pretty rare in practice, so for the time being we leave
    // this as it is and just document the corner-case behaviour. Perhaps
    // in the future we can offer a stronger guarantee, which however will
    // result in a more complicated logic.
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
    propagate_until(T t, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops<T, false>(kw_args...);

        return propagate_until_impl(detail::dfloat<T>(std::move(t)), max_steps, std::move(max_delta_t), std::move(cb),
                                    write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
    propagate_for(T delta_t, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops<T, false>(kw_args...);

        return propagate_for_impl(std::move(delta_t), max_steps, std::move(max_delta_t), std::move(cb), write_tc,
                                  with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, step_callback<T>, std::vector<T>>
    propagate_grid(std::vector<T> grid, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_t, cb] = detail::taylor_propagate_common_ops<T, true>(kw_args...);

        return propagate_grid_impl(std::move(grid), max_steps, std::move(max_delta_t), std::move(cb));
    }
};

// Prevent implicit instantiations.
// NOLINTBEGIN
#define HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST(F)                                                                          \
    extern template class detail::taylor_adaptive_base<F, taylor_adaptive<F>>;                                         \
    extern template class taylor_adaptive<F>;
// NOLINTEND

HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST(float)
HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST(double)
HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_TAYLOR_ADAPTIVE_EXTERN_INST

namespace detail
{

// Parser for the common kwargs options for the propagate_*() functions
// for the batch integrator.
template <typename T, bool Grid, bool ForceScalarMaxDeltaT, typename... KwArgs>
auto taylor_propagate_common_ops_batch(std::uint32_t batch_size, const KwArgs &...kw_args)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    igor::parser p{kw_args...};

    if constexpr (p.has_unnamed_arguments()) {
        static_assert(always_false_v<KwArgs...>, "The variadic arguments to a propagate_*() function in an "
                                                 "adaptive Taylor integrator in batch mode contain unnamed arguments.");
        throw;
    } else {
        // Max number of steps (defaults to zero).
        auto max_steps = [&p]() -> std::size_t {
            if constexpr (p.has(kw::max_steps)) {
                static_assert(std::integral<std::remove_cvref_t<decltype(p(kw::max_steps))>>,
                              "The 'max_steps' keyword argument to a propagate_*() function must be of integral type.");

                return boost::numeric_cast<std::size_t>(p(kw::max_steps));
            } else {
                return 0;
            }
        }();

        // Max delta_t (defaults to empty vector).
        // NOTE: we want an explicit copy here because
        // in the implementations of the propagate_*() functions
        // we keep on checking on max_delta_t before invoking
        // the single step function. Hence, we want to avoid
        // any risk of aliasing.
        auto max_delta_t = [&]() -> std::vector<T> {
            if constexpr (p.has(kw::max_delta_t)) {
                using type = decltype(p(kw::max_delta_t));

                if constexpr (input_rangeT<type, T>) {
                    if constexpr (ForceScalarMaxDeltaT) {
                        // LCOV_EXCL_START
                        static_assert(always_false_v<T>,
                                      "In ensemble integrations, max_delta_t must always be passed as a scalar.");

                        throw;
                        // LCOV_EXCL_STOP
                    } else {
                        std::vector<T> retval;
                        for (auto &&x : p(kw::max_delta_t)) {
                            retval.emplace_back(std::forward<decltype(x)>(x));
                        }

                        return retval;
                    }
                } else {
                    // Interpret as a scalar to be splatted.
                    static_assert(
                        std::convertible_to<type, T>,
                        "A 'max_delta_t' keyword argument of an invalid type was passed to a propagate_*() function.");

                    return std::vector<T>(boost::numeric_cast<typename std::vector<T>::size_type>(batch_size),
                                          p(kw::max_delta_t));
                }
            } else {
                return {};
            }
        }();

        // Parse the callback argument.
        auto cb = parse_propagate_cb<step_callback_batch<T>, step_callback_batch_set<T>>(p);

        // Write the Taylor coefficients (defaults to false).
        // NOTE: this won't be used in propagate_grid().
        auto write_tc = [&p]() -> bool {
            if constexpr (p.has(kw::write_tc)) {
                static_assert(
                    std::convertible_to<decltype(p(kw::write_tc)), bool>,
                    "A 'write_tc' keyword argument of an invalid type was passed to a propagate_*() function.");

                return p(kw::write_tc);
            } else {
                return false;
            }
        }();

        if constexpr (Grid) {
            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb));
        } else {
            // Continuous output (defaults to false).
            auto with_c_out = [&p]() -> bool {
                if constexpr (p.has(kw::c_output)) {
                    static_assert(
                        std::convertible_to<decltype(p(kw::c_output)), bool>,
                        "A 'c_output' keyword argument of an invalid type was passed to a propagate_*() function.");

                    return p(kw::c_output);
                } else {
                    return false;
                }
            }();

            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb), write_tc, with_c_out);
        }
    }
}

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive_batch
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");

public:
    using value_type = T;

    using nt_event_t = nt_event_batch<T>;
    using t_event_t = t_event_batch<T>;

private:
    // Struct storing the integrator data.
    struct HEYOKA_DLL_PUBLIC i_data;

    // Struct implementing the data/logic for event detection.
    struct HEYOKA_DLL_PUBLIC ed_data;

    // Pimpls.
    std::unique_ptr<i_data> m_i_data;
    std::unique_ptr<ed_data> m_ed_data;

    // Serialization.
    template <typename Archive>
    HEYOKA_DLL_LOCAL void save_impl(Archive &, unsigned) const;
    template <typename Archive>
    HEYOKA_DLL_LOCAL void load_impl(Archive &, unsigned);

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    HEYOKA_DLL_LOCAL void step_impl(const std::vector<T> &, bool);

    // Private implementation-detail constructor machinery.
    using sys_t = std::variant<std::vector<std::pair<expression, expression>>, var_ode_sys>;
    void finalise_ctor_impl(sys_t, std::vector<T>, std::uint32_t, std::vector<T>, std::optional<T>, bool, bool,
                            std::vector<T>, std::vector<t_event_t>, std::vector<nt_event_t>, bool);
    template <typename... KwArgs>
    void finalise_ctor(sys_t sys, std::vector<T> state, std::uint32_t batch_size, const KwArgs &...kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of an adaptive batch Taylor integrator contain "
                          "unnamed arguments.");
        } else {
            // Initial times (defaults to a vector of zeroes).
            auto tm = [&p, batch_size]() -> std::vector<T> {
                if constexpr (p.has(kw::time)) {
                    // NOTE: silence clang warning.
                    (void)batch_size;
                    return p(kw::time);
                } else {
                    return std::vector<T>(static_cast<typename std::vector<T>::size_type>(batch_size), T(0));
                }
            }();

            auto [high_accuracy, tol, compact_mode, pars, parallel_mode]
                = detail::taylor_adaptive_common_ops<T>(kw_args...);

            // Extract the terminal events, if any.
            auto tes = [&p]() -> std::vector<t_event_t> {
                if constexpr (p.has(kw::t_events)) {
                    return p(kw::t_events);
                } else {
                    return {};
                }
            }();

            // Extract the non-terminal events, if any.
            auto ntes = [&p]() -> std::vector<nt_event_t> {
                if constexpr (p.has(kw::nt_events)) {
                    return p(kw::nt_events);
                } else {
                    return {};
                }
            }();

            finalise_ctor_impl(std::move(sys), std::move(state), batch_size, std::move(tm), std::move(tol),
                               high_accuracy, compact_mode, std::move(pars), std::move(tes), std::move(ntes),
                               parallel_mode);
        }
    }

    // NOTE: we need to go through a private non-template constructor
    // in order to avoid having to provide definitions for the pimpled data.
    struct private_ctor_t {
    };
    explicit taylor_adaptive_batch(private_ctor_t, llvm_state);

    HEYOKA_DLL_LOCAL void check_variational(const char *) const;
    HEYOKA_DLL_LOCAL void assign_stepper(bool);

    // Input type for Taylor map computation.
    using tm_input_t = mdspan<const T, dextents<std::uint32_t, 1>>;
    const std::vector<T> &eval_taylor_map_impl(tm_input_t);

public:
    taylor_adaptive_batch();

    template <typename... KwArgs>
    explicit taylor_adaptive_batch(std::vector<std::pair<expression, expression>> sys, std::vector<T> state,
                                   std::uint32_t batch_size, const KwArgs &...kw_args)
        : taylor_adaptive_batch(private_ctor_t{}, llvm_state(kw_args...))
    {
        finalise_ctor(std::move(sys), std::move(state), batch_size, kw_args...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch(std::vector<std::pair<expression, expression>> sys, std::initializer_list<T> state,
                                   std::uint32_t batch_size, const KwArgs &...kw_args)
        : taylor_adaptive_batch(std::move(sys), std::vector<T>(state), batch_size, kw_args...)
    {
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch(var_ode_sys sys, std::vector<T> state, std::uint32_t batch_size,
                                   const KwArgs &...kw_args)
        : taylor_adaptive_batch(private_ctor_t{}, llvm_state(kw_args...))
    {
        finalise_ctor(std::move(sys), std::move(state), batch_size, kw_args...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch(var_ode_sys sys, std::initializer_list<T> state, std::uint32_t batch_size,
                                   const KwArgs &...kw_args)
        : taylor_adaptive_batch(std::move(sys), std::vector<T>(state), batch_size, kw_args...)
    {
    }

    taylor_adaptive_batch(const taylor_adaptive_batch &);
    taylor_adaptive_batch(taylor_adaptive_batch &&) noexcept;

    taylor_adaptive_batch &operator=(const taylor_adaptive_batch &);
    taylor_adaptive_batch &operator=(taylor_adaptive_batch &&) noexcept;

    ~taylor_adaptive_batch();

    [[nodiscard]] const std::variant<llvm_state, llvm_multi_state> &get_llvm_state() const;

    [[nodiscard]] const taylor_dc_t &get_decomposition() const;

    [[nodiscard]] std::uint32_t get_batch_size() const;
    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] T get_tol() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] bool get_compact_mode() const;
    [[nodiscard]] std::uint32_t get_dim() const;
    [[nodiscard]] std::uint32_t get_n_orig_sv() const noexcept;

    [[nodiscard]] const std::vector<T> &get_time() const;
    [[nodiscard]] const T *get_time_data() const;
    void set_time(const std::vector<T> &);
    void set_time(T);

    // Time set/get in double-length format.
    [[nodiscard]] std::pair<const std::vector<T> &, const std::vector<T> &> get_dtime() const;
    [[nodiscard]] std::pair<const T *, const T *> get_dtime_data() const;
    void set_dtime(const std::vector<T> &, const std::vector<T> &);
    void set_dtime(T, T);

    [[nodiscard]] const std::vector<T> &get_state() const;
    [[nodiscard]] const T *get_state_data() const;
    [[nodiscard]] T *get_state_data();
    [[nodiscard]] std::ranges::subrange<typename std::vector<T>::iterator> get_state_range();

    [[nodiscard]] const std::vector<T> &get_pars() const;
    [[nodiscard]] const T *get_pars_data() const;
    [[nodiscard]] T *get_pars_data();
    [[nodiscard]] std::ranges::subrange<typename std::vector<T>::iterator> get_pars_range();

    [[nodiscard]] const std::vector<T> &get_tc() const;

    [[nodiscard]] const std::vector<T> &get_last_h() const;

    [[nodiscard]] const std::vector<T> &get_d_output() const;
    const std::vector<T> &update_d_output(const std::vector<T> &, bool = false);
    const std::vector<T> &update_d_output(T, bool = false);

    [[nodiscard]] bool with_events() const;
    void reset_cooldowns();
    void reset_cooldowns(std::uint32_t);
    [[nodiscard]] const std::vector<t_event_t> &get_t_events() const;
    [[nodiscard]] const std::vector<std::vector<std::optional<std::pair<T, T>>>> &get_te_cooldowns() const;
    [[nodiscard]] const std::vector<nt_event_t> &get_nt_events() const;

    [[nodiscard]] const std::vector<std::pair<expression, expression>> &get_sys() const noexcept;

    void step(bool = false);
    void step_backward(bool = false);
    void step(const std::vector<T> &, bool = false);
    [[nodiscard]] const std::vector<std::tuple<taylor_outcome, T>> &get_step_res() const;

    [[nodiscard]] bool is_variational() const noexcept;
    [[nodiscard]] const std::vector<expression> &get_vargs() const;
    [[nodiscard]] std::uint32_t get_vorder() const;

    template <typename R>
        requires std::ranges::contiguous_range<R>
                 && std::same_as<T, std::remove_cvref_t<std::ranges::range_reference_t<R>>>
                 && std::integral<std::ranges::range_size_t<R>>
    const std::vector<T> &eval_taylor_map(R &&r)
    {
        // Turn r into a span.
        tm_input_t s(std::ranges::data(r), boost::numeric_cast<std::uint32_t>(std::ranges::size(r)));

        return eval_taylor_map_impl(s);
    }
    const std::vector<T> &eval_taylor_map(std::initializer_list<T>);
    [[nodiscard]] const std::vector<T> &get_tstate() const;

    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> get_vslice(std::uint32_t) const;
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> get_vslice(std::uint32_t, std::uint32_t) const;
    [[nodiscard]] const dtens::sv_idx_t &get_mindex(std::uint32_t) const;

private:
    // Implementations of the propagate_*() functions.

    // NOTE: the argument to the propagate_until_impl() function can be one of these:
    // - single scalar value,
    // - vector of values,
    // - vector of double-length values.
    // NOTE: the third case can occur only if propagate_until() is being called
    // from propagate_for().
    using puntil_arg_t = std::variant<T, std::reference_wrapper<const std::vector<T>>,
                                      std::reference_wrapper<const std::vector<detail::dfloat<T>>>>;
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_until_impl(const puntil_arg_t &, std::size_t, const std::vector<T> &, step_callback_batch<T>, bool, bool);

    // NOTE: the argument to the propagate_for_impl() function can be one of these:
    // - single scalar value,
    // - vector of values.
    using pfor_arg_t = std::variant<T, std::reference_wrapper<const std::vector<T>>>;
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_for_impl(const pfor_arg_t &, std::size_t, const std::vector<T> &, step_callback_batch<T>, bool, bool);

    std::tuple<step_callback_batch<T>, std::vector<T>>
    propagate_grid_impl(const std::vector<T> &, std::size_t, const std::vector<T> &, step_callback_batch<T>);

public:
    // NOTE: in propagate_for/until(), we can take 'ts' as const reference because it is always
    // only and immediately used to set up the internal m_pfor_ts member (which is not visible
    // from outside). Hence, even if 'ts' aliases some public integrator data, it does not matter.
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_until(const std::vector<T> &ts, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(get_batch_size(), kw_args...);

        return propagate_until_impl(ts, max_steps, max_delta_ts, std::move(cb), write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_until(T t, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(get_batch_size(), kw_args...);

        return propagate_until_impl(std::move(t), max_steps, max_delta_ts, std::move(cb), write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_for(const std::vector<T> &delta_ts, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(get_batch_size(), kw_args...);

        return propagate_for_impl(delta_ts, max_steps, max_delta_ts, std::move(cb), write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_for(T delta_t, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(get_batch_size(), kw_args...);

        return propagate_for_impl(std::move(delta_t), max_steps, max_delta_ts, std::move(cb), write_tc, with_c_out);
    }
    // NOTE: grid is taken by copy because in the implementation loop we keep on reading from it.
    // Hence, we need to avoid any aliasing issue with other public integrator data.
    template <typename... KwArgs>
    std::tuple<step_callback_batch<T>, std::vector<T>> propagate_grid(std::vector<T> grid, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb]
            = detail::taylor_propagate_common_ops_batch<T, true, false>(get_batch_size(), kw_args...);

        return propagate_grid_impl(grid, max_steps, max_delta_ts, std::move(cb));
    }
    [[nodiscard]] const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &get_propagate_res() const;
};

// Prevent implicit instantiations.
#define HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST(F) extern template class taylor_adaptive_batch<F>;

HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST(float)
HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST(double)
HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_TAYLOR_ADAPTIVE_BATCH_EXTERN_INST

template <typename T>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<T> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive<float> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive<double> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive<mppp::real128> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive<mppp::real> &);

#endif

template <typename T>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<T> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch<float> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch<double> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const taylor_adaptive_batch<mppp::real128> &);

#endif

namespace detail
{

// Boost s11n class version history for taylor_adaptive:
// - 1: added base class to taylor_adaptive.
// - 2: added the m_state_vars and m_rhs members.
// - 3: removed the mr flag from the terminal event callback siganture,
//      which resulted also in changes in the event detection data structure.
// - 4: switched to pimpl implementation for i_data.
// - 5: removed m_state_vars/m_rhs, variational ODE data.
// - 6: added parallel JIT compilation for compact mode.
inline constexpr int taylor_adaptive_s11n_version = 6;

// Boost s11n class version history for taylor_adaptive_batch:
// - 1: added the m_state_vars and m_rhs members.
// - 2: removed the mr flag from the terminal event callback siganture,
//      which resulted also in changes in the event detection data structure.
// - 3: switched to pimpl implementation for i_data.
// - 4: removed m_state_vars/m_rhs, variational ODE data.
// - 5: added parallel JIT compilation for compact mode.
inline constexpr int taylor_adaptive_batch_s11n_version = 5;

} // namespace detail

HEYOKA_END_NAMESPACE

// Set the Boost s11n class version for taylor_adaptive and taylor_adaptive_batch.
BOOST_CLASS_VERSION(heyoka::taylor_adaptive<float>, heyoka::detail::taylor_adaptive_s11n_version);
BOOST_CLASS_VERSION(heyoka::taylor_adaptive<double>, heyoka::detail::taylor_adaptive_s11n_version);
BOOST_CLASS_VERSION(heyoka::taylor_adaptive<long double>, heyoka::detail::taylor_adaptive_s11n_version);

BOOST_CLASS_VERSION(heyoka::taylor_adaptive_batch<float>, heyoka::detail::taylor_adaptive_batch_s11n_version);
BOOST_CLASS_VERSION(heyoka::taylor_adaptive_batch<double>, heyoka::detail::taylor_adaptive_batch_s11n_version);
BOOST_CLASS_VERSION(heyoka::taylor_adaptive_batch<long double>, heyoka::detail::taylor_adaptive_batch_s11n_version);

#if defined(HEYOKA_HAVE_REAL128)

BOOST_CLASS_VERSION(heyoka::taylor_adaptive<mppp::real128>, heyoka::detail::taylor_adaptive_s11n_version);

BOOST_CLASS_VERSION(heyoka::taylor_adaptive_batch<mppp::real128>, heyoka::detail::taylor_adaptive_batch_s11n_version);

#endif

#if defined(HEYOKA_HAVE_REAL)

BOOST_CLASS_VERSION(heyoka::taylor_adaptive<mppp::real>, heyoka::detail::taylor_adaptive_s11n_version);

// NOTE: this is not really necessary as the batch integrator cannot be used with real, but let's
// just leave it for consistency.
BOOST_CLASS_VERSION(heyoka::taylor_adaptive_batch<mppp::real>, heyoka::detail::taylor_adaptive_batch_s11n_version);

#endif

#endif
