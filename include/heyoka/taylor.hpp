// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
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

std::uint32_t n_pars_in_dc(const taylor_dc_t &);

taylor_dc_t taylor_add_adaptive_step_with_events(llvm_state &, llvm::Type *, llvm::Type *, const std::string &,
                                                 const std::vector<std::pair<expression, expression>> &, std::uint32_t,
                                                 bool, const std::vector<expression> &, bool, bool, std::uint32_t);

taylor_dc_t taylor_add_adaptive_step(llvm_state &, llvm::Type *, llvm::Type *, const std::string &,
                                     const std::vector<std::pair<expression, expression>> &, std::uint32_t, bool, bool,
                                     bool, std::uint32_t);

llvm::Value *taylor_c_make_sv_funcs_arr(llvm_state &, const std::vector<std::uint32_t> &);

llvm::Value *
taylor_determine_h(llvm_state &, llvm::Type *,
                   const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &,
                   const std::vector<std::uint32_t> &, llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t,
                   std::uint32_t, std::uint32_t, llvm::Value *);

std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>>
taylor_compute_jet(llvm_state &, llvm::Type *, llvm::Value *, llvm::Value *, llvm::Value *, const taylor_dc_t &,
                   const std::vector<std::uint32_t> &, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool,
                   bool, bool);

void taylor_write_tc(llvm_state &, llvm::Type *,
                     const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &,
                     const std::vector<std::uint32_t> &, llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t,
                     std::uint32_t, std::uint32_t);

std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_multihorner(llvm_state &, llvm::Type *,
                       const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &,
                       llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_ceval(llvm_state &, llvm::Type *,
                 const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &,
                 llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, bool, std::uint32_t);

std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args(llvm::LLVMContext &, llvm::Type *, const std::string &, std::uint32_t, std::uint32_t,
                             const std::vector<std::variant<variable, number, param>> &, std::uint32_t = 0);

// Add a function for computing the dense output
// via polynomial evaluation.
void taylor_add_d_out_function(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, std::uint32_t, bool,
                               bool = true);

} // namespace detail

HEYOKA_DLL_PUBLIC std::pair<taylor_dc_t, std::vector<std::uint32_t>> taylor_decompose(const std::vector<expression> &,
                                                                                      const std::vector<expression> &);
HEYOKA_DLL_PUBLIC std::pair<taylor_dc_t, std::vector<std::uint32_t>>
taylor_decompose(const std::vector<std::pair<expression, expression>> &, const std::vector<expression> &);

template <typename>
taylor_dc_t taylor_add_jet(llvm_state &, const std::string &, const std::vector<expression> &, std::uint32_t,
                           std::uint32_t, bool, bool, const std::vector<expression> & = {}, bool = false,
                           long long = 0);

template <typename>
taylor_dc_t taylor_add_jet(llvm_state &, const std::string &, const std::vector<std::pair<expression, expression>> &,
                           std::uint32_t, std::uint32_t, bool, bool, const std::vector<expression> & = {}, bool = false,
                           long long = 0);

// Prevent implicit instantiations.
#define HEYOKA_TAYLOR_ADD_JET_EXTERN_INST(F)                                                                           \
    extern template taylor_dc_t taylor_add_jet<F>(llvm_state &, const std::string &, const std::vector<expression> &,  \
                                                  std::uint32_t, std::uint32_t, bool, bool,                            \
                                                  const std::vector<expression> &, bool, long long);                   \
    extern template taylor_dc_t taylor_add_jet<F>(                                                                     \
        llvm_state &, const std::string &, const std::vector<std::pair<expression, expression>> &, std::uint32_t,      \
        std::uint32_t, bool, bool, const std::vector<expression> &, bool, long long);

HEYOKA_TAYLOR_ADD_JET_EXTERN_INST(float)
HEYOKA_TAYLOR_ADD_JET_EXTERN_INST(double)
HEYOKA_TAYLOR_ADD_JET_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADD_JET_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_TAYLOR_ADD_JET_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_TAYLOR_ADD_JET_EXTERN_INST

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

// Helper for parsing common options for the Taylor integrators.
template <typename T, typename... KwArgs>
inline auto taylor_adaptive_common_ops(const KwArgs &...kw_args)
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

// Polynomial cache type. Each entry is a polynomial
// represented as a vector of coefficients. Used
// during event detection.
template <typename T>
using taylor_poly_cache = std::vector<std::vector<T>>;

// A RAII helper to extract polys from a cache and
// return them to the cache upon destruction. Used
// during event detection.
template <typename>
class taylor_pwrap;

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
Callback parse_propagate_cb(Parser &p)
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

template <typename TA, typename U>
void taylor_adaptive_setup_sv_rhs(TA &, const U &);

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS taylor_adaptive : public detail::taylor_adaptive_base<T, taylor_adaptive<T>>
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");
    friend class HEYOKA_DLL_PUBLIC_INLINE_CLASS detail::taylor_adaptive_base<T, taylor_adaptive<T>>;
    using base_t = detail::taylor_adaptive_base<T, taylor_adaptive<T>>;
    template <typename TA, typename U>
    friend void detail::taylor_adaptive_setup_sv_rhs(TA &, const U &);

public:
    using nt_event_t = nt_event<T>;
    using t_event_t = t_event<T>;

private:
    // Struct implementing the data/logic for event detection.
    struct HEYOKA_DLL_PUBLIC ed_data {
        // The working list type used during real root isolation.
        using wlist_t = std::vector<std::tuple<T, T, detail::taylor_pwrap<T>>>;
        // The type used to store the list of isolating intervals.
        using isol_t = std::vector<std::tuple<T, T>>;
        // Polynomial translation function type.
        using pt_t = void (*)(T *, const T *) noexcept;
        // rtscc function type.
        using rtscc_t = void (*)(T *, T *, std::uint32_t *, const T *) noexcept;
        // fex_check function type.
        using fex_check_t = void (*)(const T *, const T *, const std::uint32_t *, std::uint32_t *) noexcept;

        // The vector of terminal events.
        std::vector<t_event_t> m_tes;
        // The vector of non-terminal events.
        std::vector<nt_event_t> m_ntes;
        // The jet of derivatives for the state variables
        // and the events.
        std::vector<T> m_ev_jet;
        // Vector of detected terminal events.
        std::vector<std::tuple<std::uint32_t, T, int, T>> m_d_tes;
        // The vector of cooldowns for the terminal events.
        // If an event is on cooldown, the corresponding optional
        // in this vector will contain the total time elapsed
        // since the cooldown started and the absolute value
        // of the cooldown duration.
        std::vector<std::optional<std::pair<T, T>>> m_te_cooldowns;
        // Vector of detected non-terminal events.
        std::vector<std::tuple<std::uint32_t, T, int>> m_d_ntes;
        // The LLVM state.
        llvm_state m_state;
        // The JIT compiled functions used during root finding.
        // NOTE: use default member initializers to ensure that
        // these are zero-inited by the default constructor
        // (which is defaulted).
        pt_t m_pt = nullptr;
        rtscc_t m_rtscc = nullptr;
        fex_check_t m_fex_check = nullptr;
        // The polynomial cache.
        // NOTE: it is *really* important that this is declared
        // *before* m_wlist, because m_wlist will contain references
        // to and interact with m_poly_cache during destruction,
        // and we must be sure that m_wlist is destroyed *before*
        // m_poly_cache.
        detail::taylor_poly_cache<T> m_poly_cache;
        // The working list.
        wlist_t m_wlist;
        // The list of isolating intervals.
        isol_t m_isol;

        // Constructors.
        ed_data(llvm_state, std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t, const T &);
        ed_data(const ed_data &);
        ~ed_data();

        // Delete unused bits.
        ed_data(ed_data &&) = delete;
        ed_data &operator=(const ed_data &) = delete;
        ed_data &operator=(ed_data &&) = delete;

        // The event detection function.
        void detect_events(const T &, std::uint32_t, std::uint32_t, const T &);

    private:
        // Serialisation.
        // NOTE: the def ctor is used only during deserialisation
        // via pointer.
        ed_data();
        friend class boost::serialization::access;
        void save(boost::archive::binary_oarchive &, unsigned) const;
        void load(boost::archive::binary_iarchive &, unsigned);
        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };

    // State vector.
    std::vector<T> m_state;
    // Time.
    detail::dfloat<T> m_time;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Dimension of the system.
    std::uint32_t m_dim{};
    // Taylor decomposition.
    taylor_dc_t m_dc;
    // Taylor order.
    std::uint32_t m_order{};
    // Tolerance.
    T m_tol;
    // High accuracy.
    bool m_high_accuracy{};
    // Compact mode.
    bool m_compact_mode{};
    // The steppers.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *) noexcept;
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *) noexcept;
    std::variant<step_f_t, step_f_e_t> m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // Size of the last timestep taken.
    T m_last_h = T(0);
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *) noexcept;
    d_out_f_t m_d_out_f;
    // The vector for the dense output.
    std::vector<T> m_d_out;
    // Auxiliary data/functions for event detection.
    std::unique_ptr<ed_data> m_ed_data;
    // The state variables and the rhs.
    std::vector<expression> m_state_vars, m_rhs;

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
    void finalise_ctor_impl(const std::vector<std::pair<expression, expression>> &, std::vector<T>, std::optional<T>,
                            std::optional<T>, bool, bool, std::vector<T>, std::vector<t_event_t>,
                            std::vector<nt_event_t>, bool, std::optional<long long>);
    template <typename... KwArgs>
    void finalise_ctor(const std::vector<std::pair<expression, expression>> &sys, std::vector<T> state,
                       KwArgs &&...kw_args)
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
                = detail::taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

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

            finalise_ctor_impl(sys, std::move(state), std::move(tm), std::move(tol), high_accuracy, compact_mode,
                               std::move(pars), std::move(tes), std::move(ntes), parallel_mode, std::move(prec));
        }
    }

public:
    taylor_adaptive();

    template <typename... KwArgs>
    explicit taylor_adaptive(const std::vector<std::pair<expression, expression>> &sys, std::vector<T> state,
                             KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive(const std::vector<std::pair<expression, expression>> &sys, std::initializer_list<T> state,
                             KwArgs &&...kw_args)
        : taylor_adaptive(sys, std::vector<T>(state), std::forward<KwArgs>(kw_args)...)
    {
    }

    taylor_adaptive(const taylor_adaptive &);
    taylor_adaptive(taylor_adaptive &&) noexcept;

    taylor_adaptive &operator=(const taylor_adaptive &);
    taylor_adaptive &operator=(taylor_adaptive &&) noexcept;

    ~taylor_adaptive();

    [[nodiscard]] const llvm_state &get_llvm_state() const;

    [[nodiscard]] const taylor_dc_t &get_decomposition() const;

    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] T get_tol() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] bool get_compact_mode() const;
    [[nodiscard]] std::uint32_t get_dim() const;

    [[nodiscard]] T get_time() const;
    void set_time(T);

    // Time set/get in double-length format.
    [[nodiscard]] std::pair<T, T> get_dtime() const;
    void set_dtime(T, T);

    [[nodiscard]] const std::vector<T> &get_state() const;
    [[nodiscard]] const T *get_state_data() const;
    T *get_state_data();

    [[nodiscard]] const std::vector<T> &get_pars() const;
    [[nodiscard]] const T *get_pars_data() const;
    T *get_pars_data();

    [[nodiscard]] const std::vector<T> &get_tc() const;

    [[nodiscard]] T get_last_h() const;

    [[nodiscard]] const std::vector<T> &get_d_output() const;
    const std::vector<T> &update_d_output(T, bool = false);

    [[nodiscard]] bool with_events() const;
    void reset_cooldowns();
    [[nodiscard]] const std::vector<t_event_t> &get_t_events() const;
    [[nodiscard]] const std::vector<std::optional<std::pair<T, T>>> &get_te_cooldowns() const;
    [[nodiscard]] const std::vector<nt_event_t> &get_nt_events() const;

    [[nodiscard]] const std::vector<expression> &get_state_vars() const;
    [[nodiscard]] const std::vector<expression> &get_rhs() const;

    std::tuple<taylor_outcome, T> step(bool = false);
    std::tuple<taylor_outcome, T> step_backward(bool = false);
    std::tuple<taylor_outcome, T> step(T, bool = false);

private:
    // Implementations of the propagate_*() functions.
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
    propagate_until_impl(detail::dfloat<T>, std::size_t, T, step_callback<T>, bool, bool);
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

        return propagate_until_impl(m_time + std::move(delta_t), max_steps, std::move(max_delta_t), std::move(cb),
                                    write_tc, with_c_out);
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
        auto max_delta_t = [batch_size, &p]() -> std::vector<T> {
            // NOTE: compiler warning.
            (void)batch_size;

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

    template <typename TA, typename U>
    friend void detail::taylor_adaptive_setup_sv_rhs(TA &, const U &);

public:
    using nt_event_t = nt_event_batch<T>;
    using t_event_t = t_event_batch<T>;

private:
    // Struct implementing the data/logic for event detection.
    struct HEYOKA_DLL_PUBLIC ed_data {
        // The working list type used during real root isolation.
        using wlist_t = std::vector<std::tuple<T, T, detail::taylor_pwrap<T>>>;
        // The type used to store the list of isolating intervals.
        using isol_t = std::vector<std::tuple<T, T>>;
        // Polynomial translation function type.
        using pt_t = void (*)(T *, const T *) noexcept;
        // rtscc function type.
        using rtscc_t = void (*)(T *, T *, std::uint32_t *, const T *) noexcept;
        // fex_check function type.
        using fex_check_t = void (*)(const T *, const T *, const std::uint32_t *, std::uint32_t *) noexcept;

        // The vector of terminal events.
        std::vector<t_event_t> m_tes;
        // The vector of non-terminal events.
        std::vector<nt_event_t> m_ntes;
        // The jet of derivatives for the state variables
        // and the events.
        std::vector<T> m_ev_jet;
        // The vector to store the norm infinity of the state
        // vector when using the stepper with events.
        std::vector<T> m_max_abs_state;
        // The vector to store the the maximum absolute error
        // on the Taylor series of the event equations.
        std::vector<T> m_g_eps;
        // Vector of detected terminal events.
        std::vector<std::vector<std::tuple<std::uint32_t, T, int, T>>> m_d_tes;
        // The vector of cooldowns for the terminal events.
        // If an event is on cooldown, the corresponding optional
        // in this vector will contain the total time elapsed
        // since the cooldown started and the absolute value
        // of the cooldown duration.
        std::vector<std::vector<std::optional<std::pair<T, T>>>> m_te_cooldowns;
        // Vector of detected non-terminal events.
        std::vector<std::vector<std::tuple<std::uint32_t, T, int>>> m_d_ntes;
        // The LLVM state.
        llvm_state m_state;
        // Flags to signal if we are integrating backwards in time.
        std::vector<std::uint32_t> m_back_int;
        // Output of the fast exclusion check.
        std::vector<std::uint32_t> m_fex_check_res;
        // The JIT compiled functions used during root finding.
        // NOTE: use default member initializers to ensure that
        // these are zero-inited by the default constructor
        // (which is defaulted).
        pt_t m_pt = nullptr;
        rtscc_t m_rtscc = nullptr;
        fex_check_t m_fex_check = nullptr;
        // The polynomial cache.
        // NOTE: it is *really* important that this is declared
        // *before* m_wlist, because m_wlist will contain references
        // to and interact with m_poly_cache during destruction,
        // and we must be sure that m_wlist is destroyed *before*
        // m_poly_cache.
        detail::taylor_poly_cache<T> m_poly_cache;
        // The working list.
        wlist_t m_wlist;
        // The list of isolating intervals.
        isol_t m_isol;

        // Constructors.
        ed_data(llvm_state, std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t,
                std::uint32_t);
        ed_data(const ed_data &);
        ~ed_data();

        // Delete unused bits.
        ed_data(ed_data &&) = delete;
        ed_data &operator=(const ed_data &) = delete;
        ed_data &operator=(ed_data &&) = delete;

        // The event detection function.
        void detect_events(const T *, std::uint32_t, std::uint32_t, std::uint32_t);

    private:
        // Serialisation.
        // NOTE: the def ctor is used only during deserialisation
        // via pointer.
        ed_data();
        friend class boost::serialization::access;
        void save(boost::archive::binary_oarchive &, unsigned) const;
        void load(boost::archive::binary_iarchive &, unsigned);
        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };

    // The batch size.
    std::uint32_t m_batch_size{};
    // State vectors.
    std::vector<T> m_state;
    // Times.
    std::vector<T> m_time_hi, m_time_lo;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Dimension of the system.
    std::uint32_t m_dim{};
    // Taylor decomposition.
    taylor_dc_t m_dc;
    // Taylor order.
    std::uint32_t m_order{};
    // Tolerance.
    T m_tol;
    // High accuracy.
    bool m_high_accuracy{};
    // Compact mode.
    bool m_compact_mode{};
    // The steppers.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *) noexcept;
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *) noexcept;
    std::variant<step_f_t, step_f_e_t> m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // The sizes of the last timesteps taken.
    std::vector<T> m_last_h;
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *) noexcept;
    d_out_f_t m_d_out_f;
    // The vector for the dense output.
    std::vector<T> m_d_out;
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
    // Temporary vectors used in the step()/propagate_*() implementations.
    std::vector<std::size_t> m_ts_count;
    std::vector<T> m_min_abs_h, m_max_abs_h;
    std::vector<T> m_cur_max_delta_ts;
    std::vector<detail::dfloat<T>> m_pfor_ts;
    std::vector<int> m_t_dir;
    std::vector<detail::dfloat<T>> m_rem_time;
    std::vector<T> m_time_copy_hi, m_time_copy_lo;
    std::vector<int> m_nf_detected;
    // Temporary vector used in the dense output implementation.
    std::vector<T> m_d_out_time;
    // Auxiliary data/functions for event detection.
    std::unique_ptr<ed_data> m_ed_data;
    // The state variables and the rhs.
    std::vector<expression> m_state_vars, m_rhs;

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
    void finalise_ctor_impl(const std::vector<std::pair<expression, expression>> &, std::vector<T>, std::uint32_t,
                            std::vector<T>, std::optional<T>, bool, bool, std::vector<T>, std::vector<t_event_t>,
                            std::vector<nt_event_t>, bool);
    template <typename... KwArgs>
    void finalise_ctor(const std::vector<std::pair<expression, expression>> &sys, std::vector<T> state,
                       std::uint32_t batch_size, KwArgs &&...kw_args)
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
                    return p(kw::time);
                } else {
                    return std::vector<T>(static_cast<typename std::vector<T>::size_type>(batch_size), T(0));
                }
            }();

            auto [high_accuracy, tol, compact_mode, pars, parallel_mode]
                = detail::taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

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

            finalise_ctor_impl(sys, std::move(state), batch_size, std::move(tm), std::move(tol), high_accuracy,
                               compact_mode, std::move(pars), std::move(tes), std::move(ntes), parallel_mode);
        }
    }

public:
    taylor_adaptive_batch();

    template <typename... KwArgs>
    explicit taylor_adaptive_batch(const std::vector<std::pair<expression, expression>> &sys, std::vector<T> state,
                                   std::uint32_t batch_size, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), batch_size, std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch(const std::vector<std::pair<expression, expression>> &sys,
                                   std::initializer_list<T> state, std::uint32_t batch_size, KwArgs &&...kw_args)
        : taylor_adaptive_batch(sys, std::vector<T>(state), batch_size, std::forward<KwArgs>(kw_args)...)
    {
    }

    taylor_adaptive_batch(const taylor_adaptive_batch &);
    taylor_adaptive_batch(taylor_adaptive_batch &&) noexcept;

    taylor_adaptive_batch &operator=(const taylor_adaptive_batch &);
    taylor_adaptive_batch &operator=(taylor_adaptive_batch &&) noexcept;

    ~taylor_adaptive_batch();

    [[nodiscard]] const llvm_state &get_llvm_state() const;

    [[nodiscard]] const taylor_dc_t &get_decomposition() const;

    [[nodiscard]] std::uint32_t get_batch_size() const;
    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] T get_tol() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] bool get_compact_mode() const;
    [[nodiscard]] std::uint32_t get_dim() const;

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
    T *get_state_data();

    [[nodiscard]] const std::vector<T> &get_pars() const;
    [[nodiscard]] const T *get_pars_data() const;
    T *get_pars_data();

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

    [[nodiscard]] const std::vector<expression> &get_state_vars() const;
    [[nodiscard]] const std::vector<expression> &get_rhs() const;

    void step(bool = false);
    void step_backward(bool = false);
    void step(const std::vector<T> &, bool = false);
    [[nodiscard]] const std::vector<std::tuple<taylor_outcome, T>> &get_step_res() const;

private:
    // Implementations of the propagate_*() functions.
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_until_impl(const std::vector<detail::dfloat<T>> &, std::size_t, const std::vector<T> &,
                         step_callback_batch<T>, bool, bool);
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_until_impl(const std::vector<T> &, std::size_t, const std::vector<T> &, step_callback_batch<T>, bool,
                         bool);
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_for_impl(const std::vector<T> &, std::size_t, const std::vector<T> &, step_callback_batch<T>, bool, bool);
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
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size, kw_args...);

        return propagate_until_impl(ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb),
                                    write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_until(T t, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size, kw_args...);

        // NOTE: re-use m_pfor_ts as tmp storage, as the other overload does.
        assert(m_pfor_ts.size() == m_batch_size);
        std::fill(m_pfor_ts.begin(), m_pfor_ts.end(), detail::dfloat<T>(t));
        return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb),
                                    write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_for(const std::vector<T> &delta_ts, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size, kw_args...);

        return propagate_for_impl(delta_ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb),
                                  write_tc, with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
    propagate_for(T delta_t, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size, kw_args...);

        // NOTE: this is a slight repetition of the other overload's code.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_pfor_ts[i] = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + delta_t;
        }
        return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb),
                                    write_tc, with_c_out);
    }
    // NOTE: grid is taken by copy because in the implementation loop we keep on reading from it.
    // Hence, we need to avoid any aliasing issue with other public integrator data.
    template <typename... KwArgs>
    std::tuple<step_callback_batch<T>, std::vector<T>> propagate_grid(std::vector<T> grid, const KwArgs &...kw_args)
    {
        auto [max_steps, max_delta_ts, cb]
            = detail::taylor_propagate_common_ops_batch<T, true, false>(m_batch_size, kw_args...);

        return propagate_grid_impl(grid, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb));
    }
    [[nodiscard]] const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &get_propagate_res() const
    {
        return m_prop_res;
    }
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
inline std::ostream &operator<<(std::ostream &os, const taylor_adaptive<T> &)
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
inline std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<T> &)
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
inline constexpr int taylor_adaptive_s11n_version = 3;

// Boost s11n class version history for taylor_adaptive_batch:
// - 1: added the m_state_vars and m_rhs members.
// - 2: removed the mr flag from the terminal event callback siganture,
//      which resulted also in changes in the event detection data structure.
inline constexpr int taylor_adaptive_batch_s11n_version = 2;

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
