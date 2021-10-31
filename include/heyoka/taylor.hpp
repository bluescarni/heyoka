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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

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

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &, const number &, llvm::Value *,
                                                              llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &, const param &, llvm::Value *, llvm::Value *,
                                                              std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &, std::uint32_t, std::uint32_t,
                                                 std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_c_load_diff(llvm_state &, llvm::Value *, std::uint32_t, llvm::Value *,
                                                  llvm::Value *);

HEYOKA_DLL_PUBLIC std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args_impl(llvm::LLVMContext &, const std::string &, llvm::Type *, std::uint32_t,
                                  const std::vector<std::variant<variable, number, param>> &, std::uint32_t);

// NOTE: this function will return a pair containing:
//
// - the mangled name and
// - the list of LLVM argument types
//
// for the function implementing the Taylor derivative in compact mode of the mathematical function
// called "name". The mangled name is assembled from "name", the types of the arguments args, the number
// of uvars and the scalar or vector floating-point type in use (which depends on T and batch_size).
template <typename T>
inline std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args(llvm::LLVMContext &c, const std::string &name, std::uint32_t n_uvars,
                             std::uint32_t batch_size, const std::vector<std::variant<variable, number, param>> &args,
                             std::uint32_t n_hidden_deps = 0)
{
    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(c, batch_size);

    return taylor_c_diff_func_name_args_impl(c, name, val_t, n_uvars, args, n_hidden_deps);
}

} // namespace detail

HEYOKA_DLL_PUBLIC std::pair<taylor_dc_t, std::vector<std::uint32_t>> taylor_decompose(const std::vector<expression> &,
                                                                                      const std::vector<expression> &);
HEYOKA_DLL_PUBLIC std::pair<taylor_dc_t, std::vector<std::uint32_t>>
taylor_decompose(const std::vector<std::pair<expression, expression>> &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet_dbl(llvm_state &, const std::string &, const std::vector<expression> &,
                                                 std::uint32_t, std::uint32_t, bool, bool,
                                                 const std::vector<expression> &);
HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet_ldbl(llvm_state &, const std::string &, const std::vector<expression> &,
                                                  std::uint32_t, std::uint32_t, bool, bool,
                                                  const std::vector<expression> &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet_f128(llvm_state &, const std::string &, const std::vector<expression> &,
                                                  std::uint32_t, std::uint32_t, bool, bool,
                                                  const std::vector<expression> &);

#endif

template <typename T>
taylor_dc_t taylor_add_jet(llvm_state &s, const std::string &name, const std::vector<expression> &sys,
                           std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                           const std::vector<expression> &sv_funcs = {})
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_jet_dbl(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_jet_ldbl(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_jet_f128(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet_dbl(llvm_state &, const std::string &,
                                                 const std::vector<std::pair<expression, expression>> &, std::uint32_t,
                                                 std::uint32_t, bool, bool, const std::vector<expression> &);
HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet_ldbl(llvm_state &, const std::string &,
                                                  const std::vector<std::pair<expression, expression>> &, std::uint32_t,
                                                  std::uint32_t, bool, bool, const std::vector<expression> &);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet_f128(llvm_state &, const std::string &,
                                                  const std::vector<std::pair<expression, expression>> &, std::uint32_t,
                                                  std::uint32_t, bool, bool, const std::vector<expression> &);

#endif

template <typename T>
taylor_dc_t taylor_add_jet(llvm_state &s, const std::string &name,
                           const std::vector<std::pair<expression, expression>> &sys, std::uint32_t order,
                           std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                           const std::vector<expression> &sv_funcs = {})
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_add_jet_dbl(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_add_jet_ldbl(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_add_jet_f128(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

// Enum to represent the outcome of a stepping/propagate function.
enum class taylor_outcome : std::int64_t {
    // NOTE: we make these enums start at -2**32 - 1,
    // so that we have 2**32 values in the [-2**32, -1]
    // range to use for signalling stopping terminal events.
    success = -4294967296ll - 1,      // Integration step was successful, no time/step limits were reached.
    step_limit = -4294967296ll - 2,   // Maximum number of steps reached.
    time_limit = -4294967296ll - 3,   // Time limit reached.
    err_nf_state = -4294967296ll - 4, // Non-finite state detected at the end of the timestep.
    cb_stop = -4294967296ll - 5       // Propagation stopped by callback.
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, taylor_outcome);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, event_direction);

} // namespace heyoka

// NOTE: implement a workaround for the serialisation of tuples whose first element
// is a taylor outcome. We need this because Boost.Serialization treats all enums
// as ints, which is not ok for taylor_outcome (whose underyling type will not
// be an int on most platforms). Because it is not possible to override Boost's
// enum implementation, we override the serialisation of tuples with outcomes
// as first elements, which is all we need in the serialisation of the batch
// integrator. The implementation below will be preferred over the generic tuple
// s11n because it is more specialised.
// NOTE: this workaround is not necessary for the other enums in heyoka because
// those all have ints as underlying type.
namespace boost
{

namespace serialization
{

template <typename Archive, typename... Args>
inline void save(Archive &ar, const std::tuple<heyoka::taylor_outcome, Args...> &tup, unsigned)
{
    auto tf = [&ar](const auto &x) {
        if constexpr (std::is_same_v<decltype(x), const heyoka::taylor_outcome &>) {
            ar << static_cast<std::underlying_type_t<heyoka::taylor_outcome>>(x);
        } else {
            ar << x;
        }
    };

    std::apply([&tf](const auto &...x) { (tf(x), ...); }, tup);
}

template <typename Archive, typename... Args>
inline void load(Archive &ar, std::tuple<heyoka::taylor_outcome, Args...> &tup, unsigned)
{
    auto tf = [&ar](auto &x) {
        if constexpr (std::is_same_v<decltype(x), heyoka::taylor_outcome &>) {
            std::underlying_type_t<heyoka::taylor_outcome> val{};
            ar >> val;

            x = static_cast<heyoka::taylor_outcome>(val);
        } else {
            ar >> x;
        }
    };

    std::apply([&tf](auto &...x) { (tf(x), ...); }, tup);
}

template <typename Archive, typename... Args>
inline void serialize(Archive &ar, std::tuple<heyoka::taylor_outcome, Args...> &tup, unsigned v)
{
    split_free(ar, tup, v);
}

} // namespace serialization

} // namespace boost

namespace heyoka
{

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(tol);
IGOR_MAKE_NAMED_ARGUMENT(high_accuracy);
IGOR_MAKE_NAMED_ARGUMENT(compact_mode);
IGOR_MAKE_NAMED_ARGUMENT(pars);
IGOR_MAKE_NAMED_ARGUMENT(t_events);
IGOR_MAKE_NAMED_ARGUMENT(nt_events);

// NOTE: these are used for constructing events.
IGOR_MAKE_NAMED_ARGUMENT(callback);
IGOR_MAKE_NAMED_ARGUMENT(cooldown);
IGOR_MAKE_NAMED_ARGUMENT(direction);

// NOTE: these are used in the
// propagate_*() functions.
IGOR_MAKE_NAMED_ARGUMENT(max_steps);
IGOR_MAKE_NAMED_ARGUMENT(max_delta_t);
IGOR_MAKE_NAMED_ARGUMENT(write_tc);

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

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC nt_event_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using callback_t = callable<std::conditional_t<B, void(taylor_adaptive_batch_impl<T> &, T, int, std::uint32_t),
                                                   void(taylor_adaptive_impl<T> &, T, int)>>;

private:
    expression eq;
    callback_t callback;
    event_direction dir;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &eq;
        ar &callback;
        ar &dir;
    }

    void finalise_ctor(event_direction);

public:
    nt_event_impl();

    template <typename... KwArgs>
    explicit nt_event_impl(const expression &e, callback_t cb, KwArgs &&...kw_args)
        : eq(copy(e)), callback(std::move(cb))
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of a non-terminal event contain "
                          "unnamed arguments.");
            throw;
        } else {
            // Direction (defaults to any).
            auto d = [&p]() -> event_direction {
                if constexpr (p.has(kw::direction)) {
                    return std::forward<decltype(p(kw::direction))>(p(kw::direction));
                } else {
                    return event_direction::any;
                }
            }();

            finalise_ctor(d);
        }
    }

    nt_event_impl(const nt_event_impl &);
    nt_event_impl(nt_event_impl &&) noexcept;

    nt_event_impl &operator=(const nt_event_impl &);
    nt_event_impl &operator=(nt_event_impl &&) noexcept;

    ~nt_event_impl();

    const expression &get_expression() const;
    const callback_t &get_callback() const;
    event_direction get_direction() const;
};

template <typename T, bool B>
inline std::ostream &operator<<(std::ostream &os, const nt_event_impl<T, B> &)
{
    static_assert(always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<double, true> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<long double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<long double, true> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<mppp::real128, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<mppp::real128, true> &);

#endif

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC t_event_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using callback_t = callable<std::conditional_t<B, bool(taylor_adaptive_batch_impl<T> &, bool, int, std::uint32_t),
                                                   bool(taylor_adaptive_impl<T> &, bool, int)>>;

private:
    expression eq;
    callback_t callback;
    T cooldown;
    event_direction dir;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &eq;
        ar &callback;
        ar &cooldown;
        ar &dir;
    }

    void finalise_ctor(callback_t, T, event_direction);

public:
    t_event_impl();

    template <typename... KwArgs>
    explicit t_event_impl(const expression &e, KwArgs &&...kw_args) : eq(copy(e))
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of a terminal event contain "
                          "unnamed arguments.");
            throw;
        } else {
            // Callback (defaults to empty).
            auto cb = [&p]() -> callback_t {
                if constexpr (p.has(kw::callback)) {
                    return std::forward<decltype(p(kw::callback))>(p(kw::callback));
                } else {
                    return {};
                }
            }();

            // Cooldown (defaults to -1).
            auto cd = [&p]() -> T {
                if constexpr (p.has(kw::cooldown)) {
                    return std::forward<decltype(p(kw::cooldown))>(p(kw::cooldown));
                } else {
                    return T(-1);
                }
            }();

            // Direction (defaults to any).
            auto d = [&p]() -> event_direction {
                if constexpr (p.has(kw::direction)) {
                    return std::forward<decltype(p(kw::direction))>(p(kw::direction));
                } else {
                    return event_direction::any;
                }
            }();

            finalise_ctor(std::move(cb), cd, d);
        }
    }

    t_event_impl(const t_event_impl &);
    t_event_impl(t_event_impl &&) noexcept;

    t_event_impl &operator=(const t_event_impl &);
    t_event_impl &operator=(t_event_impl &&) noexcept;

    ~t_event_impl();

    const expression &get_expression() const;
    const callback_t &get_callback() const;
    event_direction get_direction() const;
    T get_cooldown() const;
};

template <typename T, bool B>
inline std::ostream &operator<<(std::ostream &os, const t_event_impl<T, B> &)
{
    static_assert(always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<double, true> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<long double, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<long double, true> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<mppp::real128, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<mppp::real128, true> &);

#endif

} // namespace detail

template <typename T>
using nt_event = detail::nt_event_impl<T, false>;

template <typename T>
using t_event = detail::t_event_impl<T, false>;

template <typename T>
using nt_batch_event = detail::nt_event_impl<T, true>;

template <typename T>
using t_batch_event = detail::t_event_impl<T, true>;

namespace detail
{

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

template <typename T>
class HEYOKA_DLL_PUBLIC taylor_adaptive_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using nt_event_t = nt_event<T>;
    using t_event_t = t_event<T>;

private:
    // Struct implementing the data/logic for event detection.
    struct HEYOKA_DLL_PUBLIC ed_data {
        // The working list type used during real root isolation.
        using wlist_t = std::vector<std::tuple<T, T, taylor_pwrap<T>>>;
        // The type used to store the list of isolating intervals.
        using isol_t = std::vector<std::tuple<T, T>>;
        // Polynomial translation function type.
        using pt_t = void (*)(T *, const T *);
        // rtscc function type.
        using rtscc_t = void (*)(T *, T *, std::uint32_t *, const T *);
        // fex_check function type.
        using fex_check_t = void (*)(const T *, const T *, const std::uint32_t *, std::uint32_t *);

        // The vector of terminal events.
        std::vector<t_event_t> m_tes;
        // The vector of non-terminal events.
        std::vector<nt_event_t> m_ntes;
        // The jet of derivatives for the state variables
        // and the events.
        std::vector<T> m_ev_jet;
        // Vector of detected terminal events.
        std::vector<std::tuple<std::uint32_t, T, bool, int, T>> m_d_tes;
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
        // The working list.
        wlist_t m_wlist;
        // The list of isolating intervals.
        isol_t m_isol;
        // The polynomial cache.
        taylor_poly_cache<T> m_poly_cache;

        // Constructors.
        ed_data(std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t);
        ed_data(const ed_data &);
        ~ed_data();

        // Delete unused bits.
        ed_data(ed_data &&) = delete;
        ed_data &operator=(const ed_data &) = delete;
        ed_data &operator=(ed_data &&) = delete;

        // The event detection function.
        void detect_events(T, std::uint32_t, std::uint32_t, T);

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
    dfloat<T> m_time;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Dimension of the system.
    std::uint32_t m_dim;
    // Taylor decomposition.
    taylor_dc_t m_dc;
    // Taylor order.
    std::uint32_t m_order;
    // Tolerance.
    T m_tol;
    // The steppers.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *);
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *);
    std::variant<step_f_t, step_f_e_t> m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // Size of the last timestep taken.
    T m_last_h = T(0);
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *);
    d_out_f_t m_d_out_f;
    // The vector for the dense output.
    std::vector<T> m_d_out;
    // Auxiliary data/functions for event detection.
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
    // NOTE: apparently on Windows we need to re-iterate
    // here that this is going to be dll-exported.
    template <typename U>
    HEYOKA_DLL_PUBLIC void finalise_ctor_impl(const U &, std::vector<T>, T, T, bool, bool, std::vector<T>,
                                              std::vector<t_event_t>, std::vector<nt_event_t>);
    template <typename U, typename... KwArgs>
    void finalise_ctor(const U &sys, std::vector<T> state, KwArgs &&...kw_args)
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

            // Extract the terminal events, if any.
            auto tes = [&p]() -> std::vector<t_event_t> {
                if constexpr (p.has(kw::t_events)) {
                    return std::forward<decltype(p(kw::t_events))>(p(kw::t_events));
                } else {
                    return {};
                }
            }();

            // Extract the non-terminal events, if any.
            auto ntes = [&p]() -> std::vector<nt_event_t> {
                if constexpr (p.has(kw::nt_events)) {
                    return std::forward<decltype(p(kw::nt_events))>(p(kw::nt_events));
                } else {
                    return {};
                }
            }();

            finalise_ctor_impl(sys, std::move(state), time, tol, high_accuracy, compact_mode, std::move(pars),
                               std::move(tes), std::move(ntes));
        }
    }

public:
    taylor_adaptive_impl();

    template <typename... KwArgs>
    explicit taylor_adaptive_impl(const std::vector<expression> &sys, std::vector<T> state, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_impl(const std::vector<std::pair<expression, expression>> &sys, std::vector<T> state,
                                  KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), std::forward<KwArgs>(kw_args)...);
    }

    taylor_adaptive_impl(const taylor_adaptive_impl &);
    taylor_adaptive_impl(taylor_adaptive_impl &&) noexcept;

    taylor_adaptive_impl &operator=(const taylor_adaptive_impl &);
    taylor_adaptive_impl &operator=(taylor_adaptive_impl &&) noexcept;

    ~taylor_adaptive_impl();

    const llvm_state &get_llvm_state() const;

    const taylor_dc_t &get_decomposition() const;

    std::uint32_t get_order() const;
    T get_tol() const;
    std::uint32_t get_dim() const;

    T get_time() const
    {
        return static_cast<T>(m_time);
    }
    void set_time(T t)
    {
        m_time = dfloat<T>(t);
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

    T get_last_h() const
    {
        return m_last_h;
    }

    const std::vector<T> &get_d_output() const
    {
        return m_d_out;
    }
    const std::vector<T> &update_d_output(T, bool = false);

    bool with_events() const
    {
        return static_cast<bool>(m_ed_data);
    }
    void reset_cooldowns();
    const std::vector<t_event_t> &get_t_events() const
    {
        if (!m_ed_data) {
            throw std::invalid_argument("No events were defined for this integrator");
        }

        return m_ed_data->m_tes;
    }
    const auto &get_te_cooldowns() const
    {
        if (!m_ed_data) {
            throw std::invalid_argument("No events were defined for this integrator");
        }

        return m_ed_data->m_te_cooldowns;
    }
    const std::vector<nt_event_t> &get_nt_events() const
    {
        if (!m_ed_data) {
            throw std::invalid_argument("No events were defined for this integrator");
        }

        return m_ed_data->m_ntes;
    }

    std::tuple<taylor_outcome, T> step(bool = false);
    std::tuple<taylor_outcome, T> step_backward(bool = false);
    std::tuple<taylor_outcome, T> step(T, bool = false);

private:
    // Parser for the common kwargs options for the propagate_*() functions.
    template <typename... KwArgs>
    static auto propagate_common_ops(KwArgs &&...kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>, "The variadic arguments to a propagate_*() function in an "
                                                             "adaptive Taylor integrator contain unnamed arguments.");
            throw;
        } else {
            // Max number of steps (defaults to zero).
            auto max_steps = [&p]() -> std::size_t {
                if constexpr (p.has(kw::max_steps)) {
                    return std::forward<decltype(p(kw::max_steps))>(p(kw::max_steps));
                } else {
                    return 0;
                }
            }();

            // Max delta_t (defaults to positive infinity).
            auto max_delta_t = [&p]() -> T {
                if constexpr (p.has(kw::max_delta_t)) {
                    return std::forward<decltype(p(kw::max_delta_t))>(p(kw::max_delta_t));
                } else {
                    return std::numeric_limits<T>::infinity();
                }
            }();

            // Callback (defaults to empty).
            auto cb = [&p]() -> std::function<bool(taylor_adaptive_impl &)> {
                if constexpr (p.has(kw::callback)) {
                    return std::forward<decltype(p(kw::callback))>(p(kw::callback));
                } else {
                    return {};
                }
            }();

            // Write the Taylor coefficients (defaults to false).
            // NOTE: this won't be used in propagate_grid().
            auto write_tc = [&p]() -> bool {
                if constexpr (p.has(kw::write_tc)) {
                    return std::forward<decltype(p(kw::write_tc))>(p(kw::write_tc));
                } else {
                    return false;
                }
            }();

            return std::tuple{max_steps, max_delta_t, std::move(cb), write_tc};
        }
    }

    // Implementations of the propagate_*() functions.
    std::tuple<taylor_outcome, T, T, std::size_t>
    propagate_until_impl(const dfloat<T> &, std::size_t, T, std::function<bool(taylor_adaptive_impl &)>, bool);
    std::tuple<taylor_outcome, T, T, std::size_t, std::vector<T>>
    propagate_grid_impl(const std::vector<T> &, std::size_t, T, std::function<bool(taylor_adaptive_impl &)>);

public:
    // NOTE: return values:
    // - outcome,
    // - min abs(timestep),
    // - max abs(timestep),
    // - total number of nonzero steps
    //   successfully undertaken,
    // - grid of state vectors (only for propagate_grid()).
    // NOTE: the min/max timesteps are well-defined
    // only if at least 1-2 steps were taken successfully.
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t> propagate_until(T t, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_t, cb, write_tc] = propagate_common_ops(std::forward<KwArgs>(kw_args)...);

        return propagate_until_impl(dfloat<T>(t), max_steps, max_delta_t, std::move(cb), write_tc);
    }
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t> propagate_for(T delta_t, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_t, cb, write_tc] = propagate_common_ops(std::forward<KwArgs>(kw_args)...);

        return propagate_until_impl(m_time + delta_t, max_steps, max_delta_t, std::move(cb), write_tc);
    }
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, std::vector<T>> propagate_grid(const std::vector<T> &grid,
                                                                                 KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_t, cb, _] = propagate_common_ops(std::forward<KwArgs>(kw_args)...);

        return propagate_grid_impl(grid, max_steps, max_delta_t, std::move(cb));
    }
};

} // namespace detail

template <typename T>
using taylor_adaptive = detail::taylor_adaptive_impl<T>;

namespace detail
{

template <typename T>
class HEYOKA_DLL_PUBLIC taylor_adaptive_batch_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using nt_event_t = nt_batch_event<T>;
    using t_event_t = t_batch_event<T>;

private:
    // Struct implementing the data/logic for event detection.
    struct HEYOKA_DLL_PUBLIC ed_data {
        // The working list type used during real root isolation.
        using wlist_t = std::vector<std::tuple<T, T, taylor_pwrap<T>>>;
        // The type used to store the list of isolating intervals.
        using isol_t = std::vector<std::tuple<T, T>>;
        // Polynomial translation function type.
        using pt_t = void (*)(T *, const T *);
        // rtscc function type.
        using rtscc_t = void (*)(T *, T *, std::uint32_t *, const T *);
        // fex_check function type.
        using fex_check_t = void (*)(const T *, const T *, const std::uint32_t *, std::uint32_t *);

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
        std::vector<std::vector<std::tuple<std::uint32_t, T, bool, int, T>>> m_d_tes;
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
        // The working list.
        wlist_t m_wlist;
        // The list of isolating intervals.
        isol_t m_isol;
        // The polynomial cache.
        taylor_poly_cache<T> m_poly_cache;

        // Constructors.
        ed_data(std::vector<t_event_t>, std::vector<nt_event_t>, std::uint32_t, std::uint32_t, std::uint32_t);
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
    std::uint32_t m_batch_size;
    // State vectors.
    std::vector<T> m_state;
    // Times.
    std::vector<T> m_time_hi, m_time_lo;
    // The LLVM machinery.
    llvm_state m_llvm;
    // Dimension of the system.
    std::uint32_t m_dim;
    // Taylor decomposition.
    taylor_dc_t m_dc;
    // Taylor order.
    std::uint32_t m_order;
    // Tolerance.
    T m_tol;
    // The steppers.
    using step_f_t = void (*)(T *, const T *, const T *, T *, T *);
    using step_f_e_t = void (*)(T *, const T *, const T *, const T *, T *, T *);
    std::variant<step_f_t, step_f_e_t> m_step_f;
    // The vector of parameters.
    std::vector<T> m_pars;
    // The vector for the Taylor coefficients.
    std::vector<T> m_tc;
    // The sizes of the last timesteps taken.
    std::vector<T> m_last_h;
    // The function for computing the dense output.
    using d_out_f_t = void (*)(T *, const T *, const T *);
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
    // Temporary vectors used in the propagate_*() implementations.
    std::vector<std::size_t> m_ts_count;
    std::vector<T> m_min_abs_h, m_max_abs_h;
    std::vector<T> m_cur_max_delta_ts;
    std::vector<dfloat<T>> m_pfor_ts;
    std::vector<int> m_t_dir;
    std::vector<dfloat<T>> m_rem_time;
    // Temporary vector used in the dense output implementation.
    std::vector<T> m_d_out_time;
    // Auxiliary data/functions for event detection.
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
    template <typename U>
    HEYOKA_DLL_PUBLIC void finalise_ctor_impl(const U &, std::vector<T>, std::uint32_t, std::vector<T>, T, bool, bool,
                                              std::vector<T>, std::vector<t_event_t>, std::vector<nt_event_t>);
    template <typename U, typename... KwArgs>
    void finalise_ctor(const U &sys, std::vector<T> state, std::uint32_t batch_size, KwArgs &&...kw_args)
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

            // Extract the terminal events, if any.
            auto tes = [&p]() -> std::vector<t_event_t> {
                if constexpr (p.has(kw::t_events)) {
                    return std::forward<decltype(p(kw::t_events))>(p(kw::t_events));
                } else {
                    return {};
                }
            }();

            // Extract the non-terminal events, if any.
            auto ntes = [&p]() -> std::vector<nt_event_t> {
                if constexpr (p.has(kw::nt_events)) {
                    return std::forward<decltype(p(kw::nt_events))>(p(kw::nt_events));
                } else {
                    return {};
                }
            }();

            finalise_ctor_impl(sys, std::move(state), batch_size, std::move(time), tol, high_accuracy, compact_mode,
                               std::move(pars), std::move(tes), std::move(ntes));
        }
    }

public:
    taylor_adaptive_batch_impl();

    template <typename... KwArgs>
    explicit taylor_adaptive_batch_impl(const std::vector<expression> &sys, std::vector<T> state,
                                        std::uint32_t batch_size, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), batch_size, std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch_impl(const std::vector<std::pair<expression, expression>> &sys, std::vector<T> state,
                                        std::uint32_t batch_size, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), batch_size, std::forward<KwArgs>(kw_args)...);
    }

    taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &);
    taylor_adaptive_batch_impl(taylor_adaptive_batch_impl &&) noexcept;

    taylor_adaptive_batch_impl &operator=(const taylor_adaptive_batch_impl &);
    taylor_adaptive_batch_impl &operator=(taylor_adaptive_batch_impl &&) noexcept;

    ~taylor_adaptive_batch_impl();

    const llvm_state &get_llvm_state() const;

    const taylor_dc_t &get_decomposition() const;

    std::uint32_t get_batch_size() const;
    std::uint32_t get_order() const;
    T get_tol() const;
    std::uint32_t get_dim() const;

    const std::vector<T> &get_time() const
    {
        return m_time_hi;
    }
    const T *get_time_data() const
    {
        return m_time_hi.data();
    }
    void set_time(const std::vector<T> &);

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

    const std::vector<T> &get_last_h() const
    {
        return m_last_h;
    }

    const std::vector<T> &get_d_output() const
    {
        return m_d_out;
    }
    const std::vector<T> &update_d_output(const std::vector<T> &, bool = false);

    void reset_cooldowns();
    void reset_cooldowns(std::uint32_t);

    void step(bool = false);
    void step_backward(bool = false);
    void step(const std::vector<T> &, bool = false);
    const std::vector<std::tuple<taylor_outcome, T>> &get_step_res() const
    {
        return m_step_res;
    }

private:
    // Parser for the common kwargs options for the propagate_*() functions.
    template <typename... KwArgs>
    auto propagate_common_ops(KwArgs &&...kw_args) const
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments to a propagate_*() function in an "
                          "adaptive Taylor integrator in batch mode contain unnamed arguments.");
            throw;
        } else {
            // Max number of steps (defaults to zero).
            auto max_steps = [&p]() -> std::size_t {
                if constexpr (p.has(kw::max_steps)) {
                    return std::forward<decltype(p(kw::max_steps))>(p(kw::max_steps));
                } else {
                    return 0;
                }
            }();

            // Max delta_t (defaults to empty vector).
            auto max_delta_t = [&p]() {
                if constexpr (p.has(kw::max_delta_t)) {
                    if constexpr (std::is_same_v<std::vector<T>, uncvref_t<decltype(p(kw::max_delta_t))>>) {
                        return std::reference_wrapper<const std::vector<T>>(
                            static_cast<const std::vector<T> &>(p(kw::max_delta_t)));
                    } else {
                        return std::vector<T>(std::forward<decltype(p(kw::max_delta_t))>(p(kw::max_delta_t)));
                    }
                } else {
                    return std::vector<T>{};
                }
            }();

            // Callback (defaults to empty).
            auto cb = [&p]() -> std::function<bool(taylor_adaptive_batch_impl &)> {
                if constexpr (p.has(kw::callback)) {
                    return std::forward<decltype(p(kw::callback))>(p(kw::callback));
                } else {
                    return {};
                }
            }();

            // Write the Taylor coefficients (defaults to false).
            // NOTE: this won't be used in propagate_grid().
            auto write_tc = [&p]() -> bool {
                if constexpr (p.has(kw::write_tc)) {
                    return std::forward<decltype(p(kw::write_tc))>(p(kw::write_tc));
                } else {
                    return false;
                }
            }();

            // NOTE: use make_tuple so that max_delta_t is transformed
            // into a reference if it is a reference wrapper.
            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb), write_tc);
        }
    }

    // Implementations of the propagate_*() functions.
    HEYOKA_DLL_LOCAL void propagate_until_impl(const std::vector<dfloat<T>> &, std::size_t, const std::vector<T> &,
                                               std::function<bool(taylor_adaptive_batch_impl &)>, bool);
    void propagate_until_impl(const std::vector<T> &, std::size_t, const std::vector<T> &,
                              std::function<bool(taylor_adaptive_batch_impl &)>, bool);
    void propagate_for_impl(const std::vector<T> &, std::size_t, const std::vector<T> &,
                            std::function<bool(taylor_adaptive_batch_impl &)>, bool);
    std::vector<T> propagate_grid_impl(const std::vector<T> &, std::size_t, const std::vector<T> &,
                                       std::function<bool(taylor_adaptive_batch_impl &)>);

public:
    template <typename... KwArgs>
    void propagate_until(const std::vector<T> &ts, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc] = propagate_common_ops(std::forward<KwArgs>(kw_args)...);

        propagate_until_impl(ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb), write_tc);
    }
    template <typename... KwArgs>
    void propagate_for(const std::vector<T> &ts, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc] = propagate_common_ops(std::forward<KwArgs>(kw_args)...);

        propagate_for_impl(ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb), write_tc);
    }
    template <typename... KwArgs>
    std::vector<T> propagate_grid(const std::vector<T> &grid, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, _] = propagate_common_ops(std::forward<KwArgs>(kw_args)...);

        return propagate_grid_impl(grid, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, std::move(cb));
    }
    const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &get_propagate_res() const
    {
        return m_prop_res;
    }
};

} // namespace detail

template <typename T>
using taylor_adaptive_batch = detail::taylor_adaptive_batch_impl<T>;

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

// NOTE: copy the implementation of the BOOST_CLASS_VERSION macro, as it does
// not support class templates.
namespace boost::serialization
{

template <typename T>
struct version<heyoka::taylor_adaptive<T>> {
    typedef mpl::int_<1> type;
    typedef mpl::integral_c_tag tag;
    BOOST_STATIC_CONSTANT(int, value = version::type::value);
    BOOST_MPL_ASSERT((boost::mpl::less<boost::mpl::int_<1>, boost::mpl::int_<256>>));
};

template <typename T>
struct version<heyoka::taylor_adaptive_batch<T>> {
    typedef mpl::int_<1> type;
    typedef mpl::integral_c_tag tag;
    BOOST_STATIC_CONSTANT(int, value = version::type::value);
    BOOST_MPL_ASSERT((boost::mpl::less<boost::mpl::int_<1>, boost::mpl::int_<256>>));
};

} // namespace boost::serialization

#endif
