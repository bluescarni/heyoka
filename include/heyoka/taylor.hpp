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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/fmt_compat.hpp>
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
HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet(llvm_state &, const std::string &, const std::vector<expression> &,
                                             std::uint32_t, std::uint32_t, bool, bool,
                                             const std::vector<expression> & = {}, bool = false, long long = 0);

template <typename>
HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet(llvm_state &, const std::string &,
                                             const std::vector<std::pair<expression, expression>> &, std::uint32_t,
                                             std::uint32_t, bool, bool, const std::vector<expression> & = {},
                                             bool = false, long long = 0);

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

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, event_direction);

HEYOKA_END_NAMESPACE

// fmt formatters for taylor_outcome and event_direction, implemented
// on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::taylor_outcome> : heyoka::detail::ostream_formatter {
};

template <>
struct formatter<heyoka::event_direction> : heyoka::detail::ostream_formatter {
};

} // namespace fmt

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
namespace boost::serialization
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

    // NOTE: this is a right fold, which, in conjunction with the
    // builtin comma operator, ensures that the serialisation of
    // the tuple elements proceeds in the correct order and with
    // the correct sequencing.
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

} // namespace boost::serialization

HEYOKA_BEGIN_NAMESPACE

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(tol);
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
IGOR_MAKE_NAMED_ARGUMENT(c_output);

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

    // tol (defaults to undefined). Zero tolerance is considered
    // the same as undefined.
    auto tol = [&p]() -> std::optional<T> {
        if constexpr (p.has(kw::tol)) {
            auto retval = static_cast<T>(std::forward<decltype(p(kw::tol))>(p(kw::tol)));
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
            return std::forward<decltype(p(kw::compact_mode))>(p(kw::compact_mode));
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
            return std::forward<decltype(p(kw::pars))>(p(kw::pars));
        } else {
            return {};
        }
    }();

    // Parallel mode (defaults to false).
    auto parallel_mode = [&p]() -> bool {
        if constexpr (p.has(kw::parallel_mode)) {
            return std::forward<decltype(p(kw::parallel_mode))>(p(kw::parallel_mode));
        } else {
            return false;
        }
    }();

    return std::tuple{high_accuracy, std::move(tol), compact_mode, std::move(pars), parallel_mode};
}

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC nt_event_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using callback_t = callable<std::conditional_t<B, void(taylor_adaptive_batch<T> &, T, int, std::uint32_t),
                                                   void(taylor_adaptive<T> &, T, int)>>;

private:
    expression eq;
    callback_t callback;
    event_direction dir = event_direction::any;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & eq;
        ar & callback;
        ar & dir;
    }

    void finalise_ctor(event_direction);

public:
    nt_event_impl();

    template <typename... KwArgs>
    explicit nt_event_impl(expression e, callback_t cb, KwArgs &&...kw_args) : eq(std::move(e)), callback(std::move(cb))
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

    [[nodiscard]] const expression &get_expression() const;
    const callback_t &get_callback() const;
    [[nodiscard]] event_direction get_direction() const;
};

template <typename T, bool B>
inline std::ostream &operator<<(std::ostream &os, const nt_event_impl<T, B> &)
{
    static_assert(always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<float, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<float, true> &);

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

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const nt_event_impl<mppp::real, false> &);

#endif

template <typename T, bool B>
class HEYOKA_DLL_PUBLIC t_event_impl
{
    static_assert(is_supported_fp_v<T>, "Unhandled type.");

public:
    using callback_t = callable<std::conditional_t<B, bool(taylor_adaptive_batch<T> &, bool, int, std::uint32_t),
                                                   bool(taylor_adaptive<T> &, bool, int)>>;

private:
    expression eq;
    callback_t callback;
    T cooldown = 0;
    event_direction dir = event_direction::any;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & eq;
        ar & callback;
        ar & cooldown;
        ar & dir;
    }

    void finalise_ctor(callback_t, T, event_direction);

public:
    t_event_impl();

    template <typename... KwArgs>
    explicit t_event_impl(expression e, KwArgs &&...kw_args) : eq(std::move(e))
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

    [[nodiscard]] const expression &get_expression() const;
    const callback_t &get_callback() const;
    [[nodiscard]] event_direction get_direction() const;
    T get_cooldown() const;
};

template <typename T, bool B>
inline std::ostream &operator<<(std::ostream &os, const t_event_impl<T, B> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<float, false> &);
template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<float, true> &);

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

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const t_event_impl<mppp::real, false> &);

#endif

} // namespace detail

template <typename T>
using nt_event = detail::nt_event_impl<T, false>;

template <typename T>
using t_event = detail::t_event_impl<T, false>;

template <typename T>
using nt_event_batch = detail::nt_event_impl<T, true>;

template <typename T>
using t_event_batch = detail::t_event_impl<T, true>;

template <typename>
class HEYOKA_DLL_PUBLIC continuous_output;

namespace detail
{

template <typename T>
std::ostream &c_out_stream_impl(std::ostream &, const continuous_output<T> &);

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC continuous_output
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");

    template <typename>
    friend class HEYOKA_DLL_PUBLIC taylor_adaptive;

    friend std::ostream &detail::c_out_stream_impl<T>(std::ostream &, const continuous_output<T> &);

    llvm_state m_llvm_state;
    std::vector<T> m_tcs;
    std::vector<T> m_times_hi, m_times_lo;
    std::vector<T> m_output;
    using fptr_t = void (*)(T *, T *, const T *, const T *, const T *) noexcept;
    fptr_t m_f_ptr = nullptr;

    HEYOKA_DLL_LOCAL void add_c_out_function(std::uint32_t, std::uint32_t, bool);
    void call_impl(T);

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    continuous_output();
    explicit continuous_output(llvm_state &&);
    continuous_output(const continuous_output &);
    continuous_output(continuous_output &&) noexcept;
    ~continuous_output();

    continuous_output &operator=(const continuous_output &);
    continuous_output &operator=(continuous_output &&) noexcept;

    [[nodiscard]] const llvm_state &get_llvm_state() const;

    const std::vector<T> &operator()(T);
    const std::vector<T> &get_output() const;
    const std::vector<T> &get_times() const;
    const std::vector<T> &get_tcs() const;

    std::pair<T, T> get_bounds() const;
    [[nodiscard]] std::size_t get_n_steps() const;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const continuous_output<T> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output<float> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output<double> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output<mppp::real128> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output<mppp::real> &);

#endif

template <typename>
class HEYOKA_DLL_PUBLIC continuous_output_batch;

namespace detail
{

template <typename T>
std::ostream &c_out_batch_stream_impl(std::ostream &, const continuous_output_batch<T> &);

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC continuous_output_batch
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");

    template <typename>
    friend class HEYOKA_DLL_PUBLIC taylor_adaptive_batch;

    friend std::ostream &detail::c_out_batch_stream_impl<T>(std::ostream &, const continuous_output_batch<T> &);

    std::uint32_t m_batch_size = 0;
    llvm_state m_llvm_state;
    std::vector<T> m_tcs;
    std::vector<T> m_times_hi, m_times_lo;
    std::vector<T> m_output;
    std::vector<T> m_tmp_tm;
    using fptr_t = void (*)(T *, const T *, const T *, const T *, const T *) noexcept;
    fptr_t m_f_ptr = nullptr;

    HEYOKA_DLL_LOCAL void add_c_out_function(std::uint32_t, std::uint32_t, bool);
    void call_impl(const T *);

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    continuous_output_batch();
    explicit continuous_output_batch(llvm_state &&);
    continuous_output_batch(const continuous_output_batch &);
    continuous_output_batch(continuous_output_batch &&) noexcept;
    ~continuous_output_batch();

    continuous_output_batch &operator=(const continuous_output_batch &);
    continuous_output_batch &operator=(continuous_output_batch &&) noexcept;

    [[nodiscard]] const llvm_state &get_llvm_state() const;

    const std::vector<T> &operator()(const T *);
    const std::vector<T> &operator()(const std::vector<T> &);
    const std::vector<T> &operator()(T);

    const std::vector<T> &get_output() const;
    // NOTE: when documenting this function,
    // we need to warn about the padding.
    const std::vector<T> &get_times() const;
    const std::vector<T> &get_tcs() const;
    [[nodiscard]] std::uint32_t get_batch_size() const;

    std::pair<std::vector<T>, std::vector<T>> get_bounds() const;
    [[nodiscard]] std::size_t get_n_steps() const;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const continuous_output_batch<T> &)
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return os;
}

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output_batch<float> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output_batch<double> &);

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output_batch<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const continuous_output_batch<mppp::real128> &);

#endif

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

// Small helper to construct a default value for the max_delta_t
// keyword argument.
template <typename T>
HEYOKA_DLL_PUBLIC T taylor_default_max_delta_t();

// Parser for the common kwargs options for the propagate_*() functions.
template <typename T, bool Grid, typename... KwArgs>
inline auto taylor_propagate_common_ops(KwArgs &&...kw_args)
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
                return taylor_default_max_delta_t<T>();
            }
        }();

        // Callback (defaults to empty). If a step_callback with the correct
        // signature is passed as argument, return a reference wrapper to it
        // in order to avoid a useless copy.
        // NOTE: eventually here we could check if the user passed in a range
        // of elements which are (convertible to) step_callback, and automatically
        // build a step_callback_set from them.
        auto cb = [&p]() {
            using cb_func_t = step_callback<T>;

            if constexpr (p.has(kw::callback)) {
                if constexpr (std::is_same_v<uncvref_t<decltype(p(kw::callback))>, cb_func_t>) {
                    return std::ref(p(kw::callback));
                } else {
                    return cb_func_t(std::forward<decltype(p(kw::callback))>(p(kw::callback)));
                }
            } else {
                return cb_func_t{};
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

        // NOTE: use std::make_tuple() so that if cb is a reference wrapper, it is turned
        // into a reference tuple element.
        if constexpr (Grid) {
            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb), write_tc);
        } else {
            // Continuous output (defaults to false).
            auto with_c_out = [&p]() -> bool {
                if constexpr (p.has(kw::c_output)) {
                    return std::forward<decltype(p(kw::c_output))>(p(kw::c_output));
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
class HEYOKA_DLL_PUBLIC taylor_adaptive_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

#if defined(HEYOKA_HAVE_REAL)

template <typename Derived>
class HEYOKA_DLL_PUBLIC taylor_adaptive_base<mppp::real, Derived>
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_prec;
    }

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
class HEYOKA_DLL_PUBLIC taylor_adaptive : public detail::taylor_adaptive_base<T, taylor_adaptive<T>>
{
    static_assert(detail::is_supported_fp_v<T>, "Unhandled type.");
    friend class HEYOKA_DLL_PUBLIC detail::taylor_adaptive_base<T, taylor_adaptive<T>>;
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
    // NOTE: apparently on Windows we need to re-iterate
    // here that this is going to be dll-exported.
    template <typename U>
    HEYOKA_DLL_PUBLIC void finalise_ctor_impl(const U &, std::vector<T>, std::optional<T>, std::optional<T>, bool, bool,
                                              std::vector<T>, std::vector<t_event_t>, std::vector<nt_event_t>, bool,
                                              std::optional<long long>);
    template <typename U, typename... KwArgs>
    void finalise_ctor(const U &sys, std::vector<T> state, KwArgs &&...kw_args)
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
                    return std::forward<decltype(p(kw::time))>(p(kw::time));
                } else {
                    return {};
                }
            }();

            auto [high_accuracy, tol, compact_mode, pars, parallel_mode]
                = detail::taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

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

            // Fetch the precision, if provided. Zero precision
            // is considered the same as undefined.
            auto prec = [&p]() -> std::optional<long long> {
                if constexpr (p.has(kw::prec)) {
                    auto ret = static_cast<long long>(std::forward<decltype(p(kw::prec))>(p(kw::prec)));
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
    explicit taylor_adaptive(const std::vector<expression> &sys, std::vector<T> state, KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive(const std::vector<expression> &sys, std::initializer_list<T> state, KwArgs &&...kw_args)
        : taylor_adaptive(sys, std::vector<T>(state), std::forward<KwArgs>(kw_args)...)
    {
    }
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

    T get_time() const;
    void set_time(T);

    // Time set/get in double-length format.
    std::pair<T, T> get_dtime() const;
    void set_dtime(T, T);

    const std::vector<T> &get_state() const;
    const T *get_state_data() const;
    T *get_state_data();

    const std::vector<T> &get_pars() const;
    const T *get_pars_data() const;
    T *get_pars_data();

    const std::vector<T> &get_tc() const;

    T get_last_h() const;

    const std::vector<T> &get_d_output() const;
    const std::vector<T> &update_d_output(T, bool = false);

    [[nodiscard]] bool with_events() const;
    void reset_cooldowns();
    const std::vector<t_event_t> &get_t_events() const;
    const std::vector<std::optional<std::pair<T, T>>> &get_te_cooldowns() const;
    const std::vector<nt_event_t> &get_nt_events() const;

    [[nodiscard]] const std::vector<expression> &get_state_vars() const;
    [[nodiscard]] const std::vector<expression> &get_rhs() const;

    std::tuple<taylor_outcome, T> step(bool = false);
    std::tuple<taylor_outcome, T> step_backward(bool = false);
    std::tuple<taylor_outcome, T> step(T, bool = false);

private:
    // Implementations of the propagate_*() functions.
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>
    propagate_until_impl(detail::dfloat<T>, std::size_t, T, step_callback<T> &, bool, bool);
    std::tuple<taylor_outcome, T, T, std::size_t, std::vector<T>> propagate_grid_impl(std::vector<T>, std::size_t, T,
                                                                                      step_callback<T> &);

public:
    // NOTE: return values:
    // - outcome,
    // - min abs(timestep),
    // - max abs(timestep),
    // - total number of nonzero steps
    //   successfully undertaken,
    // - grid of state vectors (only for propagate_grid()),
    // - continuous output, if requested (only for propagate_for/until()).
    // NOTE: the min/max timesteps are well-defined
    // only if at least 1-2 steps were taken successfully.
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>
    propagate_until(T t, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops<T, false>(std::forward<KwArgs>(kw_args)...);

        return propagate_until_impl(detail::dfloat<T>(std::move(t)), max_steps, std::move(max_delta_t), cb, write_tc,
                                    with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>
    propagate_for(T delta_t, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_t, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops<T, false>(std::forward<KwArgs>(kw_args)...);

        return propagate_until_impl(m_time + std::move(delta_t), max_steps, std::move(max_delta_t), cb, write_tc,
                                    with_c_out);
    }
    template <typename... KwArgs>
    std::tuple<taylor_outcome, T, T, std::size_t, std::vector<T>> propagate_grid(std::vector<T> grid,
                                                                                 KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_t, cb, _]
            = detail::taylor_propagate_common_ops<T, true>(std::forward<KwArgs>(kw_args)...);

        return propagate_grid_impl(std::move(grid), max_steps, std::move(max_delta_t), cb);
    }
};

namespace detail
{

// Parser for the common kwargs options for the propagate_*() functions
// for the batch integrator.
template <typename T, bool Grid, bool ForceScalarMaxDeltaT, typename... KwArgs>
inline auto taylor_propagate_common_ops_batch(std::uint32_t batch_size, KwArgs &&...kw_args)
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
                return std::forward<decltype(p(kw::max_steps))>(p(kw::max_steps));
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
                using type = uncvref_t<decltype(p(kw::max_delta_t))>;

                if constexpr (is_any_vector_v<type> || is_any_ilist_v<type>) {
                    if constexpr (ForceScalarMaxDeltaT) {
                        // LCOV_EXCL_START
                        static_assert(always_false_v<T>,
                                      "In ensemble integrations, max_delta_t must always be passed as a scalar.");

                        throw;
                        // LCOV_EXCL_STOP
                    } else {
                        return std::forward<decltype(p(kw::max_delta_t))>(p(kw::max_delta_t));
                    }
                } else {
                    // Interpret as a scalar to be splatted.
                    return std::vector<T>(boost::numeric_cast<typename std::vector<T>::size_type>(batch_size),
                                          p(kw::max_delta_t));
                }
            } else {
                return {};
            }
        }();

        // Callback (defaults to empty). If a step_callback with the correct
        // signature is passed as argument, return a reference wrapper to it
        // in order to avoid a useless copy.
        auto cb = [&p]() {
            using cb_func_t = step_callback_batch<T>;

            if constexpr (p.has(kw::callback)) {
                if constexpr (std::is_same_v<uncvref_t<decltype(p(kw::callback))>, cb_func_t>) {
                    return std::ref(p(kw::callback));
                } else {
                    return cb_func_t(std::forward<decltype(p(kw::callback))>(p(kw::callback)));
                }
            } else {
                return cb_func_t{};
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

        // NOTE: use std::make_tuple() so that if cb is a reference wrapper, it is turned
        // into a reference tuple element.
        if constexpr (Grid) {
            return std::make_tuple(max_steps, std::move(max_delta_t), std::move(cb), write_tc);
        } else {
            // Continuous output (defaults to false).
            auto with_c_out = [&p]() -> bool {
                if constexpr (p.has(kw::c_output)) {
                    return std::forward<decltype(p(kw::c_output))>(p(kw::c_output));
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
class HEYOKA_DLL_PUBLIC taylor_adaptive_batch
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
    template <typename U>
    HEYOKA_DLL_PUBLIC void finalise_ctor_impl(const U &, std::vector<T>, std::uint32_t, std::vector<T>,
                                              std::optional<T>, bool, bool, std::vector<T>, std::vector<t_event_t>,
                                              std::vector<nt_event_t>, bool);
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
            auto tm = [&p, batch_size]() -> std::vector<T> {
                if constexpr (p.has(kw::time)) {
                    return std::forward<decltype(p(kw::time))>(p(kw::time));
                } else {
                    return std::vector<T>(static_cast<typename std::vector<T>::size_type>(batch_size), T(0));
                }
            }();

            auto [high_accuracy, tol, compact_mode, pars, parallel_mode]
                = detail::taylor_adaptive_common_ops<T>(std::forward<KwArgs>(kw_args)...);

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

            finalise_ctor_impl(sys, std::move(state), batch_size, std::move(tm), std::move(tol), high_accuracy,
                               compact_mode, std::move(pars), std::move(tes), std::move(ntes), parallel_mode);
        }
    }

public:
    taylor_adaptive_batch();

    template <typename... KwArgs>
    explicit taylor_adaptive_batch(const std::vector<expression> &sys, std::vector<T> state, std::uint32_t batch_size,
                                   KwArgs &&...kw_args)
        : m_llvm{std::forward<KwArgs>(kw_args)...}
    {
        finalise_ctor(sys, std::move(state), batch_size, std::forward<KwArgs>(kw_args)...);
    }
    template <typename... KwArgs>
    explicit taylor_adaptive_batch(const std::vector<expression> &sys, std::initializer_list<T> state,
                                   std::uint32_t batch_size, KwArgs &&...kw_args)
        : taylor_adaptive_batch(sys, std::vector<T>(state), batch_size, std::forward<KwArgs>(kw_args)...)
    {
    }
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
    T get_tol() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] bool get_compact_mode() const;
    [[nodiscard]] std::uint32_t get_dim() const;

    const std::vector<T> &get_time() const;
    const T *get_time_data() const;
    void set_time(const std::vector<T> &);
    void set_time(T);

    // Time set/get in double-length format.
    std::pair<const std::vector<T> &, const std::vector<T> &> get_dtime() const;
    std::pair<const T *, const T *> get_dtime_data() const;
    void set_dtime(const std::vector<T> &, const std::vector<T> &);
    void set_dtime(T, T);

    const std::vector<T> &get_state() const;
    const T *get_state_data() const;
    T *get_state_data();

    const std::vector<T> &get_pars() const;
    const T *get_pars_data() const;
    T *get_pars_data();

    const std::vector<T> &get_tc() const;

    const std::vector<T> &get_last_h() const;

    const std::vector<T> &get_d_output() const;
    const std::vector<T> &update_d_output(const std::vector<T> &, bool = false);
    const std::vector<T> &update_d_output(T, bool = false);

    [[nodiscard]] bool with_events() const;
    void reset_cooldowns();
    void reset_cooldowns(std::uint32_t);
    const std::vector<t_event_t> &get_t_events() const;
    const std::vector<std::vector<std::optional<std::pair<T, T>>>> &get_te_cooldowns() const;
    const std::vector<nt_event_t> &get_nt_events() const;

    [[nodiscard]] const std::vector<expression> &get_state_vars() const;
    [[nodiscard]] const std::vector<expression> &get_rhs() const;

    void step(bool = false);
    void step_backward(bool = false);
    void step(const std::vector<T> &, bool = false);
    const std::vector<std::tuple<taylor_outcome, T>> &get_step_res() const;

private:
    // Implementations of the propagate_*() functions.
    std::optional<continuous_output_batch<T>> propagate_until_impl(const std::vector<detail::dfloat<T>> &, std::size_t,
                                                                   const std::vector<T> &, step_callback_batch<T> &,
                                                                   bool, bool);
    std::optional<continuous_output_batch<T>> propagate_until_impl(const std::vector<T> &, std::size_t,
                                                                   const std::vector<T> &, step_callback_batch<T> &,
                                                                   bool, bool);
    std::optional<continuous_output_batch<T>> propagate_for_impl(const std::vector<T> &, std::size_t,
                                                                 const std::vector<T> &, step_callback_batch<T> &, bool,
                                                                 bool);
    std::vector<T> propagate_grid_impl(const std::vector<T> &, std::size_t, const std::vector<T> &,
                                       step_callback_batch<T> &);

public:
    // NOTE: in propagate_for/until(), we can take 'ts' as const reference because it is always
    // only and immediately used to set up the internal m_pfor_ts member (which is not visible
    // from outside). Hence, even if 'ts' aliases some public integrator data, it does not matter.
    template <typename... KwArgs>
    std::optional<continuous_output_batch<T>> propagate_until(const std::vector<T> &ts, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size,
                                                                         std::forward<KwArgs>(kw_args)...);

        return propagate_until_impl(ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, cb, write_tc,
                                    with_c_out); // LCOV_EXCL_LINE
    }
    template <typename... KwArgs>
    std::optional<continuous_output_batch<T>> propagate_until(T t, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size,
                                                                         std::forward<KwArgs>(kw_args)...);

        // NOTE: re-use m_pfor_ts as tmp storage, as the other overload does.
        assert(m_pfor_ts.size() == m_batch_size); // LCOV_EXCL_LINE
        std::fill(m_pfor_ts.begin(), m_pfor_ts.end(), detail::dfloat<T>(t));
        return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, cb, write_tc,
                                    with_c_out); // LCOV_EXCL_LINE
    }
    template <typename... KwArgs>
    std::optional<continuous_output_batch<T>> propagate_for(const std::vector<T> &delta_ts, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size,
                                                                         std::forward<KwArgs>(kw_args)...);

        return propagate_for_impl(delta_ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, cb, write_tc,
                                  with_c_out); // LCOV_EXCL_LINE
    }
    template <typename... KwArgs>
    std::optional<continuous_output_batch<T>> propagate_for(T delta_t, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, write_tc, with_c_out]
            = detail::taylor_propagate_common_ops_batch<T, false, false>(m_batch_size,
                                                                         std::forward<KwArgs>(kw_args)...);

        // NOTE: this is a slight repetition of the other overload's code.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_pfor_ts[i] = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + delta_t;
        }
        return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, cb, write_tc,
                                    with_c_out); // LCOV_EXCL_LINE
    }
    // NOTE: grid is taken by copy because in the implementation loop we keep on reading from it.
    // Hence, we need to avoid any aliasing issue with other public integrator data.
    template <typename... KwArgs>
    std::vector<T> propagate_grid(std::vector<T> grid, KwArgs &&...kw_args)
    {
        auto [max_steps, max_delta_ts, cb, _]
            = detail::taylor_propagate_common_ops_batch<T, true, false>(m_batch_size, std::forward<KwArgs>(kw_args)...);

        return propagate_grid_impl(grid, max_steps, max_delta_ts.empty() ? m_pinf : max_delta_ts, cb);
    }
    const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &get_propagate_res() const
    {
        return m_prop_res;
    }
};

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
inline constexpr int taylor_adaptive_s11n_version = 2;

// Boost s11n class version history for taylor_adaptive_batch:
// - 1: added the m_state_vars and m_rhs members.
inline constexpr int taylor_adaptive_batch_s11n_version = 1;

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
