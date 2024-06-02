// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/continuous_output.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/ed_data.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/i_data.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

// NOTE: this is a helper macro to reduce typing when accessing the
// data members of i_data.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define HEYOKA_TAYLOR_REF_FROM_I_DATA(name) auto &name = m_i_data->name

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T, typename Derived>
void taylor_adaptive_base<T, Derived>::save(boost::archive::binary_oarchive &, unsigned) const
{
}

template <typename T, typename Derived>
void taylor_adaptive_base<T, Derived>::load(boost::archive::binary_iarchive &, unsigned)
{
}

#if defined(HEYOKA_HAVE_REAL)

template <typename Derived>
void taylor_adaptive_base<mppp::real, Derived>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_prec;
}

template <typename Derived>
void taylor_adaptive_base<mppp::real, Derived>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_prec;
}

template <typename Derived>
mpfr_prec_t taylor_adaptive_base<mppp::real, Derived>::get_prec() const
{
    assert(m_prec >= mppp::real_prec_min() && m_prec <= mppp::real_prec_max());

    return m_prec;
}

// Helper to check that the integrator data is consistent with
// the precision. To be used at the end of construciton or before using
// the integrator data (e.g., in step(), propagate(), etc.).
template <typename Derived>
void taylor_adaptive_base<mppp::real, Derived>::data_prec_check() const
{
    const auto *dthis = static_cast<const Derived *>(this);

    const auto prec = get_prec();

    // Time, tolerance, Taylor coefficients, m_last_h, dense output
    // are all supposed to maintain the original precision after construction.
    assert(dthis->m_i_data->m_time.hi.get_prec() == prec);
    assert(dthis->m_i_data->m_time.lo.get_prec() == prec);
    assert(dthis->m_i_data->m_tol.get_prec() == prec);
    assert(std::all_of(dthis->m_i_data->m_tc.begin(), dthis->m_i_data->m_tc.end(),
                       [prec](const auto &x) { return x.get_prec() == prec; }));
    assert(dthis->m_i_data->m_last_h.get_prec() == prec);
    assert(std::all_of(dthis->m_i_data->m_d_out.begin(), dthis->m_i_data->m_d_out.end(),
                       [prec](const auto &x) { return x.get_prec() == prec; }));

    // Same goes for the event detection jet data, if present.
#if !defined(NDEBUG)
    if (dthis->m_ed_data) {
        assert(std::all_of(dthis->m_ed_data->m_ev_jet.begin(), dthis->m_ed_data->m_ev_jet.end(),
                           [prec](const auto &x) { return x.get_prec() == prec; }));
    }
#endif

    // State, pars can be changed by the user and thus need to be checked.
    for (const auto &x : dthis->m_i_data->m_state) {
        if (x.get_prec() != prec) {
            throw std::invalid_argument(fmt::format("A state variable with precision {} was detected in the state "
                                                    "vector: this is incompatible with the integrator precision of {}",
                                                    x.get_prec(), prec));
        }
    }

    for (const auto &x : dthis->m_i_data->m_pars) {
        if (x.get_prec() != prec) {
            throw std::invalid_argument(fmt::format("A value with precision {} was detected in the parameter "
                                                    "vector: this is incompatible with the integrator precision of {}",
                                                    x.get_prec(), prec));
        }
    }
}

#endif

} // namespace detail

template <typename T>
taylor_adaptive<T>::taylor_adaptive(private_ctor_t, llvm_state s) : m_i_data(std::make_unique<i_data>(std::move(s)))
{
}

template <typename T>
void taylor_adaptive<T>::finalise_ctor_impl(sys_t vsys, std::vector<T> state,
                                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                            std::optional<T> time, std::optional<T> tol, bool high_accuracy,
                                            bool compact_mode, std::vector<T> pars, std::vector<t_event_t> tes,
                                            std::vector<nt_event_t> ntes, bool parallel_mode,
                                            [[maybe_unused]] std::optional<long long> prec)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_step_f);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pars);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_high_accuracy);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_compact_mode);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_llvm);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tol);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_order);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out_f);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_vsys);

    // NOTE: this must hold because tol == 0 is interpreted
    // as undefined in finalise_ctor().
    assert(!tol || *tol != 0);

#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

    using std::isfinite;

    // Are we constructing a variational integrator?
    const auto is_variational = (vsys.index() == 1u);

    // Fetch the ODE system.
    const auto &sys = is_variational ? std::get<1>(vsys).get_sys() : std::get<0>(vsys);

    // Validate it.
    // NOTE: in a variational ODE sys, we already validated the original equations,
    // and the variational equations should not be in need of validation.
    // However, *in principle*, something could have gone wrong during the computation of the
    // variational equations (i.e., a malformed gradient() implementation somewhere in a
    // user-defined function injecting variables whose names begin with "∂"). Hence,
    // just re-validate for peace of mind.
    validate_ode_sys(sys, tes, ntes);

    // Run an immediate check on state. This is a bit redundant with other checks
    // later (e.g., state.size() must be consistent with the ODE definition, which in
    // turn cannot consist of zero equations), but it's handy to do it here so that,
    // e.g., we can immediately infer the precision if T == mppp::real.
    if (state.empty()) {
        throw std::invalid_argument("Cannot initialise an adaptive integrator with an empty state vector");
    }

    // Assign the state.
    m_state = std::move(state);

#if defined(HEYOKA_HAVE_REAL)

    // Setup the precision: it is either passed by the user
    // or automatically inferred from the state vector.
    // NOTE: this must be done early so that the precision of the integrator
    // is available for other checks later.
    if constexpr (std::is_same_v<T, mppp::real>) {
        this->m_prec = prec ? boost::numeric_cast<mpfr_prec_t>(*prec) : m_state[0].get_prec();

        if (prec) {
            // If the user explicitly specifies a precision, enforce that precision
            // on the state vector.
            // NOTE: if the user specifies an invalid precision, we are sure
            // here that an exception will be thrown: m_state is not empty
            // and prec_round() will check the input precision value.
            for (auto &val : m_state) {
                // NOTE: use directly this->m_prec in order to avoid
                // triggering an assertion in get_prec() if a bogus
                // prec value was provided by the user.
                val.prec_round(this->m_prec);
            }
        } else {
            // If the precision is automatically deduced, ensure that
            // the same precision is used for all initial values.
            // NOTE: the automatically-deduced precision will be a valid one,
            // as it is taken from a valid mppp::real (i.e., m_state[0]).
            if (std::any_of(m_state.begin() + 1, m_state.end(),
                            [this](const auto &val) { return val.get_prec() != this->get_prec(); })) {
                throw std::invalid_argument(
                    fmt::format("The precision deduced automatically from the initial state vector in a multiprecision "
                                "adaptive Taylor integrator is {}, but values with different precision(s) were "
                                "detected in the state vector",
                                this->get_prec()));
            }
        }
    }

#endif

    // Check the input state size.
    // NOTE: keep track of whether or not we need to automatically setup the initial
    // conditions in a variational integrator. This is needed because we need
    // to delay the automatic ic setup for the derivatives wrt the initial time until
    // after we have correctly set up state, pars and time in the integrator.
    bool auto_ic_setup = false;
    if (m_state.size() != sys.size()) {
        if (is_variational) {
            // Fetch the original number of equations/state variables.
            const auto n_orig_sv = std::get<1>(vsys).get_n_orig_sv();

            if (m_state.size() == n_orig_sv) [[likely]] {
                // Automatic setup of the initial conditions for the derivatives wrt
                // variables and parameters.
                detail::setup_variational_ics_varpar(m_state, std::get<1>(vsys), 1);
                auto_ic_setup = true;
            } else {
                throw std::invalid_argument(fmt::format(
                    "Inconsistent sizes detected in the initialization of a variational adaptive Taylor "
                    "integrator: the state vector has a dimension of {}, while the total number of equations is {}. "
                    "The size of the state vector must be equal either to the total number of equations, or to the "
                    "number of original (i.e., non-variational) equations, which for this system is {}",
                    m_state.size(), sys.size(), n_orig_sv));
            }
        } else [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                            "integrator: the state vector has a dimension of {}, while the number of equations is {}",
                            m_state.size(), sys.size()));
        }
    }

    // Init the time.
    if (time) {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::is_same_v<T, mppp::real>) {
            // NOTE: enforce the correct precision for mppp::real.
            m_time = detail::dfloat<T>(mppp::real{std::move(*time), this->get_prec()});
        } else {
#endif
            m_time = detail::dfloat<T>(std::move(*time));
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    } else {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::is_same_v<T, mppp::real>) {
            m_time = detail::dfloat<T>(mppp::real{mppp::real_kind::zero, this->get_prec()});
        } else {
#endif
            m_time = detail::dfloat<T>(static_cast<T>(0));
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    }

    // Parameter values.
    m_pars = std::move(pars);

#if defined(HEYOKA_HAVE_REAL)

    // Enforce the correct precision for mppp::real.
    if constexpr (std::is_same_v<T, mppp::real>) {
        for (auto &val : m_pars) {
            val.prec_round(this->get_prec());
        }
    }

#endif

    // High accuracy and compact mode flags.
    m_high_accuracy = high_accuracy;
    m_compact_mode = compact_mode;

    // Check the tolerance value.
    if (tol && (!isfinite(*tol) || *tol < 0)) {
        throw std::invalid_argument(fmt::format(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is {} instead",
            detail::fp_to_string(*tol)));
    }

    // Check the consistency of parallel vs compact mode.
    if (parallel_mode && !compact_mode) {
        throw std::invalid_argument("Parallel mode can be activated only in conjunction with compact mode");
    }

    // Store the tolerance.
    if (tol) {
        m_tol = std::move(*tol);
    } else {
        m_tol = detail::num_eps_like(m_state[0]);
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        // Force the tolerance to have the inferred precision.
        // NOTE: this is important as in taylor_add_adaptive_step*()
        // we use the tolerance value to infer the internal type.
        m_tol.prec_round(this->get_prec());
    }

#endif

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Compute the total number of params, including the rhs and the event functions.
    const auto tot_n_pars = detail::tot_n_pars_in_ode_sys(sys, tes, ntes);

    // Do we have events?
    const auto with_events = !tes.empty() || !ntes.empty();

    // Determine the order from the tolerance.
    m_order = detail::taylor_order_from_tol(m_tol);

    // Determine the external fp type.
    auto *ext_fp_t = detail::to_llvm_type<T>(m_llvm.context());

    // Determine the internal fp type.
    // NOTE: in case of mppp::real, we ensured earlier that the tolerance value
    // has the correct precision, so that llvm_type_like() will yield the correct internal type.
    auto *fp_t = detail::llvm_type_like(m_llvm, m_tol);

    // Add the stepper function.
    if (with_events) {
        std::vector<expression> ee;
        ee.reserve(boost::safe_numerics::safe<decltype(ee.size())>(tes.size()) + ntes.size());
        for (const auto &ev : tes) {
            ee.push_back(ev.get_expression());
        }
        for (const auto &ev : ntes) {
            ee.push_back(ev.get_expression());
        }

        m_dc = detail::taylor_add_adaptive_step_with_events(m_llvm, ext_fp_t, fp_t, "step_e", sys, 1, compact_mode, ee,
                                                            high_accuracy, parallel_mode, m_order);
    } else {
        m_dc = detail::taylor_add_adaptive_step(m_llvm, ext_fp_t, fp_t, "step", sys, 1, high_accuracy, compact_mode,
                                                parallel_mode, m_order);
    }

    // Fix m_pars' size, if necessary.
    if (m_pars.size() < tot_n_pars) {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::is_same_v<T, mppp::real>) {
            // For mppp::real, ensure that the appended parameter
            // values all have the inferred precision.
            m_pars.resize(boost::numeric_cast<decltype(m_pars.size())>(tot_n_pars),
                          mppp::real{mppp::real_kind::zero, this->get_prec()});
        } else {
#endif
            m_pars.resize(boost::numeric_cast<decltype(m_pars.size())>(tot_n_pars));
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    } else if (m_pars.size() > tot_n_pars) {
        throw std::invalid_argument(fmt::format(
            "Excessive number of parameter values passed to the constructor of an adaptive "
            "Taylor integrator: {} parameter value(s) were passed, but the ODE system contains only {} parameter(s)",
            m_pars.size(), tot_n_pars));
    }

    // Log runtimes in trace mode.
    spdlog::stopwatch sw;

    // Add the function for the computation of
    // the dense output.
    detail::taylor_add_d_out_function(m_llvm, detail::llvm_type_like(m_llvm, m_state[0]), m_dim, m_order, 1,
                                      high_accuracy);

    detail::get_logger()->trace("Taylor dense output runtime: {}", sw);
    sw.reset();

    // Run the jit.
    m_llvm.compile();

    detail::get_logger()->trace("Taylor LLVM compilation runtime: {}", sw);

    // Fetch the stepper.
    if (with_events) {
        m_step_f = reinterpret_cast<i_data::step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<i_data::step_f_t>(m_llvm.jit_lookup("step"));
    }

    // Fetch the function to compute the dense output.
    m_d_out_f = reinterpret_cast<i_data::d_out_f_t>(m_llvm.jit_lookup("d_out_f"));

    // Setup the vector for the Taylor coefficients.
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        // NOTE: ensure the Taylor coefficients are all generated
        // with the inferred precision.
        m_tc.resize(m_state.size() * (su32_t(m_order) + 1), mppp::real{mppp::real_kind::zero, this->get_prec()});
    } else {
#endif
        m_tc.resize(m_state.size() * (su32_t(m_order) + 1));
#if defined(HEYOKA_HAVE_REAL)
    }
#endif

    // Setup the vector for the dense output.
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        // NOTE: ensure the values are all generated with the inferred precision.
        m_d_out.resize(m_state.size(), mppp::real{mppp::real_kind::zero, this->get_prec()});
    } else {
#endif
        m_d_out.resize(m_state.size());
#if defined(HEYOKA_HAVE_REAL)
    }
#endif

    // Init the event data structure if needed.
    // NOTE: this can be done in parallel with the rest of the constructor,
    // once we have m_order/m_dim and we are done using tes/ntes.
    if (with_events) {
        m_ed_data = std::make_unique<ed_data>(m_llvm.make_similar(), std::move(tes), std::move(ntes), m_order, m_dim,
                                              m_state[0]);
    }

    if (auto_ic_setup) {
        // Finish the automatic setup of the ics for a variational
        // integrator.
        detail::setup_variational_ics_t0(m_llvm, m_state, m_pars, &m_time.hi, std::get<1>(vsys), 1, m_high_accuracy,
                                         m_compact_mode);
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        // Fix the precision of m_last_h, which was inited from '0'.
        m_last_h.prec_round(this->get_prec());

#if !defined(NDEBUG)

        // Run the precision check on the integrator data.
        // NOTE: run it only in debug mode since at this point
        // we should be sure all internal data has the correct precision.
        this->data_prec_check();

#endif
    }

#endif

    // Move vsys in.
    m_vsys = std::move(vsys);
}

template <typename T>
taylor_adaptive<T>::taylor_adaptive()
    : taylor_adaptive({prime("x"_var) = 0_dbl}, {static_cast<T>(0)}, kw::tol = static_cast<T>(1e-1))
{
}

template <typename T>
taylor_adaptive<T>::taylor_adaptive(const taylor_adaptive &other)
    : base_t(static_cast<const base_t &>(other)), m_i_data(std::make_unique<i_data>(*other.m_i_data)),
      m_ed_data(other.m_ed_data ? std::make_unique<ed_data>(*other.m_ed_data) : nullptr)
{
    if (m_ed_data) {
        m_i_data->m_step_f = reinterpret_cast<i_data::step_f_e_t>(m_i_data->m_llvm.jit_lookup("step_e"));
    } else {
        m_i_data->m_step_f = reinterpret_cast<i_data::step_f_t>(m_i_data->m_llvm.jit_lookup("step"));
    }
}

template <typename T>
taylor_adaptive<T>::taylor_adaptive(taylor_adaptive &&) noexcept = default;

template <typename T>
taylor_adaptive<T> &taylor_adaptive<T>::operator=(const taylor_adaptive &other)
{
    if (this != &other) {
        *this = taylor_adaptive(other);
    }

    return *this;
}

template <typename T>
taylor_adaptive<T> &taylor_adaptive<T>::operator=(taylor_adaptive &&) noexcept = default;

template <typename T>
taylor_adaptive<T>::~taylor_adaptive() = default;

// NOTE: the save/load patterns mimic the copy constructor logic.
template <typename T>
template <typename Archive>
void taylor_adaptive<T>::save_impl(Archive &ar, unsigned) const
{
    ar << boost::serialization::base_object<detail::taylor_adaptive_base<T, taylor_adaptive>>(*this);

    ar << m_i_data;
    ar << m_ed_data;
}

template <typename T>
template <typename Archive>
void taylor_adaptive<T>::load_impl(Archive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<taylor_adaptive<T>>::type::value)) {
        throw std::invalid_argument(fmt::format("Unable to load a taylor_adaptive integrator: "
                                                "the archive version ({}) is too old",
                                                version));
    }
    // LCOV_EXCL_STOP

    try {
        ar >> boost::serialization::base_object<detail::taylor_adaptive_base<T, taylor_adaptive>>(*this);

        ar >> m_i_data;
        ar >> m_ed_data;

        // Recover the function pointers.
        if (m_ed_data) {
            m_i_data->m_step_f = reinterpret_cast<i_data::step_f_e_t>(m_i_data->m_llvm.jit_lookup("step_e"));
        } else {
            m_i_data->m_step_f = reinterpret_cast<i_data::step_f_t>(m_i_data->m_llvm.jit_lookup("step"));
        }
        // LCOV_EXCL_START
    } catch (...) {
        // Reset to def-cted state in case of exceptions.
        *this = taylor_adaptive{};

        throw;
    }
    // LCOV_EXCL_STOP
}

template <typename T>
void taylor_adaptive<T>::save(boost::archive::binary_oarchive &ar, unsigned v) const
{
    save_impl(ar, v);
}

template <typename T>
void taylor_adaptive<T>::load(boost::archive::binary_iarchive &ar, unsigned v)
{
    load_impl(ar, v);
}

// Implementation detail to make a single integration timestep.
// The magnitude of the timestep is automatically deduced, but it will
// always be not greater than abs(max_delta_t). The propagation
// is done forward in time if max_delta_t >= 0, backwards in
// time otherwise.
//
// The function will return a pair, containing
// a flag describing the outcome of the integration,
// and the integration timestep that was used.
//
// NOTE: for the docs:
// - outcome:
//   - if nf state is detected, err_nf_state, else
//   - if terminal events trigger, return the index
//     of the first event triggering, else
//   - either time_limit or success, depending on whether
//     max_delta_t was used as a timestep or not;
// - event detection happens in the [0, h) half-open range (that is,
//   all detected events are guaranteed to trigger within
//   the [0, h) range). Thus, if the timestep ends up being zero
//   (either because max_delta_t == 0
//   or the inferred timestep is zero), then event detection is skipped
//   altogether;
// - the execution of the events' callbacks is guaranteed to proceed in
//   chronological order;
// - a timestep h == 0 will still result in m_last_h being updated (to zero)
//   and the Taylor coefficients being recorded in the internal array
//   (if wtc == true). That is, h == 0 is not treated in any special way.
template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step_impl(T max_delta_t, bool wtc)
{
    using std::abs;
    using std::isfinite;

#if !defined(NDEBUG)
    // NOTE: this is the only precondition on max_delta_t.
    using std::isnan;
    assert(!isnan(max_delta_t)); // LCOV_EXCL_LINE

    assert(m_i_data);
#endif

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        assert(max_delta_t.get_prec() == this->get_prec());

        // Run the data precision checks.
        this->data_prec_check();
    }

#endif

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_step_f);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pars);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tol);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_order);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out_f);

    auto h = max_delta_t;

    if (m_step_f.index() == 0u) {
        assert(!m_ed_data); // LCOV_EXCL_LINE

        // Invoke the vanilla stepper.
        std::get<0>(m_step_f)(m_state.data(), m_pars.data(), &m_time.hi, &h, wtc ? m_tc.data() : nullptr);

        // Update the time.
        m_time += h;

        // Store the last timestep.
        m_last_h = h;

        // Check if the time or the state vector are non-finite at the
        // end of the timestep.
        if (!isfinite(m_time)
            || std::any_of(m_state.cbegin(), m_state.cend(), [](const auto &x) { return !isfinite(x); })) {
            return std::tuple{taylor_outcome::err_nf_state, std::move(h)};
        }

        return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, std::move(h)};
    } else {
        assert(m_ed_data); // LCOV_EXCL_LINE

        auto &edd = *m_ed_data;

        // Invoke the stepper for event handling. We will record the norm infinity of the state vector +
        // event equations at the beginning of the timestep for later use.
        auto max_abs_state = detail::num_zero_like(h);
        std::get<1>(m_step_f)(edd.m_ev_jet.data(), m_state.data(), m_pars.data(), &m_time.hi, &h, &max_abs_state);

        // Compute the maximum absolute error on the Taylor series of the event equations, which we will use for
        // automatic cooldown deduction. If max_abs_state is not finite, set it to inf so that
        // in edd.detect_events() we skip event detection altogether.
        const auto g_eps = [&]() {
            if (isfinite(max_abs_state)) {
                // Are we in absolute or relative error control mode?
                const auto abs_or_rel = max_abs_state < 1;

                // Estimate the size of the largest remainder in the Taylor
                // series of both the dynamical equations and the events.
                auto max_r_size = abs_or_rel ? m_tol : (m_tol * max_abs_state);

                // NOTE: depending on m_tol, max_r_size is arbitrarily small, but the real
                // integration error cannot be too small due to floating-point truncation.
                // This is the case for instance if we use sub-epsilon integration tolerances
                // to achieve Brouwer's law. In such a case, we cap the value of g_eps,
                // using eps * max_abs_state as an estimation of the smallest number
                // that can be resolved with the current floating-point type.
                auto tmp = detail::num_eps_like(max_abs_state) * max_abs_state;

                // NOTE: the if condition in the next line is equivalent, in relative
                // error control mode, to:
                // if (m_tol < eps)
                if (max_r_size < tmp) {
                    return tmp;
                } else {
                    return max_r_size;
                }
            } else {
                return detail::num_inf_like(max_abs_state);
            }
        }();

#if defined(HEYOKA_HAVE_REAL)

        if constexpr (std::is_same_v<T, mppp::real>) {
            assert(g_eps.get_prec() == this->get_prec());
        }

#endif

        // Write unconditionally the tcs.
        std::copy(edd.m_ev_jet.data(), edd.m_ev_jet.data() + m_dim * (m_order + 1u), m_tc.data());

        // Do the event detection.
        edd.detect_events(h, m_order, m_dim, g_eps);

        // NOTE: before this point, we did not alter
        // any user-visible data in the integrator (just
        // temporary memory). From here until we start invoking
        // the callbacks, everything is noexcept, so we don't
        // risk leaving the integrator in a half-baked state.

        // Sort the events by time.
        // NOTE: the time coordinates in m_d_(n)tes is relative
        // to the beginning of the timestep. It will be negative
        // for backward integration, thus we compare using
        // abs() so that the first events are those which
        // happen closer to the beginning of the timestep.
        // NOTE: the checks inside edd.detect_events() ensure
        // that we can safely sort the events' times.
        auto cmp = [](const auto &ev0, const auto &ev1) { return detail::abs_lt(std::get<1>(ev0), std::get<1>(ev1)); };
        std::sort(edd.m_d_tes.begin(), edd.m_d_tes.end(), cmp);
        std::sort(edd.m_d_ntes.begin(), edd.m_d_ntes.end(), cmp);

        // If we have terminal events we need
        // to update the value of h.
        if (!edd.m_d_tes.empty()) {
#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<T, mppp::real>) {
                // NOTE: use set() so that no matter what happens during event detection,
                // we will not change the precision of h to an invalid value.
                h.set(std::get<1>(edd.m_d_tes[0]));
            } else {
#endif
                h = std::get<1>(edd.m_d_tes[0]);
#if defined(HEYOKA_HAVE_REAL)
            }
#endif
        }

        // Update the state.
        m_d_out_f(m_state.data(), edd.m_ev_jet.data(), &h);

        // Update the time.
        m_time += h;

        // Store the last timestep.
        m_last_h = h;

        // Check if the time or the state vector are non-finite at the
        // end of the timestep.
        if (!isfinite(m_time)
            || std::any_of(m_state.cbegin(), m_state.cend(), [](const auto &x) { return !isfinite(x); })) {
            return std::tuple{taylor_outcome::err_nf_state, std::move(h)};
        }

        // Update the cooldowns.
        for (auto &cd : edd.m_te_cooldowns) {
            if (cd) {
                // Check if the timestep we just took
                // brought this event outside the cooldown.
                auto tmp = cd->first + h;

                if (abs(tmp) >= cd->second) {
                    // We are now outside the cooldown period
                    // for this event, reset cd.
                    cd.reset();
                } else {
                    // Still in cooldown, update the
                    // time spent in cooldown.
                    cd->first = tmp;
                }
            }
        }

        // If we don't have terminal events, we will invoke the callbacks
        // of *all* the non-terminal events. Otherwise, we need to figure
        // out which non-terminal events do not happen because their time
        // coordinate is past the the first terminal event.
        const auto ntes_end_it
            = edd.m_d_tes.empty()
                  ? edd.m_d_ntes.end()
                  : std::lower_bound(edd.m_d_ntes.begin(), edd.m_d_ntes.end(), h,
                                     [](const auto &ev, const auto &t) { return detail::abs_lt(std::get<1>(ev), t); });

        // Store the time coordinate before invoking the callbacks.
        const auto new_time = m_time;

        // Invoke the callbacks of the non-terminal events, which are guaranteed
        // to happen before the first terminal event.
        for (auto it = edd.m_d_ntes.begin(); it != ntes_end_it; ++it) {
            const auto &t = *it;
            auto &cb = edd.m_ntes[std::get<0>(t)].get_callback();
            assert(cb); // LCOV_EXCL_LINE

            // NOTE: use new_time, instead of m_time, in order to prevent
            // passing the wrong time coordinate to the callback if an earlier
            // callback changed it.

#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<T, mppp::real>) {
                // NOTE: for mppp::real, we must ensure that the time coordinate
                // of the event is computed with the correct precision (we are
                // getting the time coordinate from the event detection machinery,
                // which does not enforce preservation of the correct precision).
                auto tc = static_cast<T>((new_time - m_last_h + std::get<1>(t)));
                tc.prec_round(this->get_prec());
                cb(*this, tc, std::get<2>(t));
            } else {
#endif
                cb(*this, static_cast<T>(new_time - m_last_h + std::get<1>(t)), std::get<2>(t));
#if defined(HEYOKA_HAVE_REAL)
            }
#endif
        }

        // The return value of the first
        // terminal event's callback. It will be
        // unused if there are no terminal events.
        bool te_cb_ret = false;

        if (!edd.m_d_tes.empty()) {
            // Fetch the first terminal event.
            const auto te_idx = std::get<0>(edd.m_d_tes[0]);
            assert(te_idx < edd.m_tes.size()); // LCOV_EXCL_LINE
            auto &te = edd.m_tes[te_idx];

            // Set the corresponding cooldown.
            if (te.get_cooldown() >= 0) {
                // Cooldown explicitly provided by the user, use it.
                edd.m_te_cooldowns[te_idx].emplace(0, te.get_cooldown());
            } else {
                // Deduce the cooldown automatically.
                // NOTE: if g_eps is not finite, we skipped event detection
                // altogether and thus we never end up here. If the derivative
                // of the event equation is not finite, the event is also skipped.
                edd.m_te_cooldowns[te_idx].emplace(0,
                                                   detail::taylor_deduce_cooldown(g_eps, std::get<3>(edd.m_d_tes[0])));
            }

            // Invoke the callback of the first terminal event, if it has one.
            if (te.get_callback()) {
                te_cb_ret = te.get_callback()(*this, std::get<2>(edd.m_d_tes[0]));
            }
        }

        // NOTE: event callbacks - terminal or not - cannot modify the time
        // variable, as this is going to mess up the internal time keeping logic
        // in the propagate_*() functions.
        // NOTE: use cmp_nan_eq() so that we consider NaN == NaN in the time coordinates.
        // This is necessary in case something goes wrong in the integration step
        // and the time coordinate goes NaN - in such a case, the standard equality operator
        // would trigger (because NaN != NaN) even if the time coordinate was not changed by the callback.
        if (!detail::cmp_nan_eq(m_time.hi, new_time.hi) || !detail::cmp_nan_eq(m_time.lo, new_time.lo)) {
            throw std::runtime_error("The invocation of one or more event callbacks resulted in the alteration of the "
                                     "time coordinate of the integrator - this is not supported");
        }

        if (edd.m_d_tes.empty()) {
            // No terminal events detected, return success or time limit.
            return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, std::move(h)};
        } else {
            // Terminal event detected. Fetch its index.
            const auto ev_idx = static_cast<std::int64_t>(std::get<0>(edd.m_d_tes[0]));

            // NOTE: if te_cb_ret is true, it means that the terminal event has
            // a callback and its invocation returned true (meaning that the
            // integration should continue). Otherwise, either the terminal event
            // has no callback or its callback returned false, meaning that the
            // integration must stop.
            return std::tuple{taylor_outcome{te_cb_ret ? ev_idx : (-ev_idx - 1)}, std::move(h)};
        }
    }
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step(bool wtc)
{
    // NOTE: time limit +inf means integration forward in time
    // and no time limit.
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return step_impl(mppp::real{mppp::real_kind::inf, this->get_prec()}, wtc);
    } else {
#endif
        return step_impl(std::numeric_limits<T>::infinity(), wtc);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step_backward(bool wtc)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return step_impl(mppp::real{mppp::real_kind::inf, -1, this->get_prec()}, wtc);
    } else {
#endif
        return step_impl(-std::numeric_limits<T>::infinity(), wtc);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step(T max_delta_t, bool wtc)
{
    using std::isnan;

    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A NaN max_delta_t was passed to the step() function of an adaptive Taylor integrator");
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        max_delta_t.prec_round(this->get_prec());
    }

#endif

    return step_impl(std::move(max_delta_t), wtc);
}

// Reset all cooldowns for the terminal events.
template <typename T>
void taylor_adaptive<T>::reset_cooldowns()
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    for (auto &cd : m_ed_data->m_te_cooldowns) {
        cd.reset();
    }
}

// NOTE: possible outcomes:
// - time_limit iff the propagation was performed
//   up to the final time, else
// - cb_interrupt if the propagation was interrupted
//   via callback, else
// - err_nf_state if a non-finite state was generated, else
// - an event index if a stopping terminal event was encountered.
// The callback is always executed at the end of each timestep, unless
// a non-finite state was detected.
// The continuous output is always updated at the end of each timestep,
// unless a non-finite state was detected.
template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
taylor_adaptive<T>::propagate_until_impl(detail::dfloat<T> t, std::size_t max_steps, T max_delta_t, step_callback<T> cb,
                                         bool wtc, bool with_c_out)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_order);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_llvm);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_high_accuracy);

    // Check the current time.
    if (!isfinite(m_time)) {
        throw std::invalid_argument("Cannot invoke the propagate_until() function of an adaptive Taylor integrator if "
                                    "the current time is not finite");
    }

    // Check the final time.
    if (!isfinite(t)) {
        throw std::invalid_argument(
            "A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        assert(t.hi.get_prec() == t.lo.get_prec());
        t.hi.prec_round(this->get_prec());
        t.lo.prec_round(this->get_prec());

        max_delta_t.prec_round(this->get_prec());
    }

#endif

    // Check max_delta_t.
    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A nan max_delta_t was passed to the propagate_until() function of an adaptive Taylor integrator");
    }
    if (max_delta_t <= 0) {
        throw std::invalid_argument(
            "A non-positive max_delta_t was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

    // If with_c_out is true, we always need to write the Taylor coefficients.
    wtc = wtc || with_c_out;

    // These vectors are used in the construction of the continuous output.
    // If continuous output is not requested, they will remain empty.
    std::vector<T> c_out_tcs, c_out_times_hi, c_out_times_lo;
    if (with_c_out) {
        // Push in the starting time.
        c_out_times_hi.push_back(m_time.hi);
        c_out_times_lo.push_back(m_time.lo);
    }

    // Initial values for the counters
    // and the min/max abs of the integration
    // timesteps.
    // NOTE: iter_counter is for keeping track of the max_steps
    // limits, step_counter counts the number of timesteps performed
    // with a nonzero h. Most of the time these two quantities
    // will be identical, apart from corner cases.
    std::size_t iter_counter = 0, step_counter = 0;
    // NOTE: in case of mppp::real, we know that max_delta_t has the correct
    // precision by now.
    T min_h = detail::num_inf_like(max_delta_t), max_h = detail::num_zero_like(max_delta_t);

    // Init the remaining time.
    // NOTE: for mppp::real, manipulations on rem_time involve only t and m_time: t's precision
    // has been checked, m_time is manipulated only by step(), which enforces the correct
    // precision is maintained. Thus, rem_time will always keep the correct precision.
    auto rem_time = t - m_time;

    // Check it.
    if (!isfinite(rem_time)) {
        throw std::invalid_argument("The final time passed to the propagate_until() function of an adaptive Taylor "
                                    "integrator results in an overflow condition");
    }

    // Cache the integration direction.
    const auto t_dir = (rem_time >= static_cast<T>(0));

    // Helper to create the continuous output object.
    auto make_c_out = [&]() -> std::optional<continuous_output<T>> {
        if (with_c_out) {
            if (c_out_times_hi.size() < 2u) {
                // NOTE: this means that no successful steps
                // were taken.
                return {};
            }

            // Construct the return value.
            continuous_output<T> ret(m_llvm.make_similar());

            // Fill in the data.
            ret.m_tcs = std::move(c_out_tcs);
            ret.m_times_hi = std::move(c_out_times_hi);
            ret.m_times_lo = std::move(c_out_times_lo);

            // Prepare the output vector.
            // NOTE: in case of mppp::real, we know that max_delta_t has the correct
            // precision by now.
            ret.m_output.resize(boost::numeric_cast<decltype(ret.m_output.size())>(m_dim),
                                detail::num_zero_like(max_delta_t));

            // Add the continuous output function.
            ret.add_c_out_function(m_order, m_dim, m_high_accuracy);

            return std::optional{std::move(ret)};
        } else {
            return {};
        }
    };

    // Helper to update the continuous output data after a timestep.
    auto update_c_out = [&]() {
        if (with_c_out) {
#if !defined(NDEBUG)
            const detail::dfloat<T> prev_time(c_out_times_hi.back(), c_out_times_lo.back());
#endif

            c_out_times_hi.push_back(m_time.hi);
            c_out_times_lo.push_back(m_time.lo);

#if !defined(NDEBUG)
            const detail::dfloat<T> new_time(c_out_times_hi.back(), c_out_times_lo.back());
            assert(isfinite(new_time));
            if (t_dir) {
                assert(!(new_time < prev_time));
            } else {
                assert(!(new_time > prev_time));
            }
#endif

            c_out_tcs.insert(c_out_tcs.end(), m_tc.begin(), m_tc.end());
        }
    };

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

    // Error message for when the callback modifies the time coordinate.
    constexpr auto cb_time_errmsg
        = "The invocation of the callback passed to propagate_until() resulted in the alteration of the "
          "time coordinate of the integrator - this is not supported";

    // Run the callback's pre_hook() function, if needed.
    if (with_cb) {
        // NOTE: the callback is not allowed to change the time coordinate
        // (either via pre_hook() or via its call operator, as shown later).
        // NOTE: orig_time is guaranteed to be finite
        // due to the checks done earlier on m_time.
        const auto orig_time = m_time;

        cb.pre_hook(*this);

        if (m_time != orig_time) {
            throw std::runtime_error(cb_time_errmsg);
        }
    }

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        assert((rem_time >= T(0)) == t_dir); // LCOV_EXCL_LINE
        auto dt_limit = t_dir ? std::min(detail::dfloat<T>(max_delta_t), rem_time)
                              : std::max(detail::dfloat<T>(-max_delta_t), rem_time);
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        const auto [oc, h] = step_impl(std::move(dt_limit.hi), wtc);

        if (oc == taylor_outcome::err_nf_state) {
            // If a non-finite state is detected, we do *not* want
            // to execute the propagate() callback and we do *not* want
            // to update the continuous output. Just exit.
            return std::tuple{oc, std::move(min_h), std::move(max_h), step_counter, make_c_out(), std::move(cb)};
        }

        // Update the number of steps.
        step_counter += static_cast<std::size_t>(h != 0);

        // Update min_h/max_h, but only if the outcome is success (otherwise
        // the step was artificially clamped either by a time limit or
        // by a terminal event).
        if (oc == taylor_outcome::success) {
            const auto abs_h = abs(h);
            min_h = std::min(min_h, abs_h);
            max_h = std::max(max_h, abs_h);
        }

        // Update the continuous output data.
        update_c_out();

        // Update the number of iterations.
        ++iter_counter;

        // Execute the propagate() callback, if applicable.
        if (with_cb) {
            // Store the current time coordinate before
            // executing the cb, so that we can check if
            // the cb changes the time coordinate.
            // NOTE: orig_time is guaranteed to be finite
            // because the outcome of the step is not err_nf_state.
            const auto orig_time = m_time;

            // Execute the cb.
            const auto ret_cb = cb(*this);

            // Check the time coordinate.
            if (m_time != orig_time) {
                throw std::runtime_error(cb_time_errmsg);
            }

            if (!ret_cb) {
                // Interruption via callback.
                return std::tuple{taylor_outcome::cb_stop,
                                  std::move(min_h),
                                  std::move(max_h),
                                  step_counter,
                                  make_c_out(),
                                  std::move(cb)};
            }
        }

        // The breakout conditions:
        // - a step of rem_time was used, or
        // - a stopping terminal event was detected.
        // NOTE: we check h == rem_time, instead of just
        // oc == time_limit, because clamping via max_delta_t
        // could also result in time_limit.
        const bool ste_detected = oc > taylor_outcome::success && oc < taylor_outcome{0};
        if (h == rem_time.hi || ste_detected) {
#if !defined(NDEBUG)
            if (h == rem_time.hi) {
                assert(oc == taylor_outcome::time_limit);
            }
#endif
            return std::tuple{oc, std::move(min_h), std::move(max_h), step_counter, make_c_out(), std::move(cb)};
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger (modulo wraparound)
        // as by this point we are sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            return std::tuple{taylor_outcome::step_limit,
                              std::move(min_h),
                              std::move(max_h),
                              step_counter,
                              make_c_out(),
                              std::move(cb)};
        }

        // Update the remaining time.
        // NOTE: at this point, we are sure
        // that abs(h) < abs(rem_time.hi). This implies
        // that t - m_time cannot undergo a sign
        // flip and invert the integration direction.
        assert(abs(h) < abs(rem_time.hi));
        rem_time = t - m_time;
    }
}

template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>, step_callback<T>>
taylor_adaptive<T>::propagate_for_impl(T delta_t, std::size_t max_steps, T max_delta_t, step_callback<T> cb, bool wtc,
                                       bool with_c_out)
{
    return propagate_until_impl(m_i_data->m_time + std::move(delta_t), max_steps, std::move(max_delta_t), std::move(cb),
                                wtc, with_c_out);
}

// NOTE: possible outcomes:
// - time_limit (the happy path),
// - nf_err_state in case of non-finite state
//   detected,
// - cb_stop in case of stop by callback,
// - the index of a stopping terminal event.
// The callback is always executed at the end of each timestep, unless
// a non-finite state was detected.
template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t, step_callback<T>, std::vector<T>>
taylor_adaptive<T>::propagate_grid_impl(std::vector<T> grid, std::size_t max_steps, T max_delta_t, step_callback<T> cb)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out);

    if (!isfinite(m_time)) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator if the current time is not finite");
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        max_delta_t.prec_round(this->get_prec());
    }

#endif

    // Check max_delta_t.
    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A nan max_delta_t was passed to the propagate_grid() function of an adaptive Taylor integrator");
    }
    if (max_delta_t <= 0) {
        throw std::invalid_argument(
            "A non-positive max_delta_t was passed to the propagate_grid() function of an adaptive Taylor integrator");
    }

    // Check the grid.
    if (grid.empty()) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator if the time grid is empty");
    }

    constexpr auto nf_err_msg
        = "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator";
    constexpr auto ig_err_msg
        = "A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator";

    // Check the first point.
    if (!isfinite(grid[0])) {
        throw std::invalid_argument(nf_err_msg);
    }

#if defined(HEYOKA_HAVE_REAL)

    // Ensure all grid points have the correct precision for real.
    if constexpr (std::is_same_v<T, mppp::real>) {
        for (auto &val : grid) {
            val.prec_round(this->get_prec());
        }
    }

#endif

    if (grid.size() > 1u) {
        // Establish the direction of the grid from
        // the first two points.
        if (!isfinite(grid[1])) {
            throw std::invalid_argument(nf_err_msg);
        }
        if (grid[1] == grid[0]) {
            throw std::invalid_argument(ig_err_msg);
        }

        const auto grid_direction = grid[1] > grid[0];

        // Check that the remaining points are finite and that
        // they are ordered monotonically.
        for (decltype(grid.size()) i = 2; i < grid.size(); ++i) {
            if (!isfinite(grid[i])) {
                throw std::invalid_argument(nf_err_msg);
            }

            if ((grid[i] > grid[i - 1u]) != grid_direction) {
                throw std::invalid_argument(ig_err_msg);
            }
        }
    }

    // Require that the user provides a grid starting from the
    // current integrator time, modulo the double-length correction.
    if (m_time.hi != grid[0]) {
        throw std::invalid_argument(
            fmt::format("When invoking propagate_grid(), the first element of the time grid "
                        "must match the current time coordinate - however, the first element of the time grid has a "
                        "value of {}, while the current time coordinate is {}",
                        grid[0], m_time.hi));
    }

    // Pre-allocate the return value.
    std::vector<T> retval;
    retval.reserve(boost::safe_numerics::safe<decltype(retval.size())>(grid.size()) * get_dim());

    // Initial values for the counters
    // and the min/max abs of the integration
    // timesteps.
    // NOTE: iter_counter is for keeping track of the max_steps
    // limits, step_counter counts the number of timesteps performed
    // with a nonzero h. Most of the time these two quantities
    // will be identical, apart from corner cases.
    std::size_t iter_counter = 0, step_counter = 0;
    // NOTE: in case of mppp::real, we know that max_delta_t has the correct
    // precision by now.
    T min_h = detail::num_inf_like(max_delta_t), max_h = detail::num_zero_like(max_delta_t);

    // Propagate the system up to the first grid point.
    // This is necessary in order to account for the fact
    // that the user cannot pass as starting point in the grid
    // a time coordinate which is *exactly* equal to m_time,
    // due to the usage of a double-length representation.
    // NOTE: we pass write_tc = true because some grid
    // points after the first one might end up being
    // calculated via dense output *before*
    // taking additional steps, and, in such case, we
    // must ensure the TCs are up to date.
    // NOTE: if the integrator time is already at grid[0],
    // then propagate_until() will take a zero timestep,
    // resulting in m_last_h also going to zero and no event
    // detection being performed.
    // NOTE: use the same max_steps for the initial propagation,
    // and don't pass the callback.
    const auto oc_until = std::get<0>(
        propagate_until(grid[0], kw::max_delta_t = max_delta_t, kw::max_steps = max_steps, kw::write_tc = true));

    if (oc_until != taylor_outcome::time_limit) {
        // The outcome is not time_limit, exit now.
        return std::tuple{oc_until, std::move(min_h), std::move(max_h), step_counter, std::move(cb), std::move(retval)};
    }

    // Add the first result to retval.
    // NOTE: in principle here we could have a state
    // with incorrect precision being added to retval - this
    // happens in case an event callback changes the state precision
    // at the last step of the propagate_until(). This won't have
    // ill effects on the propagate_grid() logic as retval
    // is only written to (we don't use its values in the
    // function logic).
    retval.insert(retval.end(), m_state.begin(), m_state.end());

    // Cache a double-length zero for several uses later.
    const auto dl_zero = detail::dfloat<T>(detail::num_zero_like(max_delta_t));

    // Init the remaining time.
    // NOTE: m_time is guaranteed to have the correct precision,
    // grid was checked above - thus rem_time will have the correct
    // precision.
    auto rem_time = grid.back() - m_time;

    // Check it.
    if (!isfinite(rem_time)) {
        throw std::invalid_argument("The final time passed to the propagate_grid() function of an adaptive Taylor "
                                    "integrator results in an overflow condition");
    }

    // Cache the integration direction.
    const auto t_dir = (rem_time >= dl_zero);

    // This flag, if set to something else than success,
    // is used to signal the early interruption of the integration.
    auto interrupt = taylor_outcome::success;

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

    // Error message for when the callback modifies the time coordinate.
    constexpr auto cb_time_errmsg
        = "The invocation of the callback passed to propagate_grid() resulted in the alteration of the "
          "time coordinate of the integrator - this is not supported";

    // Run the callback's pre_hook() function, if needed.
    if (with_cb) {
        // NOTE: the callback is not allowed to change the time coordinate
        // (either via pre_hook() or via its call operator, as shown later).
        // NOTE: orig_time is guaranteed to be finite
        // because propagate_until() did not return err_nf_state.
        const auto orig_time = m_time;

        cb.pre_hook(*this);

        if (m_time != orig_time) {
            throw std::runtime_error(cb_time_errmsg);
        }
    }

    // Iterate over the remaining grid points.
    for (decltype(grid.size()) cur_grid_idx = 1; cur_grid_idx < grid.size();) {
        // Establish the time range of the last
        // taken timestep.
        // NOTE: this is computed so that t0 < t1,
        // regardless of the integration direction.
        const auto t0 = std::min(m_time, m_time - m_last_h);
        const auto t1 = std::max(m_time, m_time - m_last_h);

        // Compute the state of the system via dense output for as many grid
        // points as possible, i.e., as long as the grid times
        // fall within the time range of the last step.
        while (true) {
            // Fetch the current time target.
            const auto &cur_tt = grid[cur_grid_idx];

            // NOTE: we force processing of all remaining grid points
            // if we are at the last timestep. We do this in order to avoid
            // numerical issues when deciding if the last grid point
            // falls within the range of the last step.
            if ((cur_tt >= t0 && cur_tt <= t1) || (rem_time == dl_zero)) {
                // The current time target falls within the range of
                // the last step. Compute the dense output in cur_tt.
                update_d_output(cur_tt);

                // Add the result to retval.
                retval.insert(retval.end(), m_d_out.begin(), m_d_out.end());
            } else {
                // Cannot use dense output on the current time target,
                // need to take another step.
                break;
            }

            // Move to the next time target, or break out
            // if we have no more.
            if (++cur_grid_idx == grid.size()) {
                break;
            }
        }

        if (cur_grid_idx == grid.size() || interrupt != taylor_outcome::success) {
            // Either we ran out of grid points, or the last step() invocation
            // resulted in early termination.
            break;
        }

        // Take the next step, making sure to write the Taylor coefficients
        // and to cap the timestep size so that we don't go past the
        // last grid point and we don't use a timestep exceeding max_delta_t.
        // NOTE: rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        assert((rem_time >= T(0)) == t_dir); // LCOV_EXCL_LINE
        auto dt_limit = t_dir ? std::min(detail::dfloat<T>(max_delta_t), rem_time)
                              : std::max(detail::dfloat<T>(-max_delta_t), rem_time);
        const auto [oc, h] = step_impl(std::move(dt_limit.hi), true);

        if (oc == taylor_outcome::err_nf_state) {
            // If a non-finite state is detected, we do *not* want
            // to execute the propagate() callback and we do *not* want
            // to update the return value. Just exit.
            return std::tuple{oc, std::move(min_h), std::move(max_h), step_counter, std::move(cb), std::move(retval)};
        }

        // Update the number of steps.
        step_counter += static_cast<std::size_t>(h != 0);

        // Update min_h/max_h, but only if the outcome is success (otherwise
        // the step was artificially clamped either by a time limit or
        // by a terminal event).
        if (oc == taylor_outcome::success) {
            const auto abs_h = abs(h);
            min_h = std::min(min_h, abs_h);
            max_h = std::max(max_h, abs_h);
        }

        // Update the number of iterations.
        ++iter_counter;

        // Small helper to wrap the invocation of the callback
        // while checking that the callback does not change the
        // time coordinate.
        auto wrap_cb_call = [&]() {
            // Store the current time coordinate before
            // executing the cb, so that we can check if
            // the cb changes the time coordinate.
            // NOTE: orig_time is guaranteed to contain
            // a finite value because the outcome of the step
            // was not err_nf_state.
            const auto orig_time = m_time;

            // Execute the cb.
            assert(cb);
            const auto ret_cb = cb(*this);

            // Check the time coordinate.
            if (m_time != orig_time) {
                throw std::runtime_error(cb_time_errmsg);
            }

            return ret_cb;
        };

        // Check the early interruption conditions.
        // NOTE: only one of them must be set.
        // NOTE: if the integration is stopped via callback,
        // we don't exit immediately, we process any remaining
        // grid points first.
        if (with_cb && !wrap_cb_call()) {
            // Interruption via callback.
            interrupt = taylor_outcome::cb_stop;
        } else if (oc > taylor_outcome::success && oc < taylor_outcome{0}) {
            // Interruption via stopping terminal event.
            interrupt = oc;
        } else if (iter_counter == max_steps) {
            // Interruption via max iteration limit.
            interrupt = taylor_outcome::step_limit;
        }

        // Update the remaining time.
        // NOTE: if rem_time.hi was used as a timestep,
        // it means that we hit the time limit. Force rem_time to zero
        // to signal this, avoiding inconsistencies with grid.back() - m_time
        // not going exactly to zero due to numerical issues. A zero rem_time
        // will also force the processing of all remaining grid points.
        if (h == rem_time.hi) {
            assert(oc == taylor_outcome::time_limit); // LCOV_EXCL_LINE
            rem_time = dl_zero;
        } else {
            // NOTE: this should never flip the time direction of the
            // integration for the same reasons as explained in the
            // scalar implementation of propagate_until().
            assert(abs(h) < abs(static_cast<T>(rem_time))); // LCOV_EXCL_LINE
            rem_time = grid.back() - m_time;
        }
    }

    // Return time_limit or the interrupt condition, if the integration
    // was stopped early.
    return std::tuple{interrupt == taylor_outcome::success ? taylor_outcome::time_limit : interrupt,
                      std::move(min_h),
                      std::move(max_h),
                      step_counter,
                      std::move(cb),
                      std::move(retval)};
}

template <typename T>
const llvm_state &taylor_adaptive<T>::get_llvm_state() const
{
    return m_i_data->m_llvm;
}

template <typename T>
const taylor_dc_t &taylor_adaptive<T>::get_decomposition() const
{
    return m_i_data->m_dc;
}

template <typename T>
std::uint32_t taylor_adaptive<T>::get_order() const
{
    return m_i_data->m_order;
}

template <typename T>
T taylor_adaptive<T>::get_tol() const
{
    return m_i_data->m_tol;
}

template <typename T>
bool taylor_adaptive<T>::get_high_accuracy() const
{
    return m_i_data->m_high_accuracy;
}

template <typename T>
bool taylor_adaptive<T>::get_compact_mode() const
{
    return m_i_data->m_compact_mode;
}

template <typename T>
std::uint32_t taylor_adaptive<T>::get_dim() const
{
    return m_i_data->m_dim;
}

template <typename T>
T taylor_adaptive<T>::get_time() const
{
    return static_cast<T>(m_i_data->m_time);
}

template <typename T>
std::pair<T, T> taylor_adaptive<T>::get_dtime() const
{
    return {m_i_data->m_time.hi, m_i_data->m_time.lo};
}

template <typename T>
const std::vector<T> &taylor_adaptive<T>::get_state() const
{
    return m_i_data->m_state;
}

template <typename T>
const T *taylor_adaptive<T>::get_state_data() const
{
    return m_i_data->m_state.data();
}

template <typename T>
std::ranges::subrange<typename std::vector<T>::iterator> taylor_adaptive<T>::get_state_range()
{
    return std::ranges::subrange(m_i_data->m_state.begin(), m_i_data->m_state.end());
}

template <typename T>
T *taylor_adaptive<T>::get_state_data()
{
    return m_i_data->m_state.data();
}

template <typename T>
const std::vector<T> &taylor_adaptive<T>::get_pars() const
{
    return m_i_data->m_pars;
}

template <typename T>
const T *taylor_adaptive<T>::get_pars_data() const
{
    return m_i_data->m_pars.data();
}

template <typename T>
std::ranges::subrange<typename std::vector<T>::iterator> taylor_adaptive<T>::get_pars_range()
{
    return std::ranges::subrange(m_i_data->m_pars.begin(), m_i_data->m_pars.end());
}

template <typename T>
T *taylor_adaptive<T>::get_pars_data()
{
    return m_i_data->m_pars.data();
}

template <typename T>
const std::vector<T> &taylor_adaptive<T>::get_tc() const
{
    return m_i_data->m_tc;
}

template <typename T>
T taylor_adaptive<T>::get_last_h() const
{
    return m_i_data->m_last_h;
}

template <typename T>
const std::vector<T> &taylor_adaptive<T>::get_d_output() const
{
    return m_i_data->m_d_out;
}

template <typename T>
bool taylor_adaptive<T>::with_events() const
{
    return static_cast<bool>(m_ed_data);
}

template <typename T>
const std::vector<typename taylor_adaptive<T>::t_event_t> &taylor_adaptive<T>::get_t_events() const
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    return m_ed_data->m_tes;
}

template <typename T>
const std::vector<std::optional<std::pair<T, T>>> &taylor_adaptive<T>::get_te_cooldowns() const
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    return m_ed_data->m_te_cooldowns;
}

template <typename T>
const std::vector<typename taylor_adaptive<T>::nt_event_t> &taylor_adaptive<T>::get_nt_events() const
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    return m_ed_data->m_ntes;
}

template <typename T>
const std::vector<std::pair<expression, expression>> &taylor_adaptive<T>::get_sys() const noexcept
{
    return (m_i_data->m_vsys.index() == 0) ? std::get<0>(m_i_data->m_vsys) : std::get<1>(m_i_data->m_vsys).get_sys();
}

template <typename T>
void taylor_adaptive<T>::set_time(T t)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        t.prec_round(this->get_prec());
    }
#endif

    m_i_data->m_time = detail::dfloat<T>(std::move(t));
}

template <typename T>
void taylor_adaptive<T>::set_dtime(T hi, T lo)
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        hi.prec_round(this->get_prec());
        lo.prec_round(this->get_prec());
    }
#endif

    // Check the components.
    detail::dtime_checks(hi, lo);

    m_i_data->m_time = normalise(detail::dfloat<T>(std::move(hi), std::move(lo)));
}

template <typename T>
const std::vector<T> &taylor_adaptive<T>::update_d_output(T time, bool rel_time)
{

#if defined(HEYOKA_HAVE_REAL)

    // NOTE: here it is not necessary to run the precision
    // check on the state/params, as they are never used
    // in the computation.

    if constexpr (std::is_same_v<T, mppp::real>) {
        time.prec_round(this->get_prec());
    }

#endif

    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        const auto h = m_i_data->m_last_h + std::move(time);

        m_i_data->m_d_out_f(m_i_data->m_d_out.data(), m_i_data->m_tc.data(), &h);
    } else {
        // Absolute time coordinate.
        const auto h = std::move(time) - (m_i_data->m_time - m_i_data->m_last_h);

        m_i_data->m_d_out_f(m_i_data->m_d_out.data(), m_i_data->m_tc.data(), &h.hi);
    }

    return m_i_data->m_d_out;
}

// Explicit instantiations
// NOLINTBEGIN
#define HEYOKA_TAYLOR_ADAPTIVE_INST(F)                                                                                 \
    template class HEYOKA_DLL_PUBLIC detail::taylor_adaptive_base<F, taylor_adaptive<F>>;                              \
    template class HEYOKA_DLL_PUBLIC taylor_adaptive<F>;
// NOLINTEND

HEYOKA_TAYLOR_ADAPTIVE_INST(float)
HEYOKA_TAYLOR_ADAPTIVE_INST(double)
HEYOKA_TAYLOR_ADAPTIVE_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADAPTIVE_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_TAYLOR_ADAPTIVE_INST(mppp::real)

#endif

#undef HEYOKA_TAYLOR_ADAPTIVE_INST

HEYOKA_END_NAMESPACE

#undef HEYOKA_TAYLOR_REF_FROM_I_DATA
