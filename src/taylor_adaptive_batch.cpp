// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/continuous_output.hpp>
#include <heyoka/detail/aligned_vector.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/ed_data.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/i_data.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/ta_jit_data.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/detail/vsys_data.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

// NOTE: these are helper macros to reduce typing when accessing the data members of i_data.
#define HEYOKA_TAYLOR_REF_FROM_I_DATA(name) [[maybe_unused]] auto &name = m_i_data->name
#define HEYOKA_TAYLOR_CREF_FROM_I_DATA(name) [[maybe_unused]] const auto &name = m_i_data->name

HEYOKA_BEGIN_NAMESPACE

template <typename T>
taylor_adaptive_batch<T>::taylor_adaptive_batch(private_ctor_t, llvm_state s)
    : m_i_data(std::make_unique<i_data>(std::move(s)))
{
}

template <typename T>
void taylor_adaptive_batch<T>::finalise_ctor_impl(sys_t vsys, std::vector<T> state, const std::uint32_t batch_size,
                                                  std::vector<T> time, std::optional<T> tol, const bool high_accuracy,
                                                  const bool compact_mode, std::vector<T> pars,
                                                  std::vector<t_event_t> tes, std::vector<nt_event_t> ntes,
                                                  const bool parallel_mode, const bool parjit)
{
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

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pars);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_high_accuracy);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_compact_mode);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tol);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_order);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_ta_jit_data);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_tplt_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pinf);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_minf);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_delta_ts);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_step_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_prop_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_ts_count);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_min_abs_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_max_abs_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_cur_max_delta_ts);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pfor_ts);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_t_dir);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_rem_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_nf_detected);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_vsys);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_vsys_data);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tape_sa);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_cm_tape);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_ed_data);

    auto &m_llvm_state = m_ta_jit_data->m_llvm_state;
    auto &m_d_out_f = m_ta_jit_data->m_d_out_f;

    // Init the data members.
    m_batch_size = batch_size;
    m_time_hi = std::move(time);
    m_time_lo.resize(m_time_hi.size());
    m_high_accuracy = high_accuracy;
    m_compact_mode = compact_mode;

    // Check several input params.
    if (m_batch_size == 0u) [[unlikely]] {
        throw std::invalid_argument("The batch size in an adaptive Taylor integrator cannot be zero");
    }

    if (state.size() % m_batch_size != 0u) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid size detected in the initialization of an adaptive Taylor "
                        "integrator: the state vector has a size of {}, which is not a multiple of the batch size ({})",
                        state.size(), m_batch_size)); // LCOV_EXCL_LINE
    }

    // Fetch the original number of equations/state variables.
    const auto n_orig_sv = is_variational ? std::get<1>(vsys).get_n_orig_sv()
                                          : boost::numeric_cast<std::uint32_t>(std::get<0>(vsys).size());
    // NOTE: this is ensured by validate_ode_sys().
    assert(n_orig_sv != 0u);

    // Zero init the state vector, if empty.
    if (state.empty()) {
        // NOTE: we will perform further initialisation for the variational quantities
        // at a later stage, if needed.
        state.resize(boost::safe_numerics::safe<decltype(state.size())>(n_orig_sv) * m_batch_size);
    }

    // Assign the state.
    m_state = std::move(state);

    // NOTE: keep track of whether or not we need to automatically setup the initial
    // conditions in a variational integrator. This is needed because we need
    // to delay the automatic ic setup for the derivatives wrt the initial time until
    // after we have correctly set up state, pars and time in the integrator.
    bool auto_ic_setup = false;
    if (m_state.size() / m_batch_size != sys.size()) {
        if (is_variational) {
            // NOLINTNEXTLINE(readability-inconsistent-ifelse-braces)
            if (m_state.size() / m_batch_size == n_orig_sv) [[likely]] {
                // Automatic setup of the initial conditions for the derivatives wrt
                // variables and parameters.
                detail::setup_variational_ics_varpar(m_state, std::get<1>(vsys), m_batch_size);
                auto_ic_setup = true;
            } else {
                throw std::invalid_argument(fmt::format(
                    "Inconsistent sizes detected in the initialization of a variational adaptive Taylor "
                    "integrator in batch mode: the state vector has a dimension of {} (in batches of {}), while the "
                    "total number of equations is {}. The size of the state vector must be "
                    "equal either to the total number of equations times the batch size, or to the number of original "
                    "(i.e., non-variational) equations, which for this system is {}, times the batch size",
                    m_state.size(), m_batch_size, sys.size(), n_orig_sv));
            }
            // NOLINTNEXTLINE(readability-inconsistent-ifelse-braces)
        } else [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                            "integrator: the state vector has a dimension of {} and a batch size of {}, "
                            "while the number of equations is {}",
                            m_state.size() / m_batch_size, m_batch_size, sys.size()));
        }
    }

    if (m_time_hi.size() != m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Invalid size detected in the initialization of an adaptive Taylor "
                        "integrator: the time vector has a size of {}, which is not equal to the batch size ({})",
                        m_time_hi.size(), m_batch_size));
    }

    if (tol && (!isfinite(*tol) || *tol < 0)) {
        throw std::invalid_argument(fmt::format(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is {} instead",
            detail::fp_to_string(*tol)));
    }

    if (parallel_mode && !compact_mode) {
        throw std::invalid_argument("Parallel mode can be activated only in conjunction with compact mode");
    }

    // Store the tolerance.
    if (tol) {
        m_tol = *tol;
    } else {
        m_tol = std::numeric_limits<T>::epsilon();
    }

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Compute the total number of params, including the rhs and the event functions.
    const auto tot_n_pars = detail::tot_n_pars_in_ode_sys(sys, tes, ntes);
    // In compact mode, we need to ensure that we can index into the array of parameter values
    // using std::uint32_t.
    // NOTE: in default mode the check is done inside taylor_codegen_numparam()
    // during the construction of the IR code.
    // In compact mode we cannot do that, as the determination of the index into
    // par_ptr is done *within* the IR code (compare taylor_codegen_numparam()
    // to taylor_c_diff_numparam_codegen()).
    // LCOV_EXCL_START
    if (m_compact_mode && tot_n_pars > std::numeric_limits<std::uint32_t>::max() / m_batch_size) [[unlikely]] {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives in compact mode");
    }
    // LCOV_EXCL_STOP

    // Check/setup pars.
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;
    const auto pars_req_size = su32_t(tot_n_pars) * m_batch_size;
    if (pars.empty()) {
        pars.resize(pars_req_size);
        // NOLINTNEXTLINE(readability-inconsistent-ifelse-braces)
    } else if (pars.size() != pars_req_size) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid number of parameter values passed to the constructor of an adaptive "
                        "Taylor integrator in batch mode: {} parameter value(s) were passed, but the ODE "
                        "system contains {} parameter(s) (in batches of {})",
                        pars.size(), tot_n_pars, m_batch_size));
    }

    // Assign pars.
    m_pars = std::move(pars);

    // Do we have events?
    const auto with_events = !tes.empty() || !ntes.empty();

    // Determine the order from the tolerance.
    m_order = detail::taylor_order_from_tol(m_tol);

    // Determine the external fp type.
    auto *ext_fp_t = detail::to_external_llvm_type<T>(std::get<0>(m_llvm_state).context());

    // Determine the internal fp type.
    auto *fp_t = detail::internal_llvm_type_like(std::get<0>(m_llvm_state), m_tol);

    // The state(s) which will be returned by the construction of the stepper function.
    // If we are not in compact mode, this vector will remain empty.
    std::vector<llvm_state> states;

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

        std::tie(m_dc, m_tape_sa, states)
            = detail::taylor_add_adaptive_step_with_events(std::get<0>(m_llvm_state), fp_t, "step_e", sys, batch_size,
                                                           compact_mode, ee, high_accuracy, parallel_mode, m_order);
    } else {
        std::tie(m_dc, m_tape_sa, states)
            = detail::taylor_add_adaptive_step(std::get<0>(m_llvm_state), ext_fp_t, fp_t, "step", sys, batch_size,
                                               high_accuracy, compact_mode, parallel_mode, m_order);
    }

    // Log runtimes in trace mode.
    spdlog::stopwatch sw;

    // Add the function for the computation of the dense output.
    // NOTE: in compact mode, the dense output function will be added to the main state.
    detail::taylor_add_d_out_function(std::get<0>(m_llvm_state), ext_fp_t, m_dim, m_order, m_batch_size, high_accuracy);

    detail::get_logger()->trace("Taylor batch dense output runtime: {}", sw);
    sw.reset();

    // Run the jit compilation.
    if (compact_mode) {
        // Add the main state to the list of states.
        states.push_back(std::move(std::get<0>(m_llvm_state)));

        // Reverse the list of states so that we start with the
        // compilation of the main state first, which may be bigger.
        std::ranges::reverse(states);

        // Create the multi state and assign it.
        m_llvm_state = llvm_multi_state(std::move(states), parjit);

        // Compile.
        std::get<1>(m_llvm_state).compile();

        // Create the storage for the tape of derivatives.
        const auto [sz, al] = m_tape_sa;
        m_cm_tape = detail::aligned_vector<std::byte>(detail::aligned_allocator<std::byte>(al));
        m_cm_tape.resize(boost::numeric_cast<decltype(m_cm_tape.size())>(sz));
    } else {
        std::get<0>(m_llvm_state).compile();
    }

    detail::get_logger()->trace("Taylor batch LLVM compilation runtime: {}", sw);

    // Fetch the stepper.
    assign_stepper(with_events);

    // Fetch the function to compute the dense output.
    m_d_out_f = std::visit(
        [](auto &s) { return reinterpret_cast<detail::ta_jit_data<T>::d_out_f_t>(s.jit_lookup("d_out_f")); },
        m_llvm_state);

    // Setup the vector for the Taylor coefficients.
    // NOTE: the size of m_state.size() already takes
    // into account the batch size.
    m_tc.resize(m_state.size() * (su32_t(m_order) + 1));

    // Setup m_last_h.
    m_last_h.resize(boost::numeric_cast<decltype(m_last_h.size())>(batch_size));

    // Setup the vector for the dense output.
    // NOTE: the size of m_state.size() already takes
    // into account the batch size.
    m_d_out.resize(m_state.size());

    // Prepare the temp vectors.
    m_pinf.resize(m_batch_size, std::numeric_limits<T>::infinity());
    m_minf.resize(m_batch_size, -std::numeric_limits<T>::infinity());
    // LCOV_EXCL_START
    if (m_batch_size > std::numeric_limits<std::uint32_t>::max() / 2u) {
        throw std::overflow_error("Overflow detected in the initialisation of an adaptive Taylor integrator in batch "
                                  "mode: the batch size is too large");
    }
    // LCOV_EXCL_STOP
    // NOTE: make it twice as big because we need twice the storage in step_impl().
    m_delta_ts.resize(boost::numeric_cast<decltype(m_delta_ts.size())>(m_batch_size * 2u));

    // NOTE: init the outcome to success, the rest to zero.
    m_step_res.resize(boost::numeric_cast<decltype(m_step_res.size())>(m_batch_size),
                      std::tuple{taylor_outcome::success, static_cast<T>(0)});
    m_prop_res.resize(
        boost::numeric_cast<decltype(m_prop_res.size())>(m_batch_size),
        std::tuple{taylor_outcome::success, static_cast<T>(0), static_cast<T>(0), static_cast<std::size_t>(0)});

    // Prepare the internal buffers.
    m_ts_count.resize(boost::numeric_cast<decltype(m_ts_count.size())>(m_batch_size));
    m_min_abs_h.resize(m_batch_size);
    m_max_abs_h.resize(m_batch_size);
    m_cur_max_delta_ts.resize(m_batch_size);
    m_pfor_ts.resize(boost::numeric_cast<decltype(m_pfor_ts.size())>(m_batch_size));
    m_t_dir.resize(boost::numeric_cast<decltype(m_t_dir.size())>(m_batch_size));
    m_rem_time.resize(m_batch_size);
    m_time_copy_hi.resize(m_batch_size);
    m_time_copy_lo.resize(m_batch_size);
    m_nf_detected.resize(m_batch_size);

    m_d_out_time.resize(m_batch_size);

    // Init the event data structure if needed.
    //
    // NOTE: in principle this can be done in parallel with the rest of the constructor, once we have
    // m_order/m_dim/m_batch_size and we are done using tes/ntes.
    if (with_events) {
        m_ed_data = std::make_unique<detail::ed_data_batch<T>>(m_tplt_state.make_similar(), std::move(tes),
                                                               std::move(ntes), m_order, m_dim, m_batch_size);
    }

    if (auto_ic_setup) {
        // Finish the automatic setup of the ics for a variational
        // integrator.
        detail::setup_variational_ics_t0(m_tplt_state, m_state, m_pars, m_time_hi.data(), std::get<1>(vsys),
                                         m_batch_size, m_high_accuracy);
    }

    if (is_variational) {
        m_vsys_data = std::make_unique<detail::vsys_data<T>>(std::get<1>(vsys), 0, m_tplt_state, m_batch_size,
                                                             high_accuracy, compact_mode, parjit);
    }

    // Move vsys in.
    m_vsys = std::move(vsys);
}

template <typename T>
taylor_adaptive_batch<T>::taylor_adaptive_batch() noexcept = default;

template <typename T>
taylor_adaptive_batch<T>::taylor_adaptive_batch(const taylor_adaptive_batch &other)
    : m_i_data(std::make_unique<i_data>(*other.m_i_data))
{
}

template <typename T>
taylor_adaptive_batch<T>::taylor_adaptive_batch(taylor_adaptive_batch &&) noexcept = default;

template <typename T>
taylor_adaptive_batch<T> &taylor_adaptive_batch<T>::operator=(const taylor_adaptive_batch &other)
{
    if (this != &other) {
        *this = taylor_adaptive_batch(other);
    }

    return *this;
}

template <typename T>
taylor_adaptive_batch<T> &taylor_adaptive_batch<T>::operator=(taylor_adaptive_batch &&) noexcept = default;

template <typename T>
taylor_adaptive_batch<T>::~taylor_adaptive_batch() = default;

template <typename T>
bool taylor_adaptive_batch<T>::is_variational() const noexcept
{
    return m_i_data->m_vsys.index() == 1u;
}

template <typename T>
// NOLINTNEXTLINE(bugprone-exception-escape)
std::uint32_t taylor_adaptive_batch<T>::get_n_orig_sv() const noexcept
{
    return is_variational() ? std::get<1>(m_i_data->m_vsys).get_n_orig_sv() : m_i_data->m_dim;
}

// NOTE: the save/load patterns mimic the copy constructor logic.
template <typename T>
template <typename Archive>
void taylor_adaptive_batch<T>::save_impl(Archive &ar, unsigned) const
{
    ar << m_i_data;
}

template <typename T>
template <typename Archive>
void taylor_adaptive_batch<T>::load_impl(Archive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<taylor_adaptive_batch<T>>::type::value)) {
        throw std::invalid_argument(fmt::format("Unable to load a taylor_adaptive_batch integrator: "
                                                "the archive version ({}) is too old",
                                                version));
    }
    // LCOV_EXCL_STOP

    try {
        ar >> m_i_data;

        // Recover the stepper.
        //
        // NOTE: Boost serialisation tracks shared_ptr instances - if multiple integrators sharing the same
        // m_ta_jit_data are serialised into the same archive, upon deserialisation Boost will deserialise the object
        // once and give all integrators a shared_ptr to the same instance. Each integrator's load path will then
        // redundantly re-assign the same function pointers to the same object. This is harmless because Boost
        // deserialisation is strictly sequential (so no thread-safety concerns) and the re-assigned values are
        // identical.
        assign_stepper(static_cast<bool>(m_i_data->m_ed_data));
        // LCOV_EXCL_START
    } catch (...) {
        // Reset to def-cted state in case of exceptions.
        *this = taylor_adaptive_batch{};

        throw;
    }
    // LCOV_EXCL_STOP
}

template <typename T>
void taylor_adaptive_batch<T>::save(boost::archive::binary_oarchive &ar, unsigned v) const
{
    save_impl(ar, v);
}

template <typename T>
void taylor_adaptive_batch<T>::load(boost::archive::binary_iarchive &ar, unsigned v)
{
    load_impl(ar, v);
}

template <typename T>
void taylor_adaptive_batch<T>::set_time(const std::vector<T> &new_time)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);

    // Check the dimensionality of new_time.
    if (new_time.size() != m_batch_size) {
        throw std::invalid_argument(fmt::format(
            "Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified times is {}",
            m_batch_size, new_time.size()));
    }

    // Copy over the new times.
    // NOTE: do not use std::copy(), as it gives UB if new_time
    // and m_time_hi are the same object.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_time_hi[i] = new_time[i];
    }
    // Reset the lo part.
    std::fill(m_time_lo.begin(), m_time_lo.end(), static_cast<T>(0));
}

template <typename T>
void taylor_adaptive_batch<T>::set_time(T new_time)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);

    // Set the hi part.
    std::fill(m_time_hi.begin(), m_time_hi.end(), new_time);
    // Reset the lo part.
    std::fill(m_time_lo.begin(), m_time_lo.end(), static_cast<T>(0));
}

template <typename T>
void taylor_adaptive_batch<T>::set_dtime(const std::vector<T> &hi, const std::vector<T> &lo)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);

    // Check the dimensionalities.
    if (hi.size() != m_batch_size || lo.size() != m_batch_size) {
        throw std::invalid_argument(fmt::format(
            "Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified times is ({}, {})",
            m_batch_size, hi.size(), lo.size()));
    }

    // Check the values in hi/lo.
    // NOTE: do it before ever touching m_time_hi/lo for exception safety.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        detail::dtime_checks(hi[i], lo[i]);
    }

    // Copy over the new times, ensuring proper
    // normalisation.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        const auto tmp = normalise(detail::dfloat<T>(hi[i], lo[i]));

        m_time_hi[i] = tmp.hi;
        m_time_lo[i] = tmp.lo;
    }
}

template <typename T>
void taylor_adaptive_batch<T>::set_dtime(T hi, T lo)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);

    // Check the components.
    detail::dtime_checks(hi, lo);

    // Copy over the new time, ensuring proper
    // normalisation.
    const auto tmp = normalise(detail::dfloat<T>(hi, lo));
    std::fill(m_time_hi.begin(), m_time_hi.end(), tmp.hi);
    std::fill(m_time_lo.begin(), m_time_lo.end(), tmp.lo);
}

// Implementation detail to make a single integration timestep.
// The magnitude of the timestep is automatically deduced for each
// state vector, but it will always be not greater than
// the absolute value of the corresponding element in max_delta_ts.
// For each state vector, the propagation
// is done forward in time if max_delta_t >= 0, backwards in
// time otherwise.
//
// The function will write to res a pair for each state
// vector, containing a flag describing the outcome of the integration
// and the integration timestep that was used.
//
// NOTE: for the docs:
// - document exception rethrowing behaviour when an event
//   callback throws;
// - outcome for each batch element:
//   - if nf state is detected, err_nf_state, else
//   - if terminal events trigger, return the index
//     of the first event triggering, else
//   - either time_limit or success, depending on whether
//     max_delta_t was used as a timestep or not;
// - the docs for the scalar step function are applicable to
//   the batch version too.
template <typename T>
void taylor_adaptive_batch<T>::step_impl(const std::vector<T> &max_delta_ts, bool wtc)
{
    using std::abs;
    using std::isfinite;

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pars);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_tol);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_order);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_ta_jit_data);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_delta_ts);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_step_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_nf_detected);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_cm_tape);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_ed_data);

    auto &m_step_f = m_ta_jit_data->m_step_f;
    auto &m_d_out_f = m_ta_jit_data->m_d_out_f;

    // LCOV_EXCL_START
    // Check preconditions.
    assert(max_delta_ts.size() == m_batch_size);
    assert(std::none_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
        using std::isnan;
        return isnan(x);
    }));

    // Sanity check.
    assert(m_step_res.size() == m_batch_size);
    // LCOV_EXCL_STOP

    // Copy max_delta_ts to the tmp buffer, twice.
    std::copy(max_delta_ts.begin(), max_delta_ts.end(), m_delta_ts.data());
    std::copy(max_delta_ts.begin(), max_delta_ts.end(), m_delta_ts.data() + m_batch_size);

    // Helper to check if the state vector of a batch element
    // contains a non-finite value.
    const auto check_nf_batch = [&](std::uint32_t batch_idx) {
        for (std::uint32_t i = 0; i < m_dim; ++i) {
            if (!isfinite(m_state[i * m_batch_size + batch_idx])) {
                return true;
            }
        }
        return false;
    };

    if (m_step_f.index() == 0u || m_step_f.index() == 2u) {
        assert(!m_ed_data); // LCOV_EXCL_LINE

        // Invoke the vanilla stepper.
        if (m_step_f.index() == 0u) {
            std::get<0>(m_step_f)(m_state.data(), m_pars.data(), m_time_hi.data(), m_delta_ts.data(),
                                  wtc ? m_tc.data() : nullptr);
        } else {
            assert(m_cm_tape.size() == m_i_data->m_tape_sa[0]);
            assert(m_cm_tape.get_allocator().get_alignment() == m_i_data->m_tape_sa[1]);

            std::get<2>(m_step_f)(m_state.data(), m_pars.data(), m_time_hi.data(), m_delta_ts.data(),
                                  wtc ? m_tc.data() : nullptr, m_cm_tape.data());
        }

        // Update the times and the last timesteps, and write out the result.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // The timestep that was actually used for
            // this batch element.
            const auto h = m_delta_ts[i];

            // Compute the new time in double-length arithmetic.
            const auto new_time = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + h;
            m_time_hi[i] = new_time.hi;
            m_time_lo[i] = new_time.lo;

            // Update the size of the last timestep.
            m_last_h[i] = h;

            if (!isfinite(new_time) || check_nf_batch(i)) {
                // Either the new time or state contain non-finite values,
                // return an error condition.
                m_step_res[i] = std::tuple{taylor_outcome::err_nf_state, h};
            } else {
                m_step_res[i] = std::tuple{
                    // NOTE: use here the original value of
                    // max_delta_ts, stored at the end of m_delta_ts,
                    // in case max_delta_ts aliases integrator data
                    // which was modified during the step.
                    h == m_delta_ts[m_batch_size + i] ? taylor_outcome::time_limit : taylor_outcome::success,
                    h,
                };
            }
        }
    } else {
        assert(m_ed_data); // LCOV_EXCL_LINE

        auto &edd = *m_ed_data;

        // Invoke the stepper for event handling. We will record the norm infinity of the state vector +
        // event equations at the beginning of the timestep for later use.
        if (m_step_f.index() == 1u) {
            std::get<1>(m_step_f)(edd.m_ev_jet.data(), m_state.data(), m_pars.data(), m_time_hi.data(),
                                  m_delta_ts.data(), edd.m_max_abs_state.data());
        } else {
            assert(m_cm_tape.size() == m_i_data->m_tape_sa[0]);
            assert(m_cm_tape.get_allocator().get_alignment() == m_i_data->m_tape_sa[1]);

            std::get<3>(m_step_f)(edd.m_ev_jet.data(), m_state.data(), m_pars.data(), m_time_hi.data(),
                                  m_delta_ts.data(), edd.m_max_abs_state.data(), m_cm_tape.data());
        }

        // Compute the maximum absolute error on the Taylor series of the event equations, which we will use for
        // automatic cooldown deduction. If max_abs_state is not finite, set it to inf so that
        // in edd.detect_events() we skip event detection altogether.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto max_abs_state = edd.m_max_abs_state[i];

            if (isfinite(max_abs_state)) {
                // Are we in absolute or relative error control mode?
                const auto abs_or_rel = max_abs_state < 1;

                // Estimate the size of the largest remainder in the Taylor
                // series of both the dynamical equations and the events.
                const auto max_r_size = abs_or_rel ? m_tol : (m_tol * max_abs_state);

                // NOTE: depending on m_tol, max_r_size is arbitrarily small, but the real
                // integration error cannot be too small due to floating-point truncation.
                // This is the case for instance if we use sub-epsilon integration tolerances
                // to achieve Brouwer's law. In such a case, we cap the value of g_eps,
                // using eps * max_abs_state as an estimation of the smallest number
                // that can be resolved with the current floating-point type.
                // NOTE: the if condition in the next line is equivalent, in relative
                // error control mode, to:
                // if (m_tol < std::numeric_limits<T>::epsilon())
                if (max_r_size < std::numeric_limits<T>::epsilon() * max_abs_state) {
                    edd.m_g_eps[i] = std::numeric_limits<T>::epsilon() * max_abs_state;
                } else {
                    edd.m_g_eps[i] = max_r_size;
                }
            } else {
                edd.m_g_eps[i] = std::numeric_limits<T>::infinity();
            }
        }

        // Write unconditionally the tcs.
        std::copy(edd.m_ev_jet.data(), edd.m_ev_jet.data() + m_dim * (m_order + 1u) * m_batch_size, m_tc.data());

        // Do the event detection.
        edd.detect_events(m_delta_ts.data(), m_order, m_dim, m_batch_size);

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
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto cmp
                = [](const auto &ev0, const auto &ev1) { return abs(std::get<1>(ev0)) < abs(std::get<1>(ev1)); };
            std::sort(edd.m_d_tes[i].begin(), edd.m_d_tes[i].end(), cmp);
            std::sort(edd.m_d_ntes[i].begin(), edd.m_d_ntes[i].end(), cmp);

            // If we have terminal events we need
            // to update the value of h.
            if (!edd.m_d_tes[i].empty()) {
                m_delta_ts[i] = std::get<1>(edd.m_d_tes[i][0]);
            }
        }

        // Update the state.
        m_d_out_f(m_state.data(), edd.m_ev_jet.data(), m_delta_ts.data());

        // We will use this to capture exceptions thrown while
        // executing callbacks.
        std::vector<std::pair<std::uint32_t, std::exception_ptr>> cb_eptrs;

        // Make a copy of the current times before invoking the callbacks, so that:
        // - we can notice if the callbacks change the time coordinate, and,
        // - if they do, we will use the original time coordinate in the
        //   time update logic, rather than the (wrong) time coordinate
        //   that a callback invocation might set.
        // In other words, m_time_copy will end up containing the correctly-updated
        // time coordinate, while m_time is subject to arbitrary changes via callbacks.
        std::copy(m_time_hi.begin(), m_time_hi.end(), m_time_copy_hi.begin());
        std::copy(m_time_lo.begin(), m_time_lo.end(), m_time_copy_lo.begin());

        // Run the finiteness check on the state vector
        // *before* executing the callbacks, as the execution of
        // a callback on a batch index might alter the state
        // variables of another batch index.
        // NOTE: this can probably be vectorised, if necessary.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_nf_detected[i] = check_nf_batch(i);
        }

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto h = m_delta_ts[i];

            // Compute the new time in double-length arithmetic.
            // NOTE: take the time coordinate from m_time_copy (rather than
            // m_time directly) as m_time might have been modified by incorrectly
            // implemented callbacks.
            const auto new_time = detail::dfloat<T>(m_time_copy_hi[i], m_time_copy_lo[i]) + h;
            m_time_hi[i] = new_time.hi;
            m_time_lo[i] = new_time.lo;

            // Update also m_time_copy with the new time.
            m_time_copy_hi[i] = new_time.hi;
            m_time_copy_lo[i] = new_time.lo;

            // Store the last timestep.
            m_last_h[i] = h;

            // Check if the time or the state vector are non-finite at the
            // end of the timestep.
            if (!isfinite(new_time) || m_nf_detected[i] != 0) {
                m_step_res[i] = std::tuple{taylor_outcome::err_nf_state, h};

                // Move to the next batch element.
                continue;
            }

            // Update the cooldowns.
            for (auto &cd : edd.m_te_cooldowns[i]) {
                if (cd) {
                    // Check if the timestep we just took
                    // brought this event outside the cooldown.
                    const auto tmp = cd->first + h;

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
                = edd.m_d_tes[i].empty()
                      ? edd.m_d_ntes[i].end()
                      : std::lower_bound(edd.m_d_ntes[i].begin(), edd.m_d_ntes[i].end(), h,
                                         [](const auto &ev, const auto &t) { return abs(std::get<1>(ev)) < abs(t); });

            // Flag to signal that the callback of a non-terminal event threw
            // an exception.
            bool nt_cb_exception = false;

            // Invoke the callbacks of the non-terminal events, which are guaranteed
            // to happen before the first terminal event.
            // NOTE: the loop will be exited as soon as an exception is raised
            // by a callback. This is intended to match the behaviour of the
            // scalar integrator (i.e., do not process any more events for the current
            // batch element and move to the next batch element instead).
            for (auto it = edd.m_d_ntes[i].begin(); it != ntes_end_it; ++it) {
                const auto &t = *it;
                auto &cb = edd.m_ntes[std::get<0>(t)].get_callback();
                assert(cb); // LCOV_EXCL_LINE
                try {
                    cb(*this, static_cast<T>(new_time - m_last_h[i] + std::get<1>(t)), std::get<2>(t), i);
                } catch (...) {
                    // NOTE: in case of exception, remember it, set nt_cb_exception
                    // to true and break out, so that we do not process additional events.
                    cb_eptrs.emplace_back(i, std::current_exception());
                    nt_cb_exception = true;
                    break;
                }
            }

            // Don't proceed to the terminal events if a non-terminal
            // callback threw, just move to the next batch element. This
            // is intended to match the behaviour of the
            // scalar integrator.
            if (nt_cb_exception) {
                continue;
            }

            // The return value of the first
            // terminal event's callback. It will be
            // unused if there are no terminal events.
            bool te_cb_ret = false;

            if (!edd.m_d_tes[i].empty()) {
                // Fetch the first terminal event.
                const auto te_idx = std::get<0>(edd.m_d_tes[i][0]);
                assert(te_idx < edd.m_tes.size()); // LCOV_EXCL_LINE
                auto &te = edd.m_tes[te_idx];

                // Set the corresponding cooldown.
                if (te.get_cooldown() >= 0) {
                    // Cooldown explicitly provided by the user, use it.
                    edd.m_te_cooldowns[i][te_idx].emplace(0, te.get_cooldown());
                } else {
                    // Deduce the cooldown automatically.
                    // NOTE: if m_g_eps[i] is not finite, we skipped event detection
                    // altogether and thus we never end up here. If the derivative
                    // of the event equation is not finite, the event is also skipped.
                    edd.m_te_cooldowns[i][te_idx].emplace(
                        0, detail::taylor_deduce_cooldown(edd.m_g_eps[i], std::get<3>(edd.m_d_tes[i][0])));
                }

                // Invoke the callback of the first terminal event, if it has one.
                if (te.get_callback()) {
                    try {
                        te_cb_ret = te.get_callback()(*this, std::get<2>(edd.m_d_tes[i][0]), i);
                    } catch (...) {
                        // NOTE: if an exception is raised, record it
                        // and then move on to the next batch element.
                        // This is intended to match the behaviour of
                        // the scalar integrator.
                        cb_eptrs.emplace_back(i, std::current_exception());
                        continue;
                    }
                }
            }

            if (edd.m_d_tes[i].empty()) {
                // No terminal events detected, return success or time limit.
                m_step_res[i] = std::tuple{
                    // NOTE: use here the original value of
                    // max_delta_ts, stored at the end of m_delta_ts,
                    // in case max_delta_ts aliases integrator data
                    // which was modified during the step.
                    h == m_delta_ts[m_batch_size + i] ? taylor_outcome::time_limit : taylor_outcome::success,
                    h,
                };
            } else {
                // Terminal event detected. Fetch its index.
                const auto ev_idx = static_cast<std::int64_t>(std::get<0>(edd.m_d_tes[i][0]));

                // NOTE: if te_cb_ret is true, it means that the terminal event has
                // a callback and its invocation returned true (meaning that the
                // integration should continue). Otherwise, either the terminal event
                // has no callback or its callback returned false, meaning that the
                // integration must stop.
                m_step_res[i] = std::tuple{taylor_outcome{te_cb_ret ? ev_idx : (-ev_idx - 1)}, h};
            }
        }

        // Check if any callback threw an exception.
        if (!cb_eptrs.empty()) {
            // If there's only 1 exception just rethrow it.
            if (cb_eptrs.size() == 1u) {
                std::rethrow_exception(cb_eptrs[0].second);
            }

            // Otherwise, we will assemble and throw a new exception
            // containing the messages of all thrown exceptions.
            std::string exc_msg = "Two or more exceptions were raised during the execution of event callbacks in a "
                                  "batch integrator:\n\n";

            for (auto &[i, eptr] : cb_eptrs) {
                exc_msg += fmt::format("Batch index #{}:\n", i);

                try {
                    std::rethrow_exception(eptr);
                } catch (const std::exception &ex) {
                    exc_msg += fmt::format("    Exception type: {}\n", boost::core::demangle(typeid(ex).name()));
                    exc_msg += fmt::format("    Exception message: {}\n", ex.what());
                } catch (...) {
                    // LCOV_EXCL_START
                    exc_msg += "    Exception type: unknown\n";
                    exc_msg += "    Exception message: unknown\n";
                    // LCOV_EXCL_STOP
                }

                exc_msg += '\n';
            }

            throw std::runtime_error(exc_msg);
        }

        // NOTE: event callbacks - terminal or not - cannot modify the time
        // variable, as this is going to mess up the internal time keeping logic
        // in the propagate_*() functions.
        // NOTE: use cmp_nan_eq() so that we consider NaN == NaN in the time coordinates.
        // This is necessary in case something goes wrong in the integration step
        // and the time coordinate goes NaN - in such a case, the standard equality operator
        // would trigger (because NaN != NaN) even if the time coordinate was not changed by the callback.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            if (!detail::cmp_nan_eq(m_time_hi[i], m_time_copy_hi[i])
                || !detail::cmp_nan_eq(m_time_lo[i], m_time_copy_lo[i])) {
                throw std::runtime_error(
                    fmt::format("The invocation of one or more event callbacks resulted in the alteration of the "
                                "time coordinate of the integrator at the batch index {} - this is not supported",
                                i));
            }
        }
    }
}

template <typename T>
void taylor_adaptive_batch<T>::step(bool wtc)
{
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_pinf);

    step_impl(m_pinf, wtc);
}

template <typename T>
void taylor_adaptive_batch<T>::step_backward(bool wtc)
{
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_minf);

    step_impl(m_minf, wtc);
}

template <typename T>
void taylor_adaptive_batch<T>::step(const std::vector<T> &max_delta_ts, bool wtc)
{
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_batch_size);

    // Check the dimensionality of max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(fmt::format(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}",
            m_batch_size, max_delta_ts.size()));
    }

    // Make sure no values in max_delta_ts are nan.
    if (std::any_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
            using std::isnan;
            return isnan(x);
        })) {
        throw std::invalid_argument(
            "Cannot invoke the step() function of an adaptive Taylor integrator in batch mode if "
            "one of the max timesteps is nan");
    }

    step_impl(max_delta_ts, wtc);
}

template <typename T>
std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
taylor_adaptive_batch<T>::propagate_for_impl(const pfor_arg_t &delta_ts, std::size_t max_steps,
                                             const std::vector<T> &max_delta_ts, step_callback_batch<T> cb, bool wtc,
                                             bool with_c_out)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pfor_ts);

    if (const auto *scal_ptr = std::get_if<T>(&delta_ts)) {
        // Single duration value: add it to the current times
        // and store the results in m_pfor_ts.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_pfor_ts[i] = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + *scal_ptr;
        }
    } else {
        // Vector of single-length durations.
        const auto &vec = std::get<std::reference_wrapper<const std::vector<T>>>(delta_ts).get();

        // Check the dimensionality of vec.
        if (vec.size() != m_batch_size) {
            throw std::invalid_argument(
                fmt::format("Invalid number of time intervals specified in a Taylor integrator in batch mode: "
                            "the batch size is {}, but the number of specified time intervals is {}",
                            m_batch_size, vec.size()));
        }

        // Add the durations to the current times and store the result
        // in m_pfor_ts.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_pfor_ts[i] = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + vec[i];
        }
    }

    // NOTE: max_delta_ts is checked in propagate_until_impl().
    return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts, std::move(cb), wtc, with_c_out);
}

// NOTE: possible outcomes:
//
// - 1 or more err_nf_state if 1 or more batch elements generated a non-finite state. The other elements will have their
//   outcome set by the last step taken; else,
// - all cb_stop if the integration was interrupted via a callback; else,
// - 1 or more event indices if 1 or more batch elements generated a stopping terminal event. The other elements will
//   have their outcome set by the last step taken; else,
// - all time_limit if all batch elements were propagated up to the final times; else,
// - all step_limit if the max_steps limit was reached.
//
// The callback is always executed at the end of each timestep, unless a non-finite state was detected in any batch
// element. The continuous output is always updated at the end of each timestep, unless a non-finite state was detected
// in any batch element.
template <typename T>
std::tuple<std::optional<continuous_output_batch<T>>, step_callback_batch<T>>
taylor_adaptive_batch<T>::propagate_until_impl(const puntil_arg_t &ts_, const std::size_t max_steps,
                                               const std::vector<T> &max_delta_ts_, step_callback_batch<T> cb,
                                               const bool wtc_, const bool with_c_out)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Small helper to reduce typing below.
    constexpr auto is_finite = [](const auto &x) { return isfinite(x); };

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_high_accuracy);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_order);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tplt_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_pinf);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_step_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_prop_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_ts_count);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_min_abs_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_max_abs_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_cur_max_delta_ts);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_pfor_ts);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_t_dir);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_rem_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_lo);

    assert(m_pfor_ts.size() == m_batch_size);

    // Compute the final times and store them in m_pfor_ts, if they are not already there.
    std::visit(
        [&]<typename V>(const V &v) {
            if constexpr (std::same_as<V, T>) {
                // Scalar final time. Splat the value into m_pfor_ts.
                std::ranges::fill(m_pfor_ts, detail::dfloat<T>(v));
            } else {
                using vec_t = std::remove_cvref_t<std::unwrap_reference_t<V>>;

                if constexpr (std::same_as<T, typename vec_t::value_type>) {
                    // Vector of single-length final times. Convert to double-length and copy into m_pfor_ts.
                    const auto &vec = v.get();

                    // Check the dimensionality of v.
                    if (vec.size() != m_batch_size) [[unlikely]] {
                        throw std::invalid_argument(fmt::format(
                            "Invalid number of time limits specified in a Taylor integrator in batch mode: the "
                            "batch size is {}, but the number of specified time limits is {}",
                            m_batch_size, vec.size()));
                    }

                    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                        m_pfor_ts[i] = detail::dfloat<T>(vec[i]);
                    }
                } else {
                    // Vector of double-length final times.
                    //
                    // NOTE: this is possible only if propagate_until() is called from propagate_for(), in which case v
                    // must point to m_pfor_ts.
                    assert(&v.get() == &m_pfor_ts);
                }
            }
        },
        ts_);

    // The final times are now stored in m_pfor_ts.
    const auto &ts = m_pfor_ts;

    // Set up max_delta_ts.
    const auto &max_delta_ts = max_delta_ts_.empty() ? m_pinf : max_delta_ts_;

    // Check the current times.
    if (!std::ranges::all_of(m_time_hi, is_finite) || !std::ranges::all_of(m_time_lo, is_finite)) [[unlikely]] {
        throw std::invalid_argument(
            "Cannot invoke the propagate_until() function of an adaptive Taylor integrator in batch mode if "
            "one of the current times is not finite");
    }

    // Check the final times.
    if (!std::ranges::all_of(ts, is_finite)) [[unlikely]] {
        throw std::invalid_argument("A non-finite time was passed to the propagate_until() function of an adaptive "
                                    "Taylor integrator in batch mode");
    }

    // Check max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}",
            m_batch_size, max_delta_ts.size()));
    }
    for (const auto &dt : max_delta_ts) {
        if (isnan(dt)) [[unlikely]] {
            throw std::invalid_argument("A nan max_delta_t was passed to the propagate_until() function of an adaptive "
                                        "Taylor integrator in batch mode");
        }
        if (dt <= 0) [[unlikely]] {
            throw std::invalid_argument("A non-positive max_delta_t was passed to the propagate_until() function of an "
                                        "adaptive Taylor integrator in batch mode");
        }
    }

    // If with_c_out is true, we always need to write the Taylor coefficients.
    const auto wtc = wtc_ || with_c_out;

    // These vectors are used in the construction of the continuous output. If continuous output is not requested, they
    // will remain empty.
    std::vector<T> c_out_tcs, c_out_times_hi, c_out_times_lo;
    if (with_c_out) {
        // Push in the starting time.
        c_out_times_hi.insert(c_out_times_hi.end(), m_time_hi.begin(), m_time_hi.end());
        c_out_times_lo.insert(c_out_times_lo.end(), m_time_lo.begin(), m_time_lo.end());
    }

    // Reset the counters and the min/max abs(h) vectors.
    safe_size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    // Double-length zero used in several places below.
    const auto dl_zero = detail::dfloat<T>(static_cast<T>(0));

    // Helper to compute the sign of an input rem_time: -1, 0 or 1.
    const auto sgn_rem_time = [dl_zero](const auto &rem_time) {
        return static_cast<int>(dl_zero < rem_time) - static_cast<int>(dl_zero > rem_time);
    };

    // Compute the integration directions and init the remaining times.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_rem_time[i] = ts[i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);
        if (!is_finite(m_rem_time[i])) [[unlikely]] {
            throw std::invalid_argument("The final time passed to the propagate_until() function of an adaptive Taylor "
                                        "integrator in batch mode results in an overflow condition");
        }

        m_t_dir[i] = sgn_rem_time(m_rem_time[i]);
    }

    // Helper to create the continuous output object.
    const auto make_c_out = [&] -> std::optional<continuous_output_batch<T>> {
        if (with_c_out) {
            if (c_out_times_hi.size() / m_batch_size < 2u) {
                // NOTE: this means that the first step resulted in a non-finite state in at least one batch element.
                return {};
            }

            // Construct the return value.
            continuous_output_batch<T> ret(m_tplt_state.make_similar());

            // Fill in the data.
            ret.m_batch_size = m_batch_size;
            ret.m_tcs = std::move(c_out_tcs);
            ret.m_times_hi = std::move(c_out_times_hi);
            ret.m_times_lo = std::move(c_out_times_lo);

            // Add padding to the times vectors to make the vectorised upper_bound implementation well defined.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                ret.m_times_hi.push_back((m_t_dir[i] >= 0) ? std::numeric_limits<T>::infinity()
                                                           : -std::numeric_limits<T>::infinity());
                ret.m_times_lo.push_back(static_cast<T>(0));
            }

            // Prepare the output vector.
            ret.m_output.resize(boost::numeric_cast<decltype(ret.m_output.size())>(m_dim * m_batch_size));

            // Prepare the temp time vector.
            ret.m_tmp_tm.resize(boost::numeric_cast<decltype(ret.m_tmp_tm.size())>(m_batch_size));

            // Add the continuous output function.
            ret.add_c_out_function(m_order, m_dim, m_high_accuracy);

            return std::optional{std::move(ret)};
        } else {
            return {};
        }
    };

    // Helper to update the continuous output data after a timestep.
    const auto update_c_out = [&] {
        if (with_c_out) {
#if !defined(NDEBUG)
            std::vector<detail::dfloat<T>> prev_times;
            prev_times.reserve(m_batch_size);
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                prev_times.emplace_back(c_out_times_hi[c_out_times_hi.size() - m_batch_size + i],
                                        c_out_times_lo[c_out_times_lo.size() - m_batch_size + i]);
            }
#endif

            c_out_times_hi.insert(c_out_times_hi.end(), m_time_hi.begin(), m_time_hi.end());
            c_out_times_lo.insert(c_out_times_lo.end(), m_time_lo.begin(), m_time_lo.end());

#if !defined(NDEBUG)
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                const detail::dfloat<T> new_time(c_out_times_hi[c_out_times_hi.size() - m_batch_size + i],
                                                 c_out_times_lo[c_out_times_lo.size() - m_batch_size + i]);
                assert(isfinite(new_time)); // LCOV_EXCL_LINE
                if (m_t_dir[i] == 1) {
                    assert(!(new_time < prev_times[i]));
                } else if (m_t_dir[i] == -1) {
                    assert(!(new_time > prev_times[i]));
                }
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
        // NOTE: the callback is not allowed to change the time coordinate (either via pre_hook() or via its call
        // operator, as shown later).
        //
        // NOTE: the hi/lo copies are guaranteed to be finite due to the checks performed earlier.
        std::ranges::copy(m_time_hi, m_time_copy_hi.begin());
        std::ranges::copy(m_time_lo, m_time_copy_lo.begin());

        cb.pre_hook(*this);

        if (m_time_hi != m_time_copy_hi || m_time_lo != m_time_copy_lo) [[unlikely]] {
            throw std::runtime_error(cb_time_errmsg);
        }
    }

    // Small helper to create the return value. It will consume c_out_tcs, c_out_times_hi and c_out_times_lo via
    // make_c_out, and cb.
    const auto make_retval = [&make_c_out, &cb]() { return std::make_tuple(make_c_out(), std::move(cb)); };

    while (true) {
        // Compute the max integration times for this timestep.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // NOTE: m_rem_time[i] is guaranteed to be finite: we check it explicitly both on creation and every time it
            // is updated. We also make sure that its sign at most changes to zero (this signals that a batch element is
            // done).
            assert(sgn_rem_time(m_rem_time[i]) == m_t_dir[i] || m_rem_time[i] == dl_zero);

            // Compute the time limit.
            const auto dt_limit = (m_t_dir[i] >= 0) ? std::min(detail::dfloat<T>(max_delta_ts[i]), m_rem_time[i])
                                                    : std::max(detail::dfloat<T>(-max_delta_ts[i]), m_rem_time[i]);

            // Store it.
            m_cur_max_delta_ts[i] = static_cast<T>(dt_limit);
        }

        // Run the integration timestep, making sure that we don't go past the time limit.
        //
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        step_impl(m_cur_max_delta_ts, wtc);

        // Check the outcomes of the step for each batch element, update the step counters, min_h/max_h and the
        // remaining times, and keep track of:
        //
        // - the number of batch elements which reached the time limit (n_done),
        // - whether or not any non-finite state was detected,
        // - whether or not any stopping terminal event triggered.
        std::uint32_t n_done = 0;
        bool nfs_detected = false, ste_detected = false;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            auto [oc, h] = m_step_res[i];

            if (oc == taylor_outcome::err_nf_state) [[unlikely]] {
                // Non-finite state: flag it and do nothing else.
                nfs_detected = true;
            } else {
                // Step outcome is one of:
                //
                // - success,
                // - time_limit,
                // - terminal event.
                //
                // Update the local step counters.
                //
                // NOTE: a step is counted unless the batch element had already finished before the current step, in
                // which case it is only taking zero-length steps while waiting for the other batch elements. A zero
                // remaining time means the element has finished, except on the first iteration, where it means that
                // the final time is the current time: in that case the zero-length step is counted, consistently with
                // the scalar integrator.
                const auto lane_active = (m_rem_time[i] != dl_zero) || (iter_counter == 0u);
                m_ts_count[i] = safe_size_t(m_ts_count[i]) + static_cast<std::size_t>(lane_active);

                // Update min_h/max_h only if the outcome is success (otherwise the step was artificially clamped either
                // by a time limit or by a terminal event).
                if (oc == taylor_outcome::success) {
                    const auto abs_h = abs(h);
                    m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
                    m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
                }

                // Check if this batch element reached the time limit.
                //
                // NOTE: we check h == rem_time.hi, instead of just oc == time_limit, because clamping via max_delta_t
                // could also result in time_limit.
                const auto cur_time_limit_reached = (h == m_rem_time[i].hi);
                assert(!cur_time_limit_reached || oc == taylor_outcome::time_limit);

                // Flag if we encountered a stopping terminal event.
                const auto cur_ste = (oc > taylor_outcome::success && oc < taylor_outcome{0});
                ste_detected = ste_detected || cur_ste;

                // Compute the new remaining time and check it.
                const auto new_rem_time = ts[i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);
                // NOTE: this is likely impossible to trigger, but the check is cheap.
                if (!is_finite(new_rem_time)) [[unlikely]] {
                    // LCOV_EXCL_START
                    // NOTE: just flag the non-finiteness, fill in m_prop_res[i] and continue to the next batch element.
                    nfs_detected = true;
                    m_prop_res[i]
                        = std::tuple{taylor_outcome::err_nf_state, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
                    continue;
                    // LCOV_EXCL_STOP
                }

                // Detect if rem_time changed sign. This happens if t_dir is not zero and:
                //
                // - we landed exactly on t, or
                // - we overshot t due to floating-point rounding, or
                // - the current batch element had finished already before the current step.
                //
                // A sign flip is regarded the same as reaching the time limit.
                const auto rem_time_sign_flip = (sgn_rem_time(new_rem_time) != m_t_dir[i]);

                // Check if this batch element is done, meaning that it reached its time limit. A sign change in
                // rem_time counts as reaching the time limit.
                const auto cur_done = cur_time_limit_reached || rem_time_sign_flip;
                n_done += cur_done;

                // Update the remaining time, forcibly setting it to zero in case the current batch element is done.
                m_rem_time[i] = cur_done ? dl_zero : new_rem_time;

                // If the current batch element is done, snap the time coordinate to ts[i] and make sure oc is set to
                // time_limit (if we did not run into a stopping terminal event).
                //
                // NOTE: the snapping is done in order to improve the user experience, i.e., if a user asks for a
                // propagation up to t, the final time is guaranteed to be t exactly. Strictly speaking, this can
                // introduce a small error, at the level of the floating-point epsilon. See also the explanation in
                // taylor_adaptive::propagate_until_impl().
                if (cur_done) {
                    m_time_hi[i] = ts[i].hi;
                    m_time_lo[i] = ts[i].lo;

                    // If the current batch element did *not* encounter a stopping terminal event, make sure that oc
                    // contains time_limit.
                    //
                    // NOTE: we need to do this because we could be in a situation in which oc is success but we
                    // detected a rem_time sign flip, which is regarded the same as reaching the time limit.
                    if (!cur_ste) {
                        oc = taylor_outcome::time_limit;
                    }
                }
            }

            // Write into m_prop_res.
            m_prop_res[i] = std::tuple{oc, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }

        if (nfs_detected) [[unlikely]] {
            // At least 1 batch element generated a non-finite state. In this situation, we do *not* want to execute the
            // propagate() callback and we do *not* want to update the continuous output. Just exit.
            return make_retval();
        }

        // Update the continuous output.
        update_c_out();

        // Update the iteration counter.
        ++iter_counter;

        // Execute the propagate() callback, if applicable.
        if (with_cb) {
            // Store the current time coordinate before executing the cb, so that we can check if the cb changes the
            // time coordinate.
            //
            // NOTE: the hi/lo copies are guaranteed to be finite because no non-finite outcomes were detected in the
            // step.
            std::ranges::copy(m_time_hi, m_time_copy_hi.begin());
            std::ranges::copy(m_time_lo, m_time_copy_lo.begin());

            // Execute the cb.
            const auto ret_cb = cb(*this);

            // Check the time coordinate.
            if (m_time_hi != m_time_copy_hi || m_time_lo != m_time_copy_lo) [[unlikely]] {
                throw std::runtime_error(cb_time_errmsg);
            }

            if (!ret_cb) {
                // Change m_prop_res before exiting by setting all outcomes to cb_stop regardless of the timestep
                // outcome.
                for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                    std::get<0>(m_prop_res[i]) = taylor_outcome::cb_stop;
                }

                return make_retval();
            }
        }

        // We need to break out if either we reached the final time for all batch elements, or we encountered at least 1
        // stopping terminal event. In either case, m_prop_res was already set up in the loop where we checked the
        // outcomes.
        if (n_done == m_batch_size || ste_detected) {
            return make_retval();
        }

        // Check the iteration limit.
        //
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps), then this condition will never trigger as,
        // by this point, we are sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            // We reached the max_steps limit: set the outcome for all batch elements to step_limit.
            //
            // NOTE: this is the same logic adopted when the integration is stopped by the callback (see above).
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                std::get<0>(m_prop_res[i]) = taylor_outcome::step_limit;
            }

            return make_retval();
        }
    }
}

// NOTE: possible outcomes:
//
// - the outcomes of the initial propagation up to the first grid points, if they are not all time_limit; else,
// - 1 or more err_nf_state if 1 or more batch elements generated a non-finite state. The other elements will have their
//   outcome set by the last step taken; else,
// - all cb_stop if the integration was interrupted via a callback; else,
// - 1 or more event indices if 1 or more batch elements generated a stopping terminal event. The other elements will
//   have their outcome set by the last step taken; else,
// - all time_limit if all batch elements were propagated up to the last grid points; else,
// - all step_limit if the max_steps limit was reached.
//
// In case of interruption via callback, stopping terminal event or step limit, the grid points falling within the last
// timestep are processed before returning. In the returned vector, the grid points which were not processed are set
// to NaN.
//
// The callback is executed at the end of each timestep taken after the initial propagation up to the first grid points,
// unless a non-finite state was detected in any batch element.
template <typename T>
std::tuple<step_callback_batch<T>, std::vector<T>>
taylor_adaptive_batch<T>::propagate_grid_impl(const std::vector<T> &grid_, const std::size_t max_steps,
                                              const std::vector<T> &max_delta_ts_, step_callback_batch<T> cb)
{
    using std::abs;
    using std::isnan;

    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_state);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_dim);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_pinf);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_step_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_prop_res);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_ts_count);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_min_abs_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_max_abs_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_t_dir);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_rem_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_copy_lo);

    // Helper to detect if an input value is nonfinite.
    constexpr auto is_nf = [](const auto &t) {
        using std::isfinite;
        return !isfinite(t);
    };

    if (grid_.empty()) [[unlikely]] {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if the time grid is empty");
    }

    // Check that the grid size is a multiple of m_batch_size.
    if (grid_.size() % m_batch_size != 0u) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid grid size detected in propagate_grid() for an adaptive Taylor integrator in batch mode: "
            "the grid has a size of {}, which is not a multiple of the batch size ({})",
            grid_.size(), m_batch_size));
    }

    // Check the current time coordinates.
    if (std::ranges::any_of(m_time_hi, is_nf) || std::ranges::any_of(m_time_lo, is_nf)) [[unlikely]] {
        throw std::invalid_argument("Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if "
                                    "the current time is not finite");
    }

    // Set up max_delta_ts.
    const auto &max_delta_ts = max_delta_ts_.empty() ? m_pinf : max_delta_ts_;

    // Check max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}",
            m_batch_size, max_delta_ts.size()));
    }
    for (const auto &dt : max_delta_ts) {
        if (isnan(dt)) [[unlikely]] {
            throw std::invalid_argument("A nan max_delta_t was passed to the propagate_grid() function of an adaptive "
                                        "Taylor integrator in batch mode");
        }
        if (dt <= 0) [[unlikely]] {
            throw std::invalid_argument("A non-positive max_delta_t was passed to the propagate_grid() function of an "
                                        "adaptive Taylor integrator in batch mode");
        }
    }

    // The number of batched grid points.
    const auto n_grid_points = grid_.size() / m_batch_size;

    // Create a span over the grid for ease of access.
    using grid_size_t = decltype(grid_.size());
    const mdspan<const T, dextents<grid_size_t, 2>> grid_span(grid_.data(), n_grid_points, m_batch_size);

    // Small utility to extract a row from grid_span as a std::span, so we can run range algorithms.
    const auto grid_row = [grid_span, n_grid_points](const grid_size_t i) {
        static_cast<void>(n_grid_points);
        assert(i < n_grid_points);

        return std::span(&grid_span[i, 0], boost::numeric_cast<std::size_t>(grid_span.extent(1)));
    };

    // Check the input grid points.
    constexpr auto nf_err_msg
        = "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator in batch mode";
    constexpr auto ig_err_msg = "A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                "Taylor integrator in batch mode";

    // Check the first row for finiteness.
    if (std::ranges::any_of(grid_row(0), is_nf)) [[unlikely]] {
        throw std::invalid_argument(nf_err_msg);
    }

    if (n_grid_points > 1u) {
        // Establish the direction of the grid from the first elements of the first two batches of points.
        if (std::ranges::any_of(grid_row(1), is_nf)) [[unlikely]] {
            throw std::invalid_argument(nf_err_msg);
        }
        if (grid_span[1, 0] == grid_span[0, 0]) [[unlikely]] {
            throw std::invalid_argument(ig_err_msg);
        }
        const auto grid_direction = grid_span[1, 0] > grid_span[0, 0];

        // Small helper to check the monotonicity of the grid. Given row/column indices i/j in the grid, it will compare
        // the value at [i, j] with the value at [i-1, j], and check that the two values differ and that they obey the
        // grid direction.
        const auto check_monotonicity = [grid_direction, grid_span](const grid_size_t i, const grid_size_t j) {
            assert(i > 0u);
            const auto cur_value = grid_span[i, j];
            const auto prev_value = grid_span[i - 1u, j];

            if ((cur_value == prev_value) || (cur_value > prev_value) != grid_direction) [[unlikely]] {
                throw std::invalid_argument(ig_err_msg);
            }
        };

        // Check monotonicity for the remaining batch lanes in the first two batches.
        for (std::uint32_t j = 1; j < m_batch_size; ++j) {
            check_monotonicity(1, j);
        }

        // Check that the remaining points are finite and that they are ordered monotonically.
        for (grid_size_t i = 2; i < n_grid_points; ++i) {
            for (std::uint32_t j = 0; j < m_batch_size; ++j) {
                if (is_nf(grid_span[i, j])) [[unlikely]] {
                    throw std::invalid_argument(nf_err_msg);
                }
                check_monotonicity(i, j);
            }
        }
    }

    // Require that the user provides a grid starting from the current integrator time, modulo the double-length
    // correction.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        if (m_time_hi[i] != grid_span[0, i]) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("When invoking propagate_grid(), the first element of the time grid must match the current "
                            "time coordinate - however, the first element of the time grid at batch index {} has a "
                            "value of {}, while the current time coordinate is {}",
                            i, grid_span[0, i], m_time_hi[i]));
        }
    }

    // Pre-allocate the return value.
    std::vector<T> retval;
    // NOTE: fill with NaNs, so that the missing entries are signalled with NaN if we exit early.
    retval.resize(boost::safe_numerics::safe<decltype(retval.size())>(grid_.size()) * get_dim(),
                  std::numeric_limits<T>::quiet_NaN());

    // Define a span over retval for ease of indexing.
    const mdspan<T, dextents<decltype(retval.size()), 3>> ret_span(retval.data(), n_grid_points, get_dim(),
                                                                   m_batch_size);

    // Define a span over m_d_out for ease of indexing.
    const mdspan<const T, dextents<decltype(m_d_out.size()), 2>> d_out_span(m_d_out.data(), get_dim(), m_batch_size);

    // NOTE: this is a buffer of size m_batch_size that is used in various places as temp storage.
    std::vector<T> pgrid_tmp;
    pgrid_tmp.resize(boost::numeric_cast<decltype(pgrid_tmp.size())>(m_batch_size));

    // Small helper to create the return value. It will consume cb and retval.
    const auto make_retval = [&cb, &retval]() { return std::make_tuple(std::move(cb), std::move(retval)); };

    // Propagate the system up to the first batch of grid points. This is necessary in order to account for the fact
    // that the user cannot pass as starting point in the grid a time coordinate which is *exactly* equal to m_time, due
    // to the usage of a double-length representation.
    //
    // NOTE: if the integrator time is already *exactly* at the first grid points, then propagate_until() will take a
    // zero timestep, resulting in no event detection being performed.
    //
    // NOTE: use the same max_steps for the initial propagation, and don't pass the callback.
    std::ranges::copy(grid_row(0), pgrid_tmp.begin());
    propagate_until(pgrid_tmp, kw::max_delta_t = max_delta_ts, kw::max_steps = max_steps);

    // Check the result of the integration.
    if (std::ranges::any_of(m_prop_res, [](const auto &t) { return std::get<0>(t) != taylor_outcome::time_limit; }))
        [[unlikely]] {
        // NOTE: for consistency with the scalar implementation, keep the outcomes from propagate_until() but reset
        // min/max h and the step counter.
        for (auto &[_, min_h, max_h, ts_count] : m_prop_res) {
            min_h = std::numeric_limits<T>::infinity();
            max_h = 0;
            ts_count = 0;
        }

        return make_retval();
    }

    // Add the first result to retval.
    std::ranges::copy(m_state, retval.begin());

    // Double-length zero used in several places below.
    const auto dl_zero = detail::dfloat<T>(static_cast<T>(0));

    // Helper to compute the sign of an input rem_time: -1, 0 or 1.
    const auto sgn_rem_time = [dl_zero](const auto &rem_time) {
        return static_cast<int>(dl_zero < rem_time) - static_cast<int>(dl_zero > rem_time);
    };

    // Init m_rem_time and m_t_dir.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_rem_time[i] = grid_span[n_grid_points - 1u, i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);

        // Check it.
        if (is_nf(m_rem_time[i])) [[unlikely]] {
            throw std::invalid_argument("The final time passed to the propagate_grid() function of an adaptive Taylor "
                                        "integrator in batch mode results in an overflow condition");
        }

        m_t_dir[i] = sgn_rem_time(m_rem_time[i]);
        assert((m_t_dir[i] == 0) == (n_grid_points == 1u));
    }

    // Reset the counters and the min/max abs(h) vectors. Also reset m_prop_res, so that, for a single-point grid, the
    // propagation results report zero steps (like in the scalar version), instead of the results of the initial
    // propagate_until().
    safe_size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
        m_prop_res[i] = std::tuple{taylor_outcome::time_limit, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
    }

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

    // Error message for when the callback modifies the time coordinate.
    constexpr auto cb_time_errmsg
        = "The invocation of the callback passed to propagate_grid() resulted in the alteration of the "
          "time coordinate of the integrator - this is not supported";

    // Run the callback's pre_hook() function, if needed.
    if (with_cb) {
        // NOTE: the callback is not allowed to change the time coordinate (either via pre_hook() or via its call
        // operator, as shown later).
        //
        // NOTE: the hi/lo copies are guaranteed to contain finite values because propagate_until() did not return
        // err_nf_state.
        std::ranges::copy(m_time_hi, m_time_copy_hi.begin());
        std::ranges::copy(m_time_lo, m_time_copy_lo.begin());

        cb.pre_hook(*this);

        if (m_time_hi != m_time_copy_hi || m_time_lo != m_time_copy_lo) [[unlikely]] {
            throw std::runtime_error(cb_time_errmsg);
        }
    }

    // NOTE: in general, an integration timestep will cover a different number of grid points for each batch element. We
    // thus need to track the grid index separately for each batch element. We will start with index 1 for all batch
    // elements, since all batch elements have been propagated to index 0 already.
    std::vector<grid_size_t> cur_grid_idx(
        boost::numeric_cast<typename std::vector<grid_size_t>::size_type>(m_batch_size), static_cast<grid_size_t>(1));

    // Vector of flags to keep track of the batch elements we can compute dense output for.
    //
    // NOTE: this is scratch space, reset at each iteration of the outer while loop below.
    std::vector<unsigned> dflags(boost::numeric_cast<std::vector<unsigned>::size_type>(m_batch_size));

    // Small helper to detect if there are still unprocessed grid points left for at least one batch element.
    const auto cont_cond = [n_grid_points, &cur_grid_idx] {
        return std::ranges::any_of(cur_grid_idx, [n_grid_points](const auto idx) { return idx < n_grid_points; });
    };

    // Keep on iterating until we have processed all grid points for all batch elements.
    while (true) {
        // Reset dflags.
        std::ranges::fill(dflags, 1u);

        // Compute the state of the system via dense output for as many grid points as possible.
        while (true) {
            // Establish and count for which batch elements we can still compute dense output.
            std::uint32_t counter = 0;

            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                // Fetch the grid index for the current batch element.
                const auto gidx = cur_grid_idx[i];

                if (dflags[i] && gidx < n_grid_points) {
                    // The current batch element has not been eliminated yet from dflags and it still has grid points
                    // available. Determine if we can use dense output.

                    // Fetch the grid target time and integrator time for the current batch element.
                    const auto cur_tt = grid_span[gidx, i];
                    const auto cur_time = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);

                    // NOTE: see the explanation in the scalar propagate_grid() as to why this works.
                    assert(m_t_dir[i] != 0);
                    const auto can_use_d_output
                        = ((m_t_dir[i] > 0) ? (cur_tt < cur_time) : (cur_tt > cur_time)) || (m_rem_time[i] == dl_zero);
                    dflags[i] = can_use_d_output;
                    counter += can_use_d_output;

                    // Copy over the grid target time to pgrid_tmp regardless of whether can_use_d_output is true or
                    // false. If we can compute the dense output this is needed, otherwise it is a meaningless value
                    // (the result of dense output at this point will be ignored).
                    pgrid_tmp[i] = cur_tt;
                } else {
                    // Either the batch element had already been eliminated previously, or there are no more grid points
                    // available. Make sure the batch element is marked as eliminated.
                    dflags[i] = 0;
                }
            }

            if (counter == 0u) {
                // Cannot use dense output on any of the batch elements, need to take another step.
                break;
            }

            // Compute the dense output.
            //
            // NOTE: for some batch elements, the data in pgrid_tmp may be meaningless/wrong. This should be ok, as
            // below we filter out from the dense output vector the wrong data.
            update_d_output(pgrid_tmp);

            // For the active batch elements, add the results to retval and bump up the values in cur_grid_idx.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                if (dflags[i] != 0u) {
                    const auto gidx = cur_grid_idx[i];
                    assert(gidx < n_grid_points);

                    for (std::uint32_t j = 0; j < m_dim; ++j) {
                        ret_span[gidx, j, i] = d_out_span[j, i];
                    }

                    ++cur_grid_idx[i];
                }
            }

            // Check if we exhausted all grid points for all batch elements.
            if (!cont_cond()) {
                break;
            }
        }

        // Check if we exhausted all grid points for all batch elements.
        if (!cont_cond()) {
            break;
        }

        // If the last step we took led to an interrupt condition, we will also break out.
        //
        // NOTE: the first time we execute this code, m_prop_res is guaranteed to contain only time_limit outcomes which
        // were explicitly written after the initial (successful) propagate_until().
        //
        // NOTE: interruption is signalled by an outcome of either:
        //
        // - cb_stop, or
        // - a stopping terminal event, or
        // - step_limit.
        if (std::ranges::any_of(m_prop_res, [](const auto &t) {
                const auto t_oc = std::get<0>(t);

                // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
                return t_oc == taylor_outcome::cb_stop || (t_oc > taylor_outcome::success && t_oc < taylor_outcome{0})
                       || t_oc == taylor_outcome::step_limit;
            })) {
            break;
        }

        // Compute the max integration times for this timestep, storing them in pgrid_tmp.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // NOTE: m_rem_time[i] is guaranteed to be finite: we check it explicitly both on creation and every time it
            // is updated. We also make sure that its sign at most changes to zero (this signals that a batch element is
            // done).
            assert(sgn_rem_time(m_rem_time[i]) == m_t_dir[i] || m_rem_time[i] == dl_zero);

            // Max delta_t for the current batch element.
            const auto max_delta_t = max_delta_ts[i];

            // Compute the step limit for the current batch element.
            const auto dt_limit = (m_t_dir[i] >= 0) ? std::min(detail::dfloat<T>(max_delta_t), m_rem_time[i])
                                                    : std::max(detail::dfloat<T>(-max_delta_t), m_rem_time[i]);

            pgrid_tmp[i] = static_cast<T>(dt_limit);
        }

        // Take the next step, making sure to write the Taylor coefficients and to cap the timestep size so that we
        // don't go past the last grid point and we don't use a timestep exceeding max_delta_t.
        //
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        step_impl(pgrid_tmp, true);

        // Check the outcomes of the step for each batch element, update the step counters, min_h/max_h and the
        // remaining times, and keep track of:
        //
        // - the number of batch elements which reached the time limit (n_done),
        // - whether or not any non-finite state was detected,
        // - whether or not any stopping terminal event triggered.
        std::uint32_t n_done = 0;
        bool nfs_detected = false, ste_detected = false;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            auto [oc, h] = m_step_res[i];

            if (oc == taylor_outcome::err_nf_state) [[unlikely]] {
                // Non-finite state: flag it and do nothing else.
                nfs_detected = true;
            } else {
                // Step outcome is one of:
                //
                // - success,
                // - time_limit,
                // - terminal event.
                //
                // Update the local step counters.
                //
                // NOTE: a step is counted unless the batch element had already finished before the current step, in
                // which case it is only taking zero-length steps while waiting for the other batch elements. A zero
                // remaining time means the element has finished.
                //
                // NOTE: unlike in propagate_until(), no special casing is needed for the first step: we can get here
                // only if the grid has at least 2 points, in which case all remaining times are nonzero before the
                // first step.
                const auto lane_active = (m_rem_time[i] != dl_zero);
                m_ts_count[i] = safe_size_t(m_ts_count[i]) + static_cast<std::size_t>(lane_active);

                // Update min_h/max_h only if the outcome is success (otherwise the step was artificially clamped either
                // by a time limit or by a terminal event).
                if (oc == taylor_outcome::success) {
                    const auto abs_h = abs(h);
                    m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
                    m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
                }

                // Check if this batch element reached the time limit.
                //
                // NOTE: we check h == rem_time.hi, instead of just oc == time_limit, because clamping via max_delta_t
                // could also result in time_limit.
                //
                // NOTE: once a batch element has finished, all subsequent steps have size zero and m_rem_time[i] stays
                // exactly zero, hence cur_time_limit_reached is always true and oc is always time_limit.
                const auto cur_time_limit_reached = (h == m_rem_time[i].hi);
                assert(!cur_time_limit_reached || oc == taylor_outcome::time_limit);

                // Flag if we encountered a stopping terminal event.
                const auto cur_ste = (oc > taylor_outcome::success) && (oc < taylor_outcome{0});
                ste_detected = ste_detected || cur_ste;

                // Compute the new remaining time and check it.
                const auto new_rem_time
                    = grid_span[n_grid_points - 1u, i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);
                // NOTE: this is likely impossible to trigger, but the check is cheap.
                if (is_nf(new_rem_time)) [[unlikely]] {
                    // LCOV_EXCL_START
                    // NOTE: just flag the non-finiteness, fill in m_prop_res[i] and continue to the next batch element.
                    nfs_detected = true;
                    m_prop_res[i]
                        = std::tuple{taylor_outcome::err_nf_state, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
                    continue;
                    // LCOV_EXCL_STOP
                }

                // Detect if rem_time changed sign. This happens if:
                //
                // - we landed exactly on the last grid point, or
                // - we overshot the last grid point due to floating-point rounding, or
                // - the current batch element had finished already before the current step.
                //
                // A sign flip is regarded the same as reaching the time limit.
                const auto rem_time_sign_flip = (sgn_rem_time(new_rem_time) != m_t_dir[i]);

                // Check if this batch element is done, meaning that it reached its time limit. A sign change in
                // rem_time counts as reaching the time limit.
                const auto cur_done = cur_time_limit_reached || rem_time_sign_flip;
                n_done += cur_done;

                // Update the remaining time, forcibly setting it to zero in case the current batch element is done.
                m_rem_time[i] = cur_done ? dl_zero : new_rem_time;

                // If the current batch element is done, snap the time coordinate to grid_span[n_grid_points - 1u, i]
                // and make sure oc is set to time_limit (if we did not run into a stopping terminal event).
                //
                // NOTE: the snapping is done in order to improve the user experience, i.e., the final time is
                // guaranteed to be exactly the last grid point. Strictly speaking, this can introduce a small error, at
                // the level of the floating-point epsilon. See also the explanation in
                // taylor_adaptive::propagate_until_impl().
                if (cur_done) {
                    m_time_hi[i] = grid_span[n_grid_points - 1u, i];
                    m_time_lo[i] = 0;

                    // If the current batch element did *not* encounter a stopping terminal event, make sure that oc
                    // contains time_limit.
                    //
                    // NOTE: we need to do this because we could be in a situation in which oc is success but we
                    // detected a rem_time sign flip, which is regarded the same as reaching the time limit.
                    if (!cur_ste) {
                        oc = taylor_outcome::time_limit;
                    }
                }
            }

            // Write into m_prop_res.
            m_prop_res[i] = std::tuple{oc, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }

        if (nfs_detected) [[unlikely]] {
            // At least 1 batch element generated a non-finite state. In this situation, we do *not* want to execute the
            // propagate() callback or modify retval any further. Just break and return what we have.
            break;
        }

        // Update the number of iterations.
        ++iter_counter;

        // Small helper to wrap the invocation of the callback while checking that the callback does not change the time
        // coordinate.
        const auto wrap_cb_call = [&] {
            // Store the current time coordinate before executing the cb, so that we can check if the cb changes the
            // time coordinate.
            //
            // NOTE: the hi/lo copies are guaranteed to contain finite values because no err_nf_state outcome was
            // detected in the step.
            std::ranges::copy(m_time_hi, m_time_copy_hi.begin());
            std::ranges::copy(m_time_lo, m_time_copy_lo.begin());

            // Execute the cb.
            assert(cb);
            const auto ret_cb = cb(*this);

            // Check the time coordinate.
            if (m_time_hi != m_time_copy_hi || m_time_lo != m_time_copy_lo) [[unlikely]] {
                throw std::runtime_error(cb_time_errmsg);
            }

            return ret_cb;
        };

        // Check the early interruption conditions.
        //
        // NOTE: in case of interruption, we don't exit immediately: the grid points falling within the most recent
        // step are processed first, at the beginning of the next loop iteration. After that, the loop exit is
        // guaranteed either because we processed all grid points or because of the interruption condition check.
        if (with_cb && !wrap_cb_call()) {
            // Interruption via callback.
            for (auto &t : m_prop_res) {
                std::get<0>(t) = taylor_outcome::cb_stop;
            }
        } else if (!ste_detected && n_done != m_batch_size && iter_counter == max_steps) {
            // Interruption via max iteration limit.
            //
            // NOTE: a stopping terminal event and the completion of the propagation for all batch elements take
            // precedence over step_limit.
            for (auto &t : m_prop_res) {
                std::get<0>(t) = taylor_outcome::step_limit;
            }
        }
    }

    // NOTE: at this point, we have the following possibilities:
    //
    // - a non-finite state was detected -> at least one outcome is err_nf_state,
    // - the integration was interrupted -> all outcomes are cb_stop or step_limit, or at least one outcome is a
    //   stopping terminal event,
    // - otherwise, all grid points were processed and all outcomes are time_limit.
    return make_retval();
}

template <typename T>
const std::variant<llvm_state, llvm_multi_state> &taylor_adaptive_batch<T>::get_llvm_state() const
{
    return m_i_data->m_ta_jit_data->m_llvm_state;
}

template <typename T>
const taylor_dc_t &taylor_adaptive_batch<T>::get_decomposition() const
{
    return m_i_data->m_dc;
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_order() const
{
    return m_i_data->m_order;
}

template <typename T>
T taylor_adaptive_batch<T>::get_tol() const
{
    return m_i_data->m_tol;
}

template <typename T>
bool taylor_adaptive_batch<T>::get_high_accuracy() const
{
    return m_i_data->m_high_accuracy;
}

template <typename T>
bool taylor_adaptive_batch<T>::get_compact_mode() const
{
    return m_i_data->m_compact_mode;
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_batch_size() const
{
    return m_i_data->m_batch_size;
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_dim() const
{
    return m_i_data->m_dim;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_time() const
{
    return m_i_data->m_time_hi;
}

template <typename T>
const T *taylor_adaptive_batch<T>::get_time_data() const
{
    return m_i_data->m_time_hi.data();
}

template <typename T>
std::pair<const std::vector<T> &, const std::vector<T> &> taylor_adaptive_batch<T>::get_dtime() const
{
    return std::make_pair(std::cref(m_i_data->m_time_hi), std::cref(m_i_data->m_time_lo));
}

template <typename T>
std::pair<const T *, const T *> taylor_adaptive_batch<T>::get_dtime_data() const
{
    return std::make_pair(m_i_data->m_time_hi.data(), m_i_data->m_time_lo.data());
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_state() const
{
    return m_i_data->m_state;
}

template <typename T>
const T *taylor_adaptive_batch<T>::get_state_data() const
{
    return m_i_data->m_state.data();
}

template <typename T>
std::ranges::subrange<typename std::vector<T>::iterator> taylor_adaptive_batch<T>::get_state_range()
{
    return std::ranges::subrange(m_i_data->m_state.begin(), m_i_data->m_state.end());
}

template <typename T>
T *taylor_adaptive_batch<T>::get_state_data()
{
    return m_i_data->m_state.data();
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_pars() const
{
    return m_i_data->m_pars;
}

template <typename T>
const T *taylor_adaptive_batch<T>::get_pars_data() const
{
    return m_i_data->m_pars.data();
}

template <typename T>
std::ranges::subrange<typename std::vector<T>::iterator> taylor_adaptive_batch<T>::get_pars_range()
{
    return std::ranges::subrange(m_i_data->m_pars.begin(), m_i_data->m_pars.end());
}

template <typename T>
T *taylor_adaptive_batch<T>::get_pars_data()
{
    return m_i_data->m_pars.data();
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_tc() const
{
    return m_i_data->m_tc;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_last_h() const
{
    return m_i_data->m_last_h;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_d_output() const
{
    return m_i_data->m_d_out;
}

template <typename T>
bool taylor_adaptive_batch<T>::with_events() const
{
    return static_cast<bool>(m_i_data->m_ed_data);
}

template <typename T>
const std::vector<typename taylor_adaptive_batch<T>::t_event_t> &taylor_adaptive_batch<T>::get_t_events() const
{
    if (!m_i_data->m_ed_data) [[unlikely]] {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    return m_i_data->m_ed_data->m_tes;
}

template <typename T>
const std::vector<std::vector<std::optional<std::pair<T, T>>>> &taylor_adaptive_batch<T>::get_te_cooldowns() const
{
    if (!m_i_data->m_ed_data) [[unlikely]] {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    return m_i_data->m_ed_data->m_te_cooldowns;
}

template <typename T>
const std::vector<typename taylor_adaptive_batch<T>::nt_event_t> &taylor_adaptive_batch<T>::get_nt_events() const
{
    if (!m_i_data->m_ed_data) [[unlikely]] {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    return m_i_data->m_ed_data->m_ntes;
}

template <typename T>
// NOLINTNEXTLINE(bugprone-exception-escape)
const std::vector<std::pair<expression, expression>> &taylor_adaptive_batch<T>::get_sys() const noexcept
{
    return (m_i_data->m_vsys.index() == 0) ? std::get<0>(m_i_data->m_vsys) : std::get<1>(m_i_data->m_vsys).get_sys();
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T>> &taylor_adaptive_batch<T>::get_step_res() const
{
    return m_i_data->m_step_res;
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &taylor_adaptive_batch<T>::get_propagate_res() const
{
    return m_i_data->m_prop_res;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::update_d_output(const std::vector<T> &time, bool rel_time)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_ta_jit_data);

    auto &m_d_out_f = m_ta_jit_data->m_d_out_f;

    // Check the dimensionality of time.
    if (time.size() != m_batch_size) {
        throw std::invalid_argument(fmt::format(
            "Invalid number of time coordinates specified for the dense output in a Taylor integrator in batch "
            "mode: the batch size is {}, but the number of time coordinates is {}",
            m_batch_size, time.size()));
    }

    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = m_last_h[i] + time[i];
        }
    } else {
        // Absolute time coordinate.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = static_cast<T>(time[i] - (detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) - m_last_h[i]));
        }
    }

    m_d_out_f(m_d_out.data(), m_tc.data(), m_d_out_time.data());

    return m_d_out;
}

// NOTE: there's some overlap with the code from the other overload here.
template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::update_d_output(T time, bool rel_time)
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out_time);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_last_h);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_hi);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_time_lo);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_d_out);
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_tc);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_ta_jit_data);

    auto &m_d_out_f = m_ta_jit_data->m_d_out_f;

    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = m_last_h[i] + time;
        }
    } else {
        // Absolute time coordinate.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = static_cast<T>(time - (detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) - m_last_h[i]));
        }
    }

    m_d_out_f(m_d_out.data(), m_tc.data(), m_d_out_time.data());

    return m_d_out;
}

template <typename T>
void taylor_adaptive_batch<T>::reset_cooldowns()
{
    HEYOKA_TAYLOR_REF_FROM_I_DATA(m_batch_size);

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        reset_cooldowns(i);
    }
}

template <typename T>
void taylor_adaptive_batch<T>::reset_cooldowns(std::uint32_t i)
{
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_batch_size);

    if (!m_i_data->m_ed_data) [[unlikely]] {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    if (i >= m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Cannot reset the cooldowns at batch index {}: the batch size for this integrator is only {}",
                        i, m_batch_size));
    }

    for (auto &cd : m_i_data->m_ed_data->m_te_cooldowns[i]) {
        cd.reset();
    }
}

template <typename T>
void taylor_adaptive_batch<T>::check_variational(const char *fname) const
{
    if (!is_variational()) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The function '{}()' cannot be invoked on non-variational batch integrators", fname));
    }
}

// Helper to fetch the stepper function from m_llvm_state.
// NOTE: this is exactly identical to the scalar integrator code.
// Should we write a separate common helper for this at one point?
template <typename T>
void taylor_adaptive_batch<T>::assign_stepper(bool with_events)
{
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_compact_mode);
    HEYOKA_TAYLOR_CREF_FROM_I_DATA(m_ta_jit_data);

    auto &m_step_f = m_ta_jit_data->m_step_f;
    auto &m_llvm_state = m_ta_jit_data->m_llvm_state;

    if (with_events) {
        if (m_compact_mode) {
            m_step_f = reinterpret_cast<detail::ta_jit_data<T>::c_step_f_e_t>(
                std::get<1>(m_llvm_state).jit_lookup("step_e"));
        } else {
            m_step_f
                = reinterpret_cast<detail::ta_jit_data<T>::step_f_e_t>(std::get<0>(m_llvm_state).jit_lookup("step_e"));
        }
    } else {
        if (m_compact_mode) {
            m_step_f
                = reinterpret_cast<detail::ta_jit_data<T>::c_step_f_t>(std::get<1>(m_llvm_state).jit_lookup("step"));
        } else {
            m_step_f = reinterpret_cast<detail::ta_jit_data<T>::step_f_t>(std::get<0>(m_llvm_state).jit_lookup("step"));
        }
    }
}

template <typename T>
const std::vector<expression> &taylor_adaptive_batch<T>::get_vargs() const
{
    check_variational(__func__);

    return std::get<1>(m_i_data->m_vsys).get_vargs();
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_vorder() const
{
    check_variational(__func__);

    return std::get<1>(m_i_data->m_vsys).get_order();
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::eval_taylor_map_impl(tm_input_t s)
{
    check_variational("eval_taylor_map");

    // Cache the number of variational arguments.
    const auto nvargs = std::get<1>(m_i_data->m_vsys).get_vargs().size();

    // Cache the batch size.
    const auto batch_size = m_i_data->m_batch_size;

    if (s.extent(0) % batch_size != 0u) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Unable to compute the Taylor map: the input range of values has a "
                                                "size of {}, which is not a multiple of the batch size {}",
                                                s.extent(0), batch_size));
    }

    if (s.extent(0) / batch_size != nvargs) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Unable to compute the Taylor map: the input range of values has a "
                        "size of {} (in batches of {}), but the number of variational arguments is {}",
                        s.extent(0) / batch_size, batch_size, nvargs));
    }

    // Run the compiled function.
    assert(m_i_data->m_vsys_data); // LCOV_EXCL_LINE
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto &vsys_data = *m_i_data->m_vsys_data;

    // Reshape the 1D data into 2D mdspans for cfunc's multi_eval interface.
    //
    // NOTE: we need to ensure that the extents and their products are representable as std::size_t, otherwise the
    // mdspan construction would be UB.
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;
    const auto bs = safe_size_t(batch_size);
    const auto nouts = safe_size_t(get_n_orig_sv());
    const auto nv = safe_size_t(nvargs);
    const auto np = safe_size_t(m_i_data->m_dim);

    (void)static_cast<std::size_t>(nouts * bs);
    (void)static_cast<std::size_t>(nv * bs);
    (void)static_cast<std::size_t>(np * bs);

    using out_2d = cfunc<T>::out_2d;
    using in_2d = cfunc<T>::in_2d;

    const out_2d out(vsys_data.m_tm_output.data(), static_cast<std::size_t>(nouts), static_cast<std::size_t>(bs));
    const in_2d in(s.data_handle(), static_cast<std::size_t>(nv), static_cast<std::size_t>(bs));
    const in_2d pars(m_i_data->m_state.data(), static_cast<std::size_t>(np), static_cast<std::size_t>(bs));

    // NOTE: disable batch parallelism. cfunc parallelises over the batch size, which, in typical usages of the batch
    // integrator, will be the same native SIMD size used by default in cfunc. As a consequence, no parallelisation is
    // possible, and explicitly disabling it increases performance by avoiding the parallelisation heuristic and the
    // overhead of the TBB scheduling system.
    vsys_data.m_tm_cfunc(out, in, kw::pars = pars, kw::batch_parallel = false);

    return vsys_data.m_tm_output;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::eval_taylor_map(std::initializer_list<T> il)
{
    return eval_taylor_map(std::ranges::subrange(il.begin(), il.end()));
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::get_tstate() const
{
    check_variational(__func__);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return m_i_data->m_vsys_data->m_tm_output;
}

template <typename T>
std::pair<std::uint32_t, std::uint32_t> taylor_adaptive_batch<T>::get_vslice(std::uint32_t order) const
{
    check_variational(__func__);

    const auto &dt = std::get<1>(m_i_data->m_vsys).get_dtens();

    const auto rng = dt.get_derivatives(order);

    return {
        boost::numeric_cast<std::uint32_t>(dt.index_of(rng.begin())),
        boost::numeric_cast<std::uint32_t>(dt.index_of(rng.end())),
    };
}

template <typename T>
std::pair<std::uint32_t, std::uint32_t> taylor_adaptive_batch<T>::get_vslice(std::uint32_t component,
                                                                             std::uint32_t order) const
{
    check_variational(__func__);

    const auto &dt = std::get<1>(m_i_data->m_vsys).get_dtens();

    const auto rng = dt.get_derivatives(component, order);

    return {
        boost::numeric_cast<std::uint32_t>(dt.index_of(rng.begin())),
        boost::numeric_cast<std::uint32_t>(dt.index_of(rng.end())),
    };
}

template <typename T>
const dtens::sv_idx_t &taylor_adaptive_batch<T>::get_mindex(std::uint32_t i) const
{
    check_variational(__func__);

    const auto &dt = std::get<1>(m_i_data->m_vsys).get_dtens();

    if (i >= dt.size()) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Cannot fetch the multiindex of the derivative at index {}: the index "
                                                "is not less than the total number of derivatives ({})",
                                                i, dt.size()));
    }

    return (dt.begin() + boost::numeric_cast<decltype(dt.begin() - dt.begin())>(i))->first;
}

// Explicit instantiations.
#define HEYOKA_TAYLOR_ADAPTIVE_BATCH_INST(F) template class HEYOKA_DLL_PUBLIC taylor_adaptive_batch<F>;

HEYOKA_TAYLOR_ADAPTIVE_BATCH_INST(float)
HEYOKA_TAYLOR_ADAPTIVE_BATCH_INST(double)
HEYOKA_TAYLOR_ADAPTIVE_BATCH_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADAPTIVE_BATCH_INST(mppp::real128)

#endif

#undef HEYOKA_TAYLOR_ADAPTIVE_BATCH_INST

HEYOKA_END_NAMESPACE

#undef HEYOKA_TAYLOR_REF_FROM_I_DATA

#undef HEYOKA_TAYLOR_CREF_FROM_I_DATA
