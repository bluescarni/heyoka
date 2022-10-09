// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

// Small helper to deduce the number of parameters
// present in the Taylor decomposition of an ODE system.
// NOTE: this will also include the functions of state variables,
// as they are part of the decomposition.
// NOTE: the first few entries in the decomposition are the mapping
// u variables -> state variables. These never contain any param
// by construction.
std::uint32_t n_pars_in_dc(const taylor_dc_t &dc)
{
    std::uint32_t retval = 0;

    for (const auto &p : dc) {
        retval = std::max(retval, get_param_size(p.first));
    }

    return retval;
}

namespace
{

// RAII helper to temporarily set the opt level to 0 in an llvm_state.
struct opt_disabler {
    llvm_state &m_s;
    unsigned m_orig_opt_level;

    explicit opt_disabler(llvm_state &s) : m_s(s), m_orig_opt_level(s.opt_level())
    {
        // Disable optimisations.
        m_s.opt_level() = 0;
    }

    opt_disabler(const opt_disabler &) = delete;
    opt_disabler(opt_disabler &&) noexcept = delete;
    opt_disabler &operator=(const opt_disabler &) = delete;
    opt_disabler &operator=(opt_disabler &&) noexcept = delete;

    ~opt_disabler()
    {
        // Restore the original optimisation level.
        m_s.opt_level() = m_orig_opt_level;
    }
};

// Helper to determine the optimal Taylor order for a given tolerance,
// following Jorba's prescription.
template <typename T>
std::uint32_t taylor_order_from_tol(T tol)
{
    using std::ceil;
    using std::isfinite;
    using std::log;

    // Determine the order from the tolerance.
    auto order_f = ceil(-log(tol) / 2 + 1);
    // LCOV_EXCL_START
    if (!isfinite(order_f)) {
        throw std::invalid_argument(
            "The computation of the Taylor order in an adaptive Taylor stepper produced a non-finite value");
    }
    // LCOV_EXCL_STOP
    // NOTE: min order is 2.
    order_f = std::max(T(2), order_f);

    // NOTE: cast to double as that ensures that the
    // max of std::uint32_t is exactly representable.
    // LCOV_EXCL_START
    if (order_f > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("The computation of the Taylor order in an adaptive Taylor stepper resulted "
                                  "in an overflow condition");
    }
    // LCOV_EXCL_STOP
    return static_cast<std::uint32_t>(order_f);
}

// Add to s an adaptive timestepper function with support for events. This timestepper will *not*
// propagate the state of the system. Instead, its output will be the jet of derivatives
// of all state variables and event equations, and the deduced timestep value(s).
template <typename T, typename U>
auto taylor_add_adaptive_step_with_events(llvm_state &s, const std::string &name, const U &sys, T tol,
                                          std::uint32_t batch_size, bool, bool compact_mode,
                                          const std::vector<expression> &evs, bool high_accuracy, bool parallel_mode)
{
    using std::isfinite;

    assert(!s.is_compiled());
    assert(batch_size != 0u);
    assert(isfinite(tol) && tol > 0);

    // Determine the order from the tolerance.
    const auto order = taylor_order_from_tol(tol);

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    auto [dc, ev_dc] = taylor_decompose(sys, evs);

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the LLVM type corresponding to T.
    auto *fp_t = to_llvm_type<T>(context);

    // Prepare the function prototype. The arguments are:
    // - pointer to the output jet of derivative (write only),
    // - pointer to the current state vector (read only),
    // - pointer to the parameters (read only),
    // - pointer to the time value(s) (read only),
    // - pointer to the array of max timesteps (read & write),
    // - pointer to the max_abs_state output variable (write only).
    // These pointers cannot overlap.
    std::vector<llvm::Type *> fargs(6, llvm::PointerType::getUnqual(to_llvm_type<T>(context)));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    // LCOV_EXCL_START
    if (f == nullptr) {
        throw std::invalid_argument(
            fmt::format("Unable to create a function for an adaptive Taylor stepper with name '{}'", name));
    }
    // LCOV_EXCL_STOP

    // Set the names/attributes of the function arguments.
    auto *jet_ptr = f->args().begin();
    jet_ptr->setName("jet_ptr");
    jet_ptr->addAttr(llvm::Attribute::NoCapture);
    jet_ptr->addAttr(llvm::Attribute::NoAlias);
    jet_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *state_ptr = jet_ptr + 1;
    state_ptr->setName("state_ptr");
    state_ptr->addAttr(llvm::Attribute::NoCapture);
    state_ptr->addAttr(llvm::Attribute::NoAlias);
    state_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *par_ptr = state_ptr + 1;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr = par_ptr + 1;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *h_ptr = time_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *max_abs_state_ptr = h_ptr + 1;
    max_abs_state_ptr->setName("max_abs_state_ptr");
    max_abs_state_ptr->addAttr(llvm::Attribute::NoCapture);
    max_abs_state_ptr->addAttr(llvm::Attribute::NoAlias);
    max_abs_state_ptr->addAttr(llvm::Attribute::WriteOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Create a global read-only array containing the values in ev_dc, if there
    // are any and we are in compact mode (otherwise, svf_ptr will be null).
    auto *svf_ptr = compact_mode ? taylor_c_make_sv_funcs_arr(s, ev_dc) : nullptr;

    // Compute the jet of derivatives at the given order.
    auto diff_variant = taylor_compute_jet<T>(s, state_ptr, par_ptr, time_ptr, dc, ev_dc, n_eq, n_uvars, order,
                                              batch_size, compact_mode, high_accuracy, parallel_mode);

    // Determine the integration timestep.
    auto h = taylor_determine_h(s, fp_t, diff_variant, ev_dc, svf_ptr, h_ptr, n_eq, n_uvars, order, batch_size,
                                max_abs_state_ptr);

    // Store h to memory.
    store_vector_to_memory(builder, h_ptr, h);

    // Copy the jet of derivatives to jet_ptr.
    taylor_write_tc(s, fp_t, diff_variant, ev_dc, svf_ptr, jet_ptr, n_eq, n_uvars, order, batch_size);

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass.
    // NOTE: this does nothing currently, as the optimisation
    // level is set to zero from the outside.
    s.optimise();

    return std::tuple{std::move(dc), order};
}

// NOTE: in compact mode, care must be taken when adding multiple stepper functions to the same llvm state
// with the same floating-point type, batch size and number of u variables. The potential issue there
// is that when the first stepper is added, the compact mode AD functions are created and then optimised.
// The optimisation pass might alter the functions in a way that makes them incompatible with subsequent
// uses in the second stepper (e.g., an argument might be removed from the signature because it is a
// compile-time constant). A workaround to avoid issues is to set the optimisation level to zero
// in the state, add the 2 steppers and then run a single optimisation pass. This is what we do
// in the integrators' ctors.
// NOTE: document this eventually.
template <typename T, typename U>
auto taylor_add_adaptive_step(llvm_state &s, const std::string &name, const U &sys, T tol, std::uint32_t batch_size,
                              bool high_accuracy, bool compact_mode, bool parallel_mode)
{
    using std::isfinite;

    assert(!s.is_compiled());
    assert(batch_size > 0u);
    assert(isfinite(tol) && tol > 0);

    // Determine the order from the tolerance.
    const auto order = taylor_order_from_tol(tol);

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    // NOTE: no sv_funcs needed for this stepper.
    auto [dc, sv_funcs_dc] = taylor_decompose(sys, {});

    assert(sv_funcs_dc.empty());

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    auto &builder = s.builder();
    auto &context = s.context();

    // Prepare the function prototype. The arguments are:
    // - pointer to the current state vector (read & write),
    // - pointer to the parameters (read only),
    // - pointer to the time value(s) (read only),
    // - pointer to the array of max timesteps (read & write),
    // - pointer to the Taylor coefficients output (write only).
    // These pointers cannot overlap.
    auto *fp_t = to_llvm_type<T>(context);
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);
    const std::vector<llvm::Type *> fargs(5, llvm::PointerType::getUnqual(fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument(
            fmt::format("Unable to create a function for an adaptive Taylor stepper with name '{}'", name));
    }

    // Set the names/attributes of the function arguments.
    auto *state_ptr = f->args().begin();
    state_ptr->setName("state_ptr");
    state_ptr->addAttr(llvm::Attribute::NoCapture);
    state_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *par_ptr = state_ptr + 1;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr = par_ptr + 1;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *h_ptr = time_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *tc_ptr = h_ptr + 1;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::WriteOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Compute the jet of derivatives at the given order.
    auto diff_variant = taylor_compute_jet<T>(s, state_ptr, par_ptr, time_ptr, dc, {}, n_eq, n_uvars, order, batch_size,
                                              compact_mode, high_accuracy, parallel_mode);

    // Determine the integration timestep.
    auto h = taylor_determine_h(s, fp_t, diff_variant, sv_funcs_dc, nullptr, h_ptr, n_eq, n_uvars, order, batch_size,
                                nullptr);

    // Evaluate the Taylor polynomials, producing the updated state of the system.
    auto new_state_var = high_accuracy ? taylor_run_ceval(s, fp_t, diff_variant, h, n_eq, n_uvars, order, high_accuracy,
                                                          batch_size, compact_mode)
                                       : taylor_run_multihorner(s, fp_t, diff_variant, h, n_eq, n_uvars, order,
                                                                batch_size, compact_mode);

    // Store the new state.
    // NOTE: no need to perform overflow check on n_eq * batch_size,
    // as in taylor_compute_jet() we already checked.
    if (compact_mode) {
        auto new_state = std::get<llvm::Value *>(new_state_var);

        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            auto val = builder.CreateLoad(fp_vec_t, builder.CreateInBoundsGEP(fp_vec_t, new_state, cur_var_idx));
            store_vector_to_memory(builder,
                                   builder.CreateInBoundsGEP(
                                       fp_t, state_ptr, builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))),
                                   val);
        });
    } else {
        const auto &new_state = std::get<std::vector<llvm::Value *>>(new_state_var);

        assert(new_state.size() == n_eq);

        for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
            store_vector_to_memory(builder,
                                   builder.CreateInBoundsGEP(fp_t, state_ptr, builder.getInt32(var_idx * batch_size)),
                                   new_state[var_idx]);
        }
    }

    // Store the timesteps that were used.
    store_vector_to_memory(builder, h_ptr, h);

    // Write the Taylor coefficients, if requested.
    auto nptr = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())));
    llvm_if_then_else(
        s, builder.CreateICmpNE(tc_ptr, nptr),
        [&]() {
            // tc_ptr is not null: copy the Taylor coefficients
            // for the state variables.
            taylor_write_tc(s, fp_t, diff_variant, {}, nullptr, tc_ptr, n_eq, n_uvars, order, batch_size);
        },
        []() {
            // Taylor coefficients were not requested,
            // don't do anything in this branch.
        });

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass.
    // NOTE: this does nothing currently, as the optimisation
    // level is set to zero from the outside.
    s.optimise();

    return std::tuple{std::move(dc), order};
}

} // namespace

} // namespace detail

template <typename T>
template <typename U>
void taylor_adaptive<T>::finalise_ctor_impl(const U &sys, std::vector<T> state, T time, T tol, bool high_accuracy,
                                            bool compact_mode, std::vector<T> pars, std::vector<t_event_t> tes,
                                            std::vector<nt_event_t> ntes, bool parallel_mode)
{
#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

    using std::isfinite;

    // Assign the data members.
    m_state = std::move(state);
    m_time = detail::dfloat<T>(time);
    m_pars = std::move(pars);
    m_high_accuracy = high_accuracy;
    m_compact_mode = compact_mode;

    // Check input params.
    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() != sys.size()) {
        throw std::invalid_argument(
            fmt::format("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                        "integrator: the state vector has a dimension of {}, while the number of equations is {}",
                        m_state.size(), sys.size()));
    }

    if (!isfinite(m_time)) {
        throw std::invalid_argument(
            fmt::format("Cannot initialise an adaptive Taylor integrator with a non-finite initial time of {}",
                        detail::fp_to_string(static_cast<T>(m_time))));
    }

    if (!isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(fmt::format(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is {} instead",
            detail::fp_to_string(tol)));
    }

    if (parallel_mode && !compact_mode) {
        throw std::invalid_argument("Parallel mode can be activated only in conjunction with compact mode");
    }

    // Store the tolerance.
    m_tol = tol;

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Do we have events?
    const auto with_events = !tes.empty() || !ntes.empty();

    // Temporarily disable optimisations in s, so that
    // we don't optimise twice when adding the step
    // and then the d_out.
    std::optional<detail::opt_disabler> od(m_llvm);

    // Add the stepper function.
    if (with_events) {
        std::vector<expression> ee;
        // NOTE: no need for deep copies of the expressions: ee is never mutated
        // and we will be deep-copying it anyway when we do the decomposition.
        for (const auto &ev : tes) {
            ee.push_back(ev.get_expression());
        }
        for (const auto &ev : ntes) {
            ee.push_back(ev.get_expression());
        }

        std::tie(m_dc, m_order) = detail::taylor_add_adaptive_step_with_events<T>(
            m_llvm, "step_e", sys, tol, 1, high_accuracy, compact_mode, ee, high_accuracy, parallel_mode);
    } else {
        std::tie(m_dc, m_order) = detail::taylor_add_adaptive_step<T>(m_llvm, "step", sys, tol, 1, high_accuracy,
                                                                      compact_mode, parallel_mode);
    }

    // Fix m_pars' size, if necessary.
    const auto npars = detail::n_pars_in_dc(m_dc);
    if (m_pars.size() < npars) {
        m_pars.resize(boost::numeric_cast<decltype(m_pars.size())>(npars));
    } else if (m_pars.size() > npars) {
        throw std::invalid_argument(fmt::format(
            "Excessive number of parameter values passed to the constructor of an adaptive "
            "Taylor integrator: {} parameter values were passed, but the ODE system contains only {} parameters",
            m_pars.size(), npars));
    }

    // Log runtimes in trace mode.
    spdlog::stopwatch sw;

    // Add the function for the computation of
    // the dense output.
    detail::taylor_add_d_out_function<T>(m_llvm, m_dim, m_order, 1, high_accuracy);

    detail::get_logger()->trace("Taylor dense output runtime: {}", sw);
    sw.reset();

    // Restore the original optimisation level in s.
    od.reset();

    // Run the optimisation pass manually.
    m_llvm.optimise();

    detail::get_logger()->trace("Taylor global opt pass runtime: {}", sw);
    sw.reset();

    // Run the jit.
    m_llvm.compile();

    detail::get_logger()->trace("Taylor LLVM compilation runtime: {}", sw);

    // Fetch the stepper.
    if (with_events) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }

    // Fetch the function to compute the dense output.
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));

    // Setup the vector for the Taylor coefficients.
    // LCOV_EXCL_START
    if (m_order == std::numeric_limits<std::uint32_t>::max()
        || m_state.size() > std::numeric_limits<decltype(m_tc.size())>::max() / (m_order + 1u)) {
        throw std::overflow_error("Overflow detected in the initialisation of an adaptive Taylor integrator: the order "
                                  "or the state size is too large");
    }
    // LCOV_EXCL_STOP

    m_tc.resize(m_state.size() * (m_order + 1u));

    // Setup the vector for the dense output.
    m_d_out.resize(m_state.size());

    // Init the event data structure if needed.
    // NOTE: this can be done in parallel with the rest of the constructor,
    // once we have m_order/m_dim and we are done using tes/ntes.
    if (with_events) {
        m_ed_data = std::make_unique<ed_data>(std::move(tes), std::move(ntes), m_order, m_dim);
    }
}

template <typename T>
taylor_adaptive<T>::taylor_adaptive() : taylor_adaptive({prime("x"_var) = 0_dbl}, {T(0)}, kw::tol = T(1e-1))
{
}

template <typename T>
taylor_adaptive<T>::taylor_adaptive(const taylor_adaptive &other)
    : m_state(other.m_state), m_time(other.m_time), m_llvm(other.m_llvm), m_dim(other.m_dim), m_order(other.m_order),
      m_tol(other.m_tol), m_high_accuracy(other.m_high_accuracy), m_compact_mode(other.m_compact_mode),
      m_pars(other.m_pars), m_tc(other.m_tc), m_last_h(other.m_last_h), m_d_out(other.m_d_out),
      m_ed_data(other.m_ed_data ? std::make_unique<ed_data>(*other.m_ed_data) : nullptr)
{
    // NOTE: make explicit deep copy of the decomposition.
    m_dc.reserve(other.m_dc.size());
    for (const auto &[ex, deps] : other.m_dc) {
        m_dc.emplace_back(copy(ex), deps);
    }

    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
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
    ar << m_state;
    ar << m_time;
    ar << m_llvm;
    ar << m_dim;
    ar << m_dc;
    ar << m_order;
    ar << m_tol;
    ar << m_high_accuracy;
    ar << m_compact_mode;
    ar << m_pars;
    ar << m_tc;
    ar << m_last_h;
    ar << m_d_out;
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

    ar >> m_state;
    ar >> m_time;
    ar >> m_llvm;
    ar >> m_dim;
    ar >> m_dc;
    ar >> m_order;
    ar >> m_tol;
    ar >> m_high_accuracy;
    ar >> m_compact_mode;
    ar >> m_pars;
    ar >> m_tc;
    ar >> m_last_h;
    ar >> m_d_out;
    ar >> m_ed_data;

    // Recover the function pointers.
    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
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
#endif

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
            return std::tuple{taylor_outcome::err_nf_state, h};
        }

        return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, h};
    } else {
        assert(m_ed_data); // LCOV_EXCL_LINE

        auto &edd = *m_ed_data;

        // Invoke the stepper for event handling. We will record the norm infinity of the state vector +
        // event equations at the beginning of the timestep for later use.
        T max_abs_state;
        std::get<1>(m_step_f)(edd.m_ev_jet.data(), m_state.data(), m_pars.data(), &m_time.hi, &h, &max_abs_state);

        // Compute the maximum absolute error on the Taylor series of the event equations, which we will use for
        // automatic cooldown deduction. If max_abs_state is not finite, set it to inf so that
        // in edd.detect_events() we skip event detection altogether.
        T g_eps;
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
                g_eps = std::numeric_limits<T>::epsilon() * max_abs_state;
            } else {
                g_eps = max_r_size;
            }
        } else {
            g_eps = std::numeric_limits<T>::infinity();
        }

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
        auto cmp = [](const auto &ev0, const auto &ev1) { return abs(std::get<1>(ev0)) < abs(std::get<1>(ev1)); };
        std::sort(edd.m_d_tes.begin(), edd.m_d_tes.end(), cmp);
        std::sort(edd.m_d_ntes.begin(), edd.m_d_ntes.end(), cmp);

        // If we have terminal events we need
        // to update the value of h.
        if (!edd.m_d_tes.empty()) {
            h = std::get<1>(edd.m_d_tes[0]);
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
            // Let's also reset the cooldown values, as at this point
            // they have become useless.
            reset_cooldowns();

            return std::tuple{taylor_outcome::err_nf_state, h};
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
                                     [](const auto &ev, const auto &t) { return abs(std::get<1>(ev)) < abs(t); });

        // Invoke the callbacks of the non-terminal events, which are guaranteed
        // to happen before the first terminal event.
        for (auto it = edd.m_d_ntes.begin(); it != ntes_end_it; ++it) {
            const auto &t = *it;
            const auto &cb = edd.m_ntes[std::get<0>(t)].get_callback();
            assert(cb); // LCOV_EXCL_LINE
            cb(*this, static_cast<T>(m_time - m_last_h + std::get<1>(t)), std::get<2>(t));
        }

        // The return value of the first
        // terminal event's callback. It will be
        // unused if there are no terminal events.
        bool te_cb_ret = false;

        if (!edd.m_d_tes.empty()) {
            // Fetch the first terminal event.
            const auto te_idx = std::get<0>(edd.m_d_tes[0]);
            assert(te_idx < edd.m_tes.size()); // LCOV_EXCL_LINE
            const auto &te = edd.m_tes[te_idx];

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
                                                   detail::taylor_deduce_cooldown(g_eps, std::get<4>(edd.m_d_tes[0])));
            }

            // Invoke the callback of the first terminal event, if it has one.
            if (te.get_callback()) {
                te_cb_ret = te.get_callback()(*this, std::get<2>(edd.m_d_tes[0]), std::get<3>(edd.m_d_tes[0]));
            }
        }

        if (edd.m_d_tes.empty()) {
            // No terminal events detected, return success or time limit.
            return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, h};
        } else {
            // Terminal event detected. Fetch its index.
            const auto ev_idx = static_cast<std::int64_t>(std::get<0>(edd.m_d_tes[0]));

            // NOTE: if te_cb_ret is true, it means that the terminal event has
            // a callback and its invocation returned true (meaning that the
            // integration should continue). Otherwise, either the terminal event
            // has no callback or its callback returned false, meaning that the
            // integration must stop.
            return std::tuple{taylor_outcome{te_cb_ret ? ev_idx : (-ev_idx - 1)}, h};
        }
    }
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step(bool wtc)
{
    // NOTE: time limit +inf means integration forward in time
    // and no time limit.
    return step_impl(std::numeric_limits<T>::infinity(), wtc);
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step_backward(bool wtc)
{
    return step_impl(-std::numeric_limits<T>::infinity(), wtc);
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive<T>::step(T max_delta_t, bool wtc)
{
    using std::isnan;

    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A NaN max_delta_t was passed to the step() function of an adaptive Taylor integrator");
    }

    return step_impl(max_delta_t, wtc);
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
std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>
taylor_adaptive<T>::propagate_until_impl(const detail::dfloat<T> &t, std::size_t max_steps, T max_delta_t,
                                         const std::function<bool(taylor_adaptive &)> &cb, bool wtc, bool with_c_out)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

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
    T min_h = std::numeric_limits<T>::infinity(), max_h(0);

    // Init the remaining time.
    auto rem_time = t - m_time;

    // Check it.
    if (!isfinite(rem_time)) {
        throw std::invalid_argument("The final time passed to the propagate_until() function of an adaptive Taylor "
                                    "integrator results in an overflow condition");
    }

    // Cache the integration direction.
    const auto t_dir = (rem_time >= T(0));

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

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
            ret.m_output.resize(boost::numeric_cast<decltype(ret.m_output.size())>(m_dim));

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

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        assert((rem_time >= T(0)) == t_dir); // LCOV_EXCL_LINE
        const auto dt_limit = t_dir ? std::min(detail::dfloat<T>(max_delta_t), rem_time)
                                    : std::max(detail::dfloat<T>(-max_delta_t), rem_time);
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        const auto [oc, h] = step_impl(static_cast<T>(dt_limit), wtc);

        if (oc == taylor_outcome::err_nf_state) {
            // If a non-finite state is detected, we do *not* want
            // to execute the propagate() callback and we do *not* want
            // to update the continuous output. Just exit.
            return std::tuple{oc, min_h, max_h, step_counter, make_c_out()};
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
        if (with_cb && !cb(*this)) {
            // Interruption via callback.
            return std::tuple{taylor_outcome::cb_stop, min_h, max_h, step_counter, make_c_out()};
        }

        // The breakout conditions:
        // - a step of rem_time was used, or
        // - a stopping terminal event was detected.
        // NOTE: we check h == rem_time, instead of just
        // oc == time_limit, because clamping via max_delta_t
        // could also result in time_limit.
        const bool ste_detected = oc > taylor_outcome::success && oc < taylor_outcome{0};
        if (h == static_cast<T>(rem_time) || ste_detected) {
#if !defined(NDEBUG)
            if (h == static_cast<T>(rem_time)) {
                assert(oc == taylor_outcome::time_limit);
            }
#endif
            return std::tuple{oc, min_h, max_h, step_counter, make_c_out()};
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger (modulo wraparound)
        // as by this point we are sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter, make_c_out()};
        }

        // Update the remaining time.
        // NOTE: at this point, we are sure
        // that abs(h) < abs(static_cast<T>(rem_time)). This implies
        // that t - m_time cannot undergo a sign
        // flip and invert the integration direction.
        assert(abs(h) < abs(static_cast<T>(rem_time)));
        rem_time = t - m_time;
    }
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
std::tuple<taylor_outcome, T, T, std::size_t, std::vector<T>>
taylor_adaptive<T>::propagate_grid_impl(const std::vector<T> &grid, std::size_t max_steps, T max_delta_t,
                                        const std::function<bool(taylor_adaptive &)> &cb)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    if (!isfinite(m_time)) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator if the current time is not finite");
    }

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

    // Pre-allocate the return value.
    std::vector<T> retval;
    // LCOV_EXCL_START
    if (get_dim() > std::numeric_limits<decltype(retval.size())>::max() / grid.size()) {
        throw std::overflow_error("Overflow detected in the creation of the return value of propagate_grid() in an "
                                  "adaptive Taylor integrator");
    }
    // LCOV_EXCL_STOP
    retval.reserve(grid.size() * get_dim());

    // Initial values for the counters
    // and the min/max abs of the integration
    // timesteps.
    // NOTE: iter_counter is for keeping track of the max_steps
    // limits, step_counter counts the number of timesteps performed
    // with a nonzero h. Most of the time these two quantities
    // will be identical, apart from corner cases.
    std::size_t iter_counter = 0, step_counter = 0;
    T min_h = std::numeric_limits<T>::infinity(), max_h(0);

    // Propagate the system up to the first grid point.
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
        return std::tuple{oc_until, min_h, max_h, step_counter, std::move(retval)};
    }

    // Add the first result to retval.
    retval.insert(retval.end(), m_state.begin(), m_state.end());

    // Init the remaining time.
    auto rem_time = grid.back() - m_time;

    // Check it.
    if (!isfinite(rem_time)) {
        throw std::invalid_argument("The final time passed to the propagate_grid() function of an adaptive Taylor "
                                    "integrator results in an overflow condition");
    }

    // Cache the integration direction.
    const auto t_dir = (rem_time >= T(0));

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

    // This flag, if set to something else than success,
    // is used to signal the early interruption of the integration.
    taylor_outcome interrupt = taylor_outcome::success;

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
            const auto cur_tt = grid[cur_grid_idx];

            // NOTE: we force processing of all remaining grid points
            // if we are at the last timestep. We do this in order to avoid
            // numerical issues when deciding if the last grid point
            // falls within the range of the last step.
            if ((cur_tt >= t0 && cur_tt <= t1) || (rem_time == detail::dfloat<T>(T(0)))) {
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
        const auto dt_limit = t_dir ? std::min(detail::dfloat<T>(max_delta_t), rem_time)
                                    : std::max(detail::dfloat<T>(-max_delta_t), rem_time);
        const auto [oc, h] = step_impl(static_cast<T>(dt_limit), true);

        if (oc == taylor_outcome::err_nf_state) {
            // If a non-finite state is detected, we do *not* want
            // to execute the propagate() callback and we do *not* want
            // to update the return value. Just exit.
            return std::tuple{oc, min_h, max_h, step_counter, std::move(retval)};
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

        // Check the early interruption conditions.
        // NOTE: only one of them must be set.
        if (with_cb && !cb(*this)) {
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
        // NOTE: if static_cast<T>(rem_time) was used as a timestep,
        // it means that we hit the time limit. Force rem_time to zero
        // to signal this, avoiding inconsistencies with grid.back() - m_time
        // not going exactly to zero due to numerical issues. A zero rem_time
        // will also force the processing of all remaining grid points.
        if (h == static_cast<T>(rem_time)) {
            assert(oc == taylor_outcome::time_limit); // LCOV_EXCL_LINE
            rem_time = detail::dfloat<T>(T(0));
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
    return std::tuple{interrupt == taylor_outcome::success ? taylor_outcome::time_limit : interrupt, min_h, max_h,
                      step_counter, std::move(retval)};
}

template <typename T>
const llvm_state &taylor_adaptive<T>::get_llvm_state() const
{
    return m_llvm;
}

template <typename T>
const taylor_dc_t &taylor_adaptive<T>::get_decomposition() const
{
    return m_dc;
}

template <typename T>
std::uint32_t taylor_adaptive<T>::get_order() const
{
    return m_order;
}

template <typename T>
T taylor_adaptive<T>::get_tol() const
{
    return m_tol;
}

template <typename T>
bool taylor_adaptive<T>::get_high_accuracy() const
{
    return m_high_accuracy;
}

template <typename T>
bool taylor_adaptive<T>::get_compact_mode() const
{
    return m_compact_mode;
}

template <typename T>
std::uint32_t taylor_adaptive<T>::get_dim() const
{
    return m_dim;
}

namespace
{

// NOTE: double-length normalisation assumes abs(hi) >= abs(lo), we need to enforce this
// when setting the time coordinate in double-length format.
template <typename T>
void dtime_checks(T hi, T lo)
{
    using std::abs;
    using std::isfinite;

    if (!isfinite(hi) || !isfinite(lo)) {
        throw std::invalid_argument(fmt::format("The components of the double-length representation of the time "
                                                "coordinate must both be finite, but they are {} and {} instead",
                                                detail::fp_to_string(hi), detail::fp_to_string(lo)));
    }

    if (abs(hi) < abs(lo)) {
        throw std::invalid_argument(
            fmt::format("The first component of the double-length representation of the time "
                        "coordinate ({}) must not be smaller in magnitude than the second component ({})",
                        detail::fp_to_string(hi), detail::fp_to_string(lo)));
    }
}

} // namespace

template <typename T>
void taylor_adaptive<T>::set_dtime(T hi, T lo)
{
    // Check the components.
    dtime_checks(hi, lo);

    m_time = normalise(detail::dfloat<T>(hi, lo));
}

template <typename T>
const std::vector<T> &taylor_adaptive<T>::update_d_output(T time, bool rel_time)
{
    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        const auto h = m_last_h + time;

        m_d_out_f(m_d_out.data(), m_tc.data(), &h);
    } else {
        // Absolute time coordinate.
        const auto h = time - (m_time - m_last_h);

        m_d_out_f(m_d_out.data(), m_tc.data(), &h.hi);
    }

    return m_d_out;
}

// Explicit instantiation of the implementation classes/functions.
// NOTE: on Windows apparently it is necessary to declare that
// these instantiations are meant to be dll-exported.
template class taylor_adaptive<double>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive<double>::finalise_ctor_impl(const std::vector<expression> &,
                                                                            std::vector<double>, double, double, bool,
                                                                            bool, std::vector<double>,
                                                                            std::vector<t_event_t>,
                                                                            std::vector<nt_event_t>, bool);

template HEYOKA_DLL_PUBLIC void
taylor_adaptive<double>::finalise_ctor_impl(const std::vector<std::pair<expression, expression>> &, std::vector<double>,
                                            double, double, bool, bool, std::vector<double>, std::vector<t_event_t>,
                                            std::vector<nt_event_t>, bool);

template class taylor_adaptive<long double>;

template HEYOKA_DLL_PUBLIC void
taylor_adaptive<long double>::finalise_ctor_impl(const std::vector<expression> &, std::vector<long double>, long double,
                                                 long double, bool, bool, std::vector<long double>,
                                                 std::vector<t_event_t>, std::vector<nt_event_t>, bool);

template HEYOKA_DLL_PUBLIC void taylor_adaptive<long double>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<long double>, long double, long double, bool,
    bool, std::vector<long double>, std::vector<t_event_t>, std::vector<nt_event_t>, bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive<mppp::real128>;

template HEYOKA_DLL_PUBLIC void
taylor_adaptive<mppp::real128>::finalise_ctor_impl(const std::vector<expression> &, std::vector<mppp::real128>,
                                                   mppp::real128, mppp::real128, bool, bool, std::vector<mppp::real128>,
                                                   std::vector<t_event_t>, std::vector<nt_event_t>, bool);

template HEYOKA_DLL_PUBLIC void taylor_adaptive<mppp::real128>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<mppp::real128>, mppp::real128, mppp::real128,
    bool, bool, std::vector<mppp::real128>, std::vector<t_event_t>, std::vector<nt_event_t>, bool);

#endif

template <typename T>
template <typename U>
void taylor_adaptive_batch<T>::finalise_ctor_impl(const U &sys, std::vector<T> state, std::uint32_t batch_size,
                                                  std::vector<T> time, T tol, bool high_accuracy, bool compact_mode,
                                                  std::vector<T> pars, std::vector<t_event_t> tes,
                                                  std::vector<nt_event_t> ntes, bool parallel_mode)
{
#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

    using std::isfinite;

    // Init the data members.
    m_batch_size = batch_size;
    m_state = std::move(state);
    m_time_hi = std::move(time);
    m_time_lo.resize(m_time_hi.size());
    m_pars = std::move(pars);
    m_high_accuracy = high_accuracy;
    m_compact_mode = compact_mode;

    // Check input params.
    if (m_batch_size == 0u) {
        throw std::invalid_argument("The batch size in an adaptive Taylor integrator cannot be zero");
    }

    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() % m_batch_size != 0u) {
        throw std::invalid_argument(
            fmt::format("Invalid size detected in the initialization of an adaptive Taylor "
                        "integrator: the state vector has a size of {}, which is not a multiple of the batch size ({})",
                        m_state.size(), m_batch_size));
    }

    if (m_state.size() / m_batch_size != sys.size()) {
        throw std::invalid_argument(
            fmt::format("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                        "integrator: the state vector has a dimension of {} and a batch size of {}, "
                        "while the number of equations is {}",
                        m_state.size() / m_batch_size, m_batch_size, sys.size()));
    }

    if (m_time_hi.size() != m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Invalid size detected in the initialization of an adaptive Taylor "
                        "integrator: the time vector has a size of {}, which is not equal to the batch size ({})",
                        m_time_hi.size(), m_batch_size));
    }
    // NOTE: no need to check m_time_lo for finiteness, as it
    // was inited to zero already.
    if (std::any_of(m_time_hi.begin(), m_time_hi.end(), [](const auto &x) { return !isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite initial time was detected in the initialisation of an adaptive Taylor integrator");
    }

    if (!isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(fmt::format(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is {} instead",
            detail::fp_to_string(tol)));
    }

    if (parallel_mode && !compact_mode) {
        throw std::invalid_argument("Parallel mode can be activated only in conjunction with compact mode");
    }

    // Store the tolerance.
    m_tol = tol;

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Do we have events?
    const auto with_events = !tes.empty() || !ntes.empty();

    // Temporarily disable optimisations in s, so that
    // we don't optimise twice when adding the step
    // and then the d_out.
    std::optional<detail::opt_disabler> od(m_llvm);

    // Add the stepper function.
    if (with_events) {
        std::vector<expression> ee;
        // NOTE: no need for deep copies of the expressions: ee is never mutated
        // and we will be deep-copying it anyway when we do the decomposition.
        for (const auto &ev : tes) {
            ee.push_back(ev.get_expression());
        }
        for (const auto &ev : ntes) {
            ee.push_back(ev.get_expression());
        }

        std::tie(m_dc, m_order) = detail::taylor_add_adaptive_step_with_events<T>(
            m_llvm, "step_e", sys, tol, batch_size, high_accuracy, compact_mode, ee, high_accuracy, parallel_mode);
    } else {
        std::tie(m_dc, m_order) = detail::taylor_add_adaptive_step<T>(m_llvm, "step", sys, tol, batch_size,
                                                                      high_accuracy, compact_mode, parallel_mode);
    }

    // Fix m_pars' size, if necessary.
    const auto npars = detail::n_pars_in_dc(m_dc);
    // LCOV_EXCL_START
    if (npars > std::numeric_limits<std::uint32_t>::max() / m_batch_size) {
        throw std::overflow_error("Overflow detected when computing the size of the parameter array in an adaptive "
                                  "Taylor integrator in batch mode");
    }
    // LCOV_EXCL_STOP
    if (m_pars.size() < npars * m_batch_size) {
        m_pars.resize(boost::numeric_cast<decltype(m_pars.size())>(npars * m_batch_size));
    } else if (m_pars.size() > npars * m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Excessive number of parameter values passed to the constructor of an adaptive "
                        "Taylor integrator in batch mode: {} parameter values were passed, but the ODE "
                        "system contains only {} parameters "
                        "(in batches of {})",
                        m_pars.size(), npars, m_batch_size));
    }

    // Log runtimes in trace mode.
    spdlog::stopwatch sw;

    // Add the function for the computation of
    // the dense output.
    detail::taylor_add_d_out_function<T>(m_llvm, m_dim, m_order, m_batch_size, high_accuracy);

    detail::get_logger()->trace("Taylor batch dense output runtime: {}", sw);
    sw.reset();

    // Restore the original optimisation level in s.
    od.reset();

    // Run the optimisation pass manually.
    m_llvm.optimise();

    detail::get_logger()->trace("Taylor batch global opt pass runtime: {}", sw);
    sw.reset();

    // Run the jit.
    m_llvm.compile();

    detail::get_logger()->trace("Taylor batch LLVM compilation runtime: {}", sw);

    // Fetch the stepper.
    if (with_events) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }

    // Fetch the function to compute the dense output.
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));

    // Setup the vector for the Taylor coefficients.
    // LCOV_EXCL_START
    if (m_order == std::numeric_limits<std::uint32_t>::max()
        || m_state.size() > std::numeric_limits<decltype(m_tc.size())>::max() / (m_order + 1u)) {
        throw std::overflow_error(
            "Overflow detected in the initialisation of an adaptive Taylor integrator in batch mode: the order "
            "or the state size is too large");
    }
    // LCOV_EXCL_STOP

    // NOTE: the size of m_state.size() already takes
    // into account the batch size.
    m_tc.resize(m_state.size() * (m_order + 1u));

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
                      std::tuple{taylor_outcome::success, T(0)});
    m_prop_res.resize(boost::numeric_cast<decltype(m_prop_res.size())>(m_batch_size),
                      std::tuple{taylor_outcome::success, T(0), T(0), std::size_t(0)});

    m_ts_count.resize(boost::numeric_cast<decltype(m_ts_count.size())>(m_batch_size));
    m_min_abs_h.resize(m_batch_size);
    m_max_abs_h.resize(m_batch_size);
    m_cur_max_delta_ts.resize(m_batch_size);
    m_pfor_ts.resize(boost::numeric_cast<decltype(m_pfor_ts.size())>(m_batch_size));
    m_t_dir.resize(boost::numeric_cast<decltype(m_t_dir.size())>(m_batch_size));
    m_rem_time.resize(m_batch_size);

    m_d_out_time.resize(m_batch_size);

    // Init the event data structure if needed.
    // NOTE: this can be done in parallel with the rest of the constructor,
    // once we have m_order/m_dim/m_batch_size and we are done using tes/ntes.
    if (with_events) {
        m_ed_data = std::make_unique<ed_data>(std::move(tes), std::move(ntes), m_order, m_dim, m_batch_size);
    }
}

template <typename T>
taylor_adaptive_batch<T>::taylor_adaptive_batch()
    : taylor_adaptive_batch({prime("x"_var) = 0_dbl}, {T(0)}, 1u, kw::tol = T(1e-1))
{
}

template <typename T>
taylor_adaptive_batch<T>::taylor_adaptive_batch(const taylor_adaptive_batch &other)
    // NOTE: make a manual copy of all members, apart from the function pointers.
    : m_batch_size(other.m_batch_size), m_state(other.m_state), m_time_hi(other.m_time_hi), m_time_lo(other.m_time_lo),
      m_llvm(other.m_llvm), m_dim(other.m_dim), m_order(other.m_order), m_tol(other.m_tol),
      m_high_accuracy(other.m_high_accuracy), m_compact_mode(other.m_compact_mode), m_pars(other.m_pars),
      m_tc(other.m_tc), m_last_h(other.m_last_h), m_d_out(other.m_d_out), m_pinf(other.m_pinf), m_minf(other.m_minf),
      m_delta_ts(other.m_delta_ts), m_step_res(other.m_step_res), m_prop_res(other.m_prop_res),
      m_ts_count(other.m_ts_count), m_min_abs_h(other.m_min_abs_h), m_max_abs_h(other.m_max_abs_h),
      m_cur_max_delta_ts(other.m_cur_max_delta_ts), m_pfor_ts(other.m_pfor_ts), m_t_dir(other.m_t_dir),
      m_rem_time(other.m_rem_time), m_d_out_time(other.m_d_out_time),
      m_ed_data(other.m_ed_data ? std::make_unique<ed_data>(*other.m_ed_data) : nullptr)
{
    // NOTE: make explicit deep copy of the decomposition.
    m_dc.reserve(other.m_dc.size());
    for (const auto &[ex, deps] : other.m_dc) {
        m_dc.emplace_back(copy(ex), deps);
    }

    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
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

// NOTE: the save/load patterns mimic the copy constructor logic.
template <typename T>
template <typename Archive>
void taylor_adaptive_batch<T>::save_impl(Archive &ar, unsigned) const
{
    // NOTE: save all members, apart from the function pointers.
    ar << m_batch_size;
    ar << m_state;
    ar << m_time_hi;
    ar << m_time_lo;
    ar << m_llvm;
    ar << m_dim;
    ar << m_dc;
    ar << m_order;
    ar << m_tol;
    ar << m_high_accuracy;
    ar << m_compact_mode;
    ar << m_pars;
    ar << m_tc;
    ar << m_last_h;
    ar << m_d_out;
    ar << m_pinf;
    ar << m_minf;
    ar << m_delta_ts;
    ar << m_step_res;
    ar << m_prop_res;
    ar << m_ts_count;
    ar << m_min_abs_h;
    ar << m_max_abs_h;
    ar << m_cur_max_delta_ts;
    ar << m_pfor_ts;
    ar << m_t_dir;
    ar << m_rem_time;
    ar << m_d_out_time;
    ar << m_ed_data;
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

    ar >> m_batch_size;
    ar >> m_state;
    ar >> m_time_hi;
    ar >> m_time_lo;
    ar >> m_llvm;
    ar >> m_dim;
    ar >> m_dc;
    ar >> m_order;
    ar >> m_tol;
    ar >> m_high_accuracy;
    ar >> m_compact_mode;
    ar >> m_pars;
    ar >> m_tc;
    ar >> m_last_h;
    ar >> m_d_out;
    ar >> m_pinf;
    ar >> m_minf;
    ar >> m_delta_ts;
    ar >> m_step_res;
    ar >> m_prop_res;
    ar >> m_ts_count;
    ar >> m_min_abs_h;
    ar >> m_max_abs_h;
    ar >> m_cur_max_delta_ts;
    ar >> m_pfor_ts;
    ar >> m_t_dir;
    ar >> m_rem_time;
    ar >> m_d_out_time;
    ar >> m_ed_data;

    // Recover the function pointers.
    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
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
    std::fill(m_time_lo.begin(), m_time_lo.end(), T(0));
}

template <typename T>
void taylor_adaptive_batch<T>::set_time(T new_time)
{
    // Set the hi part.
    std::fill(m_time_hi.begin(), m_time_hi.end(), new_time);
    // Reset the lo part.
    std::fill(m_time_lo.begin(), m_time_lo.end(), T(0));
}

template <typename T>
void taylor_adaptive_batch<T>::set_dtime(const std::vector<T> &hi, const std::vector<T> &lo)
{
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
        dtime_checks(hi[i], lo[i]);
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
    // Check the components.
    dtime_checks(hi, lo);

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
    auto check_nf_batch = [this](std::uint32_t batch_idx) {
        for (std::uint32_t i = 0; i < m_dim; ++i) {
            if (!isfinite(m_state[i * m_batch_size + batch_idx])) {
                return true;
            }
        }
        return false;
    };

    if (m_step_f.index() == 0u) {
        assert(!m_ed_data); // LCOV_EXCL_LINE

        std::get<0>(m_step_f)(m_state.data(), m_pars.data(), m_time_hi.data(), m_delta_ts.data(),
                              wtc ? m_tc.data() : nullptr);

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
                    h == m_delta_ts[m_batch_size + i] ? taylor_outcome::time_limit : taylor_outcome::success, h};
            }
        }
    } else {
        assert(m_ed_data); // LCOV_EXCL_LINE

        auto &edd = *m_ed_data;

        // Invoke the stepper for event handling. We will record the norm infinity of the state vector +
        // event equations at the beginning of the timestep for later use.
        std::get<1>(m_step_f)(edd.m_ev_jet.data(), m_state.data(), m_pars.data(), m_time_hi.data(), m_delta_ts.data(),
                              edd.m_max_abs_state.data());

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
            auto cmp = [](const auto &ev0, const auto &ev1) { return abs(std::get<1>(ev0)) < abs(std::get<1>(ev1)); };
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

        // We will use this to capture the first exception thrown
        // by a callback, if any.
        std::exception_ptr eptr;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto h = m_delta_ts[i];

            // Compute the new time in double-length arithmetic.
            const auto new_time = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + h;
            m_time_hi[i] = new_time.hi;
            m_time_lo[i] = new_time.lo;

            // Store the last timestep.
            m_last_h[i] = h;

            // Check if the time or the state vector are non-finite at the
            // end of the timestep.
            if (!isfinite(new_time) || check_nf_batch(i)) {
                // Let's also reset the cooldown values for this batch index,
                // as at this point they have become useless.
                reset_cooldowns(i);

                m_step_res[i] = std::tuple{taylor_outcome::err_nf_state, h};

                // Move to the next batch element.
                continue;
            }

            // Update the cooldowns.
            for (auto &cd : edd.m_te_cooldowns[i]) {
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
                = edd.m_d_tes[i].empty()
                      ? edd.m_d_ntes[i].end()
                      : std::lower_bound(edd.m_d_ntes[i].begin(), edd.m_d_ntes[i].end(), h,
                                         [](const auto &ev, const auto &t) { return abs(std::get<1>(ev)) < abs(t); });

            // Invoke the callbacks of the non-terminal events, which are guaranteed
            // to happen before the first terminal event.
            for (auto it = edd.m_d_ntes[i].begin(); it != ntes_end_it; ++it) {
                const auto &t = *it;
                const auto &cb = edd.m_ntes[std::get<0>(t)].get_callback();
                assert(cb); // LCOV_EXCL_LINE
                try {
                    cb(*this, static_cast<T>(new_time - m_last_h[i] + std::get<1>(t)), std::get<2>(t), i);
                } catch (...) {
                    if (!eptr) {
                        eptr = std::current_exception();
                    }
                }
            }

            // The return value of the first
            // terminal event's callback. It will be
            // unused if there are no terminal events.
            bool te_cb_ret = false;

            if (!edd.m_d_tes[i].empty()) {
                // Fetch the first terminal event.
                const auto te_idx = std::get<0>(edd.m_d_tes[i][0]);
                assert(te_idx < edd.m_tes.size()); // LCOV_EXCL_LINE
                const auto &te = edd.m_tes[te_idx];

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
                        0, detail::taylor_deduce_cooldown(edd.m_g_eps[i], std::get<4>(edd.m_d_tes[i][0])));
                }

                // Invoke the callback of the first terminal event, if it has one.
                if (te.get_callback()) {
                    try {
                        te_cb_ret = te.get_callback()(*this, std::get<2>(edd.m_d_tes[i][0]),
                                                      std::get<3>(edd.m_d_tes[i][0]), i);
                    } catch (...) {
                        if (!eptr) {
                            eptr = std::current_exception();
                        }
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
                    h == m_delta_ts[m_batch_size + i] ? taylor_outcome::time_limit : taylor_outcome::success, h};
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

        // Check if any callback threw an exception, and re-throw
        // it in case.
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    }
}

template <typename T>
void taylor_adaptive_batch<T>::step(bool wtc)
{
    step_impl(m_pinf, wtc);
}

template <typename T>
void taylor_adaptive_batch<T>::step_backward(bool wtc)
{
    step_impl(m_minf, wtc);
}

template <typename T>
void taylor_adaptive_batch<T>::step(const std::vector<T> &max_delta_ts, bool wtc)
{
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
std::optional<continuous_output_batch<T>> taylor_adaptive_batch<T>::propagate_for_impl(
    const std::vector<T> &delta_ts, std::size_t max_steps, const std::vector<T> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch &)> &cb, bool wtc, bool with_c_out)
{
    // Check the dimensionality of delta_ts.
    if (delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Invalid number of time intervals specified in a Taylor integrator in batch mode: "
                        "the batch size is {}, but the number of specified time intervals is {}",
                        m_batch_size, delta_ts.size()));
    }

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_pfor_ts[i] = detail::dfloat<T>(m_time_hi[i], m_time_lo[i]) + delta_ts[i];
    }

    // NOTE: max_delta_ts is checked in propagate_until_impl().
    return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts, std::move(cb), wtc, with_c_out);
}

// NOTE: possible outcomes:
// - all time_limit iff all batch elements
//   were successfully propagated up to the final times; else,
// - all cb_interrupt if the integration was interrupted via
//   a callback; else,
// - 1 or more err_nf_state if 1 or more batch elements generated
//   a non-finite state. The other elements will have their outcome
//   set by the last step taken; else,
// - 1 or more event indices if 1 or more batch elements generated
//   a stopping terminal event. The other elements will have their outcome
//   set by the last step taken.
// The callback is always executed at the end of each timestep, unless
// a non-finite state was detected in any batch element.
// The continuous output is always updated at the end of each timestep,
// unless a non-finite state was detected in any batch element.
template <typename T>
std::optional<continuous_output_batch<T>> taylor_adaptive_batch<T>::propagate_until_impl(
    const std::vector<detail::dfloat<T>> &ts, std::size_t max_steps, const std::vector<T> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch &)> &cb, bool wtc, bool with_c_out)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    // NOTE: this function is called from either the other propagate_until() overload,
    // or propagate_for(). In both cases, we have already set up correctly the dimension of ts.
    assert(ts.size() == m_batch_size); // LCOV_EXCL_LINE

    // Check the current times.
    if (std::any_of(m_time_hi.begin(), m_time_hi.end(), [](const auto &t) { return !isfinite(t); })
        || std::any_of(m_time_lo.begin(), m_time_lo.end(), [](const auto &t) { return !isfinite(t); })) {
        throw std::invalid_argument(
            "Cannot invoke the propagate_until() function of an adaptive Taylor integrator in batch mode if "
            "one of the current times is not finite");
    }

    // Check the final times.
    if (std::any_of(ts.begin(), ts.end(), [](const auto &t) { return !isfinite(t); })) {
        throw std::invalid_argument("A non-finite time was passed to the propagate_until() function of an adaptive "
                                    "Taylor integrator in batch mode");
    }

    // Check max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(fmt::format(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}",
            m_batch_size, max_delta_ts.size()));
    }
    for (const auto &dt : max_delta_ts) {
        if (isnan(dt)) {
            throw std::invalid_argument("A nan max_delta_t was passed to the propagate_until() function of an adaptive "
                                        "Taylor integrator in batch mode");
        }
        if (dt <= 0) {
            throw std::invalid_argument("A non-positive max_delta_t was passed to the propagate_until() function of an "
                                        "adaptive Taylor integrator in batch mode");
        }
    }

    // If with_c_out is true, we always need to write the Taylor coefficients.
    wtc = wtc || with_c_out;

    // These vectors are used in the construction of the continuous output.
    // If continuous output is not requested, they will remain empty.
    std::vector<T> c_out_tcs, c_out_times_hi, c_out_times_lo;
    if (with_c_out) {
        // Push in the starting time.
        c_out_times_hi.insert(c_out_times_hi.end(), m_time_hi.begin(), m_time_hi.end());
        c_out_times_lo.insert(c_out_times_lo.end(), m_time_lo.begin(), m_time_lo.end());
    }

    // Reset the counters and the min/max abs(h) vectors.
    std::size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    // Compute the integration directions and init
    // the remaining times.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_rem_time[i] = ts[i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);
        if (!isfinite(m_rem_time[i])) {
            throw std::invalid_argument("The final time passed to the propagate_until() function of an adaptive Taylor "
                                        "integrator in batch mode results in an overflow condition");
        }

        m_t_dir[i] = (m_rem_time[i] >= T(0));
    }

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

    // Helper to create the continuous output object.
    auto make_c_out = [&]() -> std::optional<continuous_output_batch<T>> {
        if (with_c_out) {
            if (c_out_times_hi.size() / m_batch_size < 2u) {
                // NOTE: this means that no successful steps
                // were taken.
                return {};
            }

            // Construct the return value.
            continuous_output_batch<T> ret(m_llvm.make_similar());

            // Fill in the data.
            ret.m_batch_size = m_batch_size;
            ret.m_tcs = std::move(c_out_tcs);
            ret.m_times_hi = std::move(c_out_times_hi);
            ret.m_times_lo = std::move(c_out_times_lo);

            // Add padding to the times vectors to make the
            // vectorised upper_bound implementation well defined.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                ret.m_times_hi.push_back(m_t_dir[i] ? std::numeric_limits<T>::infinity()
                                                    : -std::numeric_limits<T>::infinity());
                ret.m_times_lo.push_back(T(0));
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
    auto update_c_out = [&]() {
        if (with_c_out) {
#if !defined(NDEBUG)
            std::vector<detail::dfloat<T>> prev_times;
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
                assert(isfinite(new_time));
                if (m_t_dir[i]) {
                    assert(!(new_time < prev_times[i]));
                } else {
                    assert(!(new_time > prev_times[i]));
                }
            }
#endif

            c_out_tcs.insert(c_out_tcs.end(), m_tc.begin(), m_tc.end());
        }
    };

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: m_rem_time[i] is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            assert((m_rem_time[i] >= T(0)) == m_t_dir[i] || m_rem_time[i] == T(0)); // LCOV_EXCL_LINE

            // Compute the time limit.
            const auto dt_limit = m_t_dir[i] ? std::min(detail::dfloat<T>(max_delta_ts[i]), m_rem_time[i])
                                             : std::max(detail::dfloat<T>(-max_delta_ts[i]), m_rem_time[i]);

            // Store it.
            m_cur_max_delta_ts[i] = static_cast<T>(dt_limit);
        }

        // Run the integration timestep.
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        step_impl(m_cur_max_delta_ts, wtc);

        // Check the outcomes of the step for each batch element,
        // update the step counters, min_h/max_h and the remaining times
        // (if meaningful), and keep track of:
        // - the number of batch elements which reached the time limit,
        // - whether or not any non-finite state was detected,
        // - whether or not any stopping terminal event triggered.
        std::uint32_t n_done = 0;
        bool nfs_detected = false, ste_detected = false;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto [oc, h] = m_step_res[i];

            if (oc == taylor_outcome::err_nf_state) {
                // Non-finite state: flag it and do nothing else.
                nfs_detected = true;
            } else {
                // Step outcome is one of:
                // - success,
                // - time_limit,
                // - terminal event.

                // Update the local step counters.
                // NOTE: the local step counters increase only if we integrated
                // for a nonzero time.
                m_ts_count[i] += static_cast<std::size_t>(h != 0);

                // Update min_h/max_h only if the outcome is success (otherwise
                // the step was artificially clamped either by a time limit or
                // by a terminal event).
                if (oc == taylor_outcome::success) {
                    const auto abs_h = abs(h);
                    m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
                    m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
                }

                // Flag if we encountered a terminal event.
                ste_detected = ste_detected || (oc > taylor_outcome::success && oc < taylor_outcome{0});

                // Check if this batch element is done.
                // NOTE: we check h == rem_time, instead of just
                // oc == time_limit, because clamping via max_delta_t
                // could also result in time_limit.
                const auto cur_done = (h == static_cast<T>(m_rem_time[i]));
                n_done += cur_done;

                // Update the remaining times.
                if (cur_done) {
                    assert(oc == taylor_outcome::time_limit); // LCOV_EXCL_LINE

                    // Force m_rem_time[i] to zero so that
                    // zero-length steps will be taken
                    // for all remaining iterations.
                    // NOTE: if m_rem_time[i] was previously set to zero, it
                    // will end up being repeatedly set to zero here. This
                    // should be harmless.
                    m_rem_time[i] = detail::dfloat<T>(T(0));
                } else {
                    // NOTE: this should never flip the time direction of the
                    // integration for the same reasons as explained in the
                    // scalar implementation.
                    assert(abs(h) < abs(static_cast<T>(m_rem_time[i]))); // LCOV_EXCL_LINE
                    m_rem_time[i] = ts[i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);
                }
            }

            // Write into m_prop_res.
            m_prop_res[i] = std::tuple{oc, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }

        if (nfs_detected) {
            // At least 1 batch element generated a non-finite state. In this situation,
            // we do *not* want to execute the propagate() callback and we do *not* want
            // to update the continuous output. Just exit.
            return make_c_out();
        }

        // Update the continuous output.
        update_c_out();

        // Update the iteration counter.
        ++iter_counter;

        // Execute the propagate() callback, if applicable.
        if (with_cb && !cb(*this)) {
            // Change m_prop_res before exiting by setting all outcomes
            // to cb_stop regardless of the timestep outcome.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                std::get<0>(m_prop_res[i]) = taylor_outcome::cb_stop;
            }

            return make_c_out();
        }

        // We need to break out if either we reached the final time
        // for all batch elements, or we encountered at least 1 stopping
        // terminal event. In either case, m_prop_res was already set up
        // in the loop where we checked the outcomes.
        if (n_done == m_batch_size || ste_detected) {
            return make_c_out();
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger (modulo wraparound) as by this point we are
        // sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            // We reached the max_steps limit: set the outcome for all batch elements
            // to step_limit.
            // NOTE: this is the same logic adopted when the integration is stopped
            // by the callback (see above).
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                std::get<0>(m_prop_res[i]) = taylor_outcome::step_limit;
            }

            return make_c_out();
        }
    }

    // LCOV_EXCL_START
    assert(false);

    return {};
    // LCOV_EXCL_STOP
}

template <typename T>
std::optional<continuous_output_batch<T>> taylor_adaptive_batch<T>::propagate_until_impl(
    const std::vector<T> &ts, std::size_t max_steps, const std::vector<T> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch &)> &cb, bool wtc, bool with_c_out)
{
    // Check the dimensionality of ts.
    if (ts.size() != m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Invalid number of time limits specified in a Taylor integrator in batch mode: the "
                        "batch size is {}, but the number of specified time limits is {}",
                        m_batch_size, ts.size()));
    }

    // NOTE: re-use m_pfor_ts as tmp storage.
    assert(m_pfor_ts.size() == m_batch_size); // LCOV_EXCL_LINE
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_pfor_ts[i] = detail::dfloat<T>(ts[i]);
    }

    // NOTE: max_delta_ts is checked in the other propagate_until_impl() overload.
    return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts, std::move(cb), wtc, with_c_out);
}

// NOTE: possible outcomes:
// - all time_limit (the happy path),
// - at least 1 err_nf_state if a non-finite state was detected,
// - all cb_stop or all step_limit in case of interruption by,
//   respectively, callback or iteration limit,
// - at least 1 stopping terminal event.
// The callback is always executed at the end of each timestep, unless
// a non-finite state was detected.
template <typename T>
std::vector<T> taylor_adaptive_batch<T>::propagate_grid_impl(const std::vector<T> &grid, std::size_t max_steps,
                                                             const std::vector<T> &max_delta_ts,
                                                             const std::function<bool(taylor_adaptive_batch &)> &cb)
{
    using std::abs;
    using std::isnan;

    // Helper to detect if an input value is nonfinite.
    auto is_nf = [](const T &t) {
        using std::isfinite;
        return !isfinite(t);
    };

    if (grid.empty()) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if the time grid is empty");
    }

    // Check that the grid size is a multiple of m_batch_size.
    if (grid.size() % m_batch_size != 0u) {
        throw std::invalid_argument(fmt::format(
            "Invalid grid size detected in propagate_grid() for an adaptive Taylor integrator in batch mode: "
            "the grid has a size of {}, which is not a multiple of the batch size ({})",
            grid.size(), m_batch_size));
    }

    // Check the current time coordinates.
    if (std::any_of(m_time_hi.begin(), m_time_hi.end(), is_nf)
        || std::any_of(m_time_lo.begin(), m_time_lo.end(), is_nf)) {
        throw std::invalid_argument("Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if "
                                    "the current time is not finite");
    }

    // Check max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(fmt::format(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}",
            m_batch_size, max_delta_ts.size()));
    }
    for (const auto &dt : max_delta_ts) {
        if (isnan(dt)) {
            throw std::invalid_argument("A nan max_delta_t was passed to the propagate_grid() function of an adaptive "
                                        "Taylor integrator in batch mode");
        }
        if (dt <= 0) {
            throw std::invalid_argument("A non-positive max_delta_t was passed to the propagate_grid() function of an "
                                        "adaptive Taylor integrator in batch mode");
        }
    }

    // The number of grid points.
    const auto n_grid_points = grid.size() / m_batch_size;

    // Pointer to the grid data.
    const auto *const grid_ptr = grid.data();

    // Check the input grid points.
    constexpr auto nf_err_msg
        = "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator in batch mode";
    constexpr auto ig_err_msg = "A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                "Taylor integrator in batch mode";

    // Check the first point.
    if (std::any_of(grid_ptr, grid_ptr + m_batch_size, is_nf)) {
        throw std::invalid_argument(nf_err_msg);
    }
    if (n_grid_points > 1u) {
        // Establish the direction of the grid from
        // the first two batches of points.
        if (std::any_of(grid_ptr + m_batch_size, grid_ptr + m_batch_size + m_batch_size, is_nf)) {
            throw std::invalid_argument(nf_err_msg);
        }
        if (grid_ptr[m_batch_size] == grid_ptr[0]) {
            throw std::invalid_argument(ig_err_msg);
        }

        const auto grid_direction = grid_ptr[m_batch_size] > grid_ptr[0];
        for (std::uint32_t i = 1; i < m_batch_size; ++i) {
            if ((grid_ptr[m_batch_size + i] > grid_ptr[i]) != grid_direction) {
                throw std::invalid_argument(ig_err_msg);
            }
        }

        // Check that the remaining points are finite and that
        // they are ordered monotonically.
        for (decltype(grid.size()) i = 2; i < n_grid_points; ++i) {
            if (std::any_of(grid_ptr + i * m_batch_size, grid_ptr + (i + 1u) * m_batch_size, is_nf)) {
                throw std::invalid_argument(nf_err_msg);
            }

            if (std::any_of(
                    grid_ptr + i * m_batch_size, grid_ptr + (i + 1u) * m_batch_size,
                    [this, grid_direction](const T &t) { return (t > *(&t - m_batch_size)) != grid_direction; })) {
                throw std::invalid_argument(ig_err_msg);
            }
        }
    }

    // Pre-allocate the return value.
    std::vector<T> retval;
    // LCOV_EXCL_START
    if (get_dim() > std::numeric_limits<decltype(retval.size())>::max() / grid.size()) {
        throw std::overflow_error("Overflow detected in the creation of the return value of propagate_grid() in an "
                                  "adaptive Taylor integrator in batch mode");
    }
    // LCOV_EXCL_STOP
    // NOTE: fill with NaNs, so that the missing entries
    // are signalled with NaN if we exit early.
    retval.resize(grid.size() * get_dim(), std::numeric_limits<T>::quiet_NaN());

    // NOTE: this is a buffer of size m_batch_size
    // that is used in various places as temp storage.
    std::vector<T> pgrid_tmp;
    pgrid_tmp.resize(boost::numeric_cast<decltype(pgrid_tmp.size())>(m_batch_size));

    // Propagate the system up to the first batch of grid points.
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
    std::copy(grid_ptr, grid_ptr + m_batch_size, pgrid_tmp.begin());
    propagate_until(pgrid_tmp, kw::max_delta_t = max_delta_ts, kw::max_steps = max_steps, kw::write_tc = true);

    // Check the result of the integration.
    if (std::any_of(m_prop_res.begin(), m_prop_res.end(), [](const auto &t) {
            // Check if any outcome is not time_limit.
            return std::get<0>(t) != taylor_outcome::time_limit;
        })) {
        // NOTE: for consistency with the scalar implementation,
        // keep the outcomes from propagate_until() but we reset
        // min/max h and the step counter.
        for (auto &[_, min_h, max_h, ts_count] : m_prop_res) {
            min_h = std::numeric_limits<T>::infinity();
            max_h = 0;
            ts_count = 0;
        }

        return retval;
    }

    // Add the first result to retval.
    std::copy(m_state.begin(), m_state.end(), retval.begin());

    // Init the remaining times and directions.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_rem_time[i]
            = grid_ptr[(n_grid_points - 1u) * m_batch_size + i] - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);

        // Check it.
        if (!isfinite(m_rem_time[i])) {
            throw std::invalid_argument("The final time passed to the propagate_grid() function of an adaptive Taylor "
                                        "integrator in batch mode results in an overflow condition");
        }

        m_t_dir[i] = (m_rem_time[i] >= T(0));
    }

    // Cache the presence/absence of a callback.
    const auto with_cb = static_cast<bool>(cb);

    // Reset the counters and the min/max abs(h) vectors.
    std::size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    // NOTE: in general, an integration timestep will cover a different number
    // of grid points for each batch element. We thus need to track the grid
    // index separately for each batch element. We will start with index
    // 1 for all batch elements, since all batch elements have been propagated to
    // index 0 already.
    std::vector<decltype(grid.size())> cur_grid_idx(
        boost::numeric_cast<typename std::vector<decltype(grid.size())>::size_type>(m_batch_size), 1);

    // Vectors to keep track of the time range of the last taken timestep.
    std::vector<detail::dfloat<T>> t0(
        boost::numeric_cast<typename std::vector<detail::dfloat<T>>::size_type>(m_batch_size)),
        t1(t0);

    // Vector of flags to keep track of the batch elements
    // we can compute dense output for.
    std::vector<unsigned> dflags(boost::numeric_cast<std::vector<unsigned>::size_type>(m_batch_size));

    // NOTE: small helper to detect if there are still unprocessed
    // grid points for at least one batch element.
    auto cont_cond = [n_grid_points, &cur_grid_idx]() {
        return std::any_of(cur_grid_idx.begin(), cur_grid_idx.end(),
                           [n_grid_points](auto idx) { return idx < n_grid_points; });
    };

    while (cont_cond()) {
        // Establish the time ranges of the last
        // taken timestep.
        // NOTE: t0 < t1.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const detail::dfloat<T> cur_time(m_time_hi[i], m_time_lo[i]), cmp = cur_time - m_last_h[i];

            t0[i] = std::min(cur_time, cmp);
            t1[i] = std::max(cur_time, cmp);
        }

        // Reset dflags.
        std::fill(dflags.begin(), dflags.end(), 1u);

        // Compute the state of the system via dense output for as many grid
        // points as possible, i.e., as long as the grid times
        // fall within the last taken step of at least
        // one batch element.
        while (true) {
            // Establish and count for which batch elements we
            // can still compute dense output.
            std::uint32_t counter = 0;
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                // Fetch the grid index for the current batch element.
                const auto gidx = cur_grid_idx[i];

                if (dflags[i] && gidx < n_grid_points) {
                    // The current batch element has not been eliminated
                    // yet from the candidate list and it still has grid
                    // points available. Determine if the current grid point
                    // falls within the last taken step.
                    // NOTE: if we are at the last timestep for this batch
                    // element, force processing of all remaining grid points.
                    // We do this to avoid numerical issues when deciding if
                    // he last grid point falls within the range of validity
                    // of the dense output.
                    const auto idx = gidx * m_batch_size + i;
                    const auto d_avail = (grid_ptr[idx] >= t0[i] && grid_ptr[idx] <= t1[i])
                                         || (m_rem_time[i] == detail::dfloat<T>(T(0)));
                    dflags[i] = d_avail;
                    counter += d_avail;

                    // Copy over the grid point to pgrid_tmp regardless
                    // of whether d_avail is true or false.
                    pgrid_tmp[i] = grid_ptr[idx];
                } else {
                    // Either the batch element had already been eliminated
                    // previously, or there are no more grid points available.
                    // Make sure the batch element is marked as eliminated.
                    dflags[i] = 0;
                }
            }

            if (counter == 0u) {
                // Cannot use dense output on any of the batch elements,
                // need to take another step.
                break;
            }

            // Compute the dense output.
            // NOTE: for some batch elements, the data in pgrid_tmp
            // may be meaningless/wrong. This should be ok, as below
            // we filter out from the dense output vector
            // the wrong data.
            update_d_output(pgrid_tmp);

            // Add the results to retval and bump up the values in cur_grid_idx.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                if (dflags[i] != 0u) {
                    const auto gidx = cur_grid_idx[i];

                    for (std::uint32_t j = 0; j < m_dim; ++j) {
                        retval[gidx * m_batch_size * m_dim + j * m_batch_size + i] = m_d_out[j * m_batch_size + i];
                    }

                    assert(cur_grid_idx[i] < n_grid_points); // LCOV_EXCL_LINE
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
            assert(std::all_of(m_prop_res.begin(), m_prop_res.end(),
                               [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

            break;
        }

        // If the last step we took led to an interrupt condition,
        // we will also break out.
        // NOTE: the first time we execute this code, m_prop_res
        // is guaranteed to contain only time_limit outcomes after
        // the initial propagate_until().
        // NOTE: interruption is signalled by an outcome of either:
        // - cb_stop, or
        // - a stopping terminal event, or
        // - step_limit.
        if (std::any_of(m_prop_res.begin(), m_prop_res.end(), [](const auto &t) {
                const auto t_oc = std::get<0>(t);

                return t_oc == taylor_outcome::cb_stop || (t_oc > taylor_outcome::success && t_oc < taylor_outcome{0})
                       || t_oc == taylor_outcome::step_limit;
            })) {
            break;
        }

        // Take the next step, making sure to write the Taylor coefficients
        // and to cap the timestep size so that we don't go past the
        // last grid point and we don't use a timestep exceeding max_delta_t.
        // NOTE: m_rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // Max delta_t for the current batch element.
            const auto max_delta_t = max_delta_ts[i];

            // Compute the step limit for the current batch element.
            assert((m_rem_time[i] >= T(0)) == m_t_dir[i] || m_rem_time[i] == T(0)); // LCOV_EXCL_LINE
            const auto dt_limit = m_t_dir[i] ? std::min(detail::dfloat<T>(max_delta_t), m_rem_time[i])
                                             : std::max(detail::dfloat<T>(-max_delta_t), m_rem_time[i]);

            pgrid_tmp[i] = static_cast<T>(dt_limit);
        }
        step_impl(pgrid_tmp, true);

        // Check the outcomes of the step for each batch element,
        // update the step counters, min_h/max_h and the remaining times
        // (if meaningful), and keep track of
        // whether or not any non-finite state was detected.
        // This loop will also write into m_prop_res the outcomes,
        // taking them from the step() outcomes.
        bool nfs_detected = false;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto [oc, h] = m_step_res[i];

            if (oc == taylor_outcome::err_nf_state) {
                // Non-finite state: flag it and do nothing else.
                nfs_detected = true;
            } else {
                // Step outcome is one of:
                // - success,
                // - time_limit,
                // - terminal event.

                // Update the local step counters.
                // NOTE: the local step counters increase only if we integrated
                // for a nonzero time.
                m_ts_count[i] += static_cast<std::size_t>(h != 0);

                // Update min_h/max_h only if the outcome is success (otherwise
                // the step was artificially clamped either by a time limit or
                // by a terminal event).
                if (oc == taylor_outcome::success) {
                    const auto abs_h = abs(h);
                    m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
                    m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
                }

                // Update the remaining times.
                // NOTE: if static_cast<T>(m_rem_time[i]) was used as a timestep,
                // it means that we hit the time limit. Force rem_time to zero
                // to signal this, so that zero-length steps will be taken
                // for all remaining iterations, thus always triggering the
                // time_limit outcome. A zero m_rem_time[i]
                // will also force the processing of all remaining grid points.
                // NOTE: if m_rem_time[i] was previously set to zero, it
                // will end up being repeatedly set to zero here. This
                // should be harmless.
                // NOTE: we check h == rem_time, instead of just
                // oc == time_limit, because clamping via max_delta_t
                // could also result in time_limit.
                if (h == static_cast<T>(m_rem_time[i])) {
                    assert(oc == taylor_outcome::time_limit); // LCOV_EXCL_LINE
                    m_rem_time[i] = detail::dfloat<T>(T(0));
                } else {
                    // NOTE: this should never flip the time direction of the
                    // integration for the same reasons as explained in the
                    // scalar implementation of propagate_until().
                    assert(abs(h) < abs(static_cast<T>(m_rem_time[i]))); // LCOV_EXCL_LINE
                    m_rem_time[i] = grid_ptr[(n_grid_points - 1u) * m_batch_size + i]
                                    - detail::dfloat<T>(m_time_hi[i], m_time_lo[i]);
                }
            }

            // Write into m_prop_res.
            m_prop_res[i] = std::tuple{oc, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }

        if (nfs_detected) {
            // At least 1 batch element generated a non-finite state. In this situation,
            // we do *not* want to execute the propagate() callback or modify retval
            // any further. Just break and return what we have.
            break;
        }

        // Update the number of iterations.
        ++iter_counter;

        // Check the early interruption conditions.
        // NOTE: in case of cb_stop or step_limit,
        // we will overwrite the outcomes in m_prop_res.
        // The outcome for a stopping terminal event is already
        // set up properly in the previous loop.
        if (with_cb && !cb(*this)) {
            // Interruption via callback.
            for (auto &t : m_prop_res) {
                std::get<0>(t) = taylor_outcome::cb_stop;
            }
        } else if (iter_counter == max_steps) {
            // Interruption via max iteration limit.
            for (auto &t : m_prop_res) {
                std::get<0>(t) = taylor_outcome::step_limit;
            }
        }
    }

    // NOTE: at this point, we have the following possibilities:
    // - a non-finite state was detected -> at least one outcome is err_nf_state,
    // - the integration was interrupted early -> at least one outcome
    //   is either cb_stop, step_limit or a stopping terminal event,
    // - the integration finished successfully, in which case we consumed
    //   all grid points and all outcomes are time_limit.
    return retval;
}

template <typename T>
const llvm_state &taylor_adaptive_batch<T>::get_llvm_state() const
{
    return m_llvm;
}

template <typename T>
const taylor_dc_t &taylor_adaptive_batch<T>::get_decomposition() const
{
    return m_dc;
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_order() const
{
    return m_order;
}

template <typename T>
T taylor_adaptive_batch<T>::get_tol() const
{
    return m_tol;
}

template <typename T>
bool taylor_adaptive_batch<T>::get_high_accuracy() const
{
    return m_high_accuracy;
}

template <typename T>
bool taylor_adaptive_batch<T>::get_compact_mode() const
{
    return m_compact_mode;
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_batch_size() const
{
    return m_batch_size;
}

template <typename T>
std::uint32_t taylor_adaptive_batch<T>::get_dim() const
{
    return m_dim;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch<T>::update_d_output(const std::vector<T> &time, bool rel_time)
{
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
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        reset_cooldowns(i);
    }
}

template <typename T>
void taylor_adaptive_batch<T>::reset_cooldowns(std::uint32_t i)
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    if (i >= m_batch_size) {
        throw std::invalid_argument(
            fmt::format("Cannot reset the cooldowns at batch index {}: the batch size for this integrator is only {}",
                        i, m_batch_size));
    }

    for (auto &cd : m_ed_data->m_te_cooldowns[i]) {
        cd.reset();
    }
}

// Explicit instantiation of the batch implementation classes.
template class taylor_adaptive_batch<double>;

template HEYOKA_DLL_PUBLIC void
taylor_adaptive_batch<double>::finalise_ctor_impl(const std::vector<expression> &, std::vector<double>, std::uint32_t,
                                                  std::vector<double>, double, bool, bool, std::vector<double>,
                                                  std::vector<t_event_t>, std::vector<nt_event_t>, bool);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch<double>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<double>, std::uint32_t, std::vector<double>,
    double, bool, bool, std::vector<double>, std::vector<t_event_t>, std::vector<nt_event_t>, bool);

template class taylor_adaptive_batch<long double>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch<long double>::finalise_ctor_impl(
    const std::vector<expression> &, std::vector<long double>, std::uint32_t, std::vector<long double>, long double,
    bool, bool, std::vector<long double>, std::vector<t_event_t>, std::vector<nt_event_t>, bool);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch<long double>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<long double>, std::uint32_t,
    std::vector<long double>, long double, bool, bool, std::vector<long double>, std::vector<t_event_t>,
    std::vector<nt_event_t>, bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_batch<mppp::real128>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch<mppp::real128>::finalise_ctor_impl(
    const std::vector<expression> &, std::vector<mppp::real128>, std::uint32_t, std::vector<mppp::real128>,
    mppp::real128, bool, bool, std::vector<mppp::real128>, std::vector<t_event_t>, std::vector<nt_event_t>, bool);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch<mppp::real128>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<mppp::real128>, std::uint32_t,
    std::vector<mppp::real128>, mppp::real128, bool, bool, std::vector<mppp::real128>, std::vector<t_event_t>,
    std::vector<nt_event_t>, bool);

#endif

} // namespace heyoka
