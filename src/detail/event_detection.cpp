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
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <mp++/real.hpp>

#include <heyoka/detail/ed_data.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/tbb_isolated.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Evaluate the first derivative of a polynomial. Requires random-access iterator.
//
// NOTE: if T is mppp::real, the output of this function may have a precision greater than that of the poly coefficients
// and x.
template <typename InputIt, typename T>
auto poly_eval_1(InputIt a, const T &x, std::uint32_t n)
{
    assert(n >= 2u); // LCOV_EXCL_LINE

    // Init the return value.
    auto ret1 = a[n] * static_cast<T>(n);

    for (std::uint32_t i = 1; i < n; ++i) {
        // NOTE: possible optimisation for mppp::real here:
        // use fmma() directly, once exposed in mp++.
        ret1 = a[n - i] * static_cast<T>(n - i) + std::move(ret1) * x;
    }

    return ret1;
}

// Helper to detect events of terminal type.
template <typename>
struct is_terminal_event : std::false_type {};

template <typename T, bool B>
struct is_terminal_event<t_event_impl<T, B>> : std::true_type {};

template <typename T>
constexpr bool is_terminal_event_v = is_terminal_event<T>::value;

// Generic branchless sign function.
template <typename T>
int sgn(T val)
{
    return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
}

int sgn(const mppp::real &val)
{
    // NOTE: this is only used when val is guaranteed to be finite, thus this should never throw.
    assert(!val.nan_p());
    const auto ret = val.sgn();

    // NOTE: for consistency with the other implementation, don't return ret directly (as it could be different from 0,
    // -1 or 1).
    return sgn(ret);
}

// Helper to factor out common event detection code in the scalar and batch implementations.
//
// out is the output vector for the detected events. ev_vec is the vector of events. ev_idx is the index of the event
// being detected in ev_vec. te_cooldowns is the vector of cooldowns for terminal events, relevant only if Event is a
// terminal event type. roots is scratch storage for the roots detected by find_real_poly_roots() (its content is not
// important, as it will be cleared out before usage). ws is the workspace for find_real_poly_roots(), which must have
// been constructed for the polynomial degree of cfs (and, for mppp::real, with the precision of cfs and h). cfs
// contains the Taylor coefficients of the event function in ascending order (assumed finite, with at least 3
// coefficients). h is the integration timestep (assumed finite and nonzero).
//
// This function will detect the trigger time(s) of the event at index ev_idx in ev_vec and append the detected events
// to the output vector out. The appended events may include duplicates (i.e., events with the same index and trigger
// time), which may differ in the sign of the derivative.
//
// Exceptions raised by find_real_poly_roots() are propagated. Additionally, a std::runtime_error is thrown if the
// evaluation of the derivative of the event function at a root produces a non-finite value.
template <typename T, typename Event>
void detect_events_common_impl(auto &out, const std::vector<Event> &ev_vec, const std::uint32_t ev_idx,
                               [[maybe_unused]] const auto &te_cooldowns, std::vector<real_poly_root<T>> &roots,
                               frpr_workspace<T> &ws, const std::span<const T> cfs, const T &h)
{
    using std::abs;
    using std::isfinite;

    assert(isfinite(h));
    assert(h != 0);
    assert(cfs.size() > 2u);

    // Fetch the order from cfs.
    const auto order = static_cast<std::uint32_t>(cfs.size() - 1u);

    // Helper to add a detected event to out.
    //
    // NOTE: the roots detected via find_real_poly_roots() are guaranteed to be finite and to be reported in the
    // original [0, h)/(h, 0] range.
    const auto add_d_event = [&out, &cfs, order, &ev_vec, ev_idx](real_poly_root<T> &root) {
        // Get the direction of the root and an optional containing the value of the derivative of the event function at
        // the root. The latter is needed only for terminal events, but it may be computed for non-terminal events if
        // the root direction is not available from the output of poly root finding.
        auto [d_sgn, der_opt] = [&root, &cfs, order] {
            std::optional<T> d_opt;
            if (is_terminal_event_v<Event> || !root.dir) {
                // NOTE: for mppp::real poly_eval_1 may produce a value whose precision is greater than the working
                // precision. This is ok, as the derivative is used only for its sign and, for terminal events, for the
                // automatic cooldown deduction, where the cooldown values are allowed to have a different precision.
                d_opt.emplace(poly_eval_1(cfs.data(), root.t, order));

                // Check it before proceeding.
                if (!isfinite(*d_opt)) [[unlikely]] {
                    // LCOV_EXCL_START
                    throw std::runtime_error(fmt::format("The evaluation of the derivative of an event function during "
                                                         "event detection produced the non-finite value {}",
                                                         *d_opt));
                    // LCOV_EXCL_STOP
                }
            }

            if (root.dir) {
                // The direction is available, use it.
                return std::make_pair(*root.dir, std::move(d_opt));
            } else {
                // Compute the sign of the derivative from der_opt.
                assert(d_opt);
                const auto der_sgn = detail::sgn(*d_opt);
                return std::make_pair(der_sgn, std::move(d_opt));
            }
        }();

        // Fetch and cache the desired event direction.
        const auto dir = ev_vec[ev_idx].get_direction();

        // The event must be added if the event direction does not matter, or if it matches the sign of the derivative.
        if (dir == event_direction::any || static_cast<event_direction>(d_sgn) == dir) {
            if constexpr (is_terminal_event_v<Event>) {
                assert(der_opt);
                out.emplace_back(ev_idx, std::move(root.t), d_sgn, abs(std::move(*der_opt)));
            } else {
                out.emplace_back(ev_idx, std::move(root.t), d_sgn);
            }
        }
    };

    // NOTE: if we are dealing with a terminal event on cooldown, we will need to ignore roots within the cooldown
    // period. lb is the distance from the beginning of the timestep to the end of the cooldown.
    //
    // NOTE: for mppp::real, lb may come out with the wrong precision because the cooldown values are not guaranteed to
    // have the working precision. This is ok, as find_real_poly_roots() will fix lb's precision as needed.
    //
    // NOLINTNEXTLINE(misc-const-correctness)
    auto lb = [&] {
        if constexpr (is_terminal_event_v<Event>) {
            if (te_cooldowns[ev_idx]) {
                // NOTE: need to distinguish between forward and backward integration.
                //
                // NOTE: the cooldown values are finite and |first| <= second, thus the worst that can happen here is
                // the generation of a positive infinity or negative 0 (which are valid values for
                // find_real_poly_roots()).
                if (h < 0) {
                    return te_cooldowns[ev_idx]->second + te_cooldowns[ev_idx]->first;
                } else {
                    return te_cooldowns[ev_idx]->second - te_cooldowns[ev_idx]->first;
                }
            }
        }

        // NOTE: we end up here if the event is not terminal or not on cooldown.
        return num_zero_like(h);
    }();

    // Run root finding.
    find_real_poly_roots(roots, ws, cfs, h, std::move(lb));

    // Add the detected events.
    for (auto &root : roots) {
        add_d_event(root);
    }
}

} // namespace

// Helper to automatically deduce the cooldown for a terminal event. g_eps is the maximum absolute error on the Taylor
// series of the event equation (which depends on the integrator tolerance and the infinity norm of the state
// vector/event equations), abs_der is the absolute value of the time derivative of the event equation at the zero.
//
// The result is always finite and strictly positive.
template <typename T>
T taylor_deduce_cooldown(T g_eps, T abs_der)
{
    using std::isfinite;
    using std::nextafter;

    // LCOV_EXCL_START
    assert(isfinite(g_eps));
    assert(isfinite(abs_der));
    assert(g_eps > 0);
    assert(abs_der >= 0);
    // LCOV_EXCL_STOP

    // NOTE: the * 10 is a safety factor composed of:
    //
    // - 2 is the original factor from theoretical considerations,
    // - 2 factor to deal with very small values of the derivative,
    // - 2 factor to deal with the common case of event equation flipping around after the event (e.g., for collisions).
    //
    // The rest is additional safety.
    //
    // NOTE: clang-tidy suggests constness but this prevents moving out (important for mppp::real).
    //
    // NOLINTNEXTLINE(misc-const-correctness)
    auto ret = std::move(g_eps) / std::move(abs_der) * 10;

    if (isfinite(ret)) [[likely]] {
        if (ret == 0) [[unlikely]] {
            // ret can come out zero in case of underflow. In such case, we want to avoid an exactly zero cooldown.
            ret = nextafter(std::move(ret), static_cast<T>(1));
        }

        return ret;
    } else {
        // The heuristic produced a non-finite value. Since we are assuming that g_eps is strictly positive, the only
        // non-finite result for ret can be +infinity. Replace it with a large number in order to keep the spirit of a
        // very large cooldown without the headache of making it non-finite.
        assert(ret == std::numeric_limits<float>::infinity());
        return nextafter(std::move(ret), static_cast<T>(0));
    }
}

// Explicit instantiations.
#define HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST(F) template F taylor_deduce_cooldown<F>(F, F);

HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST(float)
HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST(mppp::real128)

#endif

HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST(mppp::real)

#undef HEYOKA_TAYLOR_DEDUCE_COOLDOWN_INST

// Helper to determine the sign of the polynomial with coefficients cfs at the point x.
//
// cfs is assumed to contain the coefficients in ascending degree order starting from cfs[offset] at intervals of
// stride. E.g., given cfs = [a, b, c, d, e, f, g, h], offset==1 and stride==2, the list of coefficients is [b, d, f,
// h].
//
// The polynomial is evaluated via Horner's scheme in interval arithmetic: each operation is performed in
// round-to-nearest, and its result is then widened outwards to the adjacent representable values. Since a
// round-to-nearest result differs from the exact result by at most half the gap to the adjacent representable values,
// the final interval is guaranteed to enclose the exact value of the polynomial at x.
//
// The return value is +1 or -1 if the enclosure lies entirely above or below zero, and an empty optional otherwise
// (that is, if the enclosure contains zero or is not finite).
//
// cfs must be non-empty, and both the coefficients selected from cfs for evaluation and x must be finite. For
// mppp::real, cfs and x must all have the same precision. offset must be smaller than stride, and stride must divide
// the size of cfs exactly.
template <typename T>
[[nodiscard]] std::optional<int> ed_poly_eval_sign_ival(const std::span<const T> cfs, const T &x,
                                                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                        const std::size_t offset, const std::size_t stride)
{
    using std::isfinite;

    assert(offset < stride);
    assert(!cfs.empty());
    assert(isfinite(x));

    if constexpr (std::same_as<T, mppp::real>) {
        assert(std::ranges::all_of(cfs, [x_prec = x.get_prec()](const auto &val) { return val.get_prec() == x_prec; }));
    }

    const auto cfs_size = cfs.size();
    assert(cfs_size % stride == 0u);

    // Determine the degree+1 of the polynomial.
    const auto dp1 = cfs_size / stride;

    // Helper to fetch a coefficient from cfs.
    const auto fetch_cf = [&](const std::size_t degree) -> const T & {
        assert(degree < dp1);
        return cfs[(degree * stride) + offset];
    };

    // Check that all the coefficients which participate in the evaluation are finite.
    assert(std::ranges::all_of(std::views::iota(0uz, dp1) | std::views::transform(fetch_cf),
                               [](const auto &val) { return isfinite(val); }));

    // Init the enclosure with the highest-degree coefficient.
    auto lo = fetch_cf(dp1 - 1u), hi = lo;

    // Cache the sign of x.
    const auto x_is_neg = (x < 0);

    for (auto i = dp1 - 1u; i > 0u; --i) {
        const auto &cf = fetch_cf(i - 1u);

        // Multiply the enclosure by x. The bounds swap if x is negative.
        //
        // NOLINTBEGIN(misc-const-correctness)
        auto prod_lo = x_is_neg ? hi * x : lo * x;
        auto prod_hi = x_is_neg ? lo * x : hi * x;
        // NOLINTEND(misc-const-correctness)

        // Widen the products, add the coefficient and widen again.
        lo = num_nextdown(num_nextdown(std::move(prod_lo)) + cf);
        hi = num_nextup(num_nextup(std::move(prod_hi)) + cf);
    }

    if (!isfinite(lo) || !isfinite(hi)) [[unlikely]] {
        // NOTE: this can happen in case of overflow.
        return {};
    }

    if (lo > 0) {
        return 1;
    }

    if (hi < 0) {
        return -1;
    }

    return {};
}

// Explicit instantiations.
#define HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST(F)                                                                          \
    template std::optional<int> ed_poly_eval_sign_ival<F>(std::span<const F>, const F &, std::size_t, std::size_t);

HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST(float)
HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST(double)

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST(long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST(mppp::real128)

#endif

HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST(mppp::real)

#undef HEYOKA_ED_POLY_EVAL_SIGN_IVAL_INST

// NOTE: the def ctor is used only for serialisation purposes.
template <typename T>
ed_data<T>::ed_data() = default;

// Main ctor.
template <typename T>
ed_data<T>::ed_data(const llvm_state &s, std::vector<t_event_t> tes, std::vector<nt_event_t> ntes,
                    const std::uint32_t order, const std::uint32_t dim, const T &s0)
    : m_tes(std::move(tes)), m_ntes(std::move(ntes))
{
    assert(!m_tes.empty() || !m_ntes.empty()); // LCOV_EXCL_LINE

    const auto n_tes = m_tes.size();
    const auto n_ntes = m_ntes.size();

    // Setup m_ev_jet.
    //
    // NOTE: check that we can represent the requested size for m_ev_jet using both its size type and std::uint32_t.
    // This also indirectly checks that n_tes and n_ntes are representable as 32-bit unsigned ints.
    const auto ev_jet_size = [&] {
        try {
            using safe_ev_jet_size_t = boost::safe_numerics::safe<decltype(m_ev_jet.size())>;
            const auto ret = (safe_ev_jet_size_t(n_tes) + n_ntes + dim) * (order + 1u);
            static_cast<void>(static_cast<std::uint32_t>(ret));
            return ret;
            // LCOV_EXCL_START
        } catch (...) {
            throw std::overflow_error(
                "Overflow detected in the initialisation of an adaptive Taylor integrator: the order "
                "or the state size is too large");
        }
        // LCOV_EXCL_STOP
    }();
    m_ev_jet.resize(ev_jet_size, num_zero_like(s0));

    // Setup the vector of cooldowns.
    m_te_cooldowns.resize(boost::numeric_cast<decltype(m_te_cooldowns.size())>(m_tes.size()));

    // Setup m_ev_fex_check_res.
    {
        using safe_size_t = boost::safe_numerics::safe<decltype(m_ev_fex_check_res.size())>;
        // NOTE: the content of this vector is not important, just make sure it is sized correctly.
        m_ev_fex_check_res.resize(safe_size_t(n_tes) + n_ntes);
    }

    // Setup m_ev_signs_end_step.
    {
        using safe_size_t = boost::safe_numerics::safe<decltype(m_ev_signs_end_step.size())>;
        // NOTE: the signs must be initialised in the unknown state.
        m_ev_signs_end_step.resize(safe_size_t(n_tes) + n_ntes, event_sign_end_step::unknown);
    }

    // Setup in parallel the JIT data and the poly root finding workspace.
    tbb_isolated_parallel_invoke(
        [this, &s, order, &s0] {
            // Create the ed_jit_data instance.
            m_jit_data = std::make_shared<ed_jit_data<T>>(s.make_similar());

            // Fetch the constructed state.
            auto &state = m_jit_data->m_state;

            // Fetch the scalar FP type.
            //
            // NOTE: s0 is the first value in the state vector of the integrator, from which the internal floating-point
            // type is deduced.
            auto *const fp_t = internal_llvm_type_like(state, s0);

            // Add the function for the fast exclusion check.
            llvm_ed_add_fex_check(state, "fex_check", fp_t, order, 1);

            // Compile.
            state.compile();

            // Fetch the function pointer.
            m_jit_data->m_fex_check = reinterpret_cast<ed_jit_data<T>::fex_check_t>(state.jit_lookup("fex_check"));
        },
        [this, &s, order, &s0] {
            // Infer the precision argument for the polynomial root finding workspace.
            const auto prec = [&s0] -> long long {
                static_cast<void>(s0);

                if constexpr (std::same_as<T, mppp::real>) {
                    return boost::numeric_cast<long long>(s0.get_prec());
                }

                return 0;
            }();

            // Setup the polynomial root finding workspace.
            m_frpr_ws = frpr_workspace<T>(order, prec, &s);
        });
}

// NOTE: the detected events and m_poly_roots are scratch space and don't need to be copied. m_ev_jet and
// m_ev_fex_check_res need only to be sized correctly.
template <typename T>
ed_data<T>::ed_data(const ed_data &o)
    : m_tes(o.m_tes), m_ntes(o.m_ntes), m_te_cooldowns(o.m_te_cooldowns), m_jit_data(o.m_jit_data),
      m_frpr_ws(o.m_frpr_ws), m_ev_signs_end_step(o.m_ev_signs_end_step)
{
    // NOTE: for real we rely on o.m_ev_jet to contain values with the correct precision.
    assert(!o.m_ev_jet.empty());
    m_ev_jet.resize(o.m_ev_jet.size(), num_zero_like(o.m_ev_jet[0]));
    m_ev_fex_check_res.resize(o.m_ev_fex_check_res.size());
}

template <typename T>
ed_data<T>::~ed_data() = default;

// NOTE: for serialisation, we do not want to save any scratch data. Deserialisation clears out and sizes appropriately
// the scratch data, in analogy with the copy constructor.
template <typename T>
void ed_data<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_tes;
    ar << m_ntes;

    // NOTE: for m_ev_jet we save only the size and, if necessary, the precision.
    ar << m_ev_jet.size();
    if constexpr (std::same_as<T, mppp::real>) {
        assert(!m_ev_jet.empty());
        ar << m_ev_jet[0].get_prec();
    }

    ar << m_te_cooldowns;
    ar << m_jit_data;
    ar << m_frpr_ws;
    ar << m_ev_signs_end_step;
}

template <typename T>
void ed_data<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_tes;
    ar >> m_ntes;

    // Clear and resize m_ev_jet correctly.
    m_ev_jet.clear();
    decltype(m_ev_jet.size()) ev_jet_size{};
    ar >> ev_jet_size;
    if constexpr (std::same_as<T, mppp::real>) {
        mpfr_prec_t prec{};
        ar >> prec;
        m_ev_jet.resize(ev_jet_size, mppp::real{mppp::real_kind::zero, prec});
    } else {
        m_ev_jet.resize(ev_jet_size);
    }

    ar >> m_te_cooldowns;
    ar >> m_jit_data;
    ar >> m_frpr_ws;
    ar >> m_ev_signs_end_step;

    // NOTE: these can be restored to their default-constructed state.
    m_d_tes.clear();
    m_d_ntes.clear();
    m_poly_roots.clear();

    // Make sure m_ev_fex_check_res is restored with the correct size, which we can compute.
    m_ev_fex_check_res.clear();
    m_ev_fex_check_res.resize(static_cast<decltype(m_ev_fex_check_res.size())>(m_tes.size()) + m_ntes.size());
}

// Implementation of event detection.
//
// NOTE: the list of detected events may contain duplicates which will have to be removed outside this function.
template <typename T>
void ed_data<T>::detect_events(const T &h, const std::uint32_t order, const std::uint32_t dim)
{
    using std::isfinite;

    assert(isfinite(h));

    // As a first step, the vectors of detected events must be cleared out.
    m_d_tes.clear();
    m_d_ntes.clear();

    if (h == 0) {
        // If the timestep is zero, skip event detection.
        return;
    }

    // Fetch the fex check function pointer.
    auto *const m_fex_check = m_jit_data->m_fex_check;

    // Array to store the polynomial enclosures computed by m_fex_check().
    //
    // NOTE: make sure that we create a value with the correct precision for mppp::real. The content does not matter.
    // The enclosures are not needed here, and they are ignored.
    const auto zero = num_zero_like(h);
    T enc[] = {zero, zero};

    // Helper to run event detection on a vector of events (terminal or not). 'out' is the vector of detected events,
    // 'ev_vec' the input vector of events to detect.
    const auto run_detection = [&]<typename Event>(auto &out, const std::vector<Event> &ev_vec) {
        const auto offset = is_terminal_event_v<Event> ? 0u : m_tes.size();

        for (std::uint32_t i = 0; i < ev_vec.size(); ++i) {
            // Extract the pointer to the Taylor polynomial for the current event.
            const auto *const ptr = m_ev_jet.data() + (i + dim + offset) * (order + 1u);

            // Run the fast exclusion check.
            //
            // NOTE: we are assuming that if we get here we have checked that all the finiteness preconditions for the
            // invocation of m_fex_check() hold.
            std::uint32_t fex_check_result{};
            m_fex_check(ptr, &h, &fex_check_result, enc);

            // Store the result of the check before proceeding.
            //
            // NOTE: offset + i is computed in at least 32-bit unsigned width and it is guaranteed to be <
            // m_ev_jet.size(), which elsewhere has been checked to fit 32-bit unsigned ints. Hence, no
            // overflow/wraparound here is possible.
            m_ev_fex_check_res[offset + i] = fex_check_result;
            if (fex_check_result) {
                // The fast exclusion check returned true, we can move to the next event.
                continue;
            }

            // Run the event detection.
            const std::span<const T> cfs(ptr, boost::numeric_cast<std::size_t>(order + 1u));
            detail::detect_events_common_impl(out, ev_vec, i, m_te_cooldowns, m_poly_roots, m_frpr_ws, cfs, h);
        }
    };

    run_detection(m_d_tes, m_tes);
    run_detection(m_d_ntes, m_ntes);
}

// Explicit instantiation of the book-keeping structures for event detection in the scalar integrator.
template struct ed_data<float>;
template struct ed_data<double>;

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

template struct ed_data<long double>;

#endif

#if defined(HEYOKA_HAVE_REAL128)

template struct ed_data<mppp::real128>;

#endif

template struct ed_data<mppp::real>;

// NOTE: the def ctor is used only for serialisation purposes.
template <typename T>
ed_data_batch<T>::ed_data_batch() = default;

template <typename T>
ed_data_batch<T>::ed_data_batch(const llvm_state &s, std::vector<t_event_t> tes, std::vector<nt_event_t> ntes,
                                const std::uint32_t order, const std::uint32_t dim, const std::uint32_t batch_size)
    : m_tes(std::move(tes)), m_ntes(std::move(ntes))
{
    assert(order >= 2u);
    assert(!m_tes.empty() || !m_ntes.empty());
    assert(batch_size != 0u);

    const auto n_tes = m_tes.size();
    const auto n_ntes = m_ntes.size();

    // Setup m_ev_jet.
    //
    // NOTE: check that we can represent the requested size for m_ev_jet using both its size type and std::uint32_t.
    // This also indirectly checks that n_tes and n_ntes are representable as 32-bit unsigned ints.
    const auto ev_jet_size = [&] {
        try {
            using safe_ev_jet_size_t = boost::safe_numerics::safe<decltype(m_ev_jet.size())>;
            const auto ret = (safe_ev_jet_size_t(n_tes) + n_ntes + dim) * (order + 1u) * batch_size;
            static_cast<void>(static_cast<std::uint32_t>(ret));
            return ret;
            // LCOV_EXCL_START
        } catch (...) {
            throw std::overflow_error("Overflow detected in the initialisation of an adaptive Taylor integrator in "
                                      "batch mode: the order or the state size is too large");
        }
        // LCOV_EXCL_STOP
    }();
    m_ev_jet.resize(ev_jet_size);

    // Prepare m_ed_finite.
    m_ed_finite.resize(boost::numeric_cast<decltype(m_ed_finite.size())>(batch_size));

    // Prepare m_max_abs_state.
    m_max_abs_state.resize(batch_size);

    // Prepare m_g_eps.
    m_g_eps.resize(batch_size);

    // Prepare m_d_tes.
    m_d_tes.resize(boost::numeric_cast<decltype(m_d_tes.size())>(batch_size));

    // Setup the vector of cooldowns.
    m_te_cooldowns.resize(boost::numeric_cast<decltype(m_te_cooldowns.size())>(batch_size));
    for (auto &v : m_te_cooldowns) {
        v.resize(boost::numeric_cast<decltype(v.size())>(m_tes.size()));
    }

    // Prepare m_d_ntes.
    m_d_ntes.resize(boost::numeric_cast<decltype(m_d_ntes.size())>(batch_size));

    // Prepare m_fex_check_res.
    m_fex_check_res.resize(boost::numeric_cast<decltype(m_fex_check_res.size())>(batch_size));

    // Prepare m_fex_check_enc.
    //
    // NOTE: batch_size * 2u is safe, we checked earlier that (order + 1u) * batch_size (with order guaranteed >= 2) is
    // representable both by std::uint32_t and the size type of m_ev_jet, which has the same size type as
    // m_fex_check_enc.
    m_fex_check_enc.resize(batch_size * 2u);

    // Prepare m_cb_eptrs.
    //
    // NOTE: this vector is normally kept empty. During the execution of step_impl(), it will end up containing at most
    // batch_size elements, and it is guaranteed to be cleared out when step_impl() returns. Because we reserve its
    // maximum possible size, it is guaranteed that no exceptions will be thrown when pushing back.
    m_cb_eptrs.reserve(boost::numeric_cast<decltype(m_cb_eptrs.size())>(batch_size));

    // Prepare m_scal_poly.
    //
    // NOTE: order + 1u is safe for conversion to the size type of m_scal_poly: the size type is the same as m_ev_jet's,
    // which has been proven earlier to be able to represent order + 1u.
    m_scal_poly.resize(order + 1u);

    // Setup m_ev_fex_check_res.
    {
        using safe_size_t = boost::safe_numerics::safe<decltype(m_ev_fex_check_res.size())>;
        // NOTE: the content of this vector is not important, just make sure it is sized correctly.
        m_ev_fex_check_res.resize((safe_size_t(n_tes) + n_ntes) * batch_size);
    }

    // Setup m_ev_signs_end_step.
    {
        using safe_size_t = boost::safe_numerics::safe<decltype(m_ev_signs_end_step.size())>;
        // NOTE: the signs must be initialised in the unknown state.
        m_ev_signs_end_step.resize((safe_size_t(n_tes) + n_ntes) * batch_size, event_sign_end_step::unknown);
    }

    // Setup in parallel the JIT data and the poly root finding workspace.
    tbb_isolated_parallel_invoke(
        [this, &s, order, batch_size] {
            // Create the ed_jit_data instance.
            m_jit_data = std::make_shared<ed_jit_data<T>>(s.make_similar());

            // Fetch the constructed state.
            auto &state = m_jit_data->m_state;

            // Fetch the scalar FP type.
            auto *const fp_t = to_external_llvm_type<T>(state.context());

            // Add the function for the fast exclusion check.
            //
            // NOTE: the fast exclusion check is vectorised.
            llvm_ed_add_fex_check(state, "fex_check", fp_t, order, batch_size);

            // Compile.
            state.compile();

            // Fetch the function pointer.
            m_jit_data->m_fex_check = reinterpret_cast<ed_jit_data<T>::fex_check_t>(state.jit_lookup("fex_check"));
        },
        [this, &s, order] {
            // Setup the polynomial root finding workspace.
            m_frpr_ws = frpr_workspace<T>(order, 0, &s);
        });
}

// NOTE: the scratch data is not copied, it only needs to be sized correctly (for the detected events, this means one
// empty vector per batch lane). m_poly_roots is left empty, while m_cb_eptrs must be reserved to the batch size in
// order to guarantee that pushing back into it in step_impl() does not throw.
template <typename T>
ed_data_batch<T>::ed_data_batch(const ed_data_batch &o)
    : m_tes(o.m_tes), m_ntes(o.m_ntes), m_te_cooldowns(o.m_te_cooldowns), m_jit_data(o.m_jit_data),
      m_frpr_ws(o.m_frpr_ws), m_ev_signs_end_step(o.m_ev_signs_end_step)
{
    // Fetch the batch size.
    const auto batch_size = static_cast<std::uint32_t>(o.m_d_tes.size());

    // Prepare these vectors with the correct sizes.
    m_ev_jet.resize(o.m_ev_jet.size());
    m_ed_finite.resize(o.m_ed_finite.size());
    m_max_abs_state.resize(o.m_max_abs_state.size());
    m_g_eps.resize(o.m_g_eps.size());
    m_d_tes.resize(o.m_d_tes.size());
    m_d_ntes.resize(o.m_d_ntes.size());
    m_fex_check_res.resize(o.m_fex_check_res.size());
    m_fex_check_enc.resize(o.m_fex_check_enc.size());
    m_scal_poly.resize(o.m_scal_poly.size());
    m_ev_fex_check_res.resize(o.m_ev_fex_check_res.size());

    // Reserve space in m_cb_eptrs.
    m_cb_eptrs.reserve(batch_size);
}

template <typename T>
ed_data_batch<T>::~ed_data_batch() = default;

// NOTE: for serialisation, we do not want to save any scratch data. Deserialisation clears out and sizes appropriately
// the scratch data, in analogy with the copy constructor.
template <typename T>
void ed_data_batch<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    // Start by saving the batch size and the order+1.
    ar << static_cast<std::uint32_t>(m_d_tes.size());
    ar << static_cast<std::uint32_t>(m_scal_poly.size());

    ar << m_tes;
    ar << m_ntes;
    // NOTE: for m_ev_jet we save only the size.
    ar << m_ev_jet.size();
    ar << m_te_cooldowns;
    ar << m_jit_data;
    ar << m_frpr_ws;
    ar << m_ev_signs_end_step;
}

template <typename T>
void ed_data_batch<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    // Recover the batch size and the order+1.
    std::uint32_t batch_size{};
    ar >> batch_size;
    std::uint32_t op1{};
    ar >> op1;

    ar >> m_tes;
    ar >> m_ntes;

    // Clear and resize m_ev_jet correctly.
    m_ev_jet.clear();
    decltype(m_ev_jet.size()) ev_jet_size{};
    ar >> ev_jet_size;
    m_ev_jet.resize(ev_jet_size);

    ar >> m_te_cooldowns;
    ar >> m_jit_data;
    ar >> m_frpr_ws;
    ar >> m_ev_signs_end_step;

    // Reset the scratch data.
    const auto clear_and_resize = [](auto &vec, const auto sz) {
        vec.clear();
        vec.resize(sz);
    };
    clear_and_resize(m_ed_finite, batch_size);
    clear_and_resize(m_max_abs_state, batch_size);
    clear_and_resize(m_g_eps, batch_size);
    m_d_tes.resize(batch_size);
    for (auto &v : m_d_tes) {
        v.clear();
    }
    m_d_ntes.resize(batch_size);
    for (auto &v : m_d_ntes) {
        v.clear();
    }
    clear_and_resize(m_fex_check_res, batch_size);
    clear_and_resize(m_fex_check_enc, batch_size * 2u);
    clear_and_resize(m_scal_poly, op1);
    clear_and_resize(m_ev_fex_check_res,
                     (static_cast<decltype(m_ev_fex_check_res.size())>(m_tes.size()) + m_ntes.size()) * batch_size);

    // Clear and reserve the capacity in m_cb_eptrs.
    m_cb_eptrs.clear();
    m_cb_eptrs.reserve(batch_size);

    // Clear out m_poly_roots.
    m_poly_roots.clear();
}

// Implementation of event detection.
//
// NOTE: the list of detected events may contain duplicates which will have to be removed outside this function.
//
// NOTE: batch lanes whose flag in m_ed_finite is false are skipped. The flags must be set by the caller before invoking
// this function.
template <typename T>
void ed_data_batch<T>::detect_events(const T *const h_ptr, const std::uint32_t order, const std::uint32_t dim,
                                     const std::uint32_t batch_size)
{
    using std::isfinite;

    // Clear the vectors of detected events.
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        m_d_tes[i].clear();
        m_d_ntes[i].clear();
    }

    // Fetch the fex check function pointer.
    auto *const m_fex_check = m_jit_data->m_fex_check;

    // Helper to run event detection on a vector of events (terminal or not). 'out_vec' is the vector of detected
    // events, 'ev_vec' the input vector of events to detect.
    const auto run_detection = [&]<typename Event>(auto &out_vec, const std::vector<Event> &ev_vec) {
        const auto offset = is_terminal_event_v<Event> ? 0u : m_tes.size();

        for (std::uint32_t i = 0; i < ev_vec.size(); ++i) {
            // Extract the pointer to the Taylor polynomial for the current event.
            const auto *const batch_ptr = m_ev_jet.data() + (i + dim + offset) * (order + 1u) * batch_size;

            // Run the fast exclusion check.
            //
            // NOTE: in lanes with non-finite values in h or in the Taylor coefficients of the event function, the
            // results of the check are meaningless. This is ok, because event detection is skipped for these lanes
            // below without looking at the output of m_fex_check(), and the check operates independently on each lane,
            // so that non-finite values in one lane do not affect the results in the other lanes.
            m_fex_check(batch_ptr, h_ptr, m_fex_check_res.data(), m_fex_check_enc.data());

            // Run event detection on all the batch elements.
            for (std::uint32_t j = 0; j < batch_size; ++j) {
                // Check if we are in a lane which contains non-finite data.
                if (!m_ed_finite[j]) [[unlikely]] {
                    continue;
                }

                // Load the timestep value.
                const auto h = h_ptr[j];
                assert(isfinite(h));

                if (h == 0) {
                    // If the timestep is zero, skip event detection.
                    continue;
                }

                // Store the result of the fast exclusion check before proceeding.
                //
                // NOTE: (offset + i) * batch_size + j is computed in at least 32-bit unsigned width and it is
                // guaranteed to be < m_ev_jet.size(), which elsewhere has been checked to fit 32-bit unsigned ints.
                // Hence, no overflow/wraparound here is possible.
                m_ev_fex_check_res[(offset + i) * batch_size + j] = m_fex_check_res[j];
                if (m_fex_check_res[j]) {
                    // The fast exclusion check returned true, we can move to the next batch lane.
                    continue;
                }

                // Fetch a reference to the vector of detected events.
                auto &out = out_vec[j];

                // Copy the polynomial coefficients from batch_ptr to m_scal_poly.
                for (std::uint32_t k = 0; k <= order; ++k) {
                    m_scal_poly[k] = *(batch_ptr + j + k * batch_size);
                }
                const auto *const ptr = m_scal_poly.data();

                // Run the event detection.
                const std::span<const T> cfs(ptr, boost::numeric_cast<std::size_t>(order + 1u));
                detail::detect_events_common_impl(out, ev_vec, i, m_te_cooldowns[j], m_poly_roots, m_frpr_ws, cfs, h);
            }
        }
    };

    run_detection(m_d_tes, m_tes);
    run_detection(m_d_ntes, m_ntes);
}

// Explicit instantiation of the book-keeping structures for event detection in the batch integrator.
template struct ed_data_batch<float>;
template struct ed_data_batch<double>;

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

template struct ed_data_batch<long double>;

#endif

#if defined(HEYOKA_HAVE_REAL128)

template struct ed_data_batch<mppp::real128>;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
