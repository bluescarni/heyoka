// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/numeric/conversion/cast.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <boost/multiprecision/float128.hpp>

#include <mp++/real128.hpp>

#endif

#include <fmt/ostream.h>

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/taylor.hpp>

namespace heyoka::detail
{

namespace
{

#if !defined(NDEBUG)

// Debug functions to check the computation of
// the binomial coefficients.

template <typename T>
auto boost_math_bc(std::uint32_t n_, std::uint32_t k_)
{
    const auto n = boost::numeric_cast<unsigned>(n_);
    const auto k = boost::numeric_cast<unsigned>(k_);

#if defined(HEYOKA_HAVE_REAL128)
    if constexpr (std::is_same_v<T, mppp::real128>) {
        using bf128 = boost::multiprecision::float128;

        return mppp::real128{boost::math::binomial_coefficient<bf128>(n, k).backend().value()};
    } else {
#endif
        return boost::math::binomial_coefficient<T>(n, k);
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

#endif

// Helper to fetch the per-thread poly cache.
template <typename T>
auto &get_poly_cache()
{
    thread_local std::vector<std::vector<std::vector<T>>> ret;

    return ret;
}

// Extract a poly of order n from the cache (or create a new one).
template <typename T>
auto get_poly_from_cache(std::uint32_t n)
{
    // Get/create the thread-local cache.
    auto &cache = get_poly_cache<T>();

    // Look if we have inited the cache for order n.
    if (n >= cache.size()) {
        // The cache was never used for polynomials of order
        // n, add the necessary entries.

        // NOTE: no overflow check needed here, cause the order
        // is always overflow checked in the integrator machinery.
        cache.resize(boost::numeric_cast<decltype(cache.size())>(n + 1u));
    }

    auto &pcache = cache[n];

    if (pcache.empty()) {
        // No polynomials are available, create a new one.
        return std::vector<T>(boost::numeric_cast<typename std::vector<T>::size_type>(n + 1u));
    } else {
        // Extract an existing polynomial from the cache.
        auto retval = std::move(pcache.back());
        pcache.pop_back();

        return retval;
    }
}

// Insert a poly into the cache.
template <typename T>
void put_poly_in_cache(std::vector<T> &&v)
{
    // Get/create the thread-local cache.
    auto &cache = get_poly_cache<T>();

    // Fetch the order of the polynomial.
    // NOTE: the order is the size - 1.
    assert(!v.empty());
    const auto n = v.size() - 1u;

    // Look if we have inited the cache for order n.
    if (n >= cache.size()) {
        // The cache was never used for polynomials of order
        // n, add the necessary entries.

        if (n == std::numeric_limits<decltype(cache.size())>::max()) {
            throw std::overflow_error("An overflow was detected in the polynomial cache");
        }
        cache.resize(n + 1u);
    }

    // Move v in.
    cache[n].push_back(std::move(v));
}

// Compute and return all binomial coefficients (n choose k)
// up to n = max_n.
template <typename T>
auto make_binomial_coefficients(std::uint32_t max_n)
{
    assert(max_n >= 2u);

    if (max_n > std::numeric_limits<std::uint32_t>::max() - 2u
        || (max_n + 1u) > std::numeric_limits<std::uint32_t>::max() / (max_n + 2u)) {
        throw std::overflow_error("An overflow was detected while generating a list of binomial coefficients");
    }

    std::vector<T> retval;
    retval.resize(boost::numeric_cast<decltype(retval.size())>(((max_n + 1u) * (max_n + 2u)) / 2u));

    // Fill up to n = 2.

    // 0 choose 0.
    retval[0] = 1;

    // 1 choose 0.
    retval[1] = 1;
    // 1 choose 1.
    retval[2] = 1;

    // 2 choose 0.
    retval[3] = 1;
    // 2 choose 1.
    retval[4] = 2;
    // 2 choose 2.
    retval[5] = 1;

    // Iterate using the recursion formula.
    std::uint32_t base_idx = 6;
    for (std::uint32_t n = 3; n <= max_n; base_idx += ++n) {
        // n choose 0 = 1.
        retval[base_idx] = 1;

        // NOTE: the recursion formula is valid up to k = n - 1.
        const auto prev_base_idx = base_idx - n;
        for (std::uint32_t k = 1; k < n; ++k) {
            retval[base_idx + k] = retval[prev_base_idx + k] + retval[prev_base_idx + k - 1u];
        }

        // n choose n = 1.
        retval[base_idx + n] = 1;
    }

    return retval;
}

// Fetch the index of (n choose k) in a vector produced by
// make_binomial_coefficients().
std::uint32_t bc_idx(std::uint32_t n, std::uint32_t k)
{
    assert(k <= n);

    return (n * (n + 1u)) / 2u + k;
}

// Helper to fetch the per-thread cache of binomial coefficients.
template <typename T>
auto &get_bc_cache()
{
    thread_local std::vector<std::vector<T>> ret;

    return ret;
}

// Fetch from the cache the list of all binomial
// coefficients (n choose k) up to n = max_n.
template <typename T>
const auto &get_bc_up_to(std::uint32_t max_n)
{
    auto &cache = get_bc_cache<T>();

    if (max_n >= cache.size()) {
        cache.resize(boost::numeric_cast<decltype(cache.size())>(max_n + 1u));
    }

    auto &bcache = cache[max_n];

    if (bcache.empty()) {
        // NOTE: here, in principle, rather than computing all
        // binomial coefficients for each different value of max_n,
        // we could re-use the coefficients computed for lower values
        // of max_n. I don't think this makes much of a difference in practice,
        // as the computation is done once per thread and most likely with
        // not many different values of max_n. Keep this in mind
        // in any case.
        bcache = make_binomial_coefficients<T>(max_n);
    }

    return bcache;
}

// Given an input polynomial a(x), substitute
// x with x_1 * h and write to ret the resulting
// polynomial in the new variable x_1. Requires
// random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt, typename T>
void poly_rescale(OutputIt ret, InputIt a, const T &scal, std::uint32_t n)
{
    T cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = cur_f * a[i];
        cur_f *= scal;
    }
}

// Transform the polynomial a(x) into 2**n * a(x / 2).
// Requires random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt>
void poly_rescale_p2(OutputIt ret, InputIt a, std::uint32_t n)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    value_type cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[n - i] = cur_f * a[n - i];
        cur_f *= 2;
    }
}

// Substitute the polynomial variable x with x_1 + 1,
// and write the resulting polynomial in ret. bcs
// is a vector containing the binomial coefficients
// up to to (n choose n) in the format returned
// by make_binomial_coefficients().
// Requires random-access iterators.
// NOTE: aliasing NOT allowed.
template <typename OutputIt, typename InputIt, typename BCs>
void poly_translate_1(OutputIt ret, InputIt a, std::uint32_t n, const BCs &bcs)
{
    using value_type [[maybe_unused]] = typename std::iterator_traits<InputIt>::value_type;

    // Zero out the return value.
    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = 0;
    }

    for (std::uint32_t i = 0; i <= n; ++i) {
        const auto ai = a[i];

        for (std::uint32_t k = 0; k <= i; ++k) {
            assert(bc_idx(i, k) < bcs.size());

#if !defined(NDEBUG)
            // In debug mode check that the binomial coefficient
            // was computed correctly.
            using std::abs;

            auto cmp = boost_math_bc<value_type>(i, k);
            auto bc = bcs[bc_idx(i, k)];
            assert(abs((bc - cmp) / cmp) < std::numeric_limits<value_type>::epsilon() * 1e4);
#endif

            ret[k] += ai * bcs[bc_idx(i, k)];
        }
    }
}

// Count the number of sign changes in the coefficients of polynomial a.
// Zero coefficients are skipped. Requires random-access iterator.
template <typename InputIt>
std::uint32_t count_sign_changes(InputIt a, std::uint32_t n)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    struct zero_filter {
        bool operator()(const value_type &x) const
        {
            return x != 0;
        }
    };

    // Create iterators for skipping zero coefficients in the polynomial.
    auto begin = boost::make_filter_iterator<zero_filter>(a, a + n + 1u);
    const auto end = boost::make_filter_iterator<zero_filter>(a + n + 1u, a + n + 1u);

    if (begin == end) {
        return 0;
    }

    std::uint32_t retval = 0;

    auto prev = begin;
    for (++begin; begin != end; ++begin, ++prev) {
        retval += (*begin > 0) != (*prev > 0);
    }

    return retval;
}

// Evaluate the first derivative of a polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval_1(InputIt a, T x, std::uint32_t n)
{
    assert(n >= 2u);

    // Init the return value.
    auto ret1 = a[n] * n;

    for (std::uint32_t i = 1; i < n; ++i) {
        ret1 = a[n - i] * (n - i) + ret1 * x;
    }

    return ret1;
}

// Evaluate polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval(InputIt a, T x, std::uint32_t n)
{
    auto ret = a[n];

    for (std::uint32_t i = 1; i <= n; ++i) {
        ret = a[n - i] + ret * x;
    }

    return ret;
}

// A RAII helper to extract polys from the cache and
// return them to the cache upon destruction.
template <typename T>
struct pwrap {
    explicit pwrap(std::uint32_t n) : v(get_poly_from_cache<T>(n)) {}

    // NOTE: upon move, the v of other is guaranteed
    // to become empty().
    pwrap(pwrap &&) noexcept = default;

    // Delete the rest.
    pwrap(const pwrap &) = delete;
    pwrap &operator=(const pwrap &) = delete;
    pwrap &operator=(pwrap &&) = delete;

    ~pwrap()
    {
        // NOTE: put back into cache only
        // if this was not moved-from.
        if (!v.empty()) {
            put_poly_in_cache(std::move(v));
        }
    }

    std::vector<T> v;
};

// Find the only existing root for the polynomial poly of the given order
// existing in [lb, ub].
template <typename T>
std::tuple<T, int> bracketed_root_find(const pwrap<T> &poly, std::uint32_t order, T lb, T ub)
{
    // NOTE: perhaps this should depend on T?
    constexpr boost::uintmax_t iter_limit = 100;
    boost::uintmax_t max_iter = iter_limit;

    // Ensure that root finding does not throw on error,
    // rather it will write something to errno instead.
    // https://www.boost.org/doc/libs/1_75_0/libs/math/doc/html/math_toolkit/pol_tutorial/namespace_policies.html
    using boost::math::policies::domain_error;
    using boost::math::policies::errno_on_error;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::pole_error;
    using boost::math::policies::policy;

    using pol = policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>,
                       evaluation_error<errno_on_error>>;

    // Clear out errno before running the root finding.
    errno = 0;

    // Prepare the return value.
    T ret;

#if defined(HEYOKA_HAVE_REAL128)
    if constexpr (std::is_same_v<T, mppp::real128>) {
        // NOTE: currently in order to use Boost's root finding
        // we need to also use their own float128 wrapper. In the long run we should
        // perhaps try to adapt mp++ so that this is not necessary
        // and we can use directly real128.
        using bf128 = boost::multiprecision::float128;

        // Transform the poly coefficients to bf128.
        pwrap<bf128> tmp(order);
        std::transform(poly.v.begin(), poly.v.end(), tmp.v.begin(), [](auto x) { return bf128(x.m_value); });

        auto p = boost::math::tools::toms748_solve(
            [d = std::as_const(tmp.v).data(), order](bf128 x) { return poly_eval(d, x, order); }, bf128(lb.m_value),
            bf128(ub.m_value), boost::math::tools::eps_tolerance<bf128>(), max_iter, pol{});

        ret = mppp::real128(((p.first + p.second) / 2).backend().value());
    } else {
#endif
        auto p = boost::math::tools::toms748_solve([d = poly.v.data(), order](T x) { return poly_eval(d, x, order); },
                                                   lb, ub, boost::math::tools::eps_tolerance<T>(), max_iter, pol{});

        ret = (p.first + p.second) / 2;
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif

    SPDLOG_LOGGER_DEBUG(get_logger(), "root finding iterations: {}", max_iter);

    if (errno > 0) {
        // Some error condition arose during root finding,
        // return zero and errno.
        return std::tuple{T(0), errno};
    }

    if (max_iter < iter_limit) {
        // Root finding terminated within the
        // iteration limit, return ret and success.
        return std::tuple{ret, 0};
    } else {
        // Root finding needed too many iterations,
        // return the (possibly wrong) result
        // and flag -1.
        return std::tuple{ret, -1};
    }
}

// Helper to fetch the per-thread working list used in
// poly root finding.
template <typename T>
auto &get_wlist()
{
    thread_local std::vector<std::tuple<T, T, pwrap<T>>> w_list;

    return w_list;
}

// Helper to fetch the per-thread list of isolating intervals.
template <typename T>
auto &get_isol()
{
    thread_local std::vector<std::tuple<T, T>> isol;

    return isol;
}

// Helper to detect events of terminal type.
template <typename>
struct is_terminal_event : std::false_type {
};

template <typename T>
struct is_terminal_event<t_event<T>> : std::true_type {
};

// Implementation of event detection.
template <typename T>
void taylor_detect_events_impl(std::vector<std::tuple<std::uint32_t, T, bool>> &d_tes,
                               std::vector<std::tuple<std::uint32_t, T>> &d_ntes, const std::vector<t_event<T>> &tes,
                               const std::vector<nt_event<T>> &ntes,
                               const std::vector<std::optional<std::pair<T, T>>> &cooldowns, T h,
                               const std::vector<T> &ev_jet, std::uint32_t order, std::uint32_t dim)
{
    using std::isfinite;

    if (!isfinite(h) || h == 0) {
        get_logger()->warn("event detection skipped due to an invalid timestep value of {}", h);

        return;
    }

    assert(order >= 2u);

    // Clear the vectors of detected events.
    d_tes.clear();
    d_ntes.clear();

    // Fetch a reference to the list of isolating intervals.
    auto &isol = get_isol<T>();

    // Fetch a reference to the wlist.
    auto &wl = get_wlist<T>();

    // Prepare the cache of binomial coefficients.
    const auto &bc = get_bc_up_to<T>(order);

    // Helper to run event detection on a vector of events
    // (terminal or not). 'out' is the vector of detected
    // events, 'ev_vec' the input vector of events to detect,
    // 'base_jet_idx' an index for reading the event polynomials
    // from ev_jet.
    auto run_detection = [&](auto &out, const auto &ev_vec) {
        // Check if we are doing detection of terminal events.
        using ev_type = typename uncvref_t<decltype(ev_vec)>::value_type;

        for (std::uint32_t i = 0; i < ev_vec.size(); ++i) {
            // Clear out the list of isolating intervals.
            isol.clear();

            // Reset the working list.
            wl.clear();

            // Extract the pointer to the Taylor polynomial for the
            // current event.
            const auto ptr
                = ev_jet.data() + (i + dim + (is_terminal_event<ev_type>::value ? 0u : tes.size())) * (order + 1u);

            // Helper to add a detected event to out.
            // NOTE: the root here is expected to be already rescaled
            // to the [0, h) range.
            auto add_d_event = [&](T root) {
                // NOTE: we do one last check on the root in order to
                // avoid non-finite event times. This guarantees that
                // sorting the events by time is safe.
                if (!isfinite(root)) {
                    get_logger()->warn("polynomial root finding produced a non-finite root of {} - skipping the event",
                                       root);
                    return;
                }

                // TODO multiroot in cooldown detection.
                [[maybe_unused]] const bool has_multi_roots = [&]() {
                    if constexpr (is_terminal_event<ev_type>::value) {
                        return false;
                    } else {
                        return false;
                    }
                }();

                // Fetch and cache the event direction.
                const auto dir = ev_vec[i].get_direction();

                if (dir == event_direction::any) {
                    // If the event direction does not
                    // matter, just add it.
                    if constexpr (is_terminal_event<ev_type>::value) {
                        out.emplace_back(i, root, has_multi_roots);
                    } else {
                        out.emplace_back(i, root);
                    }
                } else {
                    // Otherwise, we need to compute the derivative
                    // and record the event only if its direction
                    // matches the sign of the derivative.
                    const auto der = poly_eval_1(ptr, root, order);

                    if ((der >= 0 && dir == event_direction::positive)
                        || (der <= 0 && dir == event_direction::negative)) {
                        if constexpr (is_terminal_event<ev_type>::value) {
                            out.emplace_back(i, root, has_multi_roots);
                        } else {
                            out.emplace_back(i, root);
                        }
                    }
                }
            };

            // NOTE: if we are dealing with a terminal event on cooldown,
            // we will need to ignore roots within the cooldown period.
            // lb_offset is the value in the original [0, 1) range corresponding
            // to the end of the cooldown.
            const auto lb_offset = [&]() {
                if constexpr (is_terminal_event<ev_type>::value) {
                    if (cooldowns[i]) {
                        using std::abs;

                        // NOTE: need to distinguish between forward
                        // and backward integration.
                        if (h >= 0) {
                            return (cooldowns[i]->second - cooldowns[i]->first) / abs(h);
                        } else {
                            return (cooldowns[i]->second + cooldowns[i]->first) / abs(h);
                        }
                    }
                }

                // NOTE: we end up here if the event is not terminal
                // or not on cooldown.
                return T(0);
            }();

            if (lb_offset >= 1) {
                // NOTE: the whole integration range is in the cooldown range,
                // move to the next event.
                SPDLOG_LOGGER_DEBUG(
                    get_logger(),
                    "the integration timestep falls within the cooldown range for the terminal event {}, skipping", i);
                continue;
            }

            // Rescale it so that the range [0, h)
            // becomes [0, 1).
            pwrap<T> tmp(order);
            poly_rescale(tmp.v.data(), ptr, h, order);

            // Place the first element in the working list.
            wl.emplace_back(0, 1, std::move(tmp));

#if !defined(NDEBUG)
            auto max_wl_size = wl.size();
            auto max_isol_size = isol.size();
#endif

            // Flag to signal that the do-while loop below failed.
            bool loop_failed = false;

            do {
                // Fetch the current interval and polynomial from the working list.
                // NOTE: q(x) is the transformed polynomial whose roots in the x range [0, 1) we will
                // be looking for. lb and ub represent what 0 and 1 correspond to in the *original*
                // [0, 1) range.
                auto [lb, ub, q] = std::move(wl.back());
                wl.pop_back();

                // Check for an event at the lower bound, which occurs
                // if the constant term of the polynomial is zero. We also
                // check for finiteness of all the other coefficients, otherwise
                // we cannot really claim to have detected an event.
                // When we do proper root finding below, the
                // algorithm should be able to detect non-finite
                // polynomials.
                if (q.v[0] == T(0)
                    && std::all_of(q.v.data() + 1, q.v.data() + 1 + order, [](const auto &x) { return isfinite(x); })) {
                    // NOTE: we will have to skip the event if we are dealing
                    // with a terminal event on cooldown and the lower bound
                    // falls within the cooldown time.
                    bool skip_event = false;
                    if constexpr (is_terminal_event<ev_type>::value) {
                        if (lb < lb_offset) {
                            SPDLOG_LOGGER_DEBUG(get_logger(),
                                                "terminal event {} detected at the beginning of an isolating interval "
                                                "is subject to cooldown, ignoring",
                                                i);
                            skip_event = true;
                        }
                    }

                    if (!skip_event) {
                        // NOTE: the original range had been rescaled wrt to h.
                        // Thus, we need to rescale back when adding the detected
                        // event.
                        add_d_event(lb * h);
                    }
                }

                // Reverse it.
                pwrap<T> tmp1(order);
                std::copy(q.v.rbegin(), q.v.rend(), tmp1.v.data());

                // Translate it.
                pwrap<T> tmp2(order);
                poly_translate_1(tmp2.v.data(), tmp1.v.data(), order, bc);

                // Count the sign changes.
                const auto n_sc = count_sign_changes(tmp2.v.data(), order);

                if (n_sc == 1u) {
                    // Found isolating interval, add it to isol.
                    isol.emplace_back(lb, ub);
                } else if (n_sc > 1u) {
                    // No isolating interval found, bisect.

                    // First we transform q into 2**n * q(x/2) and store the result
                    // into tmp1.
                    poly_rescale_p2(tmp1.v.data(), q.v.data(), order);
                    // Then we take tmp1 and translate it to produce 2**n * q((x+1)/2).
                    poly_translate_1(tmp2.v.data(), tmp1.v.data(), order, bc);

                    // Finally we add tmp1 and tmp2 to the working list.
                    const auto mid = (lb + ub) / 2;
                    // NOTE: don't add the lower range if it falls
                    // entirely within the cooldown range.
                    if (lb_offset < mid) {
                        wl.emplace_back(lb, mid, std::move(tmp1));
                    } else {
                        SPDLOG_LOGGER_DEBUG(
                            get_logger(),
                            "ignoring lower interval in a bisection that would fall entirely in the cooldown period");
                    }
                    wl.emplace_back(mid, ub, std::move(tmp2));
                }

#if !defined(NDEBUG)
                max_wl_size = std::max(max_wl_size, wl.size());
                max_isol_size = std::max(max_isol_size, isol.size());
#endif

                // We want to put limits in order to avoid an endless loop when the algorithm fails.
                // The first check is on the working list size and it is based
                // on heuristic observation of the algorithm's behaviour in pathological
                // cases. The second check is that we cannot possibly find more isolating
                // intervals than the degree of the polynomial.
                if (wl.size() > 250u || isol.size() > order) {
                    get_logger()->warn(
                        "the polynomial root isolation algorithm failed during event detection: the working "
                        "list size is {} and the number of isolating intervals is {}",
                        wl.size(), isol.size());

                    loop_failed = true;

                    break;
                }

            } while (!wl.empty());

#if !defined(NDEBUG)
            SPDLOG_LOGGER_DEBUG(get_logger(), "max working list size: {}", max_wl_size);
            SPDLOG_LOGGER_DEBUG(get_logger(), "max isol list size   : {}", max_isol_size);
#endif

            if (isol.empty() || loop_failed) {
                // Don't do root finding for this event if the loop failed,
                // or if the list of isolating intervals is empty. Just
                // move to the next event.
                continue;
            }

            // Reconstruct a version of the original event polynomial
            // in which the range [0, h) is rescaled to [0, 1). We need
            // to do root finding on the rescaled polynomial because the
            // isolating intervals are also rescaled to [0, 1).
            pwrap<T> tmp1(order);
            poly_rescale(tmp1.v.data(), ptr, h, order);

            // Run the root finding in the isolating intervals.
            for (auto &[lb, ub] : isol) {
                if constexpr (is_terminal_event<ev_type>::value) {
                    // NOTE: if we are dealing with a terminal event
                    // subject to cooldown, we need to ensure that
                    // we don't look for roots before the cooldown has expired.
                    if (lb < lb_offset) {
                        // Make sure we move lb past the cooldown.
                        lb = lb_offset;

                        // NOTE: this should be ensured by the fact that
                        // we ensure above (lb_offset < mid) that we don't
                        // end up with an invalid interval.
                        assert(lb < ub);

                        // Check if the interval still contains a zero.
                        const auto f_lb = poly_eval(tmp1.v.data(), lb, order);
                        const auto f_ub = poly_eval(tmp1.v.data(), ub, order);

                        if (!(f_lb * f_ub < 0)) {
                            SPDLOG_LOGGER_DEBUG(get_logger(), "terminal event {} is subject to cooldown, ignoring", i);
                            continue;
                        }
                    }
                }

                // Run the root finding.
                const auto [root, cflag] = bracketed_root_find(tmp1, order, lb, ub);

                if (cflag == 0) {
                    // Root finding finished successfully, record the event.
                    // The found root needs to be rescaled by h.
                    add_d_event(root * h);
                } else {
                    // Root finding encountered some issue. Ignore the
                    // event and log the issue.
                    if (cflag == -1) {
                        get_logger()->warn(
                            "polynomial root finding during event detection failed due to too many iterations");
                    } else {
                        get_logger()->warn(
                            "polynomial root finding during event detection returned a nonzero errno with message '{}'",
                            std::strerror(cflag));
                    }
                }
            }
        }
    };

    run_detection(d_tes, tes);
    run_detection(d_ntes, ntes);
}

} // namespace

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, double, bool>> &d_tes,
                          std::vector<std::tuple<std::uint32_t, double>> &d_ntes,
                          const std::vector<t_event<double>> &tes, const std::vector<nt_event<double>> &ntes,
                          const std::vector<std::optional<std::pair<double, double>>> &cooldowns, double h,
                          const std::vector<double> &ev_jet, std::uint32_t order, std::uint32_t dim)
{
    taylor_detect_events_impl(d_tes, d_ntes, tes, ntes, cooldowns, h, ev_jet, order, dim);
}

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, long double, bool>> &d_tes,
                          std::vector<std::tuple<std::uint32_t, long double>> &d_ntes,
                          const std::vector<t_event<long double>> &tes, const std::vector<nt_event<long double>> &ntes,
                          const std::vector<std::optional<std::pair<long double, long double>>> &cooldowns,
                          long double h, const std::vector<long double> &ev_jet, std::uint32_t order, std::uint32_t dim)
{
    taylor_detect_events_impl(d_tes, d_ntes, tes, ntes, cooldowns, h, ev_jet, order, dim);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, mppp::real128, bool>> &d_tes,
                          std::vector<std::tuple<std::uint32_t, mppp::real128>> &d_ntes,
                          const std::vector<t_event<mppp::real128>> &tes,
                          const std::vector<nt_event<mppp::real128>> &ntes,
                          const std::vector<std::optional<std::pair<mppp::real128, mppp::real128>>> &cooldowns,
                          mppp::real128 h, const std::vector<mppp::real128> &ev_jet, std::uint32_t order,
                          std::uint32_t dim)
{
    taylor_detect_events_impl(d_tes, d_ntes, tes, ntes, cooldowns, h, ev_jet, order, dim);
}

#endif

} // namespace heyoka::detail
