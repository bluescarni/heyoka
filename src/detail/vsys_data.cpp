// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/math/special_functions/factorials.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/vsys_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/var_ode_sys.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
vsys_data<T>::vsys_data() = default;

namespace
{

// LCOV_EXCL_START

// Factorial implementations.
template <typename F>
F factorial([[maybe_unused]] std::uint32_t n, long long)
{
#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::same_as<F, long double>) {
        throw std::invalid_argument("'long double' computations are not supported on this platform");
    } else {
#endif
        return boost::math::factorial<F>(boost::numeric_cast<unsigned>(n));
#if defined(HEYOKA_ARCH_PPC)
    }
#endif
}

// LCOV_EXCL_STOP

#if defined(HEYOKA_HAVE_REAL128)

template <>
mppp::real128 factorial<mppp::real128>(std::uint32_t n, long long)
{
    return mppp::tgamma(mppp::real128(n) + 1);
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
mppp::real factorial<mppp::real>(std::uint32_t n, long long prec)
{
    mppp::real ret{mppp::real_kind::zero, boost::numeric_cast<mpfr_prec_t>(prec)};
    ::mpfr_fac_ui(ret._get_mpfr_t(), boost::numeric_cast<unsigned long>(n), MPFR_RNDN);

    return ret;
}

#endif

// Function to formulate the expressions for the evaluation of the Taylor map in a variational integrator.
//
// The first element of the return value are the expressions evaluating the Taylor map for each original state variable.
// The second element of the return value are the inputs, that is, the variations in initial conditions / parameters /
// initial time. The evaluation of the Taylor map requires also the values of the derivatives of the original state
// variables, which are passed in via the array of parameters. The indexing into the array of parameters mirrors the
// indexing into the 'dt' object - that is, the first par[0], par[1], ..., par[n_orig_sv - 1] parameters contain the
// values of the state variables, followed by the order-1 derivatives, and so on.
//
// NOTE: this needs to be templated on T due to the necessity of computing the compile-time factorial. Perhaps at one
// point we can have a dedicated constant-like factorial func that would allow us to make this a non-template function.
template <typename T>
std::pair<std::vector<expression>, std::vector<expression>>
vsys_data_create_tm_expr(const var_ode_sys &sys, const dtens &dt, const long long prec)
{
    // Cache the number of variational arguments. This is the number of input arguments.
    using vargs_size_t = decltype(sys.get_vargs().size());
    const auto nvargs = sys.get_vargs().size();

    // Cache the diff order.
    const auto order = dt.get_order();
    assert(order > 0u); // LCOV_EXCL_LINE

    // Cache the number of state variables.
    const auto n_orig_sv = sys.get_n_orig_sv();
    assert(n_orig_sv > 0u); // LCOV_EXCL_LINE

    // NOTE: there is a lot of literature about efficient multivariate polynomial evaluation. Here we are adopting the
    // simple approach of proceeding order by order and iteratively building up the powers of the input variables by
    // repeated multiplications. If necessary we can look into more optimised (and more complicated) ways of doing this,
    // such as multivariate Horner schemes.

    // Create the table of powers of the input values and init it with the order-1 powers (i.e., the original input
    // values).
    //
    // NOTE: this will end up being a table with nvargs rows and order columns. Each row will contains the
    // natural powers of an input value from 1 up to order.
    std::vector<std::vector<expression>> in_pows;
    in_pows.resize(boost::numeric_cast<decltype(in_pows.size())>(nvargs));
    // NOTE: create the vector of inputs at the same time.
    std::vector<expression> inputs;
    inputs.reserve(nvargs);
    for (vargs_size_t i = 0; i < nvargs; ++i) {
        // NOTE: we do not store the order-0 powers explicitly (as they are just 1s). Thus, we reserve only order and
        // not order + 1.
        in_pows[i].reserve(order);

        // Add the order-1 power of the input value.
        in_pows[i].emplace_back(fmt::format("delta_{}", i));

        // Store the input value.
        inputs.push_back(in_pows[i].back());
    }

    // Create the table of Taylor series terms for the state variables. This will end up being a table with n_orig_sv
    // rows and order + 1 columns. Each row will contain, order by order, the terms of the Taylor series for each state
    // variable.
    std::vector<std::vector<expression>> t_terms;
    t_terms.resize(boost::numeric_cast<decltype(t_terms.size())>(n_orig_sv));
    // Fill-in the order-0 terms (that is, the current values of the state variables).
    for (std::uint32_t i = 0; i < n_orig_sv; ++i) {
        // NOTE: overflow checking was done in the ctor.
        t_terms[i].reserve(order + 1u);

        // NOTE: the first n_orig_sv params are the current values of the state variables.
        t_terms[i].push_back(par[i]);
    }

    // Build up the Taylor series terms order by order.
    for (std::uint32_t cur_order = 1; cur_order <= order; ++cur_order) {
        // For orders higher than 1 we need to update the table of powers.
        if (cur_order > 1u) {
            for (auto &cur_in_pow : in_pows) {
                cur_in_pow.push_back(cur_in_pow[0] * cur_in_pow.back());
            }
        }

        // Iterate over the state variables.
        for (std::uint32_t sv_idx = 0; sv_idx < n_orig_sv; ++sv_idx) {
            // Fetch the subrange of multiindices for the current order and state variable.
            const auto mrng = dt.get_derivatives(sv_idx, cur_order);
            assert(!mrng.empty()); // LCOV_EXCL_LINE

            // Iterate over the subrange and compute the Taylor series terms for the current state variable and order.
            // The terms will be stored in 'terms'.
            std::vector<expression> terms;
            terms.reserve(std::ranges::size(mrng));
            for (auto m_it = mrng.begin(); m_it != mrng.end(); ++m_it) {
                const auto &[key, ex] = *m_it;

                const auto &[comp, sv] = key;
                assert(comp == sv_idx); // LCOV_EXCL_LINE
                assert(!sv.empty());    // LCOV_EXCL_LINE

                // Iterate over sv to compute the divisor and the product of input powers.
                auto div = factorial<T>(sv[0].second, prec);
                assert(sv[0].first < in_pows.size());                    // LCOV_EXCL_LINE
                assert(sv[0].second - 1u < in_pows[sv[0].first].size()); // LCOV_EXCL_LINE
                std::vector<expression> prod_terms{in_pows[sv[0].first][sv[0].second - 1u]};

                for (auto it = sv.begin() + 1; it != sv.end(); ++it) {
                    div *= factorial<T>(it->second, prec);
                    assert(it->first < in_pows.size());                  // LCOV_EXCL_LINE
                    assert(it->second - 1u < in_pows[it->first].size()); // LCOV_EXCL_LINE
                    prod_terms.push_back(in_pows[it->first][it->second - 1u]);
                }

#if defined(HEYOKA_HAVE_REAL)

                if constexpr (std::same_as<T, mppp::real>) {
                    assert(div.get_prec() == prec);
                }

#endif

                // Compute the current term of the Taylor series and add it to terms.
                auto prod = heyoka::prod(std::move(prod_terms));

                // Multiply by the value of the derivative.
                //
                // NOTE: the indexing into the parameters array mirrors the indexing of derivatives in dt.
                const auto didx = dt.index_of(m_it);
                prod *= par[boost::numeric_cast<std::uint32_t>(didx)];

                // NOTE: avoid doing the division if not necessary.
                if (div == 1) {
                    terms.push_back(prod);
                } else {
                    terms.push_back(prod / div);
                }
            }

            // Sum the components in 'terms' and add the result to t_terms.
            assert(sv_idx < t_terms.size()); // LCOV_EXCL_LINE
            t_terms[sv_idx].push_back(sum(std::move(terms)));
        }
    }

    // Compute and write out the outputs.
    std::vector<expression> outs;
    outs.reserve(n_orig_sv);
    for (std::uint32_t i = 0; i < n_orig_sv; ++i) {
        // Sum the Taylor series terms and append the result to outs.
        outs.push_back(sum(std::move(t_terms[i])));
    }

    return std::make_pair(std::move(outs), std::move(inputs));
}

} // namespace

template <typename T>
vsys_data<T>::vsys_data(const var_ode_sys &sys, const long long prec, const llvm_state &tplt,
                        const std::uint32_t batch_size, const bool high_accuracy, const bool compact_mode,
                        const bool parjit)
{
    // Fetch the dtens from sys.
    const auto &dt = sys.get_dtens();

    // Overflow check.
    //
    // NOTE: as usual with polynomials, we need to be able to compute order + 1 without overflowing.
    //
    // LCOV_EXCL_START
    if (dt.get_order() == std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        throw std::overflow_error(
            "An overflow condition was detected while setting up the data for the evaluation of a Taylor map");
    }
    // LCOV_EXCL_STOP

    // Prepare m_tm_output.
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::same_as<mppp::real, T>) {
        m_tm_output.resize(boost::safe_numerics::safe<decltype(m_tm_output.size())>(sys.get_n_orig_sv()) * batch_size,
                           // NOTE: static cast is fine here, the precision value has
                           // already been validated.
                           mppp::real{mppp::real_kind::zero, static_cast<mpfr_prec_t>(prec)});
    } else {
#endif
        m_tm_output.resize(boost::safe_numerics::safe<decltype(m_tm_output.size())>(sys.get_n_orig_sv()) * batch_size,
                           static_cast<T>(0));
#if defined(HEYOKA_HAVE_REAL)
    }
#endif

    // Create the Taylor map expressions.
    auto [tm_outs, tm_ins] = vsys_data_create_tm_expr<T>(sys, dt, prec);

    // Create the Taylor map cfunc.
    //
    // NOTE: we are ignoring the parallel_mode kwarg even if in principle we should forward it from the parallel_mode
    // setting of the Taylor integrator. The reason is that we are not implementing parallel_mode yet in cfunc, and we
    // do not want the cfunc constructor to error out if parallel_mode is passed down from the integrator.
    //
    // NOTE: we set the cfunc kwarg check_prec to false because we are checking the precision of the input
    // multiprecision arguments in the Taylor integrators code.
    m_tm_cfunc = cfunc<T>{std::move(tm_outs), std::move(tm_ins),
                          // cfunc-specific kwargs.
                          kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode, kw::prec = prec,
                          kw::check_prec = false, kw::parjit = parjit,
                          // llvm_state-specific kwargs.
                          kw::opt_level = tplt.get_opt_level(), kw::fast_math = tplt.fast_math(),
                          kw::force_avx512 = tplt.force_avx512(), kw::slp_vectorize = tplt.get_slp_vectorize(),
                          kw::code_model = tplt.get_code_model()};
}

template <typename T>
vsys_data<T>::vsys_data(const vsys_data &) = default;

template <typename T>
vsys_data<T>::~vsys_data() = default;

template <typename T>
void vsys_data<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_tm_cfunc;
    ar << m_tm_output;
}

template <typename T>
void vsys_data<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_tm_cfunc;
    ar >> m_tm_output;
}

// Explicit instantiations.
template struct vsys_data<float>;
template struct vsys_data<double>;
template struct vsys_data<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template struct vsys_data<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template struct vsys_data<mppp::real>;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
