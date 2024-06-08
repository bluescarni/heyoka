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
#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/math/special_functions/factorials.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/i_data.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/var_ode_sys.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
tm_data<T>::tm_data() = default;

namespace
{

// Factorial implementations.
template <typename F>
F factorial(std::uint32_t n, long long)
{
    return boost::math::factorial<F>(boost::numeric_cast<unsigned>(n));
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
mppp::real128 factorial<mppp::real128>(std::uint32_t n, long long)
{
    return mppp::tgamma(mppp::real128(n) + 1);
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
mppp::real factorial<mppp::real>(std::uint32_t n, long long prec)
{
    mppp::real ret{mppp::real_kind::zero, boost::numeric_cast<mpfr_prec_t>(prec)};
    ::mpfr_fac_ui(ret._get_mpfr_t(), boost::numeric_cast<unsigned long>(n), MPFR_RNDN);

    return ret;
}

#endif

// Non-compact mode implementation of the function for the evaluation
// of the Taylor map.
template <typename T>
void add_tm_func_nc_mode(llvm_state &st, const std::vector<T> &state, const var_ode_sys &sys, const dtens &dt,
                         std::uint32_t batch_size)
{
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;

    auto &builder = st.builder();
    auto &context = st.context();
    auto &md = st.module();

    // Fetch the internal and external floating-point types.
    assert(!state.empty());
    // NOTE: 'state' has been set up with the correct precision
    // in the ctor.
    auto *fp_t = detail::llvm_type_like(st, state[0]);
    auto *ext_fp_t = detail::to_llvm_type<T>(context);
    auto *ext_ptr_t = llvm::PointerType::getUnqual(ext_fp_t);

    // Cache the precision.
    const auto prec = [&]() -> long long {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::same_as<T, mppp::real>) {
            return static_cast<long long>(state[0].get_prec());
        } else {
#endif
            return 0;
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    }();

    // The function arguments:
    // - the output pointer (write-only),
    // - the input pointer (read-only),
    // - the pointer to the state vector (read-only).
    // All pointers are external. There might be overlap between input,
    // output and state vector pointers.
    std::vector<llvm::Type *> fargs(3u, ext_ptr_t);

    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE

    // Now create the function.
    auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, "tm_func", &md);
    // NOTE: a Taylor map eval function cannot call itself.
    f->addFnAttr(llvm::Attribute::NoRecurse);

    // Set the names/attributes of the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *in_ptr = out_ptr + 1;
    in_ptr->setName("in_ptr");
    in_ptr->addAttr(llvm::Attribute::NoCapture);
    in_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *state_ptr = out_ptr + 2;
    state_ptr->setName("state_ptr");
    state_ptr->addAttr(llvm::Attribute::NoCapture);
    state_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Cache the number of variational arguments. This is the number of input arguments.
    using vargs_size_t = decltype(sys.get_vargs().size());
    const auto nvargs = sys.get_vargs().size();

    // Cache the diff order.
    const auto order = dt.get_order();
    assert(order > 0u);

    // Cache the number of state variables.
    const auto n_orig_sv = sys.get_n_orig_sv();
    assert(n_orig_sv > 0u);

    // NOTE: there is a lot of literature about efficient multivariate polynomial evaluation.
    // Here we are adopting the simple approach of proceeding order by order and iteratively
    // building up the powers of the input variables by repeated multiplications. If necessary
    // we can look into more optimised (and more complicated) ways of doing this, such as
    // multivariate Horner schemes.

    // Create the table of powers of the input values and init
    // it with the order-1 powers (i.e., the original input values).
    // NOTE: this will end up being a table with nvargs rows and
    // order columns. Each row will contains the natural powers
    // of an input value from 1 up to order.
    std::vector<std::vector<llvm::Value *>> in_pows;
    in_pows.resize(boost::numeric_cast<decltype(in_pows.size())>(nvargs));
    for (vargs_size_t i = 0; i < nvargs; ++i) {
        // NOTE: we do not store the order-0 powers explicitly (as they are just 1s).
        // Thus, we reserve only order and not order + 1.
        in_pows[i].reserve(order);

        // Compute the pointer to load the inputs from.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, in_ptr, builder.getInt32(su32_t(i) * batch_size));

        // Load the value.
        in_pows[i].push_back(ext_load_vector_from_memory(st, fp_t, ptr, batch_size));
    }

    // Create the table of Taylor series terms for the state variables.
    // This will end up being a table with n_orig_sv rows and order + 1
    // columns. Each row will contain, order by order, the terms of the Taylor
    // series for each state variable.
    std::vector<std::vector<llvm::Value *>> t_terms;
    t_terms.resize(boost::numeric_cast<decltype(t_terms.size())>(n_orig_sv));
    // Fill-in the order-0 terms (that is, the current values of the state variables).
    for (std::uint32_t i = 0; i < n_orig_sv; ++i) {
        // NOTE: overflow checking was done in the ctor.
        t_terms[i].reserve(order + 1u);

        // Compute the pointer to load from.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, state_ptr, builder.getInt32(su32_t(i) * batch_size));

        // Load the value.
        t_terms[i].push_back(ext_load_vector_from_memory(st, fp_t, ptr, batch_size));
    }

    // Build up the Taylor series terms order by order.
    for (std::uint32_t cur_order = 1; cur_order <= order; ++cur_order) {
        // For orders higher than 1 we need to update the table of powers.
        if (cur_order > 1u) {
            for (auto &cur_in_pow : in_pows) {
                auto *next_pow = llvm_fmul(st, cur_in_pow[0], cur_in_pow.back());
                cur_in_pow.push_back(next_pow);
            }
        }

        // Iterate over the state variables.
        for (std::uint32_t sv_idx = 0; sv_idx < n_orig_sv; ++sv_idx) {
            // Fetch the subrange of multiindices for the current order and state variable.
            const auto mrng = dt.get_derivatives(sv_idx, cur_order);
            assert(!mrng.empty());

            // Iterate over the subrange and compute the Taylor series terms
            // for the current state variable and order. The terms will be stored
            // in 'terms'.
            std::vector<llvm::Value *> terms;
            terms.reserve(std::ranges::size(mrng));
            for (auto m_it = mrng.begin(); m_it != mrng.end(); ++m_it) {
                const auto &[key, ex] = *m_it;

                const auto &[comp, sv] = key;
                assert(comp == sv_idx);
                assert(!sv.empty());

                // Iterate over sv to compute the divisor and the product
                // of input powers.
                auto div = factorial<T>(sv[0].second, prec);
                assert(sv[0].first < in_pows.size());
                assert(sv[0].second - 1u < in_pows[sv[0].first].size());
                std::vector<llvm::Value *> prod_terms{in_pows[sv[0].first][sv[0].second - 1u]};

                for (auto it = sv.begin() + 1; it != sv.end(); ++it) {
                    div *= factorial<T>(it->second, prec);
                    assert(it->first < in_pows.size());
                    assert(it->second - 1u < in_pows[it->first].size());
                    prod_terms.push_back(in_pows[it->first][it->second - 1u]);
                }

#if defined(HEYOKA_HAVE_REAL)

                if constexpr (std::same_as<T, mppp::real>) {
                    assert(div.get_prec() == prec);
                }

#endif

                // Compute the current term of the Taylor series and add it to terms.
                auto *prod = pairwise_prod(st, prod_terms);

                // Multiply by the value of the derivative.
                const auto didx = dt.index_of(m_it);
                auto *dptr
                    = builder.CreateInBoundsGEP(ext_fp_t, state_ptr, builder.getInt32(su32_t(didx) * batch_size));
                auto *dval = ext_load_vector_from_memory(st, fp_t, dptr, batch_size);
                prod = llvm_fmul(st, prod, dval);

                // NOTE: avoid doing the division if not necessary.
                if (div == 1) {
                    terms.push_back(prod);
                } else {
                    auto *div_splat = vector_splat(builder, llvm_codegen(st, fp_t, number{std::move(div)}), batch_size);
                    terms.push_back(llvm_fdiv(st, prod, div_splat));
                }
            }

            // Sum the components in terms and add the result to t_terms.
            assert(sv_idx < t_terms.size());
            t_terms[sv_idx].push_back(pairwise_sum(st, terms));
        }
    }

    // Compute and write out the result.
    for (std::uint32_t i = 0; i < n_orig_sv; ++i) {
        // Compute the pointer to store the result to.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32(su32_t(i) * batch_size));

        // Sum the Taylor series terms.
        auto *ret = pairwise_sum(st, t_terms[i]);

        // Store it.
        ext_store_vector_to_memory(st, ptr, ret);
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    st.verify_function(f);
}

} // namespace

template <typename T>
tm_data<T>::tm_data(const var_ode_sys &sys, [[maybe_unused]] long long prec, const llvm_state &orig_s,
                    std::uint32_t batch_size)
{
    // Fetch the dtens from sys.
    const auto &dt = sys.get_dtens();

    // Overflow check.
    // NOTE: as usual with polynomials, we need to be able to compute order + 1
    // without overflowing.
    // LCOV_EXCL_START
    if (dt.get_order() == std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        throw std::overflow_error(
            "An overflow condition was detected while setting up the data for the evaluation of a Taylor map");
    }
    // LCOV_EXCL_STOP

    // Prepare m_output.
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::same_as<mppp::real, T>) {
        m_output.resize(boost::safe_numerics::safe<decltype(m_output.size())>(sys.get_n_orig_sv()) * batch_size,
                        // NOTE: static cast is fine here, the precision value has
                        // already been validated.
                        mppp::real{mppp::real_kind::zero, static_cast<mpfr_prec_t>(prec)});
    } else {
#endif
        m_output.resize(boost::safe_numerics::safe<decltype(m_output.size())>(sys.get_n_orig_sv()) * batch_size,
                        static_cast<T>(0));
#if defined(HEYOKA_HAVE_REAL)
    }
#endif

    // Create a new llvm state similar to orig_s.
    auto st = orig_s.make_similar();

    // Add the Taylor map function.
    add_tm_func_nc_mode(st, m_output, sys, dt, batch_size);

    // Compile.
    st.compile();

    // Fetch the function.
    m_tm_func = reinterpret_cast<tm_func_t>(st.jit_lookup("tm_func"));

    // Move in the state.
    m_state = std::move(st);
}

template <typename T>
tm_data<T>::tm_data(const tm_data &other) : m_state(other.m_state), m_output(other.m_output)
{
    m_tm_func = reinterpret_cast<tm_func_t>(m_state.jit_lookup("tm_func"));
}

template <typename T>
tm_data<T>::tm_data(tm_data &&) noexcept = default;

template <typename T>
tm_data<T> &tm_data<T>::operator=(tm_data &&) noexcept = default;

template <typename T>
tm_data<T>::~tm_data() = default;

template <typename T>
void tm_data<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_state;
    ar << m_output;
}

template <typename T>
void tm_data<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_state;
    ar >> m_output;

    m_tm_func = reinterpret_cast<tm_func_t>(m_state.jit_lookup("tm_func"));
}

// Explicit instantiations.
template struct tm_data<float>;
template struct tm_data<double>;
template struct tm_data<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template struct tm_data<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template struct tm_data<mppp::real>;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
