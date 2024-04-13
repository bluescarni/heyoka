// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TAYLOR_COMMON_HPP
#define HEYOKA_DETAIL_TAYLOR_COMMON_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/i_data.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Implementation detail for taylor_c_diff_func_numpar(). np_tup is a tuple containing a sequence of numbers and/or
// params, while numpar_begin points to the first number/param argument in the signature of a function for the
// computation of Taylor derivatives in compact mode. This function will generate a set of llvm values from the function
// arguments starting at numpar_begin, either splatting the number argument or loading the parameter from the parameter
// array (see the implementation of taylor_c_diff_numparam_codegen()).
template <typename Tup, typename ArgIter, std::size_t... I>
inline std::vector<llvm::Value *>
taylor_c_diff_func_numpar_codegen_impl(llvm_state &s, llvm::Type *fp_t, const Tup &np_tup, ArgIter numpar_begin,
                                       llvm::Value *par_ptr, std::uint32_t batch_size, std::index_sequence<I...>)
{
    return {taylor_c_diff_numparam_codegen(s, fp_t, std::get<I>(np_tup), numpar_begin + I, par_ptr, batch_size)...};
}

// Helper to implement the function for the differentiation of
// a function of number(s)/param(s) in compact mode. The function will always return zero,
// unless the order is 0 (in which case it will return the result of applying the functor cgen
// to the number(s)/param(s) arguments).
template <typename F, typename... NumPars>
inline llvm::Function *taylor_c_diff_func_numpar(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                 std::uint32_t batch_size, const std::string &name,
                                                 std::uint32_t n_hidden_deps, const F &cgen, const NumPars &...np)
{
    static_assert(sizeof...(np) > 0u);
    static_assert(std::conjunction_v<std::disjunction<std::is_same<NumPars, number>, std::is_same<NumPars, param>>...>);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, name, n_uvars, batch_size, {np...}, n_hidden_deps);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto par_ptr = f->args().begin() + 3;
        auto numpar_begin = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // Generate the vector of num/param arguments.
                auto np_args = taylor_c_diff_func_numpar_codegen_impl(s, fp_t, std::make_tuple(std::cref(np)...),
                                                                      numpar_begin, par_ptr, batch_size,
                                                                      std::make_index_sequence<sizeof...(np)>{});

                // Run the codegen and store the result.
                builder.CreateStore(cgen(np_args), retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(llvm_constantfp(s, val_t, 0.), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Helper to determine the optimal Taylor order for a given tolerance,
// following Jorba's prescription.
// NOTE: when T is mppp::real and tol has a low precision, the use
// of integer operands in these computations might bump up the working
// precision due to the way precision propagation works in mp++. I don't
// think there's any negative consequence here.
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
    order_f = std::max(static_cast<T>(2), order_f);

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

// Small helper to set up the m_state_vars and m_rhs members
// of an integrator. Shared among scalar and batch integrators.
template <typename TA, typename U>
void taylor_adaptive_setup_sv_rhs(TA &ta, const U &sys)
{
    for (std::uint32_t i = 0; i < ta.m_i_data->m_dim; ++i) {
        // NOTE: take the state variables from the
        // decomposition.
        assert(std::holds_alternative<variable>(ta.m_i_data->m_dc[i].first.value()));
        ta.m_i_data->m_state_vars.push_back(ta.m_i_data->m_dc[i].first);
        ta.m_i_data->m_rhs.push_back(sys[i].second);
    }
}

// Small helper to compare the absolute
// values of two input values.
template <typename T>
bool abs_lt(const T &a, const T &b)
{
    using std::abs;

    return abs(a) < abs(b);
}

#if defined(HEYOKA_HAVE_REAL)

template <>
inline bool abs_lt<mppp::real>(const mppp::real &a, const mppp::real &b)
{
    return mppp::cmpabs(a, b) < 0;
}

#endif

// Custom equality comparison that considers all NaN values equal.
template <typename T>
bool cmp_nan_eq(const T &a, const T &b)
{
    using std::isnan;

    const auto nan_a = isnan(a);
    const auto nan_b = isnan(b);

    if (!nan_a && !nan_b) {
        return a == b;
    } else {
        return nan_a && nan_b;
    }
}

// NOTE: double-length normalisation assumes abs(hi) >= abs(lo), we need to enforce this
// when setting the time coordinate in double-length format.
template <typename T>
void dtime_checks(const T &hi, const T &lo)
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

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
