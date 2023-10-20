// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

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

#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

kepF_impl::kepF_impl() : kepF_impl(0_dbl, 0_dbl, 0_dbl) {}

kepF_impl::kepF_impl(expression h, expression k, expression lam)
    : func_base("kepF", std::vector{std::move(h), std::move(k), std::move(lam)})
{
}

template <typename Archive>
void kepF_impl::serialize(Archive &ar, unsigned)
{
    ar &boost::serialization::base_object<func_base>(*this);
}

template <typename T>
expression kepF_impl::diff_impl(funcptr_map<expression> &func_map, const T &s) const
{
    assert(args().size() == 3u);

    const auto &h = args()[0];
    const auto &k = args()[1];
    const auto &lam = args()[2];

    const expression F{func{*this}};

    return (detail::diff(func_map, k, s) * sin(F) - detail::diff(func_map, h, s) * cos(F)
            + detail::diff(func_map, lam, s))
           / (1_dbl - h * sin(F) - k * cos(F));
}

expression kepF_impl::diff(funcptr_map<expression> &func_map, const std::string &s) const
{
    return diff_impl(func_map, s);
}

expression kepF_impl::diff(funcptr_map<expression> &func_map, const param &p) const
{
    return diff_impl(func_map, p);
}

namespace
{

llvm::Value *kepF_llvm_eval_impl(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                 const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr, llvm::Value *stride,
                                 std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_eval_helper(
        [&s, fp_t, batch_size](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepF_func = llvm_add_inv_kep_F(s, fp_t, batch_size);

            return s.builder().CreateCall(kepF_func, {args[0], args[1], args[2]});
        },
        fb, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

} // namespace

llvm::Value *kepF_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                  llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                  bool high_accuracy) const
{
    return kepF_llvm_eval_impl(s, fp_t, *this, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *kepF_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                               std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "kepF",
        [&s, batch_size, fp_t](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepF_func = llvm_add_inv_kep_F(s, fp_t, batch_size);

            return s.builder().CreateCall(kepF_func, {args[0], args[1], args[2]});
        },
        fb, s, fp_t, batch_size, high_accuracy);
}

} // namespace

llvm::Function *kepF_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    return kepF_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

taylor_dc_t::size_type kepF_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 3u);

    // Make a copy of h and k, since we will be soon moving this.
    // NOTE: the arguments here have already been decomposed, thus
    // args()[0-1] are non-function values.
    assert(!std::holds_alternative<func>(args()[0].value()));
    assert(!std::holds_alternative<func>(args()[1].value()));
    auto h_copy = args()[0];
    auto k_copy = args()[1];

    // Append the kepF decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Append the sin(a)/cos(a) decompositions.
    u_vars_defs.emplace_back(sin(expression{variable{fmt::format("u_{}", u_vars_defs.size() - 1u)}}),
                             std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(cos(expression{variable{fmt::format("u_{}", u_vars_defs.size() - 2u)}}),
                             std::vector<std::uint32_t>{});

    // Append the h*sin(a) and k*cos(a) decompositions.
    u_vars_defs.emplace_back(h_copy * expression{variable{fmt::format("u_{}", u_vars_defs.size() - 2u)}},
                             std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(k_copy * expression{variable{fmt::format("u_{}", u_vars_defs.size() - 2u)}},
                             std::vector<std::uint32_t>{});

    // Add the hidden deps.
    // NOTE: the hidden deps are added in this order:
    // - c = h*sin(F),
    // - d = k*cos(F),
    // - e = sin(F),
    // - f = cos(F).
    // The order in which the hidden deps are added here must match the order in which the hidden deps are consumed
    // in the Taylor diff implementations.
    (u_vars_defs.end() - 5)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));
    (u_vars_defs.end() - 5)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));
    (u_vars_defs.end() - 5)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 4u));
    (u_vars_defs.end() - 5)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 3u));

    // sin/cos hidden deps.
    (u_vars_defs.end() - 4)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 3u));
    (u_vars_defs.end() - 3)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 4u));

    return u_vars_defs.size() - 5u;
}

namespace
{

// Derivative of kepF(number, number, number).
template <typename U, typename V, typename W,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>, is_num_param<W>>, int> = 0>
llvm::Value *taylor_diff_kepF_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &, const U &num0,
                                   const V &num1, const W &num2, const std::vector<llvm::Value *> &,
                                   llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                   std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (order == 0u) {
        // Do the number codegen.
        auto *h = taylor_codegen_numparam(s, fp_t, num0, par_ptr, batch_size);
        auto *k = taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size);
        auto *lam = taylor_codegen_numparam(s, fp_t, num2, par_ptr, batch_size);

        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_F(s, fp_t, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {h, k, lam});
    } else {
        return vector_splat(builder, llvm_constantfp(s, fp_t, 0.), batch_size);
    }
}

// Derivative of kepF(number, number, var).
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *
taylor_diff_kepF_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps, const U &num0,
                      const V &num1, const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                      std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(deps.size() == 4u); // LCOV_EXCL_LINE

    auto &builder = s.builder();

    // Fetch the index of the lam variable argument.
    const auto lam_idx = uname_to_index(var.name());

    // Do the codegen for the number arguments.
    auto *h = taylor_codegen_numparam(s, fp_t, num0, par_ptr, batch_size);
    auto *k = taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size);

    if (order == 0u) {
        // Create/fetch the Kepler solver.
        auto *fkep = llvm_add_inv_kep_F(s, fp_t, batch_size);

        // Invoke and return.
        return builder.CreateCall(fkep, {h, k, taylor_fetch_diff(arr, lam_idx, 0, n_uvars)});
    }

    // Splat the order.
    auto *n = vector_splat(builder, llvm_constantfp(s, fp_t, static_cast<double>(order)), batch_size);

    // Compute the divisor: n * (1 - c^[0] - d^[0]).
    const auto c_idx = deps[0], d_idx = deps[1];
    auto *one_fp = vector_splat(builder, llvm_constantfp(s, fp_t, 1.), batch_size);
    auto *divisor = llvm_fsub(s, one_fp, taylor_fetch_diff(arr, c_idx, 0, n_uvars));
    divisor = llvm_fsub(s, divisor, taylor_fetch_diff(arr, d_idx, 0, n_uvars));
    divisor = llvm_fmul(s, n, divisor);

    // Compute the first part of the dividend: n * lam^[n] (the derivatives of h and k are zero because
    // here h and k are constants and the order is > 0).
    auto *dividend = llvm_fmul(s, n, taylor_fetch_diff(arr, lam_idx, order, n_uvars));

    // Compute the second part of the dividend only for order > 1, in order to avoid
    // an empty summation.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;

        // NOTE: iteration in the [1, order) range.
        for (std::uint32_t j = 1; j < order; ++j) {
            auto *fac = vector_splat(builder, llvm_constantfp(s, fp_t, static_cast<double>(j)), batch_size);

            auto *cnj = taylor_fetch_diff(arr, c_idx, order - j, n_uvars);
            auto *dnj = taylor_fetch_diff(arr, d_idx, order - j, n_uvars);
            auto *aj = taylor_fetch_diff(arr, idx, j, n_uvars);

            auto *tmp1 = llvm_fadd(s, cnj, dnj);
            auto *tmp2 = llvm_fmul(s, fac, aj);
            auto *tmp = llvm_fmul(s, tmp1, tmp2);
            sum.push_back(tmp);
        }

        // Update the dividend.
        dividend = llvm_fadd(s, dividend, pairwise_sum(s, sum));
    }

    return llvm_fdiv(s, dividend, divisor);
}

// LCOV_EXCL_START

// All the other cases.
template <typename U, typename V, typename W, typename... Args>
llvm::Value *taylor_diff_kepF_impl(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &, const U &, const V &,
                                   const W &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                   std::uint32_t, std::uint32_t, std::uint32_t, const Args &...)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of kepF()");
}

// LCOV_EXCL_STOP

llvm::Value *taylor_diff_kepF(llvm_state &s, llvm::Type *fp_t, const kepF_impl &f,
                              const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                              llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                              std::uint32_t batch_size)
{
    assert(f.args().size() == 3u);

    // LCOV_EXCL_START
    if (deps.size() != 4u) {
        throw std::invalid_argument(
            fmt::format("A hidden dependency vector of size 4 is expected in order to compute the Taylor "
                        "derivative of kepF(), but a vector of size {} was passed instead",
                        deps.size()));
    }
    // LCOV_EXCL_STOP

    return std::visit(
        [&](const auto &v1, const auto &v2, const auto &v3) {
            return taylor_diff_kepF_impl(s, fp_t, deps, v1, v2, v3, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value(), f.args()[1].value(), f.args()[2].value());
}

} // namespace

llvm::Value *kepF_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size, bool) const
{
    return taylor_diff_kepF(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

} // namespace detail

// NOTE: constant folding here would need a JIT-compiled version of kepF().
// Perhaps store the function pointer in a thread_local variable and keep around
// a cache of llvm states to fetch the pointer from?
expression kepF(expression h, expression k, expression lam)
{
    return expression{func{detail::kepF_impl{std::move(h), std::move(k), std::move(lam)}}};
}

#define HEYOKA_DEFINE_KEPF_OVERLOADS(type)                                                                             \
    expression kepF(expression h, type k, type lam)                                                                    \
    {                                                                                                                  \
        return kepF(std::move(h), expression{std::move(k)}, expression{std::move(lam)});                               \
    }                                                                                                                  \
    expression kepF(type h, expression k, type lam)                                                                    \
    {                                                                                                                  \
        return kepF(expression{std::move(h)}, std::move(k), expression{std::move(lam)});                               \
    }                                                                                                                  \
    expression kepF(type h, type k, expression lam)                                                                    \
    {                                                                                                                  \
        return kepF(expression{std::move(h)}, expression{std::move(k)}, std::move(lam));                               \
    }                                                                                                                  \
    expression kepF(expression h, expression k, type lam)                                                              \
    {                                                                                                                  \
        return kepF(std::move(h), std::move(k), expression{std::move(lam)});                                           \
    }                                                                                                                  \
    expression kepF(expression h, type k, expression lam)                                                              \
    {                                                                                                                  \
        return kepF(std::move(h), expression{std::move(k)}, std::move(lam));                                           \
    }                                                                                                                  \
    expression kepF(type h, expression k, expression lam)                                                              \
    {                                                                                                                  \
        return kepF(expression{std::move(h)}, std::move(k), std::move(lam));                                           \
    }

HEYOKA_DEFINE_KEPF_OVERLOADS(double)
HEYOKA_DEFINE_KEPF_OVERLOADS(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DEFINE_KEPF_OVERLOADS(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DEFINE_KEPF_OVERLOADS(mppp::real);

#endif

#undef HEYOKA_DEFINE_KEPF_OVERLOADS

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::kepF_impl)
