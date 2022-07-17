// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/log.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

pow_impl::pow_impl(expression b, expression e) : func_base("pow", std::vector{std::move(b), std::move(e)}) {}

pow_impl::pow_impl() : pow_impl(1_dbl, 1_dbl) {}

namespace
{

// NOTE: we want to allow approximate implementations of pow()
// in the following cases:
// - exponent is an integral number n (in which case we want to allow
//   transformation in a sequence of multiplications),
// - exponent is a value of type n / 2, with n an odd integral value (in which case
//   we want to give the option of implementing pow() on top of sqrt()).
bool pow_allow_approx(const pow_impl &pi)
{
    return is_integral(pi.args()[1]) || is_odd_integral_half(pi.args()[1]);
}

} // namespace

double pow_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 2u);

    return std::pow(heyoka::eval_dbl(args()[0], map, pars), heyoka::eval_dbl(args()[1], map, pars));
}

long double pow_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                const std::vector<long double> &pars) const
{
    assert(args().size() == 2u);

    return std::pow(heyoka::eval_ldbl(args()[0], map, pars), heyoka::eval_ldbl(args()[1], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 pow_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                  const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 2u);

    return mppp::pow(heyoka::eval_f128(args()[0], map, pars), heyoka::eval_f128(args()[1], map, pars));
}
#endif

void pow_impl::eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &map,
                              const std::vector<double> &pars) const
{
    assert(args().size() == 2u);

    auto out0 = out; // is this allocation needed?
    heyoka::eval_batch_dbl(out0, args()[0], map, pars);
    heyoka::eval_batch_dbl(out, args()[1], map, pars);
    for (decltype(out.size()) i = 0; i < out.size(); ++i) {
        out[i] = std::pow(out0[i], out[i]);
    }
}

double pow_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 2u) {
        throw std::invalid_argument(
            fmt::format("Inconsistent number of arguments when computing the numerical value of the "
                        "exponentiation over doubles (2 arguments were expected, but {} arguments were provided",
                        a.size()));
    }

    return std::pow(a[0], a[1]);
}

double pow_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 2u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the exponentiation");
    }

    return a[1] * std::pow(a[0], a[1] - 1.) + std::log(a[0]) * std::pow(a[0], a[1]);
}

llvm::Value *pow_impl::llvm_eval_dbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                     llvm::Value *stride, std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<double>([&s, this](const std::vector<llvm::Value *> &args,
                                               bool) { return llvm_pow(s, args[0], args[1], pow_allow_approx(*this)); },
                                    *this, s, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Value *pow_impl::llvm_eval_ldbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                      llvm::Value *stride, std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<long double>(
        [&s, this](const std::vector<llvm::Value *> &args, bool) {
            return llvm_pow(s, args[0], args[1], pow_allow_approx(*this));
        },
        *this, s, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *pow_impl::llvm_eval_f128(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                      llvm::Value *stride, std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<mppp::real128>(
        [&s, this](const std::vector<llvm::Value *> &args, bool) {
            return llvm_pow(s, args[0], args[1], pow_allow_approx(*this));
        },
        *this, s, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

#endif

namespace
{

template <typename T>
[[nodiscard]] llvm::Function *pow_llvm_c_eval(llvm_state &s, const pow_impl &pimpl, std::uint32_t batch_size,
                                              bool high_accuracy)
{
    const auto allow_approx = pow_allow_approx(pimpl);

    return llvm_c_eval_func_helper<T>(
        allow_approx ? "pow_approx" : "pow",
        [&s, allow_approx](const std::vector<llvm::Value *> &args, bool) {
            return llvm_pow(s, args[0], args[1], allow_approx);
        },
        pimpl, s, batch_size, high_accuracy);
}

} // namespace

llvm::Function *pow_impl::llvm_c_eval_func_dbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return pow_llvm_c_eval<double>(s, *this, batch_size, high_accuracy);
}

llvm::Function *pow_impl::llvm_c_eval_func_ldbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return pow_llvm_c_eval<long double>(s, *this, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *pow_impl::llvm_c_eval_func_f128(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return pow_llvm_c_eval<mppp::real128>(s, *this, batch_size, high_accuracy);
}

#endif

namespace
{

// Derivative of pow(number, number).
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, const pow_impl &f, const U &num0, const V &num1,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Check if we can use the approximated version.
    const auto allow_approx = pow_allow_approx(f);

    if (order == 0u) {
        return llvm_pow(s, taylor_codegen_numparam<T>(s, num0, par_ptr, batch_size),
                        taylor_codegen_numparam<T>(s, num1, par_ptr, batch_size), allow_approx);
    } else {
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of pow(variable, number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, const pow_impl &f, const variable &var, const U &num,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                  std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Check if we can use the approximated version.
    const auto allow_approx = pow_allow_approx(f);

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_pow(s, taylor_fetch_diff(arr, u_idx, 0, n_uvars),
                        taylor_codegen_numparam<T>(s, num, par_ptr, batch_size), allow_approx);
    }

    // NOTE: iteration in the [0, order) range
    // (i.e., order *not* included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

        // Compute the scalar factor: order * num - j * (num + 1).
        auto scal_f = [&]() -> llvm::Value * {
            if constexpr (std::is_same_v<U, number>) {
                return vector_splat(builder,
                                    codegen<T>(s, number(static_cast<T>(order)) * num
                                                      - number(static_cast<T>(j)) * (num + number(static_cast<T>(1)))),
                                    batch_size);
            } else {
                auto pc = taylor_codegen_numparam<T>(s, num, par_ptr, batch_size);
                auto jvec = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);
                auto ordvec = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);
                auto onevec = vector_splat(builder, codegen<T>(s, number(static_cast<T>(1))), batch_size);

                auto tmp1 = builder.CreateFMul(ordvec, pc);
                auto tmp2 = builder.CreateFMul(jvec, builder.CreateFAdd(pc, onevec));

                return builder.CreateFSub(tmp1, tmp2);
            }
        }();

        // Add scal_f*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(scal_f, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Compute the final divisor: order * (zero-th derivative of u_idx).
    auto ord_f = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);
    auto b0 = taylor_fetch_diff(arr, u_idx, 0, n_uvars);
    auto div = builder.CreateFMul(ord_f, b0);

    // Compute and return the result: ret_acc / div.
    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U1, typename U2,
          std::enable_if_t<!std::conjunction_v<is_num_param<U1>, is_num_param<U2>>, int> = 0>
llvm::Value *taylor_diff_pow_impl(llvm_state &, const pow_impl &, const U1 &, const U2 &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a pow()");
}

template <typename T>
llvm::Value *taylor_diff_pow(llvm_state &s, const pow_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 2u);

    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the exponentiation, but a vector of size {} was passed "
                        "instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_pow_impl<T>(s, f, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value(), f.args()[1].value());
}

} // namespace

llvm::Value *pow_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                       const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                       std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                       std::uint32_t batch_size, bool) const
{
    return taylor_diff_pow<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *pow_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size, bool) const
{
    return taylor_diff_pow<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *pow_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size, bool) const
{
    return taylor_diff_pow<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

namespace
{

// Derivative of pow(number, number).
template <typename T, typename U, typename V,
          std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &s, const pow_impl &fn, const U &n0, const V &n1,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    // Check if we can use the approximated version.
    const auto allow_approx = pow_allow_approx(fn);

    // Create the function name.
    const auto *const pow_name = allow_approx ? "pow_approx" : "pow";

    return taylor_c_diff_func_numpar<T>(
        s, n_uvars, batch_size, pow_name, 0,
        [&s, allow_approx](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 2u);
            assert(args[0] != nullptr);
            assert(args[1] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_pow(s, args[0], args[1], allow_approx);
        },
        n0, n1);
}

// Derivative of pow(variable, number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &s, const pow_impl &fn, const variable &var, const U &n,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Check if we can use the approximated version.
    const auto allow_approx = pow_allow_approx(fn);

    // Create the function name.
    const auto *const pow_name = allow_approx ? "pow_approx" : "pow";

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args<T>(context, pow_name, n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 5;
        auto exponent = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(llvm_pow(s, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), var_idx),
                                             taylor_c_diff_numparam_codegen(s, n, exponent, par_ptr, batch_size),
                                             allow_approx),
                                    retval);
            },
            [&]() {
                // Create FP vector versions of exponent and order.
                auto alpha_v = taylor_c_diff_numparam_codegen(s, n, exponent, par_ptr, batch_size);
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(0), ord, [&](llvm::Value *j) {
                    auto b_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                    auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, u_idx);

                    // Compute the factor n*alpha-j*(alpha+1).
                    auto j_v = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
                    auto fac = builder.CreateFSub(
                        builder.CreateFMul(ord_v, alpha_v),
                        builder.CreateFMul(
                            j_v,
                            builder.CreateFAdd(alpha_v, vector_splat(builder, codegen<T>(s, number{1.}), batch_size))));

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(val_t, acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(b_nj, aj))),
                                        acc);
                });

                // Finalize the result: acc / (n*b0).
                builder.CreateStore(
                    builder.CreateFDiv(builder.CreateLoad(val_t, acc),
                                       builder.CreateFMul(ord_v, taylor_c_load_diff(s, diff_ptr, n_uvars,
                                                                                    builder.getInt32(0), var_idx))),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signatures for the Taylor derivative of pow() in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U1, typename U2,
          std::enable_if_t<!std::conjunction_v<is_num_param<U1>, is_num_param<U2>>, int> = 0>
llvm::Function *taylor_c_diff_func_pow_impl(llvm_state &, const pow_impl &, const U1 &, const U2 &, std::uint32_t,
                                            std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a pow() in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_pow(llvm_state &s, const pow_impl &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 2u);

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_c_diff_func_pow_impl<T>(s, fn, v1, v2, n_uvars, batch_size);
        },
        fn.args()[0].value(), fn.args()[1].value());
}

} // namespace

llvm::Function *pow_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                 bool) const
{
    return taylor_c_diff_func_pow<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *pow_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                  bool) const
{
    return taylor_c_diff_func_pow<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *pow_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                  bool) const
{
    return taylor_c_diff_func_pow<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

std::vector<expression> pow_impl::gradient() const
{
    assert(args().size() == 2u);
    return {args()[1] * pow(args()[0], args()[1] - 1_dbl), pow(args()[0], args()[1]) * log(args()[0])};
}

namespace
{

// Wrapper for the implementation of the top-level pow() function.
// It will special-case for e == 0, 1, 2, 3, 4 and 0.5.
expression pow_wrapper_impl(expression b, expression e)
{
    if (auto num_ptr = std::get_if<number>(&e.value())) {
        if (is_zero(*num_ptr)) {
            return 1_dbl;
        }

        if (is_one(*num_ptr)) {
            return b;
        }

        if (std::visit([](const auto &v) { return v == 2; }, num_ptr->value())) {
            return square(std::move(b));
        }

        if (std::visit([](const auto &v) { return v == 3; }, num_ptr->value())) {
            return powi(std::move(b), 3);
        }

        if (std::visit([](const auto &v) { return v == 4; }, num_ptr->value())) {
            return powi(std::move(b), 4);
        }

        if (std::visit([](const auto &v) { return v == .5; }, num_ptr->value())) {
            return sqrt(std::move(b));
        }
    }

    return expression{func{pow_impl{std::move(b), std::move(e)}}};
}

} // namespace

} // namespace detail

expression pow(expression b, expression e)
{
    return detail::pow_wrapper_impl(std::move(b), std::move(e));
}

expression pow(expression b, double e)
{
    return detail::pow_wrapper_impl(std::move(b), expression{e});
}

expression pow(expression b, long double e)
{
    return detail::pow_wrapper_impl(std::move(b), expression{e});
}

#if defined(HEYOKA_HAVE_REAL128)

expression pow(expression b, mppp::real128 e)
{
    return detail::pow_wrapper_impl(std::move(b), expression{e});
}

#endif

// Natural power.
expression powi(expression b, std::uint32_t e)
{
    switch (e) {
        case 0u:
            return 1_dbl;
        case 1u:
            return b;
        case 2u:
            return square(std::move(b));
    }

    // https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    auto y = 1_dbl;

    while (e > 1u) {
        if (e % 2u == 0u) {
            b = square(std::move(b));
            e /= 2u;
        } else {
            y = b * std::move(y);
            b = square(std::move(b));
            e = (e - 1u) / 2u;
        }
    }

    return std::move(b) * std::move(y);
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::pow_impl)
