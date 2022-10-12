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

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
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

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sigmoid.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// C++ implementation.
template <typename T>
T cpp_sigmoid(T x)
{
    using std::exp;

    return 1 / (1 + exp(-x));
}

} // namespace

sigmoid_impl::sigmoid_impl(expression e) : func_base("sigmoid", std::vector{std::move(e)}) {}

sigmoid_impl::sigmoid_impl() : sigmoid_impl(0_dbl) {}

double sigmoid_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return cpp_sigmoid(heyoka::eval_dbl(args()[0], map, pars));
}

long double sigmoid_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                    const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return cpp_sigmoid(heyoka::eval_ldbl(args()[0], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 sigmoid_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                      const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return cpp_sigmoid(heyoka::eval_f128(args()[0], map, pars));
}
#endif

void sigmoid_impl::eval_batch_dbl(std::vector<double> &out,
                                  const std::unordered_map<std::string, std::vector<double>> &map,
                                  const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    heyoka::eval_batch_dbl(out, args()[0], map, pars);
    for (auto &el : out) {
        el = cpp_sigmoid(el);
    }
}

double sigmoid_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("Inconsistent number of arguments when computing the numerical value of the "
                        "sigmoid over doubles (1 argument was expected, but {} arguments were provided",
                        a.size()));
    }

    return cpp_sigmoid(a[0]);
}

double sigmoid_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 1u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the sigmoid");
    }
    auto sigma = cpp_sigmoid(a[0]);
    return sigma * (1 - sigma);
}

llvm::Value *sigmoid_impl::llvm_eval_dbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr,
                                         llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                         bool high_accuracy) const
{
    return llvm_eval_helper<double>(
        [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_sigmoid(s, args[0]); }, *this, s, eval_arr,
        par_ptr, stride, batch_size, high_accuracy);
}

llvm::Value *sigmoid_impl::llvm_eval_ldbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr,
                                          llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                          bool high_accuracy) const
{
    return llvm_eval_helper<long double>(
        [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_sigmoid(s, args[0]); }, *this, s, eval_arr,
        par_ptr, stride, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *sigmoid_impl::llvm_eval_f128(llvm_state &s, const std::vector<llvm::Value *> &eval_arr,
                                          llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                          bool high_accuracy) const
{
    return llvm_eval_helper<mppp::real128>(
        [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_sigmoid(s, args[0]); }, *this, s, eval_arr,
        par_ptr, stride, batch_size, high_accuracy);
}

#endif

namespace
{

template <typename T>
[[nodiscard]] llvm::Function *sigmoid_llvm_c_eval(llvm_state &s, const func_base &fb, std::uint32_t batch_size,
                                                  bool high_accuracy)
{
    return llvm_c_eval_func_helper<T>(
        "sigmoid", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_sigmoid(s, args[0]); }, fb, s,
        batch_size, high_accuracy);
}

} // namespace

llvm::Function *sigmoid_impl::llvm_c_eval_func_dbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return sigmoid_llvm_c_eval<double>(s, *this, batch_size, high_accuracy);
}

llvm::Function *sigmoid_impl::llvm_c_eval_func_ldbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return sigmoid_llvm_c_eval<long double>(s, *this, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *sigmoid_impl::llvm_c_eval_func_f128(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return sigmoid_llvm_c_eval<mppp::real128>(s, *this, batch_size, high_accuracy);
}

#endif

taylor_dc_t::size_type sigmoid_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Append the sigmoid decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Append the auxiliary function sigmoid(arg) * sigmoid(arg).
    u_vars_defs.emplace_back(square(expression{fmt::format("u_{}", u_vars_defs.size() - 1u)}),
                             std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 2)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));

    return u_vars_defs.size() - 2u;
}

namespace
{

// Derivative of sigmoid(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_sigmoid_impl(llvm_state &s, llvm::Type *fp_t, const sigmoid_impl &,
                                      const std::vector<std::uint32_t> &, const U &num,
                                      const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                      std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return llvm_sigmoid(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of sigmoid(variable).
llvm::Value *taylor_diff_sigmoid_impl(llvm_state &s, llvm::Type *fp_t, const sigmoid_impl &,
                                      const std::vector<std::uint32_t> &deps, const variable &var,
                                      const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                      std::uint32_t order, std::uint32_t a_idx, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_sigmoid(s, taylor_fetch_diff(arr, u_idx, 0, n_uvars));
    }

    // NOTE: iteration in the [1, order] range.
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is sigmoid(var) * sigmoid(var).
        auto *anj = taylor_fetch_diff(arr, a_idx, order - j, n_uvars);
        auto *bj = taylor_fetch_diff(arr, u_idx, j, n_uvars);
        auto *cnj = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);

        auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

        // Add j*(anj-cnj)*bj to the sum.
        auto *tmp1 = builder.CreateFSub(anj, cnj);
        auto *tmp2 = builder.CreateFMul(tmp1, bj);
        auto *tmp3 = builder.CreateFMul(tmp2, fac);

        sum.push_back(tmp3);
    }

    // Init the return value as the result of the sum.
    auto *ret_acc = pairwise_sum(builder, sum);

    // Finalise the return value: ret_acc / n.
    auto *div = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(order))), batch_size);

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_sigmoid_impl(llvm_state &, llvm::Type *, const sigmoid_impl &,
                                      const std::vector<std::uint32_t> &, const U &, const std::vector<llvm::Value *> &,
                                      llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a sigmoid");
}

llvm::Value *taylor_diff_sigmoid(llvm_state &s, llvm::Type *fp_t, const sigmoid_impl &f,
                                 const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                                 llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                 std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("A hidden dependency vector of size 1 is expected in order to compute the Taylor "
                        "derivative of the sigmoid, but a vector of size {} was passed instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_sigmoid_impl(s, fp_t, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *sigmoid_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                       const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                       std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                       std::uint32_t batch_size, bool) const
{
    return taylor_diff_sigmoid(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of sigmoid(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_sigmoid_impl(llvm_state &s, llvm::Type *fp_t, const sigmoid_impl &, const U &num,
                                                std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "sigmoid", 1,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_sigmoid(s, args[0]);
        },
        num);
}

// Derivative of sigmoid(variable).
llvm::Function *taylor_c_diff_func_sigmoid_impl(llvm_state &s, llvm::Type *fp_t, const sigmoid_impl &,
                                                const variable &var, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "sigmoid", n_uvars, batch_size, {var}, 1);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto a_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto b_idx = f->args().begin() + 5;
        auto dep_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of b_idx.
                builder.CreateStore(
                    llvm_sigmoid(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), b_idx)),
                    retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    auto anj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), a_idx);
                    auto bj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, b_idx);
                    auto cnj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), dep_idx);

                    // Compute the factor j.
                    auto fac = vector_splat(builder, builder.CreateUIToFP(j, fp_t), batch_size);

                    // Add  j*(anj-cnj)*bj into the sum.
                    auto tmp1 = builder.CreateFSub(anj, cnj);
                    auto tmp2 = builder.CreateFMul(tmp1, bj);
                    auto tmp3 = builder.CreateFMul(tmp2, fac);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(val_t, acc), tmp3), acc);
                });

                // Divide by the order to produce the return value.
                auto ord_v = vector_splat(builder, builder.CreateUIToFP(ord, fp_t), batch_size);

                builder.CreateStore(builder.CreateFDiv(builder.CreateLoad(val_t, acc), ord_v), retval);
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
                "Inconsistent function signature for the Taylor derivative of the sigmpid in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_sigmoid_impl(llvm_state &, llvm::Type *, const sigmoid_impl &, const U &,
                                                std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a sigmpid in compact mode");
}

llvm::Function *taylor_c_diff_func_sigmoid(llvm_state &s, llvm::Type *fp_t, const sigmoid_impl &fn,
                                           std::uint32_t n_uvars, std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_func_sigmoid_impl(s, fp_t, fn, v, n_uvars, batch_size); },
        fn.args()[0].value());
}

} // namespace

llvm::Function *sigmoid_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                 std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_sigmoid(s, fp_t, *this, n_uvars, batch_size);
}

std::vector<expression> sigmoid_impl::gradient() const
{
    assert(args().size() == 1u);
    // NOTE: if single-precision floats are implemented,
    // should 1_dbl become 1_flt?
    return {(1_dbl - sigmoid(args()[0])) * sigmoid(args()[0])};
}

} // namespace detail

expression sigmoid(expression e)
{
    return expression{func{detail::sigmoid_impl(std::move(e))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::sigmoid_impl)
