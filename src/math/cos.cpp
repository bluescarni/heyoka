// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/neg.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

cos_impl::cos_impl(expression e) : func_base("cos", std::vector{std::move(e)}) {}

cos_impl::cos_impl() : cos_impl(0_dbl) {}

double cos_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return std::cos(heyoka::eval_dbl(args()[0], map, pars));
}

long double cos_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return std::cos(heyoka::eval_ldbl(args()[0], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 cos_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                  const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return mppp::cos(heyoka::eval_f128(args()[0], map, pars));
}
#endif

void cos_impl::eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &map,
                              const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    heyoka::eval_batch_dbl(out, args()[0], map, pars);
    for (auto &el : out) {
        el = std::cos(el);
    }
}

double cos_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("Inconsistent number of arguments when computing the numerical value of the "
                        "cosine over doubles (1 argument was expected, but {} arguments were provided",
                        a.size()));
    }

    return std::cos(a[0]);
}

double cos_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 1u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the cosine");
    }

    return -std::sin(a[0]);
}

llvm::Value *cos_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    return llvm_eval_helper([&s](const std::vector<llvm::Value *> &args, bool) { return llvm_cos(s, args[0]); }, *this,
                            s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *cos_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                              std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "cos", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_cos(s, args[0]); }, fb, s, fp_t,
        batch_size, high_accuracy);
}

} // namespace

llvm::Function *cos_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    return cos_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

taylor_dc_t::size_type cos_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Append the sine decomposition.
    u_vars_defs.emplace_back(sin(args()[0]), std::vector<std::uint32_t>{});

    // Append the cosine decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden deps.
    (u_vars_defs.end() - 2)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed cosine).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Derivative of cos(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_cos_impl(llvm_state &s, llvm::Type *fp_t, const cos_impl &, const std::vector<std::uint32_t> &,
                                  const U &num, const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return llvm_cos(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

llvm::Value *taylor_diff_cos_impl(llvm_state &s, llvm::Type *fp_t, const cos_impl &,
                                  const std::vector<std::uint32_t> &deps, const variable &var,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_cos(s, taylor_fetch_diff(arr, u_idx, 0, n_uvars));
    }

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is sin(var).
        auto *v0 = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);
        auto *v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

        auto *fac = vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(j))), batch_size);

        // Add j*v0*v1 to the sum.
        sum.push_back(llvm_fmul(s, fac, llvm_fmul(s, v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto *ret_acc = pairwise_sum(s, sum);

    // Compute and return the result: -ret_acc / order
    auto *div = vector_splat(builder, llvm_codegen(s, fp_t, number(-static_cast<double>(order))), batch_size);

    return llvm_fdiv(s, ret_acc, div);
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_cos_impl(llvm_state &, llvm::Type *, const cos_impl &, const std::vector<std::uint32_t> &,
                                  const U &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a cosine");
}

llvm::Value *taylor_diff_cos(llvm_state &s, llvm::Type *fp_t, const cos_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        throw std::invalid_argument(
            fmt::format("A hidden dependency vector of size 1 is expected in order to compute the Taylor "
                        "derivative of the cosine, but a vector of size {} was passed instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_cos_impl(s, fp_t, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *cos_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool) const
{
    return taylor_diff_cos(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of cos(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_cos_impl(llvm_state &s, llvm::Type *fp_t, const cos_impl &, const U &num,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "cos", 1,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_cos(s, args[0]);
        },
        num);
}

// Derivative of cos(variable).
llvm::Function *taylor_c_diff_func_cos_impl(llvm_state &s, llvm::Type *fp_t, const cos_impl &, const variable &var,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "cos", n_uvars, batch_size, {var}, 1);
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
        auto diff_ptr = f->args().begin() + 2;
        auto var_idx = f->args().begin() + 5;
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
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    llvm_cos(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx)), retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    auto b_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), dep_idx);
                    auto cj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx);

                    auto j_v = vector_splat(builder, llvm_ui_to_fp(s, j, fp_t), batch_size);

                    builder.CreateStore(
                        llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, j_v, llvm_fmul(s, b_nj, cj))), acc);
                });

                // Divide by the order and negate to produce the return value.
                auto ord_v = vector_splat(builder, llvm_ui_to_fp(s, ord, fp_t), batch_size);
                builder.CreateStore(llvm_fdiv(s, builder.CreateLoad(val_t, acc), llvm_fneg(s, ord_v)), retval);
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
                "Inconsistent function signature for the Taylor derivative of the cosine in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_cos_impl(llvm_state &, llvm::Type *, const cos_impl &, const U &, std::uint32_t,
                                            std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a cosine in compact mode");
}

llvm::Function *taylor_c_diff_func_cos(llvm_state &s, llvm::Type *fp_t, const cos_impl &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_cos_impl(s, fp_t, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *cos_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_cos(s, fp_t, *this, n_uvars, batch_size);
}

std::vector<expression> cos_impl::gradient() const
{
    assert(args().size() == 1u);
    return {-sin(args()[0])};
}

} // namespace detail

expression cos(expression e)
{
    if (const auto *fptr = detail::is_neg(e)) {
        // Simplify cos(-x) to cos(x).
        assert(fptr->args().size() == 1u);
        return cos(fptr->args()[0]);
    } else {
        // Simplify cos(number) to its value.
        if (const auto *num_ptr = std::get_if<number>(&e.value())) {
            return expression{std::visit(
                [](const auto &x) {
                    using std::cos;

                    return number{cos(x)};
                },
                num_ptr->value())};
        } else {
            return expression{func{detail::cos_impl(std::move(e))}};
        }
    }
}

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::cos_impl)
