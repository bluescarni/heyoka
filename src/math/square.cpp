// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

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
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

square_impl::square_impl(expression e) : func_base("square", std::vector{std::move(e)}) {}

square_impl::square_impl() : square_impl(0_dbl) {}

void square_impl::to_stream(std::ostream &os) const
{
    assert(args().size() == 1u);

    os << args()[0] << "**2";
}

std::vector<expression> square_impl::gradient() const
{
    assert(args().size() == 1u);
    return {2_dbl * args()[0]};
}

double square_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return heyoka::eval_dbl(args()[0], map, pars) * heyoka::eval_dbl(args()[0], map, pars);
}

long double square_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                   const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return heyoka::eval_ldbl(args()[0], map, pars) * heyoka::eval_ldbl(args()[0], map, pars);
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 square_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                     const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return heyoka::eval_f128(args()[0], map, pars) * heyoka::eval_f128(args()[0], map, pars);
}
#endif

llvm::Value *square_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                    llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                    bool high_accuracy) const
{
    return llvm_eval_helper([&s](const std::vector<llvm::Value *> &args, bool) { return llvm_square(s, args[0]); },
                            *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *square_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                                 std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "square", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_square(s, args[0]); }, fb, s, fp_t,
        batch_size, high_accuracy);
}

} // namespace

llvm::Function *square_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                              bool high_accuracy) const
{
    return square_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

namespace
{

// Derivative of square(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_square_impl(llvm_state &s, llvm::Type *fp_t, const square_impl &, const U &num,
                                     const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                     std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return llvm_square(s, taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of square(variable).
llvm::Value *taylor_diff_square_impl(llvm_state &s, llvm::Type *, const square_impl &, const variable &var,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t, std::uint32_t)
{
    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    if (order == 0u) {
        return llvm_square(s, taylor_fetch_diff(arr, u_idx, 0, n_uvars));
    }

    // Compute the sum.
    std::vector<llvm::Value *> sum;
    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 0; j <= (order - 1u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto *ret = pairwise_sum(builder, sum);
        return llvm_fadd(s, ret, ret);
    } else {
        // Even order.
        auto *ak2 = taylor_fetch_diff(arr, u_idx, order / 2u, n_uvars);
        auto *sq_ak2 = builder.CreateFMul(ak2, ak2);

        for (std::uint32_t j = 0; j <= (order - 2u) / 2u; ++j) {
            auto *v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
            auto *v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

            sum.push_back(builder.CreateFMul(v0, v1));
        }

        auto *ret = pairwise_sum(builder, sum);
        return llvm_fadd(s, llvm_fadd(s, ret, ret), sq_ak2);
    }
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_square_impl(llvm_state &, llvm::Type *, const square_impl &, const U &,
                                     const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                     std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a square");
}

llvm::Value *taylor_diff_square(llvm_state &s, llvm::Type *fp_t, const square_impl &f,
                                const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                                llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the square, but a vector of size {} was passed "
                        "instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_square_impl(s, fp_t, f, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *square_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                      const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                      std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                      std::uint32_t batch_size, bool) const
{
    return taylor_diff_square(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of square(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &s, llvm::Type *fp_t, const square_impl &, const U &num,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "square", 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_square(s, args[0]);
        },
        num);
}

// Derivative of square(variable).
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &s, llvm::Type *fp_t, const square_impl &,
                                               const variable &var, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "square", n_uvars, batch_size, {var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = module.getFunction(fname);

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
        auto *ord = f->args().begin();
        auto *diff_ptr = f->args().begin() + 2;
        auto *var_idx = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order 0, invoke the function on the order 0 of var_idx.
                builder.CreateStore(
                    llvm_square(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Distinguish the odd/even cases for the order.
                llvm_if_then_else(
                    s, builder.CreateICmpEQ(builder.CreateURem(ord, builder.getInt32(2)), builder.getInt32(1)),
                    [&]() {
                        // Odd order.
                        auto *loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(1)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto *a_nj
                                = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                            auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx);

                            builder.CreateStore(
                                llvm_fadd(s, builder.CreateLoad(val_t, acc), builder.CreateFMul(a_nj, aj)), acc);
                        });

                        // Return 2 * acc.
                        auto *acc_load = builder.CreateLoad(val_t, acc);
                        builder.CreateStore(llvm_fadd(s, acc_load, acc_load), retval);
                    },
                    [&]() {
                        // Even order.

                        // Pre-compute the final term.
                        auto *ak2 = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars,
                                                       builder.CreateUDiv(ord, builder.getInt32(2)), var_idx);
                        auto *sq_ak2 = builder.CreateFMul(ak2, ak2);

                        auto *loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(ord, builder.getInt32(2)), builder.getInt32(2)),
                            builder.getInt32(1));
                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) {
                            auto *a_nj
                                = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), var_idx);
                            auto *aj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx);

                            builder.CreateStore(
                                llvm_fadd(s, builder.CreateLoad(val_t, acc), builder.CreateFMul(a_nj, aj)), acc);
                        });

                        // Return 2 * acc + ak2 * ak2.
                        auto *acc_load = builder.CreateLoad(val_t, acc);
                        builder.CreateStore(llvm_fadd(s, llvm_fadd(s, acc_load, acc_load), sq_ak2), retval);
                    });
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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of the square "
                                        "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_square_impl(llvm_state &, llvm::Type *, const square_impl &, const U &,
                                               std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a square in compact mode");
}

llvm::Function *taylor_c_diff_func_square(llvm_state &s, llvm::Type *fp_t, const square_impl &fn, std::uint32_t n_uvars,
                                          std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_func_square_impl(s, fp_t, fn, v, n_uvars, batch_size); },
        fn.args()[0].value());
}

} // namespace

llvm::Function *square_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_square(s, fp_t, *this, n_uvars, batch_size);
}

} // namespace detail

expression square(expression e)
{
    return expression{func{detail::square_impl(std::move(e))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::square_impl)
