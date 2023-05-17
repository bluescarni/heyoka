// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
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
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/sum_sq.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

sum_sq_impl::sum_sq_impl() : sum_sq_impl(std::vector<expression>{}) {}

sum_sq_impl::sum_sq_impl(std::vector<expression> v) : func_base("sum_sq", std::move(v)) {}

void sum_sq_impl::to_stream(std::ostringstream &oss) const
{
    if (args().size() == 1u) {
        // NOTE: avoid brackets if there's only 1 argument.
        stream_expression(oss, args()[0]);
        oss << "**2";
    } else {
        oss << '(';

        for (decltype(args().size()) i = 0; i < args().size(); ++i) {
            stream_expression(oss, args()[i]);
            oss << "**2";
            if (i != args().size() - 1u) {
                oss << " + ";
            }
        }

        oss << ')';
    }
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
bool sum_sq_impl::is_commutative() const
{
    return true;
}

namespace
{

llvm::Value *sum_sq_llvm_eval_impl(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                   const std::vector<llvm::Value *> &eval_arr,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size,
                                   bool high_accuracy)
{
    return llvm_eval_helper(
        [&s](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            std::vector<llvm::Value *> sqs;
            sqs.reserve(args.size());

            for (auto *val : args) {
                sqs.push_back(llvm_square(s, val));
            }

            return pairwise_sum(s, sqs);
        },
        fb, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

} // namespace

llvm::Value *sum_sq_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                    llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                    bool high_accuracy) const
{
    return sum_sq_llvm_eval_impl(s, fp_t, *this, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *sum_sq_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                                 std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "sum_sq",
        [&s](const std::vector<llvm::Value *> &args, bool) {
            std::vector<llvm::Value *> sqs;
            sqs.reserve(args.size());

            for (auto *val : args) {
                sqs.push_back(llvm_square(s, val));
            }

            return pairwise_sum(s, sqs);
        },
        fb, s, fp_t, batch_size, high_accuracy);
}

} // namespace

llvm::Function *sum_sq_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                              bool high_accuracy) const
{
    return sum_sq_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

namespace
{

llvm::Value *sum_sq_taylor_diff_impl(llvm_state &s, llvm::Type *fp_t, const sum_sq_impl &sf,
                                     const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                                     llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                     std::uint32_t order, std::uint32_t batch_size)
{
    // NOTE: this is prevented in the implementation
    // of the sum_sq() function.
    assert(!sf.args().empty());

    if (!deps.empty()) {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("The vector of hidden dependencies in the Taylor diff for a sum of squares "
                        "should be empty, but instead it has a size of {}",
                        deps.size()));
        // LCOV_EXCL_STOP
    }

    auto &builder = s.builder();

    // Each vector in v_sums will contain the terms in the summation in the formula
    // for the computation of the Taylor derivative of square() for each argument in sf.
    std::vector<std::vector<llvm::Value *>> v_sums;
    v_sums.resize(boost::numeric_cast<decltype(v_sums.size())>(sf.args().size()));

    // This function calculates the j-th term in the summation in the formula for the
    // Taylor derivative of square() for each k-th argument in sf, and appends the result
    // to the k-th entry in v_sums.
    auto looper = [&](std::uint32_t j) {
        for (decltype(sf.args().size()) k = 0; k < sf.args().size(); ++k) {
            std::visit(
                [&](const auto &v) {
                    using type = detail::uncvref_t<decltype(v)>;

                    if constexpr (std::is_same_v<type, variable>) {
                        // Variable.
                        const auto u_idx = uname_to_index(v.name());

                        auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
                        auto v1 = taylor_fetch_diff(arr, u_idx, j, n_uvars);

                        v_sums[k].push_back(llvm_fmul(s, v0, v1));
                    } else if constexpr (is_num_param_v<type>) {
                        // Number/param.

                        // NOTE: for number/params, all terms in the summation
                        // will be zero. Thus, ensure that v_sums[k] just
                        // contains a single zero.
                        if (v_sums[k].empty()) {
                            v_sums[k].push_back(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size));
                        }
                    } else {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            "An invalid argument type was encountered while trying to build the "
                            "Taylor derivative of a sum of squares");
                        // LCOV_EXCL_STOP
                    }
                },
                sf.args()[k].value());
        }
    };

    if (order % 2u == 1u) {
        // Odd order.
        for (std::uint32_t j = 0; j <= (order - 1u) / 2u; ++j) {
            looper(j);
        }

        // Pairwise sum each item in v_sums.
        std::vector<llvm::Value *> tmp;
        tmp.reserve(boost::numeric_cast<decltype(tmp.size())>(v_sums.size()));
        for (auto &v_sum : v_sums) {
            tmp.push_back(pairwise_sum(s, v_sum));
        }

        // Sum the sums.
        pairwise_sum(s, tmp);

        // Multiply by 2 and return.
        return llvm_fadd(s, tmp[0], tmp[0]);
    } else {
        // Even order.
        for (std::uint32_t j = 0; order > 0u && j <= (order - 2u) / 2u; ++j) {
            looper(j);
        }

        // Pairwise sum each item in v_sums, multiply the result by 2 and add the
        // term outside the summation.
        std::vector<llvm::Value *> tmp;
        tmp.reserve(boost::numeric_cast<decltype(tmp.size())>(v_sums.size()));
        for (decltype(sf.args().size()) k = 0; k < sf.args().size(); ++k) {
            // Compute the term outside the summation and store it in tmp.
            tmp.push_back(std::visit(
                [&](const auto &v) -> llvm::Value * {
                    using type = detail::uncvref_t<decltype(v)>;

                    if constexpr (std::is_same_v<type, variable>) {
                        // Variable.
                        auto val = taylor_fetch_diff(arr, uname_to_index(v.name()), order / 2u, n_uvars);
                        return llvm_fmul(s, val, val);
                    } else if constexpr (is_num_param_v<type>) {
                        // Number/param.
                        if (order == 0u) {
                            auto val = taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size);
                            return llvm_fmul(s, val, val);
                        } else {
                            return vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size);
                        }
                    } else {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            "An invalid argument type was encountered while trying to build the "
                            "Taylor derivative of a sum of squares");
                        // LCOV_EXCL_STOP
                    }
                },
                sf.args()[k].value()));

            // NOTE: avoid doing the pairwise sum if the order is 0, in which case
            // the items in v_sums are all empty and tmp.back() contains only the term
            // outside the summation.
            if (order > 0u) {
                auto *p_sum = pairwise_sum(s, v_sums[k]);
                // Muliply the pairwise sum by 2.
                p_sum = llvm_fadd(s, p_sum, p_sum);
                // Add it to the term outside the sum.
                tmp.back() = llvm_fadd(s, p_sum, tmp.back());
            }
        }

        // Sum the sums and return.
        return pairwise_sum(s, tmp);
    }
}

} // namespace

llvm::Value *sum_sq_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                      const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                      std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                      std::uint32_t batch_size, bool) const
{
    return sum_sq_taylor_diff_impl(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, batch_size);
}

namespace
{

llvm::Function *sum_sq_taylor_c_diff_func_impl(llvm_state &s, llvm::Type *fp_t, const sum_sq_impl &sf,
                                               std::uint32_t n_uvars, std::uint32_t batch_size)
{
    // NOTE: this is prevented in the implementation
    // of the sum() function.
    assert(!sf.args().empty());

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Build the vector of arguments needed to determine the function name.
    std::vector<std::variant<variable, number, param>> nm_args;
    nm_args.reserve(static_cast<decltype(nm_args.size())>(sf.args().size()));
    for (const auto &arg : sf.args()) {
        nm_args.push_back(std::visit(
            [](const auto &v) -> std::variant<variable, number, param> {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    // LCOV_EXCL_START
                    assert(false);
                    throw;
                    // LCOV_EXCL_STOP
                } else {
                    return v;
                }
            },
            arg.value()));
    }

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "sum_sq", n_uvars, batch_size, nm_args);
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
        // NOTE: force inline.
        f->addFnAttr(llvm::Attribute::AlwaysInline);

        // Fetch the necessary function arguments.
        auto *order = f->args().begin();
        auto *diff_arr = f->args().begin() + 2;
        auto *par_ptr = f->args().begin() + 3;
        auto *terms = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulators for each argument in the summation, and init them to zero.
        std::vector<llvm::Value *> v_accs;
        v_accs.resize(boost::numeric_cast<decltype(v_accs.size())>(sf.args().size()));
        for (auto &acc : v_accs) {
            acc = builder.CreateAlloca(val_t);
            builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);
        }

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // This function calculates the j-th term in the summation in the formula for the
        // Taylor derivative of square() for each k-th argument in sf, and accumulates the result
        // into the k-th entry in v_accs.
        auto looper = [&](llvm::Value *j) {
            for (decltype(sf.args().size()) k = 0; k < sf.args().size(); ++k) {
                std::visit(
                    [&](const auto &v) {
                        using type = detail::uncvref_t<decltype(v)>;

                        if constexpr (std::is_same_v<type, variable>) {
                            // Variable.
                            auto *v0 = taylor_c_load_diff(s, val_t, diff_arr, n_uvars, builder.CreateSub(order, j),
                                                          terms + k);
                            auto *v1 = taylor_c_load_diff(s, val_t, diff_arr, n_uvars, j, terms + k);

                            // Update the k-th accumulator.
                            builder.CreateStore(
                                llvm_fadd(s, builder.CreateLoad(val_t, v_accs[k]), llvm_fmul(s, v0, v1)), v_accs[k]);
                        } else if constexpr (is_num_param_v<type>) {
                            // Number/param: nothing to do, leave the accumulator to zero.
                        } else {
                            // LCOV_EXCL_START
                            throw std::invalid_argument(
                                "An invalid argument type was encountered while trying to build the "
                                "Taylor derivative of a sum of squares in compact mode");
                            // LCOV_EXCL_STOP
                        }
                    },
                    sf.args()[k].value());
            }
        };

        // Distinguish odd/even cases.
        auto *odd_or_even = builder.CreateICmpEQ(builder.CreateURem(order, builder.getInt32(2)), builder.getInt32(1));

        llvm_if_then_else(
            s, odd_or_even,
            [&]() {
                // Odd order.
                auto *loop_end = builder.CreateAdd(
                    builder.CreateUDiv(builder.CreateSub(order, builder.getInt32(1)), builder.getInt32(2)),
                    builder.getInt32(1));

                llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) { looper(j); });

                // Run a pairwise summation on the vector of accumulators.
                std::vector<llvm::Value *> tmp;
                tmp.reserve(v_accs.size());
                for (auto &acc : v_accs) {
                    tmp.push_back(builder.CreateLoad(val_t, acc));
                }
                auto *ret = pairwise_sum(s, tmp);

                // Return 2 * ret.
                builder.CreateStore(llvm_fadd(s, ret, ret), retval);
            },
            [&]() {
                // Even order.
                // NOTE: run the loop only if we are not at order 0.
                llvm_if_then_else(
                    s, builder.CreateICmpEQ(order, builder.getInt32(0)),
                    []() {
                        // Order 0, do nothing.
                    },
                    [&]() {
                        // Order 2 or higher.
                        auto *loop_end = builder.CreateAdd(
                            builder.CreateUDiv(builder.CreateSub(order, builder.getInt32(2)), builder.getInt32(2)),
                            builder.getInt32(1));

                        llvm_loop_u32(s, builder.getInt32(0), loop_end, [&](llvm::Value *j) { looper(j); });
                    });

                // Multiply each accumulator by two and add the term outside the summation.
                std::vector<llvm::Value *> tmp;
                tmp.reserve(v_accs.size());
                for (decltype(sf.args().size()) k = 0; k < sf.args().size(); ++k) {
                    // Load the current accumulator and multiply it by 2.
                    auto *acc_val = builder.CreateLoad(val_t, v_accs[k]);
                    auto *acc2 = llvm_fadd(s, acc_val, acc_val);

                    // Load the external term.
                    auto *ex_term = std::visit( // LCOV_EXCL_LINE
                        [&](const auto &v) -> llvm::Value * {
                            using type = detail::uncvref_t<decltype(v)>;

                            if constexpr (std::is_same_v<type, variable>) {
                                // Variable.
                                auto *val
                                    = taylor_c_load_diff(s, val_t, diff_arr, n_uvars,
                                                         builder.CreateUDiv(order, builder.getInt32(2)), terms + k);
                                return llvm_fmul(s, val, val);
                            } else if constexpr (is_num_param_v<type>) {
                                // Number/param.
                                auto *ret = builder.CreateAlloca(val_t);

                                llvm_if_then_else(
                                    s, builder.CreateICmpEQ(order, builder.getInt32(0)),
                                    [&]() {
                                        // Order 0, store the num/param.
                                        builder.CreateStore(
                                            taylor_c_diff_numparam_codegen(s, fp_t, v, terms + k, par_ptr, batch_size),
                                            ret);
                                    },
                                    [&]() {
                                        // Order 2 or higher, store zero.
                                        builder.CreateStore(
                                            vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), ret);
                                    });

                                auto *val = builder.CreateLoad(val_t, ret);

                                return llvm_fmul(s, val, val);
                            } else {
                                // LCOV_EXCL_START
                                throw std::invalid_argument(
                                    "An invalid argument type was encountered while trying to build the "
                                    "Taylor derivative of a sum of squares in compact mode");
                                // LCOV_EXCL_STOP
                            }
                        },
                        sf.args()[k].value());

                    // Compute the Taylor derivative for the current argument.
                    tmp.push_back(llvm_fadd(s, acc2, ex_term));
                }

                // Return the pairwise sum.
                builder.CreateStore(pairwise_sum(s, tmp), retval);
            });

        // Create the return value.
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
            // LCOV_EXCL_START
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of sum_sq() in compact mode detected");
            // LCOV_EXCL_STOP
        }
    }

    return f;
}

} // namespace

llvm::Function *sum_sq_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                std::uint32_t batch_size, bool) const
{
    return sum_sq_taylor_c_diff_func_impl(s, fp_t, *this, n_uvars, batch_size);
}

} // namespace detail

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::sum_sq_impl)
