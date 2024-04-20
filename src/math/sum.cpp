// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/sub.hpp>
#include <heyoka/detail/sum_sq.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#include <heyoka/detail/logging_impl.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

sum_impl::sum_impl() : sum_impl(std::vector<expression>{}) {}

sum_impl::sum_impl(std::vector<expression> v) : func_base("sum", std::move(v)) {}

void sum_impl::to_stream(std::ostringstream &oss) const
{
    if (args().empty()) {
        stream_expression(oss, 0_dbl);
        return;
    }

    if (args().size() == 1u) {
        stream_expression(oss, args()[0]);
        return;
    }

    // Partition the arguments so that all terms which are either a negative
    // number or a product in the form cf * a * ..., with cf a negative number,
    // are at the end.
    const auto fpart = [](const expression &arg) {
        if (const auto *num_ptr = std::get_if<number>(&arg.value())) {
            return !is_negative(*num_ptr);
        }

        if (const auto &fptr = std::get_if<func>(&arg.value());
            fptr != nullptr && fptr->extract<prod_impl>() != nullptr && fptr->args().size() >= 2u
            && std::holds_alternative<number>(fptr->args()[0].value())) {
            return !is_negative(std::get<number>(fptr->args()[0].value()));
        }

        return true;
    };

    auto terms = args();
    const auto neg_it = std::stable_partition(terms.begin(), terms.end(), fpart);

    // Helper to stream the positive terms.
    auto stream_pos_terms = [&]() {
        // Must have some positive terms.
        assert(neg_it != terms.begin());

        for (auto it = terms.begin(); it != neg_it; ++it) {
            stream_expression(oss, *it);

            if (it + 1 != neg_it) {
                oss << " + ";
            }
        }
    };

    // Helper to stream the negative terms.
    auto stream_neg_terms = [&]() {
        // Must have some negative terms.
        assert(neg_it != terms.end());

        auto it = neg_it;

        if (it == terms.begin()) {
            // If all terms are negative, handle
            // specially the first one: print *it with its
            // own leading '-' sign, and add a trailing
            // " - " for the next term.
            stream_expression(oss, *it);
            // NOTE: 'it' points at the beginning of terms,
            // and we know that there are at least 2 terms. Thus,
            // we are certain there is always at least one more term
            // and that we always need another " - ".
            oss << " - ";
            ++it;
        }

        for (; it != terms.end(); ++it) {
            if (const auto *num_ptr = std::get_if<number>(&it->value())) {
                stream_expression(oss, expression{-*num_ptr});
            } else {
                const auto &pfunc = std::get<func>(it->value());
                assert(pfunc.extract<prod_impl>() != nullptr);

                // Negate the first (constant negative) term of the product.
                auto new_prod_args = pfunc.args();
                assert(new_prod_args.size() >= 2u);
                new_prod_args[0] = expression{-std::get<number>(new_prod_args[0].value())};

                stream_expression(oss, prod(new_prod_args));
            }

            if (it + 1 != terms.end()) {
                oss << " - ";
            }
        }
    };

    oss << '(';

    if (neg_it == terms.begin()) {
        // The sum consists only of negative terms.
        stream_neg_terms();
    } else if (neg_it == terms.end()) {
        // The sum consists only of positive terms.
        stream_pos_terms();
    } else {
        // There are both positive and negative terms.
        stream_pos_terms();
        oss << " - ";
        stream_neg_terms();
    }

    oss << ')';
}

std::vector<expression> sum_impl::gradient() const
{
    return {args().size(), 1_dbl};
}

namespace
{

llvm::Value *sum_llvm_eval_impl(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr, llvm::Value *stride,
                                std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_eval_helper(
        [&s](std::vector<llvm::Value *> args, bool) -> llvm::Value * { return pairwise_sum(s, args); }, fb, s, fp_t,
        eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

} // namespace

llvm::Value *sum_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    return sum_llvm_eval_impl(s, fp_t, *this, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *sum_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                              std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "sum", [&s](std::vector<llvm::Value *> args, bool) { return pairwise_sum(s, args); }, fb, s, fp_t, batch_size,
        high_accuracy);
}

} // namespace

llvm::Function *sum_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    return sum_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

namespace
{

llvm::Value *sum_taylor_diff_impl(llvm_state &s, llvm::Type *fp_t, const sum_impl &sf,
                                  const std::vector<std::uint32_t> &deps, const std::vector<llvm::Value *> &arr,
                                  llvm::Value *par_ptr, std::uint32_t n_uvars,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  std::uint32_t order, std::uint32_t batch_size)
{
    // NOTE: this is prevented in the implementation
    // of the sum() function.
    assert(!sf.args().empty());

    if (!deps.empty()) {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("The vector of hidden dependencies in the Taylor diff for a sum "
                                                "should be empty, but instead it has a size of {}",
                                                deps.size()));
        // LCOV_EXCL_STOP
    }

    auto &builder = s.builder();

    // Load all values to be summed in local variables and
    // do a pairwise summation.
    std::vector<llvm::Value *> vals;
    vals.reserve(static_cast<decltype(vals.size())>(sf.args().size()));
    for (const auto &arg : sf.args()) {
        std::visit(
            [&](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    // Variable.
                    vals.push_back(taylor_fetch_diff(arr, uname_to_index(v.name()), order, n_uvars));
                } else if constexpr (is_num_param_v<type>) {
                    // Number/param.
                    if (order == 0u) {
                        vals.push_back(taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size));
                    } else {
                        vals.push_back(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size));
                    }
                } else {
                    // LCOV_EXCL_START
                    throw std::invalid_argument("An invalid argument type was encountered while trying to build the "
                                                "Taylor derivative of a sum");
                    // LCOV_EXCL_STOP
                }
            },
            arg.value());
    }

    return pairwise_sum(s, vals);
}

} // namespace

llvm::Value *sum_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                   bool) const
{
    return sum_taylor_diff_impl(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, batch_size);
}

namespace
{

llvm::Function *sum_taylor_c_diff_func_impl(llvm_state &s, llvm::Type *fp_t, const sum_impl &sf, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
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
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "sum", n_uvars, batch_size, nm_args);
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

        // Load all values to be summed in local variables and
        // do a pairwise summation.
        std::vector<llvm::Value *> vals;
        vals.reserve(static_cast<decltype(vals.size())>(sf.args().size()));
        for (decltype(sf.args().size()) i = 0; i < sf.args().size(); ++i) {
            vals.push_back(std::visit(
                [&](const auto &v) -> llvm::Value * {
                    using type = detail::uncvref_t<decltype(v)>;

                    if constexpr (std::is_same_v<type, variable>) {
                        return taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, terms + i);
                    } else if constexpr (is_num_param_v<type>) {
                        // Create the return value.
                        auto *retval = builder.CreateAlloca(val_t);

                        llvm_if_then_else(
                            s, builder.CreateICmpEQ(order, builder.getInt32(0)),
                            [&]() {
                                // If the order is zero, run the codegen.
                                builder.CreateStore(
                                    taylor_c_diff_numparam_codegen(s, fp_t, v, terms + i, par_ptr, batch_size), retval);
                            },
                            [&]() {
                                // Otherwise, return zero.
                                builder.CreateStore(
                                    vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), retval);
                            });

                        return builder.CreateLoad(val_t, retval);
                    } else {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            "An invalid argument type was encountered while trying to build the "
                            "Taylor derivative of a sum");
                        // LCOV_EXCL_STOP
                    }
                },
                sf.args()[i].value()));
        }

        builder.CreateRet(pairwise_sum(s, vals));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

} // namespace

llvm::Function *sum_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    return sum_taylor_c_diff_func_impl(s, fp_t, *this, n_uvars, batch_size);
}

// Helper to split the input sum 'e' into nested sums, each
// of which will have at most 'split' arguments.
// If 'e' is not a sum, or if it is a sum with no more than
// 'split' terms, 'e' will be returned unmodified.
// NOTE: 'e' is assumed to be a function.
// NOLINTNEXTLINE(misc-no-recursion)
expression sum_split(const expression &e, std::uint32_t split)
{
    assert(split >= 2u);
    assert(std::holds_alternative<func>(e.value()));

    const auto *sum_ptr = std::get<func>(e.value()).extract<sum_impl>();

    // NOTE: return 'e' unchanged if it is not a sum,
    // or if it is a sum that does not need to be split.
    // The latter condition is also used to terminate the
    // recursion.
    if (sum_ptr == nullptr || sum_ptr->args().size() <= split) {
        return e;
    }

    // NOTE: ret_seq will be a list
    // of sums each containing 'split' terms.
    // tmp is a temporary vector
    // used to accumulate the arguments for each
    // sum in ret_seq.
    std::vector<expression> ret_seq, tmp;
    for (const auto &arg : sum_ptr->args()) {
        tmp.push_back(arg);

        if (tmp.size() == split) {
            ret_seq.emplace_back(func{detail::sum_impl{std::move(tmp)}});

            // NOTE: tmp is practically guaranteed to be empty, but let's
            // be paranoid.
            tmp.clear();
        }
    }

    // NOTE: tmp is not empty if 'split' does not divide
    // exactly sum_ptr->args().size(). In such a case, we need to do the
    // last iteration manually.
    if (!tmp.empty()) {
        // NOTE: contrary to the previous loop, here we could
        // in principle end up creating a sum_impl with only one
        // term. In such a case, for consistency with the general
        // behaviour of sum({arg}), return arg directly.
        if (tmp.size() == 1u) {
            ret_seq.push_back(std::move(tmp[0]));
        } else {
            ret_seq.emplace_back(func{detail::sum_impl{std::move(tmp)}});
        }
    }

    // Recurse to split further, if needed.
    return sum_split(expression{func{detail::sum_impl{std::move(ret_seq)}}}, split);
}

namespace
{

// Check if ex is pow(something, 2). If it is, return
// a pointer to something, otherwise return nullptr.
const expression *is_square(const expression &ex)
{
    const auto *fptr = std::get_if<func>(&ex.value());

    if (fptr == nullptr) {
        // Not a function.
        return nullptr;
    }

    const auto *pow_ptr = fptr->extract<pow_impl>();

    if (pow_ptr == nullptr) {
        // Not a pow().
        return nullptr;
    }

    assert(pow_ptr->args().size() == 2u);

    const auto &base = pow_ptr->args()[0];
    const auto &expo = pow_ptr->args()[1];

    if (const auto *expo_num_ptr = std::get_if<number>(&expo.value())) {
        // The exponent is a number.
        if (std::visit([](const auto &v) { return v == 2; }, expo_num_ptr->value())) {
            // Exponent is 2, return the base.
            return &base;
        } else {
            // Exponent is not 2.
            return nullptr;
        }
    } else {
        // Exponent is not a number.
        return nullptr;
    }
}

} // namespace

// Transform the input sum 'e' into a sum of squares, if possible. If not,
// 'e' will be returned unchanged.
// NOTE: 'e' is assumed to be a function.
expression sum_to_sum_sq(const expression &e)
{
    assert(std::holds_alternative<func>(e.value()));

    const auto *sum_ptr = std::get<func>(e.value()).extract<sum_impl>();

    if (sum_ptr == nullptr) {
        // 'e' is not a sum.
        return e;
    }

    // 'e' is a sum, check if all arguments are squares.
    std::vector<expression> new_args;
    new_args.reserve(sum_ptr->args().size());

    for (const auto &arg : sum_ptr->args()) {
        const auto *square_arg = is_square(arg);

        if (square_arg == nullptr) {
            // A non-square argument was encountered, return.
            return e;
        } else {
            new_args.push_back(*square_arg);
        }
    }

    return expression{func{sum_sq_impl{std::move(new_args)}}};
}

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
expression sum_to_sub_impl(funcptr_map<expression> &func_map, const expression &ex)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&](const auto &v) {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, func>) {
                const auto *f_id = v.get_ptr();

                // Check if we already handled ex.
                if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                // Recursively transform sums into subs
                // in the arguments.
                std::vector<expression> new_args;
                new_args.reserve(v.args().size());
                for (const auto &orig_arg : v.args()) {
                    new_args.push_back(sum_to_sub_impl(func_map, orig_arg));
                }

                // Prepare the return value.
                std::optional<expression> retval;

                if (v.template extract<sum_impl>() == nullptr) {
                    // The current function is not a sum(). Just create
                    // a copy of it with the new args.
                    retval.emplace(v.copy(new_args));
                } else {
                    // The current function is a sum(). Partition its
                    // arguments so that those in the form -1 * a * ...
                    // are at the end.
                    const auto fpart = [](const expression &arg) {
                        if (const auto &fptr = std::get_if<func>(&arg.value());
                            fptr != nullptr && fptr->extract<prod_impl>() != nullptr && fptr->args().size() >= 2u
                            && std::holds_alternative<number>(fptr->args()[0].value())) {
                            return !is_negative_one(std::get<number>(fptr->args()[0].value()));
                        }

                        return true;
                    };

                    const auto it = std::stable_partition(new_args.begin(), new_args.end(), fpart);

                    if (it == new_args.end()) {
                        // There are no negations in the sum, just make a copy.
                        retval.emplace(v.copy(new_args));
                    } else {
                        // There are some negations in the sum.
                        // Group them into a subtrahend, negate, and transform
                        // into a subtraction.

                        // Construct the terms of the subtrahend.
                        std::vector<expression> sub_args, tmp_args;
                        for (auto s_it = it; s_it != new_args.end(); ++s_it) {
                            const auto &fn = std::get<func>(s_it->value());

                            assert(fn.template extract<prod_impl>() != nullptr);
                            assert(fn.args().size() >= 2u);
                            assert(is_negative_one(std::get<number>(fn.args()[0].value())));

                            // Build tmp_args from fn.args(), skipping the first term (-1).
                            tmp_args.clear();
                            tmp_args.reserve(fn.args().size() - 1u);
                            tmp_args.insert(tmp_args.end(), fn.args().begin() + 1, fn.args().end());

                            // Reconstruct the negated product.
                            sub_args.push_back(prod(tmp_args));
                        }

                        // Construct the subtrahend.
                        auto st = sum(sub_args);

                        if (it == new_args.begin()) {
                            // There are *only* negations, return the negation of st.
                            retval.emplace(prod({-1_dbl, std::move(st)}));
                        } else {
                            // Construct the minuend.
                            new_args.erase(it, new_args.end());
                            auto mend = sum(new_args);

                            // Construct the return value.
                            retval.emplace(sub(std::move(mend), std::move(st)));
                        }
                    }
                }

                // Put the return value into the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, *retval);
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return std::move(*retval);
            } else {
                return ex;
            }
        },
        ex.value());
}

} // namespace

// This function will transform sums containing negated terms
// (i.e., terms of the form -1 * a * ...) into subtractions. E.g.,
// (x + -1 * y) will be transformed into (x - y), thereby shaving
// away the cost of a multiplication.
std::vector<expression> sum_to_sub(const std::vector<expression> &v_ex)
{
    funcptr_map<expression> func_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(sum_to_sub_impl(func_map, e));
    }

    return retval;
}

} // namespace detail

expression sum(std::vector<expression> args)
{
    // Partition args so that all numbers are at the end.
    const auto n_end_it = std::stable_partition(
        args.begin(), args.end(), [](const expression &ex) { return !std::holds_alternative<number>(ex.value()); });

    // Constant fold the numbers.
    if (n_end_it != args.end()) {
        for (auto it = n_end_it + 1; it != args.end(); ++it) {
            *n_end_it = expression{std::get<number>(n_end_it->value()) + std::get<number>(it->value())};
        }

        // Remove all numbers but the first one.
        args.erase(n_end_it + 1, args.end());

        // Handle the special case in which the remaining number is zero.
        if (is_zero(std::get<number>(n_end_it->value()))) {
            if (args.size() == 1u) {
                assert(n_end_it == args.begin());

                // This is also the only remaining term in the sum,
                // return it.
                // NOTE: it is important to special-case this, because otherwise
                // we will fall into the args().empty() special case below, which will
                // forcibly convert the folded 0 constant into double precision.
                return *n_end_it;
            } else {
                // Besides the number 0, there are other
                // non-number terms in the sum. Remove
                // the number 0.
                args.pop_back();
            }
        }
    }

    // Special cases.
    if (args.empty()) {
        return 0_dbl;
    }

    if (args.size() == 1u) {
        return std::move(args[0]);
    }

    // Re-partition args so that all numbers are at the beginning.
    // NOTE: this results in a semi-canonical representation of sums
    // in which numbers are at the beginning.
    std::stable_partition(args.begin(), args.end(),
                          [](const expression &ex) { return std::holds_alternative<number>(ex.value()); });

    return expression{func{detail::sum_impl{std::move(args)}}};
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::sum_impl)
