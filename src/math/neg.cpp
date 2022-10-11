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
#include <heyoka/math/neg.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

neg_impl::neg_impl(expression e) : func_base("neg", std::vector{std::move(e)}) {}

neg_impl::neg_impl() : neg_impl(0_dbl) {}

void neg_impl::to_stream(std::ostream &os) const
{
    assert(!args().empty());

    os << '-' << args()[0];
}

// Derivative.
std::vector<expression> neg_impl::gradient() const
{
    assert(args().size() == 1u);
    return {-1_dbl};
}

double neg_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return -heyoka::eval_dbl(args()[0], map, pars);
}

long double neg_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return -heyoka::eval_ldbl(args()[0], map, pars);
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 neg_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                  const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return -heyoka::eval_f128(args()[0], map, pars);
}
#endif

llvm::Value *neg_impl::llvm_eval_dbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                     llvm::Value *stride, std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<double>([&s](const std::vector<llvm::Value *> &args, bool) { return llvm_neg(s, args[0]); },
                                    *this, s, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Value *neg_impl::llvm_eval_ldbl(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                      llvm::Value *stride, std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<long double>(
        [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_neg(s, args[0]); }, *this, s, eval_arr,
        par_ptr, stride, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *neg_impl::llvm_eval_f128(llvm_state &s, const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr,
                                      llvm::Value *stride, std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper<mppp::real128>(
        [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_neg(s, args[0]); }, *this, s, eval_arr,
        par_ptr, stride, batch_size, high_accuracy);
}

#endif

namespace
{

template <typename T>
[[nodiscard]] llvm::Function *neg_llvm_c_eval(llvm_state &s, const func_base &fb, std::uint32_t batch_size,
                                              bool high_accuracy)
{
    return llvm_c_eval_func_helper<T>(
        "neg", [&s](const std::vector<llvm::Value *> &args, bool) { return llvm_neg(s, args[0]); }, fb, s, batch_size,
        high_accuracy);
}

} // namespace

llvm::Function *neg_impl::llvm_c_eval_func_dbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return neg_llvm_c_eval<double>(s, *this, batch_size, high_accuracy);
}

llvm::Function *neg_impl::llvm_c_eval_func_ldbl(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return neg_llvm_c_eval<long double>(s, *this, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *neg_impl::llvm_c_eval_func_f128(llvm_state &s, std::uint32_t batch_size, bool high_accuracy) const
{
    return neg_llvm_c_eval<mppp::real128>(s, *this, batch_size, high_accuracy);
}

#endif

namespace
{

// Derivative of neg(number).
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_neg_impl(llvm_state &s, llvm::Type *fp_t, const neg_impl &, const U &num,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return s.builder().CreateFNeg(taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of neg(variable).
llvm::Value *taylor_diff_neg_impl(llvm_state &s, llvm::Type *, const neg_impl &, const variable &var,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *, std::uint32_t n_uvars,
                                  std::uint32_t order, std::uint32_t, std::uint32_t)
{
    return s.builder().CreateFNeg(taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars));
}

// All the other cases.
template <typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_neg_impl(llvm_state &, llvm::Type *, const neg_impl &, const U &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a negation");
}

llvm::Value *taylor_diff_neg(llvm_state &s, llvm::Type *fp_t, const neg_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (!deps.empty()) {
        throw std::invalid_argument(
            fmt::format("An empty hidden dependency vector is expected in order to compute the Taylor "
                        "derivative of the negation, but a vector of size {} was passed "
                        "instead",
                        deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_neg_impl(s, fp_t, f, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *neg_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool) const
{
    return taylor_diff_neg(s, fp_t, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

namespace
{

// Derivative of neg(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_neg_impl(llvm_state &s, const neg_impl &, const U &num, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
{
    auto *fp_t = to_llvm_type<T>(s.context());

    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "neg", 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_neg(s, args[0]);
        },
        num);
}

// Derivative of neg(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_neg_impl(llvm_state &s, const neg_impl &, const variable &var, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the scalar and vector floating-point types.
    auto *fp_t = to_llvm_type<T>(context);
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "neg", n_uvars, batch_size, {var});
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

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateFNeg(taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, var_idx));

        // Return the result.
        builder.CreateRet(retval);

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
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of the negation "
                                        "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_neg_impl(llvm_state &, const neg_impl &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a negation in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_neg(llvm_state &s, const neg_impl &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_neg_impl<T>(s, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *neg_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                 bool) const
{
    return taylor_c_diff_func_neg<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *neg_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                  bool) const
{
    return taylor_c_diff_func_neg<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *neg_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                  bool) const
{
    return taylor_c_diff_func_neg<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

// Small helper to detect if an expression
// is a neg function. Mutable so we can extract
// the function arguments from the return value.
func *is_neg(expression &ex)
{
    if (auto func_ptr = std::get_if<func>(&ex.value());
        func_ptr != nullptr && func_ptr->extract<neg_impl>() != nullptr) {
        return func_ptr;
    } else {
        return nullptr;
    }
}

} // namespace detail

expression neg(expression e)
{
    return expression{func{detail::neg_impl(std::move(e))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::neg_impl)
