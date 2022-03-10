// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/atanh.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

atanh_impl::atanh_impl(expression e) : func_base("atanh", std::vector{std::move(e)}) {}

atanh_impl::atanh_impl() : atanh_impl(0_dbl) {}

expression atanh_impl::diff(std::unordered_map<const void *, expression> &func_map, const std::string &s) const
{
    assert(args().size() == 1u);
    return detail::diff(func_map, args()[0], s) / (1_dbl - square(args()[0]));
}

expression atanh_impl::diff(std::unordered_map<const void *, expression> &func_map, const param &p) const
{
    assert(args().size() == 1u);
    return detail::diff(func_map, args()[0], p) / (1_dbl - square(args()[0]));
}

llvm::Value *atanh_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    if (auto vec_t = llvm::dyn_cast<llvm_vector_type>(args[0]->getType())) {
        if (const auto sfn = sleef_function_name(s.context(), "atanh", vec_t->getElementType(),
                                                 boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
            !sfn.empty()) {
            return llvm_invoke_external(
                s, sfn, vec_t, args,
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
        }
    }

    return call_extern_vec(s, args[0], "atanh");
}

llvm::Value *atanh_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return call_extern_vec(s, args[0],
#if defined(_MSC_VER)
                           // NOTE: it seems like the MSVC stdlib does not have an atanhl function,
                           // because LLVM complains about the symbol "atanhl" not being
                           // defined. Hence, use our own wrapper instead.
                           "heyoka_atanhl"
#else
                           "atanhl"
#endif
    );
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *atanh_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return call_extern_vec(s, args[0], "atanhq");
}

#endif

double atanh_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return std::atanh(heyoka::eval_dbl(args()[0], map, pars));
}

long double atanh_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                  const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return std::atanh(heyoka::eval_ldbl(args()[0], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 atanh_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                    const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return mppp::atanh(heyoka::eval_f128(args()[0], map, pars));
}
#endif

taylor_dc_t::size_type atanh_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Append arg * arg.
    u_vars_defs.emplace_back(square(args()[0]), std::vector<std::uint32_t>{});

    // Append the atanh decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed atanh).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Derivative of atanh(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_atanh_impl(llvm_state &s, const atanh_impl &f, const std::vector<std::uint32_t> &,
                                    const U &num, const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                    std::uint32_t, std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return codegen_from_values<T>(s, f, {taylor_codegen_numparam<T>(s, num, par_ptr, batch_size)});
    } else {
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    }
}

template <typename T>
llvm::Value *taylor_diff_atanh_impl(llvm_state &s, const atanh_impl &f, const std::vector<std::uint32_t> &deps,
                                    const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the index of the variable argument.
    const auto b_idx = uname_to_index(var.name());

    if (order == 0u) {
        return codegen_from_values<T>(s, f, {taylor_fetch_diff(arr, b_idx, 0, n_uvars)});
    }

    // Create the constant 1 in fp format.
    auto one_fp = vector_splat(builder, codegen<T>(s, number{1.}), batch_size);

    if (order == 1u) {
        // Special-case the first-order derivative, in order
        // to avoid an empty summation below.
        return builder.CreateFDiv(taylor_fetch_diff(arr, b_idx, 1, n_uvars),
                                  builder.CreateFSub(one_fp, taylor_fetch_diff(arr, deps[0], 0, n_uvars)));
    }

    // Create the fp version of the order.
    auto ord_fp = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);

    // Assemble the first part of the result: n*b^[n].
    auto ret = builder.CreateFMul(ord_fp, taylor_fetch_diff(arr, b_idx, order, n_uvars));

    // Compute n*(1 - c^[0]).
    auto n_c0_p1 = builder.CreateFMul(ord_fp, builder.CreateFSub(one_fp, taylor_fetch_diff(arr, deps[0], 0, n_uvars)));

    // NOTE: iteration in the [1, order) range.
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j < order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is var * var.
        auto cnj = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);
        auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

        // Add j*cnj*aj to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(cnj, aj)));
    }

    // Update ret.
    ret = builder.CreateFAdd(ret, pairwise_sum(builder, sum));

    // Divide by n*(1 - c^[0]) and return.
    return builder.CreateFDiv(ret, n_c0_p1);
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_atanh_impl(llvm_state &, const atanh_impl &, const std::vector<std::uint32_t> &, const U &,
                                    const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                    std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an inverse hyperbolic tangent");
}

template <typename T>
llvm::Value *taylor_diff_atanh(llvm_state &s, const atanh_impl &f, const std::vector<std::uint32_t> &deps,
                               const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                               std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        throw std::invalid_argument(
            "A hidden dependency vector of size 1 is expected in order to compute the Taylor "
            "derivative of the inverse hyperbolic tangent, but a vector of size {} was passed instead"_format(
                deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_atanh_impl<T>(s, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *atanh_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size, bool) const
{
    return taylor_diff_atanh<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *atanh_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                          std::uint32_t batch_size, bool) const
{
    return taylor_diff_atanh<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *atanh_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                          std::uint32_t batch_size, bool) const
{
    return taylor_diff_atanh<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

namespace
{

// Derivative of atanh(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_atanh_impl(llvm_state &s, const atanh_impl &fn, const U &num, std::uint32_t n_uvars,
                                              std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num_det<T>(s, fn, num, n_uvars, batch_size, "atanh", 1);
}

// Derivative of atanh(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_atanh_impl(llvm_state &s, const atanh_impl &fn, const variable &var,
                                              std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, "atanh", n_uvars, batch_size, {var}, 1);
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
        auto a_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto b_idx = f->args().begin() + 5;
        auto c_idx = f->args().begin() + 6;

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
                builder.CreateStore(codegen_from_values<T>(
                                        s, fn, {taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), b_idx)}),
                                    retval);
            },
            [&]() {
                // Create the constant 1 in fp format.
                auto one_fp = vector_splat(builder, codegen<T>(s, number{1.}), batch_size);

                // Compute the fp version of the order.
                auto ord_fp = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Compute n*b^[n].
                auto ret = builder.CreateFMul(ord_fp, taylor_c_load_diff(s, diff_ptr, n_uvars, ord, b_idx));

                // Compute n*(1 - c^[0]).
                auto n_c0_p1 = builder.CreateFMul(
                    ord_fp,
                    builder.CreateFSub(one_fp, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), c_idx)));

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto c_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, a_idx);

                    auto fac = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(val_t, acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(c_nj, aj))),
                                        acc);
                });

                // Update ret.
                ret = builder.CreateFAdd(ret, builder.CreateLoad(val_t, acc));

                // Divide by n*(1 - c^[0]).
                ret = builder.CreateFDiv(ret, n_c0_p1);

                // Store into retval.
                builder.CreateStore(ret, retval);
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
                "Inconsistent function signature for the Taylor derivative of the inverse hyperbolic tangent "
                "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_atanh_impl(llvm_state &, const atanh_impl &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an inverse hyperbolic tangent in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_atanh(llvm_state &s, const atanh_impl &fn, std::uint32_t n_uvars,
                                         std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_atanh_impl<T>(s, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *atanh_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                   bool) const
{
    return taylor_c_diff_func_atanh<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *atanh_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                    bool) const
{
    return taylor_c_diff_func_atanh<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *atanh_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                    bool) const
{
    return taylor_c_diff_func_atanh<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

} // namespace detail

expression atanh(expression e)
{
    return expression{func{detail::atanh_impl(std::move(e))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::atanh_impl)
