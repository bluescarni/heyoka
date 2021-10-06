// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/asinh.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
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

asinh_impl::asinh_impl(expression e) : func_base("asinh", std::vector{std::move(e)}) {}

asinh_impl::asinh_impl() : asinh_impl(0_dbl) {}

std::vector<expression> asinh_impl::gradient() const
{
    assert(args().size() == 1u);
    return {pow(square(args()[0]) + 1_dbl, -.5)};
}

llvm::Value *asinh_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    if (auto vec_t = llvm::dyn_cast<llvm_vector_type>(args[0]->getType())) {
        if (const auto sfn = sleef_function_name(s.context(), "asinh", vec_t->getElementType(),
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

    return call_extern_vec(s, args[0], "asinh");
}

llvm::Value *asinh_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return call_extern_vec(s, args[0],
#if defined(_MSC_VER)
                           // NOTE: it seems like the MSVC stdlib does not have an asinhl function,
                           // because LLVM complains about the symbol "asinhl" not being
                           // defined. Hence, use our own wrapper instead.
                           "heyoka_asinhl"
#else
                           "asinhl"
#endif
    );
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *asinh_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return call_extern_vec(s, args[0], "asinhq");
}

#endif

double asinh_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return std::asinh(heyoka::eval_dbl(args()[0], map, pars));
}

long double asinh_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map,
                                  const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return std::asinh(heyoka::eval_ldbl(args()[0], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 asinh_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map,
                                    const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return mppp::asinh(heyoka::eval_f128(args()[0], map, pars));
}
#endif

taylor_dc_t::size_type asinh_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Append arg * arg.
    u_vars_defs.emplace_back(square(args()[0]), std::vector<std::uint32_t>{});

    // Append 1 + arg * arg.
    u_vars_defs.emplace_back(1_dbl + expression{"u_{}"_format(u_vars_defs.size() - 1u)}, std::vector<std::uint32_t>{});

    // Append sqrt(1 + arg * arg).
    u_vars_defs.emplace_back(sqrt(expression{"u_{}"_format(u_vars_defs.size() - 1u)}), std::vector<std::uint32_t>{});

    // Append the asinh decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed asinh).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Derivative of asinh(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_asinh_impl(llvm_state &s, const asinh_impl &f, const std::vector<std::uint32_t> &,
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
llvm::Value *taylor_diff_asinh_impl(llvm_state &s, const asinh_impl &f, const std::vector<std::uint32_t> &deps,
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

    if (order == 1u) {
        // Special-case the first-order derivative, in order
        // to avoid an empty summation below.
        return builder.CreateFDiv(taylor_fetch_diff(arr, b_idx, 1, n_uvars),
                                  taylor_fetch_diff(arr, deps[0], 0, n_uvars));
    }

    // Create the fp version of the order.
    auto ord_fp = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);

    // Assemble the first part of the result: n*b^[n].
    auto ret = builder.CreateFMul(ord_fp, taylor_fetch_diff(arr, b_idx, order, n_uvars));

    // Compute n*c^[0].
    auto n_c0 = builder.CreateFMul(ord_fp, taylor_fetch_diff(arr, deps[0], 0, n_uvars));

    // NOTE: iteration in the [1, order) range.
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j < order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is sqrt(1 + var * var).
        auto cnj = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);
        auto aj = taylor_fetch_diff(arr, idx, j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

        // Add j*cnj*aj to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(cnj, aj)));
    }

    // Update ret.
    ret = builder.CreateFSub(ret, pairwise_sum(builder, sum));

    // Divide by n*c^[0] and return.
    return builder.CreateFDiv(ret, n_c0);
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_asinh_impl(llvm_state &, const asinh_impl &, const std::vector<std::uint32_t> &, const U &,
                                    const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                    std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an inverse hyperbolic sine");
}

template <typename T>
llvm::Value *taylor_diff_asinh(llvm_state &s, const asinh_impl &f, const std::vector<std::uint32_t> &deps,
                               const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                               std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        throw std::invalid_argument(
            "A hidden dependency vector of size 1 is expected in order to compute the Taylor "
            "derivative of the inverse hyperbolic sine, but a vector of size {} was passed instead"_format(
                deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_asinh_impl<T>(s, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *asinh_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                         std::uint32_t batch_size, bool) const
{
    return taylor_diff_asinh<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *asinh_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                          std::uint32_t batch_size, bool) const
{
    return taylor_diff_asinh<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *asinh_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                          std::uint32_t batch_size, bool) const
{
    return taylor_diff_asinh<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

namespace
{

// Derivative of asinh(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_asinh_impl(llvm_state &s, const asinh_impl &fn, const U &num, std::uint32_t,
                                              std::uint32_t batch_size)
{
    return taylor_c_diff_func_unary_num_det<T>(
        s, fn, num, batch_size,
        "heyoka_taylor_diff_asinh_{}_{}"_format(taylor_c_diff_numparam_mangle(num),
                                                taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size))),
        "the inverse hyperbolic sine", 1);
}

// Derivative of asinh(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_asinh_impl(llvm_state &s, const asinh_impl &fn, const variable &,
                                              std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    const auto fname = "heyoka_taylor_diff_asinh_var_{}_n_uvars_{}"_format(taylor_mangle_suffix(val_t), n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - par ptr,
    // - time ptr,
    // - idx of the var argument,
    // - idx of the uvar whose definition is sqrt(1 + var * var).
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context),
                                    llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t),
                                    llvm::PointerType::getUnqual(to_llvm_type<T>(context)),
                                    llvm::PointerType::getUnqual(to_llvm_type<T>(context)),
                                    llvm::Type::getInt32Ty(context),
                                    llvm::Type::getInt32Ty(context)};

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
                // Compute the fp version of the order.
                auto ord_fp = vector_splat(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

                // Compute n*b^[n].
                auto ret = builder.CreateFMul(ord_fp, taylor_c_load_diff(s, diff_ptr, n_uvars, ord, b_idx));

                // Compute n*c^[0].
                auto n_c0
                    = builder.CreateFMul(ord_fp, taylor_c_load_diff(s, diff_ptr, n_uvars, builder.getInt32(0), c_idx));

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), ord, [&](llvm::Value *j) {
                    auto c_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto aj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, a_idx);

                    auto fac = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(c_nj, aj))),
                                        acc);
                });

                // Update ret.
                ret = builder.CreateFSub(ret, builder.CreateLoad(acc));

                // Divide by n*c^[0].
                ret = builder.CreateFDiv(ret, n_c0);

                // Store into retval.
                builder.CreateStore(ret, retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

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
                "Inconsistent function signature for the Taylor derivative of the inverse hyperbolic sine "
                "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_asinh_impl(llvm_state &, const asinh_impl &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an inverse hyperbolic sine in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_asinh(llvm_state &s, const asinh_impl &fn, std::uint32_t n_uvars,
                                         std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_asinh_impl<T>(s, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *asinh_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                   bool) const
{
    return taylor_c_diff_func_asinh<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *asinh_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                    bool) const
{
    return taylor_c_diff_func_asinh<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *asinh_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                    bool) const
{
    return taylor_c_diff_func_asinh<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

} // namespace detail

expression asinh(expression e)
{
    return expression{func{detail::asinh_impl(std::move(e))}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::asinh_impl)
