// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>
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
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/erf.hpp>
#include <heyoka/math/exp.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

erf_impl::erf_impl(expression e) : func_base("erf", std::vector{std::move(e)}) {}

erf_impl::erf_impl() : erf_impl(0_dbl) {}

llvm::Value *erf_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
        if (const auto sfn = sleef_function_name(s.context(), "erf", vec_t->getElementType(),
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

    return call_extern_vec(s, args[0], "erf");
}

llvm::Value *erf_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return call_extern_vec(s, args[0],
#if defined(_MSC_VER)
                           // NOTE: it seems like the MSVC stdlib does not have an erfl function,
                           // because LLVM complains about the symbol "erfl" not being
                           // defined. Hence, use our own wrapper instead.
                           "heyoka_erfl"
#else
                           "erfl"
#endif
    );
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *erf_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return call_extern_vec(s, args[0], "heyoka_erf128");
}

#endif

double erf_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return std::erf(heyoka::eval_dbl(args()[0], map, pars));
}

long double erf_impl::eval_ldbl(const std::unordered_map<std::string, long double> &map, const std::vector<long double> &pars) const
{
    assert(args().size() == 1u);

    return std::erf(heyoka::eval_ldbl(args()[0], map, pars));
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 erf_impl::eval_f128(const std::unordered_map<std::string, mppp::real128> &map, const std::vector<mppp::real128> &pars) const
{
    assert(args().size() == 1u);

    return mppp::erf(heyoka::eval_f128(args()[0], map, pars));
}
#endif

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
erf_impl::taylor_decompose(std::vector<std::pair<expression, std::vector<std::uint32_t>>> &u_vars_defs) &&
{
    assert(args().size() == 1u);

    using namespace fmt::literals;

    // Decompose the argument.
    auto &arg = *get_mutable_args_it().first;
    if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
        arg = expression{"u_{}"_format(dres)};
    }

    // Append the erf decomposition.
    u_vars_defs.emplace_back(erf(arg), std::vector<std::uint32_t>{});

    // Append arg * arg.
    u_vars_defs.emplace_back(square(std::move(arg)), std::vector<std::uint32_t>{});

    // Append - arg * arg.
    u_vars_defs.emplace_back(-expression{"u_{}"_format(u_vars_defs.size() - 1u)}, std::vector<std::uint32_t>{});

    // Append exp( - arg * arg).
    u_vars_defs.emplace_back(exp(expression{"u_{}"_format(u_vars_defs.size() - 1u)}), std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 4)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));

    return u_vars_defs.size() - 4u;
}

namespace
{

// Variable template for the constant sqrt(pi) / 2 at different levels of precision.
template <typename T>
const auto sqrt_pi_2 = std::sqrt(boost::math::constants::pi<T>()) / 2;

#if defined(HEYOKA_HAVE_REAL128)

template <>
const mppp::real128 sqrt_pi_2<mppp::real128> = mppp::sqrt(mppp::pi_128) / 2;

#endif

// Derivative of erf(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_erf_impl(llvm_state &s, const erf_impl &f, const std::vector<std::uint32_t> &, const U &num,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return codegen_from_values<T>(s, f, {taylor_codegen_numparam<T>(s, num, par_ptr, batch_size)});
    } else {
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    }
}

// Derivative of erf(variable).
template <typename T>
llvm::Value *taylor_diff_erf_impl(llvm_state &s, const erf_impl &f, const std::vector<std::uint32_t> &deps,
                                  const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    assert(deps.size() == 1u);

    auto &builder = s.builder();

    // Fetch the index of the variable argument.
    const auto b_idx = uname_to_index(var.name());

    if (order == 0u) {
        return codegen_from_values<T>(s, f, {taylor_fetch_diff(arr, b_idx, 0, n_uvars)});
    }

    // NOTE: iteration in the [1, order] range.
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is exp(- var * var).
        auto bj = taylor_fetch_diff(arr, b_idx, j, n_uvars);
        auto cnj = taylor_fetch_diff(arr, deps[0], order - j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(j))), batch_size);

        // Add j*cnj*bj to the sum vector.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(cnj, bj)));
    }

    // Create ret with the sum.
    auto ret = pairwise_sum(builder, sum);

    // Generate the factor n * sqrt(pi) / 2.
    auto fac = number(order * sqrt_pi_2<T>);
    auto fac_s = vector_splat(builder, codegen<T>(s, fac), batch_size);

    // Multiply and return.
    return builder.CreateFDiv(ret, fac_s);
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_erf_impl(llvm_state &, const erf_impl &, const std::vector<std::uint32_t> &, const U &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of an error function");
}

template <typename T>
llvm::Value *taylor_diff_erf(llvm_state &s, const erf_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "A hidden dependency vector of size 1 is expected in order to compute the Taylor "
            "derivative of the error function, but a vector of size {} was passed instead"_format(deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_erf_impl<T>(s, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *erf_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                       const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                       std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                       std::uint32_t batch_size) const
{
    return taylor_diff_erf<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *erf_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size) const
{
    return taylor_diff_erf<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *erf_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size) const
{
    return taylor_diff_erf<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

namespace
{

// Derivative of erf(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_erf_impl(llvm_state &s, const erf_impl &fn, const U &num, std::uint32_t,
                                            std::uint32_t batch_size)
{
    using namespace fmt::literals;

    return taylor_c_diff_func_unary_num_det<T>(
        s, fn, num, batch_size,
        "heyoka_taylor_diff_erf_{}_{}"_format(taylor_c_diff_numparam_mangle(num),
                                              taylor_mangle_suffix(to_llvm_vector_type<T>(s.context(), batch_size))),
        "the error function", 1);
}

// Derivative of erf(variable).
template <typename T>
llvm::Function *taylor_c_diff_func_erf_impl(llvm_state &s, const erf_impl &fn, const variable &, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the function name.
    using namespace fmt::literals;
    const auto fname = "heyoka_taylor_diff_erf_var_{}_n_uvars_{}"_format(taylor_mangle_suffix(val_t), n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - par ptr,
    // - time ptr,
    // - idx of the var argument,
    // - idx of the uvar whose definition is exp(- var * var).
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

                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    auto c_nj = taylor_c_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), c_idx);
                    auto bj = taylor_c_load_diff(s, diff_ptr, n_uvars, j, b_idx);

                    auto fac = vector_splat(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);

                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(acc),
                                                           builder.CreateFMul(fac, builder.CreateFMul(c_nj, bj))),
                                        acc);
                });

                // Generate the factor n * sqrt(pi) /2
                auto t1_llvm = vector_splat(builder, codegen<T>(s, number(sqrt_pi_2<T>)), batch_size);
                auto fac = builder.CreateFMul(t1_llvm, ord_fp);

                // Store into retval.
                builder.CreateStore(builder.CreateFDiv(builder.CreateLoad(acc), fac), retval);
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
                "Inconsistent function signature for the Taylor derivative of the error function "
                "in compact mode detected");
        }
    }

    return f;
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_erf_impl(llvm_state &, const erf_impl &, const U &, std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of the error function in compact mode");
}

template <typename T>
llvm::Function *taylor_c_diff_func_erf(llvm_state &s, const erf_impl &fn, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    assert(fn.args().size() == 1u);

    return std::visit([&](const auto &v) { return taylor_c_diff_func_erf_impl<T>(s, fn, v, n_uvars, batch_size); },
                      fn.args()[0].value());
}

} // namespace

llvm::Function *erf_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    return taylor_c_diff_func_erf<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *erf_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    return taylor_c_diff_func_erf<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *erf_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    return taylor_c_diff_func_erf<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

expression erf_impl::diff(const std::string &s) const
{
    assert(args().size() == 1u);
#if defined(HEYOKA_HAVE_REAL128)
    auto coeff = heyoka::expression(heyoka::number(1./sqrt_pi_2<mppp::real128>));
#else
    auto coeff = heyoka::expression(heyoka::number(1./sqrt_pi_2<long double>));
#endif
    return coeff * exp(-args()[0]*args()[0]) * heyoka::diff(args()[0], s);
}

} // namespace detail

expression erf(expression e)
{
    return expression{func{detail::erf_impl(std::move(e))}};
}

} // namespace heyoka
