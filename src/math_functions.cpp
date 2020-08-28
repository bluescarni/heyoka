// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Helper to run the Taylor init phase of a unary
// function in scalar or vector format.
template <typename T>
llvm::Value *taylor_init_batch_unary_func(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                          std::uint32_t batch_size, std::uint32_t vector_size)
{
    if (vector_size == 0u) {
        // Scalar case.

        // Create the function argument. The codegen for the argument
        // comes from taylor_init_batch().
        std::vector<llvm::Value *> args_v{taylor_init_batch<T>(s, f.args()[0], arr, batch_idx, batch_size, 0)};
        assert(args_v[0] != nullptr);

        return function_codegen_from_values<T>(s, f, args_v);
    } else {
        // Vector case.
        auto &builder = s.builder();

        // Create the function argument in vector form.
        auto vec = taylor_init_batch<T>(s, f.args()[0], arr, batch_idx, batch_size, vector_size);

        // Decompose the vector into a set of scalar values.
        auto scalars = vector_to_scalars(builder, vec);

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> init_vals;
        for (auto scal : scalars) {
            init_vals.push_back(function_codegen_from_values<T>(s, f, {scal}));
        }

        // Build a vector with the results.
        return scalars_to_vector(builder, init_vals);
    }
}

template <typename T>
llvm::Value *taylor_init_batch_sin(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the sine (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_init_batch_unary_func<T>(s, f, arr, batch_idx, batch_size, vector_size);
}

// Derivative of sin(number).
template <typename T>
llvm::Value *taylor_diff_batch_sin_impl(llvm_state &s, std::uint32_t, const number &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    return taylor_diff_batch_zero<T>(s, vector_size);
}

// Derivative of sin(variable).
template <typename T>
llvm::Value *taylor_diff_batch_sin_impl(llvm_state &s, std::uint32_t idx, const variable &var, std::uint32_t order,
                                        std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                        std::uint32_t batch_size, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of sin() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // The indices for accessing the derivatives in this loop iteration:
        // - (order - j) * n_uvars * batch_size + (idx + 1) * batch_size + batch_idx,
        // - j * n_uvars * batch_size + u_idx * batch_size + batch_idx.
        // NOTE: the +1 is because we are accessing the cosine
        // of the u var, which is conventionally placed
        // right after the sine in the decomposition.
        auto arr_ptr0
            = builder.CreateInBoundsGEP(diff_arr,
                                        {builder.getInt32(0), builder.getInt32((order - j) * n_uvars * batch_size
                                                                               + (idx + 1u) * batch_size + batch_idx)},
                                        "sin_ptr");
        auto arr_ptr1 = builder.CreateInBoundsGEP(
            diff_arr,
            {builder.getInt32(0), builder.getInt32(j * n_uvars * batch_size + u_idx * batch_size + batch_idx)},
            "sin_ptr");

        // Load the values.
        auto v0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "sin_load")
                                      : load_vector_from_memory(builder, arr_ptr0, vector_size, "sin_load");
        auto v1 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr1, "sin_load")
                                      : load_vector_from_memory(builder, arr_ptr1, vector_size, "sin_load");

        auto fac = codegen<T>(s, number(static_cast<T>(j)));
        if (vector_size > 0u) {
            fac = create_constant_vector(builder, fac, vector_size);
        }

        // Add j*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = llvm_pairwise_sum(builder, sum);

    // Compute and return the result: ret_acc / order
    // NOTE: worthwhile to replace division with multiplication
    // by inverse or better let LLVM do it?
    auto div = codegen<T>(s, number(static_cast<T>(order)));
    if (vector_size > 0u) {
        div = create_constant_vector(builder, div, vector_size);
    }

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_batch_sin_impl(llvm_state &, std::uint32_t, const U &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a sine");
}

template <typename T>
llvm::Value *taylor_diff_batch_sin(llvm_state &s, const function &func, std::uint32_t idx, std::uint32_t order,
                                   std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size,
                                   const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sine (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_batch_sin_impl<T>(s, idx, v, order, n_uvars, diff_arr, batch_idx, batch_size,
                                                 vector_size, cd_uvars);
        },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression sin(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.sin";
    fc.name_ldbl() = "llvm.sin";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_sin128";
#endif
    fc.display_name() = "sin";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the sine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return cos(args[0]) * diff(args[0], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the sine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sin(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the sine in batches (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::sin(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "sine over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sin(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing "
                                        "the derivative of std::sin");
        }

        return std::cos(args[0]);
    };
    // NOTE: for sine/cosine we need a non-default decomposition because
    // we always need both sine *and* cosine in the decomposition
    // in order to compute the derivatives.
    fc.taylor_decompose_f() = [](function &&f, std::vector<expression> &u_vars_defs) {
        if (f.args().size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the Taylor decomposition of "
                                        "the sine (1 argument was expected, but "
                                        + std::to_string(f.args().size()) + " arguments were provided");
        }

        // Decompose the argument.
        auto &arg = f.args()[0];
        if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
            arg = expression{variable{"u_" + detail::li_to_string(dres)}};
        }

        // Save a copy of the decomposed argument.
        auto f_arg = arg;

        // Append the sine decomposition.
        u_vars_defs.emplace_back(std::move(f));

        // Compute the return value (pointing to the
        // decomposed sine).
        const auto retval = u_vars_defs.size() - 1u;

        // Append the cosine decomposition.
        u_vars_defs.emplace_back(cos(std::move(f_arg)));

        return retval;
    };
    fc.taylor_init_batch_dbl_f() = detail::taylor_init_batch_sin<double>;
    fc.taylor_init_batch_ldbl_f() = detail::taylor_init_batch_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_init_batch_f128_f() = detail::taylor_init_batch_sin<mppp::real128>;
#endif
    fc.taylor_diff_batch_dbl_f() = detail::taylor_diff_batch_sin<double>;
    fc.taylor_diff_batch_ldbl_f() = detail::taylor_diff_batch_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_batch_f128_f() = detail::taylor_diff_batch_sin<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_init_batch_cos(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the cosine (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_init_batch_unary_func<T>(s, f, arr, batch_idx, batch_size, vector_size);
}

// Derivative of cos(number).
template <typename T>
llvm::Value *taylor_diff_batch_cos_impl(llvm_state &s, std::uint32_t, const number &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    return taylor_diff_batch_zero<T>(s, vector_size);
}

template <typename T>
llvm::Value *taylor_diff_batch_cos_impl(llvm_state &s, std::uint32_t idx, const variable &var, std::uint32_t order,
                                        std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                        std::uint32_t batch_size, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of cos() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // The indices for accessing the derivatives in this loop iteration:
        // - (order - j) * n_uvars * batch_size + (idx - 1) * batch_size + batch_idx,
        // - j * n_uvars * batch_size + u_idx * batch_size + batch_idx.
        // NOTE: the -1 is because we are accessing the sine
        // of the u var, which is conventionally placed
        // right before the cosine in the decomposition.
        auto arr_ptr0
            = builder.CreateInBoundsGEP(diff_arr,
                                        {builder.getInt32(0), builder.getInt32((order - j) * n_uvars * batch_size
                                                                               + (idx - 1u) * batch_size + batch_idx)},
                                        "cos_ptr");
        auto arr_ptr1 = builder.CreateInBoundsGEP(
            diff_arr,
            {builder.getInt32(0), builder.getInt32(j * n_uvars * batch_size + u_idx * batch_size + batch_idx)},
            "cos_ptr");

        // Load the values.
        auto v0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "cos_load")
                                      : load_vector_from_memory(builder, arr_ptr0, vector_size, "cos_load");
        auto v1 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr1, "cos_load")
                                      : load_vector_from_memory(builder, arr_ptr1, vector_size, "cos_load");

        auto fac = codegen<T>(s, number(static_cast<T>(j)));
        if (vector_size > 0u) {
            fac = create_constant_vector(builder, fac, vector_size);
        }

        // Add j*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = llvm_pairwise_sum(builder, sum);

    // Compute and return the result: -ret_acc / order
    auto div = codegen<T>(s, number(-static_cast<T>(order)));
    if (vector_size > 0u) {
        div = create_constant_vector(builder, div, vector_size);
    }

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_batch_cos_impl(llvm_state &, std::uint32_t, const U &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a cosine");
}

template <typename T>
llvm::Value *taylor_diff_batch_cos(llvm_state &s, const function &func, std::uint32_t idx, std::uint32_t order,
                                   std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size,
                                   const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the cosine (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_batch_cos_impl<T>(s, idx, v, order, n_uvars, diff_arr, batch_idx, batch_size,
                                                 vector_size, cd_uvars);
        },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression cos(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.cos";
    fc.name_ldbl() = "llvm.cos";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_cos128";
#endif
    fc.display_name() = "cos";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when taking the derivative of the cosine (1 "
                                        "argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return -sin(args[0]) * diff(args[0], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the cosine from doubles (1 "
                                        "argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::cos(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the cosine in batches of "
                                        "doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::cos(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "cosine over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::cos(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments or derivative requested when computing the derivative of std::cos");
        }

        return -std::sin(args[0]);
    };
    fc.taylor_decompose_f() = [](function &&f, std::vector<expression> &u_vars_defs) {
        if (f.args().size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the Taylor decomposition of "
                                        "the cosine (1 argument was expected, but "
                                        + std::to_string(f.args().size()) + " arguments were provided");
        }

        // Decompose the argument.
        auto &arg = f.args()[0];
        if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
            arg = expression{variable{"u_" + detail::li_to_string(dres)}};
        }

        // Append the sine decomposition.
        u_vars_defs.emplace_back(sin(arg));

        // Append the cosine decomposition.
        u_vars_defs.emplace_back(std::move(f));

        return u_vars_defs.size() - 1u;
    };
    fc.taylor_init_batch_dbl_f() = detail::taylor_init_batch_cos<double>;
    fc.taylor_init_batch_ldbl_f() = detail::taylor_init_batch_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_init_batch_f128_f() = detail::taylor_init_batch_cos<mppp::real128>;
#endif
    fc.taylor_diff_batch_dbl_f() = detail::taylor_diff_batch_cos<double>;
    fc.taylor_diff_batch_ldbl_f() = detail::taylor_diff_batch_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_batch_f128_f() = detail::taylor_diff_batch_cos<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_init_batch_log(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the logarithm (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_init_batch_unary_func<T>(s, f, arr, batch_idx, batch_size, vector_size);
}

// Derivative of log(number).
template <typename T>
llvm::Value *taylor_diff_batch_log_impl(llvm_state &s, std::uint32_t, const number &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    return taylor_diff_batch_zero<T>(s, vector_size);
}

// Derivative of log(variable).
template <typename T>
llvm::Value *taylor_diff_batch_log_impl(llvm_state &s, std::uint32_t idx, const variable &var, std::uint32_t order,
                                        std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                        std::uint32_t batch_size, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    // NOTE: not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of log() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // The result of the summation.
    llvm::Value *ret_acc;

    // NOTE: iteration in the [1, order) range
    // (i.e., order excluded). If order is 1,
    // we need to special case as the pairwise
    // summation function requires a series
    // with at least 1 element.
    if (order > 1u) {
        std::vector<llvm::Value *> sum;
        for (std::uint32_t j = 1; j < order; ++j) {
            // The indices for accessing the derivatives in this loop iteration:
            // - (order - j) * n_uvars * batch_size + idx * batch_size + batch_idx,
            // - j * n_uvars * batch_size + u_idx * batch_size + batch_idx.
            auto arr_ptr0
                = builder.CreateInBoundsGEP(diff_arr,
                                            {builder.getInt32(0), builder.getInt32((order - j) * n_uvars * batch_size
                                                                                   + idx * batch_size + batch_idx)},
                                            "log_ptr");
            auto arr_ptr1 = builder.CreateInBoundsGEP(
                diff_arr,
                {builder.getInt32(0), builder.getInt32(j * n_uvars * batch_size + u_idx * batch_size + batch_idx)},
                "log_ptr");

            // Load the values.
            auto v0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "log_load")
                                          : load_vector_from_memory(builder, arr_ptr0, vector_size, "log_load");
            auto v1 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr1, "log_load")
                                          : load_vector_from_memory(builder, arr_ptr1, vector_size, "log_load");

            auto fac = codegen<T>(s, number(static_cast<T>(order - j)));
            if (vector_size > 0u) {
                fac = create_constant_vector(builder, fac, vector_size);
            }

            // Add (order-j)*v0*v1 to the sum.
            sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
        }

        // Compute the result of the summation.
        ret_acc = llvm_pairwise_sum(builder, sum);
    } else {
        // If the order is 1, the summation will be empty.
        // Init the result of the summation with zero.
        ret_acc = codegen<T>(s, number(static_cast<T>(0)));

        // Turn it into a vector if needed.
        if (vector_size > 0u) {
            ret_acc = create_constant_vector(builder, ret_acc, vector_size);
        }
    }

    // Finalise the return value: (b^[n] - ret_acc / n) / b^[0]
    auto arr_ptrn = builder.CreateInBoundsGEP(
        diff_arr,
        {builder.getInt32(0), builder.getInt32(order * n_uvars * batch_size + u_idx * batch_size + batch_idx)},
        "log_ptr");
    auto arr_ptr0 = builder.CreateInBoundsGEP(
        diff_arr, {builder.getInt32(0), builder.getInt32(u_idx * batch_size + batch_idx)}, "log_ptr");

    auto bn = (vector_size == 0u) ? builder.CreateLoad(arr_ptrn, "log_load")
                                  : load_vector_from_memory(builder, arr_ptrn, vector_size, "log_load");
    auto b0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "log_load")
                                  : load_vector_from_memory(builder, arr_ptr0, vector_size, "log_load");

    auto div = codegen<T>(s, number(static_cast<T>(order)));
    if (vector_size > 0u) {
        div = create_constant_vector(builder, div, vector_size);
    }

    return builder.CreateFDiv(builder.CreateFSub(bn, builder.CreateFDiv(ret_acc, div)), b0);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_batch_log_impl(llvm_state &, std::uint32_t, const U &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a logarithm");
}

template <typename T>
llvm::Value *taylor_diff_batch_log(llvm_state &s, const function &func, std::uint32_t idx, std::uint32_t order,
                                   std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size,
                                   const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the logarithm (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_batch_log_impl<T>(s, idx, v, order, n_uvars, diff_arr, batch_idx, batch_size,
                                                 vector_size, cd_uvars);
        },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression log(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.log";
    fc.name_ldbl() = "llvm.log";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_log128";
#endif
    fc.display_name() = "log";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the logarithm (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return expression{number(1.)} / args[0] * diff(args[0], s);
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the logarithm from doubles (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::log(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the logarithm in batches of "
                                        "doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::log(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "logarithm over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::log(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments or derivative requested when computing the derivative of std::log");
        }

        return 1. / args[0];
    };
    fc.taylor_init_batch_dbl_f() = detail::taylor_init_batch_log<double>;
    fc.taylor_init_batch_ldbl_f() = detail::taylor_init_batch_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_init_batch_f128_f() = detail::taylor_init_batch_log<mppp::real128>;
#endif
    fc.taylor_diff_batch_dbl_f() = detail::taylor_diff_batch_log<double>;
    fc.taylor_diff_batch_ldbl_f() = detail::taylor_diff_batch_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_batch_f128_f() = detail::taylor_diff_batch_log<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_init_batch_exp(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the exponential (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_init_batch_unary_func<T>(s, f, arr, batch_idx, batch_size, vector_size);
}

// Derivative of exp(number).
template <typename T>
llvm::Value *taylor_diff_batch_exp_impl(llvm_state &s, std::uint32_t, const number &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    return taylor_diff_batch_zero<T>(s, vector_size);
}

// Derivative of exp(variable).
template <typename T>
llvm::Value *taylor_diff_batch_exp_impl(llvm_state &s, std::uint32_t idx, const variable &var, std::uint32_t order,
                                        std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                        std::uint32_t batch_size, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of exp() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [0, order) range
    // (i.e., order excluded).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        // The indices for accessing the derivatives in this loop iteration:
        // - j * n_uvars * batch_size + idx * batch_size + batch_idx,
        // - (order - j) * n_uvars * batch_size + u_idx * batch_size + batch_idx.
        auto arr_ptr0 = builder.CreateInBoundsGEP(
            diff_arr, {builder.getInt32(0), builder.getInt32(j * n_uvars * batch_size + idx * batch_size + batch_idx)},
            "exp_ptr");
        auto arr_ptr1
            = builder.CreateInBoundsGEP(diff_arr,
                                        {builder.getInt32(0), builder.getInt32((order - j) * n_uvars * batch_size
                                                                               + u_idx * batch_size + batch_idx)},
                                        "exp_ptr");

        // Load the values.
        auto v0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "exp_load")
                                      : load_vector_from_memory(builder, arr_ptr0, vector_size, "exp_load");
        auto v1 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr1, "exp_load")
                                      : load_vector_from_memory(builder, arr_ptr1, vector_size, "exp_load");

        auto fac = codegen<T>(s, number(static_cast<T>(order - j)));
        if (vector_size > 0u) {
            fac = create_constant_vector(builder, fac, vector_size);
        }

        // Add (order-j)*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = llvm_pairwise_sum(builder, sum);

    // Finalise the return value: ret_acc / n.
    auto div = codegen<T>(s, number(static_cast<T>(order)));
    if (vector_size > 0u) {
        div = create_constant_vector(builder, div, vector_size);
    }

    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
llvm::Value *taylor_diff_batch_exp_impl(llvm_state &, std::uint32_t, const U &, std::uint32_t, std::uint32_t,
                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of an exponential");
}

template <typename T>
llvm::Value *taylor_diff_batch_exp(llvm_state &s, const function &func, std::uint32_t idx, std::uint32_t order,
                                   std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size,
                                   const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the exponential (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_batch_exp_impl<T>(s, idx, v, order, n_uvars, diff_arr, batch_idx, batch_size,
                                                 vector_size, cd_uvars);
        },
        func.args()[0].value());
}

} // namespace

} // namespace detail

expression exp(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.exp";
    fc.name_ldbl() = "llvm.exp";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_exp128";
#endif
    fc.display_name() = "exp";
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the exponential (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return exp(args[0]) * diff(args[0], s);
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the exponential from doubles (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::exp(eval_dbl(args[0], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the exponential in batches of "
                "doubles (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        eval_batch_dbl(out, args[0], map);
        for (auto &el : out) {
            el = std::exp(el);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "exponential over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::exp(args[0]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 1u || i != 0u) {
            throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                        "derivative of std::exp over doubles");
        }

        return std::exp(args[0]);
    };
    fc.taylor_init_batch_dbl_f() = detail::taylor_init_batch_exp<double>;
    fc.taylor_init_batch_ldbl_f() = detail::taylor_init_batch_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_init_batch_f128_f() = detail::taylor_init_batch_exp<mppp::real128>;
#endif
    fc.taylor_diff_batch_dbl_f() = detail::taylor_diff_batch_exp<double>;
    fc.taylor_diff_batch_ldbl_f() = detail::taylor_diff_batch_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_batch_f128_f() = detail::taylor_diff_batch_exp<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_init_batch_pow(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size)
{
    if (f.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the pow() function (2 arguments were expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    // Disable verification in the llvm
    // machinery when we are generating code
    // for the Taylor init phase. This is due
    // to the verification issue with the
    // pow intrinsic mangling.
    s.verify() = false;

    if (vector_size == 0u) {
        // Scalar case.

        // Create the function arguments. The codegen for the arguments
        // comes from taylor_init_batch.
        std::vector<llvm::Value *> args_v{taylor_init_batch<T>(s, f.args()[0], arr, batch_idx, batch_size, 0),
                                          taylor_init_batch<T>(s, f.args()[1], arr, batch_idx, batch_size, 0)};

        return function_codegen_from_values<T>(s, f, args_v);
    } else {
        // Vector case.
        auto &builder = s.builder();

        // Create the function arguments in vector form.
        auto vec0 = taylor_init_batch<T>(s, f.args()[0], arr, batch_idx, batch_size, vector_size);
        auto vec1 = taylor_init_batch<T>(s, f.args()[1], arr, batch_idx, batch_size, vector_size);

        // Decompose the vectors into sets of scalar values.
        auto scalars0 = vector_to_scalars(builder, vec0);
        auto scalars1 = vector_to_scalars(builder, vec1);
        assert(scalars0.size() == scalars1.size());

        // Invoke the function on each scalar.
        std::vector<llvm::Value *> init_vals;
        for (decltype(scalars0.size()) i = 0; i < scalars0.size(); ++i) {
            init_vals.push_back(function_codegen_from_values<T>(s, f, {scalars0[i], scalars1[i]}));
        }

        // Build a vector with the results.
        return scalars_to_vector(builder, init_vals);
    }
}

// Derivative of pow(number, number).
template <typename T>
llvm::Value *taylor_diff_batch_pow_impl(llvm_state &s, std::uint32_t, const number &, const number &, std::uint32_t,
                                        std::uint32_t, llvm::Value *, std::uint32_t, std::uint32_t,
                                        std::uint32_t vector_size, const std::unordered_map<std::uint32_t, number> &)
{
    return taylor_diff_batch_zero<T>(s, vector_size);
}

// Derivative of pow(variable, number).
template <typename T>
llvm::Value *taylor_diff_batch_pow_impl(llvm_state &s, std::uint32_t idx, const variable &var, const number &num,
                                        std::uint32_t order, std::uint32_t n_uvars, llvm::Value *diff_arr,
                                        std::uint32_t batch_idx, std::uint32_t batch_size, std::uint32_t vector_size,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of pow() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [0, order) range
    // (i.e., order *not* included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        // The indices for accessing the derivatives in this loop iteration:
        // - (order - j) * n_uvars * batch_size + u_idx * batch_size + batch_idx,
        // - j * n_uvars * batch_size + idx * batch_size + batch_idx.
        auto arr_ptr0
            = builder.CreateInBoundsGEP(diff_arr,
                                        {builder.getInt32(0), builder.getInt32((order - j) * n_uvars * batch_size
                                                                               + u_idx * batch_size + batch_idx)},
                                        "pow_ptr");
        auto arr_ptr1 = builder.CreateInBoundsGEP(
            diff_arr, {builder.getInt32(0), builder.getInt32(j * n_uvars * batch_size + idx * batch_size + batch_idx)},
            "pow_ptr");

        // Load the values.
        auto v0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "pow_load")
                                      : load_vector_from_memory(builder, arr_ptr0, vector_size, "pow_load");
        auto v1 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr1, "pow_load")
                                      : load_vector_from_memory(builder, arr_ptr1, vector_size, "pow_load");

        // Compute the scalar factor: order * num - j * (num + 1).
        auto scal_f = codegen<T>(s, number(static_cast<T>(order)) * num
                                        - number(static_cast<T>(j)) * (num + number(static_cast<T>(1))));
        if (vector_size > 0u) {
            scal_f = create_constant_vector(builder, scal_f, vector_size);
        }

        // Add scal_f*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(scal_f, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = llvm_pairwise_sum(builder, sum);

    // Compute the final divisor: order * (zero-th derivative of u_idx).
    auto ord_f = codegen<T>(s, number(static_cast<T>(order)));
    if (vector_size > 0u) {
        ord_f = create_constant_vector(builder, ord_f, vector_size);
    }

    auto arr_ptr0 = builder.CreateInBoundsGEP(
        diff_arr, {builder.getInt32(0), builder.getInt32(u_idx * batch_size + batch_idx)}, "pow_ptr");
    auto b0 = (vector_size == 0u) ? builder.CreateLoad(arr_ptr0, "pow_load")
                                  : load_vector_from_memory(builder, arr_ptr0, vector_size, "pow_load");

    auto div = builder.CreateFMul(ord_f, b0, "pow_div");

    // Compute and return the result: ret_acc / div.
    return builder.CreateFDiv(ret_acc, div);
}

// All the other cases.
template <typename T, typename U1, typename U2>
llvm::Value *taylor_diff_batch_pow_impl(llvm_state &, std::uint32_t, const U1 &, const U2 &, std::uint32_t,
                                        std::uint32_t, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                        const std::unordered_map<std::uint32_t, number> &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a pow()");
}

template <typename T>
llvm::Value *taylor_diff_batch_pow(llvm_state &s, const function &func, std::uint32_t idx, std::uint32_t order,
                                   std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size,
                                   const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    if (func.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "pow() (2 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_batch_pow_impl<T>(s, idx, v1, v2, order, n_uvars, diff_arr, batch_idx, batch_size,
                                                 vector_size, cd_uvars);
        },
        func.args()[0].value(), func.args()[1].value());
}

} // namespace

} // namespace detail

expression pow(expression e1, expression e2)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e1));
    args.emplace_back(std::move(e2));

    function fc{std::move(args)};
    fc.name_dbl() = "llvm.pow";
    fc.name_ldbl() = "llvm.pow";
#if defined(HEYOKA_HAVE_REAL128)
    fc.name_f128() = "heyoka_pow128";
#endif
    fc.display_name() = "pow";
    // Disable verification whenever
    // we codegen the pow() function, due
    // to what looks like an LLVM verification bug.
    fc.disable_verify() = true;
    fc.ty_dbl() = function::type::builtin;
    fc.ty_ldbl() = function::type::builtin;
#if defined(HEYOKA_HAVE_REAL128)
    fc.ty_f128() = function::type::external;
    // NOTE: in theory we may add ReadNone here as well,
    // but for some reason, at least up to LLVM 10,
    // this causes strange codegen issues. Revisit
    // in the future.
    fc.attributes_f128() = {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn};
#endif

    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 2u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the exponentiation (2 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        return args[1] * pow(args[0], args[1] - expression{number(1.)}) * diff(args[0], s)
               + pow(args[0], args[1]) * log(args[0]) * diff(args[1], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 2u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the exponentiation from doubles (2 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        return std::pow(eval_dbl(args[0], map), eval_dbl(args[1], map));
    };
    fc.eval_batch_dbl_f() = [](std::vector<double> &out, const std::vector<expression> &args,
                               const std::unordered_map<std::string, std::vector<double>> &map) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Inconsistent number of arguments when evaluating the exponentiation in "
                                        "batches of doubles (2 arguments were expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }
        auto out0 = out; // is this allocation needed?
        eval_batch_dbl(out0, args[0], map);
        eval_batch_dbl(out, args[1], map);
        for (decltype(out.size()) i = 0u; i < out.size(); ++i) {
            out[i] = std::pow(out0[i], out[i]);
        }
    };
    fc.eval_num_dbl_f() = [](const std::vector<double> &args) {
        if (args.size() != 2u) {
            throw std::invalid_argument("Inconsistent number of arguments when computing the numerical value of the "
                                        "exponentiation over doubles (1 argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return std::pow(args[0], args[1]);
    };
    fc.deval_num_dbl_f() = [](const std::vector<double> &args, std::vector<double>::size_type i) {
        if (args.size() != 2u || i > 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments or derivative requested when computing the derivative of std::pow");
        }
        return args[1] * std::pow(args[0], args[1] - 1.) + std::log(args[0]) * std::pow(args[0], args[1]);
    };
    fc.taylor_init_batch_dbl_f() = detail::taylor_init_batch_pow<double>;
    fc.taylor_init_batch_ldbl_f() = detail::taylor_init_batch_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_init_batch_f128_f() = detail::taylor_init_batch_pow<mppp::real128>;
#endif
    fc.taylor_diff_batch_dbl_f() = detail::taylor_diff_batch_pow<double>;
    fc.taylor_diff_batch_ldbl_f() = detail::taylor_diff_batch_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_batch_f128_f() = detail::taylor_diff_batch_pow<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
