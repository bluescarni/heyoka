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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/tfp.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Helper to run the Taylor init phase of a unary
// function.
template <typename T>
tfp taylor_u_init_unary_func(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                             bool high_accuracy)
{
    assert(f.args().size() == 1u);

    auto &builder = s.builder();

    // Do the initialisation for the function argument.
    auto arg = taylor_u_init<T>(s, f.args()[0], arr, batch_size, high_accuracy);

    // Decompose arg into scalars.
    auto scalars = vector_to_scalars(
        builder, std::visit(
                     [](const auto &a) {
                         if constexpr (std::is_same_v<detail::uncvref_t<decltype(a)>, llvm::Value *>) {
                             return a;
                         } else {
                             // NOTE: in high accuracy mode, return only
                             // the main component.
                             return a.first;
                         }
                     },
                     arg));

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> init_vals;
    for (auto scal : scalars) {
        init_vals.push_back(function_codegen_from_values<T>(s, f, {scal}));
    }

    // Build a vector with the results.
    auto ret = scalars_to_vector(builder, init_vals);

    // Turn it into a tfp and return it.
    return tfp_from_vector(s, ret, high_accuracy);
}

template <typename T>
tfp taylor_u_init_sin(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                      bool high_accuracy)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the sine (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size, high_accuracy);
}

// Derivative of sin(number).
template <typename T>
tfp taylor_diff_sin_impl(llvm_state &s, const number &, const std::vector<tfp> &, std::uint32_t, std::uint32_t,
                         std::uint32_t, std::uint32_t batch_size, bool high_accuracy)
{
    return tfp_zero<T>(s, batch_size, high_accuracy);
}

// Derivative of sin(variable).
template <typename T>
tfp taylor_diff_sin_impl(llvm_state &s, const variable &var, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                         std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of sin() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    if (order == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Overflow in the Taylor derivative of sin()");
    }
    std::vector<tfp> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the +1 is because we are accessing the cosine
        // of the u var, which is conventionally placed
        // right after the sine in the decomposition.
        auto v0 = taylor_load_derivative(arr, idx + 1u, order - j, n_uvars);
        auto v1 = taylor_load_derivative(arr, u_idx, j, n_uvars);

        auto fac = tfp_constant<T>(s, number(static_cast<T>(j)), batch_size, high_accuracy);

        // Add j*v0*v1 to the sum.
        sum.push_back(tfp_mul(s, fac, tfp_mul(s, v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = tfp_pairwise_sum(s, sum);

    // Compute and return the result: ret_acc / order
    auto div = tfp_constant<T>(s, number(static_cast<T>(order)), batch_size, high_accuracy);

    return tfp_div(s, ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
tfp taylor_diff_sin_impl(llvm_state &, const U &, const std::vector<tfp> &, std::uint32_t, std::uint32_t, std::uint32_t,
                         std::uint32_t, bool)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a sine");
}

template <typename T>
tfp taylor_diff_sin(llvm_state &s, const function &func, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                    std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the sine (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_sin_impl<T>(s, v, arr, n_uvars, order, idx, batch_size, high_accuracy);
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
#if 0
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_sin<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_sin<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_sin<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_sin<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_sin<mppp::real128>;
#endif
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
tfp taylor_u_init_cos(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                      bool high_accuracy)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the cosine (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size, high_accuracy);
}

// Derivative of cos(number).
template <typename T>
tfp taylor_diff_cos_impl(llvm_state &s, const number &, const std::vector<tfp> &, std::uint32_t, std::uint32_t,
                         std::uint32_t, std::uint32_t batch_size, bool high_accuracy)
{
    return tfp_zero<T>(s, batch_size, high_accuracy);
}

template <typename T>
tfp taylor_diff_cos_impl(llvm_state &s, const variable &var, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                         std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of cos() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [1, order] range
    // (i.e., order included).
    if (order == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Overflow in the Taylor derivative of cos()");
    }
    std::vector<tfp> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        // NOTE: the -1 is because we are accessing the sine
        // of the u var, which is conventionally placed
        // right before the cosine in the decomposition.
        auto v0 = taylor_load_derivative(arr, idx - 1u, order - j, n_uvars);
        auto v1 = taylor_load_derivative(arr, u_idx, j, n_uvars);

        auto fac = tfp_constant<T>(s, number(static_cast<T>(j)), batch_size, high_accuracy);

        // Add j*v0*v1 to the sum.
        sum.push_back(tfp_mul(s, fac, tfp_mul(s, v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = tfp_pairwise_sum(s, sum);

    // Compute and return the result: -ret_acc / order
    auto div = tfp_constant<T>(s, number(-static_cast<T>(order)), batch_size, high_accuracy);

    return tfp_div(s, ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
tfp taylor_diff_cos_impl(llvm_state &, const U &, const std::vector<tfp> &, std::uint32_t, std::uint32_t, std::uint32_t,
                         std::uint32_t, bool)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a cosine");
}

template <typename T>
tfp taylor_diff_cos(llvm_state &s, const function &func, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                    std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the cosine (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_cos_impl<T>(s, v, arr, n_uvars, order, idx, batch_size, high_accuracy);
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
#if 0
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_cos<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_cos<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_cos<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_cos<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_cos<mppp::real128>;
#endif
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
tfp taylor_u_init_log(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                      bool high_accuracy)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the logarithm (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size, high_accuracy);
}

// Derivative of log(number).
template <typename T>
tfp taylor_diff_log_impl(llvm_state &s, const number &, const std::vector<tfp> &, std::uint32_t, std::uint32_t,
                         std::uint32_t, std::uint32_t batch_size, bool high_accuracy)
{
    return tfp_zero<T>(s, batch_size, high_accuracy);
}

// Derivative of log(variable).
template <typename T>
tfp taylor_diff_log_impl(llvm_state &s, const variable &var, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                         std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    // NOTE: not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of log() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // The result of the summation.
    tfp ret_acc;

    // NOTE: iteration in the [1, order) range
    // (i.e., order excluded). If order is 1,
    // we need to special case as the pairwise
    // summation function requires a series
    // with at least 1 element.
    if (order > 1u) {
        std::vector<tfp> sum;
        for (std::uint32_t j = 1; j < order; ++j) {
            auto v0 = taylor_load_derivative(arr, idx, order - j, n_uvars);
            auto v1 = taylor_load_derivative(arr, u_idx, j, n_uvars);

            auto fac = tfp_constant<T>(s, number(static_cast<T>(order - j)), batch_size, high_accuracy);

            // Add (order-j)*v0*v1 to the sum.
            sum.push_back(tfp_mul(s, fac, tfp_mul(s, v0, v1)));
        }

        // Compute the result of the summation.
        ret_acc = tfp_pairwise_sum(s, sum);
    } else {
        // If the order is 1, the summation will be empty.
        // Init the result of the summation with zero.
        ret_acc = tfp_constant<T>(s, number(0.), batch_size, high_accuracy);
    }

    // Finalise the return value: (b^[n] - ret_acc / n) / b^[0]
    auto bn = taylor_load_derivative(arr, u_idx, order, n_uvars);
    auto b0 = taylor_load_derivative(arr, u_idx, 0, n_uvars);

    auto div = tfp_constant<T>(s, number(static_cast<T>(order)), batch_size, high_accuracy);

    return tfp_div(s, tfp_sub(s, bn, tfp_div(s, ret_acc, div)), b0);
}

// All the other cases.
template <typename T, typename U>
tfp taylor_diff_log_impl(llvm_state &, const U &, const std::vector<tfp> &, std::uint32_t, std::uint32_t, std::uint32_t,
                         std::uint32_t, bool)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a logarithm");
}

template <typename T>
tfp taylor_diff_log(llvm_state &s, const function &func, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                    std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the logarithm (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_log_impl<T>(s, v, arr, n_uvars, order, idx, batch_size, high_accuracy);
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
#if 0
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_log<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_log<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_log<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_log<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_log<mppp::real128>;
#endif
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
tfp taylor_u_init_exp(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                      bool high_accuracy)
{
    if (f.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the exponential (1 argument was expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    return taylor_u_init_unary_func<T>(s, f, arr, batch_size, high_accuracy);
}

// Derivative of exp(number).
template <typename T>
tfp taylor_diff_exp_impl(llvm_state &s, const number &, const std::vector<tfp> &, std::uint32_t, std::uint32_t,
                         std::uint32_t, std::uint32_t batch_size, bool high_accuracy)
{
    return tfp_zero<T>(s, batch_size, high_accuracy);
}

// Derivative of exp(variable).
template <typename T>
tfp taylor_diff_exp_impl(llvm_state &s, const variable &var, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                         std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of exp() (the order must be at least one)");
    }

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [0, order) range
    // (i.e., order excluded).
    std::vector<tfp> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        auto v0 = taylor_load_derivative(arr, idx, j, n_uvars);
        auto v1 = taylor_load_derivative(arr, u_idx, order - j, n_uvars);

        auto fac = tfp_constant<T>(s, number(static_cast<T>(order - j)), batch_size, high_accuracy);

        // Add (order-j)*v0*v1 to the sum.
        sum.push_back(tfp_mul(s, fac, tfp_mul(s, v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = tfp_pairwise_sum(s, sum);

    // Finalise the return value: ret_acc / n.
    auto div = tfp_constant<T>(s, number(static_cast<T>(order)), batch_size, high_accuracy);

    return tfp_div(s, ret_acc, div);
}

// All the other cases.
template <typename T, typename U>
tfp taylor_diff_exp_impl(llvm_state &, const U &, const std::vector<tfp> &, std::uint32_t, std::uint32_t, std::uint32_t,
                         std::uint32_t, bool)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of an exponential");
}

template <typename T>
tfp taylor_diff_exp(llvm_state &s, const function &func, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                    std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    if (func.args().size() != 1u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "the exponential (1 argument was expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_exp_impl<T>(s, v, arr, n_uvars, order, idx, batch_size, high_accuracy);
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
#if 0
    fc.taylor_u_init_dbl_f() = detail::taylor_u_init_exp<double>;
    fc.taylor_u_init_ldbl_f() = detail::taylor_u_init_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_u_init_f128_f() = detail::taylor_u_init_exp<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_exp<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_exp<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_exp<mppp::real128>;
#endif
#endif

    return expression{std::move(fc)};
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *taylor_init_pow(llvm_state &s, const function &f, llvm::Value *diff_arr, std::uint32_t batch_size,
                             bool high_accuracy)
{
    if (f.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor initialization phase for "
                                    "the pow() function (2 arguments were expected, but "
                                    + std::to_string(f.args().size()) + " arguments were provided");
    }

    auto &builder = s.builder();

    // Do the initialisation for the function arguments.
    auto arg0 = taylor_init<T>(s, f.args()[0], diff_arr, batch_size, high_accuracy);
    auto arg1 = taylor_init<T>(s, f.args()[1], diff_arr, batch_size, high_accuracy);

    // Decompose arg into scalars.
    auto scalars0 = vector_to_scalars(builder, arg0);
    auto scalars1 = vector_to_scalars(builder, arg1);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> init_vals;
    for (decltype(scalars0.size()) i = 0; i < scalars0.size(); ++i) {
        init_vals.push_back(function_codegen_from_values<T>(s, f, {scalars0[i], scalars1[i]}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, init_vals);
}

// Derivative of pow(number, number).
template <typename T>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, const number &, const number &, llvm::Value *, std::uint32_t,
                                  llvm::Value *, std::uint32_t, std::uint32_t batch_size, bool)
{
    return create_constant_vector(s.builder(), codegen<T>(s, number{0.}), batch_size);
}

// Derivative of pow(variable, number).
template <typename T>
llvm::Value *taylor_diff_pow_impl(llvm_state &s, const variable &var, const number &num, llvm::Value *diff_arr,
                                  std::uint32_t n_uvars, llvm::Value *order, std::uint32_t idx,
                                  std::uint32_t batch_size, bool)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Create the diff function name, appending the batch size in order to
    // avoid potential name clashes when multiple Taylor functions
    // with different vector sizes are added to the same module.
    const auto fname = "heyoka_taylor_diff_pow_" + li_to_string(batch_size);

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Prepare the function prototype. The arguments:
        // - indices of the variables,
        // - exponent,
        // - derivative order,
        // - diff array.
        auto vec_t = llvm::VectorType::get(to_llvm_type<T>(context), batch_size);
        std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                        to_llvm_type<T>(context), llvm::Type::getInt32Ty(context),
                                        llvm::PointerType::getUnqual(vec_t)};
        // The return type is vec_t.
        auto *ft = llvm::FunctionType::get(vec_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);

        // Fetch the function arguments.
        auto idx0 = f->args().begin();
        auto idx1 = f->args().begin() + 1;
        auto exponent = f->args().begin() + 2;
        auto ord = f->args().begin() + 3;
        auto diff_ptr = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        auto bb = llvm::BasicBlock::Create(context, "entry", f);
        builder.SetInsertPoint(bb);

        // Create FP vector versions of exponent and order.
        auto alpha_v = create_constant_vector(builder, exponent, batch_size);
        auto ord_v = create_constant_vector(builder, builder.CreateUIToFP(ord, to_llvm_type<T>(context)), batch_size);

        // Create the accumulator.
        auto acc = builder.CreateAlloca(vec_t);
        builder.CreateStore(create_constant_vector(builder, codegen<T>(s, number{0.}), batch_size), acc);

        // Run the loop.
        // TODO overflow check.
        llvm_loop_u32(s, f, builder.getInt32(0), ord, [&](llvm::Value *j) {
            auto b_nj = taylor_load_diff(s, diff_ptr, n_uvars, builder.CreateSub(ord, j), idx0);
            auto aj = taylor_load_diff(s, diff_ptr, n_uvars, j, idx1);

            // Compute the factor n*alpha-j*(alpha+1).
            auto j_v = create_constant_vector(builder, builder.CreateUIToFP(j, to_llvm_type<T>(context)), batch_size);
            auto fac = builder.CreateFSub(
                builder.CreateFMul(ord_v, alpha_v),
                builder.CreateFMul(
                    j_v, builder.CreateFAdd(alpha_v,
                                            create_constant_vector(builder, codegen<T>(s, number{1.}), batch_size))));

            builder.CreateStore(
                builder.CreateFAdd(builder.CreateLoad(acc), builder.CreateFMul(fac, builder.CreateFMul(b_nj, aj))),
                acc);
        });

        // Create the return value.
        builder.CreateRet(builder.CreateLoad(acc));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    // Invoke the function.
    llvm::Value *ret = builder.CreateCall(
        f, {builder.getInt32(uname_to_index(var.name())), builder.getInt32(idx), codegen<T>(s, num), order, diff_arr});

    // Finalize the result.
    auto order_v = create_constant_vector(builder, builder.CreateUIToFP(order, to_llvm_type<T>(context)), batch_size);
    return builder.CreateFDiv(
        ret, builder.CreateFMul(order_v, taylor_load_diff(s, diff_arr, n_uvars, builder.getInt32(0),
                                                          builder.getInt32(uname_to_index(var.name())))));
}

// All the other cases.
template <typename T, typename U1, typename U2>
llvm::Value *taylor_diff_pow_impl(llvm_state &, const U1 &, const U2 &, llvm::Value *, std::uint32_t, llvm::Value *,
                                  std::uint32_t, std::uint32_t, bool)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a pow()");
}

template <typename T>
llvm::Value *taylor_diff_pow(llvm_state &s, const function &func, llvm::Value *diff_arr, std::uint32_t n_uvars,
                             llvm::Value *order, std::uint32_t idx, std::uint32_t batch_size, bool high_accuracy)
{
    if (func.args().size() != 2u) {
        throw std::invalid_argument("Inconsistent number of arguments in the Taylor derivative for "
                                    "pow() (2 arguments were expected, but "
                                    + std::to_string(func.args().size()) + " arguments were provided");
    }

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_pow_impl<T>(s, v1, v2, diff_arr, n_uvars, order, idx, batch_size, high_accuracy);
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
    fc.taylor_init_dbl_f() = detail::taylor_init_pow<double>;
    fc.taylor_init_ldbl_f() = detail::taylor_init_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_init_f128_f() = detail::taylor_init_pow<mppp::real128>;
#endif
    fc.taylor_diff_dbl_f() = detail::taylor_diff_pow<double>;
    fc.taylor_diff_ldbl_f() = detail::taylor_diff_pow<long double>;
#if defined(HEYOKA_HAVE_REAL128)
    fc.taylor_diff_f128_f() = detail::taylor_diff_pow<mppp::real128>;
#endif

    return expression{std::move(fc)};
}

} // namespace heyoka
