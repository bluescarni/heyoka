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

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/tan.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

tan_impl::tan_impl(expression e) : func_base("tan", std::vector{std::move(e)}) {}

tan_impl::tan_impl() : tan_impl(0_dbl) {}

namespace
{

// Generic implementation for the codegen of tan that will invoke the external
// function fname, after the decomposition of the input argument arg into scalars.
llvm::Value *tan_codegen_impl(llvm_state &s, llvm::Value *arg, const std::string &fname)
{
    auto &builder = s.builder();

    // Decompose the argument into scalars.
    auto scalars = vector_to_scalars(builder, arg);

    // Invoke the function on each scalar.
    std::vector<llvm::Value *> retvals;
    for (auto scal : scalars) {
        retvals.push_back(llvm_invoke_external(
            s, fname, scal->getType(), {scal},
            // NOTE: in theory we may add ReadNone here as well,
            // but for some reason, at least up to LLVM 10,
            // this causes strange codegen issues. Revisit
            // in the future.
            {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, retvals);
}

} // namespace

llvm::Value *tan_impl::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(args[0]->getType())) {
        if (const auto sfn = sleef_function_name(s.context(), "tan", vec_t->getElementType(),
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

    return tan_codegen_impl(s, args[0], "tan");
}

llvm::Value *tan_impl::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return tan_codegen_impl(s, args[0], "tanl");
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *tan_impl::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &args) const
{
    assert(args.size() == 1u);
    assert(args[0] != nullptr);

    return tan_codegen_impl(s, args[0], "heyoka_tan128");
}

#endif

double tan_impl::eval_dbl(const std::unordered_map<std::string, double> &map, const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    return std::tan(heyoka::eval_dbl(args()[0], map, pars));
}

void tan_impl::eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &map,
                              const std::vector<double> &pars) const
{
    assert(args().size() == 1u);

    heyoka::eval_batch_dbl(out, args()[0], map, pars);
    for (auto &el : out) {
        el = std::tan(el);
    }
}

double tan_impl::eval_num_dbl(const std::vector<double> &a) const
{
    if (a.size() != 1u) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Inconsistent number of arguments when computing the numerical value of the "
            "tangent over doubles (1 argument was expected, but {} arguments were provided"_format(a.size()));
    }

    return std::tan(a[0]);
}

double tan_impl::deval_num_dbl(const std::vector<double> &a, std::vector<double>::size_type i) const
{
    if (a.size() != 1u || i != 0u) {
        throw std::invalid_argument("Inconsistent number of arguments or derivative requested when computing the "
                                    "numerical derivative of the tangent");
    }

    return std::tan(a[0]);
}

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
tan_impl::taylor_decompose(std::vector<std::pair<expression, std::vector<std::uint32_t>>> &u_vars_defs) &&
{
    assert(args().size() == 1u);

    // Decompose the argument.
    auto &arg = *get_mutable_args_it().first;
    if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
        arg = expression{variable{"u_" + detail::li_to_string(dres)}};
    }

    // Append the tan decomposition.
    u_vars_defs.emplace_back(tan(std::move(arg)), std::vector<std::uint32_t>{});

    // Append the auxiliary function tan(arg) * tan(arg).
    u_vars_defs.emplace_back(square(expression{variable{"u_" + detail::li_to_string(u_vars_defs.size() - 1u)}}),
                             std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 2)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 1u));

    return u_vars_defs.size() - 2u;
}

namespace
{

// Derivative of tan(number).
template <typename T, typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_tan_impl(llvm_state &s, const tan_impl &f, const std::vector<std::uint32_t> &, const U &num,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        return codegen_from_values<T>(s, f, {taylor_codegen_numparam<T>(s, num, par_ptr, batch_size)});
    } else {
        return vector_splat(s.builder(), codegen<T>(s, number{0.}), batch_size);
    }
}

template <typename T>
llvm::Value *taylor_diff_tan_impl(llvm_state &s, const tan_impl &f, const std::vector<std::uint32_t> &deps,
                                  const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    if (order == 0u) {
        return codegen_from_values<T>(s, f, {taylor_fetch_diff(arr, u_idx, 0, n_uvars)});
    }

    // NOTE: iteration in the [0, order) range
    // (i.e., order excluded).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        // NOTE: the only hidden dependency contains the index of the
        // u variable whose definition is tan(var) * tan(var).
        auto bnj = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
        auto cj = taylor_fetch_diff(arr, deps[0], j, n_uvars);

        auto fac = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order - j))), batch_size);

        // Add (n-j)*bnj*cj to the sum.
        sum.push_back(builder.CreateFMul(fac, builder.CreateFMul(bnj, cj)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Divide by order.
    ret_acc
        = builder.CreateFDiv(ret_acc, vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size));

    // Create and return the result.
    return builder.CreateFAdd(taylor_fetch_diff(arr, u_idx, order, n_uvars), ret_acc);
}

// All the other cases.
template <typename T, typename U, std::enable_if_t<!is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_tan_impl(llvm_state &, const tan_impl &, const std::vector<std::uint32_t> &, const U &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of a tangent");
}

template <typename T>
llvm::Value *taylor_diff_tan(llvm_state &s, const tan_impl &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    assert(f.args().size() == 1u);

    if (deps.size() != 1u) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "A hidden dependency vector of size 1 is expected in order to compute the Taylor "
            "derivative of the tangent, but a vector of size {} was passed instead"_format(deps.size()));
    }

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_tan_impl<T>(s, f, deps, v, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        f.args()[0].value());
}

} // namespace

llvm::Value *tan_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                       const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                       std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                       std::uint32_t batch_size) const
{
    return taylor_diff_tan<double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *tan_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size) const
{
    return taylor_diff_tan<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *tan_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                        const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr,
                                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                        std::uint32_t batch_size) const
{
    return taylor_diff_tan<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, idx, batch_size);
}

#endif

} // namespace detail

expression tan(expression e)
{
    return expression{func{detail::tan_impl(std::move(e))}};
}

} // namespace heyoka
