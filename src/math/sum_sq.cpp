// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
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

sum_sq_impl::sum_sq_impl() : sum_sq_impl(std::vector<expression>{}) {}

sum_sq_impl::sum_sq_impl(std::vector<expression> v) : func_base("sum_sq", std::move(v)) {}

void sum_sq_impl::to_stream(std::ostream &os) const
{
    if (args().size() == 1u) {
        // NOTE: avoid brackets if there's only 1 argument.
        os << args()[0] << "**2";
    } else {
        os << '(';

        for (decltype(args().size()) i = 0; i < args().size(); ++i) {
            os << args()[i] << "**2";
            if (i != args().size() - 1u) {
                os << " + ";
            }
        }

        os << ')';
    }
}

template <typename T>
expression sum_sq_impl::diff_impl(std::unordered_map<const void *, expression> &func_map, const T &x) const
{
    std::vector<expression> terms;
    terms.reserve(args().size());

    for (const auto &arg : args()) {
        terms.push_back(arg * detail::diff(func_map, arg, x));
    }

    return 2_dbl * sum(std::move(terms));
}

expression sum_sq_impl::diff(std::unordered_map<const void *, expression> &func_map, const std::string &s) const
{
    return diff_impl(func_map, s);
}

expression sum_sq_impl::diff(std::unordered_map<const void *, expression> &func_map, const param &p) const
{
    return diff_impl(func_map, p);
}

namespace
{

template <typename T>
llvm::Value *sum_sq_taylor_diff_impl(llvm_state &s, const sum_sq_impl &sf, const std::vector<std::uint32_t> &deps,
                                     const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                     std::uint32_t order, std::uint32_t batch_size)
{
    // NOTE: this is prevented in the implementation
    // of the sum_sq() function.
    assert(!sf.args().empty());

    if (!deps.empty()) {
        // LCOV_EXCL_START
        throw std::invalid_argument("The vector of hidden dependencies in the Taylor diff for a sum of squares "
                                    "should be empty, but instead it has a size of {}"_format(deps.size()));
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

                        v_sums[k].push_back(builder.CreateFMul(v0, v1));
                    } else if constexpr (is_num_param_v<type>) {
                        // Number/param.

                        // NOTE: for number/params, all terms in the summation
                        // will be zero. Thus, ensure that v_sums[k] just
                        // contains a single zero.
                        if (v_sums[k].empty()) {
                            v_sums[k].push_back(vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
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
            tmp.push_back(pairwise_sum(builder, v_sum));
        }

        // Sum the sums.
        pairwise_sum(builder, tmp);

        // Multiply by 2 and return.
        return builder.CreateFAdd(tmp[0], tmp[0]);
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
                        return builder.CreateFMul(val, val);
                    } else if constexpr (is_num_param_v<type>) {
                        // Number/param.
                        if (order == 0u) {
                            auto val = taylor_codegen_numparam<T>(s, v, par_ptr, batch_size);
                            return builder.CreateFMul(val, val);
                        } else {
                            return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
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
                auto p_sum = pairwise_sum(builder, v_sums[k]);
                // Muliply the pairwise sum by 2.
                p_sum = builder.CreateFAdd(p_sum, p_sum);
                // Add it to the term outside the sum.
                tmp.back() = builder.CreateFAdd(p_sum, tmp.back());
            }
        }

        // Sum the sums and return.
        return pairwise_sum(builder, tmp);
    }
}

} // namespace

llvm::Value *sum_sq_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                          std::uint32_t batch_size, bool) const
{
    return sum_sq_taylor_diff_impl<double>(s, *this, deps, arr, par_ptr, n_uvars, order, batch_size);
}

llvm::Value *sum_sq_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                           const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                           std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                           std::uint32_t batch_size, bool) const
{
    return sum_sq_taylor_diff_impl<long double>(s, *this, deps, arr, par_ptr, n_uvars, order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *sum_sq_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                           const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                           std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                           std::uint32_t batch_size, bool) const
{
    return sum_sq_taylor_diff_impl<mppp::real128>(s, *this, deps, arr, par_ptr, n_uvars, order, batch_size);
}

#endif

} // namespace detail

expression sum_sq(std::vector<expression> args, std::uint32_t split)
{
    if (split < 2u) {
        throw std::invalid_argument(
            "The 'split' value for a sum of squares must be at least 2, but it is {} instead"_format(split));
    }

    // Partition args so that all zeroes are at the end.
    const auto n_end_it = std::stable_partition(args.begin(), args.end(), [](const expression &ex) {
        return !std::holds_alternative<number>(ex.value()) || !is_zero(std::get<number>(ex.value()));
    });

    // If we have one or more zeroes, eliminate them
    args.erase(n_end_it, args.end());

    // Special cases.
    if (args.empty()) {
        return 0_dbl;
    }

    if (args.size() == 1u) {
        return square(std::move(args[0]));
    }

    // NOTE: ret_seq will contain a sequence
    // of sum_sqs each containing 'split' terms.
    // tmp is a temporary vector
    // used to accumulate the arguments to each
    // sum_sq in ret_seq.
    std::vector<expression> ret_seq, tmp;
    for (auto &arg : args) {
        // LCOV_EXCL_START
#if !defined(NDEBUG)
        // NOTE: there cannot be zero numbers here because
        // we removed them.
        if (auto nptr = std::get_if<number>(&arg.value()); nptr && is_zero(*nptr)) {
            assert(false);
        }
#endif
        // LCOV_EXCL_STOP

        tmp.push_back(std::move(arg));
        if (tmp.size() == split) {
            // NOTE: after the move, tmp is guaranteed to be empty.
            ret_seq.emplace_back(func{detail::sum_sq_impl{std::move(tmp)}});
            assert(tmp.empty());
        }
    }

    // NOTE: tmp is not empty if 'split' does not divide
    // exactly args.size(). In such a case, we need to do the
    // last iteration manually.
    if (!tmp.empty()) {
        // NOTE: contrary to the previous loop, here we could
        // in principle end up creating a sum_sq_impl with only one
        // term. In such a case, for consistency with the general
        // behaviour of sum_sq({arg}), return arg*arg directly.
        if (tmp.size() == 1u) {
            ret_seq.emplace_back(square(std::move(tmp[0])));
        } else {
            ret_seq.emplace_back(func{detail::sum_sq_impl{std::move(tmp)}});
        }
    }

    // Perform a sum over the sum_sqs.
    return sum(std::move(ret_seq));
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::sum_sq_impl)
