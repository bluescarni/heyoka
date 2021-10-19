// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>

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
