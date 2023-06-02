// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <sstream>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

prod_impl::prod_impl() : prod_impl(std::vector<expression>{}) {}

prod_impl::prod_impl(std::vector<expression> v) : func_base("prod", std::move(v)) {}

namespace
{

// If ex is a pow() function whose exponent is a negative
// number, then a pointer to the exponent will be returned. Otherwise,
// nullptr will be returned.
const number *ex_is_negative_pow(const expression &ex)
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

    const auto *n_exp_ptr = std::get_if<number>(&(pow_ptr->args()[1].value()));

    if (n_exp_ptr == nullptr) {
        // pow() exponent is not a number.
        return nullptr;
    }

    auto checker = [](const auto &v) {
        using std::isnan;

        return !isnan(v) && v < 0;
    };

    if (std::visit(checker, n_exp_ptr->value())) {
        return n_exp_ptr;
    } else {
        // pow() number exponent is either NaN or not negative.
        return nullptr;
    }
}

// Check if a product is a negation - that is, a product with at least
// 2 terms and whose only numerical coefficient is at the beginning with
// value -1.
bool prod_is_negation_impl(const prod_impl &p)
{
    const auto &args = p.args();

    if (args.size() < 2u) {
        return false;
    }

    if (const auto *num_ptr = std::get_if<number>(&args[0].value()); num_ptr != nullptr && is_negative_one(*num_ptr)) {
        return std::none_of(args.begin() + 1, args.end(),
                            [](const auto &ex) { return std::holds_alternative<number>(ex.value()); });
    } else {
        return false;
    }
}

} // namespace

// NOLINTNEXTLINE(misc-no-recursion)
void prod_impl::to_stream(std::ostringstream &oss) const
{
    // NOTE: prods which have 0 or 1 terms are not possible
    // when using the public API, but let's handle these special
    // cases anyway.
    if (args().empty()) {
        stream_expression(oss, 1_dbl);
        return;
    }

    if (args().size() == 1u) {
        stream_expression(oss, args()[0]);
        return;
    }

    // Special case for negation.
    if (prod_is_negation_impl(*this)) {
        oss << '-';
        prod_impl(std::vector(args().begin() + 1, args().end())).to_stream(oss);

        return;
    }

    // Partition the arguments so that pow()s with negative
    // exponents are at the end. These constitute the denominator
    // of the product.
    auto tmp_args = args();
    const auto den_it = std::stable_partition(tmp_args.begin(), tmp_args.end(),
                                              [](const auto &ex) { return ex_is_negative_pow(ex) == nullptr; });

    // Helper to stream the numerator of the product.
    auto stream_num = [&]() {
        assert(den_it != tmp_args.begin());

        // Is the numerator consisting of a single term?
        const auto single_num = (tmp_args.begin() + 1 == den_it);

        if (!single_num) {
            oss << '(';
        }

        for (auto it = tmp_args.begin(); it != den_it; ++it) {
            stream_expression(oss, *it);

            if (it + 1 != den_it) {
                oss << " * ";
            }
        }

        if (!single_num) {
            oss << ')';
        }
    };

    // Helper to stream the denominator of the product.
    auto stream_den = [&]() {
        assert(den_it != tmp_args.end());

        // Is the denominator consisting of a single term?
        const auto single_den = (den_it + 1 == tmp_args.end());

        if (!single_den) {
            oss << '(';
        }

        for (auto it = den_it; it != tmp_args.end(); ++it) {
            assert(std::holds_alternative<func>(it->value()));
            assert(std::get<func>(it->value()).extract<pow_impl>() != nullptr);
            assert(std::get<func>(it->value()).args().size() == 2u);

            // Fetch the pow()'s base and exponent.
            auto base = std::get<func>(it->value()).args()[0];
            auto exp = std::get<number>(std::get<func>(it->value()).args()[1].value());

            // Stream the pow() with negated exponent.
            stream_expression(oss, pow(std::move(base), expression{-exp}));

            if (it + 1 != tmp_args.end()) {
                oss << " * ";
            }
        }

        if (!single_den) {
            oss << ')';
        }
    };

    if (den_it == tmp_args.begin()) {
        // Product consists only of negative pow()s.
        stream_expression(oss, 1_dbl);
        oss << " / ";
        stream_den();
    } else if (den_it == tmp_args.end()) {
        // There are no negative pow()s in the prod.
        stream_num();
    } else {
        // There are some negative pow()s in the prod.
        stream_num();
        oss << " / ";
        stream_den();
    }
}

std::vector<expression> prod_impl::gradient() const
{
    const auto n_args = args().size();

    std::vector<expression> retval, tmp;
    retval.reserve(n_args);
    tmp.reserve(n_args);

    for (decltype(args().size()) i = 0; i < n_args; ++i) {
        tmp.clear();

        for (decltype(i) j = 0; j < n_args; ++j) {
            if (i != j) {
                tmp.push_back(args()[j]);
            }
        }

        retval.push_back(prod(tmp));
    }

    return retval;
}

// Simplify the arguments for a prod(). This function returns either the simplified vector of arguments,
// or a single expression directly representing the result of the product.
std::variant<std::vector<expression>, expression> prod_simplify_args(const std::vector<expression> &args_)
{
    // Step 1: flatten products in args.
    std::vector<expression> args;
    args.reserve(args_.size());
    for (const auto &arg : args_) {
        if (const auto *fptr = std::get_if<func>(&arg.value());
            fptr != nullptr && fptr->extract<detail::prod_impl>() != nullptr) {

            for (const auto &prod_arg : fptr->args()) {
                args.push_back(prod_arg);
            }
        } else {
            args.push_back(arg);
        }
    }

    // Step 2: gather common bases with numerical exponents.
    // NOTE: we use map instead of unordered_map because expression hashing always
    // requires traversing the whole expression, while std::less<expression> can
    // exit early.
    std::map<expression, number> base_exp_map;

    for (const auto &arg : args) {
        if (const auto *fptr = std::get_if<func>(&arg.value());
            fptr != nullptr && fptr->extract<detail::pow_impl>() != nullptr
            && std::holds_alternative<number>(fptr->args()[1].value())) {
            // The current argument is of the form x**exp, where exp is a number.
            const auto &exp = std::get<number>(fptr->args()[1].value());

            // Try to insert base and exponent into base_exp_map.
            const auto [it, new_item] = base_exp_map.try_emplace(fptr->args()[0], exp);

            if (!new_item) {
                // The base already existed in base_exp_map, update the exponent.
                it->second = it->second + exp;
            }
        } else {
            // The current argument is *NOT* a power with a numerical exponent. Let's try to insert
            // it into base_exp_map with an exponent of 1.
            const auto [it, new_item] = base_exp_map.try_emplace(arg, 1.);

            if (!new_item) {
                // The current argument was already in the map, update its exponent.
                it->second = it->second + number{1.};
            }
        }
    }

    // Reconstruct args from base_exp_map.
    args.clear();
    for (const auto &[base, exp] : base_exp_map) {
        args.push_back(pow(base, expression{exp}));
    }

    // Step 3: partition args so that all numbers are at the end.
    const auto n_end_it = std::stable_partition(
        args.begin(), args.end(), [](const expression &ex) { return !std::holds_alternative<number>(ex.value()); });

    // Constant fold the numbers.
    if (n_end_it != args.end()) {
        for (auto it = n_end_it + 1; it != args.end(); ++it) {
            // NOTE: do not use directly operator*() on expressions in order
            // to avoid recursion.
            *n_end_it = expression{std::get<number>(n_end_it->value()) * std::get<number>(it->value())};
        }

        // Remove all numbers but the first one.
        args.erase(n_end_it + 1, args.end());

        // Remove the remaining number if it is one, or
        // return zero if constant folding results in zero.
        if (is_one(std::get<number>(n_end_it->value()))) {
            args.pop_back();
        } else if (is_zero(std::get<number>(n_end_it->value()))) {
            return 0_dbl;
        }
    }

    // Special cases.
    if (args.empty()) {
        return 1_dbl;
    }

    if (args.size() == 1u) {
        return std::move(args[0]);
    }

    // Sort the operands in canonical order.
    std::stable_sort(args.begin(), args.end(), detail::comm_ops_lt);

    return args;
}

} // namespace detail

expression prod(const std::vector<expression> &args_)
{
    auto args = detail::prod_simplify_args(args_);

    if (std::holds_alternative<expression>(args)) {
        return std::move(std::get<expression>(args));
    } else {
        return expression{func{detail::prod_impl{std::move(std::get<std::vector<expression>>(args))}}};
    }
}

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::prod_impl)
