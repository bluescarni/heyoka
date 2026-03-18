// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <functional>
#include <utility>
#include <variant>
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/sincos_combine.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Implementation of the sincos combining functions.
//
// get_ex is a function object extracting an expression from an instance of type V.
template <typename V, typename F>
void sincos_combine_impl(std::vector<V> &dc, const F &get_ex)
{
    // First pass: create two sets containing the arguments of the sin() and cos() expressions appearing in the
    // decomposition.
    boost::unordered::unordered_flat_set<expression, std::hash<expression>> sin_args, cos_args;

    for (const auto &item : dc) {
        const auto &ex = get_ex(item);

        if (const auto *f = std::get_if<func>(&ex.value())) {
            const auto &args = f->get_func_args().get_args();

            if (f->template extract<sin_impl>() != nullptr) {
                assert(args.size() == 1u);
                sin_args.insert(args[0]);
            } else if (f->template extract<cos_impl>() != nullptr) {
                assert(args.size() == 1u);
                cos_args.insert(args[0]);
            }
        }
    }

    // Second pass: replace with the combined variants all sin() and cos() expressions whose arguments appear both in
    // sin() and cos().
    for (auto &item : dc) {
        auto &ex = get_ex(item);

        if (const auto *f = std::get_if<func>(&ex.value())) {
            const auto &args = f->get_func_args().get_args();

            if (f->template extract<sin_impl>() != nullptr) {
                assert(sin_args.contains(args[0]));
                if (cos_args.contains(args[0])) {
                    ex = combined_sin(args[0]);
                }
            } else if (f->template extract<cos_impl>() != nullptr) {
                assert(cos_args.contains(args[0]));
                if (sin_args.contains(args[0])) {
                    ex = combined_cos(args[0]);
                }
            }
        }
    }
}

} // namespace

// Replace sin()/cos() pairs operating on the same argument in a cfunc decomposition with their combined counterparts
// (combined_sin()/combined_cos()).
void sincos_combine_cfunc(std::vector<expression> &dc)
{
    sincos_combine_impl(dc, []<typename T>(T &&ex) -> auto && { return std::forward<T>(ex); });
}

// Replace sin()/cos() pairs operating on the same argument in a Taylor decomposition with their combined counterparts
// (combined_sin()/combined_cos()).
void sincos_combine_taylor(taylor_dc_t &dc)
{
    sincos_combine_impl(dc, []<typename T>(T &&ex) -> auto && { return std::forward<T>(ex).first; });
}

} // namespace detail

HEYOKA_END_NAMESPACE
