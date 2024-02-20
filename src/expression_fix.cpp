// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

class fix_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    fix_impl() : fix_impl(0_dbl) {}
    explicit fix_impl(expression x) : func_base("fix", {std::move(x)}) {}

    void to_stream(std::ostringstream &oss) const
    {
        assert(args().size() == 1u);

        oss << '{';
        stream_expression(oss, args()[0]);
        oss << '}';
    }

    // NOTE: need custom implementations here in order to ensure that
    // the result of the diff is also fixed.
    expression diff(funcptr_map<expression> &func_map, const std::string &s) const
    {
        assert(args().size() == 1u);

        return fix(detail::diff(func_map, args()[0], s));
    }
    expression diff(funcptr_map<expression> &func_map, const param &p) const
    {
        assert(args().size() == 1u);

        return fix(detail::diff(func_map, args()[0], p));
    }
};

} // namespace

bool is_fixed(const expression &ex)
{
    const auto *fptr = std::get_if<func>(&ex.value());

    return fptr != nullptr && fptr->extract<fix_impl>() != nullptr;
}

} // namespace detail

expression fix(expression x)
{
    if (detail::is_fixed(x)) {
        return x;
    } else {
        return expression{func{detail::fix_impl(std::move(x))}};
    }
}

expression fix_nn(expression x)
{
    if (std::holds_alternative<number>(x.value())) {
        return x;
    } else {
        return fix(std::move(x));
    }
}

namespace detail
{

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
expression unfix_impl(funcptr_map<expression> &func_map, const expression &ex)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_map](const auto &v) {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, func>) {
                const auto *f_id = v.get_ptr();

                // Check if we already handled ex.
                if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                // Recursively unfix the arguments.
                std::vector<expression> new_args;
                new_args.reserve(v.args().size());
                for (const auto &orig_arg : v.args()) {
                    new_args.push_back(unfix_impl(func_map, orig_arg));
                }

                // Prepare the return value.
                std::optional<expression> retval;

                if (v.template extract<fix_impl>() == nullptr) {
                    // ex is not a fixed expression, return a copy.
                    retval.emplace(v.copy(new_args));
                } else {
                    // ex is a fixed expression, unfix it.
                    assert(new_args.size() == 1u);
                    retval.emplace(new_args[0]);
                }

                // Put the return value into the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, *retval);
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return std::move(*retval);
            } else {
                // Not a function, return a copy.
                return expression{v};
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

// Helpers to remove the fix() function from an expression
// or a vector of expressions.
expression unfix(const expression &ex)
{
    detail::funcptr_map<expression> func_map;

    return detail::unfix_impl(func_map, ex);
}

std::vector<expression> unfix(const std::vector<expression> &v_ex)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        retval.push_back(detail::unfix_impl(func_map, ex));
    }

    return retval;
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT(heyoka::detail::fix_impl)
