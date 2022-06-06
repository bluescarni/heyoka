// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_CM_UTILS_HPP
#define HEYOKA_DETAIL_CM_UTILS_HPP

#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>

namespace heyoka::detail
{

// Comparision operator for LLVM functions based on their names.
struct llvm_func_name_compare {
    bool operator()(const llvm::Function *, const llvm::Function *) const;
};

std::vector<std::variant<std::uint32_t, number>> udef_to_variants(const expression &,
                                                                  const std::vector<std::uint32_t> &);

// Helper to convert a vector of variants into a variant of vectors.
// All elements of v must be of the same type, and v cannot be empty.
template <typename... T>
inline auto vv_transpose(const std::vector<std::variant<T...>> &v)
{
    assert(!v.empty());

    // Init the return value based on the type
    // of the first element of v.
    auto retval = std::visit(
        [size = v.size()](const auto &x) {
            using type = uncvref_t<decltype(x)>;

            std::vector<type> tmp;
            tmp.reserve(boost::numeric_cast<decltype(tmp.size())>(size));
            tmp.push_back(x);

            return std::variant<std::vector<T>...>(std::move(tmp));
        },
        v[0]);

    // Append the other values from v.
    for (decltype(v.size()) i = 1; i < v.size(); ++i) {
        std::visit(
            [&retval](const auto &x) {
                std::visit(
                    [&x](auto &vv) {
                        // The value type of retval.
                        using scal_t = typename uncvref_t<decltype(vv)>::value_type;

                        // The type of the current element of v.
                        using x_t = uncvref_t<decltype(x)>;

                        if constexpr (std::is_same_v<scal_t, x_t>) {
                            vv.push_back(x);
                        } else {
                            throw std::invalid_argument("Inconsistent types detected in vv_transpose()");
                        }
                    },
                    retval);
            },
            v[i]);
    }

    return retval;
}

std::function<llvm::Value *(llvm::Value *)> cm_make_arg_gen_vidx(llvm_state &, const std::vector<std::uint32_t> &);

template <typename>
std::function<llvm::Value *(llvm::Value *)> cm_make_arg_gen_vc(llvm_state &, const std::vector<number> &);

} // namespace heyoka::detail

#endif
