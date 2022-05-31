// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

#include <llvm/IR/Function.h>

#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>

namespace heyoka::detail
{

bool llvm_func_name_compare::operator()(const llvm::Function *f0, const llvm::Function *f1) const
{
    return f0->getName() < f1->getName();
}

// Helper to convert the arguments of the definition of a u variable
// into a vector of variants. u variables will be converted to their indices,
// numbers will be unchanged, parameters will be converted to their indices.
// The hidden deps will also be converted to indices.
std::vector<std::variant<std::uint32_t, number>> udef_to_variants(const expression &ex,
                                                                  const std::vector<std::uint32_t> &deps)
{
    return std::visit(
        [&deps](const auto &v) -> std::vector<std::variant<std::uint32_t, number>> {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                std::vector<std::variant<std::uint32_t, number>> retval;

                for (const auto &arg : v.args()) {
                    std::visit(
                        [&retval](const auto &x) {
                            using tp = uncvref_t<decltype(x)>;

                            if constexpr (std::is_same_v<tp, variable>) {
                                retval.emplace_back(uname_to_index(x.name()));
                            } else if constexpr (std::is_same_v<tp, number>) {
                                retval.emplace_back(x);
                            } else if constexpr (std::is_same_v<tp, param>) {
                                retval.emplace_back(x.idx());
                            } else {
                                throw std::invalid_argument(
                                    "Invalid argument encountered in an element of a decomposition: the "
                                    "argument is not a variable or a number");
                            }
                        },
                        arg.value());
                }

                // Handle the hidden deps.
                for (auto idx : deps) {
                    retval.emplace_back(idx);
                }

                return retval;
            } else {
                throw std::invalid_argument("Invalid expression encountered in a decomposition: the "
                                            "expression is not a function");
            }
        },
        ex.value());
}

} // namespace heyoka::detail
