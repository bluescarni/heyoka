// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

// NOTE: GCC warns about use of mismatched new/delete
// when creating global variables. I am not sure this is
// a real issue, as it looks like we are adopting the "canonical"
// approach for the creation of global variables (at least
// according to various sources online)
// and clang is not complaining. But let us revisit
// this issue in later LLVM versions.
#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
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

namespace
{

// Helper to check if a vector of indices consists of consecutive
// strided values:
//
// [n, n + s, n + 2*s, n + 3*s, ...],
//
// where s > 0 is the stride value. The function returns the stride value,
// or zero if either:
// - v has only 1 element, or
// - the detected stride value is not strictly positive, or
// - the values in the vector do not match the strided consecutive pattern.
std::uint32_t is_consecutive_strided(const std::vector<std::uint32_t> &v)
{
    assert(!v.empty());

    if (v.size() == 1u || v[1] <= v[0]) {
        return 0;
    }

    const std::uint32_t candidate = v[1] - v[0];

    for (decltype(v.size()) i = 2; i < v.size(); ++i) {
        // NOTE: the first check is to avoid potential
        // negative overflow in the second check.
        if (v[i] <= v[i - 1u] || v[i] - v[i - 1u] != candidate) {
            return 0;
        }
    }

    return candidate;
}

} // namespace

// This helper returns the argument generator for compact mode functions when the argument is
// a vector of u var indices.
// NOTE: here we ensure that the return value does not capture
// any LLVM object except for types and global variables. This way we ensure that
// the generator does not rely on LLVM values created at the current insertion point.
std::function<llvm::Value *(llvm::Value *)> cm_make_arg_gen_vidx(llvm_state &s, const std::vector<std::uint32_t> &ind)
{
    assert(!ind.empty()); // LCOV_EXCL_LINE

    auto &builder = s.builder();

    // Check if all indices in ind are the same.
    if (std::all_of(ind.begin() + 1, ind.end(), [&ind](const auto &n) { return n == ind[0]; })) {
        // If all indices are the same, don't construct an array, just always return
        // the same value.
        return [&builder, num = ind[0]](llvm::Value *) -> llvm::Value * { return builder.getInt32(num); };
    }

    // If ind consists of consecutive (possibly strided) indices, we can replace
    // the index array with a simple offset computation.
    if (const auto stride = is_consecutive_strided(ind); stride != 0u) {
        return [&builder, start_idx = ind[0], stride](llvm::Value *cur_call_idx) -> llvm::Value * {
            return builder.CreateAdd(builder.getInt32(start_idx),
                                     builder.CreateMul(builder.getInt32(stride), cur_call_idx));
        };
    }

    // Check if ind consists of a repeated pattern like [a, a, a, b, b, b, c, c, c, ...],
    // that is, [a X n, b X n, c X n, ...], such that [a, b, c, ...] are consecutive (possibly
    // strided) numbers.
    if (ind.size() > 1u) {
        // Determine the candidate number of repetitions.
        decltype(ind.size()) n_reps = 1;
        for (decltype(ind.size()) i = 1; i < ind.size(); ++i) {
            if (ind[i] == ind[i - 1u]) {
                ++n_reps;
            } else {
                break;
            }
        }

        if (n_reps > 1u && (ind.size() % n_reps) == 0u) {
            // There is an initial number of repetitions
            // and the vector size is a multiple of that.
            // See if the repetitions continue, and keep
            // track of the repeated indices.
            std::vector<std::uint32_t> rep_indices{ind[0]};

            bool rep_flag = true;

            // Iterate over the blocks of repetitions.
            for (decltype(ind.size()) rep_idx = 1; rep_idx < ind.size() / n_reps; ++rep_idx) {
                for (decltype(ind.size()) i = 1; i < n_reps; ++i) {
                    const auto cur_idx = rep_idx * n_reps + i;

                    if (ind[cur_idx] != ind[cur_idx - 1u]) {
                        rep_flag = false;
                        break;
                    }
                }

                if (rep_flag) {
                    rep_indices.push_back(ind[rep_idx * n_reps]);
                } else {
                    break;
                }
            }

            if (const auto stride = is_consecutive_strided(rep_indices); rep_flag && stride != 0u) {
                // The pattern is  [a X n, b X n, c X n, ...] and [a, b, c, ...]
                // are consecutive (possibly strided) numbers. The m-th value in the array can thus
                // be computed as a + floor(m / n) * stride.

#if !defined(NDEBUG)
                // Double-check the result in debug mode.
                std::vector<std::uint32_t> checker;
                for (decltype(ind.size()) i = 0; i < ind.size(); ++i) {
                    checker.push_back(boost::numeric_cast<std::uint32_t>(ind[0] + (i / n_reps) * stride));
                }
                assert(checker == ind); // LCOV_EXCL_LINE
#endif

                return [&builder, start_idx = rep_indices[0], n_reps = boost::numeric_cast<std::uint32_t>(n_reps),
                        stride](llvm::Value *cur_call_idx) -> llvm::Value * {
                    return builder.CreateAdd(
                        builder.getInt32(start_idx),
                        builder.CreateMul(builder.CreateUDiv(cur_call_idx, builder.getInt32(n_reps)),
                                          builder.getInt32(stride)));
                };
            }
        }
    }

    auto &md = s.module();

    // Generate the array of indices as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    tmp_c_vec.reserve(ind.size());
    for (const auto &val : ind) {
        tmp_c_vec.push_back(builder.getInt32(val));
    }

    // Create the array type.
    auto *arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(ind.size()));
    assert(arr_type != nullptr); // LCOV_EXCL_LINE

    // Create the constant array as a global read-only variable.
    auto *const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr); // LCOV_EXCL_LINE
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto *gvar
        = new llvm::GlobalVariable(md, const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, const_arr);

    // Return the generator.
    return [&builder, gvar, arr_type](llvm::Value *cur_call_idx) -> llvm::Value * {
        return builder.CreateLoad(builder.getInt32Ty(),
                                  builder.CreateInBoundsGEP(arr_type, gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

// This helper returns the argument generator for compact mode functions when the argument is
// a vector of floating-point constants.
// NOTE: here we ensure that the return value does not capture
// any LLVM object except for types and global variables. This way we ensure that
// the generator does not rely on LLVM values created at the current insertion point.
std::function<llvm::Value *(llvm::Value *)> cm_make_arg_gen_vc(llvm_state &s, llvm::Type *fp_t,
                                                               const std::vector<number> &vc)
{
    assert(!vc.empty()); // LCOV_EXCL_LINE

    // Check if all the numbers are the same.
    if (std::all_of(vc.begin() + 1, vc.end(), [&vc](const auto &n) { return n == vc[0]; })) {
        // If all constants are the same, don't construct an array, just always return
        // the same value.
        return [&s, fp_t, num = vc[0]](llvm::Value *) -> llvm::Value * { return llvm_codegen(s, fp_t, num); };
    }

    // Generate the array of constants as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    tmp_c_vec.reserve(vc.size());
    for (const auto &val : vc) {
        tmp_c_vec.push_back(llvm::cast<llvm::Constant>(llvm_codegen(s, fp_t, val)));
    }

    // Create the array type.
    auto *arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(vc.size()));
    assert(arr_type != nullptr); // LCOV_EXCL_LINE

    // Create the constant array as a global read-only variable.
    auto *const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr); // LCOV_EXCL_LINE
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto *gvar = new llvm::GlobalVariable(s.module(), const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage,
                                          const_arr);

    // Return the generator.
    return [&s, gvar, arr_type](llvm::Value *cur_call_idx) -> llvm::Value * {
        auto &builder = s.builder();

        return builder.CreateLoad(arr_type->getArrayElementType(),
                                  builder.CreateInBoundsGEP(arr_type, gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

std::string cm_mangle(const variable &)
{
    return "var";
}

std::string cm_mangle(const number &)
{
    return "num";
}

std::string cm_mangle(const param &)
{
    return "par";
}

// LCOV_EXCL_START
std::string cm_mangle(const func &)
{
    throw std::invalid_argument("Cannot mangle the name of a function argument");
}
// LCOV_EXCL_STOP

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
