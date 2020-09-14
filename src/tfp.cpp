// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/tfp.hpp>

namespace heyoka
{

tfp tfp_add(llvm_state &s, const tfp &x, const tfp &y)
{
    return std::visit(
        [&s](const auto &a, const auto &b) -> tfp {
            using t1 = detail::uncvref_t<decltype(a)>;
            using t2 = detail::uncvref_t<decltype(b)>;

            auto &builder = s.builder();

            if constexpr (std::is_same_v<t1, llvm::Value *> && std::is_same_v<t2, llvm::Value *>) {
                return builder.CreateFAdd(a, b);
            } else if constexpr (std::is_same_v<
                                     t1,
                                     std::pair<llvm::Value *,
                                               llvm::Value
                                                   *>> && std::is_same_v<t2, std::pair<llvm::Value *, llvm::Value *>>) {
                // Knuth's TwoSum algorithm.
                auto x = builder.CreateFAdd(a.first, b.first);
                auto z = builder.CreateFSub(x, a.first);
                auto y = builder.CreateFAdd(builder.CreateFSub(a.first, builder.CreateFSub(x, z)),
                                            builder.CreateFSub(b.first, z));

                // Double-length addition without normalisation.
                return std::pair{x, builder.CreateFAdd(y, builder.CreateFAdd(a.second, b.second))};
            } else {
                throw std::invalid_argument("Invalid combination of arguments in tfp_add(): the input tfp variants "
                                            "must contain the same types");
            }
        },
        x, y);
}

tfp tfp_neg(llvm_state &s, const tfp &x)
{
    return std::visit(
        [&s](const auto &a) -> tfp {
            auto &builder = s.builder();

            if constexpr (std::is_same_v<detail::uncvref_t<decltype(a)>, llvm::Value *>) {
                return builder.CreateFNeg(a);
            } else {
                return std::pair{builder.CreateFNeg(a.first), builder.CreateFNeg(a.second)};
            }
        },
        x);
}

// x + y = x + (-y).
tfp tfp_sub(llvm_state &s, const tfp &x, const tfp &y)
{
    return tfp_add(s, x, tfp_neg(s, y));
}

namespace detail
{

namespace
{

// Helper to invoke the fma primitive used in the implementation
// of tfp_mul().
llvm::Value *tfp_fma(llvm_state &s, llvm::Value *x, llvm::Value *y, llvm::Value *z)
{
    assert(x->getType() == y->getType());
    assert(x->getType() == z->getType());

    // Determine the scalar type of the vector arguments.
    auto x_t = llvm::cast<llvm::VectorType>(x->getType())->getElementType();

    if (x_t == llvm::Type::getX86_FP80Ty(s.context())) {
        // NOTE: there seems to be an LLVM bug when trying
        // to use the fma intrinsic with extended precision
        // arguments. For the moment, let's delegate
        // to the fmal() C function and re-visit in newer
        // LLVM versions.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x), y_scalars = vector_to_scalars(builder, y),
             z_scalars = vector_to_scalars(builder, z);

        // Execute the fma function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "fmal", llvm::Type::getX86_FP80Ty(s.context()), {x_scalars[i], y_scalars[i], z_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x), y_scalars = vector_to_scalars(builder, y),
             z_scalars = vector_to_scalars(builder, z);

        // Execute the fma function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_fma128", llvm::Type::getFP128Ty(s.context()), {x_scalars[i], y_scalars[i], z_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
#endif
    } else {
        return llvm_invoke_intrinsic(s, "llvm.fma", {x->getType()}, {x, y, z});
    }
}

// TwoProductFMA algorithm of Ogita et al.
auto tfp_eft_prod(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    auto x = builder.CreateFMul(a, b);
    auto y = detail::tfp_fma(s, a, b, builder.CreateFNeg(x));

    return std::pair{x, y};
}

} // namespace

} // namespace detail

tfp tfp_mul(llvm_state &s, const tfp &x, const tfp &y)
{
    return std::visit(
        [&s, &x, &y](const auto &a, const auto &b) -> tfp {
            using t1 = detail::uncvref_t<decltype(a)>;
            using t2 = detail::uncvref_t<decltype(b)>;

            auto &builder = s.builder();

            if constexpr (std::is_same_v<t1, llvm::Value *> && std::is_same_v<t2, llvm::Value *>) {
                return builder.CreateFMul(a, b);
            } else if constexpr (std::is_same_v<
                                     t1,
                                     std::pair<llvm::Value *,
                                               llvm::Value
                                                   *>> && std::is_same_v<t2, std::pair<llvm::Value *, llvm::Value *>>) {
                return tfp_from_vector(s, builder.CreateFMul(tfp_to_vector(s, x), tfp_to_vector(s, y)), true);
            } else {
                throw std::invalid_argument("Invalid combination of arguments in tfp_mul(): the input tfp variants "
                                            "must contain the same types");
            }
        },
        x, y);
}

tfp tfp_div(llvm_state &s, const tfp &x, const tfp &y)
{
    return std::visit(
        [&s, &x, &y](const auto &a, const auto &b) -> tfp {
            using t1 = detail::uncvref_t<decltype(a)>;
            using t2 = detail::uncvref_t<decltype(b)>;

            auto &builder = s.builder();

            if constexpr (std::is_same_v<t1, llvm::Value *> && std::is_same_v<t2, llvm::Value *>) {
                return builder.CreateFDiv(a, b);
            } else if constexpr (std::is_same_v<
                                     t1,
                                     std::pair<llvm::Value *,
                                               llvm::Value
                                                   *>> && std::is_same_v<t2, std::pair<llvm::Value *, llvm::Value *>>) {
                return tfp_from_vector(s, builder.CreateFDiv(tfp_to_vector(s, x), tfp_to_vector(s, y)), true);
            } else {
                throw std::invalid_argument("Invalid combination of arguments in tfp_div(): the input tfp variants "
                                            "must contain the same types");
            }
        },
        x, y);
}

namespace detail
{

// Helper to load the derivative of order 'order' of the u variable at index u_idx from the
// derivative array 'arr'. The total number of u variables is n_uvars.
tfp taylor_load_derivative(const std::vector<tfp> &arr, std::uint32_t u_idx, std::uint32_t order, std::uint32_t n_uvars)
{
    // Sanity check.
    assert(u_idx < n_uvars);

    // Compute the index.
    const auto idx = static_cast<decltype(arr.size())>(order) * n_uvars + u_idx;
    assert(idx < arr.size());

    return arr[idx];
}

// Pairwise summation of a vector of tfps.
// https://en.wikipedia.org/wiki/Pairwise_summation
tfp tfp_pairwise_sum(llvm_state &s, std::vector<tfp> &sum)
{
    assert(!sum.empty());

    if (sum.size() == std::numeric_limits<decltype(sum.size())>::max()) {
        throw std::overflow_error("Overflow error in tfp_pairwise_sum()");
    }

    while (sum.size() != 1u) {
        std::vector<tfp> new_sum;

        for (decltype(sum.size()) i = 0; i < sum.size(); i += 2u) {
            if (i + 1u == sum.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_sum.push_back(sum[i]);
            } else {
                new_sum.push_back(tfp_add(s, sum[i], sum[i + 1u]));
            }
        }

        new_sum.swap(sum);
    }

    return sum[0];
}

} // namespace detail

// Cast a tfp back to an LLVM floating-point vector.
llvm::Value *tfp_to_vector(llvm_state &s, const tfp &x)
{
    return std::visit(
        [&s](const auto &a) {
            if constexpr (std::is_same_v<detail::uncvref_t<decltype(a)>, llvm::Value *>) {
                return a;
            } else {
                return s.builder().CreateFAdd(a.first, a.second);
            }
        },
        x);
}

// Helper to create a tfp from an input vector. In normal mode,
// it will just return x. Otherwise, it will return x paired
// to a zero-filled error vector.
tfp tfp_from_vector(llvm_state &s, llvm::Value *x, bool high_accuracy)
{
    if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(x->getType())) {
        if (!high_accuracy) {
            // In normal mode, just return x converted to a tfp.
            return x;
        }

        // In high accuracy mode, we need to init the error component to zero.
        auto &builder = s.builder();

        // Determine the scalar type.
        auto x_t = vec_t->getElementType();

        // Fetch the vector width.
        const auto vector_size = vec_t->getNumElements();

        // Create a scalar zero of type x_t.
        auto s_zero = builder.CreateUIToFP(builder.getInt32(0), x_t);

        return std::pair{
            x, detail::create_constant_vector(builder, s_zero, boost::numeric_cast<std::uint32_t>(vector_size))};
    } else {
        throw std::invalid_argument("Cannot create a tfp from a non-vector value");
    }
}

} // namespace heyoka
