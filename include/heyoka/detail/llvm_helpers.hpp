// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_HELPERS_HPP
#define HEYOKA_DETAIL_LLVM_HELPERS_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka::detail
{

HEYOKA_DLL_PUBLIC llvm::Type *to_llvm_type_impl(llvm::LLVMContext &, const std::type_info &);

// Helper to associate a C++ type to an LLVM type.
template <typename T>
inline llvm::Type *to_llvm_type(llvm::LLVMContext &c)
{
    return to_llvm_type_impl(c, typeid(T));
}

HEYOKA_DLL_PUBLIC llvm::Type *make_vector_type(llvm::Type *, std::uint32_t);

// Helper to construct an LLVM vector type of size batch_size with elements
// of the LLVM type tp corresponding to the C++ type T. If batch_size is 1, tp
// will be returned. batch_size cannot be zero.
template <typename T>
inline llvm::Type *to_llvm_vector_type(llvm::LLVMContext &c, std::uint32_t batch_size)
{
    return make_vector_type(to_llvm_type<T>(c), batch_size);
}

HEYOKA_DLL_PUBLIC std::string llvm_mangle_type(llvm::Type *);

HEYOKA_DLL_PUBLIC std::uint32_t get_vector_size(llvm::Value *);

HEYOKA_DLL_PUBLIC std::uint64_t get_alignment(llvm::Module &, llvm::Type *);

HEYOKA_DLL_PUBLIC llvm::Value *to_size_t(llvm_state &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::GlobalVariable *make_global_zero_array(llvm::Module &, llvm::ArrayType *);

HEYOKA_DLL_PUBLIC llvm::Value *load_vector_from_memory(ir_builder &, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC void store_vector_to_memory(ir_builder &, llvm::Value *, llvm::Value *);
llvm::Value *gather_vector_from_memory(ir_builder &, llvm::Type *, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *vector_splat(ir_builder &, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC std::vector<llvm::Value *> vector_to_scalars(ir_builder &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *scalars_to_vector(ir_builder &, const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::Value *pairwise_reduce(std::vector<llvm::Value *> &,
                                               const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &);
HEYOKA_DLL_PUBLIC llvm::Value *pairwise_sum(ir_builder &, std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::CallInst *llvm_invoke_intrinsic(ir_builder &, const std::string &,
                                                        const std::vector<llvm::Type *> &,
                                                        const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::CallInst *llvm_invoke_external(llvm_state &, const std::string &, llvm::Type *,
                                                       const std::vector<llvm::Value *> &,
                                                       // NOTE: this is going to be converted into
                                                       // a vector of llvm attributes (represented
                                                       // as an enum) in the implementation.
                                                       const std::vector<int> & = {});

HEYOKA_DLL_PUBLIC void llvm_loop_u32(llvm_state &, llvm::Value *, llvm::Value *,
                                     const std::function<void(llvm::Value *)> &,
                                     const std::function<llvm::Value *(llvm::Value *)> & = {});

HEYOKA_DLL_PUBLIC void llvm_while_loop(llvm_state &, const std::function<llvm::Value *()> &,
                                       const std::function<void()> &);

HEYOKA_DLL_PUBLIC void llvm_if_then_else(llvm_state &, llvm::Value *, const std::function<void()> &,
                                         const std::function<void()> &);

HEYOKA_DLL_PUBLIC llvm::Type *pointee_type(llvm::Value *);

HEYOKA_DLL_PUBLIC std::string llvm_type_name(llvm::Type *);

HEYOKA_DLL_PUBLIC bool compare_function_signature(llvm::Function *, llvm::Type *, const std::vector<llvm::Type *> &);

HEYOKA_DLL_PUBLIC llvm::Value *call_extern_vec(llvm_state &, const std::vector<llvm::Value *> &, const std::string &);

// Math helpers.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_sincos(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_modulus(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_abs(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_min(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_max(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_min_nan(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_max_nan(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sgn(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_atan2(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_exp(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fma(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_floor(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_acos(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_acosh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_asin(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_asinh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_atan(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_atanh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_cos(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sin(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_cosh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_erf(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_log(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_neg(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sigmoid(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sinh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sqrt(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_square(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_tan(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_tanh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_pow(llvm_state &, llvm::Value *, llvm::Value *, bool = false);

template <typename>
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_csc(llvm_state &, std::uint32_t, std::uint32_t);

template <typename>
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *>
llvm_penc_interval(llvm_state &, llvm::Value *, std::uint32_t, llvm::Value *, llvm::Value *, std::uint32_t);

template <typename>
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *>
llvm_penc_cargo_shisha(llvm_state &, llvm::Value *, std::uint32_t, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_inv_kep_E_dbl(llvm_state &, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_inv_kep_E_ldbl(llvm_state &, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_inv_kep_E_f128(llvm_state &, std::uint32_t);

#endif

template <typename T>
inline llvm::Function *llvm_add_inv_kep_E(llvm_state &s, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return llvm_add_inv_kep_E_dbl(s, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return llvm_add_inv_kep_E_ldbl(s, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return llvm_add_inv_kep_E_f128(s, batch_size);
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }
}

template <typename>
HEYOKA_DLL_PUBLIC void llvm_add_inv_kep_E_wrapper(llvm_state &, std::uint32_t, const std::string &);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_add_bc_array_dbl(llvm_state &, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_add_bc_array_ldbl(llvm_state &, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *llvm_add_bc_array_f128(llvm_state &, std::uint32_t);

#endif

template <typename T>
inline llvm::Value *llvm_add_bc_array(llvm_state &s, std::uint32_t n)
{
    if constexpr (std::is_same_v<T, double>) {
        return llvm_add_bc_array_dbl(s, n);
    } else if constexpr (std::is_same_v<T, long double>) {
        return llvm_add_bc_array_ldbl(s, n);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return llvm_add_bc_array_f128(s, n);
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }
}

// Helpers to double check the modifications needed
// by LLVM deprecations.
bool llvm_depr_GEP_type_check(llvm::Value *, llvm::Type *);

// Primitives for double-length arithmetic.

// Error-free product transform.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_eft_product(llvm_state &, llvm::Value *, llvm::Value *);

// Addition.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_dl_add(llvm_state &, llvm::Value *, llvm::Value *,
                                                                      llvm::Value *, llvm::Value *);

// Multiplication.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_dl_mul(llvm_state &, llvm::Value *, llvm::Value *,
                                                                      llvm::Value *, llvm::Value *);

// Division.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_dl_div(llvm_state &, llvm::Value *, llvm::Value *,
                                                                      llvm::Value *, llvm::Value *);

// Floor.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_dl_floor(llvm_state &, llvm::Value *, llvm::Value *);

// Modulus.
HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_dl_modulus(llvm_state &, llvm::Value *, llvm::Value *,
                                                                          llvm::Value *, llvm::Value *);

// Less-than.
HEYOKA_DLL_PUBLIC llvm::Value *llvm_dl_lt(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *, llvm::Value *);

// Greater-than.
HEYOKA_DLL_PUBLIC llvm::Value *llvm_dl_gt(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *, llvm::Value *);

} // namespace heyoka::detail

#endif
