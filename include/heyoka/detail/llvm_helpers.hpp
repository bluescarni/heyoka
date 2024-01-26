// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_HELPERS_HPP
#define HEYOKA_DETAIL_LLVM_HELPERS_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

HEYOKA_DLL_PUBLIC llvm::Type *to_llvm_type_impl(llvm::LLVMContext &, const std::type_info &, bool);

// Helper to associate a C++ type to an LLVM type.
template <typename T>
inline llvm::Type *to_llvm_type(llvm::LLVMContext &c, bool err_throw = true)
{
    return to_llvm_type_impl(c, typeid(T), err_throw);
}

template <typename T>
HEYOKA_DLL_PUBLIC llvm::Type *llvm_type_like(llvm_state &, const T &);

HEYOKA_DLL_PUBLIC llvm::Type *make_vector_type(llvm::Type *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Type *llvm_ext_type(llvm::Type *);

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

std::uint32_t gl_arr_size(llvm::Value *);

HEYOKA_DLL_PUBLIC std::uint64_t get_size(llvm::Module &, llvm::Type *);

HEYOKA_DLL_PUBLIC llvm::Value *to_size_t(llvm_state &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::GlobalVariable *make_global_zero_array(llvm::Module &, llvm::ArrayType *);

HEYOKA_DLL_PUBLIC llvm::Value *load_vector_from_memory(ir_builder &, llvm::Type *, llvm::Value *, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *ext_load_vector_from_memory(llvm_state &, llvm::Type *, llvm::Value *, std::uint32_t);

llvm::Value *gather_vector_from_memory(ir_builder &, llvm::Type *, llvm::Value *);
llvm::Value *ext_gather_vector_from_memory(llvm_state &, llvm::Type *, llvm::Value *);

HEYOKA_DLL_PUBLIC void store_vector_to_memory(ir_builder &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC void ext_store_vector_to_memory(llvm_state &, llvm::Value *, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *vector_splat(ir_builder &, llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC std::vector<llvm::Value *> vector_to_scalars(ir_builder &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *scalars_to_vector(ir_builder &, const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::Value *pairwise_reduce(std::vector<llvm::Value *> &,
                                               const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &);
HEYOKA_DLL_PUBLIC llvm::Value *pairwise_sum(llvm_state &, std::vector<llvm::Value *> &);
HEYOKA_DLL_PUBLIC llvm::Value *pairwise_prod(llvm_state &, std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::CallInst *llvm_invoke_intrinsic(ir_builder &, const std::string &,
                                                        const std::vector<llvm::Type *> &,
                                                        const std::vector<llvm::Value *> &);

HEYOKA_DLL_PUBLIC llvm::CallInst *llvm_invoke_external(llvm_state &, const std::string &, llvm::Type *,
                                                       const std::vector<llvm::Value *> &);
HEYOKA_DLL_PUBLIC llvm::CallInst *llvm_invoke_external(llvm_state &, const std::string &, llvm::Type *,
                                                       const std::vector<llvm::Value *> &, const llvm::AttributeList &);

HEYOKA_DLL_PUBLIC void llvm_loop_u32(llvm_state &, llvm::Value *, llvm::Value *,
                                     const std::function<void(llvm::Value *)> &,
                                     const std::function<llvm::Value *(llvm::Value *)> & = {});

HEYOKA_DLL_PUBLIC void llvm_while_loop(llvm_state &, const std::function<llvm::Value *()> &,
                                       const std::function<void()> &);

HEYOKA_DLL_PUBLIC void llvm_if_then_else(llvm_state &, llvm::Value *, const std::function<void()> &,
                                         const std::function<void()> &);

HEYOKA_DLL_PUBLIC void llvm_switch_u32(llvm_state &, llvm::Value *, const std::function<void()> &,
                                       const std::map<std::uint32_t, std::function<void()>> &);

HEYOKA_DLL_PUBLIC std::string llvm_type_name(llvm::Type *);

void llvm_append_block(llvm::Function *, llvm::BasicBlock *);

// Math helpers.
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fadd(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fsub(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fmul(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fdiv(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fneg(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fma(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_square(llvm_state &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Constant *llvm_constantfp(llvm_state &, llvm::Type *, double);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_ult(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_uge(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_oge(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_ole(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_olt(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_ogt(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_fcmp_oeq(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_min(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_max(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_min_nan(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_max_nan(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sgn(llvm_state &, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_ui_to_fp(llvm_state &, llvm::Value *, llvm::Type *);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_abs(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_floor(llvm_state &, llvm::Value *);

HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_sincos(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_atan2(llvm_state &, llvm::Value *, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_exp(llvm_state &, llvm::Value *);
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
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sigmoid(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sinh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_sqrt(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_tan(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_tanh(llvm_state &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *llvm_pow(llvm_state &, llvm::Value *, llvm::Value *);

HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_csc(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t);

HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *> llvm_penc_interval(llvm_state &, llvm::Type *, llvm::Value *,
                                                                             std::uint32_t, llvm::Value *,
                                                                             llvm::Value *, std::uint32_t);

HEYOKA_DLL_PUBLIC std::pair<llvm::Value *, llvm::Value *>
llvm_penc_cargo_shisha(llvm_state &, llvm::Type *, llvm::Value *, std::uint32_t, llvm::Value *, std::uint32_t);

llvm::Function *llvm_add_inv_kep_E(llvm_state &, llvm::Type *, std::uint32_t);
HEYOKA_DLL_PUBLIC void llvm_add_inv_kep_E_wrapper(llvm_state &, llvm::Type *, std::uint32_t, const std::string &);

llvm::Function *llvm_add_inv_kep_F(llvm_state &, llvm::Type *, std::uint32_t);
llvm::Function *llvm_add_inv_kep_DE(llvm_state &, llvm::Type *, std::uint32_t);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_add_bc_array(llvm_state &, llvm::Type *, std::uint32_t);

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

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
