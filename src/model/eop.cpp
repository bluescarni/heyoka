// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/eop_sw_impl.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/eop.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// Helper to get/generate the function for the simultaneous computation of an angular eop quantity and its derivative
// via double-length first-order polynomial interpolation.
//
// fp_t is the scalar floating-point value that will be used in the computation. batch_size is the batch size.
// 'data' is the source of EOP data. 'name' is the name of the eop quantity. eop_data_getter is the function to
// create/fetch the eop data.
//
// The eop quantity will be returned normalised to the [0, 2pi] range.
llvm::Function *llvm_get_eop_angle_func_dl(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           const eop_data &data, const char *name,
                                           llvm::Value *(*eop_data_getter)(llvm_state &, const eop_data &,
                                                                           llvm::Type *))
{
    assert(eop_data_getter != nullptr);

    namespace hy = heyoka;
    namespace hd = hy::detail;

    auto &md = s.module();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    // Fetch the table of EOP data.
    const auto &table = data.get_table();

    // NOTE: when working with double-length eop data, we want to make sure that the table size x 2 is representable as
    // a 32-bit int. The reason for this is that, later in this function, in batch mode we will be reinterpreting the
    // table of size-2 arrays as a 1D flattened array, into which we want to be able to index via 32-bit ints.
    if (table.size() > std::numeric_limits<std::uint32_t>::max() / 2u) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::overflow_error(fmt::format("Overflow detected while generating the double-length LLVM interpolation "
                                              "function for the eop quantity '{}'",
                                              name));
        // LCOV_EXCL_STOP
    }

    // Start by creating the mangled name of the function. The mangled name will be based on:
    //
    // - the name of the eop quantity we are computing,
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data,
    // - the floating-point type.
    //
    // NOTE: '-' is intentionally chosen as the separator between timestamp and identifier. Timestamp and identifier are
    // both guaranteed not to contain '-', thus the boundary between the two is unambiguous.
    const auto fname = fmt::format("heyoka.eop_get_{}_{}p.{}.{}-{}.{}", name, name, table.size(), data.get_timestamp(),
                                   data.get_identifier(), hd::llvm_mangle_type(val_t));

    // Check if we already created the function.
    if (auto *fptr = md.getFunction(fname)) {
        return fptr;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Setup the insertion point restorer.
    const hd::ip_restorer ipr(bld);

    // Construct the function prototype. The only input is the time value, the output is the array of two values [eop,
    // eopp].
    auto *ret_t = llvm::ArrayType::get(val_t, 2);
    auto *ft = llvm::FunctionType::get(ret_t, {val_t}, false);

    // Create the function
    auto *f = hd::llvm_func_create(ft, llvm::Function::PrivateLinkage, fname, &md);

    // Fetch the time argument.
    auto *tm_val = f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Get/generate the date and eop data.
    auto *date_ptr = hd::llvm_get_eop_sw_data_date_tt_cy_j2000(s, data, fp_t, "eop");
    auto *eop_ptr = eop_data_getter(s, data, fp_t);

    // Codegen the array size (and its splatted counterpart).
    auto *arr_size = bld.getInt32(boost::numeric_cast<std::uint32_t>(table.size()));
    auto *arr_size_splat = hd::vector_splat(bld, arr_size, batch_size);

    // Locate the index in date_ptr of the time interval containing tm_val.
    auto *idx = hd::llvm_eop_sw_data_locate_date(s, date_ptr, arr_size, tm_val);

    // Codegen nan for use later.
    auto *nan_const = llvm_codegen(s, val_t, number{std::numeric_limits<double>::quiet_NaN()});

    // Codegen the type representing the eop data: an array of 2 scalars storing the hi/lo double-length parts of the
    // eop quantity.
    auto *eop_t = llvm::ArrayType::get(fp_t, 2);

    // We can now load the data from the date/eop arrays. The loaded data will be stored in the t0, t1, eop0_* and
    // eop1_* variables.
    llvm::Value *t0{}, *t1{}, *eop0_hi{}, *eop0_lo{}, *eop1_hi{}, *eop1_lo{};
    if (batch_size == 1u) {
        // Scalar implementation.

        // Storage for the values we will be loading from the date/eop arrays.
        auto *t0_alloc = bld.CreateAlloca(fp_t);
        auto *t1_alloc = bld.CreateAlloca(fp_t);
        auto *eop0_hi_alloc = bld.CreateAlloca(fp_t);
        auto *eop0_lo_alloc = bld.CreateAlloca(fp_t);
        auto *eop1_hi_alloc = bld.CreateAlloca(fp_t);
        auto *eop1_lo_alloc = bld.CreateAlloca(fp_t);

        // NOTE: in the scalar implementation, we need to branch on the value of idx: if idx == arr_size, we will return
        // NaNs, otherwise we will return the values in the arrays at indices idx and idx + 1.
        hd::llvm_if_then_else(
            s, bld.CreateICmpEQ(idx, arr_size),
            [&bld, nan_const, t0_alloc, t1_alloc, eop0_hi_alloc, eop0_lo_alloc, eop1_hi_alloc, eop1_lo_alloc]() {
                // Store the nans.
                bld.CreateStore(nan_const, t0_alloc);
                bld.CreateStore(nan_const, t1_alloc);
                bld.CreateStore(nan_const, eop0_hi_alloc);
                bld.CreateStore(nan_const, eop0_lo_alloc);
                bld.CreateStore(nan_const, eop1_hi_alloc);
                bld.CreateStore(nan_const, eop1_lo_alloc);
            },
            [&bld, idx, fp_t, eop_t, date_ptr, eop_ptr, t0_alloc, t1_alloc, eop0_hi_alloc, eop0_lo_alloc, eop1_hi_alloc,
             eop1_lo_alloc]() {
                // Compute idx + 1.
                auto *idxp1 = bld.CreateAdd(idx, bld.getInt32(1));

                // Load the date values.
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idx})), t0_alloc);
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idxp1})), t1_alloc);

                // Load the eop values.
                auto *eop0 = bld.CreateLoad(eop_t, bld.CreateInBoundsGEP(eop_t, eop_ptr, {idx}));
                auto *eop1 = bld.CreateLoad(eop_t, bld.CreateInBoundsGEP(eop_t, eop_ptr, {idxp1}));

                // Decompose into hi/lo parts.
                bld.CreateStore(bld.CreateExtractValue(eop0, {0}), eop0_hi_alloc);
                bld.CreateStore(bld.CreateExtractValue(eop0, {1}), eop0_lo_alloc);
                bld.CreateStore(bld.CreateExtractValue(eop1, {0}), eop1_hi_alloc);
                bld.CreateStore(bld.CreateExtractValue(eop1, {1}), eop1_lo_alloc);
            });

        // Fetch the values that we have stored in the allocs.
        t0 = bld.CreateLoad(fp_t, t0_alloc);
        t1 = bld.CreateLoad(fp_t, t1_alloc);
        eop0_hi = bld.CreateLoad(fp_t, eop0_hi_alloc);
        eop0_lo = bld.CreateLoad(fp_t, eop0_lo_alloc);
        eop1_hi = bld.CreateLoad(fp_t, eop1_hi_alloc);
        eop1_lo = bld.CreateLoad(fp_t, eop1_lo_alloc);
    } else {
        // Vector implementation.

        // Fetch the alignment of the scalar type.
        const auto align = hd::get_alignment(md, fp_t);

        // Establish the SIMD lanes for which idx != arr_size. These are the lanes
        // we will use for the gather operation.
        auto *mask = bld.CreateICmpNE(idx, arr_size_splat);

        // Construct a 32-bit int version of the mask.
        auto *mask32 = bld.CreateZExt(mask, idx->getType());

        // Compute idx + 1 as idx + mask. The idea of doing it like this is to avoid
        // potential overflows if idx == arr_size. The value of idxp1 will not matter
        // anyway in the masked-out lanes.
        auto *idxp1 = bld.CreateAdd(idx, mask32);

        // Load the date values, using nans as passhtru.
        t0 = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idx}), llvm::Align(align), mask,
                                    nan_const);
        t1 = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idxp1}), llvm::Align(align), mask,
                                    nan_const);

        // NOTE: we are now going to do a bit of type punning. The eop data is originally an array of size-2 arrays, but
        // we will be accessing it as a flat 1D array, which allows us to perform vector gathers more easily. This would
        // probably be technically undefined behaviour in C++, but it should be ok in the LLVM world.

        // Multiply idx and idxp1 by two to access the hi parts.
        //
        // NOTE: the multiplication is safe, as we checked earlier in this function that arr_size x 2 is representable
        // as a 32-bit int and the max possible value for idx/idxp1 is arr_size.
        auto *idx_hi = bld.CreateMul(idx, llvm::ConstantInt::get(idx->getType(), 2));
        auto *idxp1_hi = bld.CreateMul(idxp1, llvm::ConstantInt::get(idxp1->getType(), 2));

        // Add 1 to access the low parts.
        //
        // NOTE: again, use the mask to avoid possible overflow.
        auto *idx_lo = bld.CreateAdd(idx_hi, mask32);
        auto *idxp1_lo = bld.CreateAdd(idxp1_hi, mask32);

        // Load the hi/lo eop values.
        eop0_hi = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idx_hi}), llvm::Align(align),
                                         mask, nan_const);
        eop0_lo = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idx_lo}), llvm::Align(align),
                                         mask, nan_const);
        eop1_hi = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idxp1_hi}), llvm::Align(align),
                                         mask, nan_const);
        eop1_lo = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idxp1_lo}), llvm::Align(align),
                                         mask, nan_const);
    }

    // We can now proceed to perform linear interpolation in double-length arithmetic.
    //
    // NOTE: the eop hi/lo values are constructed already normalised.

    // Codegen the zero constant.
    auto *zero_const = llvm_codegen(s, val_t, number{0.});

    // t1 - t0.
    const auto t1_m_t0 = hd::llvm_dl_sub(s, t1, zero_const, t0, zero_const);
    // t1 - t.
    const auto t1_m_t = hd::llvm_dl_sub(s, t1, zero_const, tm_val, zero_const);
    // t - t0.
    const auto t_m_t0 = hd::llvm_dl_sub(s, tm_val, zero_const, t0, zero_const);
    // eop0*(t1-t).
    const auto tmp1 = hd::llvm_dl_mul(s, eop0_hi, eop0_lo, t1_m_t.first, t1_m_t.second);
    // eop1*(t-t0).
    const auto tmp2 = hd::llvm_dl_mul(s, eop1_hi, eop1_lo, t_m_t0.first, t_m_t0.second);
    // eop0*(t1-t)+eop1*(t-t0).
    const auto tmp3 = hd::llvm_dl_add(s, tmp1.first, tmp1.second, tmp2.first, tmp2.second);
    // eop = (eop0*(t1-t)+eop1*(t-t0))/(t1-t0).
    const auto eop_dl = hd::llvm_dl_div(s, tmp3.first, tmp3.second, t1_m_t0.first, t1_m_t0.second);
    // eop1-eop0.
    const auto tmp4 = hd::llvm_dl_sub(s, eop1_hi, eop1_lo, eop0_hi, eop0_lo);
    // eopp = (eop1-eop0)/(t1-t0).
    auto *eopp = hd::llvm_dl_div(s, tmp4.first, tmp4.second, t1_m_t0.first, t1_m_t0.second).first;

    // Reduce the eop to the [0, 2pi) range.

    // Fetch 2pi in double-length format.
    const auto [dl_twopi_hi, dl_twopi_lo] = hd::dl_twopi_like(s, fp_t);

    // Reduce eop modulo 2*pi in double-length precision.
    //
    // NOTE: we are ok here if the reduction is not 100% correct (that is, it is ok if eop is slightly outside the [0,
    // 2pi) range due to fp rounding effects).
    auto *eop = hd::llvm_dl_modulus(s, eop_dl.first, eop_dl.second, llvm_codegen(s, val_t, dl_twopi_hi),
                                    llvm_codegen(s, val_t, dl_twopi_lo))
                    .first;

    // Create the return value.
    llvm::Value *ret = llvm::UndefValue::get(ret_t);
    ret = bld.CreateInsertValue(ret, eop, 0);
    ret = bld.CreateInsertValue(ret, eopp, 1);

    bld.CreateRet(ret);

    return f;
}

} // namespace

llvm::Function *llvm_get_era_erap_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const eop_data &data)
{
    return llvm_get_eop_angle_func_dl(s, fp_t, batch_size, data, "era", heyoka::detail::llvm_get_eop_data_era);
}

llvm::Function *llvm_get_gmst82_gmst82p_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                             const eop_data &data)
{
    return llvm_get_eop_angle_func_dl(s, fp_t, batch_size, data, "gmst82", heyoka::detail::llvm_get_eop_data_gmst82);
}

// Helper to get/generate the function for the simultaneous computation of an eop quantity and its derivative via
// first-order polynomial interpolation.
//
// fp_t is the scalar floating-point value that will be used in the computation. batch_size is the batch size.
// 'data' is the source of EOP data. 'name' is the name of the eop quantity. eop_data_getter is the function to
// create/fetch the eop data.
llvm::Function *llvm_get_eop_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const eop_data &data,
                                  const char *name,
                                  llvm::Value *(*eop_data_getter)(llvm_state &, const eop_data &, llvm::Type *))
{
    assert(eop_data_getter != nullptr);

    namespace hy = heyoka;
    namespace hd = hy::detail;

    auto &md = s.module();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    // Fetch the table of EOP data.
    const auto &table = data.get_table();

    // Start by creating the mangled name of the function. The mangled name will be based on:
    //
    // - the name of the eop quantity we are computing,
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data,
    // - the floating-point type.
    //
    // NOTE: '-' is intentionally chosen as the separator between timestamp and identifier. Timestamp and identifier are
    // both guaranteed not to contain '-', thus the boundary between the two is unambiguous.
    const auto fname = fmt::format("heyoka.eop_get_{}_{}p.{}.{}-{}.{}", name, name, table.size(), data.get_timestamp(),
                                   data.get_identifier(), hd::llvm_mangle_type(val_t));

    // Check if we already created the function.
    if (auto *fptr = md.getFunction(fname)) {
        return fptr;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Setup the insertion point restorer.
    const hd::ip_restorer ipr(bld);

    // Construct the function prototype. The only input is the time value, the output is the array of
    // two values for the EOP quantity and its derivative.
    auto *ret_t = llvm::ArrayType::get(val_t, 2);
    auto *ft = llvm::FunctionType::get(ret_t, {val_t}, false);

    // Create the function
    auto *f = hd::llvm_func_create(ft, llvm::Function::PrivateLinkage, fname, &md);

    // Fetch the time argument.
    auto *tm_val = f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Get/generate the date and EOP data.
    auto *date_ptr = hd::llvm_get_eop_sw_data_date_tt_cy_j2000(s, data, fp_t, "eop");
    auto *eop_ptr = eop_data_getter(s, data, fp_t);

    // Codegen the array size (and its splatted counterpart).
    auto *arr_size = bld.getInt32(boost::numeric_cast<std::uint32_t>(table.size()));
    auto *arr_size_splat = hd::vector_splat(bld, arr_size, batch_size);

    // Locate the index in date_ptr of the time interval containing tm_val.
    auto *idx = hd::llvm_eop_sw_data_locate_date(s, date_ptr, arr_size, tm_val);

    // Codegen nan for use later.
    auto *nan_const = llvm_codegen(s, val_t, number{std::numeric_limits<double>::quiet_NaN()});

    // We can now load the data from the date/eop arrays. The loaded data will be stored in
    // the t0, t1, eop0 and eop1 variables.
    llvm::Value *t0{}, *t1{}, *eop0{}, *eop1{};

    if (batch_size == 1u) {
        // Scalar implementation.

        // Storage for the values we will be loading from the date/eop arrays.
        auto *t0_alloc = bld.CreateAlloca(fp_t);
        auto *t1_alloc = bld.CreateAlloca(fp_t);
        auto *eop0_alloc = bld.CreateAlloca(fp_t);
        auto *eop1_alloc = bld.CreateAlloca(fp_t);

        // NOTE: in the scalar implementation, we need to branch on the value of idx: if idx == arr_size, we will return
        // NaNs, otherwise we will return the values in the arrays at indices idx and idx + 1.
        hd::llvm_if_then_else(
            s, bld.CreateICmpEQ(idx, arr_size),
            [&bld, nan_const, t0_alloc, t1_alloc, eop0_alloc, eop1_alloc]() {
                // Store the nans.
                bld.CreateStore(nan_const, t0_alloc);
                bld.CreateStore(nan_const, t1_alloc);
                bld.CreateStore(nan_const, eop0_alloc);
                bld.CreateStore(nan_const, eop1_alloc);
            },
            [&bld, idx, fp_t, date_ptr, eop_ptr, t0_alloc, t1_alloc, eop0_alloc, eop1_alloc]() {
                // Compute idx + 1.
                auto *idxp1 = bld.CreateAdd(idx, bld.getInt32(1));

                // Load the date values.
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idx})), t0_alloc);
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idxp1})), t1_alloc);

                // Load the eop values.
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idx})), eop0_alloc);
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idxp1})), eop1_alloc);
            });

        // Fetch the values that we have stored in the allocs.
        t0 = bld.CreateLoad(fp_t, t0_alloc);
        t1 = bld.CreateLoad(fp_t, t1_alloc);
        eop0 = bld.CreateLoad(fp_t, eop0_alloc);
        eop1 = bld.CreateLoad(fp_t, eop1_alloc);
    } else {
        // Vector implementation.

        // Fetch the alignment of the scalar type.
        const auto align = hd::get_alignment(md, fp_t);

        // Establish the SIMD lanes for which idx != arr_size. These are the lanes
        // we will use for the gather operation.
        auto *mask = bld.CreateICmpNE(idx, arr_size_splat);

        // Construct a 32-bit int version of the mask.
        auto *mask32 = bld.CreateZExt(mask, idx->getType());

        // Compute idx + 1 as idx + mask. The idea of doing it like this is to avoid
        // potential overflows if idx == arr_size. The value of idxp1 will not matter
        // anyway in the masked-out lanes.
        auto *idxp1 = bld.CreateAdd(idx, mask32);

        // Load the date values, using nans as passhtru.
        t0 = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idx}), llvm::Align(align), mask,
                                    nan_const);
        t1 = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idxp1}), llvm::Align(align), mask,
                                    nan_const);

        // Load the eop values, using nans as passhtru.
        eop0 = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idx}), llvm::Align(align), mask,
                                      nan_const);
        eop1 = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, eop_ptr, {idxp1}), llvm::Align(align), mask,
                                      nan_const);
    }

    // We can now proceed to perform linear interpolation.

    // t1 - t0.
    auto *t1_m_t0 = hd::llvm_fsub(s, t1, t0);
    // t1 - t.
    auto *t1_m_t = hd::llvm_fsub(s, t1, tm_val);
    // t - t0.
    auto *t_m_t0 = hd::llvm_fsub(s, tm_val, t0);
    // eop0*(t1-t).
    auto *tmp1 = hd::llvm_fmul(s, eop0, t1_m_t);
    // eop1*(t-t0).
    auto *tmp2 = hd::llvm_fmul(s, eop1, t_m_t0);
    // eop0*(t1-t)+eop1*(t-t0).
    auto *tmp3 = hd::llvm_fadd(s, tmp1, tmp2);
    // eop = (eop0*(t1-t)+eop1*(t-t0))/(t1-t0).
    auto *eop = hd::llvm_fdiv(s, tmp3, t1_m_t0);
    // eop1-eop0.
    auto *tmp4 = hd::llvm_fsub(s, eop1, eop0);
    // eopp = (eop1-eop0)/(t1-t0).
    auto *eopp = hd::llvm_fdiv(s, tmp4, t1_m_t0);

    // Create the return value.
    llvm::Value *ret = llvm::UndefValue::get(ret_t);
    ret = bld.CreateInsertValue(ret, eop, 0);
    ret = bld.CreateInsertValue(ret, eopp, 1);

    bld.CreateRet(ret);

    return f;
}

namespace
{

#define HEYOKA_MODEL_DEFINE_GET_EOP_FUNC(name)                                                                         \
    llvm::Function *llvm_get_##name##_##name##p_func(llvm_state &s, llvm::Type *const fp_t,                            \
                                                     const std::uint32_t batch_size, const eop_data &data)             \
    {                                                                                                                  \
        return llvm_get_eop_func(s, fp_t, batch_size, data, #name, &heyoka::detail::llvm_get_eop_data_##name);         \
    }

HEYOKA_MODEL_DEFINE_GET_EOP_FUNC(pm_x);
HEYOKA_MODEL_DEFINE_GET_EOP_FUNC(pm_y);
HEYOKA_MODEL_DEFINE_GET_EOP_FUNC(dX);
HEYOKA_MODEL_DEFINE_GET_EOP_FUNC(dY);

#undef HEYOKA_MODEL_DEFINE_GET_EOP_FUNC

} // namespace

} // namespace model::detail

HEYOKA_END_NAMESPACE

// NOLINTBEGIN(cert-err58-cpp,bugprone-throwing-static-initialization)
HEYOKA_MODEL_DEFINE_EOP_SW(eop, era, eop_data);
HEYOKA_MODEL_DEFINE_EOP_SW(eop, gmst82, eop_data);
HEYOKA_MODEL_DEFINE_EOP_SW(eop, pm_x, eop_data);
HEYOKA_MODEL_DEFINE_EOP_SW(eop, pm_y, eop_data);
HEYOKA_MODEL_DEFINE_EOP_SW(eop, dX, eop_data);
HEYOKA_MODEL_DEFINE_EOP_SW(eop, dY, eop_data);
// NOLINTEND(cert-err58-cpp,bugprone-throwing-static-initialization)
