// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
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

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/era.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

// Helper to generate the function for the simultaneous computation of era and erap.
//
// fp_t is the scalar floating-point value that will be used in the computation. batch_size is the batch size.
// 'data' is the source of EOP data.
llvm::Function *llvm_get_era_erap_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const eop_data &data)
{
    namespace hy = heyoka;
    namespace hd = hy::detail;

    auto &md = s.module();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    // Fetch the table of EOP data.
    const auto &table = data.get_table();

    // Start by creating the mangled name of the function. The mangled name will be based on:
    //
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data,
    // - the floating-point type.
    const auto fname = fmt::format("heyoka.get_era_erap.{}.{}_{}.{}", table.size(), data.get_timestamp(),
                                   data.get_identifier(), hd::llvm_mangle_type(val_t));

    // Check if we already created the function.
    if (auto *fptr = md.getFunction(fname)) {
        return fptr;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // Construct the function prototype. The only input is the time value, the output is the array of
    // two values [era, erap].
    auto *ret_t = llvm::ArrayType::get(val_t, 2);
    auto *ft = llvm::FunctionType::get(ret_t, {val_t}, false);

    // Create the function
    auto *f = hd::llvm_func_create(ft, llvm::Function::PrivateLinkage, fname, &md);
    f->addFnAttr(llvm::Attribute::NoRecurse);
    f->addFnAttr(llvm::Attribute::NoUnwind);
    f->addFnAttr(llvm::Attribute::Speculatable);
    f->addFnAttr(llvm::Attribute::WillReturn);

    // Fetch the time argument.
    auto *tm_val = f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Get/generate the date and ERA data.
    auto *date_ptr = hd::llvm_get_eop_data_date_tt_cy_j2000(s, data, fp_t);
    auto *era_ptr = hd::llvm_get_eop_data_era(s, data, fp_t);

    // Codegen the array size (and its splatted counterpart).
    auto *arr_size = bld.getInt32(boost::numeric_cast<std::uint32_t>(table.size()));
    auto *arr_size_splat = hd::vector_splat(bld, arr_size, batch_size);

    // Locate the index in date_ptr of the time interval containing tm_val.
    auto *idx = hd::llvm_eop_data_locate_date(s, date_ptr, arr_size, tm_val);

    // Codegen nan for use later.
    auto *nan_const = llvm_codegen(s, val_t, number{std::numeric_limits<double>::quiet_NaN()});

    // Codegen the type representing the ERA data: an array of 2 scalars storing the
    // hi/lo double-length parts of the ERA.
    auto *era_t = llvm::ArrayType::get(fp_t, 2);

    // We can now load the data from the date/era arrays. The loaded data will be stored in
    // the t0, t1, era0_* and era1_* variables.
    llvm::Value *t0{}, *t1{}, *era0_hi{}, *era0_lo{}, *era1_hi{}, *era1_lo{};
    if (batch_size == 1u) {
        // Scalar implementation.

        // Storage for the values we will be loading from the date/era arrays.
        auto *t0_alloc = bld.CreateAlloca(fp_t);
        auto *t1_alloc = bld.CreateAlloca(fp_t);
        auto *era0_hi_alloc = bld.CreateAlloca(fp_t);
        auto *era0_lo_alloc = bld.CreateAlloca(fp_t);
        auto *era1_hi_alloc = bld.CreateAlloca(fp_t);
        auto *era1_lo_alloc = bld.CreateAlloca(fp_t);

        // NOTE: in the scalar implementation, we need to branch on the value of idx: if idx == arr_size, we will return
        // NaNs, otherwise we will return the values in the arrays at indices idx and idx + 1.
        hd::llvm_if_then_else(
            s, bld.CreateICmpEQ(idx, arr_size),
            [&bld, nan_const, t0_alloc, t1_alloc, era0_hi_alloc, era0_lo_alloc, era1_hi_alloc, era1_lo_alloc]() {
                // Store the nans.
                bld.CreateStore(nan_const, t0_alloc);
                bld.CreateStore(nan_const, t1_alloc);
                bld.CreateStore(nan_const, era0_hi_alloc);
                bld.CreateStore(nan_const, era0_lo_alloc);
                bld.CreateStore(nan_const, era1_hi_alloc);
                bld.CreateStore(nan_const, era1_lo_alloc);
            },
            [&bld, idx, fp_t, era_t, date_ptr, era_ptr, t0_alloc, t1_alloc, era0_hi_alloc, era0_lo_alloc, era1_hi_alloc,
             era1_lo_alloc]() {
                // Compute idx + 1.
                auto *idxp1 = bld.CreateAdd(idx, bld.getInt32(1));

                // Load the date values.
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idx})), t0_alloc);
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, date_ptr, {idxp1})), t1_alloc);

                // Load the era values.
                auto *era_0 = bld.CreateLoad(era_t, bld.CreateInBoundsGEP(era_t, era_ptr, {idx}));
                auto *era_1 = bld.CreateLoad(era_t, bld.CreateInBoundsGEP(era_t, era_ptr, {idxp1}));

                // Decompose into hi/lo parte.
                bld.CreateStore(bld.CreateExtractValue(era_0, {0}), era0_hi_alloc);
                bld.CreateStore(bld.CreateExtractValue(era_0, {1}), era0_lo_alloc);
                bld.CreateStore(bld.CreateExtractValue(era_1, {0}), era1_hi_alloc);
                bld.CreateStore(bld.CreateExtractValue(era_1, {1}), era1_lo_alloc);
            });

        // Fetch the values that we have stored in the allocs.
        t0 = bld.CreateLoad(fp_t, t0_alloc);
        t1 = bld.CreateLoad(fp_t, t1_alloc);
        era0_hi = bld.CreateLoad(fp_t, era0_hi_alloc);
        era0_lo = bld.CreateLoad(fp_t, era0_lo_alloc);
        era1_hi = bld.CreateLoad(fp_t, era1_hi_alloc);
        era1_lo = bld.CreateLoad(fp_t, era1_lo_alloc);
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

        // NOTE: we are now going to do a bit of type punning. The era data is originally an array of size-2
        // arrays, but we will be accessing it as a flat 1D array, which allows us to perform vector
        // gathers more easily. This would probably be technically undefined behaviour in C++, but it should be
        // ok in the LLVM world.

        // Multiply idx and idxp1 by two to access the hi parts.
        // NOTE: the multiplication is safe, as we checked when creating the era data that arr_size x 2
        // is representable as a 32-bit int and the max possible value for idx/idxp1 is arr_size.
        auto *idx_hi = bld.CreateMul(idx, llvm::ConstantInt::get(idx->getType(), 2));
        auto *idxp1_hi = bld.CreateMul(idxp1, llvm::ConstantInt::get(idxp1->getType(), 2));

        // Add 1 to access the low parts.
        // NOTE: again, use the mask to avoid possible overflow.
        auto *idx_lo = bld.CreateAdd(idx_hi, mask32);
        auto *idxp1_lo = bld.CreateAdd(idxp1_hi, mask32);

        // Load the hi/lo era values.
        era0_hi = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, era_ptr, {idx_hi}), llvm::Align(align),
                                         mask, nan_const);
        era0_lo = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, era_ptr, {idx_lo}), llvm::Align(align),
                                         mask, nan_const);
        era1_hi = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, era_ptr, {idxp1_hi}), llvm::Align(align),
                                         mask, nan_const);
        era1_lo = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, era_ptr, {idxp1_lo}), llvm::Align(align),
                                         mask, nan_const);
    }

    // We can now proceed to perform linear interpolation in double-length arithmetic.
    // NOTE: the ERA hi/lo values are constructed already normalised.

    // Codegen the zero constant.
    auto *zero_const = llvm_codegen(s, val_t, number{0.});

    // t1 - t0.
    const auto t1_m_t0 = hd::llvm_dl_sub(s, t1, zero_const, t0, zero_const);
    // t1 - t.
    const auto t1_m_t = hd::llvm_dl_sub(s, t1, zero_const, tm_val, zero_const);
    // t - t0.
    const auto t_m_t0 = hd::llvm_dl_sub(s, tm_val, zero_const, t0, zero_const);
    // era0*(t1-t).
    const auto tmp1 = hd::llvm_dl_mul(s, era0_hi, era0_lo, t1_m_t.first, t1_m_t.second);
    // era1*(t-t0).
    const auto tmp2 = hd::llvm_dl_mul(s, era1_hi, era1_lo, t_m_t0.first, t_m_t0.second);
    // era0*(t1-t)+era1*(t-t0).
    const auto tmp3 = hd::llvm_dl_add(s, tmp1.first, tmp1.second, tmp2.first, tmp2.second);
    // era = (era0*(t1-t)+era1*(t-t0))/(t1-t0).
    const auto era_dl = hd::llvm_dl_div(s, tmp3.first, tmp3.second, t1_m_t0.first, t1_m_t0.second);
    // era1-era0.
    const auto tmp4 = hd::llvm_dl_sub(s, era1_hi, era1_lo, era0_hi, era0_lo);
    // erap = (era1-era0)/(t1-t0).
    auto *erap = hd::llvm_dl_div(s, tmp4.first, tmp4.second, t1_m_t0.first, t1_m_t0.second).first;

    // Reduce the ERA to the [0, 2pi) range.

    // Fetch 2pi in double-length format.
    const auto [dl_twopi_hi, dl_twopi_lo] = hd::dl_twopi_like(s, fp_t);

    // Reduce era modulo 2*pi in double-length precision.
    // NOTE: we are ok here if the reduction is not 100% correct (that is, it is ok if era is slightly outside
    // the [0, 2pi) range due to fp rounding effects).
    auto *era = hd::llvm_dl_modulus(s, era_dl.first, era_dl.second, llvm_codegen(s, val_t, dl_twopi_hi),
                                    llvm_codegen(s, val_t, dl_twopi_lo))
                    .first;

    // Create the return value.
    llvm::Value *ret = llvm::UndefValue::get(ret_t);
    ret = bld.CreateInsertValue(ret, era, 0);
    ret = bld.CreateInsertValue(ret, erap, 1);

    bld.CreateRet(ret);

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return f;
}

namespace
{

// Small wrapper to check that we have eop data to work with in the era/erap
// implementations. It should never happen that we end up throwing here while
// using the public API, but better safe than sorry.
void era_erap_check_eop_data(const std::optional<eop_data> &odata)
{
    if (!odata) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument("Cannot use the era/erap functions without eop data");
        // LCOV_EXCL_STOP
    }
}

} // namespace

void era_impl::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<func_base>(*this);
    oa << m_eop_data;
}

void era_impl::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<func_base>(*this);
    ia >> m_eop_data;
}

// NOTE: this will be used only during serialisation.
era_impl::era_impl() : func_base("era_undefined", {heyoka::time}) {}

era_impl::era_impl(expression time_expr, eop_data data)
    // NOTE: we must be careful here with the name mangling. In order to match the
    // mangling for the eop data arrays, we must include:
    //
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data.
    //
    // If we do not do that, we risk in principle having functions with the same
    // name using different eop data.
    : func_base(fmt::format("era_{}_{}_{}", data.get_table().size(), data.get_timestamp(), data.get_identifier()),
                {std::move(time_expr)}),
      m_eop_data(std::move(data))
{
}

std::vector<expression> era_impl::gradient() const
{
    era_erap_check_eop_data(m_eop_data);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return {erap(kw::time_expr = args()[0], kw::eop_data = *m_eop_data)};
}

namespace
{

// Small wrapper for use in the implementation of the llvm evaluation of the era.
llvm::Value *llvm_era_eval_helper(llvm_state &s, llvm::Value *arg, llvm::Type *fp_t, std::uint32_t batch_size,
                                  const eop_data &data)
{
    // Fetch/create the function for the computation of era/erap.
    auto *era_erap_f = llvm_get_era_erap_func(s, fp_t, batch_size, data);

    // Invoke it.
    auto &bld = s.builder();
    auto *era_erap = bld.CreateCall(era_erap_f, arg);

    // Fetch the era and return it.
    return bld.CreateExtractValue(era_erap, 0);
}

} // namespace

llvm::Value *era_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    era_erap_check_eop_data(m_eop_data);

    return heyoka::detail::llvm_eval_helper(
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        [&s, fp_t, batch_size, &data = *m_eop_data](const std::vector<llvm::Value *> &args, bool) {
            return llvm_era_eval_helper(s, args[0], fp_t, batch_size, data);
        },
        *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *era_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    era_erap_check_eop_data(m_eop_data);

    return heyoka::detail::llvm_c_eval_func_helper(
        get_name(),
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        [&s, fp_t, batch_size, &data = *m_eop_data](const std::vector<llvm::Value *> &args, bool) {
            return llvm_era_eval_helper(s, args[0], fp_t, batch_size, data);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

// NOTE: here we implement a custom decomposition for era() which injects erap() into the decomposition. We do this
// because the Taylor derivatives of the era use the value of erap, thus by inserting it into the decomposition we
// can compute it once at the beginning and then re-use it afterwards.
taylor_dc_t::size_type era_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 1u);

    era_erap_check_eop_data(m_eop_data);

    // Append the erap decomposition.
    u_vars_defs.emplace_back(erap(kw::time_expr = args()[0], kw::eop_data = *m_eop_data), std::vector<std::uint32_t>{});

    // Append the era decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Setup the hidden dep for the era (the erap does not have hidden deps).
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the decomposed era).
    return u_vars_defs.size() - 1u;
}

namespace
{

// Derivative of era(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Value *taylor_diff_era_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &, const U &num,
                                  const std::vector<llvm::Value *> &,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t batch_size,
                                  const eop_data &data)
{
    namespace hd = heyoka::detail;

    if (order == 0u) {
        return llvm_era_eval_helper(s, hd::taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size), fp_t, batch_size,
                                    data);
    } else {
        return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of era(variable).
//
// NOTE: era is defined as:
//
// era(b(t)) = c_0(b(t)) + c_1(b(t))*b(t),
//
// where c0 and c1 are step functions (hence with null derivatives). Taking the first order derivative wrt t:
//
// era'(b(t)) = c_1(b(t))*b'(t),
//
// and, generalising,
//
// era^[n] = c_1 * b^[n],
//
// where c_1 is erap(b(t)).
llvm::Value *taylor_diff_era_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                  const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                                  const eop_data &data)
{
    namespace hd = heyoka::detail;

    // Fetch the index of the variable.
    const auto b_idx = hd::uname_to_index(var.name());

    // Load b^[n].
    auto *bn = hd::taylor_fetch_diff(arr, b_idx, order, n_uvars);

    if (order == 0u) {
        // Evaluate the era and return it.
        return llvm_era_eval_helper(s, bn, fp_t, batch_size, data);
    } else {
        // Fetch the value of erap from the hidden dep.
        auto *erap_val = hd::taylor_fetch_diff(arr, deps[0], 0, n_uvars);

        // Return erap*b^[n].
        return hd::llvm_fmul(s, erap_val, bn);
    }
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Value *taylor_diff_era_impl(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &, const U &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, const eop_data &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of an era()");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Value *era_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                   bool) const
{
    assert(args().size() == 1u);
    assert(deps.size() == 1u);

    era_erap_check_eop_data(m_eop_data);

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_era_impl(s, fp_t, deps, v, arr, par_ptr, n_uvars, order, batch_size, *m_eop_data);
        },
        args()[0].value());
}

namespace
{

// Derivative of era(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Function *taylor_c_diff_func_era_impl(llvm_state &s, llvm::Type *fp_t, const era_impl &fn, const U &num,
                                            std::uint32_t n_uvars, std::uint32_t batch_size, const eop_data &data)
{
    return heyoka::detail::taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, fn.get_name(), 1,
        [&s, fp_t, batch_size, &data](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_era_eval_helper(s, args[0], fp_t, batch_size, data);
        },
        num);
}

// Derivative of era(variable).
llvm::Function *taylor_c_diff_func_era_impl(llvm_state &s, llvm::Type *fp_t, const era_impl &fn, const variable &var,
                                            std::uint32_t n_uvars, std::uint32_t batch_size, const eop_data &data)
{
    namespace hd = heyoka::detail;

    auto &md = s.module();
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    const auto na_pair = hd::taylor_c_diff_func_name_args(ctx, fp_t, fn.get_name(), n_uvars, batch_size, {var}, 1);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);
    if (f != nullptr) {
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // The return type is val_t.
    auto *ft = llvm::FunctionType::get(val_t, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *ord = f->args().begin();
    auto *diff_ptr = f->args().begin() + 2;
    auto *b_idx = f->args().begin() + 5;
    auto *dep_idx = f->args().begin() + 6;

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Create the return value.
    auto *retval = bld.CreateAlloca(val_t);

    // Load b^[n].
    auto *bn = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, b_idx);

    hd::llvm_if_then_else(
        s, bld.CreateICmpEQ(ord, bld.getInt32(0)),
        [&]() {
            // For order 0, invoke the era on the order 0 of b_idx.
            auto *era_val = llvm_era_eval_helper(s, bn, fp_t, batch_size, data);

            // Store the result.
            bld.CreateStore(era_val, retval);
        },
        [&]() {
            // For order > 0, we must compute erap*b^[n].

            // Load the value of erap.
            auto *erap = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, bld.getInt32(0), dep_idx);

            // Compute and store erap*b^[n].
            bld.CreateStore(hd::llvm_fmul(s, erap, bn), retval);
        });

    // Return the result.
    bld.CreateRet(bld.CreateLoad(val_t, retval));

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return f;
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Function *taylor_c_diff_func_era_impl(llvm_state &, llvm::Type *, const era_impl &, const U &, std::uint32_t,
                                            std::uint32_t, const eop_data &)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an era() in compact mode");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Function *era_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    era_erap_check_eop_data(m_eop_data);

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_func_era_impl(s, fp_t, *this, v, n_uvars, batch_size, *m_eop_data); },
        args()[0].value());
}

void erap_impl::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<func_base>(*this);
    oa << m_eop_data;
}

void erap_impl::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<func_base>(*this);
    ia >> m_eop_data;
}

// NOTE: this will be used only during serialisation.
erap_impl::erap_impl() : func_base("erap_undefined", {heyoka::time}) {}

erap_impl::erap_impl(expression time_expr, eop_data data)
    // NOTE: we must be careful here with the name mangling. In order to match the
    // mangling for the eop data arrays, we must include:
    //
    // - the total number of rows in the eop data table,
    // - the timestamp and identifier of the eop data.
    //
    // If we do not do that, we risk in principle having functions with the same
    // name using different eop data.
    : func_base(fmt::format("erap_{}_{}_{}", data.get_table().size(), data.get_timestamp(), data.get_identifier()),
                {std::move(time_expr)}),
      m_eop_data(std::move(data))
{
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::vector<expression> erap_impl::gradient() const
{
    return {0_dbl};
}

namespace
{

// Small wrapper for use in the implementation of the llvm evaluation of the erap.
llvm::Value *llvm_erap_eval_helper(llvm_state &s, llvm::Value *arg, llvm::Type *fp_t, std::uint32_t batch_size,
                                   const eop_data &data)
{
    // Fetch/create the function for the computation of era/erap.
    auto *era_erap_f = llvm_get_era_erap_func(s, fp_t, batch_size, data);

    // Invoke it.
    auto &bld = s.builder();
    auto *era_erap = bld.CreateCall(era_erap_f, arg);

    // Fetch the erap and return it.
    return bld.CreateExtractValue(era_erap, 1);
}

} // namespace

llvm::Value *erap_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                  llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                  bool high_accuracy) const
{
    era_erap_check_eop_data(m_eop_data);

    return heyoka::detail::llvm_eval_helper(
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        [&s, fp_t, batch_size, &data = *m_eop_data](const std::vector<llvm::Value *> &args, bool) {
            return llvm_erap_eval_helper(s, args[0], fp_t, batch_size, data);
        },
        *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *erap_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    era_erap_check_eop_data(m_eop_data);

    return heyoka::detail::llvm_c_eval_func_helper(
        get_name(),
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        [&s, fp_t, batch_size, &data = *m_eop_data](const std::vector<llvm::Value *> &args, bool) {
            return llvm_erap_eval_helper(s, args[0], fp_t, batch_size, data);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

namespace
{

// Derivative of erap(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Value *taylor_diff_erap_impl(llvm_state &s, llvm::Type *fp_t, const U &num, const std::vector<llvm::Value *> &,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t batch_size,
                                   const eop_data &data)
{
    namespace hd = heyoka::detail;

    if (order == 0u) {
        return llvm_erap_eval_helper(s, hd::taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size), fp_t,
                                     batch_size, data);
    } else {
        return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of erap(variable).
//
// NOTE: the derivatives of erap() are all zero (except of course for the order-0 derivative).
llvm::Value *taylor_diff_erap_impl(llvm_state &s, llvm::Type *fp_t, const variable &var,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                                   const eop_data &data)
{
    namespace hd = heyoka::detail;

    if (order == 0u) {
        // Fetch the index of the variable.
        const auto b_idx = hd::uname_to_index(var.name());

        // Load b^[0].
        auto *b0 = hd::taylor_fetch_diff(arr, b_idx, 0, n_uvars);

        // Evaluate the erap and return it.
        return llvm_erap_eval_helper(s, b0, fp_t, batch_size, data);
    } else {
        // Return zero.
        return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Value *taylor_diff_erap_impl(llvm_state &, llvm::Type *, const U &, const std::vector<llvm::Value *> &,
                                   llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, const eop_data &)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of an erap()");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Value *erap_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                    bool) const
{
    assert(args().size() == 1u);

    era_erap_check_eop_data(m_eop_data);

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_erap_impl(s, fp_t, v, arr, par_ptr, n_uvars, order, batch_size, *m_eop_data);
        },
        args()[0].value());
}

namespace
{

// Derivative of erap(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Function *taylor_c_diff_func_erap_impl(llvm_state &s, llvm::Type *fp_t, const erap_impl &fn, const U &num,
                                             std::uint32_t n_uvars, std::uint32_t batch_size, const eop_data &data)
{
    return heyoka::detail::taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, fn.get_name(), 0,
        [&s, fp_t, batch_size, &data](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_erap_eval_helper(s, args[0], fp_t, batch_size, data);
        },
        num);
}

// Derivative of erap(variable).
llvm::Function *taylor_c_diff_func_erap_impl(llvm_state &s, llvm::Type *fp_t, const erap_impl &fn, const variable &var,
                                             std::uint32_t n_uvars, std::uint32_t batch_size, const eop_data &data)
{
    namespace hd = heyoka::detail;

    auto &md = s.module();
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    const auto na_pair = hd::taylor_c_diff_func_name_args(ctx, fp_t, fn.get_name(), n_uvars, batch_size, {var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);
    if (f != nullptr) {
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // The return type is val_t.
    auto *ft = llvm::FunctionType::get(val_t, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *ord = f->args().begin();
    auto *diff_ptr = f->args().begin() + 2;
    auto *b_idx = f->args().begin() + 5;

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Create the return value.
    auto *retval = bld.CreateAlloca(val_t);

    hd::llvm_if_then_else(
        s, bld.CreateICmpEQ(ord, bld.getInt32(0)),
        [&]() {
            // For order 0, invoke the erap on the order 0 of b_idx.

            // Load b^[0].
            auto *b0 = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, bld.getInt32(0), b_idx);

            // Compute the erap.
            auto *erap_val = llvm_erap_eval_helper(s, b0, fp_t, batch_size, data);

            // Store the result.
            bld.CreateStore(erap_val, retval);
        },
        [&]() {
            // For order > 0, we return 0.
            bld.CreateStore(llvm_codegen(s, val_t, number{0.}), retval);
        });

    // Return the result.
    bld.CreateRet(bld.CreateLoad(val_t, retval));

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return f;
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Function *taylor_c_diff_func_erap_impl(llvm_state &, llvm::Type *, const erap_impl &, const U &, std::uint32_t,
                                             std::uint32_t, const eop_data &)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an erap() in compact mode");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Function *erap_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                              std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    era_erap_check_eop_data(m_eop_data);

    return std::visit(
        [&](const auto &v) {
            return taylor_c_diff_func_erap_impl(s, fp_t, *this, v, n_uvars, batch_size, *m_eop_data);
        },
        args()[0].value());
}

expression era_func_impl(expression time_expr, eop_data data)
{
    return expression{func{era_impl{std::move(time_expr), std::move(data)}}};
}

expression erap_func_impl(expression time_expr, eop_data data)
{
    return expression{func{erap_impl{std::move(time_expr), std::move(data)}}};
}

} // namespace model::detail

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::model::detail::era_impl)

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::model::detail::erap_impl)
