// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL)

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/erfa_decls.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/dayfrac.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// Convert the input TT time into the number of days elapsed since January 1st.
//
// The input time is measured in days since the epoch of J2000. The return value is the number of TT days elapsed since
// January 1st 00:00 UTC of the calendar year of tt.
double tt_to_dayfrac(const double tt) noexcept
{
    using heyoka::detail::dfloat;
    using heyoka::detail::eft_add_knuth;
    using heyoka::detail::get_logger;

    // Step 0: convert tt into a double-length normalised Julian date.
    //
    // NOTE: 2451545.0 is the TT Julian date of J2000.
    const auto d_tt = dfloat(eft_add_knuth(2451545.0, tt));

    // Step 1: convert the input TT JD to UTC.
    double tai1{}, tai2{};
    // NOTE: eraTttai() always returns 0, no need to check.
    eraTttai(d_tt.hi, d_tt.lo, &tai1, &tai2);
    double utc1{}, utc2{};
    if (const auto ret = eraTaiutc(tai1, tai2, &utc1, &utc2); ret != 0) [[unlikely]] {
        // LCOV_EXCL_START
        get_logger()->warn("Converting the TAI JD ({}, {}) to UTC in heyoka_tt_to_dayfrac() resulted in error code {}",
                           tai1, tai2, ret);
        // LCOV_EXCL_STOP
    }

    // Step 2: extract the year from the UTC JD.
    int iy{}, im{}, id{};
    double fd{};
    if (const auto ret = eraJd2cal(utc1, utc2, &iy, &im, &id, &fd); ret != 0) [[unlikely]] {
        // LCOV_EXCL_START
        get_logger()->warn(
            "Converting the UTC JD ({}, {}) to a calendar date in heyoka_tt_to_dayfrac() resulted in error code {}",
            utc1, utc2, ret);
        // LCOV_EXCL_STOP
    }

    // Step 3: convert the UTC year into a UTC JD.
    double yutc1{}, yutc2{};
    if (const auto ret = eraCal2jd(iy, 1, 1, &yutc1, &yutc2); ret != 0) [[unlikely]] {
        // LCOV_EXCL_START
        get_logger()->warn(
            "Converting the calendar date ({}, 1, 1) to a UTC JD in heyoka_tt_to_dayfrac() resulted in error code {}",
            iy, ret);
        // LCOV_EXCL_STOP
    }

    // Step 4: convert the UTC year to TT.
    double ytai1{}, ytai2{};
    if (const auto ret = eraUtctai(yutc1, yutc2, &ytai1, &ytai2); ret != 0) [[unlikely]] {
        // LCOV_EXCL_START
        get_logger()->warn("Converting the UTC JD ({}, {}) to TAI in heyoka_tt_to_dayfrac() resulted in error code {}",
                           yutc1, yutc2, ret);
        // LCOV_EXCL_STOP
    }
    double ytt1{}, ytt2{};
    // NOTE: eraTaitt() always returns 0, no need to check.
    eraTaitt(ytai1, ytai2, &ytt1, &ytt2);

    // Step 5: convert to dfloat and compute the return value.
    const auto d_ytt = dfloat(eft_add_knuth(ytt1, ytt2));

    const auto d_ret = d_tt - d_ytt;
    const auto ret = static_cast<double>(d_ret);

    if (!isfinite(d_ret)) [[unlikely]] {
        // LCOV_EXCL_START
        get_logger()->warn("A non-finite return value of ({}, {}) was generated in heyoka_tt_to_dayfrac()", d_ret.hi,
                           d_ret.lo);
        // LCOV_EXCL_STOP
    }

    // NOTE: negative return values could occur due to floating-point rounding.
    return (ret >= 0) ? ret : 0.;
}

} // namespace

void dayfrac_impl::save(boost::archive::binary_oarchive &oa, const unsigned) const
{
    oa << boost::serialization::base_object<func_base>(*this);
}

void dayfrac_impl::load(boost::archive::binary_iarchive &ia, const unsigned)
{
    ia >> boost::serialization::base_object<func_base>(*this);
}

dayfrac_impl::dayfrac_impl() : dayfrac_impl(heyoka::time) {}

dayfrac_impl::dayfrac_impl(expression e) : func_base("dayfrac", {std::move(e)}) {}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::vector<expression> dayfrac_impl::gradient() const
{
    return {1_dbl};
}

expression dayfrac_func_impl(expression tm)
{
    return expression{func{dayfrac_impl{std::move(tm)}}};
}

namespace
{

// Helper to convert a single scalar floating-point value into a double-precision value.
llvm::Value *fp_to_dbl(llvm_state &s, llvm::Value *x)
{
    namespace hd = heyoka::detail;

    assert(x != nullptr);

    // Fetch the type of the argument.
    auto *x_t = x->getType();

    assert(!llvm::isa<llvm::FixedVectorType>(x_t));

    auto &bld = s.builder();

    auto *dbl_t = bld.getDoubleTy();
    if (x_t == dbl_t) {
        // x is a double already, just return it.
        return x;
    }

    if (x_t->isFloatTy()) {
        // x is a float, extend it to double.
        return bld.CreateFPExt(x, dbl_t);
    }

    if (x_t->isX86_FP80Ty() || x_t->isFP128Ty()) {
        // x is one of the supported floating-point types wider than double. Truncate it.
        return bld.CreateFPTrunc(x, dbl_t);
    }

#if defined(HEYOKA_HAVE_REAL)

    if (hd::llvm_is_real(x_t) > 0) {
        // x is a real.
        return hd::llvm_real_to_double(s, x);
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(fmt::format(
        "Unable to convert an LLVM value of type '{}' into a double-precision value", hd::llvm_type_name(x_t)));
    // LCOV_EXCL_STOP
}

// Helper to convert the single scalar double-precision value x into the floating-point type fp_t.
llvm::Value *dbl_to_fp(llvm_state &s, llvm::Value *x, llvm::Type *fp_t)
{
    namespace hd = heyoka::detail;

    assert(x != nullptr);
    assert(fp_t != nullptr);
    assert(!llvm::isa<llvm::FixedVectorType>(fp_t));
    assert(x->getType()->isDoubleTy());

    auto &bld = s.builder();

    if (fp_t->isDoubleTy()) {
        // fp_t is double, return x unchanged.
        return x;
    }

    if (fp_t->isFloatTy()) {
        // fp_t is float, truncate x and return.
        return bld.CreateFPTrunc(x, fp_t);
    }

    if (fp_t->isX86_FP80Ty() || fp_t->isFP128Ty()) {
        // fp_t is one of the supported floating-point types wider than double. Extend x and return.
        return bld.CreateFPExt(x, fp_t);
    }

#if defined(HEYOKA_HAVE_REAL)

    if (hd::llvm_is_real(fp_t) > 0) {
        // fp_t is a real type.
        return hd::llvm_double_to_real(s, x, fp_t);
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Unable to convert an LLVM double-precision value to the type '{}'", hd::llvm_type_name(fp_t)));
    // LCOV_EXCL_STOP
}

// LLVM evaluation of dayfrac().
llvm::Value *dayfrac_llvm_eval_impl(llvm_state &s, llvm::Value *x)
{
    namespace hd = heyoka::detail;

    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the type of the argument.
    auto *x_t = x->getType();

    // Prepare the input/output array to invoke the C function.
    const auto vector_size = hd::get_vector_size(x);
    auto *dbl_t = bld.getDoubleTy();
    auto *inout_arr_t = llvm::ArrayType::get(dbl_t, vector_size);
    auto *inout_arr = bld.CreateAlloca(inout_arr_t);

    // Fetch a pointer to the beginning of the array.
    auto *arr_begin_ptr = bld.CreateInBoundsGEP(inout_arr_t, inout_arr, {bld.getInt32(0), bld.getInt32(0)});

    // Build the attributes for the invocation of the external function.
    const auto attr_list = llvm::AttributeList::get(ctx, llvm::AttributeList::FunctionIndex,
                                                    {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});

    if (llvm::isa<llvm::FixedVectorType>(x_t)) {
        // x is a vector.

        // Decompose x into scalars.
        auto scalars = hd::vector_to_scalars(bld, x);

        // Convert the scalars to double and store them into the array.
        for (std::uint32_t i = 0; i < vector_size; ++i) {
            bld.CreateStore(fp_to_dbl(s, scalars[i]), bld.CreateInBoundsGEP(dbl_t, arr_begin_ptr, {bld.getInt32(i)}));
        }

        // Invoke the external function.
        hd::llvm_invoke_external(s, "heyoka_tt_to_dayfrac", bld.getVoidTy(), {arr_begin_ptr, bld.getInt32(vector_size)},
                                 attr_list);

        // Load the return values from the array and convert them to the scalar counterpart of x_t.
        for (std::uint32_t i = 0; i < vector_size; ++i) {
            auto *dbl_ret = bld.CreateLoad(dbl_t, bld.CreateInBoundsGEP(dbl_t, arr_begin_ptr, {bld.getInt32(i)}));
            scalars[i] = dbl_to_fp(s, dbl_ret, x_t->getScalarType());
        }

        // Turn the scalars into a vector and return.
        return hd::scalars_to_vector(bld, scalars);
    } else {
        // x is a scalar.
        assert(vector_size == 1u);

        // Transform x into a double and store it into the array.
        bld.CreateStore(fp_to_dbl(s, x), arr_begin_ptr);

        // Invoke the external function.
        hd::llvm_invoke_external(s, "heyoka_tt_to_dayfrac", bld.getVoidTy(), {arr_begin_ptr, bld.getInt32(1)},
                                 attr_list);

        // Load the return value from the array.
        auto *dbl_ret = bld.CreateLoad(dbl_t, arr_begin_ptr);

        // Convert it to x_t and return it.
        return dbl_to_fp(s, dbl_ret, x_t);
    }
}

} // namespace

llvm::Value *dayfrac_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                     llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                     bool high_accuracy) const
{
    return heyoka::detail::llvm_eval_helper(
        [&s](const std::vector<llvm::Value *> &args, bool) { return dayfrac_llvm_eval_impl(s, args[0]); }, *this, s,
        fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *dayfrac_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                               bool high_accuracy) const
{
    return heyoka::detail::llvm_c_eval_func_helper(
        get_name(), [&s](const std::vector<llvm::Value *> &args, bool) { return dayfrac_llvm_eval_impl(s, args[0]); },
        *this, s, fp_t, batch_size, high_accuracy);
}

namespace
{

// Derivative of dayfrac(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Value *taylor_diff_dayfrac_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &, const U &num,
                                      const std::vector<llvm::Value *> &,
                                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                      llvm::Value *par_ptr, std::uint32_t, std::uint32_t order,
                                      std::uint32_t batch_size)
{
    namespace hd = heyoka::detail;

    if (order == 0u) {
        return dayfrac_llvm_eval_impl(s, hd::taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size));
    } else {
        return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of dayfrac(variable).
//
// NOTE: dayfrac is defined as follows:
//
// dayfrac(b(t)) = b(t) + c(b(t)),
//
// where c is a step function (hence with null derivatives). Taking the first order derivative wrt t:
//
// dayfrac'(b(t)) = b'(t),
//
// and, generalising,
//
// dayfrac^[n] = b^[n].
llvm::Value *taylor_diff_dayfrac_impl(llvm_state &s, llvm::Type *, const std::vector<std::uint32_t> &,
                                      const variable &var, const std::vector<llvm::Value *> &arr, llvm::Value *,
                                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                      std::uint32_t n_uvars, std::uint32_t order, std::uint32_t)
{
    namespace hd = heyoka::detail;

    // Fetch the index of the variable.
    const auto b_idx = hd::uname_to_index(var.name());

    // Load b^[n].
    auto *bn = hd::taylor_fetch_diff(arr, b_idx, order, n_uvars);

    if (order == 0u) {
        // Evaluate dayfrac for b^[0] and return it.
        return dayfrac_llvm_eval_impl(s, bn);
    } else {
        // Just return b^[n].
        return bn;
    }
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Value *taylor_diff_dayfrac_impl(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &, const U &,
                                      const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                      std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of dayfrc()");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Value *dayfrac_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                       const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                       std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                       std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);
    assert(deps.empty());

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_dayfrac_impl(s, fp_t, deps, v, arr, par_ptr, n_uvars, order, batch_size);
        },
        args()[0].value());
}

namespace
{

// Derivative of dayfrac(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Function *taylor_c_diff_func_dayfrac_impl(llvm_state &s, llvm::Type *fp_t, const dayfrac_impl &fn, const U &num,
                                                std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return heyoka::detail::taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, fn.get_name(), 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return dayfrac_llvm_eval_impl(s, args[0]);
        },
        num);
}

// Derivative of dayfrac(variable).
llvm::Function *taylor_c_diff_func_dayfrac_impl(llvm_state &s, llvm::Type *fp_t, const dayfrac_impl &fn,
                                                const variable &var, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    namespace hd = heyoka::detail;

    auto &md = s.module();
    auto &bld = s.builder();
    auto &ctx = s.context();

    const auto [fname, fargs] = hd::taylor_c_diff_func_name_args(ctx, fp_t, fn.get_name(), n_uvars, batch_size, {var});

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);
    if (f != nullptr) {
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

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

    // Load b^[n].
    auto *bn = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, b_idx);

    hd::llvm_if_then_else(
        s, bld.CreateICmpEQ(ord, bld.getInt32(0)),
        [&]() {
            // For order 0, compute dayfrac() for the order 0 of b_idx.

            // Evaluate.
            auto *dayfrac_val = dayfrac_llvm_eval_impl(s, bn);

            // Store the result.
            bld.CreateStore(dayfrac_val, retval);
        },
        [&]() {
            // For order > 0, we just return b^[n].
            bld.CreateStore(bn, retval);
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
llvm::Function *taylor_c_diff_func_dayfrac_impl(llvm_state &, llvm::Type *, const dayfrac_impl &, const U &,
                                                std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of dayfrac() in compact mode");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Function *dayfrac_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                 std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    return std::visit(
        [&](const auto &v) { return taylor_c_diff_func_dayfrac_impl(s, fp_t, *this, v, n_uvars, batch_size); },
        args()[0].value());
}

} // namespace model::detail

HEYOKA_END_NAMESPACE

// NOTE: this is the implementation function called from within LLVM.
extern "C" HEYOKA_DLL_PUBLIC void heyoka_tt_to_dayfrac(double *const inout, const std::uint32_t size) noexcept
{
    assert(size > 0u);

    for (std::uint32_t i = 0; i < size; ++i) {
        inout[i] = heyoka::model::detail::tt_to_dayfrac(inout[i]);
    }
}

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::model::detail::dayfrac_impl)
