// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#if defined(HEYOKA_HAVE_REAL)

#include <cassert>
#include <charconv>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/Config/llvm-config.h>
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
#include <llvm/Support/Casting.h>

#include <mp++/integer.hpp>
#include <mp++/real.hpp>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/real_helpers.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Various static checks.
static_assert(sizeof(mppp::real) == sizeof(mppp::mpfr_struct_t));
static_assert(alignof(mppp::real) == alignof(mppp::mpfr_struct_t));
static_assert(mppp::real_prec_min() > 0);
static_assert(std::is_signed_v<mpfr_prec_t>);
static_assert(std::is_signed_v<mpfr_sign_t>);
static_assert(std::is_signed_v<mpfr_exp_t>);
static_assert(std::is_signed_v<real_rnd_t>);
// NOTE: we want to make extra sure long long can represent any mpfr_prec_t.
static_assert(std::numeric_limits<mpfr_prec_t>::max() <= std::numeric_limits<long long>::max());

// Helper to generate the function attributes list to
// be used when invoking MPFR primitives.
llvm::AttributeList get_mpfr_attr_list(llvm::LLVMContext &context)
{
    return llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                    {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});
}

} // namespace

// Determine if the input type is heyoka.real.N,
// and, in such case, return N. Otherwise, return 0.
mpfr_prec_t llvm_is_real(llvm::Type *t)
{
    if (auto *ptr = llvm::dyn_cast<llvm::StructType>(t)) {
        const auto sname = ptr->getStructName();

        if (
#if LLVM_VERSION_MAJOR < 18
            sname.startswith("heyoka.real.")
#else
            sname.starts_with("heyoka.real.")
#endif
        ) {
            // LCOV_EXCL_START
            if (sname.size() <= 12u) {
                throw std::invalid_argument(fmt::format(
                    "Invalid name detected for an LLVM type corresponding to mppp::real: '{}'", std::string(sname)));
            }
            // LCOV_EXCL_STOP

            mpfr_prec_t value = 0;

            const auto ret = std::from_chars(sname.data() + 12, sname.data() + sname.size(), value);

            // LCOV_EXCL_START
            if (ret.ec != std::errc{}) {
                throw std::invalid_argument("The determination of the precision of an LLVM type corresponding to "
                                            "mppp::real resulted in an error condition");
            }

            if (value < mppp::real_prec_min() || value > mppp::real_prec_max()) {
                throw std::invalid_argument(fmt::format(
                    "An invalid precision of {} was determined for an LLVM type corresponding to mppp::real", value));
            }
            // LCOV_EXCL_STOP

            // Double check the limb array size is consistent with the precision value.
            assert(mppp::prec_to_nlimbs(value) == ptr->elements()[2]->getArrayNumElements());

            return value;
        }
    }

    return 0;
}

// Negation.
llvm::Value *llvm_real_fneg(llvm_state &s, llvm ::Value *x)
{
    assert(x != nullptr);
    assert(llvm_is_real(x->getType()) != 0);

    auto &builder = s.builder();

    // NOTE: the current implementation of mpfr_neg() just flips
    // the sign of the _mpfr_sign member (see the MPFR_CHANGE_SIGN()
    // macro in the MPFR source tree). Thus, we do the same thing.
    auto *orig_sign = builder.CreateExtractValue(x, {0});
    auto *new_sign = builder.CreateNeg(orig_sign);

    return builder.CreateInsertValue(x, new_sign, {0});
}

namespace
{

// Small helper to codegen the MPFR_RNDN constant.
llvm::Constant *llvm_mpfr_rndn(llvm_state &s)
{
    return llvm::ConstantInt::getSigned(to_llvm_type<real_rnd_t>(s.context()),
                                        boost::numeric_cast<std::int64_t>(static_cast<real_rnd_t>(MPFR_RNDN)));
}

// Small helper to codegen an MPFR precision value
// as an LLVM constant.
llvm::Constant *llvm_mpfr_prec(llvm_state &s, mpfr_prec_t prec)
{
    return llvm::ConstantInt::getSigned(to_llvm_type<mpfr_prec_t>(s.context()),
                                        boost::numeric_cast<std::int64_t>(prec));
}

// Construct an mpfr view from the input heyoka.real.N r. An mpfr view is a pair consisting
// of 1) an mpfr_struct_t instance and 2) the limb array to which the mpfr_struct_t points.
std::pair<llvm::Value *, llvm::Value *> llvm_real_to_mpfr_view(llvm_state &s, llvm::Value *r)
{
    const auto real_prec = llvm_is_real(r->getType());

    assert(real_prec != 0);

    auto &builder = s.builder();

    // Generate the precision as an LLVM constant.
    auto *prec_const = llvm_mpfr_prec(s, real_prec);

    // Fetch the limb array type.
    auto *struct_fp_t = llvm::cast<llvm::StructType>(r->getType());
    auto *limb_arr_t = struct_fp_t->getElementType(2u);

    // Create the limb array and store into it the limbs from r.
    auto *limb_arr = builder.CreateAlloca(limb_arr_t);
    builder.CreateStore(builder.CreateExtractValue(r, {2u}), limb_arr);

    // Create the mpfr_struct_t.
    auto *real_t = to_llvm_type<mppp::real>(s.context());
    auto *mpfr_struct_inst = builder.CreateAlloca(real_t);

    // Store the precision.
    builder.CreateStore(
        prec_const, builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(0)}));

    // Store the sign.
    builder.CreateStore(
        builder.CreateExtractValue(r, {0u}),
        builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(1)}));

    // Store the exponent.
    builder.CreateStore(
        builder.CreateExtractValue(r, {1u}),
        builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(2)}));

    // Store the pointer to the limb array.
    builder.CreateStore(
        builder.CreateInBoundsGEP(limb_arr_t, limb_arr, {builder.getInt32(0), builder.getInt32(0)}),
        builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(3)}));

    return {mpfr_struct_inst, limb_arr};
}

// Create an mpfr view with an undefined value, with the precision of the input type fp_t
// (which must be a heyoka.real.N). This is used to create return values for the functions in the mpfr API.
std::pair<llvm::Value *, llvm::Value *> llvm_undef_mpfr_view(llvm_state &s, llvm::Type *fp_t)
{
    const auto real_prec = llvm_is_real(fp_t);

    assert(real_prec != 0);

    auto &builder = s.builder();

    // Generate the precision as an LLVM constant.
    auto *prec_const = llvm_mpfr_prec(s, real_prec);

    // Fetch the limb array type.
    auto *struct_fp_t = llvm::cast<llvm::StructType>(fp_t);
    auto *limb_arr_t = struct_fp_t->getElementType(2u);

    // Create the limb array.
    // NOTE: the limb array will contain undefined values,
    // under the assumption that the MPFR functions only care about the
    // precision of the result (and not sign, exponent and significand).
    // If that turns out not to be true, we can always codegen a zero real
    // constant with appropriate precision and use its data, instead of leaving
    // things undefined.
    // NOTE: currently the mpfr_custom_init_set() macro sets something for sign and
    // exponent, in addition to the precision. Perhaps we could invoke it here and
    // then pick up the sign/exponent values as compile-time constants?
    auto *limb_arr = builder.CreateAlloca(limb_arr_t);

    // Create the mpfr_struct_t.
    auto *real_t = to_llvm_type<mppp::real>(s.context());
    auto *mpfr_struct_inst = builder.CreateAlloca(real_t);

    // Store the precision.
    builder.CreateStore(
        prec_const, builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(0)}));

    // Store the pointer to the limb array.
    builder.CreateStore(
        builder.CreateInBoundsGEP(limb_arr_t, limb_arr, {builder.getInt32(0), builder.getInt32(0)}),
        builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(3)}));

    return {mpfr_struct_inst, limb_arr};
}

// Load the data from the input mpfr view (mpfr_struct_inst, limb_arr) into a heyoka.real.N
// instance of type fp_t.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_mpfr_view_to_real(llvm_state &s, llvm::Value *mpfr_struct_inst, llvm::Value *limb_arr,
                                    llvm::Type *fp_t)
{
    assert(llvm_is_real(fp_t) != 0);

    auto &builder = s.builder();

    auto *real_t = to_llvm_type<mppp::real>(s.context());
    auto *struct_fp_t = llvm::cast<llvm::StructType>(fp_t);
    auto *limb_arr_t = struct_fp_t->getElementType(2u);

    // Init the return value.
    llvm::Value *res = llvm::UndefValue::get(fp_t);

#if !defined(NDEBUG)

    // In debug mode, double check that the precision in the view matches
    // the precision of fp_t.

    auto *prec_t = to_llvm_type<mpfr_prec_t>(s.context());

    // Load the precision value from the view.
    auto *prec_ptr = builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(0)});
    auto *prec_value = builder.CreateLoad(prec_t, prec_ptr);

    // Check that it matches the precision of fp_t.
    llvm_invoke_external(s, "heyoka_assert_real_match_precs_mpfr_view_to_real", builder.getVoidTy(),
                         {prec_value, llvm_mpfr_prec(s, llvm_is_real(fp_t))});

#endif

    auto *sign_ptr = builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(1)});
    res = builder.CreateInsertValue(res, builder.CreateLoad(struct_fp_t->getElementType(0u), sign_ptr), {0});

    auto *exp_ptr = builder.CreateInBoundsGEP(real_t, mpfr_struct_inst, {builder.getInt32(0), builder.getInt32(2)});
    res = builder.CreateInsertValue(res, builder.CreateLoad(struct_fp_t->getElementType(1u), exp_ptr), {1});

    res = builder.CreateInsertValue(res, builder.CreateLoad(limb_arr_t, limb_arr), {2});

    return res;
}

} // namespace

// Helper to construct an n-ary LLVM function corresponding to the MPFR primitive 'mpfr_name'.
// The operands will be of type 'fp_t' (which must be a heyoka.real.N). The name of the function
// will be built from mpfr_name. The return type of the MPFR primitive is assumed to be int.
llvm::Function *real_nary_op(llvm_state &s, llvm::Type *fp_t, const std::string &mpfr_name, unsigned nargs)
{
    assert(nargs > 0u);

    auto &md = s.module();
    auto &context = s.context();
    auto &builder = s.builder();

    const auto real_prec = llvm_is_real(fp_t);

    assert(real_prec > 0);

    const auto fname = fmt::format("heyoka.real.{}.{}", real_prec, mpfr_name);

    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        auto *orig_bb = builder.GetInsertBlock();

        // Create the function type and the function.
        const std::vector<llvm::Type *> fargs(boost::numeric_cast<std::vector<llvm::Type *>::size_type>(nargs), fp_t);
        auto *ft = llvm::FunctionType::get(fp_t, fargs, false);
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);
        f->addFnAttr(llvm::Attribute::NoUnwind);
        f->addFnAttr(llvm::Attribute::Speculatable);
        f->addFnAttr(llvm::Attribute::WillReturn);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create an undef value for the result and add it as first function argument.
        auto [real_res, limb_arr_res] = llvm_undef_mpfr_view(s, fp_t);
        std::vector<llvm::Value *> mpfr_args;
        mpfr_args.push_back(real_res);

        // Create the mpfr views for the input arguments and add them as function arguments.
        for (auto i = 0u; i < nargs; ++i) {
            mpfr_args.push_back(llvm_real_to_mpfr_view(s, f->args().begin() + i).first);
        }

        // Add the rounding mode.
        mpfr_args.push_back(llvm_mpfr_rndn(s));

        // Invoke the MPFR primitive.
        llvm_invoke_external(s, mpfr_name, to_llvm_type<int>(context), mpfr_args, get_mpfr_attr_list(context));

        // Assemble the result.
        auto *res = llvm_mpfr_view_to_real(s, real_res, limb_arr_res, fp_t);

        builder.CreateRet(res);

        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Compute sin/cos at the same time.
// NOTE: this needs a custom implementation due to the double
// return value in the MPFR primitive.
std::pair<llvm::Value *, llvm::Value *> llvm_real_sincos(llvm_state &s, llvm::Value *x)
{
    auto &md = s.module();
    auto &context = s.context();
    auto &builder = s.builder();

    auto *fp_t = x->getType();

    const auto real_prec = llvm_is_real(fp_t);

    assert(real_prec > 0);

    const auto fname = fmt::format("heyoka.real.{}.sincos", real_prec);

    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        auto *orig_bb = builder.GetInsertBlock();

        // Create the function type and the function.
        auto *ret_t = llvm::ArrayType::get(fp_t, 2);
        auto *ft = llvm::FunctionType::get(ret_t, {fp_t}, false);
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);
        f->addFnAttr(llvm::Attribute::NoUnwind);
        f->addFnAttr(llvm::Attribute::Speculatable);
        f->addFnAttr(llvm::Attribute::WillReturn);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create undef values for the results and add them as initial function arguments.
        auto [real_res_sin, limb_arr_res_sin] = llvm_undef_mpfr_view(s, fp_t);
        auto [real_res_cos, limb_arr_res_cos] = llvm_undef_mpfr_view(s, fp_t);

        std::vector<llvm::Value *> mpfr_args{real_res_sin, real_res_cos};

        // Create the mpfr view for the input argument and add it to the function arguments.
        mpfr_args.push_back(llvm_real_to_mpfr_view(s, f->args().begin()).first);

        // Add the rounding mode.
        mpfr_args.push_back(llvm_mpfr_rndn(s));

        // Invoke the MPFR primitive.
        llvm_invoke_external(s, "mpfr_sin_cos", to_llvm_type<int>(context), mpfr_args, get_mpfr_attr_list(context));

        // Assemble the result.
        auto *res_sin = llvm_mpfr_view_to_real(s, real_res_sin, limb_arr_res_sin, fp_t);
        auto *res_cos = llvm_mpfr_view_to_real(s, real_res_cos, limb_arr_res_cos, fp_t);

        llvm::Value *res = llvm::UndefValue::get(ret_t);
        res = builder.CreateInsertValue(res, res_sin, {0});
        res = builder.CreateInsertValue(res, res_cos, {1});

        builder.CreateRet(res);

        builder.SetInsertPoint(orig_bb);
    }

    auto *ret = builder.CreateCall(f, {x});

    return {builder.CreateExtractValue(ret, {0}), builder.CreateExtractValue(ret, {1})};
}

namespace
{

// Helper to construct an n-ary LLVM function corresponding to the MPFR comparison primitive 'mpfr_name'.
// The operands will be of type 'fp_t' (which must be a heyoka.real.N), the return type is bool. The name of the
// function will be built from mpfr_name. The MPFR primitive must not require a rounding mode argument and it must
// return an int.
llvm::Function *real_nary_cmp(llvm_state &s, llvm::Type *fp_t, const std::string &mpfr_name, unsigned nargs)
{
    assert(nargs > 0u);

    auto &md = s.module();
    auto &context = s.context();
    auto &builder = s.builder();

    const auto real_prec = llvm_is_real(fp_t);

    assert(real_prec > 0);

    const auto fname = fmt::format("heyoka.real.{}.{}", real_prec, mpfr_name);

    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        auto *orig_bb = builder.GetInsertBlock();

        // Create the function type and the function.
        const std::vector<llvm::Type *> fargs(boost::numeric_cast<std::vector<llvm::Type *>::size_type>(nargs), fp_t);
        auto *ft = llvm::FunctionType::get(builder.getInt1Ty(), fargs, false);
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);
        f->addFnAttr(llvm::Attribute::NoUnwind);
        f->addFnAttr(llvm::Attribute::Speculatable);
        f->addFnAttr(llvm::Attribute::WillReturn);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the mpfr views for the input arguments and add them as function arguments.
        std::vector<llvm::Value *> mpfr_args;
        mpfr_args.reserve(boost::numeric_cast<decltype(mpfr_args.size())>(nargs));
        for (auto i = 0u; i < nargs; ++i) {
            mpfr_args.push_back(llvm_real_to_mpfr_view(s, f->args().begin() + i).first);
        }

        // Invoke the MPFR primitive.
        auto *cmp_ret
            = llvm_invoke_external(s, mpfr_name, to_llvm_type<int>(context), mpfr_args, get_mpfr_attr_list(context));

        // Truncate the result to a boolean and return.
        builder.CreateRet(builder.CreateTrunc(cmp_ret, builder.getInt1Ty()));

        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

} // namespace

// NOTE: fcmp ULT means that it must return true if either operand is nan, or if a < b.
llvm::Value *llvm_real_fcmp_ult(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "heyoka_mpfr_fcmp_ult", 2u);

    return s.builder().CreateCall(f, {a, b});
}

// NOTE: fcmp UGE means that it must return true if either operand is nan, or if a >= b.
llvm::Value *llvm_real_fcmp_uge(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "heyoka_mpfr_fcmp_uge", 2u);

    return s.builder().CreateCall(f, {a, b});
}

// NOTE: fcmp OGE means that it will return true if neither operand is nan and a >= b.
// This corresponds to the semantics of the mpfr_greaterequal_p() function.
llvm::Value *llvm_real_fcmp_oge(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "mpfr_greaterequal_p", 2u);

    return s.builder().CreateCall(f, {a, b});
}

// NOTE: fcmp OLE means that it will return true if neither operand is nan and a <= b.
// This corresponds to the semantics of the mpfr_lessequal_p() function.
llvm::Value *llvm_real_fcmp_ole(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "mpfr_lessequal_p", 2u);

    return s.builder().CreateCall(f, {a, b});
}

// NOTE: fcmp OLT means that it will return true if neither operand is nan and a < b.
// This corresponds to the semantics of the mpfr_less_p() function.
llvm::Value *llvm_real_fcmp_olt(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "mpfr_less_p", 2u);

    return s.builder().CreateCall(f, {a, b});
}

// NOTE: fcmp OGT means that it will return true if neither operand is nan and a > b.
// This corresponds to the semantics of the mpfr_greater_p() function.
llvm::Value *llvm_real_fcmp_ogt(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "mpfr_greater_p", 2u);

    return s.builder().CreateCall(f, {a, b});
}

// NOTE: fcmp OEQ means that it will return true if neither operand is nan and a == b.
// This corresponds to the semantics of the mpfr_equal_p() function.
llvm::Value *llvm_real_fcmp_oeq(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto *f = real_nary_cmp(s, a->getType(), "mpfr_equal_p", 2u);

    return s.builder().CreateCall(f, {a, b});
}

llvm::Value *llvm_real_fcmp_one(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // Compute a == b.
    auto *ret = llvm_real_fcmp_oeq(s, a, b);

    // NOTE: this creates a logical NOT.
    return s.builder().CreateICmpEQ(ret, llvm::ConstantInt::getNullValue(ret->getType()));
}

llvm::Value *llvm_real_fnz(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Check if x is zero.
    auto *f = real_nary_cmp(s, x->getType(), "mpfr_zero_p", 1u);
    auto *ret = builder.CreateCall(f, x);

    // NOTE: this creates a logical NOT.
    return builder.CreateICmpEQ(ret, llvm::ConstantInt::getNullValue(ret->getType()));
}

// Convert the input unsigned integral value n to the real type fp_t.
llvm::Value *llvm_real_ui_to_fp(llvm_state &s, llvm::Value *n, llvm::Type *fp_t)
{
    assert(n != nullptr);
    assert(fp_t != nullptr);

    auto &md = s.module();
    auto &context = s.context();
    auto &builder = s.builder();

    const auto real_prec = llvm_is_real(fp_t);
    assert(real_prec > 0);

    // Fetch the integral type and its bit width.
    auto *llvm_int_t = llvm::cast<llvm::IntegerType>(n->getType());
    const auto source_int_width = llvm_int_t->getBitWidth();

    // We will be using mpfr_set_ui(), which takes an unsigned long
    // as input. If the source integer type is wider than unsigned long, we
    // need to error out.
    constexpr auto ul_width = static_cast<unsigned>(std::numeric_limits<unsigned long>::digits);
    if (source_int_width > ul_width) {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot convert an LLVM integer of type '{}' to a real: the bit width is too large",
                        llvm_type_name(llvm_int_t)));
        // LCOV_EXCL_STOP
    }

    const auto fname = fmt::format("heyoka.real.{}.ui_{}_to_fp", real_prec, source_int_width);

    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        auto *orig_bb = builder.GetInsertBlock();

        // Create the function type and the function.
        const std::vector<llvm::Type *> fargs{llvm_int_t};
        auto *ft = llvm::FunctionType::get(fp_t, fargs, false);
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);
        f->addFnAttr(llvm::Attribute::NoUnwind);
        f->addFnAttr(llvm::Attribute::Speculatable);
        f->addFnAttr(llvm::Attribute::WillReturn);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create an undef value for the result and add it as first function argument.
        auto [real_res, limb_arr_res] = llvm_undef_mpfr_view(s, fp_t);
        std::vector<llvm::Value *> mpfr_args;
        mpfr_args.push_back(real_res);

        // Add the input unsigned integer value, extended to unsigned long if necessary.
        if (source_int_width == ul_width) {
            mpfr_args.push_back(f->args().begin());
        } else {
            mpfr_args.push_back(builder.CreateZExt(f->args().begin(), to_llvm_type<unsigned long>(context)));
        }

        // Add the rounding mode.
        mpfr_args.push_back(llvm_mpfr_rndn(s));

        // Invoke the MPFR primitive.
        llvm_invoke_external(s, "mpfr_set_ui", to_llvm_type<int>(context), mpfr_args, get_mpfr_attr_list(context));

        // Assemble the result.
        auto *res = llvm_mpfr_view_to_real(s, real_res, limb_arr_res, fp_t);

        builder.CreateRet(res);

        builder.SetInsertPoint(orig_bb);
    }

    return builder.CreateCall(f, n);
}

// Sign function.
// NOTE: this is implemented on top of mpfr_sgn(), which behaves consistently with the branchless
// sign function (0 < x) - (x < 0) used for the basic floating-point types. In particular, mpfr_sgn()
// returns 0 if x is NaN.
llvm::Value *llvm_real_sgn(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto *fp_t = x->getType();

    const auto real_prec = llvm_is_real(fp_t);

    assert(real_prec > 0);

    // NOTE: this needs to behave like llvm_sgn() - in particular, it needs to return a 32-bit
    // integer value of either 1, 0 or -1. Hence we cannot rely directly on real_nary_cmp(),
    // which ends up returning a bool instead.

    auto &md = s.module();
    auto &context = s.context();
    auto &builder = s.builder();

    const auto fname = fmt::format("heyoka.real.{}.sgn", real_prec);

    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        auto *orig_bb = builder.GetInsertBlock();

        // Create the function type and the function.
        const std::vector<llvm::Type *> fargs{fp_t};
        auto *ft = llvm::FunctionType::get(builder.getInt32Ty(), fargs, false);
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);
        f->addFnAttr(llvm::Attribute::NoUnwind);
        f->addFnAttr(llvm::Attribute::Speculatable);
        f->addFnAttr(llvm::Attribute::WillReturn);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the mpfr view for the input argument and add it as function argument.
        const std::vector<llvm::Value *> mpfr_args{llvm_real_to_mpfr_view(s, f->args().begin()).first};

        // Invoke the MPFR primitive.
        auto *int_t = to_llvm_type<int>(context);
        auto *cmp_ret = llvm_invoke_external(s, "heyoka_mpfr_sgn", int_t, mpfr_args, get_mpfr_attr_list(context));

        // Compute the int32 return value: cmp_ret == 0 ? 0 : (cmp_ret < 0 ? -1 : 1).
        auto *int32_t = builder.getInt32Ty();
        auto *cmp_ret_zero = builder.CreateICmpEQ(cmp_ret, llvm::ConstantInt::get(int_t, 0u));
        auto *cmp_ret_neg = builder.CreateICmpSLT(cmp_ret, llvm::ConstantInt::get(int_t, 0u));
        auto *ret = builder.CreateSelect(cmp_ret_zero, builder.getInt32(0),
                                         builder.CreateSelect(cmp_ret_neg, llvm::ConstantInt::getSigned(int32_t, -1),
                                                              llvm::ConstantInt::getSigned(int32_t, 1)));

        builder.CreateRet(ret);

        builder.SetInsertPoint(orig_bb);
    }

    return builder.CreateCall(f, x);
}

// Utility to create a real with precision p
// whose value is the epsilon at that precision.
// NOTE: for consistency with the epsilons returned for the other
// types, we return here 2**-(prec - 1). See:
// https://en.wikipedia.org/wiki/Machine_epsilon
// NOTE: mp++ 0.28 has a dedicated function for this,
// we can just switch to that implementation
// when we bump up the mp++ required version.
mppp::real eps_from_prec(mpfr_prec_t p)
{
    assert(p >= mppp::real_prec_min() && p <= mppp::real_prec_max());

    return mppp::real{1ul, boost::numeric_cast<mpfr_exp_t>(-(p - 1)), p};
}

} // namespace detail

HEYOKA_END_NAMESPACE

#if !defined(NDEBUG)

extern "C" HEYOKA_DLL_PUBLIC void heyoka_assert_real_match_precs_mpfr_view_to_real(mpfr_prec_t p1,
                                                                                   mpfr_prec_t p2) noexcept
{
    assert(p1 == p2);
}

#endif

// Wrapper to implement ULT comparison semantics for real types.
extern "C" HEYOKA_DLL_PUBLIC int heyoka_mpfr_fcmp_ult(const mppp::mpfr_struct_t *a,
                                                      const mppp::mpfr_struct_t *b) noexcept
{
    assert(a != nullptr);
    assert(b != nullptr);
    assert(mpfr_get_prec(a) == mpfr_get_prec(b));

    if (mpfr_nan_p(a) != 0 || mpfr_nan_p(b) != 0) {
        return 1;
    } else {
        return ::mpfr_less_p(a, b);
    }
}

// Wrapper to implement UGE comparison semantics for real types.
extern "C" HEYOKA_DLL_PUBLIC int heyoka_mpfr_fcmp_uge(const mppp::mpfr_struct_t *a,
                                                      const mppp::mpfr_struct_t *b) noexcept
{
    assert(a != nullptr);
    assert(b != nullptr);
    assert(mpfr_get_prec(a) == mpfr_get_prec(b));

    if (mpfr_nan_p(a) != 0 || mpfr_nan_p(b) != 0) {
        return 1;
    } else {
        return ::mpfr_greaterequal_p(a, b);
    }
}

// Wrapper to invoke the mpfr_sgn() macro.
extern "C" HEYOKA_DLL_PUBLIC int heyoka_mpfr_sgn(const mppp::mpfr_struct_t *x) noexcept
{
    return mpfr_sgn(x);
}

#endif
