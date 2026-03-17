// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <llvm/Analysis/VectorUtils.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ModRef.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/vector_math.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

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

namespace
{

// Helper to determine if the scalar type tp can use the LLVM
// math intrinsics.
// NOTE: this is needed because for some types the use of LLVM intrinsics
// might be buggy or not possible at all (e.g., for mppp::real).
bool llvm_stype_can_use_math_intrinsics(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);
    assert(!tp->isVectorTy());

    auto &context = s.context();

    // NOTE: by default we assume it is safe to invoke the LLVM intrinsics
    // on the fundamental C++ floating-point types.
    // NOTE: to_external_llvm_type fails by returning nullptr - in such a case
    // (which I don't think is currently possible) then the
    // comparison to tp will fail as tp is not null.
    return tp == to_external_llvm_type<float>(context, false) || tp == to_external_llvm_type<double>(context, false)
           || tp == to_external_llvm_type<long double>(context, false);
}

// Helper to lookup/insert the declaration of an LLVM intrinsic into a module.
// NOTE: here "name" is expected to be the overloaded name of the intrinsic (that is, without type information).
// NOTE: types and nargs are needed independently of each other. For instance, llvm.pow is an
// intrinsic with 2 arguments but the types argument has only 1 element because both arguments
// must have the same type. I.e., the intrinsic is type-dependent on a single type only (not 2).
// NOTE: for intrinsics, it is not necessary to set up manually the function attributes, as LLVM takes care of it.
llvm::Function *llvm_lookup_intrinsic(ir_builder &builder, const std::string &name,
                                      const std::vector<llvm::Type *> &types, unsigned nargs)
{
    // NOTE: we used to have an assert(types.size() <= nargs), but as it turns out there are sometimes overloaded
    // intrinsics with no input arguments.
    assert(boost::starts_with(name, "llvm."));

    // Fetch the intrinsic ID from the name.
    const auto intrinsic_ID =
#if LLVM_VERSION_MAJOR < 20
        llvm::Function::lookupIntrinsicID(name);
#else
        llvm::Intrinsic::lookupIntrinsicID(name);
#endif
    // LCOV_EXCL_START
    if (intrinsic_ID == llvm::Intrinsic::not_intrinsic) {
        throw std::invalid_argument(fmt::format("Cannot fetch the ID of the intrinsic '{}'", name));
    }
    // LCOV_EXCL_STOP

    // Fetch the declaration.
    // NOTE: for generic intrinsics to work, we need to specify
    // the desired argument type(s). See:
    // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
    // And the docs of the getDeclaration() function.
    assert(builder.GetInsertBlock() != nullptr);
    assert(builder.GetInsertBlock()->getModule() != nullptr);
    auto *f =
#if LLVM_VERSION_MAJOR < 20
        llvm::Intrinsic::getDeclaration(builder.GetInsertBlock()->getModule(), intrinsic_ID, types);
#else
        llvm::Intrinsic::getOrInsertDeclaration(builder.GetInsertBlock()->getModule(), intrinsic_ID, types);
#endif
    assert(f != nullptr);
    // It does not make sense to have a definition of an intrinsic.
    assert(f->isDeclaration());

    // Check the number of arguments.
    // LCOV_EXCL_START
    if (f->arg_size() != nargs) {
        throw std::invalid_argument(fmt::format("Incorrect number of arguments for the intrinsic '{}': {} are "
                                                "expected, but {} are needed instead",
                                                name, nargs, f->arg_size()));
    }
    // LCOV_EXCL_STOP

    return f;
}

// Helper to invoke an intrinsic function with arguments 'args'. 'types' are the argument type(s) for overloaded
// intrinsics.
//
// NOTE: types and args are needed independently of each other. For instance, llvm.pow() is an intrinsic with 2
// arguments but the types argument has only 1 element because both arguments always have the same type. I.e., the
// intrinsic is type-dependent on a single type only (not 2).
llvm::CallInst *llvm_invoke_intrinsic(ir_builder &builder, const std::string &name,
                                      const std::vector<llvm::Type *> &types, const std::vector<llvm::Value *> &args)
{
    auto *callee_f = llvm_lookup_intrinsic(builder, name, types, boost::numeric_cast<unsigned>(args.size()));

    // Create the function call.
    auto *r = builder.CreateCall(callee_f, args);
    assert(r != nullptr);

    return r;
}

// Generate a set of function attributes to be used when invoking
// external math functions.
//
// The idea here is to copy the attributes from LLVM's math intrinsics,
// which unlock several optimisation opportunities thanks to the way the default LLVM
// floating-point environment is set up:
//
// https://llvm.org/docs/LangRef.html#floating-point-environment
llvm::AttributeList llvm_ext_math_func_attrs(llvm_state &s)
{
    // NOTE: use the fabs() f64 intrinsic - hopefully it does not matter
    // which intrinsic we pick.
    auto *f = llvm_lookup_intrinsic(s.builder(), "llvm.fabs", {to_external_llvm_type<double>(s.context())}, 1);
    assert(f != nullptr);

    return f->getAttributes();
}

// Add a pointer to the llvm.used global variable of a module:
//
// https://llvm.org/docs/LangRef.html#the-llvm-used-global-variable
//
// If the llvm.used variable does not exist yet, create it.
//
// NOTE: this has quadratic complexity when appending ptr to an existing array. It should not be a problem for the type
// of use we do as we expect just a few entries in this array, but something to keep in mind.
//
// NOTE: if ptr is already in the llvm.used array, then this will be a no-op.
void llvm_append_used(llvm_state &s, llvm::Constant *ptr)
{
    assert(ptr != nullptr);
    assert(ptr->getType()->isPointerTy());

    auto &md = s.module();
    auto &ctx = s.context();

    // Fetch the pointer type.
    auto *ptr_type = llvm::PointerType::getUnqual(ctx);

    if (auto *orig_used = md.getGlobalVariable("llvm.used")) {
        // The llvm.used variable exists already.

        // Fetch the original initializer.
        assert(orig_used->hasInitializer());
        auto *orig_init = llvm::cast<llvm::ConstantArray>(orig_used->getInitializer());

        // Construct a new initializer with the original values plus the new pointer.
        std::vector<llvm::Constant *> arr_values;
        arr_values.reserve(
            boost::safe_numerics::safe<decltype(arr_values.size())>(orig_init->getType()->getNumElements()) + 1);
        for (decltype(orig_init->getType()->getNumElements()) i = 0; i < orig_init->getType()->getNumElements(); ++i) {
            auto *orig_el = orig_init->getAggregateElement(boost::numeric_cast<unsigned>(i));
            assert(orig_el->getType()->isPointerTy());

            // NOTE: if ptr was already in the llvm.used vector, just bail out early.
            if (orig_el->isElementWiseEqual(ptr)) {
                return;
            }

            arr_values.push_back(orig_el);
        }
        arr_values.push_back(ptr);

        // Create the new array.
        auto *used_array_type = llvm::ArrayType::get(ptr_type, boost::numeric_cast<std::uint64_t>(arr_values.size()));
        auto *used_arr = llvm::ConstantArray::get(used_array_type, arr_values);

        // Remove the original one.
        orig_used->eraseFromParent();

        // Add the new global variable.
        auto *g_used_arr = new llvm::GlobalVariable(md, used_arr->getType(), true,
                                                    llvm::GlobalVariable::AppendingLinkage, used_arr, "llvm.used");
        g_used_arr->setSection("llvm.metadata");
    } else {
        // The llvm.used variable does not exist yet, create it.
        auto *used_array_type = llvm::ArrayType::get(ptr_type, 1);
        const std::vector<llvm::Constant *> arr_values{ptr};
        auto *used_arr = llvm::ConstantArray::get(used_array_type, arr_values);
        auto *g_used_arr = new llvm::GlobalVariable(md, used_arr->getType(), true,
                                                    llvm::GlobalVariable::AppendingLinkage, used_arr, "llvm.used");
        g_used_arr->setSection("llvm.metadata");
    }
}

// Attach the vfabi attributes to "call", which must be a call to a function with scalar arguments.
// The necessary vfabi information is stored in vfi. The function returns "call".
// The attributes of the scalar function will be attached to the vector variants.
// NOTE: this will insert the declarations of the vector variants into the module, if needed
// (plus all the boilerplate necessary for preventing the declarations from being optimised out).
llvm::CallInst *llvm_add_vfabi_attrs(llvm_state &s, llvm::CallInst *call, const std::vector<vf_info> &vfi)
{
    assert(call != nullptr);

    const auto *f = call->getCalledFunction();
    const auto *ft = f->getFunctionType();

    const auto num_args = ft->getNumParams();

    assert(num_args != 0u);
    assert(std::all_of(ft->param_begin(), ft->param_end(), [](auto *p) { return p != nullptr && !p->isVectorTy(); }));

    auto &context = s.context();
    auto &builder = s.builder();

    // Can we use the faster but less precise vectorised implementations?
    const auto use_fast_math = builder.getFastMathFlags().approxFunc();

    if (!vfi.empty()) {
        // There exist vector variants of the scalar function.
        auto &md = s.module();

        // Fetch the type of the scalar arguments.
        auto *scal_t = ft->getParamType(0);

        // Attach the "vector-function-abi-variant" attribute to the call so that LLVM's auto-vectorizer can take
        // advantage of these vector variants.
        std::vector<std::string> vf_abi_strs;
        vf_abi_strs.reserve(vfi.size());
        for (const auto &el : vfi) {
            // Fetch the vf_abi attr string (either the low-precision
            // or standard version).
            const auto &vf_abi_attr
                = (use_fast_math && !el.lp_vf_abi_attr.empty()) ? el.lp_vf_abi_attr : el.vf_abi_attr;
            vf_abi_strs.push_back(vf_abi_attr);
        }
        call->addFnAttr(llvm::Attribute::get(context, "vector-function-abi-variant",
                                             fmt::format("{}", fmt::join(vf_abi_strs, ","))));

        // Now we need to:
        // - add the declarations of the vector variants to the module,
        // - ensure that these declarations are not removed by the optimiser,
        //   otherwise the vector variants will not be picked up.

        // Remember the original insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // Add the vector variants and the boilerplate
        // to prevent them from being removed.
        for (const auto &el : vfi) {
            assert(el.width > 0u);
            assert(el.nargs == num_args);

            // Fetch the vector function name from el (either the low-precision
            // or standard version).
            const auto &el_name = (use_fast_math && !el.lp_name.empty()) ? el.lp_name : el.name;

            // The vector type for the current variant.
            auto *cur_vec_t = make_vector_type(scal_t, el.width);

            // The signature of the current variant.
            auto *vec_ft = llvm::FunctionType::get(
                cur_vec_t,
                std::vector<llvm::Type *>(boost::numeric_cast<std::vector<llvm::Type *>::size_type>(num_args),
                                          cur_vec_t),
                false);

            // Try to lookup the variant in the module.
            auto *vf_ptr = md.getFunction(el_name);

            if (vf_ptr == nullptr) {
                // The declaration of the variant is not there yet, create it.
                vf_ptr = llvm_func_create(vec_ft, llvm::Function::ExternalLinkage, el_name, &md);

                // NOTE: setting the attributes on the vector variant is not strictly required
                // for the auto-vectorizer to work. However, in other parts of the code, the vector
                // variants are invoked directly (via llvm_invoke_external()) and in those cases
                // proper attributes do help the optimiser. Thus, we want to make sure
                // that the attributes are set consistently regardless of where the declarations
                // of the vector variants are created. The convention we follow is that the attributes
                // of the vector variants must match the attributes of the scalar counterpart.
                vf_ptr->setAttributes(f->getAttributes());
            } else {
                // The declaration of the variant is already there.
                // Check that the signatures and attributes match.
                assert(vf_ptr->getFunctionType() == vec_ft);
                assert(vf_ptr->getAttributes() == f->getAttributes());
            }

            // Ensure that the variant is not optimised out because it is not explicitly used in the code.
            //
            // NOTE: llvm_append_used() will not insert vf_ptr if it is in the llvm.used array already.
            llvm_append_used(s, vf_ptr);
        }

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return call;
}

// Helper to invoke a scalar function with arguments which may or may not
// be vectors.
//
// In the former case, the call will be decomposed into a sequence of calls with scalar arguments,
// and the return values will be re-assembled as a vector. Vector arguments must all have the same size.
//
// In the latter case, this function will be equivalent to invoking the scalar function on the scalar arguments.
//
// In both cases, all arguments must be of the same type.
//
// make_s_call will be used to generate the scalar call on the scalar arguments.
//
// vfi contains the information about the vector variants of the scalar function. The information in vfi
// will be attached to the scalar call(s).
llvm::Value *
llvm_scalarise_vector_call(llvm_state &s, const std::vector<llvm::Value *> &args,
                           const std::function<llvm::CallInst *(const std::vector<llvm::Value *> &)> &make_s_call,
                           const std::vector<vf_info> &vfi)
{
    assert(!args.empty());
    // Make sure all arguments are of the same type.
    assert(std::all_of(args.begin() + 1, args.end(),
                       [&args](const auto &arg) { return arg->getType() == args[0]->getType(); }));
    // Make sure all arguments are either vectors or scalars.
    assert(std::ranges::all_of(args, [](const auto &arg) { return arg->getType()->isVectorTy(); })
           || std::ranges::all_of(args, [](const auto &arg) { return !arg->getType()->isVectorTy(); }));

    auto &builder = s.builder();

    // Decompose each argument into a vector of scalars.
    std::vector<std::vector<llvm::Value *>> scalars;
    scalars.reserve(args.size());
    for (const auto &arg : args) {
        scalars.push_back(vector_to_scalars(builder, arg));
    }

    // Fetch the vector size.
    const auto vec_size = scalars[0].size();
    assert(vec_size > 0u);

    // LCOV_EXCL_START
    // Make sure the vector size is the same for all arguments.
    assert(std::all_of(scalars.begin() + 1, scalars.end(),
                       [vec_size](const auto &arg) { return arg.size() == vec_size; }));
    // LCOV_EXCL_STOP

    // Invoke the function on each set of scalars.
    std::vector<llvm::Value *> retvals, scal_args;
    for (decltype(scalars[0].size()) i = 0; i < vec_size; ++i) {
        // Setup the vector of scalar arguments.
        scal_args.clear();
        for (const auto &scal_set : scalars) {
            scal_args.push_back(scal_set[i]);
        }

        // Invoke the scalar function, add the vector variants info, and store the scalar result.
        auto *s_call = make_s_call(scal_args);

#if !defined(NDEBUG)

        auto *called_f = s_call->getCalledFunction();
        // All arguments in the function call must be scalars.
        assert(std::all_of(called_f->arg_begin(), called_f->arg_end(),
                           [](const auto &arg) { return !arg.getType()->isVectorTy(); }));

#endif

        retvals.emplace_back(llvm_add_vfabi_attrs(s, s_call, vfi));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, retvals);
}

// Helper to invoke an external scalar math function with arguments
// which may be vectors.
// The call will be decomposed into a sequence of calls with scalar arguments,
// and the return values will be re-assembled as a vector.
// If the arguments are vectors, they must all have the same size.
// vfi contains the information about the vector variants of the scalar function.
// attrs is the list of attributes to attach to the scalar function.
// It is assumed that the return type of the math function is the same as the
// arguments' type.
llvm::Value *llvm_scalarise_ext_math_vector_call(llvm_state &s, const std::vector<llvm::Value *> &args,
                                                 const std::string &fname, const std::vector<vf_info> &vfi,
                                                 const llvm::AttributeList &attrs)
{
    // NOTE: this is not supposed to be used with intrinsics.
    assert(!boost::starts_with(fname, "llvm."));

    return llvm_scalarise_vector_call(
        s, args,
        [&](const std::vector<llvm::Value *> &scal_args) {
            assert(!scal_args.empty());
            assert(!args.empty());
            assert(scal_args[0]->getType() == args[0]->getType()->getScalarType());
            return llvm_invoke_external(s, fname, scal_args[0]->getType(), scal_args, attrs);
        },
        vfi);
}

// Get the suffix of a C math function corresponding to the scalar type scal_t. For instance,
// for sin() we have the following possibilities:
//
// - sinf(float) (suffix "f"),
// - sin(double) (suffix ""),
// - sinl(long double) (suffix "l"),
// - sinq(__float128) (suffix "q").
//
// If no C math function is available for the type scal_t, return an empty value.
std::optional<std::string> get_cmath_func_suffix(llvm_state &s, llvm::Type *scal_t)
{
    assert(scal_t != nullptr);
    assert(!scal_t->isVectorTy());

    auto &context = s.context();

    if (scal_t == to_external_llvm_type<float>(context, false)) {
        return "f";
    }

    if (scal_t == to_external_llvm_type<double>(context, false)) {
        return "";
    }

    if (scal_t == to_external_llvm_type<long double>(context, false)) {
        return "l";
    }

#if defined(HEYOKA_HAVE_REAL128)

    if (scal_t == to_external_llvm_type<mppp::real128>(context, false)) {
        return "q";
    }

#endif

    return {};
}

// Helper to invoke an external vector function with arguments args, automatically handling mismatches between the width
// of the vector function and the width of the arguments.
//
// vfi is a vector of vf_info instances listing the available implementations of the vector function (each one
// supporting a different vector width). attrs is the set of attributes to attach to the invocation(s) of the vector
// function.
//
// This function has several preconditions:
//
// - there must be at least 1 arg,
// - vfi cannot be empty,
// - all args must be vectors of the same type with a size greater than 1.
llvm::Value *llvm_invoke_vector_impl(llvm_state &s, const std::vector<vf_info> &vfi, const llvm::AttributeList &attrs,
                                     const std::vector<llvm::Value *> &args)
{
    assert(!args.empty());
    assert(!vfi.empty());

    // Check that all arguments are non-null.
    assert(std::ranges::all_of(args, [](const auto &arg) { return arg != nullptr; }));

    // Check that all arguments are of the same type.
    assert(std::ranges::all_of(args.begin() + 1, args.end(),
                               [&args](const auto &arg) { return arg->getType() == args[0]->getType(); }));

    const auto nargs = args.size();

    // Fetch the argument type.
    auto *x_t = args[0]->getType();

    // Ensure that the arguments are vectors.
    auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(x_t);
    assert(vec_t != nullptr);

    // Fetch the vector width.
    const auto vector_width = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
    assert(vector_width > 1u);

    // Fetch the builder.
    auto &bld = s.builder();

    // Can we use the faster but less precise vectorised implementations?
    const auto use_fast_math = bld.getFastMathFlags().approxFunc();

    // Lookup a vector implementation with width *greater than or equal to* vector_width.
    //
    // NOLINTNEXTLINE(modernize-use-ranges)
    auto vfi_it = std::lower_bound(vfi.begin(), vfi.end(), vector_width,
                                   [](const auto &vfi_item, std::uint32_t n) { return vfi_item.width < n; });

    if (vfi_it == vfi.end()) {
        // All vector implementations have a SIMD width *less than* vector_width. We will need
        // to decompose the vector arguments into smaller vectors, perform the calculations
        // on the smaller vectors, and reassemble the results into a single large vector.

        // Step back to the widest available vector implementation and fetch its width.
        --vfi_it;
        const auto available_vector_width = vfi_it->width;
        assert(available_vector_width > 0u);
        assert(vfi_it->nargs == nargs);

        // Fetch the vector type matching the chosen implementation.
        auto *available_vec_t = make_vector_type(vec_t->getScalarType(), available_vector_width);

        // Fetch the vector function name (either the low-precision or standard version).
        const auto &vf_name = (use_fast_math && !vfi_it->lp_name.empty()) ? vfi_it->lp_name : vfi_it->name;

        // Compute the number of chunks into which the original vector arguments will be split.
        const auto n_chunks = (vector_width / available_vector_width)
                              + static_cast<std::uint32_t>(vector_width % available_vector_width != 0u);

        // Prepare the vector of results of the invocations of the vector implementations.
        std::vector<llvm::Value *> vec_results;
        vec_results.reserve(n_chunks);

        // Prepare the vector of arguments for the invocations of the vector implementations.
        std::vector<llvm::Value *> vec_args;
        vec_args.reserve(nargs);

        // Prepare the mask vector.
        std::vector<int> mask;
        mask.reserve(vector_width);

        for (std::uint32_t i = 0; i < n_chunks; ++i) {
            // Construct the mask vector for the current iteration.
            mask.clear();
            const auto chunk_begin = i * available_vector_width;
            // NOTE: special case for the last iteration.
            const auto chunk_end = (i == n_chunks - 1u) ? vector_width : (chunk_begin + available_vector_width);
            for (auto idx = chunk_begin; idx != chunk_end; ++idx) {
                mask.push_back(boost::numeric_cast<int>(idx));
            }
            // Pad the mask if needed (this will happen only at the last iteration).
            // NOTE: the pad value is the last value in the original (large) vector.
            mask.insert(mask.end(), available_vector_width - mask.size(), boost::numeric_cast<int>(vector_width - 1u));

            // Build the vector of arguments.
            vec_args.clear();
            for (const auto &arg : args) {
                vec_args.push_back(bld.CreateShuffleVector(arg, mask));
            }

            // Invoke the vector implementation and add the result to vec_results.
            vec_results.push_back(llvm_invoke_external(s, vf_name, available_vec_t, vec_args, attrs));
        }

        // Reassemble vec_results into a large vector.
        auto *ret = llvm::concatenateVectors(bld, vec_results);

        // We need one last shuffle to trim the padded values at the end of ret (if any).
        mask.clear();
        for (std::uint32_t idx = 0; idx < vector_width; ++idx) {
            mask.push_back(boost::numeric_cast<int>(idx));
        }
        return bld.CreateShuffleVector(ret, mask);
    } else if (vfi_it->width == vector_width) {
        // We have a vector implementation with exactly the correct width. Use it.
        assert(vfi_it->nargs == nargs);

        // Fetch the vector function name (either the low-precision
        // or standard version).
        const auto &vf_name = (use_fast_math && !vfi_it->lp_name.empty()) ? vfi_it->lp_name : vfi_it->name;

        // Invoke it.
        return llvm_invoke_external(s, vf_name, vec_t, args, attrs);
    } else {
        // We have a vector implemention with SIMD width *greater than* vector_width. We need
        // to pad the input arguments, invoke the SIMD implementation, trim the result and return.

        // Fetch the width of the vector implementation.
        const auto available_vector_width = vfi_it->width;
        assert(available_vector_width > 0u);
        assert(vfi_it->nargs == nargs);

        // Fetch the vector type matching the chosen implementation.
        auto *available_vec_t = make_vector_type(vec_t->getScalarType(), available_vector_width);

        // Fetch the vector function name (either the low-precision or standard version).
        const auto &vf_name = (use_fast_math && !vfi_it->lp_name.empty()) ? vfi_it->lp_name : vfi_it->name;

        // Prepare the mask vector.
        std::vector<int> mask;
        mask.reserve(available_vector_width);
        for (std::uint32_t idx = 0; idx < vector_width; ++idx) {
            mask.push_back(boost::numeric_cast<int>(idx));
        }
        // Pad the mask with the last value in the original vector.
        mask.insert(mask.end(), available_vector_width - vector_width, boost::numeric_cast<int>(vector_width - 1u));

        // Prepare the vector of arguments for the invocation of the vector implementation.
        std::vector<llvm::Value *> vec_args;
        vec_args.reserve(nargs);
        for (const auto &arg : args) {
            vec_args.push_back(bld.CreateShuffleVector(arg, mask));
        }

        // Invoke the vector implementation.
        auto *ret = llvm_invoke_external(s, vf_name, available_vec_t, vec_args, attrs);

        // We need one last shuffle to trim the padded values at the end of ret.
        mask.resize(vector_width);
        return bld.CreateShuffleVector(ret, mask);
    }
}

} // namespace

// Implementation of an LLVM math function built on top of a function from the C math library, if possible.
//
// base_name is the name of the double-precision variant of the C math function. base_name will also be used to create
// the MPFR name for the real implementation.
llvm::Value *llvm_math_cmath(llvm_state &s, const std::string &base_name, const std::vector<llvm::Value *> &args)
{
    assert(!args.empty());
    assert(!base_name.empty());

    // Check that all arguments are non-null.
    assert(std::ranges::all_of(args, [](const auto &arg) { return arg != nullptr; }));

    // Check that all arguments are of the same type.
    assert(std::ranges::all_of(args.begin() + 1, args.end(),
                               [&args](const auto &arg) { return arg->getType() == args[0]->getType(); }));

    const auto nargs = args.size();

    auto &builder = s.builder();

    // Determine the type and scalar type of the arguments.
    auto *x_t = args[0]->getType();
    auto *scal_t = x_t->getScalarType();

    // Check if we have a cmath function available for the implementation.
    if (const auto suffix = get_cmath_func_suffix(s, scal_t)) {
        // Build the function name.
        const auto scal_name = base_name + *suffix;

        // Lookup the scalar name in the vector function info map.
        const auto &vfi = lookup_vf_info(scal_name);

        // Execute the generators, if available.
        for (const auto &vf : vfi) {
            if (vf.gen) {
                vf.gen(s);
            }
        }

        // Fetch the math function attributes.
        // NOTE: these will be used in all math function invocations
        // to ensure that scalar and vector versions are declared consistently
        // with the same attributes.
        const auto attrs = llvm_ext_math_func_attrs(s);

        if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(x_t)) {
            // The inputs are vectors. Fetch their SIMD width.
            const auto vector_width = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());

            if (vector_width == 1u || vfi.empty()) {
                // If the vector width is 1, or we do not have any vector implementation available,
                // we scalarise the function call.
                return llvm_scalarise_ext_math_vector_call(s, args, scal_name, vfi, attrs);
            } else {
                // The vector width is > 1 and we have one or more vector implementations available. Use them.
                return llvm_invoke_vector_impl(s, vfi, attrs, args);
            }
        } else {
            // The input is **not** a vector. Invoke the scalar function attaching vector
            // variants if available.
            auto *ret = llvm_invoke_external(s, scal_name, scal_t, args, attrs);
            return llvm_add_vfabi_attrs(s, ret, vfi);
        }
    }

#if defined(HEYOKA_HAVE_REAL)

    // NOTE: this handles only the scalar case.
    if (llvm_is_real(x_t) != 0) {
        auto *f = real_nary_op(s, x_t, "mpfr_" + base_name, boost::numeric_cast<unsigned>(nargs));
        return builder.CreateCall(f, args);
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Invalid type '{}' encountered in the LLVM implementation of the C math function '{}'",
                    llvm_type_name(x_t), base_name));
    // LCOV_EXCL_STOP
}

// Implementation of an LLVM math function built on top of an intrinsic (if possible).
//
// intr_name is the name of the intrinsic (without type information), f128/real_name are the names of the functions to
// be used for the real128/real implementations (if these cannot be implemented on top of the LLVM intrinsics).
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_math_intr(llvm_state &s, const std::string &intr_name,
#if defined(HEYOKA_HAVE_REAL128)
                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                            const std::string &f128_name,
#endif
#if defined(HEYOKA_HAVE_REAL)
                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                            const std::string &real_name,
#endif

                            const std::vector<llvm::Value *> &args)
{
    assert(!args.empty());
    assert(boost::starts_with(intr_name, "llvm."));

    // Check that all arguments are non-null.
    assert(std::ranges::all_of(args, [](const auto &arg) { return arg != nullptr; }));

    // Check that all arguments are of the same type.
    assert(std::ranges::all_of(args.begin() + 1, args.end(),
                               [&args](const auto &arg) { return arg->getType() == args[0]->getType(); }));

    const auto nargs = args.size();

    // Determine the type and scalar type of the arguments.
    auto *x_t = args[0]->getType();
    auto *scal_t = x_t->getScalarType();

    auto &builder = s.builder();

    if (llvm_stype_can_use_math_intrinsics(s, scal_t)) {
        // We can use the LLVM intrinsics for the given scalar type.

        // Lookup the intrinsic that would be used
        // in the scalar implementation.
        auto *s_intr = llvm_lookup_intrinsic(builder, intr_name, {scal_t}, boost::numeric_cast<unsigned>(nargs));

        // Lookup the scalar intrinsic name in the vector function info map.
        const auto &vfi = lookup_vf_info(std::string(s_intr->getName()));

        // Execute the generators, if available.
        for (const auto &vf : vfi) {
            if (vf.gen) {
                vf.gen(s);
            }
        }

        if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(x_t)) {
            // The inputs are vectors. Fetch their SIMD width.
            const auto vector_width = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());

            if (vector_width == 1u || vfi.empty()) {
                // If the vector width is 1, or we do not have any vector implementation available,
                // we let LLVM handle it.
                return llvm_invoke_intrinsic(builder, intr_name, {x_t}, args);
            } else {
                // The vector width is > 1 and we have one or more vector implementations available. Use them.
                return llvm_invoke_vector_impl(s, vfi, s_intr->getAttributes(), args);
            }
        } else {
            // The input is **not** a vector. Invoke the scalar intrinsic attaching vector
            // variants if available.
            auto *ret = builder.CreateCall(s_intr, args);
            return llvm_add_vfabi_attrs(s, ret, vfi);
        }
    }

#if defined(HEYOKA_HAVE_REAL128)

    // NOTE: this handles both the scalar and vector cases.
    if (scal_t == to_external_llvm_type<mppp::real128>(s.context(), false)) {
        // Lookup the scalar function's name in the vector function info map.
        const auto &vfi = lookup_vf_info(f128_name);

        // Execute the generators, if available.
        for (const auto &vf : vfi) {
            if (vf.gen) {
                vf.gen(s);
            }
        }

        return llvm_scalarise_ext_math_vector_call(s, args, f128_name, vfi,
                                                   // NOTE: use the standard math function attributes.
                                                   llvm_ext_math_func_attrs(s));
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    // NOTE: this handles only the scalar case.
    if (llvm_is_real(x_t) != 0) {
        auto *f = real_nary_op(s, x_t, real_name, boost::numeric_cast<unsigned>(nargs));
        return builder.CreateCall(f, args);
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Invalid type '{}' encountered in the implementation of the intrinsic-based math function '{}'",
                    llvm_type_name(x_t), intr_name));
    // LCOV_EXCL_STOP
}

// Helper to load into a vector of size vector_size the sequential scalar data starting at ptr.
// If vector_size is 1, a scalar is loaded instead.
llvm::Value *load_vector_from_memory(ir_builder &builder, llvm::Type *tp, llvm::Value *ptr, std::uint32_t vector_size)
{
    // LCOV_EXCL_START
    assert(vector_size > 0u);
    assert(llvm::isa<llvm::PointerType>(ptr->getType()));
    assert(!llvm::isa<llvm::FixedVectorType>(ptr->getType()));
    // LCOV_EXCL_STOP

    if (vector_size == 1u) {
        // Scalar case.
        return builder.CreateLoad(tp, ptr);
    }

    // Create the vector type.
    auto *vector_t = make_vector_type(tp, vector_size);
    assert(vector_t != nullptr); // LCOV_EXCL_LINE

    // Create the mask (all 1s).
    auto *mask = llvm::ConstantInt::get(make_vector_type(builder.getInt1Ty(), vector_size), 1u);

    // Create the passthrough value. This can stay undefined as it is never used
    // due to the mask being all 1s.
    auto *passthru = llvm::UndefValue::get(vector_t);

    // Invoke the intrinsic.
    auto *ret = llvm_invoke_intrinsic(builder, "llvm.masked.expandload", {vector_t}, {ptr, mask, passthru});

    return ret;
}

// This is like load_vector_from_memory(), except that the pointee of ptr might differ from the type of the loaded
// value (e.g., in the case of real). This is supposed to be used when loading data created outside the LLVM
// JIT world.
llvm::Value *ext_load_vector_from_memory(llvm_state &s, llvm::Type *tp, llvm::Value *ptr, std::uint32_t vector_size)
{
    auto &builder = s.builder();

#if defined(HEYOKA_HAVE_REAL)
    if (const auto real_prec = llvm_is_real(tp->getScalarType())) {
        // LCOV_EXCL_START
        if (tp->isVectorTy()) {
            throw std::invalid_argument("Cannot load a vector of reals");
        }
        // LCOV_EXCL_STOP

        auto &context = s.context();

        // Fetch the limb type.
        auto *limb_t = to_external_llvm_type<mp_limb_t>(context);

        // Fetch the external real struct type.
        auto *real_t = to_external_llvm_type<mppp::real>(context);

        // Compute the number of limbs in the internal real type.
        const auto nlimbs = mppp::prec_to_nlimbs(real_prec);

#if !defined(NDEBUG)

        // In debug mode, we want to assert that the precision of the internal
        // type matches exactly the precision of the external variable.

        // Load the precision from the external value.
        auto *prec_t = to_external_llvm_type<mpfr_prec_t>(context);
        auto *prec_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(0)});
        auto *prec = builder.CreateLoad(prec_t, prec_ptr);

        llvm_assert(s, builder.CreateICmpEQ(
                           prec, llvm::ConstantInt::getSigned(prec_t, boost::numeric_cast<std::int64_t>(real_prec))));
#endif

        // Init the return value.
        llvm::Value *ret = llvm::UndefValue::get(tp);

        // Read and insert the sign.
        auto *sign_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(1)});
        auto *sign = builder.CreateLoad(to_external_llvm_type<mpfr_sign_t>(context), sign_ptr);
        ret = builder.CreateInsertValue(ret, sign, {0u});

        // Read and insert the exponent.
        auto *exp_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(2)});
        auto *exp = builder.CreateLoad(to_external_llvm_type<mpfr_exp_t>(context), exp_ptr);
        ret = builder.CreateInsertValue(ret, exp, {1u});

        // Load in a local variable the input pointer to the limbs.
        auto *limb_ptr_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(3)});
        auto *limb_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(context), limb_ptr_ptr);

        // Load and insert the limbs.
        for (std::size_t i = 0; i < nlimbs; ++i) {
            auto *cur_limb_ptr
                = builder.CreateInBoundsGEP(limb_t, limb_ptr, builder.getInt32(boost::numeric_cast<std::uint32_t>(i)));
            auto *limb = builder.CreateLoad(limb_t, cur_limb_ptr);
            ret = builder.CreateInsertValue(ret, limb, {2u, boost::numeric_cast<std::uint32_t>(i)});
        }

        return ret;
    } else {
#endif
        return load_vector_from_memory(builder, tp, ptr, vector_size);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Helper to store the content of vector vec to the pointer ptr. If vec is not a vector,
// a plain store will be performed.
void store_vector_to_memory(ir_builder &builder, llvm::Value *ptr, llvm::Value *vec)
{
    // LCOV_EXCL_START
    assert(llvm::isa<llvm::PointerType>(ptr->getType()));
    assert(!llvm::isa<llvm::FixedVectorType>(ptr->getType()));
    // LCOV_EXCL_STOP

    if (auto *vector_t = llvm::dyn_cast<llvm::FixedVectorType>(vec->getType())) {
        // Determine the vector size.
        const auto vector_size = boost::numeric_cast<std::uint32_t>(vector_t->getNumElements());

        // Create the mask (all 1s).
        auto *mask = llvm::ConstantInt::get(make_vector_type(builder.getInt1Ty(), vector_size), 1u);

        // Invoke the intrinsic.
        llvm_invoke_intrinsic(builder, "llvm.masked.compressstore", {vector_t}, {vec, ptr, mask});
    } else {
        // Not a vector, store vec directly.
        builder.CreateStore(vec, ptr);
    }
}

// This is like store_vector_to_memory(), except that the pointee of ptr might differ from the type of the value to
// be stored (e.g., in the case of real). This is supposed to be used when storing data created created inside LLVM into
// a pointer that will then be used by code outside the LLVM realm.
void ext_store_vector_to_memory(llvm_state &s, llvm::Value *ptr, llvm::Value *vec)
{
    auto &builder = s.builder();

#if defined(HEYOKA_HAVE_REAL)
    if (const auto real_prec = llvm_is_real(vec->getType()->getScalarType())) {
        // LCOV_EXCL_START
        if (vec->getType()->isVectorTy()) {
            throw std::invalid_argument("Cannot store a vector of reals");
        }
        // LCOV_EXCL_STOP

        auto &context = s.context();

        // Fetch the limb type.
        auto *limb_t = to_external_llvm_type<mp_limb_t>(context);

        // Fetch the external real struct type.
        auto *real_t = to_external_llvm_type<mppp::real>(context);

        // Compute the number of limbs in the internal real type.
        const auto nlimbs = mppp::prec_to_nlimbs(real_prec);

#if !defined(NDEBUG)

        // In debug mode, we want to assert that the precision of the internal
        // type matches exactly the precision of the external variable.

        // Load the precision from the external value.
        auto *prec_t = to_external_llvm_type<mpfr_prec_t>(context);
        auto *out_prec_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(0)});
        auto *prec = builder.CreateLoad(prec_t, out_prec_ptr);

        llvm_assert(s, builder.CreateICmpEQ(
                           prec, llvm::ConstantInt::getSigned(prec_t, boost::numeric_cast<std::int64_t>(real_prec))));

#endif

        // Store the sign.
        auto *out_sign_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(1)});
        builder.CreateStore(builder.CreateExtractValue(vec, {0u}), out_sign_ptr);

        // Store the exponent.
        auto *out_exp_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(2)});
        builder.CreateStore(builder.CreateExtractValue(vec, {1u}), out_exp_ptr);

        // Load in a local variable the output pointer to the limbs.
        auto *out_limb_ptr_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(3)});
        auto *out_limb_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(context), out_limb_ptr_ptr);

        // Store the limbs.
        for (std::size_t i = 0; i < nlimbs; ++i) {
            auto *cur_limb_ptr = builder.CreateInBoundsGEP(limb_t, out_limb_ptr,
                                                           builder.getInt32(boost::numeric_cast<std::uint32_t>(i)));
            builder.CreateStore(builder.CreateExtractValue(vec, {2u, boost::numeric_cast<std::uint32_t>(i)}),
                                cur_limb_ptr);
        }
    } else {
#endif
        store_vector_to_memory(builder, ptr, vec);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Gather a vector of type vec_tp from ptrs.
//
// ptrs is assumed to be either a single pointer or a vector of pointers into an array of scalar values
// of type vec_tp->getScalarType(). The array is assumed to be properly aligned for the scalar values.
//
// If vec_tp is a vector type, then ptrs must be a vector of the same size. The returned value
// will be a vector of values gathered from the addresses specified in ptrs.
//
// Otherwise, ptrs must be a single pointer and the returned value is a scalar (that is, the function
// behaves like a scalar load from ptrs).
llvm::Value *gather_vector_from_memory(ir_builder &builder, llvm::Type *vec_tp, llvm::Value *ptrs)
{
    assert(ptrs->getType()->getScalarType()->isPointerTy());

    if (llvm::isa<llvm::FixedVectorType>(vec_tp)) {
        // LCOV_EXCL_START
        assert(llvm::isa<llvm::FixedVectorType>(ptrs->getType()));
        assert(llvm::cast<llvm::FixedVectorType>(vec_tp)->getNumElements()
               == llvm::cast<llvm::FixedVectorType>(ptrs->getType())->getNumElements());
        // LCOV_EXCL_STOP

        // Fetch the alignment of the scalar type.
        const auto align = get_alignment(*builder.GetInsertBlock()->getModule(), vec_tp->getScalarType());

        return builder.CreateMaskedGather(vec_tp, ptrs, llvm::Align(align));
    } else {
        // LCOV_EXCL_START
        assert(!llvm::isa<llvm::FixedVectorType>(ptrs->getType()));
        // LCOV_EXCL_STOP

        return builder.CreateLoad(vec_tp, ptrs);
    }
}

// Same as above, but for external loads.
llvm::Value *ext_gather_vector_from_memory(llvm_state &s, llvm::Type *tp, llvm::Value *ptr)
{
    auto &builder = s.builder();

#if defined(HEYOKA_HAVE_REAL)
    if (llvm_is_real(tp->getScalarType()) != 0) {
        // LCOV_EXCL_START
        if (tp->isVectorTy()) {
            throw std::invalid_argument("Cannot gather from memory a vector of reals");
        }
        // LCOV_EXCL_STOP

        assert(!llvm::isa<llvm::FixedVectorType>(ptr->getType()));

        return ext_load_vector_from_memory(s, tp, ptr, 1);
    } else {
#endif
        return gather_vector_from_memory(builder, tp, ptr);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Create a SIMD vector of size vector_size filled with the value c. If vector_size is 1,
// c will be returned.
llvm::Value *vector_splat(ir_builder &builder, llvm::Value *c, std::uint32_t vector_size)
{
    // LCOV_EXCL_START
    assert(vector_size > 0u);
    assert(!llvm::isa<llvm::FixedVectorType>(c->getType()));
    // LCOV_EXCL_STOP

    if (vector_size == 1u) {
        return c;
    }

    return builder.CreateVectorSplat(boost::numeric_cast<unsigned>(vector_size), c);
}

llvm::Type *make_vector_type(llvm::Type *t, std::uint32_t vector_size)
{
    // LCOV_EXCL_START
    assert(t != nullptr);
    assert(vector_size > 0u);
    assert(!llvm::isa<llvm::FixedVectorType>(t));
    // LCOV_EXCL_STOP

    if (vector_size == 1u) {
        return t;
    } else {
        auto *retval = llvm::FixedVectorType::get(t, boost::numeric_cast<unsigned>(vector_size));

        assert(retval != nullptr); // LCOV_EXCL_LINE

        return retval;
    }
}

// Convert the input LLVM vector to a std::vector of values. If vec is not a vector,
// return {vec}.
std::vector<llvm::Value *> vector_to_scalars(ir_builder &builder, llvm::Value *vec)
{
    if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(vec->getType())) {
        // Fetch the vector width.
        auto vector_size = vec_t->getNumElements();

        assert(vector_size != 0u); // LCOV_EXCL_LINE

        // Extract the vector elements one by one.
        std::vector<llvm::Value *> ret;
        for (decltype(vector_size) i = 0; i < vector_size; ++i) {
            ret.push_back(builder.CreateExtractElement(vec, boost::numeric_cast<std::uint64_t>(i)));
            assert(ret.back() != nullptr); // LCOV_EXCL_LINE
        }

        return ret;
    } else {
        return {vec};
    }
}

// Convert a std::vector of values into an LLVM vector of the corresponding size.
// If scalars contains only 1 value, return that value.
llvm::Value *scalars_to_vector(ir_builder &builder, const std::vector<llvm::Value *> &scalars)
{
    assert(!scalars.empty());

    // Fetch the vector size.
    const auto vector_size = scalars.size();

    if (vector_size == 1u) {
        return scalars[0];
    }

    // Fetch the scalar type.
    auto *scalar_t = scalars[0]->getType();

    // Create the corresponding vector type.
    auto *vector_t = make_vector_type(scalar_t, boost::numeric_cast<std::uint32_t>(vector_size));
    assert(vector_t != nullptr);

    // Create an empty vector.
    llvm::Value *vec = llvm::UndefValue::get(vector_t);
    assert(vec != nullptr);

    // Fill it up.
    for (auto i = 0u; i < vector_size; ++i) {
        assert(scalars[i]->getType() == scalar_t);

        vec = builder.CreateInsertElement(vec, scalars[i], i);
    }

    return vec;
}

// Helper to invoke an external function called 'name' with arguments args and return type ret_type.
llvm::CallInst *llvm_invoke_external(llvm_state &s, const std::string &name, llvm::Type *ret_type,
                                     const std::vector<llvm::Value *> &args, const llvm::AttributeList &attrs)
{
    // Look up the name in the global module table.
    auto *callee_f = s.module().getFunction(name);

    if (callee_f == nullptr) {
        // The function does not exist yet, make the prototype.
        std::vector<llvm::Type *> arg_types;
        arg_types.reserve(args.size());
        for (auto *a : args) {
            arg_types.push_back(a->getType());
        }
        auto *ft = llvm::FunctionType::get(ret_type, arg_types, false);
        callee_f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &s.module());

        // Add the function attributes.
        callee_f->setAttributes(attrs);
    } else {
        // LCOV_EXCL_START

        // The function declaration exists already. Check that it is only a
        // declaration and not a definition.
        if (!callee_f->isDeclaration()) {
            throw std::invalid_argument(fmt::format("Cannot call the function '{}' as an external function, because "
                                                    "it is defined as an internal module function",
                                                    name));
        }
        // Check the number of arguments.
        if (callee_f->arg_size() != args.size()) {
            throw std::invalid_argument(
                fmt::format("Incorrect # of arguments passed while calling the external function '{}': {} "
                            "are expected, but {} were provided instead",
                            name, callee_f->arg_size(), args.size()));
        }
        // NOTE: in the future we should consider adding more checks here
        // (e.g., argument types, return type, attributes, etc.).

        // LCOV_EXCL_STOP
    }

    // Create the function call.
    auto *r = s.builder().CreateCall(callee_f, args);
    assert(r != nullptr);
    // NOTE: we used to have r->setTailCall(true) here, but:
    // - when optimising, the tail call attribute is automatically
    //   added,
    // - it is not 100% clear to me whether it is always safe to enable it:
    // https://llvm.org/docs/CodeGenerator.html#tail-calls

    return r;
}

llvm::CallInst *llvm_invoke_external(llvm_state &s, const std::string &name, llvm::Type *ret_type,
                                     const std::vector<llvm::Value *> &args)
{
    return llvm_invoke_external(s, name, ret_type, args, {});
}

// Create a floating-point constant of type fp_t containing
// the value val.
llvm::Constant *llvm_constantfp([[maybe_unused]] llvm_state &s, llvm::Type *fp_t, double val)
{
    if (fp_t->getScalarType()->isFloatingPointTy()) {
        // NOTE: if fp_t is a vector type, the constant value
        // will be splatted in the return value.
        return llvm::ConstantFP::get(fp_t, val);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm::cast<llvm::Constant>(llvm_codegen(s, fp_t, number{val}));
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Unable to generate a floating-point constant of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

// Convert the input unsigned integral value(s) n to the floating-point type fp_t.
// If n is a scalar/vector, then fp_t must also be a scalar/vector type.
llvm::Value *llvm_ui_to_fp(llvm_state &s, llvm::Value *n, llvm::Type *fp_t)
{
    assert(n != nullptr);
    assert(fp_t != nullptr);

    assert(n->getType()->getScalarType()->isIntegerTy());

#if !defined(NDEBUG)
    if (n->getType()->isVectorTy()) {
        assert(fp_t->isVectorTy());
        assert(llvm::cast<llvm::FixedVectorType>(n->getType())->getNumElements()
               == llvm::cast<llvm::FixedVectorType>(fp_t)->getNumElements());
    } else {
        assert(!fp_t->isVectorTy());
    }
#endif

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return s.builder().CreateUIToFP(n, fp_t);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_ui_to_fp(s, n, fp_t);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot convert an unsigned integral value to the LLVM type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

namespace
{

// Helper to generate a global string constant as a null-terminated char array.
//
// A pointer to the beginning of the array will be returned.
//
// NOTE: this is similar to the old CreateGlobalStringPtr() LLVM function which has recently been deprecated.
llvm::Value *llvm_create_global_string_ptr(llvm_state &s, const char *str)
{
    auto &bld = s.builder();

    // Create the global variable.
    auto *gv = bld.CreateGlobalString(str);

    // Fetch and return a pointer to the first element of the array.
    return bld.CreateInBoundsGEP(gv->getValueType(), gv, {bld.getInt32(0), bld.getInt32(0)});
}

} // namespace

HEYOKA_DLL_PUBLIC void llvm_assert([[maybe_unused]] llvm_state &s, [[maybe_unused]] llvm::Value *val,
                                   [[maybe_unused]] std::source_location loc)
{

#if !defined(NDEBUG)

    // NOTE: run the assertion check only if we are not optimising the JIT compilation. The idea here is that the
    // assertion check will result in invoking an external C function, which will likely impede a lot of optimisations
    // and reduce the diversity of tested IR code.
    if (s.get_opt_level() > 0u) {
        return;
    }

    auto &bld = s.builder();

    assert(val != nullptr);
    assert(val->getType()->getScalarType() == bld.getInt1Ty());

    // Transfer the file/function name strings into the LLVM world.
    //
    // NOTE: it may be possible that the pointers returned by loc refer to strings with static storage duration, in
    // which case we could just copy the pointers. However I cannot find any conclusive reference at this time that
    // guarantees this.
    auto *file_name = llvm_create_global_string_ptr(s, loc.file_name());
    auto *function_name = llvm_create_global_string_ptr(s, loc.function_name());

    assert(file_name->getType()->isPointerTy());
    assert(function_name->getType()->isPointerTy());

    // Build the assertion condition.
    auto *cond = val;
    if (llvm::isa<llvm::FixedVectorType>(val->getType())) {
        // val is a vector: the assertion condition is true if all SIMD lanes are true, false otherwise.
        cond = bld.CreateAndReduce(cond);
    }

    // Check it.
    llvm_if_then_else(
        s, cond, []() {},
        [&s, &bld, file_name, function_name, &loc]() {
            llvm_invoke_external(s, "heyoka_llvm_assertion_failure", bld.getVoidTy(),
                                 {bld.getInt64(boost::numeric_cast<std::uint64_t>(loc.line())),
                                  bld.getInt64(boost::numeric_cast<std::uint64_t>(loc.column())), file_name,
                                  function_name});
        });

#endif
}

// Helper to add the nocapture attribute to a pointer function argument. The syntax changes in LLVM 21, hence the need
// for a wrapper.
void llvm_add_no_capture_argattr([[maybe_unused]] llvm_state &s, llvm::Argument *arg)
{
    assert(arg != nullptr);
    assert(arg->getType()->isPointerTy());

#if LLVM_VERSION_MAJOR <= 20

    arg->addAttr(llvm::Attribute::NoCapture);

#else

    arg->addAttr(llvm::Attribute::getWithCaptureInfo(s.context(), llvm::CaptureInfo::none()));

#endif
}

} // namespace detail

HEYOKA_END_NAMESPACE

#if !defined(NDEBUG)

// LCOV_EXCL_START

extern "C" HEYOKA_DLL_PUBLIC [[noreturn]] void heyoka_llvm_assertion_failure(const std::uint64_t line,
                                                                             const std::uint64_t column,
                                                                             const char *file_name,
                                                                             const char *function_name) noexcept
{
    heyoka::detail::get_logger()->critical("LLVM assertion failure in file '{}', function '{}', line={}, column={}",
                                           file_name, function_name, line, column);

    assert(false);
}

// LCOV_EXCL_STOP

#endif

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
