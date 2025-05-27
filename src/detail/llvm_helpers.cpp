// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <llvm/Analysis/VectorUtils.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

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
#include <heyoka/detail/type_traits.hpp>
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

// Helper to associate a C++ integral type
// to an LLVM integral type.
template <typename T>
llvm::Type *int_to_llvm(llvm::LLVMContext &c)
{
    static_assert(std::is_integral_v<T>);

    // NOTE: need to add +1 to the digits for signed integers,
    // as ::digits does not account for the sign bit.
    return llvm::Type::getIntNTy(c, static_cast<unsigned>(std::numeric_limits<T>::digits)
                                        + static_cast<unsigned>(std::is_signed_v<T>));
};

// The global type map to associate a C++ type to an LLVM type.
// NOLINTNEXTLINE(cert-err58-cpp)
const auto type_map = []() {
    std::unordered_map<std::type_index, llvm::Type *(*)(llvm::LLVMContext &)> retval;

    // Try to associate C++ float to LLVM float.
    if (is_ieee754_binary32<float>) {
        retval[typeid(float)] = [](llvm::LLVMContext &c) { return llvm::Type::getFloatTy(c); };
    }

    // Try to associate C++ double to LLVM double.
    if (is_ieee754_binary64<double>) {
        retval[typeid(double)] = [](llvm::LLVMContext &c) { return llvm::Type::getDoubleTy(c); };
    }

    // Try to associate C++ long double to an LLVM fp type.
    if (is_ieee754_binary64<long double>) {
        retval[typeid(long double)] = [](llvm::LLVMContext &c) {
            // IEEE double-precision format (this is the case on MSVC for instance).
            return llvm::Type::getDoubleTy(c);
        };
    } else if (is_x86_fp80<long double>) {
        retval[typeid(long double)] = [](llvm::LLVMContext &c) {
            // x86 extended precision format.
            return llvm::Type::getX86_FP80Ty(c);
        };
    } else if (is_ieee754_binary128<long double>) {
        retval[typeid(long double)] = [](llvm::LLVMContext &c) {
            // IEEE quadruple-precision format (e.g., ARM 64).
            return llvm::Type::getFP128Ty(c);
        };
    }

#if defined(HEYOKA_HAVE_REAL128)

    // Associate mppp::real128 to fp128.
    static_assert(is_ieee754_binary128<mppp::real128>);
    retval[typeid(mppp::real128)] = [](llvm::LLVMContext &c) { return llvm::Type::getFP128Ty(c); };

#endif

#if defined(HEYOKA_HAVE_REAL)

    retval[typeid(mppp::real)] = [](llvm::LLVMContext &c) -> llvm::Type * {
        if (auto *ptr = llvm::StructType::getTypeByName(c, "heyoka.real")) {
            return ptr;
        }

        auto *ret = llvm::StructType::create(
            {to_external_llvm_type<mpfr_prec_t>(c), to_external_llvm_type<mpfr_sign_t>(c),
             to_external_llvm_type<mpfr_exp_t>(c), llvm::PointerType::getUnqual(to_external_llvm_type<mp_limb_t>(c))},
            "heyoka.real");

        assert(ret != nullptr);
        assert(llvm::StructType::getTypeByName(c, "heyoka.real") == ret);

        return ret;
    };

#endif

    // Associate a few unsigned/signed integral types.
    retval[typeid(unsigned)] = int_to_llvm<unsigned>;
    retval[typeid(unsigned long)] = int_to_llvm<unsigned long>;
    retval[typeid(unsigned long long)] = int_to_llvm<unsigned long long>;
    retval[typeid(int)] = int_to_llvm<int>;
    retval[typeid(long)] = int_to_llvm<long>;
    retval[typeid(long long)] = int_to_llvm<long long>;

    return retval;
}();

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
// NOTE: this has quadratic complexity when appending ptr to an existing
// array. It should not be a problem for the type of use we do as we expect
// just a few entries in this array, but something to keep in mind.
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

        // Construct a new initializer with the original values
        // plus the new pointer.
        std::vector<llvm::Constant *> arr_values;
        arr_values.reserve(
            boost::safe_numerics::safe<decltype(arr_values.size())>(orig_init->getType()->getNumElements()) + 1);
        for (decltype(orig_init->getType()->getNumElements()) i = 0; i < orig_init->getType()->getNumElements(); ++i) {
            auto *orig_el = orig_init->getAggregateElement(boost::numeric_cast<unsigned>(i));
            assert(orig_el->getType()->isPointerTy());

            // NOTE: if ptr was already in the llvm.used vector, just bail
            // out early.
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

            // Ensure that the variant is not optimised out because it is not
            // explicitly used in the code.
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

// Helper to invoke an external vector function with arguments args, automatically handling
// mismatches between the width of the vector function and the width of the arguments.
//
// vfi is a vector of vf_info instances listing the available implementations of the vector function
// (each one supporting a different vector width). attrs is the set of attributes to attach to the
// invocation(s) of the vector function.
//
// This function has several preconditions:
//
// - there must be at least 1 arg,
// - vfi cannot be empty,
// - all args must be vectors of the same type with a size greater than 1.
llvm::Value *llvm_invoke_vector_impl(llvm_state &s, const auto &vfi, const auto &attrs, auto *...args)
{
    constexpr auto nargs = sizeof...(args);
    static_assert(nargs > 0u);
    static_assert((std::same_as<llvm::Value *, decltype(args)> && ...));

    assert(((args != nullptr) && ...));
    assert(!vfi.empty());

    // Build an array with the original arguments.
    const std::array orig_args{args...};

    // Check that all arguments have the same type.
    auto *x_t = orig_args[0]->getType();
    assert(((args->getType() == x_t) && ...));

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
        const auto n_chunks = vector_width / available_vector_width
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
            for (std::size_t arg_idx = 0; arg_idx < nargs; ++arg_idx) {
                vec_args.push_back(bld.CreateShuffleVector(orig_args[arg_idx], mask));
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
        return llvm_invoke_external(s, vf_name, vec_t, {args...}, attrs);
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
        for (std::size_t arg_idx = 0; arg_idx < nargs; ++arg_idx) {
            vec_args.push_back(bld.CreateShuffleVector(orig_args[arg_idx], mask));
        }

        // Invoke the vector implementation.
        auto *ret = llvm_invoke_external(s, vf_name, available_vec_t, vec_args, attrs);

        // We need one last shuffle to trim the padded values at the end of ret.
        mask.resize(vector_width);
        return bld.CreateShuffleVector(ret, mask);
    }
}

// Implementation of an LLVM math function built on top of an intrinsic (if possible).
// intr_name is the name of the intrinsic (without type information),
// f128/real_name are the names of the functions to be used for the
// real128/real implementations (if these cannot be implemented
// on top of the LLVM intrinsics).
template <typename... Args>
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

                            Args *...args)
{
    constexpr auto nargs = sizeof...(Args);
    static_assert(nargs > 0u);
    static_assert((std::is_same_v<llvm::Value, Args> && ...));

    assert(boost::starts_with(intr_name, "llvm."));

    assert(((args != nullptr) && ...));

    // Check that all arguments have the same type.
    const std::array arg_types = {args->getType()...};
    assert(((args->getType() == arg_types[0]) && ...));

    // Determine the type and scalar type of the arguments.
    auto *x_t = arg_types[0];
    auto *scal_t = x_t->getScalarType();

    auto &builder = s.builder();

    if (llvm_stype_can_use_math_intrinsics(s, scal_t)) {
        // We can use the LLVM intrinsics for the given scalar type.

        // Lookup the intrinsic that would be used
        // in the scalar implementation.
        auto *s_intr = llvm_lookup_intrinsic(builder, intr_name, {scal_t}, boost::numeric_cast<unsigned>(nargs));

        // Lookup the scalar intrinsic name in the vector function info map.
        const auto &vfi = lookup_vf_info(std::string(s_intr->getName()));

        if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(x_t)) {
            // The inputs are vectors. Fetch their SIMD width.
            const auto vector_width = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());

            if (vector_width == 1u || vfi.empty()) {
                // If the vector width is 1, or we do not have any vector implementation available,
                // we let LLVM handle it.
                return llvm_invoke_intrinsic(builder, intr_name, {x_t}, {args...});
            } else {
                // The vector width is > 1 and we have one or more vector implementations available. Use them.
                return llvm_invoke_vector_impl(s, vfi, s_intr->getAttributes(), args...);
            }
        } else {
            // The input is **not** a vector. Invoke the scalar intrinsic attaching vector
            // variants if available.
            auto *ret = builder.CreateCall(s_intr, {args...});
            return llvm_add_vfabi_attrs(s, ret, vfi);
        }
    }

#if defined(HEYOKA_HAVE_REAL128)

    // NOTE: this handles both the scalar and vector cases.
    if (scal_t == to_external_llvm_type<mppp::real128>(s.context(), false)) {
        return llvm_scalarise_ext_math_vector_call(s, {args...}, f128_name, lookup_vf_info(f128_name),
                                                   // NOTE: use the standard math function attributes.
                                                   llvm_ext_math_func_attrs(s));
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    // NOTE: this handles only the scalar case.
    if (llvm_is_real(x_t) != 0) {
        auto *f = real_nary_op(s, x_t, real_name, boost::numeric_cast<unsigned>(nargs));
        return builder.CreateCall(f, {args...});
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Invalid type '{}' encountered in the implementation of the intrinsic-based math function '{}'",
                    llvm_type_name(x_t), intr_name));
    // LCOV_EXCL_STOP
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

// Implementation of an LLVM math function built on top of a
// function from the C math library, if possible. base_name is the name
// of the double-precision variant of the C math function. base_name
// will also be used to create the MPFR name for the real implementation.
template <typename... Args>
llvm::Value *llvm_math_cmath(llvm_state &s, const std::string &base_name, Args *...args)
{
    constexpr auto nargs = sizeof...(Args);
    static_assert(nargs > 0u);
    static_assert((std::is_same_v<llvm::Value, Args> && ...));

    assert(!base_name.empty());

    assert(((args != nullptr) && ...));

    // Check that all arguments have the same type.
    const std::array arg_types = {args->getType()...};
    assert(((args->getType() == arg_types[0]) && ...));

    auto &builder = s.builder();

    // Determine the type and scalar type of the arguments.
    auto *x_t = arg_types[0];
    auto *scal_t = x_t->getScalarType();

    // Check if we have a cmath function available for the implementation.
    if (const auto suffix = get_cmath_func_suffix(s, scal_t)) {
        // Build the function name.
        const auto scal_name = base_name + *suffix;

        // Lookup the scalar name in the vector function info map.
        const auto &vfi = lookup_vf_info(scal_name);

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
                return llvm_scalarise_ext_math_vector_call(s, {args...}, scal_name, vfi, attrs);
            } else {
                // The vector width is > 1 and we have one or more vector implementations available. Use them.
                return llvm_invoke_vector_impl(s, vfi, attrs, args...);
            }
        } else {
            // The input is **not** a vector. Invoke the scalar function attaching vector
            // variants if available.
            auto *ret = llvm_invoke_external(s, scal_name, scal_t, {args...}, attrs);
            return llvm_add_vfabi_attrs(s, ret, vfi);
        }
    }

#if defined(HEYOKA_HAVE_REAL)

    // NOTE: this handles only the scalar case.
    if (llvm_is_real(x_t) != 0) {
        auto *f = real_nary_op(s, x_t, "mpfr_" + base_name, boost::numeric_cast<unsigned>(nargs));
        return builder.CreateCall(f, {args...});
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Invalid type '{}' encountered in the LLVM implementation of the C math function '{}'",
                    llvm_type_name(x_t), base_name));
    // LCOV_EXCL_STOP
}

} // namespace

// Implementation of the function to associate a C++ type to
// an LLVM type.
llvm::Type *to_external_llvm_type_impl(llvm::LLVMContext &c, const std::type_info &tp, bool err_throw)
{
    const auto it = type_map.find(tp);

    constexpr auto *err_msg = "Unable to associate the C++ type '{}' to an LLVM type";

    if (it == type_map.end()) {
        // LCOV_EXCL_START
        return err_throw ? throw std::invalid_argument(fmt::format(err_msg, boost::core::demangle(tp.name())))
                         : nullptr;
        // LCOV_EXCL_STOP
    } else {
        auto *ret = it->second(c);

        if (ret == nullptr) {
            // LCOV_EXCL_START
            return err_throw ? throw std::invalid_argument(fmt::format(err_msg, boost::core::demangle(tp.name())))
                             : nullptr;
            // LCOV_EXCL_STOP
        }

        return ret;
    }
}

// Helper to produce a unique string for the type t.
std::string llvm_mangle_type(llvm::Type *t)
{
    assert(t != nullptr);

    if (auto *v_t = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
        // If the type is a vector, get the name of the element type
        // and append the vector size.
        return fmt::format("{}_{}", llvm_type_name(v_t->getElementType()), v_t->getNumElements());
    } else if (auto *arr_t = llvm::dyn_cast<llvm::ArrayType>(t)) {
        // Similar idea if the type is an array.
        return fmt::format("array_{}_{}", llvm_type_name(arr_t->getElementType()), arr_t->getNumElements());
    } else {
        // Otherwise just return the type name.
        return llvm_type_name(t);
    }
}

// Helper to determine the vector size of x. If x is not
// of type llvm::FixedVectorType, 1 will be returned.
std::uint32_t get_vector_size(llvm::Value *x)
{
    if (const auto *vector_t = llvm::dyn_cast<llvm::FixedVectorType>(x->getType())) {
        return boost::numeric_cast<std::uint32_t>(vector_t->getNumElements());
    } else {
        return 1;
    }
}

// Small helper to compute the size of a global array.
std::uint32_t gl_arr_size(llvm::Value *v)
{
    return boost::numeric_cast<std::uint32_t>(
        llvm::cast<llvm::ArrayType>(llvm::cast<llvm::GlobalVariable>(v)->getValueType())->getNumElements());
}

// Fetch the alignment of a type.
std::uint64_t get_alignment(llvm::Module &md, llvm::Type *tp)
{
    return md.getDataLayout().getABITypeAlign(tp).value();
}

// Fetch the alloc size of a type. This should be
// equivalent to the sizeof() operator in C++.
// Requires a non-scalable type.
std::uint64_t get_size(llvm::Module &md, llvm::Type *tp)
{
    assert(!md.getDataLayout().getTypeAllocSize(tp).isScalable());

    return boost::numeric_cast<std::uint64_t>(md.getDataLayout().getTypeAllocSize(tp).getFixedValue());
}

// Convert the input integral value n to the type std::size_t.
// If an upcast is needed, it will be performed via zero extension.
llvm::Value *to_size_t(llvm_state &s, llvm::Value *n)
{
    // Get the bit width of the type of n.
    const auto n_bw = llvm::cast<llvm::IntegerType>(n->getType()->getScalarType())->getBitWidth();

    // Fetch the LLVM type corresponding to size_t, and its bit width.
    auto *lst = to_external_llvm_type<std::size_t>(s.context());
    const auto lst_bw = llvm::cast<llvm::IntegerType>(lst)->getBitWidth();
    assert(lst_bw == static_cast<unsigned>(std::numeric_limits<std::size_t>::digits)); // LCOV_EXCL_LINE

    if (lst_bw == n_bw) {
        // n is of type size_t, return it unchanged.
        assert(n->getType()->getScalarType() == lst); // LCOV_EXCL_LINE
        return n;
    } else {
        // Get the vector size of the type of n.
        const auto n_vs = get_vector_size(n);

        // Fetch the target type for the cast.
        auto *tgt_t = make_vector_type(lst, n_vs);

        if (n_bw > lst_bw) {
            // The type of n is bigger than size_t, truncate.
            return s.builder().CreateTrunc(n, tgt_t); // LCOV_EXCL_LINE
        } else {
            // The type of n is smaller than size_t, extend.
            return s.builder().CreateZExt(n, tgt_t);
        }
    }
}

// Helper to create a global zero-inited array variable in the module m
// with type t. The array is mutable and with internal linkage.
// NOTE: this works with real as well, as long as every element of the global
// array is written to before being read. I.e., in the case of real, this will result
// in an array of zero-inited structs, which are not valid real values, but as long
// as we never read an array element before writing to it, we will be ok.
llvm::GlobalVariable *make_global_zero_array(llvm::Module &m, llvm::ArrayType *t)
{
    assert(t != nullptr); // LCOV_EXCL_LINE

    // Make the global array.
    auto *gl_arr = new llvm::GlobalVariable(m, t, false, llvm::GlobalVariable::PrivateLinkage,
                                            llvm::ConstantAggregateZero::get(t));

    // Return it.
    return gl_arr;
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

        llvm_invoke_external(
            s, "heyoka_assert_real_match_precs_ext_load", builder.getVoidTy(),
            {prec, llvm::ConstantInt::getSigned(prec_t, boost::numeric_cast<std::int64_t>(real_prec))});

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
        auto *limb_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(limb_t), limb_ptr_ptr);

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

        llvm_invoke_external(
            s, "heyoka_assert_real_match_precs_ext_store", builder.getVoidTy(),
            {prec, llvm::ConstantInt::getSigned(prec_t, boost::numeric_cast<std::int64_t>(real_prec))});

#endif

        // Store the sign.
        auto *out_sign_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(1)});
        builder.CreateStore(builder.CreateExtractValue(vec, {0u}), out_sign_ptr);

        // Store the exponent.
        auto *out_exp_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(2)});
        builder.CreateStore(builder.CreateExtractValue(vec, {1u}), out_exp_ptr);

        // Load in a local variable the output pointer to the limbs.
        auto *out_limb_ptr_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(3)});
        auto *out_limb_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(limb_t), out_limb_ptr_ptr);

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

// Pairwise reduction of a vector of LLVM values.
llvm::Value *pairwise_reduce(std::vector<llvm::Value *> &vals,
                             const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &f)
{
    assert(!vals.empty());
    assert(f);

    // LCOV_EXCL_START
    if (vals.size() == std::numeric_limits<decltype(vals.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_reduce()");
    }
    // LCOV_EXCL_STOP

    while (vals.size() != 1u) {
        std::vector<llvm::Value *> new_vals;

        for (decltype(vals.size()) i = 0; i < vals.size(); i += 2u) {
            if (i + 1u == vals.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_vals.push_back(vals[i]);
            } else {
                new_vals.push_back(f(vals[i], vals[i + 1u]));
            }
        }

        new_vals.swap(vals);
    }

    return vals[0];
}

// Pairwise summation of a vector of LLVM values.
// https://en.wikipedia.org/wiki/Pairwise_summation
llvm::Value *pairwise_sum(llvm_state &s, std::vector<llvm::Value *> &sum)
{
    return pairwise_reduce(sum, [&s](llvm::Value *a, llvm::Value *b) -> llvm::Value * { return llvm_fadd(s, a, b); });
}

// Pairwise product of a vector of LLVM values.
llvm::Value *pairwise_prod(llvm_state &s, std::vector<llvm::Value *> &prod)
{
    return pairwise_reduce(prod, [&s](llvm::Value *a, llvm::Value *b) -> llvm::Value * { return llvm_fmul(s, a, b); });
}

// Helper to invoke an intrinsic function with arguments 'args'. 'types' are the argument type(s) for
// overloaded intrinsics.
// NOTE: types and args are needed independently of each other. For instance, llvm.pow() is an
// intrinsic with 2 arguments but the types argument has only 1 element because both arguments
// always have the same type. I.e., the intrinsic is type-dependent on a single type only (not 2).
llvm::CallInst *llvm_invoke_intrinsic(ir_builder &builder, const std::string &name,
                                      const std::vector<llvm::Type *> &types, const std::vector<llvm::Value *> &args)
{
    auto *callee_f = llvm_lookup_intrinsic(builder, name, types, boost::numeric_cast<unsigned>(args.size()));

    // Create the function call.
    auto *r = builder.CreateCall(callee_f, args);
    assert(r != nullptr);

    return r;
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

// Append bb to the list of blocks of the function f
void llvm_append_block(llvm::Function *f, llvm::BasicBlock *bb)
{
    f->insert(f->end(), bb);
}

// Create an LLVM for loop in the form:
//
// for (auto i = begin; i < end; i = next_cur(i)) {
//   body(i);
// }
//
// The default implementation of i = next_cur(i),
// if next_cur is not provided, is ++i.
//
// begin/end must be 32-bit unsigned integer values.
void llvm_loop_u32(llvm_state &s, llvm::Value *begin, llvm::Value *end, const std::function<void(llvm::Value *)> &body,
                   const std::function<llvm::Value *(llvm::Value *)> &next_cur)
{
    assert(body);
    assert(begin->getType() == end->getType());
    assert(begin->getType() == s.builder().getInt32Ty());

    auto &context = s.context();
    auto &builder = s.builder();

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Pre-create loop and afterloop blocks. Note that these have just
    // been created, they have not been inserted yet in the IR.
    auto *loop_bb = llvm::BasicBlock::Create(context);
    auto *after_bb = llvm::BasicBlock::Create(context);

    // NOTE: we need a special case if the body of the loop is
    // never to be executed (that is, begin >= end).
    // In such a case, we will jump directly to after_bb.
    // NOTE: unsigned integral comparison.
    auto *skip_cond = builder.CreateICmp(llvm::CmpInst::ICMP_UGE, begin, end);
    builder.CreateCondBr(skip_cond, after_bb, loop_bb);

    // Get a reference to the current block for
    // later usage in the phi node.
    auto *preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    llvm_append_block(f, loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto *cur = builder.CreatePHI(builder.getInt32Ty(), 2);
    cur->addIncoming(begin, preheader_bb);

    // Execute the loop body and the post-body code.
    llvm::Value *next{};
    try {
        body(cur);

        // Compute the next value of the iteration. Use the next_cur
        // function if provided, otherwise, by default, increase cur by 1.
        // NOTE: addition works regardless of integral signedness.
        next = next_cur ? next_cur(cur) : builder.CreateAdd(cur, builder.getInt32(1));
    } catch (...) {
        // NOTE: at this point after_bb has not been
        // inserted into any parent, and thus it will not
        // be cleaned up automatically. Do it manually.
        after_bb->deleteValue();

        throw;
    }

    // Compute the end condition.
    // NOTE: we use the unsigned less-than predicate.
    auto *end_cond = builder.CreateICmp(llvm::CmpInst::ICMP_ULT, next, end);

    // Get a reference to the current block for later use,
    // and insert the "after loop" block.
    auto *loop_end_bb = builder.GetInsertBlock();
    llvm_append_block(f, after_bb);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(end_cond, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    cur->addIncoming(next, loop_end_bb);
}

// Small helper to fetch a string representation
// of an LLVM type.
std::string llvm_type_name(llvm::Type *t)
{
    assert(t != nullptr);

    std::string retval;
    llvm::raw_string_ostream ostr(retval);

    t->print(ostr, false, true);

    return std::move(ostr.str());
}

// Create an LLVM if statement in the form:
//
// if (cond) {
//   then_f();
// } else {
//   else_f();
// }
void llvm_if_then_else(llvm_state &s, llvm::Value *cond,
                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                       const std::function<void()> &then_f, const std::function<void()> &else_f)
{
    auto &context = s.context();
    auto &builder = s.builder();

    assert(cond->getType() == builder.getInt1Ty());

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Create and insert the "then" block.
    auto *then_bb = llvm::BasicBlock::Create(context, "", f);

    // Create but do not insert the "else" and merge blocks.
    auto *else_bb = llvm::BasicBlock::Create(context);
    auto *merge_bb = llvm::BasicBlock::Create(context);

    // Create the conditional jump.
    builder.CreateCondBr(cond, then_bb, else_bb);

    // Emit the code for the "then" branch.
    builder.SetInsertPoint(then_bb);
    try {
        then_f();
    } catch (...) {
        // NOTE: else_bb and merge_bb have not been
        // inserted into any parent yet, clean them
        // up manually.
        else_bb->deleteValue();
        merge_bb->deleteValue();

        throw;
    }

    // Jump to the merge block.
    builder.CreateBr(merge_bb);

    // Emit the "else" block.
    llvm_append_block(f, else_bb);
    builder.SetInsertPoint(else_bb);
    try {
        else_f();
    } catch (...) {
        // NOTE: merge_bb has not been
        // inserted into any parent yet, clean it
        // up manually.
        merge_bb->deleteValue();

        throw;
    }

    // Jump to the merge block.
    builder.CreateBr(merge_bb);

    // Emit the merge block.
    llvm_append_block(f, merge_bb);
    builder.SetInsertPoint(merge_bb);
}

// Create a switch statement of the type:
//
// switch (val) {
//      default:
//          default_f();
//          break;
//      case_i:
//          case_i_f();
//          break;
// }
//
// where the pairs (case_i, case_i_f) are the elements in the 'cases' argument.
// val must be a 32-bit int.
void llvm_switch_u32(llvm_state &s, llvm::Value *val, const std::function<void()> &default_f,
                     const std::map<std::uint32_t, std::function<void()>> &cases)
{
    auto &context = s.context();
    auto &builder = s.builder();

    assert(val->getType() == builder.getInt32Ty());

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Prepare the blocks for the cases.
    std::deque<llvm::BasicBlock *> cases_blocks;
    for ([[maybe_unused]] const auto &cp : cases) {
        cases_blocks.push_back(llvm::BasicBlock::Create(context));
    }

    // Helper to clean up the uninserted blocks in case of exceptions.
    auto bb_cleanup = [&cases_blocks]() {
        for (auto *bb : cases_blocks) {
            bb->deleteValue();
        }
    };

    // Create and insert the default block.
    auto *default_bb = llvm::BasicBlock::Create(context, "", f);

    // Create but do not insert the merge block.
    auto *merge_bb = llvm::BasicBlock::Create(context);

    // Create the switch instruction.
    auto *sw_inst = builder.CreateSwitch(val, default_bb);

    // Emit the code for the default case.
    builder.SetInsertPoint(default_bb);
    try {
        default_f();
    } catch (...) {
        bb_cleanup();

        // NOTE: merge_bb has not been
        // inserted into any parent yet.
        merge_bb->deleteValue();

        throw;
    }

    // Jump to the merge block.
    builder.CreateBr(merge_bb);

    // Emit the cases blocks.
    for (const auto &[idx, case_f] : cases) {
        // Grab the block for the current case.
        auto *cur_bb = cases_blocks.front();

        // Insert it.
        llvm_append_block(f, cur_bb);
        builder.SetInsertPoint(cur_bb);

        // Pop it from cases_blocks, as now cur_bb is managed
        // by the builder and does not need cleanup in case
        // of exceptions any more.
        cases_blocks.pop_front();

        // Emit the code for the current case.
        try {
            case_f();
        } catch (...) {
            bb_cleanup();
            merge_bb->deleteValue();

            throw;
        }

        // Jump to the merge block.
        builder.CreateBr(merge_bb);

        // Add the case to the switch instruction.
        sw_inst->addCase(builder.getInt32(idx), cur_bb);
    }

    // Emit the merge block.
    llvm_append_block(f, merge_bb);
    builder.SetInsertPoint(merge_bb);
}

// Create an LLVM for loop in the form:
//
// while (cond()) {
//   body();
// }
void llvm_while_loop(llvm_state &s, const std::function<llvm::Value *()> &cond, const std::function<void()> &body)
{
    assert(body);
    assert(cond);

    auto &context = s.context();
    auto &builder = s.builder();

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto *f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Do a first evaluation of cond.
    // NOTE: if this throws, we have not created any block
    // yet, no need for manual cleanup.
    auto *cmp = cond();
    assert(cmp != nullptr);
    assert(cmp->getType() == builder.getInt1Ty());

    // Pre-create loop and afterloop blocks. Note that these have just
    // been created, they have not been inserted yet in the IR.
    auto *loop_bb = llvm::BasicBlock::Create(context);
    auto *after_bb = llvm::BasicBlock::Create(context);

    // NOTE: we need a special case if the body of the loop is
    // never to be executed (that is, cond returns false).
    // In such a case, we will jump directly to after_bb.
    builder.CreateCondBr(builder.CreateNot(cmp), after_bb, loop_bb);

    // Get a reference to the current block for
    // later usage in the phi node.
    auto *preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    llvm_append_block(f, loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto *cur = builder.CreatePHI(builder.getInt1Ty(), 2);
    cur->addIncoming(cmp, preheader_bb);

    // Execute the loop body and the post-body code.
    try {
        body();

        // Compute the end condition.
        cmp = cond();
        assert(cmp != nullptr);
        assert(cmp->getType() == builder.getInt1Ty());
    } catch (...) {
        // NOTE: at this point after_bb has not been
        // inserted into any parent, and thus it will not
        // be cleaned up automatically. Do it manually.
        after_bb->deleteValue();

        throw;
    }

    // Get a reference to the current block for later use,
    // and insert the "after loop" block.
    auto *loop_end_bb = builder.GetInsertBlock();
    llvm_append_block(f, after_bb);

    // Insert the conditional branch into the end of loop_end_bb.
    builder.CreateCondBr(cmp, loop_bb, after_bb);

    // Any new code will be inserted in after_bb.
    builder.SetInsertPoint(after_bb);

    // Add a new entry to the PHI node for the backedge.
    cur->addIncoming(cmp, loop_end_bb);
}

llvm::Value *llvm_fadd(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFAdd(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_add", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fadd values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fsub(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFSub(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_sub", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fsub values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fmul(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFMul(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_mul", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fmul values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fdiv(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFDiv(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_div", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fdiv values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fneg(llvm_state &s, llvm::Value *a)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFNeg(a);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fneg(s, a);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fneg values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
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

llvm::Value *llvm_fcmp_ult(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpULT(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ult(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ult values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_uge(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpUGE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_uge(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_uge values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ule(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpULE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ule(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ule values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_oge(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOGE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_oge(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_oge values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ole(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOLE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ole(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ole values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_olt(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOLT(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_olt(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_olt values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ogt(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOGT(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ogt(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ogt values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_oeq(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOEQ(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_oeq(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_oeq values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_one(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpONE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_one(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_one values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

// Check if the input floating-point value(s) x is anything other
// than zero (including NaN).
llvm::Value *llvm_fnz(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = x->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpUNE(x, llvm::ConstantFP::get(x->getType(), 0.));
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fnz(s, x);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Unable to invoke llvm_fnz() on values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

// Helper to compute sin and cos simultaneously.
// NOTE: although there exists a SLEEF function for computing sin/cos
// at the same time, we cannot use it directly because it returns a pair
// of SIMD vectors rather than a single one and that does not play
// well with the calling conventions. In theory we could write a wrapper
// for these sincos functions using pointers for output values,
// but compiling such a wrapper requires correctly
// setting up the SIMD compilation flags. Perhaps we can consider this in the
// future to improve performance.
// NOTE: for the vfabi machinery, I think we would need to create internal scalar
// and vector functions that implement the sincos() primitive. Then we would call
// the scalar primitive attaching the vfabi info about the vector variants. For this
// to work it looks like we would need a list of SIMD widths supported on the
// CPU, possibly implemented in target_features.
// NOTE: another possible improvement is an optimisation pass that automatically detects
// sin/cos usages that can be compressed in a single sincos call. If this were to work,
// we could just implement this a sin + cos and let the optimisation pass do
// the heavy lifting.
std::pair<llvm::Value *, llvm::Value *> llvm_sincos(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    [[maybe_unused]] auto *x_t = x->getType();
    [[maybe_unused]] auto *scal_t = x_t->getScalarType();

    // NOTE: real128 has a specialised primitive for this.
#if defined(HEYOKA_HAVE_REAL128)

    auto &context = s.context();

    if (scal_t == to_external_llvm_type<mppp::real128>(context, false)) {
        auto &builder = s.builder();

        // Convert the vector argument to scalars.
        auto x_scalars = vector_to_scalars(builder, x);

        // Execute the sincosq() function on the scalar values and store
        // the results in res_scalars.
        // NOTE: need temp storage because sincosq uses pointers
        // for output values.
        auto *s_all = builder.CreateAlloca(scal_t);
        auto *c_all = builder.CreateAlloca(scal_t);
        std::vector<llvm::Value *> res_sin, res_cos;
        for (const auto &x_scal : x_scalars) {
            llvm_invoke_external(s, "sincosq", builder.getVoidTy(), {x_scal, s_all, c_all},
                                 llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                                          {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn}));

            res_sin.emplace_back(builder.CreateLoad(scal_t, s_all));
            res_cos.emplace_back(builder.CreateLoad(scal_t, c_all));
        }

        // Reconstruct the return value as a vector.
        return {scalars_to_vector(builder, res_sin), scalars_to_vector(builder, res_cos)};
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(x_t) != 0) {
        return llvm_real_sincos(s, x);
    }

#endif

    return {llvm_sin(s, x), llvm_cos(s, x)};
}

// Helper to compute abs(x_v).
llvm::Value *llvm_abs(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.fabs",
#if defined(HEYOKA_HAVE_REAL128)
                          "fabsq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_abs",
#endif
                          x);
}

// Minimum value, floating-point arguments. Implemented as std::min():
// return (b < a) ? b : a;
llvm::Value *llvm_min(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    return builder.CreateSelect(llvm_fcmp_olt(s, b, a), b, a);
}

// Maximum value, floating-point arguments. Implemented as std::max():
// return (a < b) ? b : a;
llvm::Value *llvm_max(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    return builder.CreateSelect(llvm_fcmp_olt(s, a, b), b, a);
}

// Same as llvm_min(), but returns NaN if any operand is NaN:
// return (b == b) ? ((b < a) ? b : a) : b;
llvm::Value *llvm_min_nan(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    auto *b_not_nan = llvm_fcmp_oeq(s, b, b);
    auto *b_lt_a = llvm_fcmp_olt(s, b, a);

    return builder.CreateSelect(b_not_nan, builder.CreateSelect(b_lt_a, b, a), b);
}

// Same as llvm_max(), but returns NaN if any operand is NaN:
// return (b == b) ? ((a < b) ? b : a) : b;
llvm::Value *llvm_max_nan(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    auto *b_not_nan = llvm_fcmp_oeq(s, b, b);
    auto *a_lt_b = llvm_fcmp_olt(s, a, b);

    return builder.CreateSelect(b_not_nan, builder.CreateSelect(a_lt_b, b, a), b);
}

// Branchless sign function.
// NOTE: requires FP value.
// NOTE: this will return 0 if val is NaN.
llvm::Value *llvm_sgn(llvm_state &s, llvm::Value *val)
{
    assert(val != nullptr);

    auto &builder = s.builder();

    auto *x_t = val->getType()->getScalarType();

    if (x_t->isFloatingPointTy()) {
        // Build the zero constant.
        auto *zero = llvm_constantfp(s, val->getType(), 0.);

        // Run the comparisons.
        auto *cmp0 = llvm_fcmp_olt(s, zero, val);
        auto *cmp1 = llvm_fcmp_olt(s, val, zero);

        // Convert to int32.
        llvm::Type *int_type = make_vector_type(builder.getInt32Ty(), get_vector_size(val));
        auto *icmp0 = builder.CreateZExt(cmp0, int_type);
        auto *icmp1 = builder.CreateZExt(cmp1, int_type);

        // Compute and return the result.
        return builder.CreateSub(icmp0, icmp1);
    }

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(val->getType()) != 0) {
        return llvm_real_sgn(s, val);
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sgn()",
                                            llvm_type_name(val->getType())));
    // LCOV_EXCL_STOP
}

// Two-argument arctan.
llvm::Value *llvm_atan2(llvm_state &s, llvm::Value *y, llvm::Value *x)
{
    return llvm_math_cmath(s, "atan2", y, x);
}

// Exponential.
llvm::Value *llvm_exp(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.exp",
#if defined(HEYOKA_HAVE_REAL128)
                          "expq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_exp",
#endif
                          x);
}

// Fused multiply-add.
llvm::Value *llvm_fma(llvm_state &s, llvm::Value *x, llvm::Value *y, llvm::Value *z)
{
    return llvm_math_intr(s, "llvm.fma",
#if defined(HEYOKA_HAVE_REAL128)
                          "fmaq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_fma",
#endif
                          x, y, z);
}

// Floor.
llvm::Value *llvm_floor(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.floor",
#if defined(HEYOKA_HAVE_REAL128)
                          "floorq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_floor",
#endif
                          x);
}

// Trunc.
llvm::Value *llvm_trunc(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.trunc",
#if defined(HEYOKA_HAVE_REAL128)
                          "truncq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_trunc",
#endif
                          x);
}

// is_finite().
llvm::Value *llvm_is_finite(llvm_state &s, llvm::Value *x)
{
    assert(x != nullptr);

    auto *x_t = x->getType();

    if (x_t->getScalarType()->isFloatingPointTy()) {
        // Codegen +- inf.
        auto *pinf = llvm_codegen(s, x_t, number{std::numeric_limits<double>::infinity()});
        auto *minf = llvm_codegen(s, x_t, number{-std::numeric_limits<double>::infinity()});

        // Check that if x is not +- inf or NaN.
        auto *x_not_pinf = llvm_fcmp_one(s, x, pinf);
        auto *x_not_minf = llvm_fcmp_one(s, x, minf);

        // Put the conditions together and return.
        return s.builder().CreateLogicalAnd(x_not_pinf, x_not_minf);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x_t) != 0) {
        return llvm_real_isfinite(s, x);
#endif
        // LCOV_EXCL_START
    } else [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid type '{}' encountered in the LLVM implementation of is_finite()", llvm_type_name(x_t)));
    }
    // LCOV_EXCL_STOP
}

// Check if the input floating-point value is a natural number.
llvm::Value *llvm_is_natural(llvm_state &s, llvm::Value *x)
{
    // Is x finite?
    auto *x_finite = llvm_is_finite(s, x);

    // Is x>=0?
    auto *x_ge_0 = llvm_fcmp_oge(s, x, llvm_codegen(s, x->getType(), number{0.}));

    // Is x an integral value?
    auto *x_int = llvm_fcmp_oeq(s, x, llvm_trunc(s, x));

    // Put the conditions together and return.
    auto &bld = s.builder();
    auto *ret = bld.CreateLogicalAnd(x_finite, x_ge_0);
    ret = bld.CreateLogicalAnd(ret, x_int);

    return ret;
}

// Add a function to count the number of sign changes in the coefficients
// of a polynomial of degree n. The coefficients are SIMD vectors of size batch_size
// and scalar type scal_t.
llvm::Function *llvm_add_csc(llvm_state &s, llvm::Type *scal_t, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding a sign changes counter function");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(scal_t);

    // Fetch the vector floating-point type.
    auto *tp = make_vector_type(scal_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka_csc_degree_{}_{}", n, llvm_mangle_type(tp));

    // The function arguments:
    // - pointer to the return value,
    // - pointer to the array of coefficients.
    // NOTE: both pointers are to the scalar counterparts
    // of the vector types, so that we can call this from regular
    // C++ code. The second pointer is to an external type.
    const std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(builder.getInt32Ty()),
                                          llvm::PointerType::getUnqual(ext_fp_t)};

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is void.
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *out_ptr = f->args().begin();
        out_ptr->setName("out_ptr");
        out_ptr->addAttr(llvm::Attribute::NoCapture);
        out_ptr->addAttr(llvm::Attribute::NoAlias);
        out_ptr->addAttr(llvm::Attribute::WriteOnly);

        auto *cf_ptr = f->args().begin() + 1;
        cf_ptr->setName("cf_ptr");
        cf_ptr->addAttr(llvm::Attribute::NoCapture);
        cf_ptr->addAttr(llvm::Attribute::NoAlias);
        cf_ptr->addAttr(llvm::Attribute::ReadOnly);

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Fetch the type for storing the last_nz_idx variable.
        auto *last_nz_idx_t = make_vector_type(builder.getInt32Ty(), batch_size);

        // The initial last nz idx is zero for all batch elements.
        auto *last_nz_idx = builder.CreateAlloca(last_nz_idx_t);
        builder.CreateStore(llvm::ConstantInt::get(last_nz_idx_t, 0u), last_nz_idx);

        // NOTE: last_nz_idx is an index into the poly coefficient vector. Thus, in batch
        // mode, when loading from a vector of indices, we will have to apply an offset.
        // For instance, for batch_size = 4 and last_nz_idx = [0, 1, 1, 2], the actual
        // memory indices to load the scalar coefficients from are:
        // - 0 * 4 + 0 = 0
        // - 1 * 4 + 1 = 5
        // - 1 * 4 + 2 = 6
        // - 2 * 4 + 3 = 11.
        // That is, last_nz_idx * batch_size + offset, where offset is [0, 1, 2, 3].
        llvm::Value *offset = nullptr;
        if (batch_size == 1u) {
            // In scalar mode the offset is simply zero.
            offset = builder.getInt32(0);
        } else {
            offset = llvm::UndefValue::get(make_vector_type(builder.getInt32Ty(), batch_size));
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                offset = builder.CreateInsertElement(offset, builder.getInt32(i), i);
            }
        }

        // Init the vector of coefficient pointers with the base pointer value.
        auto *cf_ptr_v = vector_splat(builder, cf_ptr, batch_size);

        // Init the return value with zero.
        auto *retval = builder.CreateAlloca(last_nz_idx_t);
        builder.CreateStore(llvm::ConstantInt::get(last_nz_idx_t, 0u), retval);

        // The iteration range is [1, n].
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n + 1u), [&](llvm::Value *cur_n) {
            // Load the current poly coefficient(s).
            auto *cur_cf = ext_load_vector_from_memory(
                s, scal_t,
                builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, builder.CreateMul(cur_n, builder.getInt32(batch_size))),
                batch_size);

            // Load the last nonzero coefficient(s).
            auto *last_nz_ptr_idx = builder.CreateAdd(
                offset, builder.CreateMul(builder.CreateLoad(last_nz_idx_t, last_nz_idx),
                                          vector_splat(builder, builder.getInt32(batch_size), batch_size)));
            auto *last_nz_ptr = builder.CreateInBoundsGEP(ext_fp_t, cf_ptr_v, last_nz_ptr_idx);
            auto *last_nz_cf = ext_gather_vector_from_memory(s, cur_cf->getType(), last_nz_ptr);

            // Compute the sign of the current coefficient(s).
            auto *cur_sgn = llvm_sgn(s, cur_cf);

            // Compute the sign of the last nonzero coefficient(s).
            auto *last_nz_sgn = llvm_sgn(s, last_nz_cf);

            // Add them and check if the result is zero (this indicates a sign change).
            auto *cmp = builder.CreateICmpEQ(builder.CreateAdd(cur_sgn, last_nz_sgn),
                                             llvm::ConstantInt::get(cur_sgn->getType(), 0u));

            // We also need to check if last_nz_sgn is zero. If that is the case, it means
            // we haven't found any nonzero coefficient yet for the polynomial and we must
            // not modify retval yet.
            auto *zero_cmp = builder.CreateICmpEQ(last_nz_sgn, llvm::ConstantInt::get(last_nz_sgn->getType(), 0u));
            cmp = builder.CreateSelect(zero_cmp, llvm::ConstantInt::get(cmp->getType(), 0u), cmp);

            // Update retval.
            builder.CreateStore(
                builder.CreateAdd(builder.CreateLoad(last_nz_idx_t, retval), builder.CreateZExt(cmp, last_nz_idx_t)),
                retval);

            // Update last_nz_idx.
            builder.CreateStore(
                builder.CreateSelect(builder.CreateICmpEQ(cur_sgn, llvm::ConstantInt::get(cur_sgn->getType(), 0u)),
                                     builder.CreateLoad(last_nz_idx_t, last_nz_idx),
                                     vector_splat(builder, cur_n, batch_size)),
                last_nz_idx);
        });

        // Store the result.
        store_vector_to_memory(builder, out_ptr, builder.CreateLoad(last_nz_idx_t, retval));

        // Return.
        builder.CreateRetVoid();

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Compute the enclosure of the polynomial of order n with coefficients stored in cf_ptr
// over the interval [h_lo, h_hi] using interval arithmetics. The polynomial coefficients
// are vectors of size batch_size and scalar type fp_t. cf_ptr is an external pointer.
// NOTE: the interval arithmetic implementation here is not 100% correct, because
// we do not account for floating-point truncation. In order to be mathematically
// correct, we would need to adjust the results of interval arithmetic add/mul via
// a std::nextafter()-like function. See here for an example:
// https://stackoverflow.com/questions/10420848/how-do-you-get-the-next-value-in-the-floating-point-sequence
// http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node46.html
// Perhaps another alternative would be to employ FP primitives with explicit rounding modes,
// which are available in LLVM. For mppp::real, we could employ the MPFR primitives
// with specific rounding modes.
std::pair<llvm::Value *, llvm::Value *> llvm_penc_interval(llvm_state &s, llvm::Type *fp_t, llvm::Value *cf_ptr,
                                                           std::uint32_t n, llvm::Value *h_lo, llvm::Value *h_hi,
                                                           std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(fp_t != nullptr);
    assert(batch_size > 0u);
    assert(cf_ptr != nullptr);
    assert(h_lo != nullptr);
    assert(h_hi != nullptr);
    assert(llvm::isa<llvm::PointerType>(cf_ptr->getType()));

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while implementing the computation of the enclosure of a "
                                  "polynomial via interval arithmetic");
    }
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Helper to implement the sum of two intervals.
    // NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
    auto ival_sum = [&s](llvm::Value *a_lo, llvm::Value *a_hi, llvm::Value *b_lo, llvm::Value *b_hi) {
        return std::make_pair(llvm_fadd(s, a_lo, b_lo), llvm_fadd(s, a_hi, b_hi));
    };

    // Helper to implement the product of two intervals.
    auto ival_prod = [&s](llvm::Value *a_lo, llvm::Value *a_hi, llvm::Value *b_lo, llvm::Value *b_hi) {
        auto *tmp1 = llvm_fmul(s, a_lo, b_lo);
        auto *tmp2 = llvm_fmul(s, a_lo, b_hi);
        auto *tmp3 = llvm_fmul(s, a_hi, b_lo);
        auto *tmp4 = llvm_fmul(s, a_hi, b_hi);

        // NOTE: here we are not correctly propagating NaNs,
        // for which we would need to use llvm_min/max_nan(),
        // which however incur in a noticeable performance
        // penalty. Thus, even in presence of all finite
        // Taylor coefficients and integration timestep, it could
        // conceivably happen that NaNs are generated in the
        // multiplications above and they are not correctly propagated
        // in these min/max functions, thus ultimately leading to an
        // incorrect result. This however looks like a very unlikely
        // occurrence.
        auto *cmp1 = llvm_min(s, tmp1, tmp2);
        auto *cmp2 = llvm_min(s, tmp3, tmp4);
        auto *cmp3 = llvm_max(s, tmp1, tmp2);
        auto *cmp4 = llvm_max(s, tmp3, tmp4);

        return std::make_pair(llvm_min(s, cmp1, cmp2), llvm_max(s, cmp3, cmp4));
    };

    // Fetch the vector type.
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);

    // Create the lo/hi components of the accumulator.
    auto *acc_lo = builder.CreateAlloca(fp_vec_t);
    auto *acc_hi = builder.CreateAlloca(fp_vec_t);

    // Init the accumulator's lo/hi components with the highest-order coefficient.
    auto *ho_cf = ext_load_vector_from_memory(
        s, fp_t,
        builder.CreateInBoundsGEP(ext_fp_t, cf_ptr,
                                  builder.CreateMul(builder.getInt32(n), builder.getInt32(batch_size))),
        batch_size);
    builder.CreateStore(ho_cf, acc_lo);
    builder.CreateStore(ho_cf, acc_hi);

    // Run the Horner scheme (starting from 1 because we already consumed the
    // highest-order coefficient).
    llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n + 1u), [&](llvm::Value *i) {
        // Load the current coefficient.
        // NOTE: we are iterating backwards from the high-order coefficients
        // to the low-order ones.
        auto *ptr = builder.CreateInBoundsGEP(
            ext_fp_t, cf_ptr,
            builder.CreateMul(builder.CreateSub(builder.getInt32(n), i), builder.getInt32(batch_size)));
        auto *cur_cf = ext_load_vector_from_memory(s, fp_t, ptr, batch_size);

        // Multiply the accumulator by h.
        auto [acc_h_lo, acc_h_hi]
            = ival_prod(builder.CreateLoad(fp_vec_t, acc_lo), builder.CreateLoad(fp_vec_t, acc_hi), h_lo, h_hi);

        // Update the value of the accumulator.
        auto [new_acc_lo, new_acc_hi] = ival_sum(cur_cf, cur_cf, acc_h_lo, acc_h_hi);
        builder.CreateStore(new_acc_lo, acc_lo);
        builder.CreateStore(new_acc_hi, acc_hi);
    });

    // Return the lo/hi components of the accumulator.
    return {builder.CreateLoad(fp_vec_t, acc_lo), builder.CreateLoad(fp_vec_t, acc_hi)};
}

// Compute the enclosure of the polynomial of order n with coefficients stored in cf_ptr
// over an interval using the Cargo-Shisha algorithm. The polynomial coefficients
// are vectors of size batch_size and scalar type fp_t. The interval of the independent variable
// is [0, h] if h >= 0, [h, 0] otherwise. cf_ptr is an external pointer.
// NOTE: the Cargo-Shisha algorithm produces tighter bounds, but it has quadratic complexity
// and it seems to be less well-behaved numerically in corner cases. It might still be worth it up to double-precision
// computations, where the practical slowdown wrt interval arithmetics is smaller.
std::pair<llvm::Value *, llvm::Value *> llvm_penc_cargo_shisha(llvm_state &s, llvm::Type *fp_t, llvm::Value *cf_ptr,
                                                               std::uint32_t n, llvm::Value *h,
                                                               std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(fp_t != nullptr);
    assert(batch_size > 0u);
    assert(cf_ptr != nullptr);
    assert(h != nullptr);
    assert(llvm::isa<llvm::PointerType>(cf_ptr->getType()));

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while implementing the computation of the enclosure of a "
                                  "polynomial via the Cargo-Shisha algorithm");
    }
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // bj_series will contain the terms of the series
    // for the computation of bj. old_bj_series will be
    // used to deal with the fact that the pairwise sum
    // consumes the input vector.
    std::vector<llvm::Value *> bj_series, old_bj_series;

    // Init the current power of h with h itself.
    auto *cur_h_pow = h;

    // Compute the first value, b0, and add it to bj_series.
    auto *b0 = ext_load_vector_from_memory(s, fp_t, cf_ptr, batch_size);
    bj_series.push_back(b0);

    // Init min/max bj with b0.
    auto *min_bj = b0, *max_bj = b0;

    // Main iteration.
    for (std::uint32_t j = 1u; j <= n; ++j) {
        // Compute the new term of the series.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, builder.getInt32(j * batch_size));
        auto *cur_cf = ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
        auto *new_term = llvm_fmul(s, cur_cf, cur_h_pow);
        new_term = llvm_fdiv(s, new_term,
                             vector_splat(builder,
                                          llvm_codegen(s, fp_t,
                                                       binomial(number_like(s, fp_t, static_cast<double>(n)),
                                                                number_like(s, fp_t, static_cast<double>(j)))),
                                          batch_size));

        // Add it to bj_series.
        bj_series.push_back(new_term);

        // Update all elements of bj_series (apart from the last one).
        for (std::uint32_t i = 0; i < j; ++i) {
            bj_series[i] = llvm_fmul(s, bj_series[i],
                                     vector_splat(builder,
                                                  llvm_codegen(s, fp_t,
                                                               number_like(s, fp_t, static_cast<double>(j))
                                                                   / number_like(s, fp_t, static_cast<double>(j - i))),
                                                  batch_size));
        }

        // Compute the new bj.
        old_bj_series = bj_series;
        auto *cur_bj = pairwise_sum(s, bj_series);
        old_bj_series.swap(bj_series);

        // Update min/max_bj.
        min_bj = llvm_min(s, min_bj, cur_bj);
        max_bj = llvm_max(s, max_bj, cur_bj);

        // Update cur_h_pow, if we are not at the last iteration.
        if (j != n) {
            cur_h_pow = llvm_fmul(s, cur_h_pow, h);
        }
    }

    return {min_bj, max_bj};
}

// Helper to create a global const array containing
// all binomial coefficients up to (n, n). The coefficients are stored
// as scalars and the return value is a pointer to the first coefficient.
// The array has shape (n + 1, n + 1) and it is stored in row-major format.
llvm::Value *llvm_add_bc_array(llvm_state &s, llvm::Type *fp_t, std::uint32_t n)
{
    // Overflow check.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || (n + 1u) > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding an array of binomial coefficients");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();

    // Fetch the array type.
    auto *arr_type = llvm::ArrayType::get(fp_t, boost::numeric_cast<std::uint64_t>((n + 1u) * (n + 1u)));

    // Generate the binomials as constants.
    std::vector<llvm::Constant *> bc_const;
    for (std::uint32_t i = 0; i <= n; ++i) {
        for (std::uint32_t j = 0; j <= n; ++j) {
            // NOTE: the Boost implementation requires j <= i. We don't care about
            // j > i anyway.
            const auto val = (j <= i) ? binomial(number_like(s, fp_t, static_cast<double>(i)),
                                                 number_like(s, fp_t, static_cast<double>(j)))
                                      : number_like(s, fp_t, 0.);
            bc_const.push_back(llvm::cast<llvm::Constant>(llvm_codegen(s, fp_t, val)));
        }
    }

    // Create the global array.
    auto *bc_const_arr = llvm::ConstantArray::get(arr_type, bc_const);
    auto *g_bc_const_arr = new llvm::GlobalVariable(md, bc_const_arr->getType(), true,
                                                    llvm::GlobalVariable::PrivateLinkage, bc_const_arr);

    // Get out a pointer to the beginning of the array.
    return builder.CreateInBoundsGEP(bc_const_arr->getType(), g_bc_const_arr,
                                     {builder.getInt32(0), builder.getInt32(0)});
}

namespace
{
// RAII helper to temporarily disable fast
// math flags in a builder.
class fmf_disabler
{
    ir_builder *m_builder;
    const llvm::FastMathFlags m_orig_fmf;

public:
    explicit fmf_disabler(ir_builder &b) : m_builder(&b), m_orig_fmf(m_builder->getFastMathFlags())
    {
        // Reset the fast math flags.
        m_builder->setFastMathFlags(llvm::FastMathFlags{});
    }
    ~fmf_disabler()
    {
        // Restore the original fast math flags.
        m_builder->setFastMathFlags(m_orig_fmf);
    }

    fmf_disabler(const fmf_disabler &) = delete;
    fmf_disabler(fmf_disabler &&) = delete;

    fmf_disabler &operator=(const fmf_disabler &) = delete;
    fmf_disabler &operator=(fmf_disabler &&) = delete;
};

} // namespace

// Error-free transformation of the product of two floating point numbers
// using an FMA. This is algorithm 2.5 here:
// https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
std::pair<llvm::Value *, llvm::Value *> llvm_eft_product(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *x = llvm_fmul(s, a, b);
    auto *y = llvm_fma(s, a, b, llvm_fneg(s, x));

    return {x, y};
}

// Addition.
// NOTE: this is an LLVM port of the original code in NTL.
// See the C++ implementation in dfloat.hpp for an explanation.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_add(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *S = llvm_fadd(state, x_hi, y_hi);
    auto *T = llvm_fadd(state, x_lo, y_lo);
    auto *e = llvm_fsub(state, S, x_hi);
    auto *f = llvm_fsub(state, T, x_lo);

    auto *t1 = llvm_fsub(state, S, e);
    t1 = llvm_fsub(state, x_hi, t1);
    auto *s = llvm_fsub(state, y_hi, e);
    s = llvm_fadd(state, s, t1);

    t1 = llvm_fsub(state, T, f);
    t1 = llvm_fsub(state, x_lo, t1);
    auto *t = llvm_fsub(state, y_lo, f);
    t = llvm_fadd(state, t, t1);

    s = llvm_fadd(state, s, T);
    auto *H = llvm_fadd(state, S, s);
    auto *h = llvm_fsub(state, S, H);
    h = llvm_fadd(state, h, s);

    h = llvm_fadd(state, h, t);
    e = llvm_fadd(state, H, h);
    f = llvm_fsub(state, H, e);
    f = llvm_fadd(state, f, h);

    return {e, f};
}

// Subtraction.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_sub(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    return llvm_dl_add(state, x_hi, x_lo, llvm_fneg(state, y_hi), llvm_fneg(state, y_lo));
}

// Multiplication.
// NOTE: this is procedure mul2() from here:
// https://link.springer.com/content/pdf/10.1007/BF01397083.pdf
// The mul12() function is replaced with the FMA-based llvm_eft_product().
// NOTE: the code in NTL looks identical to Dekker's.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_mul(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto [c, cc] = llvm_eft_product(s, x_hi, y_hi);

    // cc = x*yy + xx*y + cc.
    auto *x_yy = llvm_fmul(s, x_hi, y_lo);
    auto *xx_y = llvm_fmul(s, x_lo, y_hi);
    cc = llvm_fadd(s, llvm_fadd(s, x_yy, xx_y), cc);

    // The normalisation step.
    auto *z = llvm_fadd(s, c, cc);
    auto *zz = llvm_fadd(s, llvm_fsub(s, c, z), cc);

    return {z, zz};
}

// Division.
// NOTE: this is procedure div2() from here:
// https://link.springer.com/content/pdf/10.1007/BF01397083.pdf
// The mul12() function is replaced with the FMA-based llvm_eft_product().
// NOTE: the code in NTL looks identical to Dekker's.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_div(llvm_state &s,
                                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                    llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi,
                                                    llvm::Value *y_lo)
{
    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *c = llvm_fdiv(s, x_hi, y_hi);

    auto [u, uu] = llvm_eft_product(s, c, y_hi);

    // cc = (x_hi - u - uu + x_lo - c * y_lo) / y_hi.
    auto *cc = llvm_fsub(s, x_hi, u);
    cc = llvm_fsub(s, cc, uu);
    cc = llvm_fadd(s, cc, x_lo);
    cc = llvm_fsub(s, cc, llvm_fmul(s, c, y_lo));
    cc = llvm_fdiv(s, cc, y_hi);

    // The normalisation step.
    auto *z = llvm_fadd(s, c, cc);
    auto *zz = llvm_fadd(s, llvm_fsub(s, c, z), cc);

    return {z, zz};
}

// Floor.
// NOTE: code taken from NTL:
// https://github.com/libntl/ntl/blob/main/src/quad_float1.cpp#L239
std::pair<llvm::Value *, llvm::Value *> llvm_dl_floor(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo)
{
    // LCOV_EXCL_START
    assert(x_hi != nullptr);
    assert(x_lo != nullptr);
    assert(x_hi->getType() == x_lo->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = x_hi->getType();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    // Floor x_hi.
    auto *fhi = llvm_floor(s, x_hi);

    // NOTE: we want to distinguish the scalar/vector codepaths, as the vectorised implementation
    // does more work.
    const auto vec_size = get_vector_size(x_hi);

    if (vec_size == 1u) {
        auto *ret_hi_ptr = builder.CreateAlloca(fp_t);
        auto *ret_lo_ptr = builder.CreateAlloca(fp_t);

        llvm_if_then_else(
            s, llvm_fcmp_oeq(s, fhi, x_hi),
            [&]() {
                // floor(x_hi) == x_hi, that is, x_hi is already
                // an integral value.

                // Floor the low part.
                auto *flo = llvm_floor(s, x_lo);

                // Normalise.
                auto *z = llvm_fadd(s, fhi, flo);
                auto *zz = llvm_fadd(s, llvm_fsub(s, fhi, z), flo);

                // Store.
                builder.CreateStore(z, ret_hi_ptr);
                builder.CreateStore(zz, ret_lo_ptr);
            },
            [&]() {
                // floor(x_hi) != x_hi. Just need to set the low part to zero.
                builder.CreateStore(fhi, ret_hi_ptr);
                builder.CreateStore(llvm_constantfp(s, fp_t, 0.), ret_lo_ptr);
            });

        return {builder.CreateLoad(fp_t, ret_hi_ptr), builder.CreateLoad(fp_t, ret_lo_ptr)};
    } else {
        // Get a vector of zeroes.
        auto *zero_vec = llvm_constantfp(s, fp_t, 0.);

        // Floor the low part.
        auto *flo = llvm_floor(s, x_lo);

        // Select flo or zero_vec, depending on fhi == x_hi.
        auto *ret_lo = builder.CreateSelect(llvm_fcmp_oeq(s, fhi, x_hi), flo, zero_vec);

        // Normalise.
        auto *z = llvm_fadd(s, fhi, ret_lo);
        auto *zz = llvm_fadd(s, llvm_fsub(s, fhi, z), ret_lo);

        return {z, zz};
    }
}

// Helper to reduce x modulo y, that is, to compute:
// x - y * floor(x / y).
std::pair<llvm::Value *, llvm::Value *> llvm_dl_modulus(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo,
                                                        llvm::Value *y_hi, llvm::Value *y_lo)
{
    // LCOV_EXCL_START
    assert(x_hi != nullptr);
    assert(x_lo != nullptr);
    assert(y_hi != nullptr);
    assert(y_lo != nullptr);
    assert(x_hi->getType() == x_lo->getType());
    assert(x_hi->getType() == y_hi->getType());
    assert(x_hi->getType() == y_lo->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto [xoy_hi, xoy_lo] = llvm_dl_div(s, x_hi, x_lo, y_hi, y_lo);
    auto [fl_hi, fl_lo] = llvm_dl_floor(s, xoy_hi, xoy_lo);
    auto [prod_hi, prod_lo] = llvm_dl_mul(s, y_hi, y_lo, fl_hi, fl_lo);

    return llvm_dl_sub(s, x_hi, x_lo, prod_hi, prod_lo);
}

// Less-than.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_dl_lt(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *cond1 = llvm_fcmp_olt(state, x_hi, y_hi);
    auto *cond2 = llvm_fcmp_oeq(state, x_hi, y_hi);
    auto *cond3 = llvm_fcmp_olt(state, x_lo, y_lo);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto *cond4 = builder.CreateLogicalAnd(cond2, cond3);
    auto *cond = builder.CreateLogicalOr(cond1, cond4);

    return cond;
}

// Greater-than.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_dl_gt(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *cond1 = llvm_fcmp_ogt(state, x_hi, y_hi);
    auto *cond2 = llvm_fcmp_oeq(state, x_hi, y_hi);
    auto *cond3 = llvm_fcmp_ogt(state, x_lo, y_lo);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto *cond4 = builder.CreateLogicalAnd(cond2, cond3);
    auto *cond = builder.CreateLogicalOr(cond1, cond4);

    return cond;
}

// Inverse cosine.
llvm::Value *llvm_acos(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "acos", x);
}

// Inverse hyperbolic cosine.
llvm::Value *llvm_acosh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "acosh", x);
}

// Inverse sine.
llvm::Value *llvm_asin(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "asin", x);
}

// Inverse hyperbolic sine.
llvm::Value *llvm_asinh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "asinh", x);
}

// Inverse tangent.
llvm::Value *llvm_atan(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "atan", x);
}

// Inverse hyperbolic tangent.
llvm::Value *llvm_atanh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "atanh", x);
}

// Cosine.
llvm::Value *llvm_cos(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.cos",
#if defined(HEYOKA_HAVE_REAL128)
                          "cosq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_cos",
#endif
                          x);
}

// Sine.
llvm::Value *llvm_sin(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.sin",
#if defined(HEYOKA_HAVE_REAL128)
                          "sinq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_sin",
#endif
                          x);
}

// Hyperbolic cosine.
llvm::Value *llvm_cosh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "cosh", x);
}

// Error function.
llvm::Value *llvm_erf(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "erf", x);
}

// Natural logarithm.
llvm::Value *llvm_log(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.log",
#if defined(HEYOKA_HAVE_REAL128)
                          "logq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_log",
#endif
                          x);
}

// Sigmoid.
llvm::Value *llvm_sigmoid(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    // Create the 1 constant.
    auto *one_fp = llvm_constantfp(s, x->getType(), 1.);

    // Compute -x.
    auto *m_x = llvm_fneg(s, x);

    // Compute e^(-x).
    auto *e_m_x = llvm_exp(s, m_x);

    // Return 1 / (1 + e_m_arg).
    return llvm_fdiv(s, one_fp, llvm_fadd(s, one_fp, e_m_x));
}

// Hyperbolic sine.
llvm::Value *llvm_sinh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "sinh", x);
}

// Square root.
llvm::Value *llvm_sqrt(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.sqrt",
#if defined(HEYOKA_HAVE_REAL128)
                          "sqrtq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_sqrt",
#endif
                          x);
}

// Squaring.
llvm::Value *llvm_square(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    if (x->getType()->getScalarType()->isFloatingPointTy()) {
        return s.builder().CreateFMul(x, x);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "mpfr_sqr", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of square()",
                                                llvm_type_name(x->getType())));
    }
}

// Tangent.
llvm::Value *llvm_tan(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "tan", x);
}

// Hyperbolic tangent.
llvm::Value *llvm_tanh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "tanh", x);
}

// Exponentiation.
llvm::Value *llvm_pow(llvm_state &s, llvm::Value *x, llvm::Value *y)
{
    return llvm_math_intr(s, "llvm.pow",
#if defined(HEYOKA_HAVE_REAL128)
                          "powq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_pow",
#endif
                          x, y);
}

template <typename T>
llvm::Type *to_internal_llvm_type(llvm_state &s, [[maybe_unused]] long long prec)
{
    auto &c = s.context();

#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::same_as<T, mppp::real>) {
        // Checks on prec.
        // LCOV_EXCL_START
        // NOLINTNEXTLINE(misc-redundant-expression)
        if (prec == 0 || prec < mppp::real_prec_min() || prec > mppp::real_prec_max()) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("An invalid precision value of {} was passed to to_internal_llvm_type()", prec));
        }
        // LCOV_EXCL_STOP

        const auto name = fmt::format("heyoka.real.{}", prec);

        if (auto *ptr = llvm::StructType::getTypeByName(c, name)) {
            return ptr;
        }

        // Compute the required number of limbs.
        // NOTE: this is a computation done in the implementation of mppp::real and
        // reproduced here. We should consider exposing this functionality in mp++.
        const auto nlimbs = boost::numeric_cast<std::uint64_t>((prec / GMP_NUMB_BITS)
                                                               + static_cast<int>((prec % GMP_NUMB_BITS) != 0));

        // Fetch the limb array type.
        auto *limb_arr_t = llvm::ArrayType::get(to_external_llvm_type<mp_limb_t>(c), nlimbs);

        auto *ret = llvm::StructType::create(
            {to_external_llvm_type<mpfr_sign_t>(c), to_external_llvm_type<mpfr_exp_t>(c), limb_arr_t}, name);

        assert(ret != nullptr);
        assert(llvm::StructType::getTypeByName(c, name) == ret);

        return ret;
    } else {
#endif
        // NOTE: for anything else than mppp::real, the internal and
        // external types coincide.
        return to_external_llvm_type<T>(c);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Type *to_internal_llvm_type<float>(llvm_state &, long long);

template HEYOKA_DLL_PUBLIC llvm::Type *to_internal_llvm_type<double>(llvm_state &, long long);

template HEYOKA_DLL_PUBLIC llvm::Type *to_internal_llvm_type<long double>(llvm_state &, long long);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC llvm::Type *to_internal_llvm_type<mppp::real128>(llvm_state &, long long);

#endif

#if defined(HEYOKA_HAVE_REAL)

template HEYOKA_DLL_PUBLIC llvm::Type *to_internal_llvm_type<mppp::real>(llvm_state &, long long);

#endif

// This helper returns the type to be used for the internal LLVM representation
// of the input value x.
template <typename T>
llvm::Type *internal_llvm_type_like(llvm_state &s, [[maybe_unused]] const T &x)
{
    auto &c = s.context();

#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        return to_internal_llvm_type<T>(s, x.get_prec());
    } else {
#endif
        // NOTE: for anything else than mppp::real, the internal and
        // external types coincide.
        return to_external_llvm_type<T>(c);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Type *internal_llvm_type_like<float>(llvm_state &, const float &);

template HEYOKA_DLL_PUBLIC llvm::Type *internal_llvm_type_like<double>(llvm_state &, const double &);

template HEYOKA_DLL_PUBLIC llvm::Type *internal_llvm_type_like<long double>(llvm_state &, const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC llvm::Type *internal_llvm_type_like<mppp::real128>(llvm_state &, const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template HEYOKA_DLL_PUBLIC llvm::Type *internal_llvm_type_like<mppp::real>(llvm_state &, const mppp::real &);

#endif

// Fetch the external llvm type corresponding to the input external llvm type.
llvm::Type *make_external_llvm_type(llvm::Type *fp_t)
{
    if (fp_t->isFloatingPointTy() || fp_t->isIntegerTy()) {
        return fp_t;
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return to_external_llvm_type<mppp::real>(fp_t->getContext());
#endif
        // LCOV_EXCL_START
    } else {
        throw std::invalid_argument(
            fmt::format("Cannot associate an external LLVM type to the internal LLVM type '{}'", llvm_type_name(fp_t)));
    }
    // LCOV_EXCL_STOP
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

// Utility to create an identical copy of the type tp in the context of the state s.
// NOTE: although it may sound like this is a read-only operation on tp, it is not,
// since we are potentially poking into the context of tp during operations. Thus, this
// function cannot be called concurrently from multiple threads on the same tp object,
// or even on different tp objects defined in the same context.
// NOTE: this handles only floating-point (vector) types at this time, extending
// to integral types should be fairly easy.
// NOTE: perhaps this function could be made more generic for arbitrary struct types
// by (recursively) reading the struct layout and then reproducing it in the target
// context. Like this, we could avoid special casing for the mppp::real types.
llvm::Type *llvm_clone_type(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);

    // Fetch the target context.
    auto &ctx = s.context();

    // Construct the scalar type first, then we will convert
    // to a vector if needed.
    auto *tp_scal = tp->getScalarType();
    llvm::Type *ret_scal_t = nullptr;

#define HEYOKA_LLVM_CLONE_TYPE_IMPL(tid)                                                                               \
    case llvm::Type::tid##TyID:                                                                                        \
        ret_scal_t = llvm::Type::get##tid##Ty(ctx);                                                                    \
        break

    // NOTE: gcov seems to get a bit confused by the macro usage.
    // LCOV_EXCL_START
    switch (tp_scal->getTypeID()) {
        HEYOKA_LLVM_CLONE_TYPE_IMPL(Float);
        HEYOKA_LLVM_CLONE_TYPE_IMPL(Double);
        HEYOKA_LLVM_CLONE_TYPE_IMPL(X86_FP80);
        HEYOKA_LLVM_CLONE_TYPE_IMPL(FP128);
        default: {

#if defined(HEYOKA_HAVE_REAL)

            if (const auto prec = llvm_is_real(tp_scal); prec != 0) {
                // tp_scal is the internal counterpart of mppp::real.
                ret_scal_t = to_internal_llvm_type<mppp::real>(s, prec);
                break;
            } else if (tp_scal == to_external_llvm_type<mppp::real>(tp_scal->getContext())) {
                // tp_scal is mppp::real.
                ret_scal_t = to_external_llvm_type<mppp::real>(ctx);
                break;
            }

#endif

            throw std::invalid_argument(
                fmt::format("Cannot clone the LLVM type '{}' to another context", llvm_type_name(tp)));
        }
    }

#undef HEYOKA_LLVM_CLONE_TYPE_IMPL
    // LCOV_EXCL_STOP

    assert(ret_scal_t != nullptr);

    if (tp->isVectorTy()) {
        // tp is a vector type.
        if (const auto *vtp = llvm::dyn_cast<llvm::FixedVectorType>(tp)) [[likely]] {
            return make_vector_type(ret_scal_t, boost::numeric_cast<std::uint32_t>(vtp->getNumElements()));
        } else {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("Cannot clone the LLVM type '{}' to another context - the type is "
                                                    "a vector type whose size is not fixed",
                                                    llvm_type_name(tp)));
            // LCOV_EXCL_STOP
        }
    } else {
        // tp is a scalar type.
        return ret_scal_t;
    }
}

// Implementation of the std::upper_bound() algorithm for floating-point types.
//
// Given an array of sorted scalar values beginning at ptr and of size arr_size (a 32-bit int), this function
// will return the index of the first element in the array that is *greater than* v. If no such element exists,
// arr_size will be returned. v can be a scalar or a vector.
//
// The algorithm is short enough to be reproduced here:
//
// template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type, class Compare>
// ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
// {
//     ForwardIt it;
//     typename std::iterator_traits<ForwardIt>::difference_type count, step;
//     count = std::distance(first, last);
//
//     while (count > 0)
//     {
//         it = first;
//         step = count / 2;
//         std::advance(it, step);
//
//         if (!comp(value, *it))
//         {
//             first = ++it;
//             count -= step + 1;
//         }
//         else
//             count = step;
//     }
//
//     return first;
// }
//
// Particular care must be taken for the vector implementation: while in a scalar implementation
// the bisection loop is never entered if count == 0, in the vector implementation we will be entering
// the bisection loop with count == 0 whenever a SIMD lane has finished but the other SIMD lanes have not.
// In a loop iteration with count == 0, the following happens:
//
// - step is set to 0 and 'it' remains inited to 'first';
// - 'it' may be pointing one past the end of the array, and thus we must take care
//   of *not* dereferencing it if that is the case. We thus have two possibilities:
//   - 'it' points somewhere in the array: in this case, we know that 'it' points to
//     the first array element greater than 'v', and thus '!comp(value, *it)' evaluates
//     to false, we end up in the 'count = step' branch, where count is set again
//     to 0. Thus, neither 'first' nor 'count' are altered;
//   - 'it' points one past the end: in this case, we must avoid reading from it and we must
//     replace the condition '!comp(value, *it)' with 'false', so that we end up in the
//     'count = step' branch.
llvm::Value *llvm_upper_bound(llvm_state &s, llvm::Value *ptr, llvm::Value *arr_size, llvm::Value *v)
{
    assert(ptr != nullptr);
    // NOTE: this will also check that ptr is not a vector of pointers.
    assert(ptr->getType()->isPointerTy());
    assert(arr_size != nullptr);
    assert(v != nullptr);

    auto &bld = s.builder();

    // NOTE: infer the scalar type from v.
    auto *scal_t = v->getType()->getScalarType();

    // Fetch the 32-bit int type.
    auto *int32_tp = bld.getInt32Ty();
    // NOTE: this will also check that arr_size is not a vector of values.
    assert(arr_size->getType() == int32_tp);

    // Determine the batch size.
    std::uint32_t batch_size = 1;
    if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(v->getType())) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
        assert(batch_size != 0u);
    }

    // Splat ptr to the batch size.
    auto *arr_ptr = vector_splat(bld, ptr, batch_size);

    // The type to be used for indexing into the scalar array.
    auto *idx_vec_t = make_vector_type(int32_tp, batch_size);

    // The type that will be loaded from the scalar array.
    auto *fp_vec_t = make_vector_type(scal_t, batch_size);

    // Create arrays of 32-bit constants for use below.
    auto *zero_vec_i32 = vector_splat(bld, bld.getInt32(0), batch_size);
    auto *one_vec_i32 = vector_splat(bld, bld.getInt32(1), batch_size);
    auto *two_vec_i32 = vector_splat(bld, bld.getInt32(2), batch_size);

    // Create the 'first' and 'count' variables.
    // NOTE: as "iterator" type we will be using a 32-bit int.
    auto *first = bld.CreateAlloca(idx_vec_t);
    auto *count = bld.CreateAlloca(idx_vec_t);

    // The 'first' iterator is inited with zeros (as it points to the beginning of the array).
    bld.CreateStore(zero_vec_i32, first);

    // 'count' is inited with the size of the array.
    auto *arr_size_splat = vector_splat(bld, arr_size, batch_size);
    bld.CreateStore(arr_size_splat, count);

    // Iterate as long as all elements of 'count' are > 0.
    llvm_while_loop(
        s,
        [&bld, count, idx_vec_t, zero_vec_i32, batch_size]() -> llvm::Value * {
            auto *cmp = bld.CreateICmpUGT(bld.CreateLoad(idx_vec_t, count), zero_vec_i32);

            // NOTE: in scalar mode, no reduction is needed.
            return (batch_size == 1u) ? cmp : bld.CreateOrReduce(cmp);
        },
        [&bld, &s, first, count, idx_vec_t, batch_size, scal_t, fp_vec_t, v, one_vec_i32, arr_ptr, arr_size_splat,
         two_vec_i32]() {
            // Load the value stored in 'first' - this will be the iterator we will
            // be using in the current iteration of the loop.
            llvm::Value *cur_first = bld.CreateLoad(idx_vec_t, first);
            auto *it = cur_first;

            // Compute the step value for the current iteration: step = count / 2.
            llvm::Value *cur_count = bld.CreateLoad(idx_vec_t, count);
            auto *step = bld.CreateUDiv(cur_count, two_vec_i32);

            // Advance 'it' by step.
            it = bld.CreateAdd(it, step);

            // Load the value(s) from 'it' into 'cur_value'. 'mask' is used only in vector mode, otherwise it
            // remains null.
            llvm::Value *cur_value{}, *mask{};
            if (batch_size == 1u) {
                // Normal scalar load.
                cur_value = bld.CreateLoad(scal_t, bld.CreateInBoundsGEP(scal_t, arr_ptr, {it}));
            } else {
                // NOTE: as explained above, in vector mode we must take care to avoid loading from 'it'
                // if it points one past the end. We accomplish this with a masked gather.

                // Fetch the alignment of the scalar type.
                const auto align = get_alignment(s.module(), scal_t);

                // Identify the SIMD lane(s) which are *not* reading past the end of the array, storing the
                // result in 'mask'.
                mask = bld.CreateICmpNE(it, arr_size_splat);

                // As a passthru value for the masked gather, use the v value itself.
                auto *passthru = v;

                // Masked gather with passthru.
                cur_value = bld.CreateMaskedGather(fp_vec_t, bld.CreateInBoundsGEP(scal_t, arr_ptr, {it}),
                                                   llvm::Align(align), mask, passthru);
            }

            // Run the comparison.
            // NOTE: the original comparison would be '!comp(value, *it)', which translates to '!(v < *it)' in
            // the current code. *it can never be NaN (apart from a corner case in vector mode when the passthru
            // value is NaN, but we take care of this later ANDing the mask). v could be NaN, in which case we want
            // '!(v < *it)' to evaluate to true because we want to consider NaN greater than non-NaN. In order to do
            // this, we flip the comparison around to '*it <= v', and we implement it via the ULE predicate, which
            // returns true if either v is NaN or '*it <= v'.
            auto *cmp = llvm_fcmp_ule(s, cur_value, v);
            if (batch_size != 1u) {
                // NOTE: in vector mode, we must take care that cmp for the masked-out lanes evaluates to false.
                assert(mask != nullptr);
                cmp = bld.CreateAnd(cmp, mask);
            }

            // We now need to update 'first' and 'count'. Branch on the batch size for efficiency.
            if (batch_size == 1u) {
                // Scalar implementation.
                llvm_if_then_else(
                    s, cmp,
                    [&bld, one_vec_i32, it, first, count, step, cur_count]() {
                        // Assign it + 1 to first.
                        auto *itp1 = bld.CreateAdd(it, one_vec_i32);
                        bld.CreateStore(itp1, first);

                        // Compute new_count = count - (step + 1).
                        auto *stepp1 = bld.CreateAdd(step, one_vec_i32);
                        auto *new_count = bld.CreateSub(cur_count, stepp1);

                        // Assign new_count to count.
                        bld.CreateStore(new_count, count);
                    },
                    [&bld, step, count]() {
                        // NOTE: no update of 'first' needed here.
                        // Assign step to count.
                        bld.CreateStore(step, count);
                    });
            } else {
                // Vector implementation.

                // Compute it + 1.
                auto *itp1 = bld.CreateAdd(it, one_vec_i32);

                // Compute step + 1.
                auto *stepp1 = bld.CreateAdd(step, one_vec_i32);

                // Compute count - (step + 1).
                auto *count_m_stepp1 = bld.CreateSub(cur_count, stepp1);

                // Compute the new first = cmp ? (it + 1) : cur_first.
                auto *new_first = bld.CreateSelect(cmp, itp1, cur_first);

                // Compute the new count = cmp ? (count - (step + 1)) : step.
                auto *new_count = bld.CreateSelect(cmp, count_m_stepp1, step);

                // Store the new first and the new count.
                bld.CreateStore(new_count, count);
                bld.CreateStore(new_first, first);
            }
        });

    // The return value is the value stored in 'first'.
    return bld.CreateLoad(idx_vec_t, first);
}

} // namespace detail

HEYOKA_END_NAMESPACE

#if !defined(NDEBUG)

#if defined(HEYOKA_HAVE_REAL)

extern "C" HEYOKA_DLL_PUBLIC void heyoka_assert_real_match_precs_ext_load(mpfr_prec_t p1, mpfr_prec_t p2) noexcept
{
    assert(p1 == p2);
}

extern "C" HEYOKA_DLL_PUBLIC void heyoka_assert_real_match_precs_ext_store(mpfr_prec_t p1, mpfr_prec_t p2) noexcept
{
    assert(p1 == p2);
}

#endif

#endif

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
