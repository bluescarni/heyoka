// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/ADT/APFloat.h>
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

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/sleef.hpp>
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

namespace heyoka::detail
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
// NOLINTNEXTLINE(cert-err58-cpp,readability-function-cognitive-complexity)
const auto type_map = []() {
    std::unordered_map<std::type_index, llvm::Type *(*)(llvm::LLVMContext &)> retval;

    // Try to associate C++ float to LLVM float.
    if (std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24) {
        retval[typeid(float)] = [](llvm::LLVMContext &c) { return llvm::Type::getFloatTy(c); };
    }

    // Try to associate C++ double to LLVM double.
    if (std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53) {
        retval[typeid(double)] = [](llvm::LLVMContext &c) { return llvm::Type::getDoubleTy(c); };
    }

    // Try to associate C++ long double to an LLVM fp type.
    if (std::numeric_limits<long double>::is_iec559) {
        if (std::numeric_limits<long double>::digits == 53) {
            retval[typeid(long double)] = [](llvm::LLVMContext &c) {
                // IEEE double-precision format (this is the case on MSVC for instance).
                return llvm::Type::getDoubleTy(c);
            };
        } else if (std::numeric_limits<long double>::digits == 64) {
            retval[typeid(long double)] = [](llvm::LLVMContext &c) {
                // x86 extended precision format.
                return llvm::Type::getX86_FP80Ty(c);
            };
        } else if (std::numeric_limits<long double>::digits == 113) {
            retval[typeid(long double)] = [](llvm::LLVMContext &c) {
                // IEEE quadruple-precision format (e.g., ARM 64).
                return llvm::Type::getFP128Ty(c);
            };
        }
    }

#if defined(HEYOKA_HAVE_REAL128)

    // Associate mppp::real128 to fp128.
    retval[typeid(mppp::real128)] = [](llvm::LLVMContext &c) { return llvm::Type::getFP128Ty(c); };

#endif

#if defined(HEYOKA_HAVE_REAL)

    retval[typeid(mppp::real)] = [](llvm::LLVMContext &c) -> llvm::Type * {
#if LLVM_VERSION_MAJOR >= 12
        if (auto *ptr = llvm::StructType::getTypeByName(c, "heyoka.real")) {
            return ptr;
        }

        auto *ret = llvm::StructType::create({to_llvm_type<real_prec_t>(c), to_llvm_type<real_sign_t>(c),
                                              to_llvm_type<real_exp_t>(c),
                                              llvm::PointerType::getUnqual(to_llvm_type<real_limb_t>(c))},
                                             "heyoka.real");

        assert(ret != nullptr);
        assert(llvm::StructType::getTypeByName(c, "heyoka.real") == ret);

        return ret;
#else
        // NOTE: in earlier LLVM versions, make this an unnamed struct.
        auto *ret = llvm::StructType::get(c, {to_llvm_type<real_prec_t>(c), to_llvm_type<real_sign_t>(c),
                                              to_llvm_type<real_exp_t>(c),
                                              llvm::PointerType::getUnqual(to_llvm_type<real_limb_t>(c))});

        assert(ret != nullptr);

        return ret;
#endif
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

} // namespace

// Implementation of the function to associate a C++ type to
// an LLVM type.
llvm::Type *to_llvm_type_impl(llvm::LLVMContext &c, const std::type_info &tp, bool err_throw)
{
    const auto it = type_map.find(tp);

    const auto *err_msg = "Unable to associate the C++ type '{}' to an LLVM type";

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

    if (auto *v_t = llvm::dyn_cast<llvm_vector_type>(t)) {
        // If the type is a vector, get the name of the element type
        // and append the vector size.
        return fmt::format("{}_{}", llvm_type_name(v_t->getElementType()), v_t->getNumElements());
    } else {
        // Otherwise just return the type name.
        return llvm_type_name(t);
    }
}

// Helper to determine the vector size of x. If x is a scalar,
// 1 will be returned.
std::uint32_t get_vector_size(llvm::Value *x)
{
    if (auto *vector_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
        return boost::numeric_cast<std::uint32_t>(vector_t->getNumElements());
    } else {
        return 1;
    }
}

// Fetch the alignment of a type.
std::uint64_t get_alignment(llvm::Module &md, llvm::Type *tp)
{
    return md.getDataLayout().getABITypeAlignment(tp);
}

// Convert the input integral value n to the type std::size_t.
// If an upcast is needed, it will be performed via zero extension.
llvm::Value *to_size_t(llvm_state &s, llvm::Value *n)
{
    // Get the bit width of the type of n.
    const auto n_bw = llvm::cast<llvm::IntegerType>(n->getType()->getScalarType())->getBitWidth();

    // Fetch the LLVM type corresponding to size_t, and its bit width.
    auto *lst = to_llvm_type<std::size_t>(s.context());
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
    auto *gl_arr = new llvm::GlobalVariable(m, t, false, llvm::GlobalVariable::InternalLinkage,
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
    assert(!llvm::isa<llvm_vector_type>(ptr->getType()));
    // LCOV_EXCL_STOP

    if (vector_size == 1u) {
        // Scalar case.
        return builder.CreateLoad(tp, ptr);
    }

    // Create the vector type.
    auto vector_t = make_vector_type(tp, vector_size);
    assert(vector_t != nullptr); // LCOV_EXCL_LINE

    // Create the mask (all 1s).
    auto mask = llvm::ConstantInt::get(make_vector_type(builder.getInt1Ty(), vector_size), 1u);

    // Create the passthrough value. This can stay undefined as it is never used
    // due to the mask being all 1s.
    auto passthru = llvm::UndefValue::get(vector_t);

    // Invoke the intrinsic.
    auto ret = llvm_invoke_intrinsic(builder, "llvm.masked.expandload", {vector_t}, {ptr, mask, passthru});

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
        auto *limb_t = to_llvm_type<real_limb_t>(context);

        // Fetch the external real struct type.
        auto *real_t = to_llvm_type<mppp::real>(context);

        // Compute the number of limbs in the internal real type.
        const auto nlimbs = mppp::prec_to_nlimbs(real_prec);

#if !defined(NDEBUG)

        // In debug mode, we want to assert that the precision of the internal
        // type matches exactly the precision of the external variable.
        if (s.opt_level() == 0u) {
            // Load the precision from the external value.
            auto *prec_t = to_llvm_type<real_prec_t>(context);
            auto *prec_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(0)});
            auto *prec = builder.CreateLoad(prec_t, prec_ptr);

            llvm_invoke_external(
                s, "heyoka_assert_real_match_precs_ext_load", builder.getVoidTy(),
                {prec, llvm::ConstantInt::getSigned(prec_t, boost::numeric_cast<std::int64_t>(real_prec))});
        }

#endif

        // Init the return value.
        llvm::Value *ret = llvm::UndefValue::get(tp);

        // Read and insert the sign.
        auto *sign_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(1)});
        auto *sign = builder.CreateLoad(to_llvm_type<real_sign_t>(context), sign_ptr);
        ret = builder.CreateInsertValue(ret, sign, {0u});

        // Read and insert the exponent.
        auto *exp_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(2)});
        auto *exp = builder.CreateLoad(to_llvm_type<real_exp_t>(context), exp_ptr);
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
    assert(!llvm::isa<llvm_vector_type>(ptr->getType()));
    // LCOV_EXCL_STOP

    if (auto vector_t = llvm::dyn_cast<llvm_vector_type>(vec->getType())) {
        // Determine the vector size.
        const auto vector_size = boost::numeric_cast<std::uint32_t>(vector_t->getNumElements());

        // Create the mask (all 1s).
        auto mask = llvm::ConstantInt::get(make_vector_type(builder.getInt1Ty(), vector_size), 1u);

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
        auto *limb_t = to_llvm_type<real_limb_t>(context);

        // Fetch the external real struct type.
        auto *real_t = to_llvm_type<mppp::real>(context);

        // Compute the number of limbs in the internal real type.
        const auto nlimbs = mppp::prec_to_nlimbs(real_prec);

#if !defined(NDEBUG)

        // In debug mode, we want to assert that the precision of the internal
        // type matches exactly the precision of the external variable.
        if (s.opt_level() == 0u) {
            // Load the precision from the external value.
            auto *prec_t = to_llvm_type<real_prec_t>(context);
            auto *out_prec_ptr = builder.CreateInBoundsGEP(real_t, ptr, {builder.getInt32(0), builder.getInt32(0)});
            auto *prec = builder.CreateLoad(prec_t, out_prec_ptr);

            llvm_invoke_external(
                s, "heyoka_assert_real_match_precs_ext_store", builder.getVoidTy(),
                {prec, llvm::ConstantInt::getSigned(prec_t, boost::numeric_cast<std::int64_t>(real_prec))});
        }

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

// Gather a vector of type vec_tp from ptrs. If vec_tp is a vector type, then ptrs
// must be a vector of pointers of the same size and the returned value is also a vector
// of that size. Otherwise, ptrs must be a single scalar pointer and the returned value is a scalar.
llvm::Value *gather_vector_from_memory(ir_builder &builder, llvm::Type *vec_tp, llvm::Value *ptrs)
{
    if (llvm::isa<llvm_vector_type>(vec_tp)) {
        // LCOV_EXCL_START
        assert(llvm::isa<llvm_vector_type>(ptrs->getType()));
        assert(llvm::cast<llvm_vector_type>(vec_tp)->getNumElements()
               == llvm::cast<llvm_vector_type>(ptrs->getType())->getNumElements());
        // LCOV_EXCL_STOP

        // Fetch the alignment of the scalar type.
        const auto align = get_alignment(*builder.GetInsertBlock()->getModule(), vec_tp->getScalarType());

        return builder.CreateMaskedGather(
#if LLVM_VERSION_MAJOR >= 13
            // NOTE: new initial argument required since LLVM 13
            // (the vector type to gather).
            vec_tp,
#endif
            ptrs,
#if LLVM_VERSION_MAJOR == 10
            boost::numeric_cast<unsigned>(align)
#else
            llvm::Align(align)
#endif
        );
    } else {
        // LCOV_EXCL_START
        assert(!llvm::isa<llvm_vector_type>(ptrs->getType()));
        // LCOV_EXCL_STOP

        return builder.CreateLoad(vec_tp, ptrs);
    }
}

// Create a SIMD vector of size vector_size filled with the value c. If vector_size is 1,
// c will be returned.
llvm::Value *vector_splat(ir_builder &builder, llvm::Value *c, std::uint32_t vector_size)
{
    // LCOV_EXCL_START
    assert(vector_size > 0u);
    assert(!llvm::isa<llvm_vector_type>(c->getType()));
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
    assert(!llvm::isa<llvm_vector_type>(t));
    // LCOV_EXCL_STOP

    if (vector_size == 1u) {
        return t;
    } else {
        auto retval = llvm_vector_type::get(t, boost::numeric_cast<unsigned>(vector_size));

        assert(retval != nullptr); // LCOV_EXCL_LINE

        return retval;
    }
}

// Convert the input LLVM vector to a std::vector of values. If vec is not a vector,
// return {vec}.
std::vector<llvm::Value *> vector_to_scalars(ir_builder &builder, llvm::Value *vec)
{
    if (auto vec_t = llvm::dyn_cast<llvm_vector_type>(vec->getType())) {
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
    auto scalar_t = scalars[0]->getType();

    // Create the corresponding vector type.
    auto vector_t = make_vector_type(scalar_t, boost::numeric_cast<std::uint32_t>(vector_size));
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

// Helper to invoke an intrinsic function with arguments 'args'. 'types' are the argument type(s) for
// overloaded intrinsics.
llvm::CallInst *llvm_invoke_intrinsic(ir_builder &builder, const std::string &name,
                                      const std::vector<llvm::Type *> &types, const std::vector<llvm::Value *> &args)
{
    // Fetch the intrinsic ID from the name.
    const auto intrinsic_ID = llvm::Function::lookupIntrinsicID(name);
    if (intrinsic_ID == 0) {
        throw std::invalid_argument(fmt::format("Cannot fetch the ID of the intrinsic '{}'", name));
    }

    // Fetch the declaration.
    // NOTE: for generic intrinsics to work, we need to specify
    // the desired argument type(s). See:
    // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
    // And the docs of the getDeclaration() function.
    assert(builder.GetInsertBlock() != nullptr); // LCOV_EXCL_LINE
    auto *callee_f = llvm::Intrinsic::getDeclaration(builder.GetInsertBlock()->getModule(), intrinsic_ID, types);
    if (callee_f == nullptr) {
        throw std::invalid_argument(fmt::format("Error getting the declaration of the intrinsic '{}'", name));
    }
    if (!callee_f->isDeclaration()) {
        // It does not make sense to have a definition of a builtin.
        throw std::invalid_argument(fmt::format("The intrinsic '{}' must be only declared, not defined", name));
    }

    // Check the number of arguments.
    if (callee_f->arg_size() != args.size()) {
        throw std::invalid_argument(
            fmt::format("Incorrect # of arguments passed while calling the intrinsic '{}': {} are "
                        "expected, but {} were provided instead",
                        name, callee_f->arg_size(), args.size()));
    }

    // Create the function call.
    auto *r = builder.CreateCall(callee_f, args);
    assert(r != nullptr);

    return r;
}

// Helper to invoke an external function called 'name' with arguments args and return type ret_type.
llvm::CallInst *llvm_invoke_external(llvm_state &s, const std::string &name, llvm::Type *ret_type,
                                     const std::vector<llvm::Value *> &args, const std::vector<int> &attrs)
{
    // Look up the name in the global module table.
    auto *callee_f = s.module().getFunction(name);

    if (callee_f == nullptr) {
        // The function does not exist yet, make the prototype.
        std::vector<llvm::Type *> arg_types;
        for (auto *a : args) {
            arg_types.push_back(a->getType());
        }
        auto *ft = llvm::FunctionType::get(ret_type, arg_types, false);
        callee_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
        if (callee_f == nullptr) {
            throw std::invalid_argument(
                fmt::format("Unable to create the prototype for the external function '{}'", name));
        }

        // Add the function attributes.
        for (const auto &att : attrs) {
            // NOTE: convert back to the LLVM attribute enum.
            callee_f->addFnAttr(boost::numeric_cast<llvm::Attribute::AttrKind>(att));
        }
    } else {
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
        // NOTE: perhaps in the future we should consider adding more checks here
        // (e.g., argument types, return type).
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
    auto f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Pre-create loop and afterloop blocks. Note that these have just
    // been created, they have not been inserted yet in the IR.
    auto *loop_bb = llvm::BasicBlock::Create(context);
    auto *after_bb = llvm::BasicBlock::Create(context);

    // NOTE: we need a special case if the body of the loop is
    // never to be executed (that is, begin >= end).
    // In such a case, we will jump directly to after_bb.
    // NOTE: unsigned integral comparison.
    auto skip_cond = builder.CreateICmp(llvm::CmpInst::ICMP_UGE, begin, end);
    builder.CreateCondBr(skip_cond, after_bb, loop_bb);

    // Get a reference to the current block for
    // later usage in the phi node.
    auto preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    f->getBasicBlockList().push_back(loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto cur = builder.CreatePHI(builder.getInt32Ty(), 2);
    cur->addIncoming(begin, preheader_bb);

    // Execute the loop body and the post-body code.
    llvm::Value *next;
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
    auto end_cond = builder.CreateICmp(llvm::CmpInst::ICMP_ULT, next, end);

    // Get a reference to the current block for later use,
    // and insert the "after loop" block.
    auto loop_end_bb = builder.GetInsertBlock();
    f->getBasicBlockList().push_back(after_bb);

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

    return ostr.str();
}

// This function will return true if:
//
// - the return type of f is ret, and
// - the argument types of f are the same as in 'args'.
//
// Otherwise, the function will return false.
bool compare_function_signature(llvm::Function *f, llvm::Type *ret, const std::vector<llvm::Type *> &args)
{
    assert(f != nullptr);
    assert(ret != nullptr);

    if (ret != f->getReturnType()) {
        // Mismatched return types.
        return false;
    }

    auto it = f->arg_begin();
    for (auto arg_type : args) {
        if (it == f->arg_end() || it->getType() != arg_type) {
            // f has fewer arguments than args, or the current
            // arguments' types do not match.
            return false;
        }
        ++it;
    }

    // In order for the signatures to match,
    // we must be at the end of f's arguments list
    // (otherwise f has more arguments than args).
    return it == f->arg_end();
}

// Create an LLVM if statement in the form:
// if (cond) {
//   then_f();
// } else {
//   else_f();
// }
void llvm_if_then_else(llvm_state &s, llvm::Value *cond, const std::function<void()> &then_f,
                       const std::function<void()> &else_f)
{
    auto &context = s.context();
    auto &builder = s.builder();

    assert(cond->getType() == builder.getInt1Ty());

    // Fetch the current function.
    assert(builder.GetInsertBlock() != nullptr);
    auto f = builder.GetInsertBlock()->getParent();
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
    f->getBasicBlockList().push_back(else_bb);
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
    f->getBasicBlockList().push_back(merge_bb);
    builder.SetInsertPoint(merge_bb);
}

// Helper to invoke an external function with vector arguments.
// The call will be decomposed into a sequence of calls with scalar arguments,
// and the return values will be re-assembled as a vector.
// NOTE: there are some assumptions about valid function attributes
// in this implementation, need to keep these into account when using
// this helper.
llvm::Value *call_extern_vec(llvm_state &s, const std::vector<llvm::Value *> &args, const std::string &fname)
{
    // LCOV_EXCL_START
    assert(!args.empty());
    // Make sure all vector arguments are of the same type.
    assert(std::all_of(args.begin() + 1, args.end(),
                       [&args](const auto &arg) { return arg->getType() == args[0]->getType(); }));
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Decompose each argument into a vector of scalars.
    std::vector<std::vector<llvm::Value *>> scalars;
    for (const auto &arg : args) {
        scalars.push_back(vector_to_scalars(builder, arg));
    }

    // Fetch the vector size.
    auto vec_size = scalars[0].size();

    // Fetch the type of the scalar arguments.
    const auto scal_t = scalars[0][0]->getType();

    // LCOV_EXCL_START
    // Make sure the vector size is the same for all arguments.
    assert(std::all_of(scalars.begin() + 1, scalars.end(),
                       [vec_size](const auto &arg) { return arg.size() == vec_size; }));
    // LCOV_EXCL_STOP

    // Invoke the function on each set of scalars.
    std::vector<llvm::Value *> retvals, scal_args;
    for (decltype(vec_size) i = 0; i < vec_size; ++i) {
        // Setup the vector of scalar arguments.
        scal_args.clear();
        for (const auto &scal_set : scalars) {
            scal_args.push_back(scal_set[i]);
        }

        // Invoke the function and store the scalar result.
        retvals.push_back(llvm_invoke_external(
            s, fname, scal_t, scal_args,
            // NOTE: in theory we may add ReadNone here as well,
            // but for some reason, at least up to LLVM 10,
            // this causes strange codegen issues. Revisit
            // in the future.
            {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
    }

    // Build a vector with the results.
    return scalars_to_vector(builder, retvals);
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
    auto f = builder.GetInsertBlock()->getParent();
    assert(f != nullptr);

    // Do a first evaluation of cond.
    // NOTE: if this throws, we have not created any block
    // yet, no need for manual cleanup.
    auto cmp = cond();
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
    auto preheader_bb = builder.GetInsertBlock();

    // Add the loop block and start insertion into it.
    f->getBasicBlockList().push_back(loop_bb);
    builder.SetInsertPoint(loop_bb);

    // Create the phi node and add the first pair of arguments.
    auto cur = builder.CreatePHI(builder.getInt1Ty(), 2);
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
    auto loop_end_bb = builder.GetInsertBlock();
    f->getBasicBlockList().push_back(after_bb);

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
        auto *f = real_nary_op(s, fp_t, "fadd", "mpfr_add", 2u);

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
        auto *f = real_nary_op(s, fp_t, "fsub", "mpfr_sub", 2u);

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
        auto *f = real_nary_op(s, fp_t, "fmul", "mpfr_mul", 2u);

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
        auto *f = real_nary_op(s, fp_t, "fdiv", "mpfr_div", 2u);

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

// Helper to compute sin and cos simultaneously.
std::pair<llvm::Value *, llvm::Value *> llvm_sincos(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();
    auto &builder = s.builder();

    // Determine the scalar type of the vector arguments.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        if (auto vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            // NOTE: although there exists a SLEEF function for computing sin/cos
            // at the same time, we cannot use it directly because it returns a pair
            // of SIMD vectors rather than a single one and that does not play
            // well with the calling conventions. In theory we could write a wrapper
            // for these sincos functions using pointers for output values,
            // but compiling such a wrapper requires correctly
            // setting up the SIMD compilation flags. Perhaps we can consider this in the
            // future to improve performance.
            const auto sfn_sin = sleef_function_name(context, "sin", vec_t->getElementType(),
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
            const auto sfn_cos = sleef_function_name(context, "cos", vec_t->getElementType(),
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));

            if (!sfn_sin.empty() && !sfn_cos.empty()) {
                auto ret_sin = llvm_invoke_external(
                    s, sfn_sin, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});

                auto ret_cos = llvm_invoke_external(
                    s, sfn_cos, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});

                return {ret_sin, ret_cos};
            }
        }

        // Compute sin and cos via intrinsics.
        auto *sin_x = llvm_invoke_intrinsic(builder, "llvm.sin", {x->getType()}, {x});
        auto *cos_x = llvm_invoke_intrinsic(builder, "llvm.cos", {x->getType()}, {x});

        return {sin_x, cos_x};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        // NOTE: for __float128 we cannot use the intrinsics, we need
        // to call an external function.

        // Convert the vector argument to scalars.
        auto x_scalars = vector_to_scalars(builder, x);

        // Execute the sincosq() function on the scalar values and store
        // the results in res_scalars.
        // NOTE: need temp storage because sincosq uses pointers
        // for output values.
        auto s_all = builder.CreateAlloca(x_t);
        auto c_all = builder.CreateAlloca(x_t);
        std::vector<llvm::Value *> res_sin, res_cos;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            llvm_invoke_external(s, "sincosq", builder.getVoidTy(), {x_scalars[i], s_all, c_all},
                                 {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});

            res_sin.emplace_back(builder.CreateLoad(x_t, s_all));
            res_cos.emplace_back(builder.CreateLoad(x_t, c_all));
        }

        // Reconstruct the return value as a vector.
        return {scalars_to_vector(builder, res_sin), scalars_to_vector(builder, res_cos)};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        return llvm_real_sincos(s, x);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sincos()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Helper to compute abs(x_v).
llvm::Value *llvm_abs(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t->isFloatingPointTy()) {
#if defined(HEYOKA_HAVE_REAL128)
        if (x_t == to_llvm_type<mppp::real128>(s.context(), false)) {
            return call_extern_vec(s, {x}, "fabsq");
        } else {
#endif
            return llvm_invoke_intrinsic(s.builder(), "llvm.fabs", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
        }
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "abs", "mpfr_abs", 1u);

        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to abs values of type '{}'", llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Helper to reduce x modulo y, that is, to compute:
// x - y * floor(x / y).
llvm::Value *llvm_modulus(llvm_state &s, llvm::Value *x, llvm::Value *y)
{
    auto &builder = s.builder();

#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto *x_t = x->getType()->getScalarType();

    auto &context = s.context();

    if (x_t == llvm::Type::getFP128Ty(context)) {
        return call_extern_vec(s, {x, y}, "heyoka_modulus128");
    } else {
#endif
        auto *quo = llvm_fdiv(s, x, y);
        auto *fl_quo = llvm_invoke_intrinsic(builder, "llvm.floor", {quo->getType()}, {quo});

        return llvm_fsub(s, x, llvm_fmul(s, y, fl_quo));
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
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
        llvm::Type *int_type;
        if (auto *v_t = llvm::dyn_cast<llvm_vector_type>(cmp0->getType())) {
            int_type
                = make_vector_type(builder.getInt32Ty(), boost::numeric_cast<std::uint32_t>(v_t->getNumElements()));
        } else {
            int_type = builder.getInt32Ty();
        }
        auto *icmp0 = builder.CreateZExt(cmp0, int_type);
        auto *icmp1 = builder.CreateZExt(cmp1, int_type);

        // Compute and return the result.
        return builder.CreateSub(icmp0, icmp1);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(val->getType()) != 0) {
        return llvm_real_sgn(s, val);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sgn()",
                                                llvm_type_name(val->getType())));
        // LCOV_EXCL_STOP
    }
}

// Two-argument arctan.
llvm::Value *llvm_atan2(llvm_state &s, llvm::Value *y, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(y != nullptr);
    assert(x != nullptr);
    assert(y->getType() == x->getType());
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the arguments.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "atan2", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {y, x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {y, x}, "atan2");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {y, x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an atan2l function,
                               // because LLVM complains about the symbol "atan2l" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_atan2l"
#else
                               "atan2l"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {y, x}, "atan2q");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "atan2", "mpfr_atan2", 2u);
        return s.builder().CreateCall(f, {y, x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of atan2()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Exponential.
llvm::Value *llvm_exp(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn
                = sleef_function_name(context, "exp", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s.builder(), "llvm.exp", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "expq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "exp", "mpfr_exp", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of exp()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Fused multiply-add.
llvm::Value *llvm_fma(llvm_state &s, llvm::Value *x, llvm::Value *y, llvm::Value *z)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    assert(y != nullptr);
    assert(z != nullptr);
    assert(x->getType() == y->getType());
    assert(x->getType() == z->getType());
    // LCOV_EXCL_STOP

    auto *x_t = x->getType()->getScalarType();

    if (x_t->isFloatingPointTy()) {
#if defined(HEYOKA_HAVE_REAL128)
        if (x_t == to_llvm_type<mppp::real128>(s.context(), false)) {
            return call_extern_vec(s, {x, y, z}, "fmaq");
        } else {
#endif
            return llvm_invoke_intrinsic(s.builder(), "llvm.fma", {x->getType()}, {x, y, z});
#if defined(HEYOKA_HAVE_REAL128)
        }
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "fma", "mpfr_fma", 3u);
        return s.builder().CreateCall(f, {x, y, z});
#endif
        // LCOV_EXCL_START
    } else {
        throw std::invalid_argument(fmt::format("Unable to fma values of type '{}'", llvm_type_name(x->getType())));
    }
    // LCOV_EXCL_STOP
}

// Floor.
llvm::Value *llvm_floor(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t->isFloatingPointTy()) {
#if defined(HEYOKA_HAVE_REAL128)
        if (x_t == to_llvm_type<mppp::real128>(s.context(), false)) {
            return call_extern_vec(s, {x}, "floorq");
        } else {
#endif
            return llvm_invoke_intrinsic(s.builder(), "llvm.floor", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
        }
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "floor", "mpfr_floor", 1u);

        return s.builder().CreateCall(f, {x});
#endif
        // LCOV_EXCL_START
    } else {
        throw std::invalid_argument(fmt::format("Unable to floor values of type '{}'", llvm_type_name(x->getType())));
    }
    // LCOV_EXCL_STOP
}

// Add a function to count the number of sign changes in the coefficients
// of a polynomial of degree n. The coefficients are SIMD vectors of size batch_size
// and scalar type T.
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

    // Fetch the vector floating-point type.
    auto *tp = make_vector_type(scal_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka_csc_degree_{}_{}", n, llvm_mangle_type(tp));

    // The function arguments:
    // - pointer to the return value,
    // - pointer to the array of coefficients.
    // NOTE: both pointers are to the scalar counterparts
    // of the vector types, so that we can call this from regular
    // C++ code.
    const std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(builder.getInt32Ty()),
                                          llvm::PointerType::getUnqual(scal_t)};

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
            auto *cur_cf = load_vector_from_memory(
                builder, scal_t,
                builder.CreateInBoundsGEP(scal_t, cf_ptr, builder.CreateMul(cur_n, builder.getInt32(batch_size))),
                batch_size);

            // Load the last nonzero coefficient(s).
            auto *last_nz_ptr_idx = builder.CreateAdd(
                offset, builder.CreateMul(builder.CreateLoad(last_nz_idx_t, last_nz_idx),
                                          vector_splat(builder, builder.getInt32(batch_size), batch_size)));
            auto *last_nz_ptr = builder.CreateInBoundsGEP(scal_t, cf_ptr_v, last_nz_ptr_idx);
            auto *last_nz_cf = gather_vector_from_memory(builder, cur_cf->getType(), last_nz_ptr);

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

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        if (!compare_function_signature(f, builder.getVoidTy(), fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the sign changes counter function detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

// Compute the enclosure of the polynomial of order n with coefficients stored in cf_ptr
// over the interval [h_lo, h_hi] using interval arithmetics. The polynomial coefficients
// are vectors of size batch_size and scalar type T.
// NOTE: the interval arithmetic implementation here is not 100% correct, because
// we do not account for floating-point truncation. In order to be mathematically
// correct, we would need to adjust the results of interval arithmetic add/mul via
// a std::nextafter()-like function. See here for an example:
// https://stackoverflow.com/questions/10420848/how-do-you-get-the-next-value-in-the-floating-point-sequence
// http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node46.html
// Perhaps another alternative would be to employ FP primitives with explicit rounding modes,
// which are available in LLVM.
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
    auto *ho_cf = load_vector_from_memory(
        builder, fp_t,
        builder.CreateInBoundsGEP(fp_t, cf_ptr, builder.CreateMul(builder.getInt32(n), builder.getInt32(batch_size))),
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
            fp_t, cf_ptr, builder.CreateMul(builder.CreateSub(builder.getInt32(n), i), builder.getInt32(batch_size)));
        auto *cur_cf = load_vector_from_memory(builder, fp_t, ptr, batch_size);

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
// are vectors of size batch_size and scalar type T. The interval of the independent variable
// is [0, h] if h >= 0, [h, 0] otherwise.
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

    // bj_series will contain the terms of the series
    // for the computation of bj. old_bj_series will be
    // used to deal with the fact that the pairwise sum
    // consumes the input vector.
    std::vector<llvm::Value *> bj_series, old_bj_series;

    // Init the current power of h with h itself.
    auto *cur_h_pow = h;

    // Compute the first value, b0, and add it to bj_series.
    auto *b0 = load_vector_from_memory(builder, fp_t, cf_ptr, batch_size);
    bj_series.push_back(b0);

    // Init min/max bj with b0.
    auto *min_bj = b0, *max_bj = b0;

    // Main iteration.
    for (std::uint32_t j = 1u; j <= n; ++j) {
        // Compute the new term of the series.
        auto *ptr = builder.CreateInBoundsGEP(fp_t, cf_ptr, builder.getInt32(j * batch_size));
        auto *cur_cf = load_vector_from_memory(builder, fp_t, ptr, batch_size);
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

namespace
{

#if !defined(NDEBUG)

// Small helper to compute pi at the precision of the floating-point type tp.
// Used only for debugging purposes.
// NOTE: this is a repetition of number_like(), perhaps we can abstract this?
number inv_kep_E_pi_like(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);

    auto &context = s.context();

    if (tp == to_llvm_type<float>(context, false)) {
        return number{boost::math::constants::pi<float>()};
    } else if (tp == to_llvm_type<double>(context, false)) {
        return number{boost::math::constants::pi<double>()};
    } else if (tp == to_llvm_type<long double>(context, false)) {
        return number{boost::math::constants::pi<long double>()};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (tp == to_llvm_type<mppp::real128>(context, false)) {
        return number{mppp::pi_128};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto prec = llvm_is_real(tp)) {
        return number{mppp::real_pi(prec)};
#endif
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Unable to create a number of type '{}' from the constant pi", llvm_type_name(tp)));
    // LCOV_EXCL_STOP
}

#endif

std::pair<number, number> inv_kep_E_dl_twopi_like(llvm_state &s, llvm::Type *fp_t)
{
    if (fp_t->isFloatingPointTy() &&
#if LLVM_VERSION_MAJOR >= 13
        fp_t->isIEEE()
#else
        !fp_t->isPPC_FP128Ty()
#endif
    ) {
        namespace bmp = boost::multiprecision;

        // NOTE: we will be generating the pi constant at 4x the maximum precision possible,
        // which is currently 113 bits for quadruple precision.
        constexpr auto ndigits = 113u * 4u;

#if !defined(NDEBUG)
        // Sanity check.
        const auto &sem = fp_t->getFltSemantics();
        const auto prec = llvm::APFloatBase::semanticsPrecision(sem);

        assert(prec <= 113u);
#endif

        // Fetch 2pi in extended precision.
        using mp_fp_t = bmp::number<bmp::cpp_bin_float<ndigits, bmp::digit_base_2>>;
        const auto mp_twopi = 2 * boost::math::constants::pi<mp_fp_t>();

        auto impl = [&](auto val) {
            using type = decltype(val);

#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (std::is_same_v<type, mppp::real128>) {
                using bmp_float128 = bmp::cpp_bin_float_quad;

                const auto twopi_hi = static_cast<bmp_float128>(mp_twopi);
                const auto twopi_lo = static_cast<bmp_float128>(mp_twopi - mp_fp_t(twopi_hi));

                assert(twopi_hi + twopi_lo == twopi_hi); // LCOV_EXCL_LINE

                return std::make_pair(number{mppp::real128{boost::lexical_cast<std::string>(twopi_hi)}},
                                      number{mppp::real128{boost::lexical_cast<std::string>(twopi_lo)}});
            } else {
#endif
                const auto twopi_hi = static_cast<type>(mp_twopi);
                const auto twopi_lo = static_cast<type>(mp_twopi - twopi_hi);

                assert(twopi_hi + twopi_lo == twopi_hi); // LCOV_EXCL_LINE

                return std::make_pair(number{twopi_hi}, number{twopi_lo});
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif
        };

        auto &context = s.context();

        if (fp_t == to_llvm_type<float>(context, false)) {
            return impl(0.f);
        } else if (fp_t == to_llvm_type<double>(context, false)) {
            return impl(0.);
        } else if (fp_t == to_llvm_type<long double>(context, false)) {
            return impl(0.l);
#if defined(HEYOKA_HAVE_REAL128)
        } else if (fp_t == to_llvm_type<mppp::real128>(context, false)) {
            return impl(mppp::real128(0));
#endif
        }

        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Cannot generate a double-length 2*pi approximation for the type '{}'",
                                                detail::llvm_type_name(fp_t)));
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto prec = llvm_is_real(fp_t)) {
        // Overflow check.
        // LCOV_EXCL_START
        if (prec > std::numeric_limits<real_prec_t>::max() / 4) {
            throw std::overflow_error("Overflow detected in inv_kep_E_dl_twopi_like()");
        }
        // LCOV_EXCL_STOP

        // Generate the 2*pi constant with prec * 4 precision.
        auto twopi = mppp::real_pi(prec * 4);
        mppp::mul_2ui(twopi, twopi, 1ul);

        // Fetch the hi/lo components in precision prec.
        auto twopi_hi = mppp::real{twopi, prec};
        auto twopi_lo = mppp::real{std::move(twopi) - twopi_hi, prec};

        assert(twopi_hi + twopi_lo == twopi_hi); // LCOV_EXCL_LINE

        return std::make_pair(number(std::move(twopi_hi)), number(std::move(twopi_lo)));
#endif
    } else {
        throw std::invalid_argument(fmt::format("Cannot generate a double-length 2*pi approximation for the type '{}'",
                                                detail::llvm_type_name(fp_t)));
    }
    // LCOV_EXCL_STOP
}

// Small helper to return the epsilon of the floating-point type tp as a number.
// NOTE: this is a repetition of number_like(), perhaps we can abstract this?
number inv_kep_E_eps_like(llvm_state &s, llvm::Type *tp)
{
    assert(tp != nullptr);

    auto &context = s.context();

    if (tp == to_llvm_type<float>(context, false)) {
        return number{std::numeric_limits<float>::epsilon()};
    } else if (tp == to_llvm_type<double>(context, false)) {
        return number{std::numeric_limits<double>::epsilon()};
    } else if (tp == to_llvm_type<long double>(context, false)) {
        return number{std::numeric_limits<long double>::epsilon()};
#if defined(HEYOKA_HAVE_REAL128)
    } else if (tp == to_llvm_type<mppp::real128>(context, false)) {
        return number{std::numeric_limits<mppp::real128>::epsilon()};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (const auto prec = llvm_is_real(tp)) {
        // NOTE: for consistency with the epsilons returned for the other
        // types, we return here 2**-(prec - 1). See:
        // https://en.wikipedia.org/wiki/Machine_epsilon
        return number(mppp::real{1ul, boost::numeric_cast<mpfr_exp_t>(-(prec - 1)), prec});
#endif
    }

    // LCOV_EXCL_START
    throw std::invalid_argument(
        fmt::format("Unable to create a number version of the epsilon for the type '{}'", llvm_type_name(tp)));
    // LCOV_EXCL_STOP
}

} // namespace

// Implementation of the inverse Kepler equation.
llvm::Function *llvm_add_inv_kep_E(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch vector floating-point type.
    auto *tp = make_vector_type(fp_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka.inv_kep_E.{}", llvm_mangle_type(tp));

    // The function arguments:
    // - eccentricity,
    // - mean anomaly.
    const std::vector<llvm::Type *> fargs{tp, tp};

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is tp.
        auto *ft = llvm::FunctionType::get(tp, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ecc_arg = f->args().begin();
        auto M_arg = f->args().begin() + 1;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Is the eccentricity a quiet NaN or less than 0?
        auto *ecc_is_nan_or_neg = llvm_fcmp_ult(s, ecc_arg, llvm_constantfp(s, tp, 0.));
        // Is the eccentricity >= 1?
        auto *ecc_is_gte1 = llvm_fcmp_oge(s, ecc_arg, llvm_constantfp(s, tp, 1.));

        // Is the eccentricity NaN or out of range?
        // NOTE: this is a logical OR.
        auto *ecc_invalid = builder.CreateSelect(
            ecc_is_nan_or_neg, llvm::ConstantInt::getAllOnesValue(ecc_is_nan_or_neg->getType()), ecc_is_gte1);

        // Replace invalid eccentricity values with quiet NaNs.
        auto *ecc = builder.CreateSelect(ecc_invalid, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()),
                                         ecc_arg);

        // Create the return value.
        auto *retval = builder.CreateAlloca(tp);

        // Fetch 2pi in double-length precision.
        const auto [dl_twopi_hi, dl_twopi_lo] = inv_kep_E_dl_twopi_like(s, fp_t);

#if !defined(NDEBUG)
        assert(dl_twopi_hi == number_like(s, fp_t, 2.) * inv_kep_E_pi_like(s, fp_t)); // LCOV_EXCL_LINE
#endif

        // Reduce M modulo 2*pi in extended precision.
        auto *M = llvm_dl_modulus(s, M_arg, llvm_constantfp(s, tp, 0.),
                                  vector_splat(builder, llvm_codegen(s, fp_t, dl_twopi_hi), batch_size),
                                  vector_splat(builder, llvm_codegen(s, fp_t, dl_twopi_lo), batch_size))
                      .first;

        // Compute the initial guess from the usual elliptic expansion
        // to the third order in eccentricities:
        // E = M + e*sin(M) + e**2*sin(M)*cos(M) + e**3*sin(M)*(3/2*cos(M)**2 - 1/2) + ...
        auto [sin_M, cos_M] = llvm_sincos(s, M);
        // e*sin(M).
        auto *e_sin_M = llvm_fmul(s, ecc, sin_M);
        // e*cos(M).
        auto *e_cos_M = llvm_fmul(s, ecc, cos_M);
        // e**2.
        auto *e2 = llvm_fmul(s, ecc, ecc);
        // cos(M)**2.
        auto *cos_M_2 = llvm_fmul(s, cos_M, cos_M);

        // 3/2 and 1/2 constants.
        auto *c_3_2 = vector_splat(builder, llvm_codegen(s, fp_t, number_like(s, fp_t, 3.) / number_like(s, fp_t, 2.)),
                                   batch_size);
        auto *c_1_2 = vector_splat(builder, llvm_codegen(s, fp_t, number_like(s, fp_t, 1.) / number_like(s, fp_t, 2.)),
                                   batch_size);

        // M + e*sin(M).
        auto *tmp1 = llvm_fadd(s, M, e_sin_M);
        // e**2*sin(M)*cos(M).
        auto *tmp2 = llvm_fmul(s, e_sin_M, e_cos_M);
        // e**3*sin(M).
        auto *tmp3 = llvm_fmul(s, e2, e_sin_M);
        // 3/2*cos(M)**2 - 1/2.
        auto *tmp4 = llvm_fsub(s, llvm_fmul(s, c_3_2, cos_M_2), c_1_2);

        // Put it together.
        auto *ig1 = llvm_fadd(s, tmp1, tmp2);
        auto *ig2 = llvm_fmul(s, tmp3, tmp4);
        auto *ig = llvm_fadd(s, ig1, ig2);

        // Make extra sure the initial guess is in the [0, 2*pi) range.
        auto *lb = llvm_constantfp(s, tp, 0.);
        auto *ub = vector_splat(builder, llvm_codegen(s, fp_t, nextafter(dl_twopi_hi, number_like(s, fp_t, 0.))),
                                batch_size);
        // NOTE: perhaps a dedicated clamp() primitive could give better
        // performance for real?
        ig = llvm_max(s, ig, lb);
        ig = llvm_min(s, ig, ub);

        // Store it.
        builder.CreateStore(ig, retval);

        // Create the counter.
        auto *counter = builder.CreateAlloca(builder.getInt32Ty());
        builder.CreateStore(builder.getInt32(0), counter);

        // Variables to store sin(E) and cos(E).
        auto *sin_E = builder.CreateAlloca(tp);
        auto *cos_E = builder.CreateAlloca(tp);

        // Write the initial values for sin_E and cos_E.
        auto sin_cos_E = llvm_sincos(s, builder.CreateLoad(tp, retval));
        builder.CreateStore(sin_cos_E.first, sin_E);
        builder.CreateStore(sin_cos_E.second, cos_E);

        // Variable to hold the value of f(E) = E - e*sin(E) - M.
        auto *fE = builder.CreateAlloca(tp);
        // Helper to compute f(E).
        auto fE_compute = [&]() {
            auto ret = llvm_fmul(s, ecc, builder.CreateLoad(tp, sin_E));
            ret = llvm_fsub(s, builder.CreateLoad(tp, retval), ret);
            return llvm_fsub(s, ret, M);
        };
        // Compute and store the initial value of f(E).
        builder.CreateStore(fE_compute(), fE);

        // Create a variable to hold the result of the tolerance check
        // computed at the beginning of each iteration of the main loop.
        // This is "true" if the loop needs to continue, or "false" if the
        // loop can stop because we achieved the desired tolerance.
        // NOTE: this is only allocated, it will be immediately written to
        // the first time loop_cond is evaluated.
        auto *vec_bool_t = make_vector_type(builder.getInt1Ty(), batch_size);
        auto *tol_check_ptr = builder.CreateAlloca(vec_bool_t);

        // Define the stopping condition functor.
        // NOTE: hard-code this for the time being.
        auto *max_iter = builder.getInt32(50);
        auto loop_cond =
            [&,
             // NOTE: tolerance is 4 * eps.
             tol = vector_splat(builder, llvm_codegen(s, fp_t, inv_kep_E_eps_like(s, fp_t) * number_like(s, fp_t, 4.)),
                                batch_size)]() -> llvm::Value * {
            auto *c_cond = builder.CreateICmpULT(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter);

            // Keep on iterating as long as abs(f(E)) > tol.
            // NOTE: need reduction only in batch mode.
            auto *tol_check = llvm_fcmp_ogt(s, llvm_abs(s, builder.CreateLoad(tp, fE)), tol);
            auto *tol_cond = (batch_size == 1u) ? tol_check : builder.CreateOrReduce(tol_check);

            // Store the result of the tolerance check.
            builder.CreateStore(tol_check, tol_check_ptr);

            // NOTE: this is a way of creating a logical AND.
            return builder.CreateSelect(c_cond, tol_cond, llvm::ConstantInt::get(tol_cond->getType(), 0u));
        };

        // Run the loop.
        llvm_while_loop(s, loop_cond, [&, one_c = llvm_constantfp(s, tp, 1.)]() {
            // Compute the new value.
            auto *old_val = builder.CreateLoad(tp, retval);
            auto *new_val = llvm_fdiv(s, builder.CreateLoad(tp, fE),
                                      llvm_fsub(s, one_c, llvm_fmul(s, ecc, builder.CreateLoad(tp, cos_E))));
            new_val = llvm_fsub(s, old_val, new_val);

            // Bisect if new_val > ub.
            // NOTE: '>' is fine here, ub is the maximum allowed value.
            auto *bcheck = llvm_fcmp_ogt(s, new_val, ub);
            new_val = builder.CreateSelect(
                bcheck,
                llvm_fmul(s,
                          vector_splat(builder,
                                       llvm_codegen(s, fp_t, number_like(s, fp_t, 1.) / number_like(s, fp_t, 2.)),
                                       batch_size),
                          llvm_fadd(s, old_val, ub)),
                new_val);

            // Bisect if new_val < lb.
            bcheck = llvm_fcmp_olt(s, new_val, lb);
            new_val = builder.CreateSelect(
                bcheck,
                llvm_fmul(s,
                          vector_splat(builder,
                                       llvm_codegen(s, fp_t, number_like(s, fp_t, 1.) / number_like(s, fp_t, 2.)),
                                       batch_size),
                          llvm_fadd(s, old_val, lb)),
                new_val);

            // Store the new value.
            builder.CreateStore(new_val, retval);

            // Update sin_E/cos_E.
            sin_cos_E = llvm_sincos(s, new_val);
            builder.CreateStore(sin_cos_E.first, sin_E);
            builder.CreateStore(sin_cos_E.second, cos_E);

            // Update f(E).
            builder.CreateStore(fE_compute(), fE);

            // Update the counter.
            builder.CreateStore(
                builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), counter), builder.getInt32(1)), counter);
        });

        // Check the counter.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(builder.CreateLoad(builder.getInt32Ty(), counter), max_iter),
            [&]() {
                // Load the tol_check variable.
                auto *tol_check = builder.CreateLoad(vec_bool_t, tol_check_ptr);

                // Set to quiet NaN in the return value all the lanes for which tol_check is 1.
                auto *old_val = builder.CreateLoad(tp, retval);
                auto *new_val = builder.CreateSelect(
                    tol_check, llvm_constantfp(s, tp, std::numeric_limits<double>::quiet_NaN()), old_val);
                builder.CreateStore(new_val, retval);

                llvm_invoke_external(s, "heyoka_inv_kep_E_max_iter", builder.getVoidTy(), {},
                                     {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});
            },
            []() {});

        // Return the result.
        builder.CreateRet(builder.CreateLoad(tp, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        if (!compare_function_signature(f, tp, fargs)) {
            throw std::invalid_argument("Inconsistent function signature for the inverse Kepler equation detected");
        }
    }

    return f;
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
                                                    llvm::GlobalVariable::InternalLinkage, bc_const_arr);

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
    ir_builder &m_builder;
    llvm::FastMathFlags m_orig_fmf;

public:
    explicit fmf_disabler(ir_builder &b) : m_builder(b), m_orig_fmf(m_builder.getFastMathFlags())
    {
        // Reset the fast math flags.
        m_builder.setFastMathFlags(llvm::FastMathFlags{});
    }
    ~fmf_disabler()
    {
        // Restore the original fast math flags.
        m_builder.setFastMathFlags(m_orig_fmf);
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
    fmf_disabler fd(builder);

    auto x = llvm_fmul(s, a, b);
    auto y = llvm_fma(s, a, b, llvm_fneg(s, x));

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
    fmf_disabler fd(builder);

    auto S = llvm_fadd(state, x_hi, y_hi);
    auto T = llvm_fadd(state, x_lo, y_lo);
    auto e = llvm_fsub(state, S, x_hi);
    auto f = llvm_fsub(state, T, x_lo);

    auto t1 = llvm_fsub(state, S, e);
    t1 = llvm_fsub(state, x_hi, t1);
    auto s = llvm_fsub(state, y_hi, e);
    s = llvm_fadd(state, s, t1);

    t1 = llvm_fsub(state, T, f);
    t1 = llvm_fsub(state, x_lo, t1);
    auto t = llvm_fsub(state, y_lo, f);
    t = llvm_fadd(state, t, t1);

    s = llvm_fadd(state, s, T);
    auto H = llvm_fadd(state, S, s);
    auto h = llvm_fsub(state, S, H);
    h = llvm_fadd(state, h, s);

    h = llvm_fadd(state, h, t);
    e = llvm_fadd(state, H, h);
    f = llvm_fsub(state, H, e);
    f = llvm_fadd(state, f, h);

    return {e, f};
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
    fmf_disabler fd(builder);

    auto [c, cc] = llvm_eft_product(s, x_hi, y_hi);

    // cc = x*yy + xx*y + cc.
    auto x_yy = llvm_fmul(s, x_hi, y_lo);
    auto xx_y = llvm_fmul(s, x_lo, y_hi);
    cc = llvm_fadd(s, llvm_fadd(s, x_yy, xx_y), cc);

    // The normalisation step.
    auto z = llvm_fadd(s, c, cc);
    auto zz = llvm_fadd(s, llvm_fsub(s, c, z), cc);

    return {z, zz};
}

// Division.
// NOTE: this is procedure div2() from here:
// https://link.springer.com/content/pdf/10.1007/BF01397083.pdf
// The mul12() function is replaced with the FMA-based llvm_eft_product().
// NOTE: the code in NTL looks identical to Dekker's.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_div(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    fmf_disabler fd(builder);

    auto *c = llvm_fdiv(s, x_hi, y_hi);

    auto [u, uu] = llvm_eft_product(s, c, y_hi);

    // cc = (x_hi - u - uu + x_lo - c * y_lo) / y_hi.
    auto *cc = llvm_fsub(s, x_hi, u);
    cc = llvm_fsub(s, cc, uu);
    cc = llvm_fadd(s, cc, x_lo);
    cc = llvm_fsub(s, cc, llvm_fmul(s, c, y_lo));
    cc = llvm_fdiv(s, cc, y_hi);

    // The normalisation step.
    auto z = llvm_fadd(s, c, cc);
    auto zz = llvm_fadd(s, llvm_fsub(s, c, z), cc);

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

    auto fp_t = x_hi->getType();

    // Temporarily disable the fast math flags.
    fmf_disabler fd(builder);

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
                auto z = llvm_fadd(s, fhi, flo);
                auto zz = llvm_fadd(s, llvm_fsub(s, fhi, z), flo);

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
        auto z = llvm_fadd(s, fhi, ret_lo);
        auto zz = llvm_fadd(s, llvm_fsub(s, fhi, z), ret_lo);

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
    fmf_disabler fd(builder);

    auto [xoy_hi, xoy_lo] = llvm_dl_div(s, x_hi, x_lo, y_hi, y_lo);
    auto [fl_hi, fl_lo] = llvm_dl_floor(s, xoy_hi, xoy_lo);
    auto [prod_hi, prod_lo] = llvm_dl_mul(s, y_hi, y_lo, fl_hi, fl_lo);

    return llvm_dl_add(s, x_hi, x_lo, llvm_fneg(s, prod_hi), llvm_fneg(s, prod_lo));
}

// Less-than.
llvm::Value *llvm_dl_lt(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    fmf_disabler fd(builder);

    auto cond1 = llvm_fcmp_olt(state, x_hi, y_hi);
    auto cond2 = llvm_fcmp_oeq(state, x_hi, y_hi);
    auto cond3 = llvm_fcmp_olt(state, x_lo, y_lo);
    // NOTE: this is a logical AND.
    auto cond4 = builder.CreateSelect(cond2, cond3, llvm::ConstantInt::getNullValue(cond3->getType()));
    // NOTE: this is a logical OR.
    auto cond = builder.CreateSelect(cond1, llvm::ConstantInt::getAllOnesValue(cond4->getType()), cond4);

    return cond;
}

// Greater-than.
llvm::Value *llvm_dl_gt(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    fmf_disabler fd(builder);

    auto cond1 = llvm_fcmp_ogt(state, x_hi, y_hi);
    auto cond2 = llvm_fcmp_oeq(state, x_hi, y_hi);
    auto cond3 = llvm_fcmp_ogt(state, x_lo, y_lo);
    // NOTE: this is a logical AND.
    auto cond4 = builder.CreateSelect(cond2, cond3, llvm::ConstantInt::getNullValue(cond3->getType()));
    // NOTE: this is a logical OR.
    auto cond = builder.CreateSelect(cond1, llvm::ConstantInt::getAllOnesValue(cond4->getType()), cond4);

    return cond;
}

// Helper to create a wrapper for kepE() usable from C++ code.
// Input/output is done through pointers.
void llvm_add_inv_kep_E_wrapper(llvm_state &s, llvm::Type *scal_t, std::uint32_t batch_size, const std::string &name)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Make sure the function does not exist already.
    assert(md.getFunction(name) == nullptr); // LCOV_EXCL_LINE

    // Add the implementation function.
    auto *impl_f = llvm_add_inv_kep_E(s, scal_t, batch_size);

    // Fetch the external type.
    auto *ext_fp_t = llvm_ext_type(scal_t);

    // The function arguments:
    // - output pointer (write only),
    // - input ecc and mean anomaly pointers (read only).
    // No overlap allowed.
    std::vector<llvm::Type *> fargs(3u, llvm::PointerType::getUnqual(ext_fp_t));
    // The return type is void.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    // Create the function
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &md);
    assert(f != nullptr); // LCOV_EXCL_LINE

    // Fetch the current insertion block.
    auto orig_bb = builder.GetInsertBlock();

    // Setup the function arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto ecc_ptr = f->args().begin() + 1;
    ecc_ptr->setName("ecc_ptr");
    ecc_ptr->addAttr(llvm::Attribute::NoCapture);
    ecc_ptr->addAttr(llvm::Attribute::NoAlias);
    ecc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto M_ptr = f->args().begin() + 2;
    M_ptr->setName("M_ptr");
    M_ptr->addAttr(llvm::Attribute::NoCapture);
    M_ptr->addAttr(llvm::Attribute::NoAlias);
    M_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Load the data from the pointers.
    auto *ecc = ext_load_vector_from_memory(s, scal_t, ecc_ptr, batch_size);
    auto *M = ext_load_vector_from_memory(s, scal_t, M_ptr, batch_size);

    // Invoke the implementation function.
    auto *ret = builder.CreateCall(impl_f, {ecc, M});

    // Store the result.
    ext_store_vector_to_memory(s, out_ptr, ret);

    // Return.
    builder.CreateRetVoid();

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);
}

// Inverse cosine.
llvm::Value *llvm_acos(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "acos", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "acos");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an acos function,
                               // because LLVM complains about the symbol "acosl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_acosl"
#else
                               "acosl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "acosq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "acos", "mpfr_acos", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of acos()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Inverse hyperbolic cosine.
llvm::Value *llvm_acosh(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "acosh", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "acosh");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an acosh function,
                               // because LLVM complains about the symbol "acoshl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_acoshl"
#else
                               "acoshl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "acoshq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "acosh", "mpfr_acosh", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of acosh()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Inverse sine.
llvm::Value *llvm_asin(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "asin", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "asin");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an asin function,
                               // because LLVM complains about the symbol "asinl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_asinl"
#else
                               "asinl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "asinq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "asin", "mpfr_asin", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of asin()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Inverse hyperbolic sine.
llvm::Value *llvm_asinh(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "asinh", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "asinh");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an asinh function,
                               // because LLVM complains about the symbol "asinhl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_asinhl"
#else
                               "asinhl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "asinhq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "asinh", "mpfr_asinh", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of asinh()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Inverse tangent.
llvm::Value *llvm_atan(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "atan", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "atan");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an atan function,
                               // because LLVM complains about the symbol "atanl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_atanl"
#else
                               "atanl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "atanq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "atan", "mpfr_atan", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of atan()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Inverse hyperbolic tangent.
llvm::Value *llvm_atanh(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "atanh", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "atanh");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an atanh function,
                               // because LLVM complains about the symbol "atanhl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_atanhl"
#else
                               "atanhl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "atanhq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "atanh", "mpfr_atanh", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of atanh()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Cosine.
llvm::Value *llvm_cos(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn
                = sleef_function_name(context, "cos", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s.builder(), "llvm.cos", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "cosq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "cos", "mpfr_cos", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of cos()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Sine.
llvm::Value *llvm_sin(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn
                = sleef_function_name(context, "sin", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s.builder(), "llvm.sin", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "sinq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "sin", "mpfr_sin", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sin()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Hyperbolic cosine.
llvm::Value *llvm_cosh(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "cosh", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "cosh");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an cosh function,
                               // because LLVM complains about the symbol "coshl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_coshl"
#else
                               "coshl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "coshq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "cosh", "mpfr_cosh", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of cosh()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Error function.
llvm::Value *llvm_erf(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn
                = sleef_function_name(context, "erf", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "erf");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an erf function,
                               // because LLVM complains about the symbol "erfl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_erfl"
#else
                               "erfl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "erfq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "erf", "mpfr_erf", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of erf()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Natural logarithm.
llvm::Value *llvm_log(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn
                = sleef_function_name(context, "log", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s.builder(), "llvm.log", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "logq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "log", "mpfr_log", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of log()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
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
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "sinh", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "sinh");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an sinh function,
                               // because LLVM complains about the symbol "sinhl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_sinhl"
#else
                               "sinhl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "sinhq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "sinh", "mpfr_sinh", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sinh()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Square root.
llvm::Value *llvm_sqrt(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of x.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "sqrt", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s.builder(), "llvm.sqrt", {x->getType()}, {x});
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "sqrtq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "sqrt", "mpfr_sqrt", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sqrt()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
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
        auto *f = real_nary_op(s, x->getType(), "square", "mpfr_sqr", 1u);
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
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn
                = sleef_function_name(context, "tan", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "tan");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an tan function,
                               // because LLVM complains about the symbol "tanl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_tanl"
#else
                               "tanl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "tanq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "tan", "mpfr_tan", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of tan()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Hyperbolic tangent.
llvm::Value *llvm_tanh(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the argument.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false)) {
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType())) {
            if (const auto sfn = sleef_function_name(context, "tanh", x_t,
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return call_extern_vec(s, {x}, "tanh");
    } else if (x_t == to_llvm_type<long double>(context, false)) {
        return call_extern_vec(s, {x},
#if defined(_MSC_VER)
                               // NOTE: it seems like the MSVC stdlib does not have an tanh function,
                               // because LLVM complains about the symbol "tanhl" not being
                               // defined. Hence, use our own wrapper instead.
                               "heyoka_tanhl"
#else
                               "tanhl"
#endif
        );
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        return call_extern_vec(s, {x}, "tanhq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "tanh", "mpfr_tanh", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of tanh()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

// Exponentiation.
llvm::Value *llvm_pow(llvm_state &s, llvm::Value *x, llvm::Value *y, bool allow_approx)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    assert(y != nullptr);
    assert(x->getType() == y->getType());
    // LCOV_EXCL_STOP

    auto &context = s.context();

    // Determine the scalar type of the arguments.
    auto *x_t = x->getType()->getScalarType();

    if (x_t == to_llvm_type<double>(context, false) || x_t == to_llvm_type<long double>(context, false)) {
        // NOTE: we want to try the SLEEF route only if we are *not* allowing
        // an approximated implementation.
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x->getType()); !allow_approx && vec_t != nullptr) {
            if (const auto sfn
                = sleef_function_name(context, "pow", x_t, boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x, y},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        auto *ret = llvm_invoke_intrinsic(s.builder(), "llvm.pow", {x->getType()}, {x, y});

        if (allow_approx) {
            ret->setHasApproxFunc(true);
        }

        return ret;
#if defined(HEYOKA_HAVE_REAL128)
    } else if (x_t == to_llvm_type<mppp::real128>(context, false)) {
        // NOTE: in principle we can detect here if y is a (vector) constant,
        // e.g., -3/2, and in such case we could do something like replacing
        // powq with sqrtq + mul/div. However the accuracy implications of this
        // are not clear: we know that allowapprox for double precision does not have
        // catastrophic effects in the Brouwer's law test, but OTOH allow_approx perhaps
        // transforms a * b**(-3/2) into a / (b * sqrt(b)), but all we can do here is to
        // transform it into a * 1/(b * sqrt(b)) instead (as we don't have access to a from here),
        // which looks perhaps worse accuracy wise? It seems like we need to run some extensive
        // testing before taking these decisions, both from the point of view of performance
        // *and* accuracy.
        //
        // The same applies to the real implementation.
        return call_extern_vec(s, {x, y}, "powq");
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        // NOTE: there is a convenient mpfr_rec_sqrt() function which looks very handy
        // for possibly optimising the case of exponent == -3/2.
        auto *f = real_nary_op(s, x->getType(), "pow", "mpfr_pow", 2u);
        return s.builder().CreateCall(f, {x, y});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of pow()",
                                                llvm_type_name(x->getType())));
        // LCOV_EXCL_STOP
    }
}

template <typename T>
llvm::Type *llvm_type_like(llvm_state &s, [[maybe_unused]] const T &x)
{
    auto &c = s.context();

#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        const auto name = fmt::format("heyoka.real.{}", x.get_prec());

#if LLVM_VERSION_MAJOR >= 12
        if (auto *ptr = llvm::StructType::getTypeByName(c, name)) {
            return ptr;
        }
#else
        if (auto *ptr = s.module().getTypeByName(name)) {
            return ptr;
        }
#endif

        // Fetch the limb array type.
        auto *limb_arr_t
            = llvm::ArrayType::get(to_llvm_type<real_limb_t>(c), boost::numeric_cast<std::uint64_t>(x.get_nlimbs()));

        auto *ret
            = llvm::StructType::create({to_llvm_type<real_sign_t>(c), to_llvm_type<real_exp_t>(c), limb_arr_t}, name);

        assert(ret != nullptr);
#if LLVM_VERSION_MAJOR >= 12
        assert(llvm::StructType::getTypeByName(c, name) == ret);
#else
        assert(s.module().getTypeByName(name) == ret);
#endif

        return ret;
    } else {
#endif
        return to_llvm_type<T>(c);
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC llvm::Type *llvm_type_like<float>(llvm_state &, const float &);

template HEYOKA_DLL_PUBLIC llvm::Type *llvm_type_like<double>(llvm_state &, const double &);

template HEYOKA_DLL_PUBLIC llvm::Type *llvm_type_like<long double>(llvm_state &, const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC llvm::Type *llvm_type_like<mppp::real128>(llvm_state &, const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template HEYOKA_DLL_PUBLIC llvm::Type *llvm_type_like<mppp::real>(llvm_state &, const mppp::real &);

#endif

// Compute the LLVM data type to be used for loading external data
// into an LLVM variable of type fp_t.
llvm::Type *llvm_ext_type(llvm::Type *fp_t)
{
    if (fp_t->isFloatingPointTy()) {
        return fp_t;
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return to_llvm_type<mppp::real>(fp_t->getContext());
#endif
        // LCOV_EXCL_START
    } else {
        throw std::invalid_argument(
            fmt::format("Cannot compute the external type for the LLVM type '{}'", llvm_type_name(fp_t)));
    }
    // LCOV_EXCL_STOP
}

// Convert the input unsigned integral value n to the floating-point type fp_t.
// Vector types/values are not supported.
llvm::Value *llvm_ui_to_fp(llvm_state &s, llvm::Value *n, llvm::Type *fp_t)
{
    assert(n != nullptr);
    assert(fp_t != nullptr);
    assert(!n->getType()->isVectorTy());
    assert(!fp_t->isVectorTy());

    if (fp_t->isFloatingPointTy()) {
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

} // namespace heyoka::detail

// NOTE: this function will be called by the LLVM implementation
// of the inverse Kepler function when the maximum number of iterations
// is exceeded.
extern "C" HEYOKA_DLL_PUBLIC void heyoka_inv_kep_E_max_iter() noexcept
{
    heyoka::detail::get_logger()->warn("iteration limit exceeded while solving the elliptic inverse Kepler equation");
}

#if !defined(NDEBUG)

#if defined(HEYOKA_HAVE_REAL)

extern "C" HEYOKA_DLL_PUBLIC void heyoka_assert_real_match_precs_ext_load(heyoka::detail::real_prec_t p1,
                                                                          heyoka::detail::real_prec_t p2) noexcept
{
    assert(p1 == p2);
}

extern "C" HEYOKA_DLL_PUBLIC void heyoka_assert_real_match_precs_ext_store(heyoka::detail::real_prec_t p1,
                                                                           heyoka::detail::real_prec_t p2) noexcept
{
    assert(p1 == p2);
}

#endif

#endif

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
