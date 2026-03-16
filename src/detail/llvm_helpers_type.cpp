// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/llvm_state.hpp>

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
}

// The global type map to associate a C++ type to an LLVM type.
// NOLINTNEXTLINE(cert-err58-cpp,bugprone-throwing-static-initialization)
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

        auto *ret
            = llvm::StructType::create({to_external_llvm_type<mpfr_prec_t>(c), to_external_llvm_type<mpfr_sign_t>(c),
                                        to_external_llvm_type<mpfr_exp_t>(c), llvm::PointerType::getUnqual(c)},
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

} // namespace

// Implementation of the function to associate a C++ type to an LLVM type.
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

// Helper to fetch the internal llvm type corresponding to the C++ type T.
//
// The internal type is the type used in the JITted code to manipulate values which in C++ are of type T. It coincides
// with the external llvm type for all supported types apart for mppp::real, which has a representation in the JITted
// code different from its C++ representation.
//
// Because mppp::real's precision is a runtime property (i.e., not encoded in the type), if T is mppp::real then the
// precision must be passed as second argument to this function.
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

        // Assemble the type name.
        const auto name = fmt::format("heyoka.real.{}", prec);

        // Check if we have already defined the type in the current context.
        if (auto *ptr = llvm::StructType::getTypeByName(c, name)) {
            return ptr;
        }

        // Compute the required number of limbs.
        //
        // NOTE: this is a computation done in the implementation of mppp::real and reproduced here. We should consider
        // exposing this functionality in mp++.
        const auto nlimbs = boost::numeric_cast<std::uint64_t>((prec / GMP_NUMB_BITS)
                                                               + static_cast<int>((prec % GMP_NUMB_BITS) != 0));

        // Fetch the limb array type.
        auto *limb_arr_t = llvm::ArrayType::get(to_external_llvm_type<mp_limb_t>(c), nlimbs);

        // Define the real type.
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
        //
        // NOLINTNEXTLINE(readability-inconsistent-ifelse-braces)
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

// Helper to check if the input type is an IEEE-like floating-point type.
//
// NOTE: LLVM<=20 had an isIEEE() method for this, but it got slightly changed in LLVM 21 so that now it is called
// isIEEELikeFPTy() and it *excludes* 80-bit extended precision. For our internal use, we want to consider 80-bit
// extended precision as IEEE-like.
bool llvm_is_ieee_like_fp(llvm::Type *tp)
{
    assert(tp != nullptr);

#if LLVM_VERSION_MAJOR <= 20

    return tp->isFloatingPointTy() && tp->isIEEE();

#else

    return tp->isFloatingPointTy() && (tp->isIEEELikeFPTy() || tp->isX86_FP80Ty());

#endif
}

} // namespace detail

HEYOKA_END_NAMESPACE
