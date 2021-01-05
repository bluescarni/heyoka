// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#if defined(HEYOKA_WITH_SLEEF)

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <tuple>
#include <unordered_map>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <sleef.h>

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka::detail
{

namespace
{

// The key type used in the sleef maps. It consists
// of a function name (e.g., "pow", "cos", etc.) and
// a SIMD vector width.
using sleef_key_t = std::tuple<std::string, std::uint32_t>;

// Hasher for sleef_key_t.
struct sleef_key_hasher {
    std::size_t operator()(const sleef_key_t &k) const
    {
        auto retval = std::hash<std::string>{}(std::get<0>(k));
        retval += std::hash<std::uint32_t>{}(std::get<1>(k));

        return retval;
    }
};

// sleef map type.
using sleef_map_t = std::unordered_map<sleef_key_t, std::string, sleef_key_hasher>;

// Helper to construct the sleef map for the double-precision type.
auto make_sleef_map_dbl()
{
    const auto &features = get_target_features();

    sleef_map_t retval;

    // sin().
    if (features.avx512f) {
        retval[{"sin", 8}] = "Sleef_sind8_u10avx512f";
        retval[{"sin", 4}] = "Sleef_sind4_u10avx2";
        retval[{"sin", 2}] = "Sleef_sind2_u10avx2128";
    } else if (features.avx2) {
        retval[{"sin", 4}] = "Sleef_sind4_u10avx2";
        retval[{"sin", 2}] = "Sleef_sind2_u10avx2128";
    } else if (features.avx) {
        retval[{"sin", 4}] = "Sleef_sind4_u10avx";
        retval[{"sin", 2}] = "Sleef_sind2_u10sse4";
    } else if (features.sse2) {
        retval[{"sin", 2}] = "Sleef_sind2_u10sse2";
    }

    // cos().
    if (features.avx512f) {
        retval[{"cos", 8}] = "Sleef_cosd8_u10avx512f";
        retval[{"cos", 4}] = "Sleef_cosd4_u10avx2";
        retval[{"cos", 2}] = "Sleef_cosd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"cos", 4}] = "Sleef_cosd4_u10avx2";
        retval[{"cos", 2}] = "Sleef_cosd2_u10avx2128";
    } else if (features.avx) {
        retval[{"cos", 4}] = "Sleef_cosd4_u10avx";
        retval[{"cos", 2}] = "Sleef_cosd2_u10sse4";
    } else if (features.sse2) {
        retval[{"cos", 2}] = "Sleef_cosd2_u10sse2";
    }

    // log().
    if (features.avx512f) {
        retval[{"log", 8}] = "Sleef_logd8_u10avx512f";
        retval[{"log", 4}] = "Sleef_logd4_u10avx2";
        retval[{"log", 2}] = "Sleef_logd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"log", 4}] = "Sleef_logd4_u10avx2";
        retval[{"log", 2}] = "Sleef_logd2_u10avx2128";
    } else if (features.avx) {
        retval[{"log", 4}] = "Sleef_logd4_u10avx";
        retval[{"log", 2}] = "Sleef_logd2_u10sse4";
    } else if (features.sse2) {
        retval[{"log", 2}] = "Sleef_logd2_u10sse2";
    }

    // exp().
    if (features.avx512f) {
        retval[{"exp", 8}] = "Sleef_expd8_u10avx512f";
        retval[{"exp", 4}] = "Sleef_expd4_u10avx2";
        retval[{"exp", 2}] = "Sleef_expd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"exp", 4}] = "Sleef_expd4_u10avx2";
        retval[{"exp", 2}] = "Sleef_expd2_u10avx2128";
    } else if (features.avx) {
        retval[{"exp", 4}] = "Sleef_expd4_u10avx";
        retval[{"exp", 2}] = "Sleef_expd2_u10sse4";
    } else if (features.sse2) {
        retval[{"exp", 2}] = "Sleef_expd2_u10sse2";
    }

    // pow().
    if (features.avx512f) {
        retval[{"pow", 8}] = "Sleef_powd8_u10avx512f";
        retval[{"pow", 4}] = "Sleef_powd4_u10avx2";
        retval[{"pow", 2}] = "Sleef_powd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"pow", 4}] = "Sleef_powd4_u10avx2";
        retval[{"pow", 2}] = "Sleef_powd2_u10avx2128";
    } else if (features.avx) {
        retval[{"pow", 4}] = "Sleef_powd4_u10avx";
        retval[{"pow", 2}] = "Sleef_powd2_u10sse4";
    } else if (features.sse2) {
        retval[{"pow", 2}] = "Sleef_powd2_u10sse2";
    }

    // tan().
    if (features.avx512f) {
        retval[{"tan", 8}] = "Sleef_tand8_u10avx512f";
        retval[{"tan", 4}] = "Sleef_tand4_u10avx2";
        retval[{"tan", 2}] = "Sleef_tand2_u10avx2128";
    } else if (features.avx2) {
        retval[{"tan", 4}] = "Sleef_tand4_u10avx2";
        retval[{"tan", 2}] = "Sleef_tand2_u10avx2128";
    } else if (features.avx) {
        retval[{"tan", 4}] = "Sleef_tand4_u10avx";
        retval[{"tan", 2}] = "Sleef_tand2_u10sse4";
    } else if (features.sse2) {
        retval[{"tan", 2}] = "Sleef_tand2_u10sse2";
    }

    return retval;
}

} // namespace

// Fetch an appropriate sleef function name, given the name of the mathematical
// function f, the desired SIMD width s and the scalar floating-point type t.
// If no sleef function is available, return an empty string.
std::string sleef_function_name(llvm::LLVMContext &c, const std::string &f, llvm::Type *t, std::uint32_t s)
{
    if (t == llvm::Type::getDoubleTy(c)) {
        static const auto sleef_map = detail::make_sleef_map_dbl();

        const auto it = sleef_map.find({f, s});

        if (it == sleef_map.end()) {
            return "";
        } else {
            return it->second;
        }
    } else {
        return "";
    }
}

// NOTE: this function is here only to introduce a fake
// dependency of heyoka on libsleef. This is needed
// in certain setups that would otherwise unlink libsleef
// from heyoka after detecting that no symbol from libsleef
// is being used (even though we need libsleef to be linked
// in for use in the JIT machinery).
HEYOKA_DLL_PUBLIC double sleef_dummy_f(double x)
{
    return ::Sleef_sin_u10(x);
}

} // namespace heyoka::detail

#else

#include <cstdint>
#include <string>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/sleef.hpp>

namespace heyoka::detail
{

// If heyoka is not configured with sleef support, sleef_function_name() will always return
// an empty string.
std::string sleef_function_name(llvm::LLVMContext &, const std::string &, llvm::Type *, std::uint32_t)
{
    return "";
}

} // namespace heyoka::detail

#endif
