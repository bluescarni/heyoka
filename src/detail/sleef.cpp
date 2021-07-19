// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

#include <iostream>

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
    } else if (features.aarch64) {
        retval[{"sin", 2}] = "Sleef_sind2_u10advsimd";
    } else if (features.vsx3) {
        std::cout << "SLEEF SIN VSX3\n";
        retval[{"sin", 2}] = "Sleef_sind2_u10vsx3";
    } else if (features.vsx) {
        retval[{"sin", 2}] = "Sleef_sind2_u10vsx";
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
    } else if (features.aarch64) {
        retval[{"cos", 2}] = "Sleef_cosd2_u10advsimd";
    } else if (features.vsx3) {
        std::cout << "SLEEF COS VSX3\n";
        retval[{"cos", 2}] = "Sleef_cosd2_u10vsx3";
    } else if (features.vsx) {
        retval[{"cos", 2}] = "Sleef_cosd2_u10vsx";
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
    } else if (features.aarch64) {
        retval[{"log", 2}] = "Sleef_logd2_u10advsimd";
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
    } else if (features.aarch64) {
        retval[{"exp", 2}] = "Sleef_expd2_u10advsimd";
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
    } else if (features.aarch64) {
        retval[{"pow", 2}] = "Sleef_powd2_u10advsimd";
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
    } else if (features.aarch64) {
        retval[{"tan", 2}] = "Sleef_tand2_u10advsimd";
    }

    // asin().
    if (features.avx512f) {
        retval[{"asin", 8}] = "Sleef_asind8_u10avx512f";
        retval[{"asin", 4}] = "Sleef_asind4_u10avx2";
        retval[{"asin", 2}] = "Sleef_asind2_u10avx2128";
    } else if (features.avx2) {
        retval[{"asin", 4}] = "Sleef_asind4_u10avx2";
        retval[{"asin", 2}] = "Sleef_asind2_u10avx2128";
    } else if (features.avx) {
        retval[{"asin", 4}] = "Sleef_asind4_u10avx";
        retval[{"asin", 2}] = "Sleef_asind2_u10sse4";
    } else if (features.sse2) {
        retval[{"asin", 2}] = "Sleef_asind2_u10sse2";
    } else if (features.aarch64) {
        retval[{"asin", 2}] = "Sleef_asind2_u10advsimd";
    }

    // acos().
    if (features.avx512f) {
        retval[{"acos", 8}] = "Sleef_acosd8_u10avx512f";
        retval[{"acos", 4}] = "Sleef_acosd4_u10avx2";
        retval[{"acos", 2}] = "Sleef_acosd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"acos", 4}] = "Sleef_acosd4_u10avx2";
        retval[{"acos", 2}] = "Sleef_acosd2_u10avx2128";
    } else if (features.avx) {
        retval[{"acos", 4}] = "Sleef_acosd4_u10avx";
        retval[{"acos", 2}] = "Sleef_acosd2_u10sse4";
    } else if (features.sse2) {
        retval[{"acos", 2}] = "Sleef_acosd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"acos", 2}] = "Sleef_acosd2_u10advsimd";
    }

    // atan().
    if (features.avx512f) {
        retval[{"atan", 8}] = "Sleef_atand8_u10avx512f";
        retval[{"atan", 4}] = "Sleef_atand4_u10avx2";
        retval[{"atan", 2}] = "Sleef_atand2_u10avx2128";
    } else if (features.avx2) {
        retval[{"atan", 4}] = "Sleef_atand4_u10avx2";
        retval[{"atan", 2}] = "Sleef_atand2_u10avx2128";
    } else if (features.avx) {
        retval[{"atan", 4}] = "Sleef_atand4_u10avx";
        retval[{"atan", 2}] = "Sleef_atand2_u10sse4";
    } else if (features.sse2) {
        retval[{"atan", 2}] = "Sleef_atand2_u10sse2";
    } else if (features.aarch64) {
        retval[{"atan", 2}] = "Sleef_atand2_u10advsimd";
    }

    // cosh().
    if (features.avx512f) {
        retval[{"cosh", 8}] = "Sleef_coshd8_u10avx512f";
        retval[{"cosh", 4}] = "Sleef_coshd4_u10avx2";
        retval[{"cosh", 2}] = "Sleef_coshd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"cosh", 4}] = "Sleef_coshd4_u10avx2";
        retval[{"cosh", 2}] = "Sleef_coshd2_u10avx2128";
    } else if (features.avx) {
        retval[{"cosh", 4}] = "Sleef_coshd4_u10avx";
        retval[{"cosh", 2}] = "Sleef_coshd2_u10sse4";
    } else if (features.sse2) {
        retval[{"cosh", 2}] = "Sleef_coshd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"cosh", 2}] = "Sleef_coshd2_u10advsimd";
    }

    // sinh().
    if (features.avx512f) {
        retval[{"sinh", 8}] = "Sleef_sinhd8_u10avx512f";
        retval[{"sinh", 4}] = "Sleef_sinhd4_u10avx2";
        retval[{"sinh", 2}] = "Sleef_sinhd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"sinh", 4}] = "Sleef_sinhd4_u10avx2";
        retval[{"sinh", 2}] = "Sleef_sinhd2_u10avx2128";
    } else if (features.avx) {
        retval[{"sinh", 4}] = "Sleef_sinhd4_u10avx";
        retval[{"sinh", 2}] = "Sleef_sinhd2_u10sse4";
    } else if (features.sse2) {
        retval[{"sinh", 2}] = "Sleef_sinhd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"sinh", 2}] = "Sleef_sinhd2_u10advsimd";
    }

    // tanh().
    if (features.avx512f) {
        retval[{"tanh", 8}] = "Sleef_tanhd8_u10avx512f";
        retval[{"tanh", 4}] = "Sleef_tanhd4_u10avx2";
        retval[{"tanh", 2}] = "Sleef_tanhd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"tanh", 4}] = "Sleef_tanhd4_u10avx2";
        retval[{"tanh", 2}] = "Sleef_tanhd2_u10avx2128";
    } else if (features.avx) {
        retval[{"tanh", 4}] = "Sleef_tanhd4_u10avx";
        retval[{"tanh", 2}] = "Sleef_tanhd2_u10sse4";
    } else if (features.sse2) {
        retval[{"tanh", 2}] = "Sleef_tanhd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"tanh", 2}] = "Sleef_tanhd2_u10advsimd";
    }

    // asinh().
    if (features.avx512f) {
        retval[{"asinh", 8}] = "Sleef_asinhd8_u10avx512f";
        retval[{"asinh", 4}] = "Sleef_asinhd4_u10avx2";
        retval[{"asinh", 2}] = "Sleef_asinhd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"asinh", 4}] = "Sleef_asinhd4_u10avx2";
        retval[{"asinh", 2}] = "Sleef_asinhd2_u10avx2128";
    } else if (features.avx) {
        retval[{"asinh", 4}] = "Sleef_asinhd4_u10avx";
        retval[{"asinh", 2}] = "Sleef_asinhd2_u10sse4";
    } else if (features.sse2) {
        retval[{"asinh", 2}] = "Sleef_asinhd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"asinh", 2}] = "Sleef_asinhd2_u10advsimd";
    }

    // acosh().
    if (features.avx512f) {
        retval[{"acosh", 8}] = "Sleef_acoshd8_u10avx512f";
        retval[{"acosh", 4}] = "Sleef_acoshd4_u10avx2";
        retval[{"acosh", 2}] = "Sleef_acoshd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"acosh", 4}] = "Sleef_acoshd4_u10avx2";
        retval[{"acosh", 2}] = "Sleef_acoshd2_u10avx2128";
    } else if (features.avx) {
        retval[{"acosh", 4}] = "Sleef_acoshd4_u10avx";
        retval[{"acosh", 2}] = "Sleef_acoshd2_u10sse4";
    } else if (features.sse2) {
        retval[{"acosh", 2}] = "Sleef_acoshd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"acosh", 2}] = "Sleef_acoshd2_u10advsimd";
    }

    // atanh().
    if (features.avx512f) {
        retval[{"atanh", 8}] = "Sleef_atanhd8_u10avx512f";
        retval[{"atanh", 4}] = "Sleef_atanhd4_u10avx2";
        retval[{"atanh", 2}] = "Sleef_atanhd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"atanh", 4}] = "Sleef_atanhd4_u10avx2";
        retval[{"atanh", 2}] = "Sleef_atanhd2_u10avx2128";
    } else if (features.avx) {
        retval[{"atanh", 4}] = "Sleef_atanhd4_u10avx";
        retval[{"atanh", 2}] = "Sleef_atanhd2_u10sse4";
    } else if (features.sse2) {
        retval[{"atanh", 2}] = "Sleef_atanhd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"atanh", 2}] = "Sleef_atanhd2_u10advsimd";
    }

    // erf().
    if (features.avx512f) {
        retval[{"erf", 8}] = "Sleef_erfd8_u10avx512f";
        retval[{"erf", 4}] = "Sleef_erfd4_u10avx2";
        retval[{"erf", 2}] = "Sleef_erfd2_u10avx2128";
    } else if (features.avx2) {
        retval[{"erf", 4}] = "Sleef_erfd4_u10avx2";
        retval[{"erf", 2}] = "Sleef_erfd2_u10avx2128";
    } else if (features.avx) {
        retval[{"erf", 4}] = "Sleef_erfd4_u10avx";
        retval[{"erf", 2}] = "Sleef_erfd2_u10sse4";
    } else if (features.sse2) {
        retval[{"erf", 2}] = "Sleef_erfd2_u10sse2";
    } else if (features.aarch64) {
        retval[{"erf", 2}] = "Sleef_erfd2_u10advsimd";
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
