// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/vector_math.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// LCOV_EXCL_START

namespace
{

using vf_map_t = std::unordered_map<std::string, std::vector<vf_info>>;

// NOTE: make_vfinfo() is necessary if *any* vectorisation backend is active,
// but at the moment we have only SLEEF.
#if defined(HEYOKA_WITH_SLEEF)

auto make_vfinfo(const char *s_name, std::string v_name, std::string lp_v_name, std::uint32_t width,
                 std::uint32_t nargs)
{
    assert(nargs == 1u || nargs == 2u);

    auto ret = vf_info{std::move(v_name), {}, std::move(lp_v_name), {}, width, nargs};
    ret.vf_abi_attr = fmt::format("_ZGV_LLVM_N{}{}_{}({})", width, nargs == 1u ? "v" : "vv", s_name, ret.name);
    ret.lp_vf_abi_attr = fmt::format("_ZGV_LLVM_N{}{}_{}({})", width, nargs == 1u ? "v" : "vv", s_name, ret.lp_name);
    return ret;
}

#endif

#if defined(HEYOKA_WITH_SLEEF)

// NOTE: helper to fetch the suffix of the low-precision version of the mathematical
// function "sleef_base_name" in SLEEF.
// NOTE: by default, the low-precision versions are denoted by the "u35" suffix
// (indicating 3.5 ULPs of precision). For some functions, the "u35" versions are not available
// and we return the standard-precision suffix instead ("u10").
auto sleef_get_lp_suffix(const std::string &sleef_base_name) -> std::string
{
    static const std::unordered_map<std::string, std::string> lp_suffix_map
        = {{"acosh", "u10"}, {"asinh", "u10"}, {"atanh", "u10"}, {"erf", "u10"}, {"exp", "u10"}, {"pow", "u10"}};

    if (auto it = lp_suffix_map.find(sleef_base_name); it == lp_suffix_map.end()) {
        return "u35";
    } else {
        return it->second;
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto add_vfinfo_sleef(vf_map_t &retval, const char *scalar_name, const char *sleef_base_name, std::string_view sleef_tp,
                      std::uint32_t nargs = 1)
{
    assert(sleef_tp == "d" || sleef_tp == "f");
    assert(retval.find(scalar_name) == retval.end());
    assert(nargs > 0u);

    auto make_sleef_vfinfo = [&](std::uint32_t width, const char *iset) {
        return make_vfinfo(scalar_name, fmt::format("Sleef_{}{}{}_u10{}", sleef_base_name, sleef_tp, width, iset),
                           fmt::format("Sleef_{}{}{}_{}{}", sleef_base_name, sleef_tp, width,
                                       sleef_get_lp_suffix(sleef_base_name), iset),
                           width, nargs);
    };

    const auto &features = get_target_features();

    // NOTE: we need to select the SIMD width(s) based on the floating-point type (sleef_tp).
    // All supported SIMD extensions start with a minimum width of 2 for double-precision
    // and 4 for single-precision, possibly supporting larger widths. So we use these two
    // values for the computation.
    const std::uint32_t base_simd_width = sleef_tp == "d" ? 2 : 4;

    if (features.avx512f) {
        retval[scalar_name]
            = {make_sleef_vfinfo(base_simd_width, "avx2128"), make_sleef_vfinfo(base_simd_width * 2u, "avx2"),
               make_sleef_vfinfo(base_simd_width * 4u, "avx512f")};
    } else if (features.avx2) {
        retval[scalar_name]
            = {make_sleef_vfinfo(base_simd_width, "avx2128"), make_sleef_vfinfo(base_simd_width * 2u, "avx2")};
    } else if (features.avx) {
        retval[scalar_name]
            = {make_sleef_vfinfo(base_simd_width, "sse4"), make_sleef_vfinfo(base_simd_width * 2u, "avx")};
    } else if (features.sse2) {
        retval[scalar_name] = {make_sleef_vfinfo(base_simd_width, "sse2")};
    } else if (features.aarch64) {
        retval[scalar_name] = {make_sleef_vfinfo(base_simd_width, "advsimd")};
    } else if (features.vsx) {
        // NOTE: at this time the sleef conda package for PPC64 does not seem
        // to provide VSX3 functions. Thus, for now we use only the
        // VSX implementations.
        retval[scalar_name] = {make_sleef_vfinfo(base_simd_width, "vsx")};
    }
}

#endif

auto make_vf_map()
{

    vf_map_t retval;

#if defined(HEYOKA_WITH_SLEEF)

    // NOTE: currently we are not adding here any sqrt() implementation provided
    // by sleef, on the assumption that usually sqrt() is implemented directly in hardware
    // and thus there's no need to go through sleef. This is certainly true for x86,
    // but I am not 100% sure for the other archs. Let's keep this in mind.
    // NOTE: the same holds for things like abs() and floor().

    // Single-precision.
    add_vfinfo_sleef(retval, "llvm.sin.f32", "sin", "f");
    add_vfinfo_sleef(retval, "llvm.cos.f32", "cos", "f");
    add_vfinfo_sleef(retval, "llvm.log.f32", "log", "f");
    add_vfinfo_sleef(retval, "llvm.exp.f32", "exp", "f");
    add_vfinfo_sleef(retval, "llvm.pow.f32", "pow", "f", 2);
    add_vfinfo_sleef(retval, "sinhf", "sinh", "f");
    add_vfinfo_sleef(retval, "coshf", "cosh", "f");
    add_vfinfo_sleef(retval, "asinf", "asin", "f");
    add_vfinfo_sleef(retval, "acosf", "acos", "f");
    add_vfinfo_sleef(retval, "asinhf", "asinh", "f");
    add_vfinfo_sleef(retval, "acoshf", "acosh", "f");
    add_vfinfo_sleef(retval, "tanf", "tan", "f");
    add_vfinfo_sleef(retval, "tanhf", "tanh", "f");
    add_vfinfo_sleef(retval, "atanf", "atan", "f");
    add_vfinfo_sleef(retval, "atanhf", "atanh", "f");
    add_vfinfo_sleef(retval, "atan2f", "atan2", "f", 2);
    add_vfinfo_sleef(retval, "erff", "erf", "f");

    // Double-precision.
    add_vfinfo_sleef(retval, "llvm.sin.f64", "sin", "d");
    add_vfinfo_sleef(retval, "llvm.cos.f64", "cos", "d");
    add_vfinfo_sleef(retval, "llvm.log.f64", "log", "d");
    add_vfinfo_sleef(retval, "llvm.exp.f64", "exp", "d");
    add_vfinfo_sleef(retval, "llvm.pow.f64", "pow", "d", 2);
    add_vfinfo_sleef(retval, "sinh", "sinh", "d");
    add_vfinfo_sleef(retval, "cosh", "cosh", "d");
    add_vfinfo_sleef(retval, "asin", "asin", "d");
    add_vfinfo_sleef(retval, "acos", "acos", "d");
    add_vfinfo_sleef(retval, "asinh", "asinh", "d");
    add_vfinfo_sleef(retval, "acosh", "acosh", "d");
    add_vfinfo_sleef(retval, "tan", "tan", "d");
    add_vfinfo_sleef(retval, "tanh", "tanh", "d");
    add_vfinfo_sleef(retval, "atan", "atan", "d");
    add_vfinfo_sleef(retval, "atanh", "atanh", "d");
    add_vfinfo_sleef(retval, "atan2", "atan2", "d", 2);
    add_vfinfo_sleef(retval, "erf", "erf", "d");

#endif

#if !defined(NDEBUG)

    // Checks in debug mode.
    for (const auto &[key, value] : retval) {
        assert(!value.empty());
        assert(std::none_of(value.begin(), value.end(), [](const auto &v) {
            return v.width == 0u || v.nargs == 0u || v.name.empty() || v.vf_abi_attr.empty();
        }));
        assert(std::is_sorted(value.begin(), value.end(),
                              [](const auto &v1, const auto &v2) { return v1.width < v2.width; }));
    }

#endif

    return retval;
}

} // namespace

// LCOV_EXCL_STOP

const std::vector<vf_info> &lookup_vf_info(const std::string &name)
{
    static const std::vector<vf_info> lookup_fail;

    static const auto vf_map = make_vf_map();

    if (const auto it = vf_map.find(name); it != vf_map.end()) {
        return it->second;
    } else {
        return lookup_fail;
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
