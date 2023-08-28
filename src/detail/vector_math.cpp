// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

auto make_vfinfo(const char *s_name, std::string v_name, std::uint32_t width, std::uint32_t nargs)
{
    assert(nargs == 1u || nargs == 2u);

    auto ret = vf_info{std::move(v_name), {}, width, nargs};
    ret.vf_abi_attr = fmt::format("_ZGV_LLVM_N{}{}_{}({})", width, nargs == 1u ? "v" : "vv", s_name, ret.name);
    return ret;
}

#if defined(HEYOKA_WITH_SLEEF)

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto add_vfinfo_sleef(vf_map_t &retval, const char *scalar_name, const char *sleef_base_name, const char *sleef_tp,
                      std::uint32_t nargs = 1)
{
    assert(retval.find(scalar_name) == retval.end());
    assert(nargs > 0u);

    auto make_sleef_vfinfo = [&](std::uint32_t width, const char *iset) {
        return make_vfinfo(scalar_name, fmt::format("Sleef_{}{}{}_u10{}", sleef_base_name, sleef_tp, width, iset),
                           width, nargs);
    };

    const auto &features = get_target_features();

    if (features.avx512f) {
        retval[scalar_name]
            = {make_sleef_vfinfo(2, "avx2128"), make_sleef_vfinfo(4, "avx2"), make_sleef_vfinfo(8, "avx512f")};
    } else if (features.avx2) {
        retval[scalar_name] = {make_sleef_vfinfo(2, "avx2128"), make_sleef_vfinfo(4, "avx2")};
    } else if (features.avx) {
        retval[scalar_name] = {make_sleef_vfinfo(2, "sse4"), make_sleef_vfinfo(4, "avx")};
    } else if (features.sse2) {
        retval[scalar_name] = {make_sleef_vfinfo(2, "sse2")};
    } else if (features.aarch64) {
        retval[scalar_name] = {make_sleef_vfinfo(2, "advsimd")};
    } else if (features.vsx) {
        // NOTE: at this time the sleef conda package for PPC64 does not seem
        // to provide VSX3 functions. Thus, for now we use only the
        // VSX implementations.
        retval[scalar_name] = {make_sleef_vfinfo(2, "vsx")};
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

    // Double-precision.
    add_vfinfo_sleef(retval, "llvm.sin.f64", "sin", "d");
    add_vfinfo_sleef(retval, "llvm.cos.f64", "cos", "d");
    add_vfinfo_sleef(retval, "llvm.log.f64", "log", "d");
    add_vfinfo_sleef(retval, "llvm.exp.f64", "exp", "d");
    add_vfinfo_sleef(retval, "llvm.pow.f64", "pow", "d", 2);

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
