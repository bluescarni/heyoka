// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/eop_sw_impl.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/model/sw.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

#define HEYOKA_MODEL_DEFINE_GET_SW_FUNC(name)                                                                          \
    llvm::Function *llvm_get_##name##_##name##p_func(llvm_state &s, llvm::Type *const fp_t,                            \
                                                     const std::uint32_t batch_size, const sw_data &data)              \
    {                                                                                                                  \
        return heyoka::detail::llvm_get_eop_sw_func(s, "sw", fp_t, batch_size, data, #name,                            \
                                                    &heyoka::detail::llvm_get_sw_data_##name);                         \
    }

HEYOKA_MODEL_DEFINE_GET_SW_FUNC(Ap_avg);
HEYOKA_MODEL_DEFINE_GET_SW_FUNC(f107);
HEYOKA_MODEL_DEFINE_GET_SW_FUNC(f107a_center81);

#undef HEYOKA_MODEL_DEFINE_GET_SW_FUNC

} // namespace

} // namespace model::detail

HEYOKA_END_NAMESPACE

// NOLINTBEGIN(cert-err58-cpp,bugprone-throwing-static-initialization)
HEYOKA_MODEL_DEFINE_EOP_SW(sw, Ap_avg, sw_data);
HEYOKA_MODEL_DEFINE_EOP_SW(sw, f107, sw_data);
HEYOKA_MODEL_DEFINE_EOP_SW(sw, f107a_center81, sw_data);
// NOLINTEND(cert-err58-cpp,bugprone-throwing-static-initialization)
