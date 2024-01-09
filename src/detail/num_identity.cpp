// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/num_identity.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

num_identity_impl::num_identity_impl(expression e) : func_base("num_identity", std::vector{std::move(e)})
{
    assert(std::holds_alternative<number>(args()[0].value()));
}

num_identity_impl::num_identity_impl() : num_identity_impl(0_dbl) {}

llvm::Value *num_identity_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t,
                                            [[maybe_unused]] const std::vector<std::uint32_t> &deps,
                                            const std::vector<llvm::Value *> &, llvm::Value *par_ptr, llvm::Value *,
                                            std::uint32_t, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                            bool) const
{
    assert(args().size() == 1u);
    assert(std::holds_alternative<number>(args()[0].value()));
    assert(deps.empty());

    if (order == 0u) {
        return taylor_codegen_numparam(s, fp_t, std::get<number>(args()[0].value()), par_ptr, batch_size);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

llvm::Function *num_identity_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                      std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);
    assert(std::holds_alternative<number>(args()[0].value()));

    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "num_identity", 0,
        [](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return args[0];
        },
        std::get<number>(args()[0].value()));
}

expression num_identity(expression e)
{
    return expression{func{num_identity_impl{std::move(e)}}};
}

} // namespace detail

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::num_identity_impl)
