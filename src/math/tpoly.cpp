// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/binomial.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/tpoly.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

tpoly_impl::tpoly_impl() : tpoly_impl(par[0], par[1]) {}

tpoly_impl::tpoly_impl(expression b, expression e)
    : func_base("tpoly", std::vector<expression>{std::move(b), std::move(e)})
{
    if (!std::holds_alternative<param>(args()[0].value())) {
        throw std::invalid_argument("Cannot construct a time polynomial from a non-param argument");
    }
    m_b_idx = std::get<param>(args()[0].value()).idx();

    if (!std::holds_alternative<param>(args()[1].value())) {
        throw std::invalid_argument("Cannot construct a time polynomial from a non-param argument");
    }
    m_e_idx = std::get<param>(args()[1].value()).idx();

    if (m_e_idx <= m_b_idx) {
        throw std::invalid_argument(
            "Cannot construct a time polynomial from param indices {} and {}: the first index is not less than the second"_format(
                m_b_idx, m_e_idx));
    }
}

void tpoly_impl::to_stream(std::ostream &os) const
{
    os << "tpoly({}, {})"_format(m_b_idx, m_e_idx);
}

namespace
{

template <typename T>
llvm::Value *taylor_diff_tpoly_impl(llvm_state &s, const tpoly_impl &tp, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                    std::uint32_t order, std::uint32_t batch_size)
{
    assert(tp.m_e_idx > tp.m_b_idx);
    assert(std::holds_alternative<param>(tp.args()[0].value()));
    assert(std::holds_alternative<param>(tp.args()[1].value()));

    const auto n = (tp.m_e_idx - tp.m_b_idx) - 1u;

    auto &builder = s.builder();

    // Null retval if the diff order is larger than the
    // polynomial order.
    if (order > n) {
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    }

    // Load the time value.
    auto tm = load_vector_from_memory(builder, time_ptr, batch_size);

    // Init the return value with the highest-order coefficient (scaled by the corresponding
    // binomial coefficient).
    assert(tp.m_e_idx > 0u);
    auto bc = binomial<T>(n, order);
    auto ret = taylor_codegen_numparam<T>(s, param{tp.m_e_idx - 1u}, par_ptr, batch_size);
    ret = builder.CreateFMul(ret, vector_splat(builder, codegen<T>(s, number{bc}), batch_size));

    // Horner evaluation of polynomial derivative.
    for (std::uint32_t i_ = 1; i_ <= n - order; ++i_) {
        // NOTE: need to invert i because Horner's method
        // proceeds backwards.
        const auto i = n - order - i_;

        // Compute the binomial coefficient.
        bc = binomial<T>(i + order, order);

        // Load the poly coefficient from the par array and multiply it by bc.
        auto cf = taylor_codegen_numparam<T>(s, param{tp.m_b_idx + i + order}, par_ptr, batch_size);
        cf = builder.CreateFMul(cf, vector_splat(builder, codegen<T>(s, number{bc}), batch_size));

        // Horner iteration.
        ret = builder.CreateFAdd(cf, builder.CreateFMul(ret, tm));
    }

    return ret;
}

} // namespace

llvm::Value *tpoly_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                         const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                         llvm::Value *time_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                         std::uint32_t batch_size, bool) const
{
    return taylor_diff_tpoly_impl<double>(s, *this, par_ptr, time_ptr, order, batch_size);
}

llvm::Value *tpoly_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                          const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                          llvm::Value *time_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                          std::uint32_t batch_size, bool) const
{
    return taylor_diff_tpoly_impl<long double>(s, *this, par_ptr, time_ptr, order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *tpoly_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &,
                                          const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                          llvm::Value *time_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                          std::uint32_t batch_size, bool) const
{
    return taylor_diff_tpoly_impl<mppp::real128>(s, *this, par_ptr, time_ptr, order, batch_size);
}

#endif

namespace
{

template <typename T>
llvm::Function *taylor_c_diff_tpoly_impl(llvm_state &s, const tpoly_impl &tp, std::uint32_t n_uvars,
                                         std::uint32_t batch_size)
{
    assert(tp.m_e_idx > tp.m_b_idx);
    assert(std::holds_alternative<param>(tp.args()[0].value()));
    assert(std::holds_alternative<param>(tp.args()[1].value()));

    // Make the poly degree a compile-time (JIT) constant.
    const auto n_const = (tp.m_e_idx - tp.m_b_idx) - 1u;

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    // NOTE: we mangle on the poly degree as well, so that we will be
    // generating a different function for each polynomial degree.
    const auto na_pair = taylor_c_diff_func_name_args<T>(
        context, "tpoly_{}"_format(n_const), n_uvars, batch_size,
        {std::get<param>(tp.args()[0].value()), std::get<param>(tp.args()[1].value())});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // Helper to fetch the (i, j) binomial coefficient from
        // a precomputed global array. The returned value is already
        // splatted.
        auto get_bc = [&, bc_ptr = llvm_add_bc_array<T>(s, n_const)](llvm::Value *i, llvm::Value *j) {
            auto idx = builder.CreateMul(i, builder.getInt32(n_const + 1u));
            idx = builder.CreateAdd(idx, j);

            auto val = builder.CreateLoad(builder.CreateInBoundsGEP(bc_ptr, {idx}));

            return vector_splat(builder, val, batch_size);
        };

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto par_ptr = f->args().begin() + 3;
        auto t_ptr = f->args().begin() + 4;
        auto b_idx = f->args().begin() + 5;
        auto e_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        // Cache the order of the polynomial.
        auto n = builder.getInt32(n_const);

        // Null retval if the diff order is larger than the
        // polynomial order.
        // NOTE: unsigned comparison.
        llvm_if_then_else(
            s, builder.CreateICmpUGT(ord, n),
            [&]() { builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval); },
            [&]() {
                // Load the time value.
                auto tm = load_vector_from_memory(builder, t_ptr, batch_size);

                // Init the return value with the highest-order coefficient (scaled by the corresponding
                // binomial coefficient).
                auto bc = get_bc(n, ord);
                auto cf = load_vector_from_memory(
                    builder,
                    builder.CreateInBoundsGEP(par_ptr,
                                              {builder.CreateMul(builder.getInt32(batch_size),
                                                                 builder.CreateSub(e_idx, builder.getInt32(1)))}),
                    batch_size);
                cf = builder.CreateFMul(cf, bc);
                builder.CreateStore(cf, retval);

                // Horner evaluation of polynomial derivative.
                llvm_loop_u32(
                    s, builder.getInt32(1), builder.CreateAdd(builder.CreateSub(n, ord), builder.getInt32(1)),
                    [&](llvm::Value *i_) {
                        // NOTE: need to invert i because Horner's method
                        // proceeds backwards.
                        auto i = builder.CreateSub(builder.CreateSub(n, ord), i_);

                        // Get the binomial coefficient.
                        bc = get_bc(builder.CreateAdd(i, ord), ord);

                        // Load the poly coefficient from the par array and multiply it by bc.
                        auto cf_idx = builder.CreateMul(builder.CreateAdd(builder.CreateAdd(b_idx, i), ord),
                                                        builder.getInt32(batch_size));
                        cf = load_vector_from_memory(builder, builder.CreateInBoundsGEP(par_ptr, {cf_idx}), batch_size);
                        cf = builder.CreateFMul(cf, bc);

                        // Horner iteration.
                        auto new_val = builder.CreateFAdd(cf, builder.CreateFMul(builder.CreateLoad(retval), tm));

                        // Update retval.
                        builder.CreateStore(new_val, retval);
                    });
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of tpoly() in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace

llvm::Function *tpoly_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                   bool) const
{
    return taylor_c_diff_tpoly_impl<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *tpoly_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                    bool) const
{
    return taylor_c_diff_tpoly_impl<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *tpoly_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                                    bool) const
{
    return taylor_c_diff_tpoly_impl<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

// Small helper to detect if an expression
// is a tpoly function.
bool is_tpoly(const expression &ex)
{
    if (auto func_ptr = std::get_if<func>(&ex.value());
        func_ptr != nullptr && func_ptr->extract<tpoly_impl>() != nullptr) {
        return true;
    } else {
        return false;
    }
}

} // namespace detail

expression tpoly(expression b, expression e)
{
    return expression{func{detail::tpoly_impl{std::move(b), std::move(e)}}};
}

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::tpoly_impl)
