// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_impl.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

template <typename Data>
void eop_sw_impl_base<Data>::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<func_base>(*this);
    oa << m_data;
}

template <typename Data>
void eop_sw_impl_base<Data>::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<func_base>(*this);
    ia >> m_data;
}

// NOTE: this will be used only during serialisation.
template <typename Data>
eop_sw_impl_base<Data>::eop_sw_impl_base() : func_base("eop_sw_undefined", {heyoka::time})
{
}

template <typename Data>
eop_sw_impl_base<Data>::eop_sw_impl_base(const std::string &descr, const std::string &name, expression time_expr,
                                         Data data)
    // NOTE: in the creation of the function name, we must incorporate the following:
    //
    // - the descriptor,
    // - the name of the EOP/SW quantity or its derivative,
    // - the total number of rows in the data table,
    // - the timestamp and identifier of the data.
    //
    // In particular, it is important that the size, timestamp and identifier of the data table are included, as these
    // three values are used to uniquely identify an EOP/SW dataset (they are also used in the name mangling scheme for
    // the JIT-compiled EOP/SW data arrays and the LLVM functions for the simultaneous computation of EOP/SW quantities
    // and their derivatives).
    //
    // NOTE: '-' is intentionally chosen as the separator between timestamp and identifier. Timestamp and identifier are
    // both guaranteed not to contain '-', thus the boundary between the two is unambiguous.
    : func_base(fmt::format("{}_{}_{}_{}-{}", descr, name, data.get_table().size(), data.get_timestamp(),
                            data.get_identifier()),
                {std::move(time_expr)}),
      m_data(std::move(data))
{
    assert(!descr.empty());
    assert(!name.empty());
}

template <typename Data>
eop_sw_impl_base<Data>::eop_sw_impl_base(const eop_sw_impl_base &) = default;

template <typename Data>
eop_sw_impl_base<Data>::eop_sw_impl_base(eop_sw_impl_base &&) noexcept = default;

// LCOV_EXCL_START

template <typename Data>
eop_sw_impl_base<Data> &eop_sw_impl_base<Data>::operator=(const eop_sw_impl_base &) = default;

template <typename Data>
eop_sw_impl_base<Data> &eop_sw_impl_base<Data>::operator=(eop_sw_impl_base &&) noexcept = default;

// LCOV_EXCL_STOP

// Small wrapper to check that we have eop/sw data to work with in eop_sw_impl_base. It should never happen that we end
// up throwing here while using the public API, but better safe than sorry.
template <typename Data>
const Data &eop_sw_impl_base<Data>::checked_get_data() const
{
    if (m_data) [[likely]] {
        return *m_data;
    }

    // LCOV_EXCL_START
    throw std::invalid_argument("Error: missing EOP/SW data");
    // LCOV_EXCL_STOP
}

// Small helper that returns the evaluation of the EOP/SW quantity or its derivative via the invocation of the combined
// LLVM function produced by get_llvm_eval_f().
//
// Factored out because it is re-used in several places below.
//
// The component argument selects whether the EOP/SW quantity (0) or its derivative (1) is returned, arg is the
// evaluation argument, fp_t the scalar type to be used for the evaluation.
template <typename Data>
llvm::Value *eop_sw_impl_base<Data>::llvm_eval_helper(llvm_state &s, const unsigned component, llvm::Value *const arg,
                                                      llvm::Type *const fp_t, const std::uint32_t batch_size) const
{
    assert(component == 0u || component == 1u);

    // Fetch/create the function for the computation of the EOP/SW quantity and its derivative.
    auto *const f = get_llvm_eval_f(s, fp_t, batch_size, checked_get_data());

    // Invoke it.
    auto &bld = s.builder();
    auto *const x_xp = bld.CreateCall(f, arg);

    // Fetch the value and return it.
    return bld.CreateExtractValue(x_xp, component);
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC eop_sw_impl_base<eop_data>;
template class HEYOKA_DLL_PUBLIC eop_sw_impl_base<sw_data>;

template <typename Data>
void eop_sw_impl<Data>::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<eop_sw_impl_base<Data>>(*this);
}

template <typename Data>
void eop_sw_impl<Data>::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<eop_sw_impl_base<Data>>(*this);
}

// NOTE: this will be used only during serialisation.
template <typename Data>
eop_sw_impl<Data>::eop_sw_impl() = default;

template <typename Data>
eop_sw_impl<Data>::eop_sw_impl(const std::string &descr, const std::string &name, expression time_expr, Data data)
    : eop_sw_impl_base<Data>(descr, name, std::move(time_expr), std::move(data))
{
}

template <typename Data>
eop_sw_impl<Data>::~eop_sw_impl() = default;

template <typename Data>
eop_sw_impl<Data>::eop_sw_impl(const eop_sw_impl &) = default;

template <typename Data>
eop_sw_impl<Data>::eop_sw_impl(eop_sw_impl &&) noexcept = default;

// LCOV_EXCL_START

template <typename Data>
eop_sw_impl<Data> &eop_sw_impl<Data>::operator=(const eop_sw_impl &) = default;

template <typename Data>
eop_sw_impl<Data> &eop_sw_impl<Data>::operator=(eop_sw_impl &&) noexcept = default;

// LCOV_EXCL_STOP

template <typename Data>
llvm::Value *eop_sw_impl<Data>::llvm_evaluate(llvm_state &s, const std::vector<llvm::Value *> &llvm_args,
                                              llvm::Type *const val_t, llvm::Value *, bool) const
{
    assert(llvm_args.size() == 1u);

    // Determine the batch size.
    const auto batch_size = heyoka::detail::get_vector_size(val_t);

    // Run the evaluation and return the result.
    return this->llvm_eval_helper(s, 0, llvm_args[0], val_t->getScalarType(), batch_size);
}

// NOTE: here we implement a custom decomposition which injects the derivative of the EOP/SW quantity into the
// decomposition. We do this because the Taylor derivatives of the EOP/SW quantity use the value of its derivative, thus
// by inserting it into the decomposition we can compute it once at order 0 and then re-use it for the higher orders.
template <typename Data>
taylor_dc_t::size_type eop_sw_impl<Data>::eop_sw_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(this->args().size() == 1u);

    // Append the decomposition of the derivative.
    u_vars_defs.emplace_back(this->gradient()[0], std::vector<std::uint32_t>{});

    // Append the decomposition of the EOP/SW quantity.
    u_vars_defs.emplace_back(std::move(*this).ex_from_this(), std::vector<std::uint32_t>{});

    // Setup the hidden dep for the EOP/SW quantity (its derivative does not have hidden deps).
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the decomposed EOP/SW quantity).
    return u_vars_defs.size() - 1u;
}

template <typename Data>
llvm::Value *
eop_sw_impl<Data>::taylor_diff(llvm_state &s, llvm::Type *const fp_t, const std::vector<std::uint32_t> &deps,
                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                               const std::vector<llvm::Value *> &arr, llvm::Value *const par_ptr, llvm::Value *,
                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                               const std::uint32_t n_uvars, const std::uint32_t order, std::uint32_t,
                               const std::uint32_t batch_size, bool) const
{
    assert(this->args().size() == 1u);
    assert(deps.size() == 1u);

    namespace hd = heyoka::detail;

    return std::visit(
        [&]<typename T>(const T &v) -> llvm::Value * {
            if constexpr (hd::is_num_param_v<T>) {
                // Number/param argument.
                if (order == 0u) {
                    return this->llvm_eval_helper(s, 0, hd::taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size),
                                                  fp_t, batch_size);
                } else {
                    return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
                }
            } else if constexpr (std::same_as<T, variable>) {
                // Variable argument.
                //
                // NOTE: the EOP/SW quantity x is defined as:
                //
                // x(b(t)) = c_0(b(t)) + c_1(b(t))*b(t),
                //
                // where c0 and c1 are step functions (hence with null derivatives). Taking the first order derivative
                // wrt t:
                //
                // x'(b(t)) = c_1(b(t))*b'(t),
                //
                // and, generalising,
                //
                // x^[n] = c_1 * b^[n],
                //
                // where c_1 is xp(b(t)).

                // Fetch the index of the variable.
                const auto b_idx = hd::uname_to_index(v.name());

                // Load b^[n].
                auto *const bn = hd::taylor_fetch_diff(arr, b_idx, order, n_uvars);

                if (order == 0u) {
                    // Evaluate the EOP/SW quantity for b^[0] and return it.
                    return this->llvm_eval_helper(s, 0, bn, fp_t, batch_size);
                } else {
                    // Fetch the value of the derivative of the EOP/SW quantity from the hidden dep.
                    auto *const xp_val = hd::taylor_fetch_diff(arr, deps[0], 0, n_uvars);

                    // Return xp*b^[n].
                    return hd::llvm_fmul(s, xp_val, bn);
                }
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor "
                                            "derivative of an EOP/SW quantity");
                // LCOV_EXCL_STOP
            }
        },
        this->args()[0].value());
}

template <typename Data>
llvm::Function *eop_sw_impl<Data>::taylor_c_diff_func(llvm_state &s, llvm::Type *const fp_t,
                                                      const std::uint32_t n_uvars, const std::uint32_t batch_size,
                                                      bool) const
{
    assert(this->args().size() == 1u);

    namespace hd = heyoka::detail;

    return std::visit(
        [&]<typename T>(const T &v) -> llvm::Function * {
            if constexpr (hd::is_num_param_v<T>) {
                // Number/param argument.
                return hd::taylor_c_diff_func_numpar(
                    s, fp_t, n_uvars, batch_size, this->get_name(), 1,
                    [this, &s, fp_t, batch_size](const auto &llvm_args) {
                        // LCOV_EXCL_START
                        assert(llvm_args.size() == 1u);
                        assert(llvm_args[0] != nullptr);
                        // LCOV_EXCL_STOP

                        return this->llvm_eval_helper(s, 0, llvm_args[0], fp_t, batch_size);
                    },
                    v);
            } else if constexpr (std::same_as<T, variable>) {
                // Variable argument.
                auto &md = s.module();
                auto &bld = s.builder();
                auto &ctx = s.context();

                // Fetch the vector floating-point type.
                auto *const val_t = hd::make_vector_type(fp_t, batch_size);

                const auto [fname, fargs]
                    = hd::taylor_c_diff_func_name_args(ctx, fp_t, this->get_name(), n_uvars, batch_size, {v}, 1);

                // Try to see if we already created the function.
                auto *f = md.getFunction(fname);
                if (f != nullptr) {
                    return f;
                }

                // The function was not created before, do it now.

                // Setup the insertion point restorer.
                const hd::ip_restorer ipr(bld);

                // The return type is val_t.
                auto *const ft = llvm::FunctionType::get(val_t, fargs, false);
                // Create the function
                f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
                assert(f != nullptr);

                // Fetch the necessary function arguments.
                auto *const ord = f->args().begin();
                auto *const diff_ptr = f->args().begin() + 2;
                auto *const b_idx = f->args().begin() + 5;
                auto *const dep_idx = f->args().begin() + 6;

                // Create a new basic block to start insertion into.
                bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

                // Create the return value.
                auto *const retval = bld.CreateAlloca(val_t);

                hd::llvm_if_then_else(
                    s, bld.CreateICmpEQ(ord, bld.getInt32(0)),
                    [&]() {
                        // For order 0, compute the EOP/SW quantity for the order 0 of b_idx.

                        // Load b^[0].
                        auto *const b0 = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, bld.getInt32(0), b_idx);

                        // Evaluate.
                        auto *const x_val = this->llvm_eval_helper(s, 0, b0, fp_t, batch_size);

                        // Store the result.
                        bld.CreateStore(x_val, retval);
                    },
                    [&]() {
                        // For order > 0, we must compute xp*b^[n].

                        // Load b^[n].
                        auto *const bn = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, b_idx);

                        // Load the value of xp.
                        auto *const xp = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, bld.getInt32(0), dep_idx);

                        // Compute and store xp*b^[n].
                        bld.CreateStore(hd::llvm_fmul(s, xp, bn), retval);
                    });

                // Return the result.
                bld.CreateRet(bld.CreateLoad(val_t, retval));

                return f;
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor "
                                            "derivative of an EOP/SW quantity in compact mode");
                // LCOV_EXCL_STOP
            }
        },
        this->args()[0].value());
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC eop_sw_impl<eop_data>;
template class HEYOKA_DLL_PUBLIC eop_sw_impl<sw_data>;

template <typename Data>
void eop_sw_p_impl<Data>::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<eop_sw_impl_base<Data>>(*this);
}

template <typename Data>
void eop_sw_p_impl<Data>::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<eop_sw_impl_base<Data>>(*this);
}

// NOTE: this will be used only during serialisation.
template <typename Data>
eop_sw_p_impl<Data>::eop_sw_p_impl() = default;

template <typename Data>
eop_sw_p_impl<Data>::eop_sw_p_impl(const std::string &descr, const std::string &name, expression time_expr, Data data)
    : eop_sw_impl_base<Data>(descr, name, std::move(time_expr), std::move(data))
{
}

template <typename Data>
eop_sw_p_impl<Data>::~eop_sw_p_impl() = default;

template <typename Data>
eop_sw_p_impl<Data>::eop_sw_p_impl(const eop_sw_p_impl &) = default;

template <typename Data>
eop_sw_p_impl<Data>::eop_sw_p_impl(eop_sw_p_impl &&) noexcept = default;

// LCOV_EXCL_START

template <typename Data>
eop_sw_p_impl<Data> &eop_sw_p_impl<Data>::operator=(const eop_sw_p_impl &) = default;

template <typename Data>
eop_sw_p_impl<Data> &eop_sw_p_impl<Data>::operator=(eop_sw_p_impl &&) noexcept = default;

// LCOV_EXCL_STOP

// NOTE: the gradient of the derivatives of EOP/SW quantities is always zero.
template <typename Data>
std::vector<expression> eop_sw_p_impl<Data>::gradient() const
{
    return {0_dbl};
}

template <typename Data>
llvm::Value *eop_sw_p_impl<Data>::llvm_evaluate(llvm_state &s, const std::vector<llvm::Value *> &llvm_args,
                                                llvm::Type *const val_t, llvm::Value *, bool) const
{
    assert(llvm_args.size() == 1u);

    // Determine the batch size.
    const auto batch_size = heyoka::detail::get_vector_size(val_t);

    // Run the evaluation and return the result.
    return this->llvm_eval_helper(s, 1, llvm_args[0], val_t->getScalarType(), batch_size);
}

template <typename Data>
llvm::Value *eop_sw_p_impl<Data>::taylor_diff(llvm_state &s, llvm::Type *const fp_t,
                                              [[maybe_unused]] const std::vector<std::uint32_t> &deps,
                                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                              const std::vector<llvm::Value *> &arr, llvm::Value *const par_ptr,
                                              llvm::Value *,
                                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                              const std::uint32_t n_uvars, const std::uint32_t order, std::uint32_t,
                                              const std::uint32_t batch_size, bool) const
{
    assert(this->args().size() == 1u);
    assert(deps.empty());

    namespace hd = heyoka::detail;

    return std::visit(
        [&]<typename T>(const T &v) -> llvm::Value * {
            if constexpr (hd::is_num_param_v<T>) {
                // Number/param argument.
                if (order == 0u) {
                    return this->llvm_eval_helper(s, 1, hd::taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size),
                                                  fp_t, batch_size);
                } else {
                    return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
                }
            } else if constexpr (std::same_as<T, variable>) {
                // Variable argument.
                //
                // NOTE: the Taylor derivatives of the derivative of an EOP/SW quantity are all zero (except of course
                // for the order-0 derivatives).
                if (order == 0u) {
                    // Fetch the index of the variable.
                    const auto b_idx = hd::uname_to_index(v.name());

                    // Load b^[0].
                    auto *const b0 = hd::taylor_fetch_diff(arr, b_idx, 0, n_uvars);

                    // Evaluate the derivative and return it.
                    return this->llvm_eval_helper(s, 1, b0, fp_t, batch_size);
                } else {
                    // Return zero.
                    return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
                }
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor "
                                            "derivative of the derivative of an EOP/SW quantity");
                // LCOV_EXCL_STOP
            }
        },
        this->args()[0].value());
}

template <typename Data>
llvm::Function *eop_sw_p_impl<Data>::taylor_c_diff_func(llvm_state &s, llvm::Type *const fp_t,
                                                        const std::uint32_t n_uvars, const std::uint32_t batch_size,
                                                        bool) const
{
    assert(this->args().size() == 1u);

    namespace hd = heyoka::detail;

    return std::visit(
        [&]<typename T>(const T &v) -> llvm::Function * {
            if constexpr (hd::is_num_param_v<T>) {
                // Number/param argument.
                return hd::taylor_c_diff_func_numpar(
                    s, fp_t, n_uvars, batch_size, this->get_name(), 0,
                    [this, &s, fp_t, batch_size](const auto &llvm_args) {
                        // LCOV_EXCL_START
                        assert(llvm_args.size() == 1u);
                        assert(llvm_args[0] != nullptr);
                        // LCOV_EXCL_STOP

                        return this->llvm_eval_helper(s, 1, llvm_args[0], fp_t, batch_size);
                    },
                    v);
            } else if constexpr (std::same_as<T, variable>) {
                // Variable argument.
                auto &md = s.module();
                auto &bld = s.builder();
                auto &ctx = s.context();

                // Fetch the vector floating-point type.
                auto *const val_t = hd::make_vector_type(fp_t, batch_size);

                const auto [fname, fargs]
                    = hd::taylor_c_diff_func_name_args(ctx, fp_t, this->get_name(), n_uvars, batch_size, {v});

                // Try to see if we already created the function.
                auto *f = md.getFunction(fname);
                if (f != nullptr) {
                    return f;
                }

                // The function was not created before, do it now.

                // Setup the insertion point restorer.
                const hd::ip_restorer ipr(bld);

                // The return type is val_t.
                auto *const ft = llvm::FunctionType::get(val_t, fargs, false);
                // Create the function
                f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
                assert(f != nullptr);

                // Fetch the necessary function arguments.
                auto *const ord = f->args().begin();
                auto *const diff_ptr = f->args().begin() + 2;
                auto *const b_idx = f->args().begin() + 5;

                // Create a new basic block to start insertion into.
                bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

                // Create the return value.
                auto *const retval = bld.CreateAlloca(val_t);

                hd::llvm_if_then_else(
                    s, bld.CreateICmpEQ(ord, bld.getInt32(0)),
                    [&]() {
                        // For order 0, compute the derivative of the EOP/SW quantity for the order 0 of b_idx.

                        // Load b^[0].
                        auto *const b0 = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, bld.getInt32(0), b_idx);

                        // Compute the derivative of the EOP/SW quantity.
                        auto *const xp_val = this->llvm_eval_helper(s, 1, b0, fp_t, batch_size);

                        // Store the result.
                        bld.CreateStore(xp_val, retval);
                    },
                    [&]() {
                        // For order > 0, we return 0.
                        bld.CreateStore(llvm_codegen(s, val_t, number{0.}), retval);
                    });

                // Return the result.
                bld.CreateRet(bld.CreateLoad(val_t, retval));

                return f;
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor "
                                            "derivative of the derivative of an EOP/SW quantity in compact mode");
                // LCOV_EXCL_STOP
            }
        },
        this->args()[0].value());
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC eop_sw_p_impl<eop_data>;
template class HEYOKA_DLL_PUBLIC eop_sw_p_impl<sw_data>;

} // namespace model::detail

HEYOKA_END_NAMESPACE
