// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/eop_sw_helpers.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/sw.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

// Helper to get/generate the function for the computation of a sw quantity.
//
// fp_t is the scalar floating-point value that will be used in the computation. batch_size is the batch size.
// 'data' is the source of SW data. 'name' is the name of the sw quantity. sw_data_getter is the function to
// create/fetch the sw data.
llvm::Function *llvm_get_sw_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const sw_data &data,
                                 const char *name,
                                 llvm::Value *(*sw_data_getter)(llvm_state &, const sw_data &, llvm::Type *))
{
    assert(sw_data_getter != nullptr);

    namespace hy = heyoka;
    namespace hd = hy::detail;

    auto &md = s.module();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    // Fetch the table of SW data.
    const auto &table = data.get_table();

    // Start by creating the mangled name of the function. The mangled name will be based on:
    //
    // - the name of the sw quantity we are computing,
    // - the total number of rows in the sw data table,
    // - the timestamp and identifier of the sw data,
    // - the floating-point type.
    const auto fname = fmt::format("heyoka.get_{}.{}.{}_{}.{}", name, table.size(), data.get_timestamp(),
                                   data.get_identifier(), hd::llvm_mangle_type(val_t));

    // Check if we already created the function.
    if (auto *fptr = md.getFunction(fname)) {
        return fptr;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // Construct the function prototype. The only input is the time value, the output is value of the SW quantity.
    auto *ft = llvm::FunctionType::get(val_t, {val_t}, false);

    // Create the function
    auto *f = hd::llvm_func_create(ft, llvm::Function::PrivateLinkage, fname, &md);
    f->addFnAttr(llvm::Attribute::NoRecurse);
    f->addFnAttr(llvm::Attribute::NoUnwind);
    f->addFnAttr(llvm::Attribute::Speculatable);
    f->addFnAttr(llvm::Attribute::WillReturn);

    // Fetch the time argument.
    auto *tm_val = f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Get/generate the date and SW data.
    auto *date_ptr = hd::llvm_get_eop_sw_data_date_tt_cy_j2000(s, data, fp_t, "sw");
    auto *sw_ptr = sw_data_getter(s, data, fp_t);

    // Codegen the array size (and its splatted counterpart).
    auto *arr_size = bld.getInt32(boost::numeric_cast<std::uint32_t>(table.size()));
    auto *arr_size_splat = hd::vector_splat(bld, arr_size, batch_size);

    // Locate the index in date_ptr of the time interval containing tm_val.
    auto *idx = hd::llvm_eop_sw_data_locate_date(s, date_ptr, arr_size, tm_val);

    // Codegen nan for use later.
    auto *nan_const = llvm_codegen(s, val_t, number{std::numeric_limits<double>::quiet_NaN()});

    // We can now load the data from the sw array. The loaded data will be stored in the sw variable.
    llvm::Value *sw{};

    if (batch_size == 1u) {
        // Scalar implementation.

        // Storage for the sw value.
        auto *sw_alloc = bld.CreateAlloca(fp_t);

        // NOTE: in the scalar implementation, we need to branch on the value of idx: if idx == arr_size, we will return
        // NaN, otherwise we will return the value in the array at index idx.
        hd::llvm_if_then_else(
            s, bld.CreateICmpEQ(idx, arr_size),
            [&bld, nan_const, sw_alloc]() {
                // Store the nan.
                bld.CreateStore(nan_const, sw_alloc);
            },
            [&bld, idx, fp_t, sw_ptr, sw_alloc]() {
                // Load the sw value.
                bld.CreateStore(bld.CreateLoad(fp_t, bld.CreateInBoundsGEP(fp_t, sw_ptr, {idx})), sw_alloc);
            });

        // Fetch the value that we have stored in the alloc.
        sw = bld.CreateLoad(fp_t, sw_alloc);
    } else {
        // Vector implementation.

        // Fetch the alignment of the scalar type.
        const auto align = hd::get_alignment(md, fp_t);

        // Establish the SIMD lanes for which idx != arr_size. These are the lanes
        // we will use for the gather operation.
        auto *mask = bld.CreateICmpNE(idx, arr_size_splat);

        // Load the sw value, using nans as passhtru.
        sw = bld.CreateMaskedGather(val_t, bld.CreateInBoundsGEP(fp_t, sw_ptr, {idx}), llvm::Align(align), mask,
                                    nan_const);
    }

    // Create the return value.
    bld.CreateRet(sw);

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return f;
}

namespace
{

// Small wrapper to check that we have sw data to work with in the sw
// implementations. It should never happen that we end up throwing here while
// using the public API, but better safe than sorry.
void sw_check_sw_data(const std::optional<sw_data> &odata)
{
    if (!odata) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument("Error: missing SW data");
        // LCOV_EXCL_STOP
    }
}

// NOTE: here we are essentially implementing a string-based vtable to dispatch the implementation details of sw_impl.
// This allows us to minimise the amount of boilerplate with respect to a solution based, e.g., on
// standard OOP patterns.
using sw_impl_func_t = llvm::Function *(*)(llvm_state &, llvm::Type *, std::uint32_t, const sw_data &);

// NOLINTNEXTLINE(cert-err58-cpp)
const std::unordered_map<std::string, sw_impl_func_t> sw_impl_funcs_map
    = {{"Ap_avg",
        [](llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const sw_data &data) {
            return llvm_get_sw_func(s, fp_t, batch_size, data, "Ap_avg", &heyoka::detail::llvm_get_sw_data_Ap_avg);
        }},
       {"f107",
        [](llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const sw_data &data) {
            return llvm_get_sw_func(s, fp_t, batch_size, data, "f107", &heyoka::detail::llvm_get_sw_data_f107);
        }},
       {"f107a_center81", [](llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, const sw_data &data) {
            return llvm_get_sw_func(s, fp_t, batch_size, data, "f107a_center81",
                                    &heyoka::detail::llvm_get_sw_data_f107a_center81);
        }}};

} // namespace

void sw_impl::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<func_base>(*this);
    oa << m_sw_name;
    oa << m_sw_data;
}

void sw_impl::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<func_base>(*this);
    ia >> m_sw_name;
    ia >> m_sw_data;
}

// NOTE: this will be used only during serialisation.
sw_impl::sw_impl() : func_base("sw_undefined", {heyoka::time}) {}

sw_impl::sw_impl(std::string name, expression time_expr, sw_data data)
    // NOTE: we must be careful here with the name mangling. In order to match the
    // mangling for the sw data arrays, we must include:
    //
    // - the total number of rows in the sw data table,
    // - the timestamp and identifier of the sw data.
    //
    // If we do not do that, we risk in principle having functions with the same
    // name using different sw data.
    : func_base(
          fmt::format("sw_{}_{}_{}_{}", name, data.get_table().size(), data.get_timestamp(), data.get_identifier()),
          {std::move(time_expr)}),
      m_sw_name(std::move(name)), m_sw_data(std::move(data))
{
}

std::vector<expression> sw_impl::gradient() const
{
    sw_check_sw_data(m_sw_data);

    return {0_dbl};
}

namespace
{

// Small wrapper for use in the implementation of the llvm evaluation of a sw quantity.
llvm::Value *llvm_sw_eval_helper(llvm_state &s, llvm::Value *arg, llvm::Type *fp_t, std::uint32_t batch_size,
                                 const sw_data &data, const std::string &sw_name)
{
    // Fetch/create the function for the computation of the sw quantity.
    auto *sw_f = sw_impl_funcs_map.at(sw_name)(s, fp_t, batch_size, data);

    // Invoke it and return the result.
    auto &bld = s.builder();
    return bld.CreateCall(sw_f, arg);
}

} // namespace

llvm::Value *sw_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                bool high_accuracy) const
{
    sw_check_sw_data(m_sw_data);

    return heyoka::detail::llvm_eval_helper(
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        [this, &s, fp_t, batch_size, &data = *m_sw_data](const std::vector<llvm::Value *> &args, bool) {
            return llvm_sw_eval_helper(s, args[0], fp_t, batch_size, data, m_sw_name);
        },
        *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *sw_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                          bool high_accuracy) const
{
    sw_check_sw_data(m_sw_data);

    return heyoka::detail::llvm_c_eval_func_helper(
        get_name(),
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        [this, &s, fp_t, batch_size, &data = *m_sw_data](const std::vector<llvm::Value *> &args, bool) {
            return llvm_sw_eval_helper(s, args[0], fp_t, batch_size, data, m_sw_name);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

namespace
{

// Derivative of sw(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Value *taylor_diff_sw_impl(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &, const U &num,
                                 const std::vector<llvm::Value *> &,
                                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                 llvm::Value *par_ptr, std::uint32_t, std::uint32_t order, std::uint32_t batch_size,
                                 const sw_data &data, const std::string &sw_name)
{
    namespace hd = heyoka::detail;

    if (order == 0u) {
        return llvm_sw_eval_helper(s, hd::taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size), fp_t, batch_size,
                                   data, sw_name);
    } else {
        return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of sw(variable).
//
// NOTE: the derivatives of sw() are all zero (except of course for the order-0 derivative).
llvm::Value *taylor_diff_sw_impl(llvm_state &s, llvm::Type *fp_t, const variable &var,
                                 const std::vector<llvm::Value *> &arr, llvm::Value *,
                                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                 std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                                 const sw_data &data, const std::string &sw_name)
{
    namespace hd = heyoka::detail;

    if (order == 0u) {
        // Fetch the index of the variable.
        const auto b_idx = hd::uname_to_index(var.name());

        // Load b^[0].
        auto *b0 = hd::taylor_fetch_diff(arr, b_idx, 0, n_uvars);

        // Evaluate the sw and return it.
        return llvm_sw_eval_helper(s, b0, fp_t, batch_size, data, sw_name);
    } else {
        // Return zero.
        return hd::vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Value *taylor_diff_sw_impl(llvm_state &, llvm::Type *, const U &, const std::vector<llvm::Value *> &,
                                 llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t, const sw_data &,
                                 const std::string &)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of a sw quantity");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Value *sw_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  std::uint32_t n_uvars, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                  bool) const
{
    assert(args().size() == 1u);

    sw_check_sw_data(m_sw_data);

    return std::visit(
        [&](const auto &v) {
            return taylor_diff_sw_impl(s, fp_t, v, arr, par_ptr, n_uvars, order, batch_size, *m_sw_data, m_sw_name);
        },
        args()[0].value());
}

namespace
{

// Derivative of sw(number).
template <typename U>
    requires(heyoka::detail::is_num_param_v<U>)
llvm::Function *taylor_c_diff_func_sw_impl(llvm_state &s, llvm::Type *fp_t, const sw_impl &fn, const U &num,
                                           std::uint32_t n_uvars, std::uint32_t batch_size, const sw_data &data,
                                           const std::string &sw_name)
{
    return heyoka::detail::taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, fn.get_name(), 0,
        [&s, fp_t, batch_size, &data, &sw_name](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 1u);
            assert(args[0] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_sw_eval_helper(s, args[0], fp_t, batch_size, data, sw_name);
        },
        num);
}

// Derivative of sw(variable).
llvm::Function *taylor_c_diff_func_sw_impl(llvm_state &s, llvm::Type *fp_t, const sw_impl &fn, const variable &var,
                                           std::uint32_t n_uvars, std::uint32_t batch_size, const sw_data &data,
                                           const std::string &sw_name)
{
    namespace hd = heyoka::detail;

    auto &md = s.module();
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = hd::make_vector_type(fp_t, batch_size);

    const auto na_pair = hd::taylor_c_diff_func_name_args(ctx, fp_t, fn.get_name(), n_uvars, batch_size, {var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);
    if (f != nullptr) {
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // The return type is val_t.
    auto *ft = llvm::FunctionType::get(val_t, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *ord = f->args().begin();
    auto *diff_ptr = f->args().begin() + 2;
    auto *b_idx = f->args().begin() + 5;

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Create the return value.
    auto *retval = bld.CreateAlloca(val_t);

    hd::llvm_if_then_else(
        s, bld.CreateICmpEQ(ord, bld.getInt32(0)),
        [&]() {
            // For order 0, compute the sw for the order 0 of b_idx.

            // Load b^[0].
            auto *b0 = hd::taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, bld.getInt32(0), b_idx);

            // Compute the sw.
            auto *sw_val = llvm_sw_eval_helper(s, b0, fp_t, batch_size, data, sw_name);

            // Store the result.
            bld.CreateStore(sw_val, retval);
        },
        [&]() {
            // For order > 0, we return 0.
            bld.CreateStore(llvm_codegen(s, val_t, number{0.}), retval);
        });

    // Return the result.
    bld.CreateRet(bld.CreateLoad(val_t, retval));

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return f;
}

// LCOV_EXCL_START

// All the other cases.
template <typename U>
llvm::Function *taylor_c_diff_func_sw_impl(llvm_state &, llvm::Type *, const sw_impl &, const U &, std::uint32_t,
                                           std::uint32_t, const sw_data &, const std::string &)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of an sw quantity in compact mode");
}

// LCOV_EXCL_STOP

} // namespace

llvm::Function *sw_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                            std::uint32_t batch_size, bool) const
{
    assert(args().size() == 1u);

    sw_check_sw_data(m_sw_data);

    return std::visit(
        [&](const auto &v) {
            return taylor_c_diff_func_sw_impl(s, fp_t, *this, v, n_uvars, batch_size, *m_sw_data, m_sw_name);
        },
        args()[0].value());
}

expression Ap_avg_func_impl(expression time_expr, sw_data data)
{
    return expression{func{sw_impl{"Ap_avg", std::move(time_expr), std::move(data)}}};
}

expression f107_func_impl(expression time_expr, sw_data data)
{
    return expression{func{sw_impl{"f107", std::move(time_expr), std::move(data)}}};
}

expression f107a_center81_func_impl(expression time_expr, sw_data data)
{
    return expression{func{sw_impl{"f107a_center81", std::move(time_expr), std::move(data)}}};
}

} // namespace model::detail

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::model::detail::sw_impl)
