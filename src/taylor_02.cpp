// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/ranges_to.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

// NOTE: GCC warns about use of mismatched new/delete
// when creating global variables. I am not sure this is
// a real issue, as it looks like we are adopting the "canonical"
// approach for the creation of global variables (at least
// according to various sources online)
// and clang is not complaining. But let us revisit
// this issue in later LLVM versions.
#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Function to split the central part of the decomposition (i.e., the definitions of the u variables that do not
// represent state variables) into segments. Within a segment, the definition of a u variable does not depend on any
// other u variable defined within that segment.
//
// The goal of this segmentation is to group together expressions whose ordered iterations for the computation of the
// Taylor derivatives can be performed independently of each other. This allows for a compact codegen of the
// computation.
//
// NOTE: the hidden deps do not need to be considered as dependencies, because in the Taylor derivatives formulae we
// only access previous-orders derivatives of the hidden deps, never same-order derivatives.
std::vector<taylor_dc_t> taylor_segment_dc(const taylor_dc_t &dc, std::uint32_t n_eq)
{
    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Helper that takes in input the definition ex of a u variable, and returns
    // in output the list of indices of the u variables on which ex depends.
    const auto udef_args_indices = [](const expression &ex) -> std::vector<std::uint32_t> {
        return std::visit(
            [](const auto &v) -> std::vector<std::uint32_t> {
                using type = std::remove_cvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    std::vector<std::uint32_t> retval;

                    for (const auto &arg : v.args()) {
                        std::visit(
                            [&retval](const auto &x) {
                                using tp = std::remove_cvref_t<decltype(x)>;

                                if constexpr (std::is_same_v<tp, variable>) {
                                    retval.push_back(uname_to_index(x.name()));
                                } else if constexpr (!std::is_same_v<tp, number> && !std::is_same_v<tp, param>) {
                                    throw std::invalid_argument(
                                        "Invalid argument encountered in an element of a Taylor decomposition: the "
                                        "argument is not a variable or a number/param");
                                }
                            },
                            arg.value());
                    }

                    return retval;
                } else {
                    throw std::invalid_argument("Invalid expression encountered in a Taylor decomposition: the "
                                                "expression is not a function");
                }
            },
            ex.value());
    };

    // Init the return value.
    std::vector<taylor_dc_t> s_dc;

    // cur_limit_idx is initially the index of the first
    // u variable which is not a state variable.
    auto cur_limit_idx = n_eq;
    for (std::uint32_t i = n_eq; i < dc.size() - n_eq; ++i) {
        // NOTE: at the very first iteration of this for loop,
        // no block has been created yet. Do it now.
        if (i == n_eq) {
            assert(s_dc.empty());
            s_dc.emplace_back();
        } else {
            assert(!s_dc.empty());
        }

        const auto &[ex, deps] = dc[i];

        // Determine the u indices on which ex depends.
        const auto u_indices = udef_args_indices(ex);

        if (std::ranges::any_of(u_indices, [cur_limit_idx](auto idx) { return idx >= cur_limit_idx; })) {
            // The current expression depends on one or more variables
            // within the current block. Start a new block and
            // update cur_limit_idx with the start index of the new block.
            s_dc.emplace_back();
            cur_limit_idx = i;
        }

        // Append ex to the current block.
        s_dc.back().emplace_back(ex, deps);
    }

#if !defined(NDEBUG)

    // Verify s_dc.
    decltype(dc.size()) counter = 0;
    for (const auto &s : s_dc) {
        // No segment can be empty.
        assert(!s.empty());

        for (const auto &[ex, _] : s) {
            // All the indices in the definitions of the
            // u variables in the current block must be
            // less than counter + n_eq (which is the starting
            // index of the block).
            const auto u_indices = udef_args_indices(ex);
            assert(std::ranges::all_of(u_indices, [idx_limit = counter + n_eq](auto idx) { return idx < idx_limit; }));
        }

        // Update the counter.
        counter += s.size();
    }

    assert(counter == dc.size() - static_cast<decltype(dc.size())>(n_eq) * 2u);

#endif

    get_logger()->debug("Taylor decomposition N of segments: {}", s_dc.size());
    get_logger()->trace("Taylor decomposition segment runtime: {}", sw);

    return s_dc;
}

} // namespace

// Create a global read-only array containing the values in sv_funcs_dc, if any
// (otherwise, the return value will be null). This is for use in the adaptive steppers
// when employing compact mode.
llvm::Value *taylor_c_make_sv_funcs_arr(llvm_state &s, const std::vector<std::uint32_t> &sv_funcs_dc)
{
    auto &builder = s.builder();

    if (sv_funcs_dc.empty()) {
        return nullptr;
    } else {
        auto *arr_type
            = llvm::ArrayType::get(builder.getInt32Ty(), boost::numeric_cast<std::uint64_t>(sv_funcs_dc.size()));
        std::vector<llvm::Constant *> sv_funcs_dc_const;
        sv_funcs_dc_const.reserve(sv_funcs_dc.size());
        for (auto idx : sv_funcs_dc) {
            sv_funcs_dc_const.emplace_back(builder.getInt32(idx));
        }
        auto *sv_funcs_dc_arr = llvm::ConstantArray::get(arr_type, sv_funcs_dc_const);
        auto *g_sv_funcs_dc = new llvm::GlobalVariable(s.module(), sv_funcs_dc_arr->getType(), true,
                                                       llvm::GlobalVariable::PrivateLinkage, sv_funcs_dc_arr);

        // Get out a pointer to the beginning of the array.
        return builder.CreateInBoundsGEP(arr_type, g_sv_funcs_dc, {builder.getInt32(0), builder.getInt32(0)});
    }
}

namespace
{

// Compute the derivative of order "order" of a state variable.
// ex is the formula for the first-order derivative of the state variable (which
// is either a u variable or a number/param), n_uvars the number of variables in
// the decomposition, arr the array containing the derivatives of all u variables
// up to order - 1.
llvm::Value *taylor_compute_sv_diff(llvm_state &s, llvm::Type *fp_t, const expression &ex,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                    std::uint32_t order, std::uint32_t batch_size)
{
    assert(order > 0u);

    auto &builder = s.builder();

    return std::visit(
        [&](const auto &v) -> llvm::Value * {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, variable>) {
                // Extract the index of the u variable in the expression
                // of the first-order derivative.
                const auto u_idx = uname_to_index(v.name());

                // Fetch from arr the derivative of order 'order - 1' of the u variable at u_idx.
                auto ret = taylor_fetch_diff(arr, u_idx, order - 1u, n_uvars);

                // We have to divide the derivative by order to get the normalised derivative of the state variable.
                return llvm_fdiv(
                    s, ret,
                    vector_splat(builder, llvm_codegen(s, fp_t, number(static_cast<double>(order))), batch_size));
            } else if constexpr (std::is_same_v<type, number> || std::is_same_v<type, param>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0. No need for normalization as the only
                // nonzero value that can be produced here is the first-order
                // derivative.
                if (order == 1u) {
                    return taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size);
                } else {
                    return llvm_constantfp(s, make_vector_type(fp_t, batch_size), 0.);
                }
            } else {
                assert(false);
                return nullptr;
            }
        },
        ex.value());
}

// Helper to construct the global arrays needed for the computation of the
// derivatives of the state variables in compact mode. The first part of the
// return value is a set of 6 arrays:
// - the indices of the state variables whose time derivative is a u variable, paired to
// - the indices of the u variables appearing in the derivatives, and
// - the indices of the state variables whose time derivative is a constant, paired to
// - the values of said constants, and
// - the indices of the state variables whose time derivative is a param, paired to
// - the indices of the params.
// The second part of the return value is a boolean flag that will be true if
// the time derivatives of all state variables are u variables, false otherwise.
std::pair<std::array<llvm::GlobalVariable *, 6>, bool>
taylor_c_make_sv_diff_globals(llvm_state &s, llvm::Type *fp_t, const taylor_dc_t &dc, std::uint32_t n_uvars)
{
    auto &context = s.context();
    auto &builder = s.builder();
    auto &md = s.module();

    // Build iteratively the output values as vectors of constants.
    std::vector<llvm::Constant *> var_indices, vars, num_indices, nums, par_indices, pars;

    // Keep track of how many time derivatives
    // of the state variables are u variables.
    std::uint32_t n_der_vars = 0;

    // NOTE: the derivatives of the state variables are at the end of the decomposition.
    for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
        std::visit(
            [&](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    ++n_der_vars;
                    // NOTE: remove from i the n_uvars offset to get the
                    // true index of the state variable.
                    var_indices.push_back(builder.getInt32(i - n_uvars));
                    vars.push_back(builder.getInt32(uname_to_index(v.name())));
                } else if constexpr (std::is_same_v<type, number>) {
                    num_indices.push_back(builder.getInt32(i - n_uvars));
                    nums.push_back(llvm::cast<llvm::Constant>(llvm_codegen(s, fp_t, v)));
                } else if constexpr (std::is_same_v<type, param>) {
                    par_indices.push_back(builder.getInt32(i - n_uvars));
                    pars.push_back(builder.getInt32(v.idx()));
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].first.value());
    }

    // Flag to signal that the time derivatives of all state variables are u variables.
    assert(dc.size() >= n_uvars); // LCOV_EXCL_LINE
    const auto all_der_vars = (n_der_vars == (dc.size() - n_uvars));

    assert(var_indices.size() == vars.size()); // LCOV_EXCL_LINE
    assert(num_indices.size() == nums.size()); // LCOV_EXCL_LINE
    assert(par_indices.size() == pars.size()); // LCOV_EXCL_LINE

    // Turn the vectors into global read-only LLVM arrays.

    // Variables.
    auto *var_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(var_indices.size()));

    auto *var_indices_arr = llvm::ConstantArray::get(var_arr_type, var_indices);
    auto *g_var_indices = new llvm::GlobalVariable(md, var_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::PrivateLinkage, var_indices_arr);

    auto *vars_arr = llvm::ConstantArray::get(var_arr_type, vars);
    auto *g_vars
        = new llvm::GlobalVariable(md, vars_arr->getType(), true, llvm::GlobalVariable::PrivateLinkage, vars_arr);

    // Numbers.
    auto *num_indices_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(num_indices.size()));
    auto *num_indices_arr = llvm::ConstantArray::get(num_indices_arr_type, num_indices);
    auto *g_num_indices = new llvm::GlobalVariable(md, num_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::PrivateLinkage, num_indices_arr);

    auto *nums_arr_type = llvm::ArrayType::get(fp_t, boost::numeric_cast<std::uint64_t>(nums.size()));
    auto *nums_arr = llvm::ConstantArray::get(nums_arr_type, nums);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *g_nums
        = new llvm::GlobalVariable(md, nums_arr->getType(), true, llvm::GlobalVariable::PrivateLinkage, nums_arr);

    // Params.
    auto *par_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(par_indices.size()));

    auto *par_indices_arr = llvm::ConstantArray::get(par_arr_type, par_indices);
    auto *g_par_indices = new llvm::GlobalVariable(md, par_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::PrivateLinkage, par_indices_arr);

    auto *pars_arr = llvm::ConstantArray::get(par_arr_type, pars);
    auto *g_pars
        = new llvm::GlobalVariable(md, pars_arr->getType(), true, llvm::GlobalVariable::PrivateLinkage, pars_arr);

    return std::pair{std::array{g_var_indices, g_vars, g_num_indices, g_nums, g_par_indices, g_pars}, all_der_vars};
}

// Helper to compute and store the derivatives of the state variables in compact mode at order 'order'.
// svd_gl is the return value of taylor_c_make_sv_diff_globals(), which contains
// the indices/constants necessary for the computation.
void taylor_c_compute_sv_diffs(llvm_state &s, llvm::Type *fp_t,
                               const std::pair<std::array<llvm::GlobalVariable *, 6>, bool> &svd_gl,
                               llvm::Value *diff_arr, llvm::Value *par_ptr, std::uint32_t n_uvars, llvm::Value *order,
                               std::uint32_t batch_size)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    // Fetch the global arrays and
    // the all_der_vars flag.
    const auto &sv_diff_gl = svd_gl.first;
    const auto all_der_vars = svd_gl.second;

    auto &builder = s.builder();

    // Recover the number of state variables whose derivatives are given
    // by u variables, numbers and params.
    const auto n_vars = gl_arr_size(sv_diff_gl[0]);
    const auto n_nums = gl_arr_size(sv_diff_gl[2]);
    const auto n_pars = gl_arr_size(sv_diff_gl[4]);

    // Fetch the vector type.
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);

    // Handle the u variables definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_vars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        // NOTE: if the time derivatives of all state variables are u variables, there's
        // no need to lookup the index in the global array (which will just contain
        // the values in the [0, n_vars] range).
        auto *sv_idx = all_der_vars
                           ? cur_idx
                           : builder.CreateLoad(builder.getInt32Ty(),
                                                builder.CreateInBoundsGEP(sv_diff_gl[0]->getValueType(), sv_diff_gl[0],
                                                                          {builder.getInt32(0), cur_idx}));

        // Fetch the index of the u variable.
        auto *u_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(sv_diff_gl[1]->getValueType(), sv_diff_gl[1], {builder.getInt32(0), cur_idx}));

        // Fetch from diff_arr the derivative of order 'order - 1' of the u variable u_idx.
        auto *ret
            = taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars, builder.CreateSub(order, builder.getInt32(1)), u_idx);

        // We have to divide the derivative by 'order' in order
        // to get the normalised derivative of the state variable.
        ret = llvm_fdiv(s, ret, vector_splat(builder, llvm_ui_to_fp(s, order, fp_vec_t->getScalarType()), batch_size));

        // Store the derivative.
        taylor_c_store_diff(s, fp_vec_t, diff_arr, n_uvars, order, sv_idx, ret);
    });

    // Handle the number definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_nums), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        auto *sv_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(sv_diff_gl[2]->getValueType(), sv_diff_gl[2], {builder.getInt32(0), cur_idx}));

        // Fetch the constant.
        auto *num = builder.CreateLoad(
            fp_vec_t->getScalarType(),
            builder.CreateInBoundsGEP(sv_diff_gl[3]->getValueType(), sv_diff_gl[3], {builder.getInt32(0), cur_idx}));

        // If the first-order derivative is being requested,
        // do the codegen for the constant itself, otherwise
        // return 0. No need for normalization as the only
        // nonzero value that can be produced here is the first-order
        // derivative.
        auto *cmp_cond = builder.CreateICmpEQ(order, builder.getInt32(1));
        auto *ret
            = builder.CreateSelect(cmp_cond, vector_splat(builder, num, batch_size), llvm_constantfp(s, fp_vec_t, 0.));

        // Store the derivative.
        taylor_c_store_diff(s, fp_vec_t, diff_arr, n_uvars, order, sv_idx, ret);
    });

    // Handle the param definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_pars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        auto *sv_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(sv_diff_gl[4]->getValueType(), sv_diff_gl[4], {builder.getInt32(0), cur_idx}));

        // Fetch the index of the param.
        auto *par_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(sv_diff_gl[5]->getValueType(), sv_diff_gl[5], {builder.getInt32(0), cur_idx}));

        // If the first-order derivative is being requested,
        // do the codegen for the constant itself, otherwise
        // return 0. No need for normalization as the only
        // nonzero value that can be produced here is the first-order
        // derivative.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(1)),
            [&]() {
                // Derivative of order 1. Fetch the value from par_ptr.
                // NOTE: param{0} is unused, its only purpose is type tagging.
                taylor_c_store_diff(s, fp_vec_t, diff_arr, n_uvars, order, sv_idx,
                                    taylor_c_diff_numparam_codegen(s, fp_t, param{0}, par_idx, par_ptr, batch_size));
            },
            [&]() {
                // Derivative of order > 1, return 0.
                taylor_c_store_diff(s, fp_vec_t, diff_arr, n_uvars, order, sv_idx, llvm_constantfp(s, fp_vec_t, 0.));
            });
    });
}

// Helper to create and return the prototype of a driver function for
// the computation of Taylor derivatives in compact mode. s is the llvm state
// in which we are operating, cur_idx the index of the driver.
llvm::Function *taylor_cm_make_driver_proto(llvm_state &s, unsigned cur_idx)
{
    auto &builder = s.builder();
    auto &md = s.module();
    auto &ctx = s.context();

    // The arguments to the driver are:
    // - a pointer to the tape,
    // - pointers to par and time,
    // - the current diff order.
    auto *ptr_tp = llvm::PointerType::getUnqual(ctx);
    const std::vector<llvm::Type *> fargs{ptr_tp, ptr_tp, ptr_tp, builder.getInt32Ty()};

    // The driver does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE

    // Now create the driver.
    const auto cur_name = fmt::format("heyoka.cm_jet.driver_{}", cur_idx);
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, cur_name, &md);
    // NOTE: the driver cannot call itself recursively.
    f->addFnAttr(llvm::Attribute::NoRecurse);

    // Add the arguments' attributes.
    // NOTE: no aliasing is assumed between the pointer
    // arguments.
    auto *tape_arg = f->args().begin();
    tape_arg->setName("tape_ptr");
    tape_arg->addAttr(llvm::Attribute::NoCapture);
    tape_arg->addAttr(llvm::Attribute::NoAlias);

    auto *par_ptr_arg = tape_arg + 1;
    par_ptr_arg->setName("par_ptr");
    par_ptr_arg->addAttr(llvm::Attribute::NoCapture);
    par_ptr_arg->addAttr(llvm::Attribute::NoAlias);
    par_ptr_arg->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr_arg = tape_arg + 2;
    time_ptr_arg->setName("time_ptr");
    time_ptr_arg->addAttr(llvm::Attribute::NoCapture);
    time_ptr_arg->addAttr(llvm::Attribute::NoAlias);
    time_ptr_arg->addAttr(llvm::Attribute::ReadOnly);

    auto *order_arg = tape_arg + 3;
    order_arg->setName("order");

    return f;
}

// Helper to codegen the computation of the Taylor derivatives for a block.
//
// s is the llvm state in which we are operating, func is the LLVM function for the computation of the Taylor
// derivatives in the block, ncalls the number of times it must be called, gens the generators for the function
// arguments, tape/par/time_ptr the pointers to the tape/parameter value(s)/time value(s), cur_order the order of the
// derivative, fp_vec_type the internal vector type used for computations, n_uvars the total number of u variables,
// oiter_rng_arr a pointer to the first element of the ordered iteration ranges for the function func.
void taylor_cm_codegen_block_diff(llvm_state &s, llvm::Function *func, std::uint32_t ncalls, const auto &gens,
                                  llvm::Value *tape_ptr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                  llvm::Value *cur_order, llvm::Type *fp_vec_type, std::uint32_t n_uvars,
                                  llvm::Value *oiter_rng_arr)
{
    // LCOV_EXCL_START
    assert(ncalls > 0u);
    assert(!gens.empty());
    assert(std::ranges::all_of(gens, [](const auto &f) { return static_cast<bool>(f); }));
    // LCOV_EXCL_STOP

    // We will be manually unrolling loops if ncalls is small enough.
    // This seems to help with compilation times.
    constexpr auto max_unroll_n = 5u;

    // Fetch the builder for the current state.
    auto &bld = s.builder();

    // Fetch the size-2 array type used to represent an iteration range.
    auto *i32_t = bld.getInt32Ty();
    auto *i32_pair_t = llvm::ArrayType::get(i32_t, 2);

    // Load the begin/end values from the range.
    // NOTE: here we are using a GEP with 2 indices: the first one gets us to the cur_order-th range, the second one
    // fetches the first element of the pair.
    auto *begin_ptr = bld.CreateInBoundsGEP(i32_pair_t, oiter_rng_arr, {cur_order, bld.getInt32(0)});
    auto *end_ptr = bld.CreateInBoundsGEP(i32_pair_t, oiter_rng_arr, {cur_order, bld.getInt32(1)});
    auto *begin = bld.CreateLoad(i32_t, begin_ptr);
    auto *end = bld.CreateLoad(i32_t, end_ptr);

    // Loop over the ordered iteration indices.
    llvm_loop_u32(s, begin, end, [&](llvm::Value *cur_oiter_idx) {
        if (ncalls > max_unroll_n) {
            // Loop over the number of calls.
            llvm_loop_u32(s, bld.getInt32(0), bld.getInt32(ncalls), [&](llvm::Value *cur_call_idx) {
                // Create the u variable index from the first generator.
                auto *u_idx = gens[0](cur_call_idx);

                // Initialise the vector of arguments with which func must be called. The following
                // initial arguments are always present:
                // - current Taylor order,
                // - u index of the variable,
                // - tape of derivatives,
                // - pointer to the param values,
                // - pointer to the time value(s).
                std::vector<llvm::Value *> args{cur_order, u_idx, tape_ptr, par_ptr, time_ptr};

                // Create the arguments from the other generators.
                for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                    args.push_back(gens[i](cur_call_idx));
                }

                // Compute the pointer to the accumulator.
                auto *acc_ptr = bld.CreateInBoundsGEP(
                    fp_vec_type, tape_ptr, {bld.CreateAdd(bld.CreateMul(cur_order, bld.getInt32(n_uvars)), u_idx)});

                // Add it to the list of arguments.
                args.push_back(acc_ptr);

                // Add the current iteration index to the list of arguments.
                args.push_back(cur_oiter_idx);

                // Perform a single iteration of the calculation of the derivative.
                bld.CreateCall(func, args);
            });
        } else {
            // The manually-unrolled version of the above.
            for (std::uint32_t idx = 0; idx < ncalls; ++idx) {
                auto *cur_call_idx = bld.getInt32(idx);

                const auto u_idx = gens[0](cur_call_idx);

                std::vector<llvm::Value *> args{cur_order, u_idx, tape_ptr, par_ptr, time_ptr};

                for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                    args.push_back(gens[i](cur_call_idx));
                }

                auto *acc_ptr = bld.CreateInBoundsGEP(
                    fp_vec_type, tape_ptr, {bld.CreateAdd(bld.CreateMul(cur_order, bld.getInt32(n_uvars)), u_idx)});
                args.push_back(acc_ptr);
                args.push_back(cur_oiter_idx);

                bld.CreateCall(func, args);
            }
        }
    });
}

// This map contains a list of functions for the calculation in compact-mode of Taylor derivatives. Each function is
// mapped to a pair, containing:
//
// - the number of times the function is to be invoked,
// - a list of function objects (aka, the generators) that generate the arguments for the invocation.
//
// NOTE: we use maps with name-based comparison for the functions. This ensures that the order in which these functions
// are invoked is always the same. If we used directly pointer comparisons instead, the order could vary across
// different executions and different platforms. The name mangling we do when creating the function names should ensure
// that there are no possible name collisions.
using taylor_cm_seg_f_list_t
    = std::map<llvm::Function *, std::pair<std::uint32_t, std::vector<std::function<llvm::Value *(llvm::Value *)>>>,
               llvm_func_name_compare>;

// This structure maps a function for the computation of a Taylor derivative in compact mode to an LLVM pointer to the
// beginning of the array of ordered iteration ranges for that function.
//
// The array has size order + 1 and it contains arrays of size 2, representing (for each differentiation order) the
// half-open range of ordered iteration indices.
//
// A global instance of this map is shared amongst several states: if the same function/array pair is defined in
// multiple states, it will show up multiple times in this map.
using taylor_cm_f_oiter_rng_map_t = boost::unordered_flat_map<llvm::Function *, llvm::Value *>;

// TODO fix docs.
// Helper to codegen the computation of the Taylor derivatives in compact mode for a segment.
//
// s is the llvm state in which we are operating, fp_vec_type the internal vector type we are using
// for computations, seg_map is the taylor_cm_seg_f_list_t containing the list of functions for the computation
// of Taylor derivatives within a segment, n_uvars the total number of u variables in the decomposition.
void taylor_cm_codegen_segment_diff_impl(llvm_state &s, llvm::Type *fp_vec_type, const taylor_cm_seg_f_list_t &seg_map,
                                         std::uint32_t n_uvars, const taylor_cm_f_oiter_rng_map_t &oiter_map)
{
    // Fetch the current builder.
    auto &bld = s.builder();

    // Fetch the arguments from the driver prototype.
    auto *driver_f = bld.GetInsertBlock()->getParent();
    assert(driver_f != nullptr);
    assert(driver_f->arg_size() == 4u);
    auto *tape_ptr = driver_f->args().begin();
    auto *par_ptr = driver_f->args().begin() + 1;
    auto *time_ptr = driver_f->args().begin() + 2;
    auto *cur_order = driver_f->args().begin() + 3;

    // Generate the code for the computation of the derivatives for this segment.
    for (const auto &[func, fpair] : seg_map) {
        const auto &[ncalls, gens] = fpair;

        // Fetch the pointer to the array of ordered iteration ranges for the function.
        assert(oiter_map.contains(func));
        auto *oiter_rng_arr = oiter_map.find(func)->second;

        taylor_cm_codegen_block_diff(s, func, ncalls, gens, tape_ptr, par_ptr, time_ptr, cur_order, fp_vec_type,
                                     n_uvars, oiter_rng_arr);
    }
}

// Helper to generate the code for the computation of the Taylor derivatives for a segment in compact mode.
//
// TODO remove parallel_flag.
// seg is the segment, start_u_idx the index of the first u variable in the segment, s the llvm state we are operating
// in, fp_t the internal scalar floating-point type, batch_size the batch size, n_uvars the total number of u variables,
// high_accuracy the high accuracy flag, oiter_map is the map of ordered iteration ranges, order is the maximum Taylor
// differentiation order.
taylor_cm_seg_f_list_t taylor_cm_codegen_segment_diff(const auto &seg, std::uint32_t start_u_idx, llvm_state &s,
                                                      llvm::Type *fp_t, std::uint32_t batch_size, std::uint32_t n_uvars,
                                                      bool high_accuracy, bool parallel_mode,
                                                      taylor_cm_f_oiter_rng_map_t &oiter_map, std::uint32_t order)
{
    auto &bld = s.builder();
    auto &md = s.module();

    // Fetch the size-2 array type used to represent an iteration range.
    auto *i32_pair_t = llvm::ArrayType::get(bld.getInt32Ty(), 2);

    // Fetch the type for the array of ordered iteration ranges.
    auto *rng_arr_t = llvm::ArrayType::get(i32_pair_t, order + 1u);

    // Fetch the internal vector type.
    auto *fp_vec_type = make_vector_type(fp_t, batch_size);

    // This structure maps a function to sets of arguments with which the function is to be called. For instance, if
    // function f(x, y, z) is to be called as f(a, b, c) and f(d, e, f), then tmp_map will contain {f : [[a, b, c], [d,
    // e, f]]}. After construction, we have verified that for each function in the map the sets of arguments have all
    // the same size.
    //
    // NOTE: again, here and below we use name-based ordered maps for the functions. This ensures that the invocations
    // of cm_make_arg_gen_*(), which create several global variables, always happen in a well-defined order. If we used
    // an unordered map instead, the variables would be created in a "random" order, which would result in a unnecessary
    // miss for the in-memory cache machinery when two logically-identical LLVM modules are considered different because
    // of the difference in the order of declaration of global variables.
    std::map<llvm::Function *, std::vector<std::vector<std::variant<std::uint32_t, number>>>, llvm_func_name_compare>
        tmp_map;

    for (const auto &[ex, deps] : seg) {
        // Get the function for a single iteration of the computation of the derivative.
        assert(std::holds_alternative<func>(ex.value()));
        const auto &f = std::get<func>(ex.value());
        auto *func = f.taylor_c_diff_get_single_iter_func(s, fp_t, n_uvars, batch_size, high_accuracy);

        // Insert the function into tmp_map.
        const auto [it, new_in_tmp_map] = tmp_map.try_emplace(func);

        assert(new_in_tmp_map || !it->second.empty()); // LCOV_EXCL_LINE

        // Convert the variables/constants in the current dc
        // element into a set of indices/constants.
        const auto cdiff_args = udef_to_variants(ex, deps);

        // LCOV_EXCL_START
        if (!new_in_tmp_map && it->second.back().size() - 1u != cdiff_args.size()) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Inconsistent arity detected in a Taylor derivative function in compact "
                            "mode: the same function is being called with both {} and {} arguments",
                            it->second.back().size() - 1u, cdiff_args.size()));
        }
        // LCOV_EXCL_STOP

        // Add the new set of arguments.
        it->second.emplace_back();
        // Add the idx of the u variable.
        it->second.back().emplace_back(start_u_idx);
        // Add the actual function arguments.
        it->second.back().insert(it->second.back().end(), cdiff_args.begin(), cdiff_args.end());

        // Update start_u_idx.
        ++start_u_idx;

        // Add the function to oiter_map, if necessary.
        const auto [o_it, new_in_oiter_map] = oiter_map.try_emplace(func);
        if (new_in_oiter_map) {
            // func is a new function (i.e., it does not show up in oiter_map). Fetch its number of unordered/ordered
            // iterations.
            const auto n_iters = f.taylor_c_diff_get_n_iters(order);

            // Build a view to transform the pairs in n_iters into ordered iteration ranges. I.e., if we have a pair (n,
            // m), this will be transformed into the pair (n, n + m).
            auto rng_view
                = n_iters | std::views::transform([i32_pair_t, &bld](const auto &p) {
                      return llvm::ConstantArray::get(
                          i32_pair_t, {bld.getInt32(p.first),
                                       // NOTE: the summation here is safe because in func::taylor_c_diff_get_n_iters()
                                       // we check that we can represent p.first + p.second as a 32-bit integer.
                                       bld.getInt32(p.first + p.second)});
                  });

            // Build the array of ordered iteration ranges.
            std::vector<llvm::Constant *> rng_arr_const(std::ranges::begin(rng_view), std::ranges::end(rng_view));
            auto *rng_arr = llvm::ConstantArray::get(rng_arr_t, rng_arr_const);
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            auto *gv_rng_arr
                = new llvm::GlobalVariable(md, rng_arr->getType(), true, llvm::GlobalVariable::PrivateLinkage, rng_arr);

            // Fetch a pointer to the beginning of the array.
            llvm::Value *rng_arr_ptr = bld.CreateInBoundsGEP(rng_arr_t, gv_rng_arr, {bld.getInt32(0), bld.getInt32(0)});

            // Assign it into oiter_map.
            o_it->second = rng_arr_ptr;
        }
    }

    // Now we build the transposition of tmp_map: from {f : [[a, b, c], [d, e, f]]} to {f : [[a, d], [b, e], [c, f]]}.
    std::map<llvm::Function *, std::vector<std::variant<std::vector<std::uint32_t>, std::vector<number>>>,
             llvm_func_name_compare>
        tmp_map_transpose;
    for (const auto &[func, vv] : tmp_map) {
        assert(!vv.empty()); // LCOV_EXCL_LINE

        // Add the function.
        const auto [it, ins_status] = tmp_map_transpose.try_emplace(func);
        assert(ins_status); // LCOV_EXCL_LINE

        const auto n_calls = vv.size();
        const auto n_args = vv[0].size();
        // NOTE: n_args must be at least 1 because the u idx is prepended to the actual function arguments in the
        // tmp_map entries.
        assert(n_args >= 1u); // LCOV_EXCL_LINE

        for (decltype(vv[0].size()) i = 0; i < n_args; ++i) {
            // Build the vector of values corresponding to the current argument index.
            std::vector<std::variant<std::uint32_t, number>> tmp_c_vec;
            tmp_c_vec.reserve(n_calls);
            for (decltype(vv.size()) j = 0; j < n_calls; ++j) {
                tmp_c_vec.push_back(vv[j][i]);
            }

            // Turn tmp_c_vec (a vector of variants) into a variant of vectors, and insert the result.
            it->second.push_back(vv_transpose(tmp_c_vec));
        }
    }

    // Create the taylor_cm_seg_f_list_t for the current segment.
    taylor_cm_seg_f_list_t seg_map;

    for (const auto &[func, vv] : tmp_map_transpose) {
        // NOTE: vv.size() is now the number of arguments. We know it cannot be zero because the functions to compute
        // the Taylor derivatives in compact mode always have at least 1 argument (i.e., the index of the u variable
        // whose derivative is being computed).
        assert(!vv.empty()); // LCOV_EXCL_LINE

        // Add the function.
        const auto [it, ins_status] = seg_map.try_emplace(func);
        assert(ins_status); // LCOV_EXCL_LINE

        // Set the number of calls for this function.
        it->second.first
            = std::visit([](const auto &x) { return boost::numeric_cast<std::uint32_t>(x.size()); }, vv[0]);
        assert(it->second.first > 0u); // LCOV_EXCL_LINE

        // Create the g functions for each argument.
        for (const auto &v : vv) {
            it->second.second.push_back(std::visit(
                [&s, fp_t](const auto &x) {
                    using type = uncvref_t<decltype(x)>;

                    if constexpr (std::is_same_v<type, std::vector<std::uint32_t>>) {
                        return cm_make_arg_gen_vidx(s, x);
                    } else {
                        return cm_make_arg_gen_vc(s, fp_t, x);
                    }
                },
                v));
        }
    }

    // Generate the code for the computation of the Taylor derivatives.
    if (parallel_mode) {
        // TODO remove.
        throw std::invalid_argument("not supported!!!\n");
    } else {
        taylor_cm_codegen_segment_diff_impl(s, fp_vec_type, seg_map, n_uvars, oiter_map);
    }

    return seg_map;
}

// Data structure representing a single partition in a partitioned decomposition.
//
// A partitioned decomposition is the central part of a Taylor decomposition reorganised into a set of partitions. Each
// partition contains all expressions in the decomposition sharing the same formula for the calculation of the Taylor
// derivatives. Thus, for instance, one partition may contain all expressions of type variable * variable, another
// partition may contain all expressions of type par * variable, another partition may contain all expressions of type
// cos(variable), and so on.
//
// Typically, the Taylor derivatives formulae consist of summations in which only one or two terms involve the
// current-order derivatives of the arguments. The rest of the terms in the summations involve only previous-orders
// derivatives. Thus, a large fraction of the computation can be performed concurrently (i.e., coroutine-style) across
// multiple expressions of the same type while disregarding data dependencies between expressions. For instance, if a
// partition consists of two variable * variable subexpressions, we begin with the first iteration for the computation
// of the Taylor derivative of the first expression and then we move to perform the first iteration for the second
// expression. Then we move to the second iterations, first for the first expression and then for the second expression.
// And so on. This allows us to employ a cache-friendly memory access pattern to the tape of derivatives in which we are
// accessing the previous-orders derivatives order-by-order. This also reduces the number of branch instructions because
// we are using a single loop to compute multiple Taylor derivatives at the same time (instead of using one loop per
// derivative).
struct dc_partition {
    // The LLVM function used to perform a single iteration in the computation of the Taylor derivatives of the
    // expressions in the partition.
    llvm::Function *f = nullptr;
    // A pointer to the first element of a global constant array containing the number of unordered iterations
    // (represented as 32-bit ints) for each differentiation order. The size of the array is order + 1.
    llvm::Value *n_uiters_ptr = nullptr;
    // The total number of subexpressions in the partition. This is also the total number of invocations of f.
    std::uint32_t n_ex = 0;
    // The function objects to generate a subset of the arguments for f.
    std::vector<std::function<llvm::Value *(llvm::Value *)>> generators;
};

// Helper to build the partitioned counterpart of the input segmented decomposition.
//
// This function will codegen into s all the single-iteration functions used for the computation of the Taylor
// derivatives in compact mode. It will also codegen global arrays containing the number of unordered iterations for
// each partition (i.e., one array of size order + 1 per partition), an array containing the max number of unordered
// iterations (calculated across the partitions) for each differentiation order, and several arrays containing arguments
// for the functions computing the Taylor derivatives.
//
// s is the llvm_state in which we are operating and to which functions and global variables will be added. fp_t is the
// scalar floating-point type to be used in the computation of the Taylor derivatives. s_dc is the segmented
// decomposition. n_eq is the number of differential equations. n_uvars is the total number of u variables in the
// decomposition. order is the maximum Taylor order. batch_size is the batch size. high_accuracy the high accuracy flag.
//
// The two return values are the paritioned decomposition and a pointer to the beginning of the array of max number of
// unordered iterations.
std::pair<std::vector<dc_partition>, llvm::Value *>
build_partitioned_dc(llvm_state &s, llvm::Type *fp_t, const std::vector<taylor_dc_t> &s_dc,
                     // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                     const std::uint32_t n_eq, const std::uint32_t n_uvars, const std::uint32_t order,
                     const std::uint32_t batch_size, const bool high_accuracy)
{
    auto &bld = s.builder();
    auto &md = s.module();

    // Fetch the 32-bit integer type.
    auto *i32_t = bld.getInt32Ty();

    // Initialise the flattened version of the partitioned decomposition, containing, for each expression ex in s_dc:
    //
    // - the LLVM function for the computation of a single iteration of the Taylor derivative of ex,
    // - the index of ex in the original decomposition,
    // - ex itself,
    // - a subset of the arguments for the LLVM function, codegenned either as 32-bit indices values (for the variable
    //   and par arguments of ex, and for the hidden dependencies) or as numbers (for the constant arguments of ex).
    std::vector<
        std::tuple<llvm::Function *, std::uint32_t, expression, std::vector<std::variant<std::uint32_t, number>>>>
        flat_pdc;
    // NOTE: the segmented decomposition s_dc represents the central part of the original decomposition. Thus, the index
    // of its first subexpression is n_eq.
    std::uint32_t cur_idx = n_eq;
    for (const auto &[ex, deps] : s_dc | std::views::join) {
        assert(std::holds_alternative<func>(ex.value()));
        flat_pdc.emplace_back(
            std::get<func>(ex.value()).taylor_c_diff_get_single_iter_func(s, fp_t, n_uvars, batch_size, high_accuracy),
            // NOTE: cur_idx++ here is safe thanks to the overflow checks in taylor_compute_jet().
            cur_idx++, ex, udef_to_variants(ex, deps));
    }

    // Sort flat_pdc according to the name of the LLVM function.
    // NOTE: use stable_sort() because within each partition we want to maintain the relative order from the segmented
    // decomposition.
    std::ranges::stable_sort(flat_pdc, [](const auto &t1, const auto &t2) {
        return llvm_func_name_compare{}(std::get<0>(t1), std::get<0>(t2));
    });

    // Prepare the vector of max_nuiters, inited to zeroes.
    std::vector<std::uint32_t> max_nuiters;
    max_nuiters.resize(boost::numeric_cast<decltype(max_nuiters.size())>(order + 1u));

    // Create a view to split flat_pdc into groups based on the LLVM function name.
    auto gview = flat_pdc | std::views::chunk_by([](const auto &t1, const auto &t2) {
                     auto *f1 = std::get<0>(t1);
                     auto *f2 = std::get<0>(t2);

                     return f1->getName() == f2->getName();
                 });

    // Setup the type of the (max_)n_uiters arrays.
    auto *n_uiters_arr_t = llvm::ArrayType::get(i32_t, order + 1u);

    // Build up the partitioned decomposition.
    std::vector<dc_partition> ret;
    for (auto r : gview) {
        static_assert(std::ranges::random_access_range<decltype(r)>);
        assert(!std::ranges::empty(r));

        // Fetch a reference to the first element in the group.
        const auto &front = *std::ranges::begin(r);

        // Fetch the first LLVM function in the group (all the items in the group are assumed to contain the same
        // function).
        auto *f = std::get<0>(front);

        // Construct a view over the n_uiters vector.
        const auto &ex = std::get<2>(front);
        assert(std::holds_alternative<func>(ex.value()));
        auto n_uiters_view = std::get<func>(ex.value()).taylor_c_diff_get_n_iters(order)
                             | std::views::transform([](const auto &p) { return p.first; });

        // Update the max_nuiters vector.
        for (std::uint32_t i = 0; i <= order; ++i) {
            max_nuiters[i] = std::max(max_nuiters[i], n_uiters_view[i]);
        }

        // Setup the LLVM counterpart of n_uiters.
        auto n_uiters_const_view
            = std::move(n_uiters_view) | std::views::transform([&bld](const auto idx) { return bld.getInt32(idx); });
        std::vector<llvm::Constant *> n_uiters_arr_const(std::ranges::begin(n_uiters_const_view),
                                                         std::ranges::end(n_uiters_const_view));
        auto *n_uiters_arr = llvm::ConstantArray::get(n_uiters_arr_t, n_uiters_arr_const);
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        auto *gv_n_uiters_arr = new llvm::GlobalVariable(md, n_uiters_arr->getType(), true,
                                                         llvm::GlobalVariable::PrivateLinkage, n_uiters_arr);

        // Fetch a pointer to the first element of the array.
        auto *n_uiters_ptr = bld.CreateInBoundsGEP(n_uiters_arr_t, gv_n_uiters_arr, {bld.getInt32(0), bld.getInt32(0)});

        // Init the vector of generators.
        std::vector<std::function<llvm::Value *(llvm::Value *)>> generators;

        // The first generator must produce the index of the u variable on which we are operating.
        auto u_idx_view = r | std::views::transform([](const auto &tup) { return std::get<1>(tup); });
        generators.push_back(cm_make_arg_gen_vidx(s, {std::ranges::begin(u_idx_view), std::ranges::end(u_idx_view)}));

        // The remaining generators handle the arguments of the u variable and the hidden deps. Infer their number from
        // the first element in the group.
        const auto &first_var_vec = std::get<3>(front);
        const auto n_gens = first_var_vec.size();

        // TODO explain.
        for (decltype(first_var_vec.size()) gen_idx = 0; gen_idx < n_gens; ++gen_idx) {
            auto view = r | std::views::transform([gen_idx](const auto &tup) {
                            const auto &cur_var_vec = std::get<3>(tup);

                            if (gen_idx >= cur_var_vec.size()) [[unlikely]] {
                                // TODO throw.
                            }

                            return cur_var_vec[gen_idx];
                        });

            // TODO rename.
            const auto foo = vv_transpose(std::vector(std::ranges::begin(view), std::ranges::end(view)));
            generators.push_back(std::visit(
                [&s, fp_t]<typename T>(const T &x) {
                    if constexpr (std::same_as<T, std::vector<std::uint32_t>>) {
                        return cm_make_arg_gen_vidx(s, x);
                    } else {
                        return cm_make_arg_gen_vc(s, fp_t, x);
                    }
                },
                foo));
        }

        // Construct and append the partition.
        ret.push_back({.f = f,
                       .n_uiters_ptr = n_uiters_ptr,
                       .n_ex = boost::numeric_cast<std::uint32_t>(std::ranges::size(r)),
                       .generators = std::move(generators)});
    }

    // Codegen the max_nuiters vector.
    auto max_n_uiters_const_view
        = max_nuiters | std::views::transform([&bld](const auto n) { return bld.getInt32(n); });
    std::vector<llvm::Constant *> max_n_uiters_arr_const(std::ranges::begin(max_n_uiters_const_view),
                                                         std::ranges::end(max_n_uiters_const_view));
    auto *max_n_uiters_arr = llvm::ConstantArray::get(n_uiters_arr_t, max_n_uiters_arr_const);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *gv_max_n_uiters_arr = new llvm::GlobalVariable(md, max_n_uiters_arr->getType(), true,
                                                         llvm::GlobalVariable::PrivateLinkage, max_n_uiters_arr);

    // Fetch a pointer to the first element of the array.
    auto *max_n_uiters_ptr
        = bld.CreateInBoundsGEP(n_uiters_arr_t, gv_max_n_uiters_arr, {bld.getInt32(0), bld.getInt32(0)});

    // Assemble the return value.
    return std::make_pair(std::move(ret), max_n_uiters_ptr);
}

// Helper to codegen the computation of the Taylor derivatives in compact mode via driver functions implemented across
// multiple LLVM states. main_state is the state in which the stepper is defined, main_fp_t the internal scalar
// floating-point type as defined in the main state, main_par/main_time/main_tape_ptr the parameters/time/tape pointers
// as defined in the main state, dc the Taylor decomposition, s_dc its segmented counterpart, n_eq the number of
// equations/state variables, order the Taylor order, batch_size the batch size, high_accuracy the high accuracy flag,
// parallel_mode the parallel mode flag, max_svf_idx the maximum index in the decomposition of the sv funcs (or zero if
// there are no sv funcs).
//
// The return value is a list of states in which the driver functions have been defined.
std::vector<llvm_state> taylor_compute_jet_multi(llvm_state &main_state, llvm::Type *main_fp_t,
                                                 llvm::Value *main_par_ptr, llvm::Value *main_time_ptr,
                                                 llvm::Value *main_tape_ptr, const taylor_dc_t &dc,
                                                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                 const std::vector<taylor_dc_t> &s_dc, std::uint32_t n_eq,
                                                 std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                                                 bool high_accuracy, bool parallel_mode, std::uint32_t max_svf_idx)
{
    // Fetch the internal vector type for the main state.
    auto *main_fp_vec_type = make_vector_type(main_fp_t, batch_size);

    // Init the list of states.
    //
    // NOTE: we use lists here because it is convenient to have pointer/reference stability when iteratively
    // constructing the set of states.
    std::list<llvm_state> states;

    // Push back a new state and use it as initial current state.
    //
    // NOTE: like this, we always end up creating at least one driver function and a state, even in the degenerate case
    // of an empty decomposition, which is suboptimal peformance-wise. I do not think however that it is worth it to
    // complicate the code to avoid this corner-case pessimisation.
    states.push_back(main_state.make_similar());
    auto *cur_state = &states.back();

    // Generate the global arrays for the computation of the derivatives
    // of the state variables in the main state.
    const auto svd_gl = taylor_c_make_sv_diff_globals(main_state, main_fp_t, dc, n_uvars);

    // Build the partitioned decomposition and the array of max unordered iterations.
    const auto [pdc, max_n_uiters_arr]
        = build_partitioned_dc(main_state, main_fp_t, s_dc, n_eq, n_uvars, order, batch_size, high_accuracy);

    // Structure used to log, in trace mode, the breakdown of each segment. For each segment, this structure contains
    // the number of invocations of each function in the segment. It will be unused if we are not tracing.
    std::vector<std::vector<std::uint32_t>> segment_bd;

    // Are we tracing?
    const auto is_tracing = get_logger()->should_log(spdlog::level::trace);

    // Do we need to compute the last-order derivatives for the sv_funcs?
    const auto need_svf_lo = max_svf_idx >= n_eq;

    // Index of the state we are currently operating on.
    boost::safe_numerics::safe<unsigned> cur_state_idx = 0;

    // NOTE: unlike in compiled functions, we cannot at the same time declare and invoke the drivers from the main
    // module as the invocation happens from within an LLVM loop. Thus, we first define the drivers in the states and
    // add their declarations in the main state, and only at a later stage we perform the invocation of the drivers in
    // the main state.

    // Declarations of the drivers in the main state.
    std::vector<llvm::Function *> main_driver_decls;
    // Add the declaration for the first driver.
    main_driver_decls.push_back(taylor_cm_make_driver_proto(main_state, cur_state_idx));

    // The driver function for the evaluation of the segment containing max_svf_idx. Will remain null if we do not need
    // to compute the last-order derivatives for the sv funcs.
    llvm::Function *max_svf_driver = nullptr;

    // Add the driver declaration to the current state, and start insertion into the driver.
    cur_state->builder().SetInsertPoint(llvm::BasicBlock::Create(
        cur_state->context(), "entry", taylor_cm_make_driver_proto(*cur_state, cur_state_idx)));

    // Variable to keep track of how many evaluation functions have been invoked in the current state.
    boost::safe_numerics::safe<std::size_t> n_evalf = 0;

    // Limit of function evaluations per state.
    //
    // NOTE: this has not been really properly tuned, needs more investigation.
    constexpr auto max_n_evalf = 200u;

    // Variable to keep track of the index of the first u variable in a segment.
    auto start_u_idx = n_eq;

    // Helper to finalise the current driver function and create a new one.
    auto start_new_driver = [&cur_state, &states, &main_state, &n_evalf, &cur_state_idx, &main_driver_decls]() {
        // Finalise the current driver.
        cur_state->builder().CreateRetVoid();

        // Create the new current state.
        states.push_back(main_state.make_similar());
        cur_state = &states.back();

        // Reset/update the counters.
        n_evalf = 0;
        ++cur_state_idx;

        // Add the driver declaration to the main state.
        main_driver_decls.push_back(taylor_cm_make_driver_proto(main_state, cur_state_idx));

        // Add the driver declaration to the current state,
        // and start insertion into the driver.
        cur_state->builder().SetInsertPoint(llvm::BasicBlock::Create(
            cur_state->context(), "entry", taylor_cm_make_driver_proto(*cur_state, cur_state_idx)));
    };

    // Setup the map of ordered iteration ranges. This will be iteratively built up and used in the next loop.
    taylor_cm_f_oiter_rng_map_t oiter_map;

    // Iterate over the segments in s_dc and codegen the code for the computation of Taylor derivatives.
    for (const auto &seg : s_dc) {
        // Cache the number of expressions in the segment.
        const auto seg_n_ex = static_cast<std::uint32_t>(seg.size());

        // Are we in the segment containing max_svf_idx? We are if:
        //
        // - we need to compute the last-order derivatives of the sv funcs,
        // - max_svf_idx is somewhere within this segment.
        //
        // In such a case, we create a driver specifically for this segment, which we will invoke again at the end of
        // this function to compute the last-order derivatives of the sv funcs.
        const auto is_svf_seg = need_svf_lo && max_svf_idx >= start_u_idx && max_svf_idx < (start_u_idx + seg_n_ex);

        if (n_evalf > max_n_evalf || is_svf_seg) {
            // Either we have codegenned enough blocks for this state, or we are in the max_svf_idx state. Finalise the
            // current driver and start the new one.
            start_new_driver();

            // Assign max_svf_driver if needed.
            if (is_svf_seg) {
                assert(max_svf_driver == nullptr);
                max_svf_driver = main_driver_decls.back();
            }
        }

        // Fetch the internal fp type for the current state.
        auto *fp_t = llvm_clone_type(*cur_state, main_fp_t);

        // Codegen the computation of the derivatives for this segment.
        const auto seg_map = taylor_cm_codegen_segment_diff(seg, start_u_idx, *cur_state, fp_t, batch_size, n_uvars,
                                                            high_accuracy, parallel_mode, oiter_map, order);

        // Update the number of invoked evaluation functions.
        n_evalf = std::accumulate(seg_map.begin(), seg_map.end(), n_evalf,
                                  [](auto a, const auto &p) { return a + p.second.first; });

        // Update start_u_idx.
        start_u_idx += seg_n_ex;

        // If we codegenned the max_svf_idx driver, start immediately a new driver. We want the max_svf_idx driver to
        // contain the codegen for a single segment and nothing more, otherwise we end up doing unnecessary work when
        // computing the last-order derivatives of the sv funcs.
        if (is_svf_seg) {
            start_new_driver();
        }

        // LCOV_EXCL_START
        // Update segment_bd if needed.
        if (is_tracing) {
            segment_bd.emplace_back();

            for (const auto &p : seg_map) {
                segment_bd.back().push_back(p.second.first);
            }
        }
        // LCOV_EXCL_STOP
    }

    // We need one last return statement for the last added state.
    cur_state->builder().CreateRetVoid();

    // LCOV_EXCL_START
    // Log segment_bd, if needed.
    if (is_tracing) {
        get_logger()->trace("Taylor function maps breakdown: {}", segment_bd);
    }
    // LCOV_EXCL_STOP

    // Back in the main state, we begin by invoking all the drivers with order zero. That is, we are computing the
    // initial values of the u variables.
    //
    // NOTE: this is an ordered iteration which relies on the guarantees that at differentiation order 0:
    //
    // - there is always exactly one ordered iteration,
    // - the single ordered iteration never reads from the accumulators, it only writes to them.
    auto &main_bld = main_state.builder();
    for (auto *cur_driver_f : main_driver_decls) {
        main_bld.CreateCall(cur_driver_f, {main_tape_ptr, main_par_ptr, main_time_ptr, main_bld.getInt32(0)});
    }

    // We then introduce a handy helper to zero out the area in the tape containing the accumulators for the current
    // order. This will be called at the beginning of the next loop.
    const auto zero_accumulators = [&](llvm::Value *cur_order) {
        llvm_loop_u32(main_state, main_bld.getInt32(n_eq), main_bld.getInt32(n_uvars), [&](llvm::Value *cur_var_idx) {
            // Fetch the value type of the tape.
            auto *main_val_t = make_vector_type(main_fp_t, batch_size);

            // Zero out the value of cur_var_idx at order cur_order.
            taylor_c_store_diff(main_state, main_val_t, main_tape_ptr, n_uvars, cur_order, cur_var_idx,
                                llvm_codegen(main_state, main_val_t, number{0.}));
        });
    };

    // Next, we compute all derivatives up to order 'order - 1'.
    llvm_loop_u32(main_state, main_bld.getInt32(1), main_bld.getInt32(order), [&](llvm::Value *cur_order) {
        // State variables first.
        taylor_c_compute_sv_diffs(main_state, main_fp_t, svd_gl, main_tape_ptr, main_par_ptr, n_uvars, cur_order,
                                  batch_size);

        // Zero out the accumulators for the current order.
        zero_accumulators(cur_order);

        // Next, we run the unordered iterations.

        // Load the max number of unordered iterations for the current order.
        auto *main_i32_t = main_bld.getInt32Ty();
        auto *max_n_uiters_ptr = main_bld.CreateInBoundsGEP(main_i32_t, max_n_uiters_arr, {cur_order});
        auto *max_n_uiters = main_bld.CreateLoad(main_i32_t, max_n_uiters_ptr);

        llvm_loop_u32(main_state, main_bld.getInt32(0), max_n_uiters, [&](llvm::Value *cur_u_iter) {
            // Iterate over the partitions.
            for (const auto &part : pdc) {
                const auto &gens = part.generators;

                // Load the number of iterations for the current partition at the given order.
                auto *n_uiters_ptr = main_bld.CreateInBoundsGEP(main_i32_t, part.n_uiters_ptr, {cur_order});
                auto *n_uiters = main_bld.CreateLoad(main_i32_t, n_uiters_ptr);

                // The unordered iterations for the current partition need to be performed only if cur_u_iter <
                // n_uiters.
                llvm_if_then_else(
                    main_state, main_bld.CreateICmpULT(cur_u_iter, n_uiters),
                    [&]() {
                        // Iterate over the total number of expressions in the partition.
                        llvm_loop_u32(main_state, main_bld.getInt32(0), main_bld.getInt32(part.n_ex),
                                      [&](llvm::Value *cur_call_idx) {
                                          // Create the u variable index from the first generator.
                                          auto *u_idx = gens[0](cur_call_idx);

                                          // Initialise the vector of arguments with which func must be called. The
                                          // following
                                          // initial arguments are always present:
                                          // - current Taylor order,
                                          // - u index of the variable,
                                          // - tape of derivatives,
                                          // - pointer to the param values,
                                          // - pointer to the time value(s).
                                          std::vector<llvm::Value *> args{cur_order, u_idx, main_tape_ptr, main_par_ptr,
                                                                          main_time_ptr};

                                          // Create the arguments from the other generators.
                                          for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                                              args.push_back(gens[i](cur_call_idx));
                                          }

                                          // Compute the pointer to the accumulator.
                                          auto *acc_ptr = main_bld.CreateInBoundsGEP(
                                              main_fp_vec_type, main_tape_ptr,
                                              {main_bld.CreateAdd(
                                                  main_bld.CreateMul(cur_order, main_bld.getInt32(n_uvars)), u_idx)});

                                          // Add it to the list of arguments.
                                          args.push_back(acc_ptr);

                                          // Add the current iteration index to the list of arguments.
                                          args.push_back(cur_u_iter);

                                          // Perform a single iteration of the calculation of the derivative.
                                          main_bld.CreateCall(part.f, args);
                                      });
                    },
                    []() {});
            }
        });

        // Next, run the ordered iterations.
        for (auto *cur_driver_f : main_driver_decls) {
            main_bld.CreateCall(cur_driver_f, {main_tape_ptr, main_par_ptr, main_time_ptr, cur_order});
        }
    });

    // Next, we compute the last-order derivatives for the state variables.
    taylor_c_compute_sv_diffs(main_state, main_fp_t, svd_gl, main_tape_ptr, main_par_ptr, n_uvars,
                              main_bld.getInt32(order), batch_size);

    // Finally, we compute the last-order derivatives for the sv_funcs, if needed. Because the sv funcs correspond to u
    // variables somewhere in the decomposition, we will have to compute the last-order derivatives of the u variables
    // until we are sure all sv_funcs derivatives have been properly computed.
    if (need_svf_lo) {
        assert(max_svf_driver != nullptr);

        // What we do here is to iterate over all the drivers, invoke them one by one, and break out when we have
        // detected max_svf_driver.
        for (auto *cur_driver_f : main_driver_decls) {
            main_bld.CreateCall(cur_driver_f, {main_tape_ptr, main_par_ptr, main_time_ptr, main_bld.getInt32(order)});

            if (cur_driver_f == max_svf_driver) {
                break;
            }
        }
    }

    // Return the states.
    return ranges_to<std::vector<llvm_state>>(states | std::views::as_rvalue);
}

// Helper for the computation of a jet of derivatives in compact mode, used in taylor_compute_jet(). The return values
// are the size/alignment requirements for the tape of derivatives and the list of states in which the drivers are
// implemented. All LLVM values and types passed to this function are defined in the main state.
std::pair<std::array<std::size_t, 2>, std::vector<llvm_state>> taylor_compute_jet_compact_mode(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    llvm_state &main_state, llvm::Type *main_fp_t, llvm::Value *order0, llvm::Value *main_par_ptr,
    llvm::Value *main_time_ptr, llvm::Value *main_tape_ptr, const taylor_dc_t &dc,
    const std::vector<std::uint32_t> &sv_funcs_dc, std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order,
    std::uint32_t batch_size, bool high_accuracy, bool parallel_mode)
{
    auto &main_bld = main_state.builder();
    auto &main_md = main_state.module();

    // Determine the vector type corresponding to main_fp_t.
    auto *main_fp_vec_t = make_vector_type(main_fp_t, batch_size);

    // Fetch the external type corresponding to fp_type.
    auto *main_ext_fp_t = make_external_llvm_type(main_fp_t);

    // Split dc into segments.
    const auto s_dc = taylor_segment_dc(dc, n_eq);

    // Determine the maximum u variable index appearing in sv_funcs_dc, or zero
    // if sv_funcs_dc is empty.
    const auto max_svf_idx
        = sv_funcs_dc.empty() ? static_cast<std::uint32_t>(0) : *std::ranges::max_element(sv_funcs_dc);

    // Determine the total number of elements to be stored in the tape of derivatives. We will be storing all the
    // derivatives of the u variables up to order 'order - 1', the derivatives of order 'order' of the state variables
    // and the derivatives of order 'order' of the sv_funcs.
    //
    // NOTE: if sv_funcs_dc is empty, or if all its indices are not greater than the indices of the state variables,
    // then we don't need additional slots after the sv derivatives. If we need additional slots, allocate another full
    // column of derivatives, as it is complicated at this stage to know exactly how many slots we will need.
    //
    // NOTE: overflow checking for this computation has been performed externally.
    const auto tot_tape_N = (max_svf_idx < n_eq) ? ((n_uvars * order) + n_eq) : (n_uvars * (order + 1u));

    // Total required size in bytes for the tape.
    const auto tape_sz = boost::safe_numerics::safe<std::size_t>(get_size(main_md, main_fp_vec_t)) * tot_tape_N;

    // Tape alignment.
    const auto tape_al = boost::numeric_cast<std::size_t>(get_alignment(main_md, main_fp_vec_t));

    // Log the runtime of IR construction in trace mode.
    spdlog::stopwatch sw;

    // NOTE: tape_ptr is used as temporary storage for the current function, but it is provided externally from
    // dynamically-allocated memory in order to avoid stack overflow. This creates a situation in which LLVM cannot
    // elide stores into tape_ptr (even if it figures out a way to avoid storing intermediate results into it) because
    // LLVM must assume that some other function may use these stored values later. Thus, we declare via an intrinsic
    // that the lifetime of tape_ptr begins here and ends at the end of the function, so that LLVM can assume that any
    // value stored in it cannot be possibly used outside this function.
    main_bld.CreateLifetimeStart(main_tape_ptr, main_bld.getInt64(tape_sz));

    // Copy the order-0 derivatives of the state variables into the tape.
    // NOTE: overflow checking is already done in the parent function.
    llvm_loop_u32(main_state, main_bld.getInt32(0), main_bld.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
        // Fetch the pointer from order0.
        auto *ptr = main_bld.CreateInBoundsGEP(main_ext_fp_t, order0,
                                               main_bld.CreateMul(cur_var_idx, main_bld.getInt32(batch_size)));

        // Load as a vector.
        auto *vec = ext_load_vector_from_memory(main_state, main_fp_t, ptr, batch_size);

        // Store into tape_ptr.
        taylor_c_store_diff(main_state, main_fp_vec_t, main_tape_ptr, n_uvars, main_bld.getInt32(0), cur_var_idx, vec);
    });

    // Codegen the computation of the Taylor derivatives across multiple states.
    auto states = taylor_compute_jet_multi(main_state, main_fp_t, main_par_ptr, main_time_ptr, main_tape_ptr, dc, s_dc,
                                           n_eq, n_uvars, order, batch_size, high_accuracy, parallel_mode, max_svf_idx);

    get_logger()->trace("Taylor IR creation compact mode runtime: {}", sw);

    // Return the tape size/alignment and the list of states containing the drivers.
    return std::make_pair(std::array<std::size_t, 2>{tape_sz, tape_al}, std::move(states));
}

// Given an input pointer 'in', load the first n * batch_size values in it as n vectors
// with size batch_size. If batch_size is 1, the values will be loaded as scalars.
// 'in' is an external pointer.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto taylor_load_values(llvm_state &s, llvm::Type *fp_t, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    auto &builder = s.builder();

    // Fetch the external type corresponding to fp_t.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    std::vector<llvm::Value *> retval;
    for (std::uint32_t i = 0; i < n; ++i) {
        // Fetch the pointer from in.
        // NOTE: overflow checking is done in the parent function.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, in, builder.getInt32(i * batch_size));

        // Load the value in vector mode.
        retval.push_back(ext_load_vector_from_memory(s, fp_t, ptr, batch_size));
    }

    return retval;
}

} // namespace

// Helper function to compute the jet of Taylor derivatives up to a given order.
//
// n_eq is the number of equations/variables in the ODE sys, dc its Taylor decomposition, n_uvars the total number of u
// variables in the decomposition. order is the max derivative order desired, batch_size the batch size. order0 is a
// pointer to an array of (at least) n_eq * batch_size scalar elements containing the derivatives of order 0. par_ptr is
// a pointer to an array containing the numerical values of the parameters, time_ptr a pointer to the time value(s),
// tape_ptr a pointer to the tape of derivatives (only in compact mode, otherwise a null value). sv_funcs are the
// indices, in the decomposition, of the functions of state variables.
//
// order0, par_ptr and time_ptr are all external pointers.
//
// The return value is a variant containing either:
//
// - in compact mode, the size/alignment requirements for the tape of derivatives and the list of states in which the
//   driver functions are implemented, or
// - the jet of derivatives of the state variables and sv_funcs up to order 'order'.
std::variant<std::pair<std::array<std::size_t, 2>, std::vector<llvm_state>>, std::vector<llvm::Value *>>
taylor_compute_jet(llvm_state &s, llvm::Type *fp_t, llvm::Value *order0, llvm::Value *par_ptr, llvm::Value *time_ptr,
                   llvm::Value *tape_ptr, const taylor_dc_t &dc, const std::vector<std::uint32_t> &sv_funcs_dc,
                   std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                   bool compact_mode, bool high_accuracy, bool parallel_mode)
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(n_eq > 0u);
    assert(order > 0u);
    assert((tape_ptr != nullptr) == compact_mode);
    // LCOV_EXCL_STOP

    // Overflow checks.
    using safe_u32_t = boost::safe_numerics::safe<std::uint32_t>;
    try {
        // We must be able to represent n_uvars * (order + 1) as a 32-bit unsigned integer. This is the maximum total
        // number of derivatives we will have to compute and store, with the +1 taking into account the extra slots that
        // might be needed by sv_funcs_dc. If sv_funcs_dc is empty, we need only n_uvars * order + n_eq derivatives.
        static_cast<void>(static_cast<std::uint32_t>(n_uvars * (safe_u32_t(order) + 1u)));

        // We also need to be able to index up to n_eq * batch_size in order0.
        static_cast<void>(static_cast<std::uint32_t>(safe_u32_t(n_eq) * batch_size));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives");
    }
    // LCOV_EXCL_STOP

    if (compact_mode) {
        return taylor_compute_jet_compact_mode(s, fp_t, order0, par_ptr, time_ptr, tape_ptr, dc, sv_funcs_dc, n_eq,
                                               n_uvars, order, batch_size, high_accuracy, parallel_mode);
    } else {
        // Log the runtime of IR construction in trace mode.
        spdlog::stopwatch sw;

        // Init the derivatives array with the order 0 of the state variables.
        auto diff_arr = taylor_load_values(s, fp_t, order0, n_eq, batch_size);

        // Compute the order-0 derivatives of the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            diff_arr.push_back(taylor_diff(s, fp_t, dc[i].first, dc[i].second, diff_arr, par_ptr, time_ptr, n_uvars, 0,
                                           i, batch_size, high_accuracy));
        }

        // Compute the derivatives order by order, starting from 1 to order excluded.
        // We will compute the highest derivatives of the state variables separately
        // in the last step.
        for (std::uint32_t cur_order = 1; cur_order < order; ++cur_order) {
            // Begin with the state variables.
            // NOTE: the derivatives of the state variables
            // are at the end of the decomposition vector.
            for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
                diff_arr.push_back(
                    taylor_compute_sv_diff(s, fp_t, dc[i].first, diff_arr, par_ptr, n_uvars, cur_order, batch_size));
            }

            // Now the other u variables.
            for (auto i = n_eq; i < n_uvars; ++i) {
                diff_arr.push_back(taylor_diff(s, fp_t, dc[i].first, dc[i].second, diff_arr, par_ptr, time_ptr, n_uvars,
                                               cur_order, i, batch_size, high_accuracy));
            }
        }

        // Compute the last-order derivatives for the state variables.
        for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
            diff_arr.push_back(
                taylor_compute_sv_diff(s, fp_t, dc[i].first, diff_arr, par_ptr, n_uvars, order, batch_size));
        }

        // If there are sv funcs, we need to compute their last-order derivatives too:
        // we will need to compute the derivatives of the u variables up to
        // the maximum index in sv_funcs_dc.
        const auto max_svf_idx
            = sv_funcs_dc.empty() ? static_cast<std::uint32_t>(0) : *std::ranges::max_element(sv_funcs_dc);

        // NOTE: if there are no sv_funcs, max_svf_idx is set to zero
        // above, thus we never enter the loop.
        // NOTE: <= because max_svf_idx is an index, not a size.
        for (std::uint32_t i = n_eq; i <= max_svf_idx; ++i) {
            diff_arr.push_back(taylor_diff(s, fp_t, dc[i].first, dc[i].second, diff_arr, par_ptr, time_ptr, n_uvars,
                                           order, i, batch_size, high_accuracy));
        }

#if !defined(NDEBUG)
        if (sv_funcs_dc.empty()) {
            assert(diff_arr.size() == static_cast<decltype(diff_arr.size())>(n_uvars) * order + n_eq);
        } else {
            // NOTE: we use std::max<std::uint32_t>(n_eq, max_svf_idx + 1u) here because
            // the sv funcs could all be aliases of the state variables themselves,
            // in which case in the previous loop we ended up appending nothing.
            assert(diff_arr.size()
                   == static_cast<decltype(diff_arr.size())>(n_uvars) * order
                          + std::max<std::uint32_t>(n_eq, max_svf_idx + 1u));
        }
#endif

        // Extract the derivatives of the state variables and sv_funcs from diff_arr.
        std::vector<llvm::Value *> retval;
        for (std::uint32_t o = 0; o <= order; ++o) {
            for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
                retval.push_back(taylor_fetch_diff(diff_arr, var_idx, o, n_uvars));
            }
            for (auto idx : sv_funcs_dc) {
                retval.push_back(taylor_fetch_diff(diff_arr, idx, o, n_uvars));
            }
        }

        get_logger()->trace("Taylor IR creation default mode runtime: {}", sw);

        return retval;
    }
}

// Small helper to construct a default value for the max_delta_t
// keyword argument. The default value is +inf.
template <typename T>
T taylor_default_max_delta_t()
{
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        // NOTE: the precision here does not matter,
        // it will be rounded to the correct precision in
        // any case.
        return mppp::real{mppp::real_kind::inf, 128};
    } else {
#endif
        return std::numeric_limits<T>::infinity();
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC float taylor_default_max_delta_t<float>();

template HEYOKA_DLL_PUBLIC double taylor_default_max_delta_t<double>();

template HEYOKA_DLL_PUBLIC long double taylor_default_max_delta_t<long double>();

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC mppp::real128 taylor_default_max_delta_t<mppp::real128>();

#endif

#if defined(HEYOKA_HAVE_REAL)

template HEYOKA_DLL_PUBLIC mppp::real taylor_default_max_delta_t<mppp::real>();

#endif

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
