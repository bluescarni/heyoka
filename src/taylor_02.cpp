// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

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
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/logging_impl.hpp>
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

// Function to split the central part of the decomposition (i.e., the definitions of the u variables
// that do not represent state variables) into parallelisable segments. Within a segment,
// the definition of a u variable does not depend on any u variable defined within that segment.
// NOTE: the hidden deps are not considered as dependencies.
// NOTE: the segments in the return value will contain shallow copies of the
// expressions in dc.
std::vector<taylor_dc_t> taylor_segment_dc(const taylor_dc_t &dc, std::uint32_t n_eq)
{
    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Helper that takes in input the definition ex of a u variable, and returns
    // in output the list of indices of the u variables on which ex depends.
    auto udef_args_indices = [](const expression &ex) -> std::vector<std::uint32_t> {
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

        if (std::any_of(u_indices.begin(), u_indices.end(),
                        [cur_limit_idx](auto idx) { return idx >= cur_limit_idx; })) {
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
            assert(std::all_of(u_indices.begin(), u_indices.end(),
                               [idx_limit = counter + n_eq](auto idx) { return idx < idx_limit; }));
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
                                                       llvm::GlobalVariable::InternalLinkage, sv_funcs_dc_arr);

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

                // Fetch from arr the derivative
                // of order 'order - 1' of the u variable at u_idx. The index is:
                // (order - 1) * n_uvars + u_idx.
                auto ret = taylor_fetch_diff(arr, u_idx, order - 1u, n_uvars);

                // We have to divide the derivative by order
                // to get the normalised derivative of the state variable.
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
                                                   llvm::GlobalVariable::InternalLinkage, var_indices_arr);

    auto *vars_arr = llvm::ConstantArray::get(var_arr_type, vars);
    auto *g_vars
        = new llvm::GlobalVariable(md, vars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, vars_arr);

    // Numbers.
    auto *num_indices_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(num_indices.size()));
    auto *num_indices_arr = llvm::ConstantArray::get(num_indices_arr_type, num_indices);
    auto *g_num_indices = new llvm::GlobalVariable(md, num_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, num_indices_arr);

    auto *nums_arr_type = llvm::ArrayType::get(fp_t, boost::numeric_cast<std::uint64_t>(nums.size()));
    auto *nums_arr = llvm::ConstantArray::get(nums_arr_type, nums);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *g_nums
        = new llvm::GlobalVariable(md, nums_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, nums_arr);

    // Params.
    auto *par_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(par_indices.size()));

    auto *par_indices_arr = llvm::ConstantArray::get(par_arr_type, par_indices);
    auto *g_par_indices = new llvm::GlobalVariable(md, par_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, par_indices_arr);

    auto *pars_arr = llvm::ConstantArray::get(par_arr_type, pars);
    auto *g_pars
        = new llvm::GlobalVariable(md, pars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, pars_arr);

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

// Helper to perform the computation of the Taylor derivatives in compact mode across
// multiple LLVM states.
auto taylor_compute_jet_multi(llvm_state &main_state, llvm::Type *main_fp_t, llvm::Value *main_par_ptr,
                              llvm::Value *main_time_ptr, llvm::Value *main_tape_ptr, const taylor_dc_t &dc,
                              const std::vector<taylor_dc_t> &s_dc, const std::vector<std::uint32_t> &sv_funcs_dc,
                              std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                              bool high_accuracy, bool parallel_mode, std::uint32_t max_svf_idx)
{
    // TODO implement.
    (void)parallel_mode;

    // Generate the global arrays for the computation of the derivatives
    // of the state variables in the main state.
    const auto svd_gl = taylor_c_make_sv_diff_globals(main_state, main_fp_t, dc, n_uvars);

    // Structure used to log, in trace mode, the breakdown of each segment.
    // For each segment, this structure contains the number of invocations
    // of each function in the segment. It will be unused if we are not tracing.
    std::vector<std::vector<std::uint32_t>> segment_bd;

    // Are we tracing?
    const auto is_tracing = get_logger()->should_log(spdlog::level::trace);

    // List of evaluation functions in a segment.
    //
    // This map contains a list of functions for the compact-mode evaluation of Taylor derivatives.
    // Each function is mapped to a pair, containing:
    //
    // - the number of times the function is to be invoked,
    // - a list of functors (generators) that generate the arguments for
    //   the invocation.
    //
    // NOTE: we use maps with name-based comparison for the functions. This ensures that the order in which these
    // functions are invoked is always the same. If we used directly pointer
    // comparisons instead, the order could vary across different executions and different platforms. The name
    // mangling we do when creating the function names should ensure that there are no possible name collisions.
    using seg_f_list_t
        = std::map<llvm::Function *, std::pair<std::uint32_t, std::vector<std::function<llvm::Value *(llvm::Value *)>>>,
                   llvm_func_name_compare>;

    // Init the list of states.
    // NOTE: we use lists here because it is convenient to have
    // pointer/reference stability when iteratively constructing
    // the set of states.
    std::list<llvm_state> states;

    // Push back a new state and use it as initial current state.
    // NOTE: like this, we always end up creating at least one driver
    // function and a state, even in the degenerate case of an empty decomposition,
    // which is suboptimal peformance-wise.
    // I do not think however that it is worth it to complicate the code to avoid
    // this corner-case pessimisation.
    states.push_back(main_state.make_similar());
    auto *cur_state = &states.back();

    // Index of the state we are currently operating on.
    boost::safe_numerics::safe<unsigned> cur_state_idx = 0;

    // Helper to create and return the prototype of a driver function in the state s.
    auto make_driver_proto = [](llvm_state &s, unsigned cur_idx) {
        auto &builder = s.builder();
        auto &md = s.module();
        auto &ctx = s.context();

        // The arguments to the driver are:
        // - a pointer to the tape,
        // - pointers to par and time,
        // - the current diff order.
        auto *ptr_tp = llvm::PointerType::getUnqual(ctx);
        std::vector<llvm::Type *> fargs{ptr_tp, ptr_tp, ptr_tp, builder.getInt32Ty()};

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

        return f;
    };

    // TODO doc fix.
    // Helper to compute the Taylor derivatives for a block.
    // func is the LLVM function for the computation of the Taylor derivative in the block,
    // ncalls the number of times it must be called, gens the generators for the
    // function arguments and cur_order the order of the derivative. s is the llvm state
    // in which we are computing the derivatives.
    auto block_diff = [n_uvars](llvm_state &s, llvm::Function *func, std::uint32_t ncalls, const auto &gens,
                                llvm::Value *tape_ptr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                llvm::Value *cur_order, llvm::Type *fp_vec_type) {
        // LCOV_EXCL_START
        assert(ncalls > 0u);
        assert(!gens.empty());
        assert(std::ranges::all_of(gens, [](const auto &f) { return static_cast<bool>(f); }));
        // LCOV_EXCL_STOP

        // Fetch the builder for the current state.
        auto &bld = s.builder();

        // We will be manually unrolling loops if ncalls is small enough.
        // This seems to help with compilation times.
        constexpr auto max_unroll_n = 5u;

        if (ncalls > max_unroll_n) {
            // Loop over the number of calls.
            llvm_loop_u32(s, bld.getInt32(0), bld.getInt32(ncalls), [&](llvm::Value *cur_call_idx) {
                // Create the u variable index from the first generator.
                auto u_idx = gens[0](cur_call_idx);

                // Initialise the vector of arguments with which func must be called. The following
                // initial arguments are always present:
                // - current Taylor order,
                // - u index of the variable,
                // - tape of derivatives,
                // - pointer to the param values,
                // - pointer to the time value(s).
                std::vector<llvm::Value *> args{cur_order, u_idx, tape_ptr, par_ptr, time_ptr};

                // Create the other arguments via the generators.
                for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                    args.push_back(gens[i](cur_call_idx));
                }

                // Calculate the derivative and store the result.
                taylor_c_store_diff(s, fp_vec_type, tape_ptr, n_uvars, cur_order, u_idx, bld.CreateCall(func, args));
            });
        } else {
            // The manually-unrolled version of the above.
            for (std::uint32_t idx = 0; idx < ncalls; ++idx) {
                auto *cur_call_idx = bld.getInt32(idx);
                auto u_idx = gens[0](cur_call_idx);
                std::vector<llvm::Value *> args{cur_order, u_idx, tape_ptr, par_ptr, time_ptr};

                for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                    args.push_back(gens[i](cur_call_idx));
                }

                taylor_c_store_diff(s, fp_vec_type, tape_ptr, n_uvars, cur_order, u_idx, bld.CreateCall(func, args));
            }
        }
    };

    // NOTE: unlike in compiled functions, we cannot at the same time
    // declare and invoke the drivers from the main module as the invocation
    // happens from within an LLVM loop. Thus, we first define the drivers
    // in the states and add their declarations in the main state, and only
    // at a later stage we perform the invocation of the drivers in the
    // main state.

    // Declarations of the drivers in the main state.
    std::vector<llvm::Function *> main_driver_decls;
    // Add the declaration for the first driver.
    main_driver_decls.push_back(make_driver_proto(main_state, cur_state_idx));

    // Add the driver declaration to the current state,
    // and start insertion into the driver.
    cur_state->builder().SetInsertPoint(
        llvm::BasicBlock::Create(cur_state->context(), "entry", make_driver_proto(*cur_state, cur_state_idx)));

    // Variable to keep track of how many blocks have been codegenned
    // in the current state.
    boost::safe_numerics::safe<unsigned> n_cg_blocks = 0;

    // Limit of codegenned blocks per state.
    // NOTE: this has not been really properly tuned,
    // needs more investigation.
    constexpr auto max_n_cg_blocks = 20u;

    // Variable to keep track of the u variable
    // on whose definition we are operating.
    auto cur_u_idx = n_eq;

    // Iterate over the segments in s_dc.
    for (const auto &seg : s_dc) {
        if (n_cg_blocks > max_n_cg_blocks) {
            // We have codegenned enough blocks for this state. Create the return
            // value for the current driver, and move to the next one.
            cur_state->builder().CreateRetVoid();

            // Create the new current state.
            states.push_back(main_state.make_similar());
            cur_state = &states.back();

            // Reset/update the counters.
            n_cg_blocks = 0;
            ++cur_state_idx;

            // Add the driver declaration to the main state.
            main_driver_decls.push_back(make_driver_proto(main_state, cur_state_idx));

            // Add the driver declaration to the current state,
            // and start insertion into the driver.
            cur_state->builder().SetInsertPoint(
                llvm::BasicBlock::Create(cur_state->context(), "entry", make_driver_proto(*cur_state, cur_state_idx)));
        }

        // Fetch the internal fp type and its vector counterpart for the current state.
        auto *fp_t = llvm_clone_type(*cur_state, main_fp_t);
        auto *fp_vec_type = make_vector_type(fp_t, batch_size);

        // Fetch the current builder.
        auto &cur_builder = cur_state->builder();

        // This structure maps a function to sets of arguments
        // with which the function is to be called. For instance, if function
        // f(x, y, z) is to be called as f(a, b, c) and f(d, e, f), then tmp_map
        // will contain {f : [[a, b, c], [d, e, f]]}.
        // After construction, we have verified that for each function
        // in the map the sets of arguments have all the same size.
        // NOTE: again, here and below we use name-based ordered maps for the functions.
        // This ensures that the invocations of cm_make_arg_gen_*(), which create several
        // global variables, always happen in a well-defined order. If we used an unordered map instead,
        // the variables would be created in a "random" order, which would result in a
        // unnecessary miss for the in-memory cache machinery when two logically-identical
        // LLVM modules are considered different because of the difference in the order
        // of declaration of global variables.
        std::map<llvm::Function *, std::vector<std::vector<std::variant<std::uint32_t, number>>>,
                 llvm_func_name_compare>
            tmp_map;

        for (const auto &ex : seg) {
            // Get the function for the computation of the derivative.
            auto *func = taylor_c_diff_func(*cur_state, fp_t, ex.first, n_uvars, batch_size, high_accuracy);

            // Insert the function into tmp_map.
            const auto [it, is_new_func] = tmp_map.try_emplace(func);

            assert(is_new_func || !it->second.empty()); // LCOV_EXCL_LINE

            // Convert the variables/constants in the current dc
            // element into a set of indices/constants.
            const auto cdiff_args = udef_to_variants(ex.first, ex.second);

            // LCOV_EXCL_START
            if (!is_new_func && it->second.back().size() - 1u != cdiff_args.size()) {
                throw std::invalid_argument(
                    fmt::format("Inconsistent arity detected in a Taylor derivative function in compact "
                                "mode: the same function is being called with both {} and {} arguments",
                                it->second.back().size() - 1u, cdiff_args.size()));
            }
            // LCOV_EXCL_STOP

            // Add the new set of arguments.
            it->second.emplace_back();
            // Add the idx of the u variable.
            it->second.back().emplace_back(cur_u_idx);
            // Add the actual function arguments.
            it->second.back().insert(it->second.back().end(), cdiff_args.begin(), cdiff_args.end());

            ++cur_u_idx;
        }

        // Now we build the transposition of tmp_map: from {f : [[a, b, c], [d, e, f]]}
        // to {f : [[a, d], [b, e], [c, f]]}.
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
            // NOTE: n_args must be at least 1 because the u idx
            // is prepended to the actual function arguments in
            // the tmp_map entries.
            assert(n_args >= 1u); // LCOV_EXCL_LINE

            for (decltype(vv[0].size()) i = 0; i < n_args; ++i) {
                // Build the vector of values corresponding
                // to the current argument index.
                std::vector<std::variant<std::uint32_t, number>> tmp_c_vec;
                for (decltype(vv.size()) j = 0; j < n_calls; ++j) {
                    tmp_c_vec.push_back(vv[j][i]);
                }

                // Turn tmp_c_vec (a vector of variants) into a variant
                // of vectors, and insert the result.
                it->second.push_back(vv_transpose(tmp_c_vec));
            }
        }

        // Create the seg_f_list_t for the current segment.
        seg_f_list_t seg_map;

        for (const auto &[func, vv] : tmp_map_transpose) {
            // NOTE: vv.size() is now the number of arguments. We know it cannot
            // be zero because the functions to compute the Taylor derivatives
            // in compact mode always have at least 1 argument (i.e., the index
            // of the u variable whose derivative is being computed).
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
                    [cur_state, fp_t](const auto &x) {
                        using type = uncvref_t<decltype(x)>;

                        if constexpr (std::is_same_v<type, std::vector<std::uint32_t>>) {
                            return cm_make_arg_gen_vidx(*cur_state, x);
                        } else {
                            return cm_make_arg_gen_vc(*cur_state, fp_t, x);
                        }
                    },
                    v));
            }
        }

        // Fetch the arguments from the driver prototype.
        auto *driver_f = cur_builder.GetInsertBlock()->getParent();
        auto *tape_ptr = driver_f->args().begin();
        auto *par_ptr = driver_f->args().begin() + 1;
        auto *time_ptr = driver_f->args().begin() + 2;
        auto *cur_order = driver_f->args().begin() + 3;

        // Compute the derivatives for this segment.
        for (const auto &[func, fpair] : seg_map) {
            const auto &[ncalls, gens] = fpair;

            block_diff(*cur_state, func, ncalls, gens, tape_ptr, par_ptr, time_ptr, cur_order, fp_vec_type);
        }

        // Update the number of codegenned blocks.
        n_cg_blocks += seg_map.size();

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

    // Back in the main state, we begin by invoking all the drivers with order zero.
    // That is, we are computing the initial values of the u variables.
    auto &main_bld = main_state.builder();
    for (auto *cur_driver_f : main_driver_decls) {
        main_bld.CreateCall(cur_driver_f, {main_tape_ptr, main_par_ptr, main_time_ptr, main_bld.getInt32(0)});
    }

    // Next, we compute all derivatives up to order 'order - 1'.
    llvm_loop_u32(main_state, main_bld.getInt32(1), main_bld.getInt32(order), [&](llvm::Value *cur_order) {
        // State variables first.
        taylor_c_compute_sv_diffs(main_state, main_fp_t, svd_gl, main_tape_ptr, main_par_ptr, n_uvars, cur_order,
                                  batch_size);

        // The other u variables.
        for (auto *cur_driver_f : main_driver_decls) {
            main_bld.CreateCall(cur_driver_f, {main_tape_ptr, main_par_ptr, main_time_ptr, cur_order});
        }
    });

    // Next, we compute the last-order derivatives for the state variables.
    taylor_c_compute_sv_diffs(main_state, main_fp_t, svd_gl, main_tape_ptr, main_par_ptr, n_uvars,
                              main_bld.getInt32(order), batch_size);

    // Compute the last-order derivatives for the sv_funcs, if any. Because the sv funcs
    // correspond to u variables somewhere in the decomposition, we will have to compute the
    // last-order derivatives of the u variables until we are sure all sv_funcs derivatives
    // have been properly computed.
    if (max_svf_idx >= n_eq) {
        // Monitor the starting index of the current
        // segment while iterating on the segments.
        auto cur_start_u_idx = n_eq;

        for (decltype(s_dc.size()) seg_idx = 0; seg_idx < s_dc.size(); ++seg_idx) {
            if (cur_start_u_idx > max_svf_idx) {
                // We computed all the necessary derivatives, break out.
                break;
            }

            // Invoke the driver for the current segment.
            main_bld.CreateCall(main_driver_decls[seg_idx],
                                {main_tape_ptr, main_par_ptr, main_time_ptr, main_bld.getInt32(order)});

            // Update cur_start_u_idx.
            cur_start_u_idx += static_cast<std::uint32_t>(s_dc[seg_idx].size());
        }
    }
}

// Helper for the computation of a jet of derivatives in compact mode,
// used in taylor_compute_jet(). The return value are the size/alignment
// requirements for the tape of derivatives. All LLVM values and types
// passed to this function are defined in the main state.
std::array<std::size_t, 2> taylor_compute_jet_compact_mode(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    llvm_state &main_state, llvm::Type *main_fp_t, llvm::Value *order0, llvm::Value *par_ptr, llvm::Value *time_ptr,
    llvm::Value *tape_ptr, const taylor_dc_t &dc, const std::vector<std::uint32_t> &sv_funcs_dc, std::uint32_t n_eq,
    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool parallel_mode)
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

    // Determine the total number of elements to be stored in the tape of derivatives.
    // We will be storing all the derivatives of the u variables
    // up to order 'order - 1', the derivatives of order
    // 'order' of the state variables and the derivatives
    // of order 'order' of the sv_funcs.
    // NOTE: if sv_funcs_dc is empty, or if all its indices are not greater
    // than the indices of the state variables, then we don't need additional
    // slots after the sv derivatives. If we need additional slots, allocate
    // another full column of derivatives, as it is complicated at this stage
    // to know exactly how many slots we will need.
    // NOTE: overflow checking for this computation has been performed externally.
    const auto tot_tape_N = (max_svf_idx < n_eq) ? (n_uvars * order + n_eq) : (n_uvars * (order + 1u));

    // Total required size in bytes for the tape.
    const auto tape_sz = boost::safe_numerics::safe<std::size_t>(get_size(main_md, main_fp_vec_t)) * tot_tape_N;

    // Tape alignment.
    const auto tape_al = boost::numeric_cast<std::size_t>(get_alignment(main_md, main_fp_vec_t));

    // Log the runtime of IR construction in trace mode.
    spdlog::stopwatch sw;

    // NOTE: tape_ptr is used as temporary storage for the current function,
    // but it is provided externally from dynamically-allocated memory in order to avoid stack overflow.
    // This creates a situation in which LLVM cannot elide stores into tape_ptr
    // (even if it figures out a way to avoid storing intermediate results into
    // it) because LLVM must assume that some other function may
    // use these stored values later. Thus, we declare via an intrinsic that the
    // lifetime of tape_ptr begins here and ends at the end of the function,
    // so that LLVM can assume that any value stored in it cannot be possibly
    // used outside this function.
    main_bld.CreateLifetimeStart(tape_ptr, main_bld.getInt64(tape_sz));

    // Copy over the order-0 derivatives of the state variables.
    // NOTE: overflow checking is already done in the parent function.
    llvm_loop_u32(main_state, main_bld.getInt32(0), main_bld.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
        // Fetch the pointer from order0.
        auto *ptr = main_bld.CreateInBoundsGEP(main_ext_fp_t, order0,
                                               main_bld.CreateMul(cur_var_idx, main_bld.getInt32(batch_size)));

        // Load as a vector.
        auto *vec = ext_load_vector_from_memory(main_state, main_fp_t, ptr, batch_size);

        // Store into tape_ptr.
        taylor_c_store_diff(main_state, main_fp_vec_t, tape_ptr, n_uvars, main_bld.getInt32(0), cur_var_idx, vec);
    });

    get_logger()->trace("Taylor IR creation compact mode runtime: {}", sw);

    // Return the array of derivatives of the u variables and its type.
    return std::make_pair(diff_arr, static_cast<llvm::Type *>(diff_array_type));
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

// Helper function to compute the jet of Taylor derivatives up to a given order. n_eq
// is the number of equations/variables in the ODE sys, dc its Taylor decomposition,
// n_uvars the total number of u variables in the decomposition.
// order is the max derivative order desired, batch_size the batch size.
// order0 is a pointer to an array of (at least) n_eq * batch_size scalar elements
// containing the derivatives of order 0. par_ptr is a pointer to an array containing
// the numerical values of the parameters, time_ptr a pointer to the time value(s),
// tape_ptr a pointer to the tape of derivatives (only in compact mode, otherwise
// a null value). sv_funcs are the indices, in the decomposition, of the functions of state
// variables.
//
// order0, par_ptr and time_ptr are all external pointers.
//
// The return value is a variant containing either:
// - in compact mode, the size/alignment requirements for the tape of derivatives,
// - otherwise, the jet of derivatives of the state variables and sv_funcs
//   up to order 'order'.
std::variant<std::array<std::size_t, 2>, std::vector<llvm::Value *>>
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

    // Make sure we can represent n_uvars * (order + 1) as a 32-bit
    // unsigned integer. This is the maximum total number of derivatives we will have to compute
    // and store, with the +1 taking into account the extra slots that might be needed by sv_funcs_dc.
    // If sv_funcs_dc is empty, we need only n_uvars * order + n_eq derivatives.
    // LCOV_EXCL_START
    if (order == std::numeric_limits<std::uint32_t>::max()
        || n_uvars > std::numeric_limits<std::uint32_t>::max() / (order + 1u)) {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives");
    }

    // We also need to be able to index up to n_eq * batch_size in order0.
    if (n_eq > std::numeric_limits<std::uint32_t>::max() / batch_size) {
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
        const auto max_svf_idx = sv_funcs_dc.empty() ? static_cast<std::uint32_t>(0)
                                                     : *std::max_element(sv_funcs_dc.begin(), sv_funcs_dc.end());

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
