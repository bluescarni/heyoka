// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/math_wrappers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/tfp.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Simplify a Taylor decomposition by removing
// common subexpressions.
std::vector<expression> taylor_decompose_cse(std::vector<expression> &v_ex, std::vector<expression>::size_type n_eq)
{
    // A Taylor decomposition is supposed
    // to have n_eq variables at the beginning,
    // n_eq variables at the end and possibly
    // extra variables in the middle.
    assert(v_ex.size() >= n_eq * 2u);

    using idx_t = std::vector<expression>::size_type;

    std::vector<expression> retval;

    // expression -> idx map. This will end up containing
    // all the unique expressions from v_ex, and it will
    // map them to their indices in retval (which will
    // in general differ from their indices in v_ex).
    std::unordered_map<expression, idx_t> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // Add the definitions of the first n_eq
    // variables in terms of u variables.
    // No need to modify anything here.
    for (idx_t i = 0; i < n_eq; ++i) {
        retval.emplace_back(std::move(v_ex[i]));
    }

    for (auto i = n_eq; i < v_ex.size() - n_eq; ++i) {
        auto &ex = v_ex[i];

        // Rename the u variables in ex.
        rename_variables(ex, uvars_rename);

        if (auto it = ex_map.find(ex); it == ex_map.end()) {
            // This is the first occurrence of ex in the
            // decomposition. Add it to retval.
            retval.emplace_back(ex);

            // Add ex to ex_map, mapping it to
            // the index it corresponds to in retval
            // (let's call it j).
            ex_map.emplace(std::move(ex), retval.size() - 1u);

            // Update uvars_rename. This will ensure that
            // occurrences of the variable 'u_i' in the next
            // elements of v_ex will be renamed to 'u_j'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace("u_" + li_to_string(i), "u_" + li_to_string(retval.size() - 1u));
            assert(res.second);
        } else {
            // ex is a redundant expression. This means
            // that it already appears in retval at index
            // it->second. Don't add anything to retval,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace("u_" + li_to_string(i), "u_" + li_to_string(it->second));
            assert(res.second);
        }
    }

    // Handle the derivatives of the state variables at the
    // end of the decomposition. We just need to ensure that
    // the u variables in their definitions are renamed with
    // the new indices.
    for (auto i = v_ex.size() - n_eq; i < v_ex.size(); ++i) {
        auto &ex = v_ex[i];

        rename_variables(ex, uvars_rename);

        retval.emplace_back(std::move(ex));
    }

    return retval;
}

#if !defined(NDEBUG)

// Helper to verify a Taylor decomposition.
void verify_taylor_dec(const std::vector<expression> &orig, const std::vector<expression> &dc)
{
    using idx_t = std::vector<expression>::size_type;

    const auto n_eq = orig.size();

    assert(dc.size() >= n_eq * 2u);

    // The first n_eq expressions of u variables
    // must be just variables.
    for (idx_t i = 0; i < n_eq; ++i) {
        assert(std::holds_alternative<variable>(dc[i].value()));
    }

    // From n_eq to dc.size() - n_eq, the expressions
    // must contain variables only in the u_n form,
    // where n < i.
    for (auto i = n_eq; i < dc.size() - n_eq; ++i) {
        for (const auto &var : get_variables(dc[i])) {
            assert(var.rfind("u_", 0) == 0);
            assert(uname_to_index(var) < i);
        }
    }

    // From dc.size() - n_eq to dc.size(), the expressions
    // must be either variables in the u_n form, where n < i,
    // or numbers.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        std::visit(
            [i](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    assert(v.name().rfind("u_", 0) == 0);
                    assert(uname_to_index(v.name()) < i);
                } else if (!std::is_same_v<type, number>) {
                    assert(false);
                }
            },
            dc[i].value());
    }

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of state variables or other u variables,
    // and store it in subs_map.
    for (idx_t i = 0; i < dc.size() - n_eq; ++i) {
        subs_map.emplace("u_" + li_to_string(i), subs(dc[i], subs_map));
    }

    // Reconstruct the right-hand sides of the system
    // and compare them to the original ones.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        assert(subs(dc[i], subs_map) == orig[i - (dc.size() - n_eq)]);
    }
}

#endif

} // namespace

} // namespace detail

// Taylor decomposition with automatic deduction
// of variables.
std::vector<expression> taylor_decompose(std::vector<expression> v_ex)
{
    if (v_ex.empty()) {
        throw std::invalid_argument("Cannot decompose a system of zero equations");
    }

    // Determine the variables in the system of equations.
    std::vector<std::string> vars;
    for (const auto &ex : v_ex) {
        auto ex_vars = get_variables(ex);
        vars.insert(vars.end(), std::make_move_iterator(ex_vars.begin()), std::make_move_iterator(ex_vars.end()));
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
    }

    if (vars.size() != v_ex.size()) {
        throw std::invalid_argument("The number of deduced variables for a Taylor decomposition ("
                                    + std::to_string(vars.size()) + ") differs from the number of equations ("
                                    + std::to_string(v_ex.size()) + ")");
    }

    // Cache the number of equations/variables
    // for later use.
    const auto n_eq = v_ex.size();

    // Create the map for renaming the variables to u_i.
    // The renaming will be done in alphabetical order.
    std::unordered_map<std::string, std::string> repl_map;
    for (decltype(vars.size()) i = 0; i < vars.size(); ++i) {
        [[maybe_unused]] const auto eres = repl_map.emplace(vars[i], "u_" + detail::li_to_string(i));
        assert(eres.second);
    }

#if !defined(NDEBUG)
    // Store a copy of the original system for checking later.
    const auto orig_v_ex = v_ex;
#endif

    // Rename the variables in the original equations.
    for (auto &ex : v_ex) {
        rename_variables(ex, repl_map);
    }

    // Init the vector containing the definitions
    // of the u variables. It begins with a list
    // of the original variables of the system.
    std::vector<expression> u_vars_defs;
    for (const auto &var : vars) {
        u_vars_defs.emplace_back(variable{var});
    }

    // Create a copy of the original equations in terms of u variables.
    // We will be reusing this below.
    auto v_ex_copy = v_ex;

    // Run the decomposition on each equation.
    for (decltype(v_ex.size()) i = 0; i < v_ex.size(); ++i) {
        // Decompose the current equation.
        if (const auto dres = taylor_decompose_in_place(std::move(v_ex[i]), u_vars_defs)) {
            // NOTE: if the equation was decomposed
            // (that is, it is not constant or a single variable),
            // we have to update the original definition
            // of the equation in v_ex_copy
            // so that it points to the u variable
            // that now represents it.
            v_ex_copy[i] = expression{variable{"u_" + detail::li_to_string(dres)}};
        }
    }

    // Append the (possibly updated) definitions of the diff equations
    // in terms of u variables.
    for (auto &ex : v_ex_copy) {
        u_vars_defs.emplace_back(std::move(ex));
    }

#if !defined(NDEBUG)
    // Verify the decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
#endif

    // Simplify the decomposition.
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
#endif

    return u_vars_defs;
}

// Taylor decomposition from lhs and rhs
// of a system of equations.
std::vector<expression> taylor_decompose(std::vector<std::pair<expression, expression>> sys)
{
    if (sys.empty()) {
        throw std::invalid_argument("Cannot decompose a system of zero equations");
    }

    // Determine the variables in the system of equations
    // from the lhs of the equations. We need to ensure that:
    // - all the lhs expressions are variables
    //   and there are no duplicates,
    // - all the variables in the rhs expressions
    //   appear in the lhs expressions.
    // Note that not all variables in the lhs
    // need to appear in the rhs.

    // This will eventually contain the list
    // of all variables in the system.
    std::vector<std::string> lhs_vars;
    // Maintain a set as well to check for duplicates.
    std::unordered_set<std::string> lhs_vars_set;
    // The set of variables in the rhs.
    std::unordered_set<std::string> rhs_vars_set;

    for (const auto &p : sys) {
        const auto &lhs = p.first;
        const auto &rhs = p.second;

        // Infer the variable from the current lhs.
        std::visit(
            [&lhs, &lhs_vars, &lhs_vars_set](const auto &v) {
                if constexpr (std::is_same_v<detail::uncvref_t<decltype(v)>, variable>) {
                    // Check if this is a duplicate variable.
                    if (auto res = lhs_vars_set.emplace(v.name()); res.second) {
                        // Not a duplicate, add it to lhs_vars.
                        lhs_vars.emplace_back(v.name());
                    } else {
                        // Duplicate, error out.
                        throw std::invalid_argument(
                            "Error in the Taylor decomposition of a system of equations: the variable '" + v.name()
                            + "' appears in the left-hand side twice");
                    }
                } else {
                    std::ostringstream oss;
                    oss << lhs;

                    throw std::invalid_argument("Error in the Taylor decomposition of a system of equations: the "
                                                "left-hand side contains the expression '"
                                                + oss.str() + "', which is not a variable");
                }
            },
            lhs.value());

        // Update the global list of variables
        // for the rhs.
        for (auto &var : get_variables(rhs)) {
            rhs_vars_set.emplace(std::move(var));
        }
    }

    // Check that all variables in the rhs appear in the lhs.
    for (const auto &var : rhs_vars_set) {
        if (lhs_vars_set.find(var) == lhs_vars_set.end()) {
            throw std::invalid_argument("Error in the Taylor decomposition of a system of equations: the variable '"
                                        + var + "' appears in the right-hand side but not in the left-hand side");
        }
    }

    // Cache the number of equations/variables.
    const auto n_eq = sys.size();
    assert(n_eq == lhs_vars.size());

    // Create the map for renaming the variables to u_i.
    // The renaming will be done following the order of the lhs
    // variables.
    std::unordered_map<std::string, std::string> repl_map;
    for (decltype(lhs_vars.size()) i = 0; i < lhs_vars.size(); ++i) {
        [[maybe_unused]] const auto eres = repl_map.emplace(lhs_vars[i], "u_" + detail::li_to_string(i));
        assert(eres.second);
    }

#if !defined(NDEBUG)
    // Store a copy of the original rhs for checking later.
    std::vector<expression> orig_rhs;
    for (const auto &[_, rhs_ex] : sys) {
        orig_rhs.push_back(rhs_ex);
    }
#endif

    // Rename the variables in the original equations.
    for (auto &[_, rhs_ex] : sys) {
        rename_variables(rhs_ex, repl_map);
    }

    // Init the vector containing the definitions
    // of the u variables. It begins with a list
    // of the original lhs variables of the system.
    std::vector<expression> u_vars_defs;
    for (const auto &var : lhs_vars) {
        u_vars_defs.emplace_back(variable{var});
    }

    // Create a copy of the original equations in terms of u variables.
    // We will be reusing this below.
    auto sys_copy = sys;

    // Run the decomposition on each equation.
    for (decltype(sys.size()) i = 0; i < sys.size(); ++i) {
        // Decompose the current equation.
        if (const auto dres = taylor_decompose_in_place(std::move(sys[i].second), u_vars_defs)) {
            // NOTE: if the equation was decomposed
            // (that is, it is not constant or a single variable),
            // we have to update the original definition
            // of the equation in sys_copy
            // so that it points to the u variable
            // that now represents it.
            sys_copy[i].second = expression{variable{"u_" + detail::li_to_string(dres)}};
        }
    }

    // Append the (possibly updated) definitions of the diff equations
    // in terms of u variables.
    for (auto &[_, rhs] : sys_copy) {
        u_vars_defs.emplace_back(std::move(rhs));
    }

#if !defined(NDEBUG)
    // Verify the decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
#endif

    // Simplify the decomposition.
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
#endif

    return u_vars_defs;
}

namespace detail
{

namespace
{

// Add a function to the llvm_state s for the evaluation
// of a polynomial via Estrin's scheme. The polynomial in question
// is the Taylor expansion that updates the state in a Taylor
// integrator at the end of the timestep. nvars is the number
// of variables in the ODE system, order is the Taylor order,
// batch_size the batch size (will be 1 in the scalar
// Taylor integrator, > 1 in the batch integrator).
template <typename T>
void taylor_add_estrin(llvm_state &s, const std::string &name, std::uint32_t nvars, std::uint32_t order,
                       std::uint32_t batch_size)
{
    assert(s.module().getNamedValue(name) == nullptr);

    auto &builder = s.builder();

    // Fetch the SIMD vector size from s.
    const auto vector_size = s.vector_size<T>();

    // Prepare the main function prototype. The arguments are:
    // - an output pointer into which we will be writing
    //   the updated state,
    // - an input pointer with the jet of derivatives
    //   (which also includes the current state at order 0),
    // - an input pointer with the integration timesteps.
    std::vector<llvm::Type *> fargs(3u, llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    assert(f != nullptr);

    // Setup the function arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::WriteOnly);
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto jet_ptr = out_ptr + 1;
    jet_ptr->setName("jet_ptr");
    jet_ptr->addAttr(llvm::Attribute::ReadOnly);
    jet_ptr->addAttr(llvm::Attribute::NoCapture);
    jet_ptr->addAttr(llvm::Attribute::NoAlias);

    auto h_ptr = jet_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::ReadOnly);
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Helper to run the Estrin scheme on the polynomial
    // whose coefficients are stored in cf_vec. This
    // will consume cf_vec.
    // https://en.wikipedia.org/wiki/Estrin%27s_scheme
    auto run_estrin = [&builder](std::vector<llvm::Value *> &cf_vec, llvm::Value *h) {
        assert(!cf_vec.empty());

        while (cf_vec.size() != 1u) {
            // Fill in the vector of coefficients for the next iteration.
            std::vector<llvm::Value *> new_cf_vec;

            for (decltype(cf_vec.size()) i = 0; i < cf_vec.size(); i += 2u) {
                if (i + 1u == cf_vec.size()) {
                    // We are at the last element of the vector
                    // and the size of the vector is odd. Just append
                    // the existing coefficient.
                    new_cf_vec.push_back(cf_vec[i]);
                } else {
                    new_cf_vec.push_back(builder.CreateFAdd(cf_vec[i], builder.CreateFMul(cf_vec[i + 1u], h)));
                }
            }

            // Replace the vector of coefficients
            // with the new one.
            new_cf_vec.swap(cf_vec);

            // Update h if we are not at the last iteration.
            if (cf_vec.size() != 1u) {
                h = builder.CreateFMul(h, h);
            }
        }

        return cf_vec[0];
    };

    if (vector_size == 0u) {
        // Scalar mode.
        for (std::uint32_t var_idx = 0; var_idx < nvars; ++var_idx) {
            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                // Load the polynomial coefficients from the jet
                // of derivatives.
                std::vector<llvm::Value *> cf_vec;
                for (std::uint32_t o = 0; o < order + 1u; ++o) {
                    auto cf_ptr = builder.CreateInBoundsGEP(
                        jet_ptr, builder.getInt32(o * nvars * batch_size + var_idx * batch_size + batch_idx),
                        "cf_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx) + "_" + li_to_string(o) + "_ptr");
                    cf_vec.emplace_back(builder.CreateLoad(
                        cf_ptr, "cf_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx) + "_" + li_to_string(o)));
                }

                // Load the integration timestep. This is common to all
                // variables and varies only by batch_idx.
                llvm::Value *h = builder.CreateLoad(builder.CreateInBoundsGEP(h_ptr, builder.getInt32(batch_idx),
                                                                              "h_" + li_to_string(batch_idx) + "_ptr"),
                                                    "h_" + li_to_string(batch_idx));

                // Run the Estrin scheme.
                auto eval = run_estrin(cf_vec, h);

                // Store the result of the evaluation.
                auto res_ptr = builder.CreateInBoundsGEP(out_ptr, builder.getInt32(var_idx * batch_size + batch_idx),
                                                         "res_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx)
                                                             + "_ptr");
                builder.CreateStore(eval, res_ptr);
            }
        }
    } else {
        // Vector mode.
        const auto n_sub_batch = batch_size / vector_size;

        for (std::uint32_t var_idx = 0; var_idx < nvars; ++var_idx) {
            for (std::uint32_t batch_idx = 0; batch_idx < n_sub_batch * vector_size; batch_idx += vector_size) {
                std::vector<llvm::Value *> cf_vec;
                for (std::uint32_t o = 0; o < order + 1u; ++o) {
                    auto cf_ptr = builder.CreateInBoundsGEP(
                        jet_ptr, builder.getInt32(o * nvars * batch_size + var_idx * batch_size + batch_idx),
                        "cf_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx) + "_" + li_to_string(o) + "_ptr");
                    cf_vec.emplace_back(load_vector_from_memory(builder, cf_ptr, vector_size));
                }

                llvm::Value *h
                    = load_vector_from_memory(builder,
                                              builder.CreateInBoundsGEP(h_ptr, builder.getInt32(batch_idx),
                                                                        "h_" + li_to_string(batch_idx) + "_ptr"),
                                              vector_size);

                auto eval = run_estrin(cf_vec, h);

                auto res_ptr = builder.CreateInBoundsGEP(out_ptr, builder.getInt32(var_idx * batch_size + batch_idx),
                                                         "res_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx)
                                                             + "_ptr");

                detail::store_vector_to_memory(builder, res_ptr, eval, vector_size);
            }

            for (std::uint32_t batch_idx = n_sub_batch * vector_size; batch_idx < batch_size; ++batch_idx) {
                std::vector<llvm::Value *> cf_vec;
                for (std::uint32_t o = 0; o < order + 1u; ++o) {
                    auto cf_ptr = builder.CreateInBoundsGEP(
                        jet_ptr, builder.getInt32(o * nvars * batch_size + var_idx * batch_size + batch_idx),
                        "cf_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx) + "_" + li_to_string(o) + "_ptr");
                    cf_vec.emplace_back(builder.CreateLoad(
                        cf_ptr, "cf_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx) + "_" + li_to_string(o)));
                }

                llvm::Value *h = builder.CreateLoad(builder.CreateInBoundsGEP(h_ptr, builder.getInt32(batch_idx),
                                                                              "h_" + li_to_string(batch_idx) + "_ptr"),
                                                    "h_" + li_to_string(batch_idx));

                auto eval = run_estrin(cf_vec, h);

                auto res_ptr = builder.CreateInBoundsGEP(out_ptr, builder.getInt32(var_idx * batch_size + batch_idx),
                                                         "res_" + li_to_string(var_idx) + "_" + li_to_string(batch_idx)
                                                             + "_ptr");
                builder.CreateStore(eval, res_ptr);
            }
        }
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(name);
}

} // namespace

template <typename T>
template <typename U>
void taylor_adaptive_impl<T>::finalise_ctor_impl(U sys, std::vector<T> state, T time, T tol, bool high_accuracy)
{
    // Assign the data members.
    m_state = std::move(state);
    m_time = time;

    // Check input params.
    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() != sys.size()) {
        throw std::invalid_argument("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                                    "integrator: the state vector has a dimension of "
                                    + std::to_string(m_state.size()) + ", while the number of equations is "
                                    + std::to_string(sys.size()));
    }

    if (!detail::isfinite(m_time)) {
        throw std::invalid_argument("Cannot initialise an adaptive Taylor integrator with a non-finite initial time of "
                                    + detail::li_to_string(m_time));
    }

    if (!detail::isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is " + li_to_string(tol)
            + " instead");
    }

    // Add the stepper function.
    m_dc = taylor_add_adaptive_step<T>(m_llvm, "step", std::move(sys), tol, 1, high_accuracy);

    // Run the jit.
    m_llvm.compile();

    // Fetch the stepper.
    m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(const taylor_adaptive_impl &other)
    // NOTE: make a manual copy of all members, apart from the function pointer.
    : m_state(other.m_state), m_time(other.m_time), m_llvm(other.m_llvm), m_dc(other.m_dc)
{
    m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(taylor_adaptive_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_impl<T> &taylor_adaptive_impl<T>::operator=(const taylor_adaptive_impl &other)
{
    if (this != &other) {
        *this = taylor_adaptive_impl(other);
    }

    return *this;
}

template <typename T>
taylor_adaptive_impl<T> &taylor_adaptive_impl<T>::operator=(taylor_adaptive_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_impl<T>::~taylor_adaptive_impl() = default;

// Implementation detail to make a single integration timestep.
// The magnitude of the timestep is automatically deduced, but it will
// always be not greater than abs(max_delta_t). The propagation
// is done forward in time if max_delta_t >= 0, backwards in
// time otherwise.
//
// The function will return a pair, containing
// a flag describing the outcome of the integration,
// abd the integration timestep that was used.
//
// NOTE: the safer adaptive timestep from
// Jorba still needs to be implemented.
template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step_impl(T max_delta_t)
{
    // Check the current state before invoking the stepper.
    if (std::any_of(m_state.cbegin(), m_state.cend(), [](const auto &x) { return !detail::isfinite(x); })) {
        return std::tuple{taylor_outcome::err_nf_state, T(0)};
    }

    // Invoke the stepper.
    auto h = max_delta_t;
    m_step_f(m_state.data(), &h);

    // Update the time.
    m_time += h;

    return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, h};
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step()
{
    // NOTE: time limit +inf means integration forward in time
    // and no time limit.
    return step_impl(std::numeric_limits<T>::infinity());
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step_backward()
{
    return step_impl(-std::numeric_limits<T>::infinity());
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step(T max_delta_t)
{
    using std::isnan;

    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A NaN max_delta_t was passed to the step() function of an adaptive Taylor integrator");
    }

    return step_impl(max_delta_t);
}

template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t> taylor_adaptive_impl<T>::propagate_for(T delta_t, std::size_t max_steps)
{
    return propagate_until(m_time + delta_t, max_steps);
}

template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t> taylor_adaptive_impl<T>::propagate_until(T t, std::size_t max_steps)
{
    if (!detail::isfinite(t)) {
        throw std::invalid_argument(
            "A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

    // Initial values for the counter,
    // the min/max abs of the integration
    // timesteps, and min/max Taylor orders.
    std::size_t step_counter = 0;
    T min_h = std::numeric_limits<T>::infinity(), max_h(0);

    if (t == m_time) {
        return std::tuple{taylor_outcome::time_limit, min_h, max_h, step_counter};
    }

    if ((t > m_time && !detail::isfinite(t - m_time)) || (t < m_time && !detail::isfinite(m_time - t))) {
        throw std::overflow_error("The time limit passed to the propagate_until() function is too large and it "
                                  "results in an overflow condition");
    }

    if (t > m_time) {
        while (true) {
            const auto [res, h] = step_impl(t - m_time);

            if (res != taylor_outcome::success && res != taylor_outcome::time_limit) {
                return std::tuple{res, min_h, max_h, step_counter};
            }

            // Update the number of steps
            // completed successfully.
            ++step_counter;

            // Break out if the time limit is reached,
            // *before* updating the min_h/max_h values.
            if (m_time >= t) {
                break;
            }

            // Update min_h/max_h.
            assert(h >= 0);
            min_h = std::min(min_h, h);
            max_h = std::max(max_h, h);

            // Check the max number of steps stopping criterion.
            if (max_steps != 0u && step_counter == max_steps) {
                return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter};
            }
        }
    } else {
        while (true) {
            const auto [res, h] = step_impl(t - m_time);

            if (res != taylor_outcome::success && res != taylor_outcome::time_limit) {
                return std::tuple{res, min_h, max_h, step_counter};
            }

            ++step_counter;

            if (m_time <= t) {
                break;
            }

            assert(h < 0);
            min_h = std::min(min_h, -h);
            max_h = std::max(max_h, -h);

            if (max_steps != 0u && step_counter == max_steps) {
                return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter};
            }
        }
    }

    return std::tuple{taylor_outcome::time_limit, min_h, max_h, step_counter};
}

template <typename T>
void taylor_adaptive_impl<T>::set_time(T t)
{
    if (!detail::isfinite(t)) {
        throw std::invalid_argument("Non-finite time " + detail::li_to_string(t)
                                    + " passed to the set_time() function of an adaptive Taylor integrator");
    }

    m_time = t;
}

template <typename T>
void taylor_adaptive_impl<T>::set_state(const std::vector<T> &state)
{
    if (&state == &m_state) {
        // Check that state and m_state are not the same object,
        // otherwise std::copy() cannot be used.
        return;
    }

    if (state.size() != m_state.size()) {
        throw std::invalid_argument(
            "The state vector passed to the set_state() function of an adaptive Taylor integrator has a size of "
            + std::to_string(state.size()) + ", which is inconsistent with the size of the current state vector ("
            + std::to_string(m_state.size()) + ")");
    }

    if (std::any_of(state.begin(), state.end(), [](const T &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite state vector was passed to the set_state() function of an adaptive Taylor integrator");
    }

    // Do the copy.
    std::copy(state.begin(), state.end(), m_state.begin());
}

template <typename T>
const llvm_state &taylor_adaptive_impl<T>::get_llvm_state() const
{
    return m_llvm;
}

template <typename T>
const std::vector<expression> &taylor_adaptive_impl<T>::get_decomposition() const
{
    return m_dc;
}

// Explicit instantiation of the implementation classes/functions.
template class taylor_adaptive_impl<double>;
template void taylor_adaptive_impl<double>::finalise_ctor_impl(std::vector<expression>, std::vector<double>, double,
                                                               double, bool);
template void taylor_adaptive_impl<double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                               std::vector<double>, double, double, bool);
template class taylor_adaptive_impl<long double>;
template void taylor_adaptive_impl<long double>::finalise_ctor_impl(std::vector<expression>, std::vector<long double>,
                                                                    long double, long double, bool);
template void taylor_adaptive_impl<long double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                                    std::vector<long double>, long double, long double,
                                                                    bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_impl<mppp::real128>;
template void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(std::vector<expression>,
                                                                      std::vector<mppp::real128>, mppp::real128,
                                                                      mppp::real128, bool);
template void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                                      std::vector<mppp::real128>, mppp::real128,
                                                                      mppp::real128, bool);

#endif

} // namespace detail

namespace detail
{

template <typename T>
template <typename U>
void taylor_adaptive_batch_impl<T>::finalise_ctor_impl(U sys, std::vector<T> states, std::uint32_t batch_size,
                                                       std::vector<T> times, T rtol, T atol, unsigned opt_level)
{
    using std::ceil;
    using std::exp;
    using std::log;

    // Init the data members.
    m_batch_size = batch_size;
    m_states = std::move(states);
    m_times = std::move(times);
    m_rtol = rtol;
    m_atol = atol;

    // Check input params.
    if (m_batch_size == 0u) {
        throw std::invalid_argument("The batch size in an adaptive Taylor integrator cannot be zero");
    }

    if (std::any_of(m_states.begin(), m_states.end(), [](const auto &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial states of an adaptive Taylor integrator");
    }

    if (m_states.size() % m_batch_size != 0u) {
        throw std::invalid_argument("Invalid size detected in the initialization of an adaptive Taylor "
                                    "integrator: the states vector has a size of "
                                    + std::to_string(m_states.size()) + ", which is not a multiple of the batch size ("
                                    + std::to_string(m_batch_size) + ")");
    }

    if (m_states.size() / m_batch_size != sys.size()) {
        throw std::invalid_argument("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                                    "integrator: the states vector has a dimension of "
                                    + std::to_string(m_states.size() / m_batch_size)
                                    + ", while the number of equations is " + std::to_string(sys.size()));
    }

    if (m_times.size() != m_batch_size) {
        throw std::invalid_argument("Invalid size detected in the initialization of an adaptive Taylor "
                                    "integrator: the times vector has a size of "
                                    + std::to_string(m_times.size()) + ", which is not equal to the batch size ("
                                    + std::to_string(m_batch_size) + ")");
    }

    if (std::any_of(m_times.begin(), m_times.end(), [](const auto &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite initial time was detected in the initialisation of an adaptive Taylor integrator");
    }

    if (!detail::isfinite(m_rtol) || m_rtol <= 0) {
        throw std::invalid_argument(
            "The relative tolerance in an adaptive Taylor integrator must be finite and positive, but it is "
            + li_to_string(m_rtol) + " instead");
    }

    if (!detail::isfinite(m_atol) || m_atol <= 0) {
        throw std::invalid_argument(
            "The absolute tolerance in an adaptive Taylor integrator must be finite and positive, but it is "
            + li_to_string(m_atol) + " instead");
    }

    // Compute the two possible orders for the integration, ensuring that
    // they are at least 2.
    const auto order_r_f = std::max(T(2), ceil(-log(m_rtol) / 2 + 1));
    const auto order_a_f = std::max(T(2), ceil(-log(m_atol) / 2 + 1));

    if (!detail::isfinite(order_r_f) || !detail::isfinite(order_a_f)) {
        throw std::invalid_argument(
            "The computation of the Taylor orders in an adaptive Taylor integrator produced non-finite values");
    }
    // NOTE: static cast is safe because we know that T is at least
    // a double-precision IEEE type.
    if (order_r_f > static_cast<T>(std::numeric_limits<std::uint32_t>::max())
        || order_a_f > static_cast<T>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("The computation of the max Taylor orders in an adaptive Taylor integrator resulted "
                                  "in an overflow condition");
    }
    // Record the Taylor orders.
    m_order_r = static_cast<std::uint32_t>(order_r_f);
    m_order_a = static_cast<std::uint32_t>(order_a_f);

    // Record the number of variables
    // before consuming sys.
    const auto n_vars = sys.size();

    // Add the functions for computing
    // the jet of normalised derivatives.
    m_dc = m_llvm.add_taylor_jet_batch<T>("jet_r", sys, m_order_r, batch_size);
    if (m_order_r != m_order_a) {
        // NOTE: add the absolute tolerance jet function only
        // if the relative and absolute orders differ.
        m_llvm.add_taylor_jet_batch<T>("jet_a", std::move(sys), m_order_a, batch_size);
    }

    // Add the functions to update the state vector.
    // NOTE: static cast is safe because we successfully
    // added the functions for the derivatives.
    taylor_add_estrin<T>(m_llvm, "estrin_r", static_cast<std::uint32_t>(n_vars), m_order_r, batch_size);
    if (m_order_r != m_order_a) {
        taylor_add_estrin<T>(m_llvm, "estrin_a", static_cast<std::uint32_t>(n_vars), m_order_a, batch_size);
    }

    // Change the optimisation level
    // and run the optimisation pass.
    m_llvm.opt_level() = opt_level;
    m_llvm.optimise();

    // Run the jit.
    m_llvm.compile();

    // Fetch the compiled functions for computing
    // the jet of derivatives.
    m_jet_f_r = m_llvm.fetch_taylor_jet_batch<T>("jet_r");
    if (m_order_r == m_order_a) {
        m_jet_f_a = m_jet_f_r;
    } else {
        m_jet_f_a = m_llvm.fetch_taylor_jet_batch<T>("jet_a");
    }

    // Fetch the function for updating the state vector
    // at the end of the integration timestep.
    m_update_f_r = reinterpret_cast<s_update_f_t>(m_llvm.jit_lookup("estrin_r"));
    if (m_order_r == m_order_a) {
        m_update_f_a = m_update_f_r;
    } else {
        m_update_f_a = reinterpret_cast<s_update_f_t>(m_llvm.jit_lookup("estrin_a"));
    }

    // Init the jet vector. Its maximum size is n_vars * (max_order + 1) * batch_size.
    // NOTE: n_vars must be nonzero because we successfully
    // created a Taylor jet function from sys.
    using jet_size_t = decltype(m_jet.size());
    const auto max_order = std::max(m_order_r, m_order_a);
    if (max_order >= std::numeric_limits<jet_size_t>::max()
        || (static_cast<jet_size_t>(max_order) + 1u) > std::numeric_limits<jet_size_t>::max() / n_vars
        || (static_cast<jet_size_t>(max_order) + 1u) * n_vars > std::numeric_limits<jet_size_t>::max() / batch_size) {
        throw std::overflow_error("The computation of the size of the jet of derivatives in an adaptive Taylor "
                                  "integrator resulted in an overflow condition");
    }
    m_jet.resize((static_cast<jet_size_t>(max_order) + 1u) * n_vars * batch_size);

    // Check the values of the derivatives
    // for the initial state.

    // Copy the current state to the order zero
    // of the jet of derivatives.
    auto jet_ptr = m_jet.data();
    std::copy(m_states.begin(), m_states.end(), jet_ptr);

    // Compute the jet of derivatives at max order.
    if (m_order_r > m_order_a) {
        m_jet_f_r(jet_ptr);
    } else {
        m_jet_f_a(jet_ptr);
    }

    // Check the computed derivatives, starting from order 1.
    if (std::any_of(jet_ptr + (n_vars * batch_size), jet_ptr + m_jet.size(),
                    [](const T &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "Non-finite value(s) detected in the jet of derivatives corresponding to the initial "
            "state of an adaptive batch Taylor integrator");
    }

    // Pre-compute the inverse orders. This spares
    // us a few divisions in the stepping function.
    m_inv_order.resize(static_cast<jet_size_t>(max_order) + 1u);
    for (jet_size_t i = 1; i < max_order + 1u; ++i) {
        m_inv_order[i] = 1 / static_cast<T>(i);
    }

    // Pre-compute the factors by which rho must
    // be multiplied in order to determine the
    // integration timestep.
    m_rhofac_r = 1 / (exp(T(1)) * exp(T(1))) * exp((T(-7) / T(10)) / (m_order_r - 1u));
    m_rhofac_a = 1 / (exp(T(1)) * exp(T(1))) * exp((T(-7) / T(10)) / (m_order_a - 1u));

    // Prepare the temporary variables for use in the
    // stepping functions.
    m_max_abs_states.resize(static_cast<jet_size_t>(m_batch_size));
    m_use_abs_tol.resize(static_cast<decltype(m_use_abs_tol.size())>(m_batch_size));
    m_max_abs_diff_om1.resize(static_cast<jet_size_t>(m_batch_size));
    m_max_abs_diff_o.resize(static_cast<jet_size_t>(m_batch_size));
    m_rho_om1.resize(static_cast<jet_size_t>(m_batch_size));
    m_rho_o.resize(static_cast<jet_size_t>(m_batch_size));
    m_h.resize(static_cast<jet_size_t>(m_batch_size));
    m_pinf.resize(static_cast<jet_size_t>(m_batch_size), std::numeric_limits<T>::infinity());
    m_minf.resize(static_cast<jet_size_t>(m_batch_size), -std::numeric_limits<T>::infinity());
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &other)
    // NOTE: make a manual copy of all members, apart from the function pointers.
    : m_batch_size(other.m_batch_size), m_states(other.m_states), m_times(other.m_times), m_rtol(other.m_rtol),
      m_atol(other.m_atol), m_order_r(other.m_order_r), m_order_a(other.m_order_a), m_inv_order(other.m_inv_order),
      m_rhofac_r(other.m_rhofac_r), m_rhofac_a(other.m_rhofac_a), m_llvm(other.m_llvm), m_jet(other.m_jet),
      m_dc(other.m_dc), m_max_abs_states(other.m_max_abs_states), m_use_abs_tol(other.m_use_abs_tol),
      m_max_abs_diff_om1(other.m_max_abs_diff_om1), m_max_abs_diff_o(other.m_max_abs_diff_o),
      m_rho_om1(other.m_rho_om1), m_rho_o(other.m_rho_o), m_h(other.m_h), m_pinf(other.m_pinf), m_minf(other.m_minf)
{
    // Fetch the compiled functions for computing
    // the jet of derivatives.
    m_jet_f_r = m_llvm.fetch_taylor_jet_batch<T>("jet_r");
    if (m_order_r == m_order_a) {
        m_jet_f_a = m_jet_f_r;
    } else {
        m_jet_f_a = m_llvm.fetch_taylor_jet_batch<T>("jet_a");
    }

    // Same for the functions for the state update.
    m_update_f_r = reinterpret_cast<s_update_f_t>(m_llvm.jit_lookup("estrin_r"));
    if (m_order_r == m_order_a) {
        m_update_f_a = m_update_f_r;
    } else {
        m_update_f_a = reinterpret_cast<s_update_f_t>(m_llvm.jit_lookup("estrin_a"));
    }
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl(taylor_adaptive_batch_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_batch_impl<T> &taylor_adaptive_batch_impl<T>::operator=(const taylor_adaptive_batch_impl &other)
{
    if (this != &other) {
        *this = taylor_adaptive_batch_impl(other);
    }

    return *this;
}

template <typename T>
taylor_adaptive_batch_impl<T> &
taylor_adaptive_batch_impl<T>::operator=(taylor_adaptive_batch_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_batch_impl<T>::~taylor_adaptive_batch_impl() = default;

// Implementation detail to make a single integration timestep.
// The magnitude of the timestep is automatically deduced for each
// state vector, but it will always be not greater than
// the absolute value of the corresponding element in max_delta_ts.
// For each state vector, the propagation
// is done forward in time if max_delta_t >= 0, backwards in
// time otherwise.
//
// The function will write to res a triple for each state
// vector, containing a flag describing the outcome of the integration,
// the integration timestep that was used and the
// Taylor order that was used.
//
// NOTE: the safer adaptive timestep from
// Jorba still needs to be implemented.
template <typename T>
void taylor_adaptive_batch_impl<T>::step_impl(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &res,
                                              const std::vector<T> &max_delta_ts)
{
    using std::abs;
    using std::isnan;
    using std::pow;

    // Check preconditions.
    assert(std::none_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) { return isnan(x); }));
    assert(max_delta_ts.size() == m_batch_size);

    // Cache locally the batch size.
    const auto batch_size = m_batch_size;

    // Prepare res.
    res.resize(batch_size);
    std::fill(res.begin(), res.end(), std::tuple{taylor_outcome::success, T(0), std::uint32_t(0)});

    // Cache the number of variables in the system.
    assert(m_states.size() % batch_size == 0u);
    const auto nvars = static_cast<std::uint32_t>(m_states.size() / batch_size);

    // Compute the norm infinity of each state vector.
    assert(m_max_abs_states.size() == batch_size);
    std::fill(m_max_abs_states.begin(), m_max_abs_states.end(), T(0));
    for (std::uint32_t i = 0; i < nvars; ++i) {
        for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const auto s_idx = i * batch_size + batch_idx;

            if (detail::isfinite(m_states[s_idx])) {
                m_max_abs_states[batch_idx] = std::max(m_max_abs_states[batch_idx], abs(m_states[s_idx]));
            } else {
                // Mark the current state vector as non-finite in res.
                // NOTE: the timestep and order have already
                // been set to zero via the fill() above.
                std::get<0>(res[batch_idx]) = taylor_outcome::err_nf_state;
            }
        }
    }

    // Compute the Taylor order for this timestep.
    // For each state vector, we determine the Taylor
    // order based on the norm infinity, and we take the
    // maximum.
    // NOTE: this means that we might end up using a higher
    // order than necessary in some elements of the batch.
    assert(m_use_abs_tol.size() == batch_size);
    std::uint32_t max_order = 0;
    for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (std::get<0>(res[batch_idx]) != taylor_outcome::success) {
            // If the current state vector is not finite, skip it
            // for the purpose of determining the max order.
            continue;
        }

        const auto use_abs_tol = m_rtol * m_max_abs_states[batch_idx] <= m_atol;
        const auto cur_order = use_abs_tol ? m_order_a : m_order_r;
        max_order = std::max(max_order, cur_order);

        // Record whether we are using absolute or relative
        // tolerance for this element of the batch.
        m_use_abs_tol[batch_idx] = use_abs_tol;
    }

    if (max_order == 0u) {
        // If max_order is still zero, it means that all state vectors
        // contain non-finite values. Exit.
        return;
    }

    assert(max_order >= 2u);

    // Copy the current state to the order zero
    // of the jet of derivatives.
    auto jet_ptr = m_jet.data();
    std::copy(m_states.begin(), m_states.end(), jet_ptr);

    // Compute the jet of derivatives.
    // NOTE: this will be computed to the max order.
    if (max_order == m_order_a) {
        m_jet_f_a(jet_ptr);
    } else {
        m_jet_f_r(jet_ptr);
    }

    // Now we compute an estimation of the radius of convergence of the Taylor
    // series at orders 'order' and 'order - 1'. We start by computing
    // the norm infinity of the derivatives at orders 'order - 1' and
    // 'order'.
    assert(m_max_abs_diff_om1.size() == batch_size);
    assert(m_max_abs_diff_o.size() == batch_size);
    std::fill(m_max_abs_diff_om1.begin(), m_max_abs_diff_om1.end(), T(0));
    std::fill(m_max_abs_diff_o.begin(), m_max_abs_diff_o.end(), T(0));
    for (std::uint32_t i = 0; i < nvars; ++i) {
        for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (std::get<0>(res[batch_idx]) != taylor_outcome::success) {
                // If the current state is not finite or resulted in non-finite
                // derivatives, skip it.
                continue;
            }

            // Establish if we are using absolute or relative
            // tolerance for this state vector.
            const auto use_abs_tol = m_use_abs_tol[batch_idx];

            // Determine the order for the current state vector.
            const auto cur_order = use_abs_tol ? m_order_a : m_order_r;

            // Load the values of the derivatives.
            const auto diff_om1 = jet_ptr[(cur_order - 1u) * nvars * batch_size + i * batch_size + batch_idx];
            const auto diff_o = jet_ptr[cur_order * nvars * batch_size + i * batch_size + batch_idx];

            if (!detail::isfinite(diff_om1) || !detail::isfinite(diff_o)) {
                // If the current state resulted in non-finite derivatives,
                // mark it and skip it.
                std::get<0>(res[batch_idx]) = taylor_outcome::err_nf_derivative;

                continue;
            }

            // Update the max abs.
            m_max_abs_diff_om1[batch_idx] = std::max(m_max_abs_diff_om1[batch_idx], abs(diff_om1));
            m_max_abs_diff_o[batch_idx] = std::max(m_max_abs_diff_o[batch_idx], abs(diff_o));
        }
    }

    // Estimate rho at orders 'order - 1' and 'order',
    // and compute the integration timestep.
    assert(m_rho_om1.size() == batch_size);
    assert(m_rho_o.size() == batch_size);
    assert(m_h.size() == batch_size);
    for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (std::get<0>(res[batch_idx]) != taylor_outcome::success) {
            // If the current state is non finite or it resulted
            // in non-finite derivatives, set the timestep to
            // zero and skip it.
            m_h[batch_idx] = 0;

            continue;
        }

        // Establish if we are using absolute or relative
        // tolerance for this state vector.
        const auto use_abs_tol = m_use_abs_tol[batch_idx];

        // Determine the order for the current state vector.
        const auto cur_order = use_abs_tol ? m_order_a : m_order_r;

        // Compute the rhos.
        const auto rho_om1 = use_abs_tol ? pow(1 / m_max_abs_diff_om1[batch_idx], m_inv_order[cur_order - 1u])
                                         : pow(m_max_abs_states[batch_idx] / m_max_abs_diff_om1[batch_idx],
                                               m_inv_order[cur_order - 1u]);
        const auto rho_o = use_abs_tol
                               ? pow(1 / m_max_abs_diff_o[batch_idx], m_inv_order[cur_order])
                               : pow(m_max_abs_states[batch_idx] / m_max_abs_diff_o[batch_idx], m_inv_order[cur_order]);

        if (isnan(rho_om1) || isnan(rho_o)) {
            // Mark the presence of NaN rho in res.
            std::get<0>(res[batch_idx]) = taylor_outcome::err_nan_rho;

            // Set the timestep to zero.
            m_h[batch_idx] = 0;
        } else {
            // Compute the minimum.
            const auto rho_m = std::min(rho_o, rho_om1);

            // Compute the timestep.
            auto h = rho_m * (use_abs_tol ? m_rhofac_a : m_rhofac_r);

            // Make sure h does not exceed abs(max_delta_t).
            const auto abs_delta_t = abs(max_delta_ts[batch_idx]);
            if (h > abs_delta_t) {
                h = abs_delta_t;
                std::get<0>(res[batch_idx]) = taylor_outcome::time_limit;
            }

            if (max_delta_ts[batch_idx] < T(0)) {
                // When propagating backwards, invert the sign of the timestep.
                h = -h;
            }

            // Store the integration timestep
            // for the current state vector.
            m_h[batch_idx] = h;
        }
    }

    // Update the state.
    // NOTE: this will update the state using the max order.
    if (max_order == m_order_a) {
        m_update_f_a(m_states.data(), jet_ptr, m_h.data());
    } else {
        m_update_f_r(m_states.data(), jet_ptr, m_h.data());
    }

    // Update the times, store the timesteps and order in res.
    for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (std::get<0>(res[batch_idx]) != taylor_outcome::success) {
            // If some failure mode was detected, don't update
            // the times or the return values.
            continue;
        }

        m_times[batch_idx] += m_h[batch_idx];
        std::get<1>(res[batch_idx]) = m_h[batch_idx];
        std::get<2>(res[batch_idx]) = m_use_abs_tol[batch_idx] ? m_order_a : m_order_r;
    }
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &res)
{
    return step_impl(res, m_pinf);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step_backward(std::vector<std::tuple<taylor_outcome, T, std::uint32_t>> &res)
{
    return step_impl(res, m_minf);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::set_times(const std::vector<T> &t)
{
    if (&t == &m_times) {
        // Check that t and m_times are not the same object,
        // otherwise std::copy() cannot be used.
        return;
    }

    if (t.size() != m_times.size()) {
        throw std::invalid_argument("Inconsistent sizes when setting the times in a batch Taylor integrator: the new "
                                    "times vector has a size of "
                                    + std::to_string(t.size()) + ", while the existing times vector has a size of "
                                    + std::to_string(m_times.size()));
    }

    if (std::any_of(t.begin(), t.end(), [](const auto &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite time value was detected while setting the times in a batch Taylor integrator");
    }

    // Do the copy.
    std::copy(t.begin(), t.end(), m_times.begin());
}

template <typename T>
void taylor_adaptive_batch_impl<T>::set_states(const std::vector<T> &states)
{
    if (&states == &m_states) {
        // Check that states and m_states are not the same object,
        // otherwise std::copy() cannot be used.
        return;
    }

    if (states.size() != m_states.size()) {
        throw std::invalid_argument("The states vector passed to the set_states() function of an adaptive batch Taylor "
                                    "integrator has a size of "
                                    + std::to_string(states.size())
                                    + ", which is inconsistent with the size of the current states vector ("
                                    + std::to_string(m_states.size()) + ")");
    }

    if (std::any_of(states.begin(), states.end(), [](const T &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument("A non-finite states vector was passed to the set_states() function of an adaptive "
                                    "batch Taylor integrator");
    }

    // Do the copy.
    std::copy(states.begin(), states.end(), m_states.begin());
}

template <typename T>
const llvm_state &taylor_adaptive_batch_impl<T>::get_llvm_state() const
{
    return m_llvm;
}

template <typename T>
const std::vector<expression> &taylor_adaptive_batch_impl<T>::get_decomposition() const
{
    return m_dc;
}

// Explicit instantiation of the batch implementation classes.
template class taylor_adaptive_batch_impl<double>;
template void taylor_adaptive_batch_impl<double>::finalise_ctor_impl(std::vector<expression>, std::vector<double>,
                                                                     std::uint32_t, std::vector<double>, double, double,
                                                                     unsigned);
template void taylor_adaptive_batch_impl<double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                                     std::vector<double>, std::uint32_t,
                                                                     std::vector<double>, double, double, unsigned);

template class taylor_adaptive_batch_impl<long double>;
template void taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(std::vector<expression>,
                                                                          std::vector<long double>, std::uint32_t,
                                                                          std::vector<long double>, long double,
                                                                          long double, unsigned);
template void taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(
    std::vector<std::pair<expression, expression>>, std::vector<long double>, std::uint32_t, std::vector<long double>,
    long double, long double, unsigned);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_batch_impl<mppp::real128>;
template void taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(std::vector<expression>,
                                                                            std::vector<mppp::real128>, std::uint32_t,
                                                                            std::vector<mppp::real128>, mppp::real128,
                                                                            mppp::real128, unsigned);
template void taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(
    std::vector<std::pair<expression, expression>>, std::vector<mppp::real128>, std::uint32_t,
    std::vector<mppp::real128>, mppp::real128, mppp::real128, unsigned);

#endif

} // namespace detail

namespace detail
{

llvm::Value *taylor_load_diff(llvm_state &s, llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                              llvm::Value *u_idx)
{
    auto &builder = s.builder();
    // TODO overflow check.
    auto ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx)});
    return builder.CreateLoad(ptr);
}

namespace
{

// RAII helper to temporarily disable most fast math flags that might
// be set in an LLVM builder. On destruction, the original fast math
// flags will be restored.
struct fm_disabler {
    llvm_state &m_s;
    llvm::FastMathFlags m_orig_fmf;

    explicit fm_disabler(llvm_state &s) : m_s(s), m_orig_fmf(m_s.builder().getFastMathFlags())
    {
        // Set the new flags (allow only fp contract).
        llvm::FastMathFlags fmf;
        fmf.setAllowContract();
        m_s.builder().setFastMathFlags(fmf);
    }
    ~fm_disabler()
    {
        // Restore the original flags.
        m_s.builder().setFastMathFlags(m_orig_fmf);
    }
};

// Compute the derivative of order "order" of a state variable.
// ex is the formula for the first-order derivative of the state variable (which
// is either a u variable or a number), n_uvars the number of variables in
// the decomposition, arr the array containing the derivatives of all u variables
// up to order - 1.
template <typename T>
tfp taylor_compute_sv_diff(llvm_state &s, const expression &ex, const std::vector<tfp> &arr, std::uint32_t n_uvars,
                           std::uint32_t order, std::uint32_t batch_size, bool high_accuracy)
{
    assert(order > 0u);

    return std::visit(
        [&](const auto &v) -> tfp {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, variable>) {
                // Extract the index of the u variable in the expression
                // of the first-order derivative.
                const auto u_idx = uname_to_index(v.name());

                // Fetch from arr the derivative
                // of order 'order - 1' of the u variable at u_idx. The index is:
                // (order - 1) * n_uvars + u_idx.
                auto ret = taylor_load_derivative(arr, u_idx, order - 1u, n_uvars);

                // We have to divide the derivative by order
                // to get the normalised derivative of the state variable.
                return tfp_div(s, ret, tfp_constant<T>(s, number(static_cast<T>(order)), batch_size, high_accuracy));
            } else if constexpr (std::is_same_v<type, number>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0. No need for normalization as the only
                // nonzero value that can be produced here is the first-order
                // derivative.
                return tfp_constant<T>(s, (order == 1u) ? v : number{0.}, batch_size, high_accuracy);
            } else {
                assert(false);
                return tfp{};
            }
        },
        ex.value());
}

// Helper function to compute the jet of Taylor derivatives up to a given order. n_eq
// is the number of equations/variables in the ODE sys, dc its Taylor decomposition,
// n_uvars the total number of u variables in the decomposition.
// order is the max derivative order desired, batch_size the batch size, high_accuracy
// specifies whether to use extended precision techniques in the computation.
// order0 contains the zero order derivatives of the state variables.
//
// The return value is the jet of derivatives of all u variables up to order 'order - 1',
// plus the derivatives of order 'order' of the state variables.
template <typename T>
auto taylor_compute_jet(llvm_state &s, std::vector<tfp> order0, const std::vector<expression> &dc, std::uint32_t n_eq,
                        std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size, bool high_accuracy)
{
    assert(order0.size() == n_eq);

    // Init the return value with the order 0 of the state variables.
    auto retval(std::move(order0));

    // Compute the order-0 derivatives of the other u variables.
    for (auto i = n_eq; i < n_uvars; ++i) {
        retval.push_back(taylor_u_init<T>(s, dc[i], retval, batch_size, high_accuracy));
    }

    // Compute the derivatives order by order, starting from 1 to order excluded.
    // We will compute the highest derivatives of the state variables separately
    // in the last step.
    for (std::uint32_t cur_order = 1; cur_order < order; ++cur_order) {
        // Begin with the state variables.
        // NOTE: the derivatives of the state variables
        // are at the end of the decomposition vector.
        for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
            retval.push_back(
                taylor_compute_sv_diff<T>(s, dc[i], retval, n_uvars, cur_order, batch_size, high_accuracy));
        }

        // Now the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            retval.push_back(taylor_diff<T>(s, dc[i], retval, n_uvars, cur_order, i, batch_size, high_accuracy));
        }
    }

    // Compute the last-order derivatives for the state variables.
    for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
        retval.push_back(taylor_compute_sv_diff<T>(s, dc[i], retval, n_uvars, order, batch_size, high_accuracy));
    }

    assert(retval.size() == static_cast<decltype(retval.size())>(n_uvars) * order + n_eq);

    return retval;
}

// Given an input pointer 'in', load the first n * batch_size values in it as n tfp values
// with vector size batch_size.
template <typename T>
auto taylor_load_values_as_tfp(llvm_state &s, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size,
                               bool high_accuracy)
{
    assert(batch_size > 0u);

    // Overflow check.
    if (n > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("Overflow while loading values as tfps");
    }

    auto &builder = s.builder();

    std::vector<tfp> retval;
    for (std::uint32_t i = 0; i < n; ++i) {
        // Fetch the pointer from in.
        auto ptr = builder.CreateInBoundsGEP(in, {builder.getInt32(i * batch_size)});

        // Load the value in vector mode.
        auto v = load_vector_from_memory(builder, ptr, batch_size);

        // Create the tfp and add it to retval.
        retval.push_back(tfp_from_vector(s, v, high_accuracy));
    }

    return retval;
}

// Given an input pointer 'in', load the first n * batch_size values in it as n
// LLVM vectors of size batch_size.
auto taylor_load_values(llvm_state &s, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    // Overflow check.
    if (n > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        // TODO fix message
        throw std::overflow_error("Overflow while loading values as tfps");
    }

    auto &builder = s.builder();

    std::vector<llvm::Value *> retval;
    for (std::uint32_t i = 0; i < n; ++i) {
        // Fetch the pointer from in.
        auto ptr = builder.CreateInBoundsGEP(in, {builder.getInt32(i * batch_size)});

        // Load the value in vector mode.
        retval.push_back(load_vector_from_memory(builder, ptr, batch_size));
    }

    return retval;
}

void taylor_store_diff(llvm_state &s, llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                       std::uint32_t u_idx, llvm::Value *val)
{
    auto &builder = s.builder();
    // TODO overflow check.
    auto ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), builder.getInt32(u_idx))});
    builder.CreateStore(val, ptr);
}

template <typename T>
llvm::Value *taylor_calculate_sv_diff(llvm_state &s, const expression &ex, llvm::Value *diff_arr, std::uint32_t n_uvars,
                                      llvm::Value *order, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    return std::visit(
        [&](const auto &v) -> llvm::Value * {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, variable>) {
                // Extract the index of the u variable in the expression
                // of the first-order derivative.
                const auto u_idx = uname_to_index(v.name());

                // Fetch from arr the derivative of order 'order - 1' of the u variable u_idx.
                auto ret = taylor_load_diff(s, diff_arr, n_uvars, builder.CreateSub(order, builder.getInt32(1)),
                                            builder.getInt32(u_idx));

                // We have to divide the derivative by 'order'
                // to get the normalised derivative of the state variable.
                return builder.CreateFDiv(
                    ret, create_constant_vector(builder, builder.CreateUIToFP(order, to_llvm_type<T>(s.context())),
                                                batch_size));
            } else if constexpr (std::is_same_v<type, number>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0. No need for normalization as the only
                // nonzero value that can be produced here is the first-order
                // derivative.
                auto cmp_cond = builder.CreateICmpEQ(order, builder.getInt32(1));
                return builder.CreateSelect(cmp_cond, create_constant_vector(builder, codegen<T>(s, v), batch_size),
                                            create_constant_vector(builder, codegen<T>(s, number{0.}), batch_size));
            } else {
                assert(false);
                return nullptr;
            }
        },
        ex.value());
}

// TODO fix docs.
// Helper function to compute the jet of Taylor derivatives up to a given order. n_eq
// is the number of equations/variables in the ODE sys, dc its Taylor decomposition,
// n_uvars the total number of u variables in the decomposition.
// order is the max derivative order desired, batch_size the batch size, high_accuracy
// specifies whether to use extended precision techniques in the computation.
// order0 contains the zero order derivatives of the state variables.
//
// The return value is the jet of derivatives of all u variables up to order 'order - 1',
// plus the derivatives of order 'order' of the state variables.
template <typename T>
llvm::Value *taylor_calculate_jet(llvm_state &s, llvm::Function *func, const std::vector<llvm::Value *> &order0,
                                  const std::vector<expression> &dc, std::uint32_t n_eq, std::uint32_t n_uvars,
                                  std::uint32_t order, std::uint32_t batch_size, bool high_accuracy)
{
    assert(order0.size() == n_eq);
    assert(n_eq > 0u);

    auto &builder = s.builder();

    // Prepare the array that will contain the jet of derivatives.
    // We will be storing all the derivatives of the u variables
    // up to order 'order - 1', plus the derivatives of order
    // 'order' of the state variables only.
    // TODO overflow check.
    auto array_type = llvm::ArrayType::get(order0[0]->getType(), static_cast<std::uint64_t>(n_uvars * order + n_eq));
    auto diff_arr = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type, 0, "diff_arr"),
                                              {builder.getInt32(0), builder.getInt32(0)});

    // Copy over the order0 derivatives of the state variables.
    for (std::uint32_t i = 0; i < n_eq; ++i) {
        taylor_store_diff(s, diff_arr, n_uvars, builder.getInt32(0), i, order0[i]);
    }

    // Run the init for the other u variables.
    for (auto i = n_eq; i < n_uvars; ++i) {
        auto val = taylor_init<T>(s, dc[i], diff_arr, batch_size, high_accuracy);
        taylor_store_diff(s, diff_arr, n_uvars, builder.getInt32(0), i, val);
    }

    // Compute all derivatives up to order 'order - 1'.
    llvm_loop_u32(s, func, builder.getInt32(1), builder.getInt32(order), [&](llvm::Value *cur_order) {
        // Begin with the state variables.
        // NOTE: the derivatives of the state variables
        // are at the end of the decomposition vector.
        for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
            taylor_store_diff(s, diff_arr, n_uvars, cur_order, i - n_uvars,
                              taylor_calculate_sv_diff<T>(s, dc[i], diff_arr, n_uvars, cur_order, batch_size));
        }

        // Now the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            taylor_store_diff(s, diff_arr, n_uvars, cur_order, i,
                              taylor_diff2<T>(s, dc[i], diff_arr, n_uvars, cur_order, i, batch_size, high_accuracy));
        }
    });

    // Compute the last-order derivatives for the state variables.
    for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
        taylor_store_diff(
            s, diff_arr, n_uvars, builder.getInt32(order), i - n_uvars,
            taylor_calculate_sv_diff<T>(s, dc[i], diff_arr, n_uvars, builder.getInt32(order), batch_size));
    }

    return diff_arr;
}

template <typename T, typename U>
auto taylor_add_jet_impl(llvm_state &s, const std::string &name, U sys, std::uint32_t order, std::uint32_t batch_size,
                         bool high_accuracy)
{
    if (s.is_compiled()) {
        throw std::invalid_argument("A function for the computation of the jet of Taylor derivatives cannot be added "
                                    "to an llvm_state after compilation");
    }

    if (order == 0u) {
        throw std::invalid_argument("The order of a Taylor jet cannot be zero");
    }

    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size of a Taylor jet cannot be zero");
    }

    auto &builder = s.builder();

    // NOTE: in high accuracy mode we need
    // to disable fast math flags in the builder.
    std::optional<fm_disabler> fmd;
    if (high_accuracy) {
        fmd.emplace(s);
    }

    // Record the number of equations/variables.
    const auto n_eq = sys.size();

    // Decompose the system of equations.
    auto dc = taylor_decompose(std::move(sys));

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = dc.size() - n_eq;

    // Prepare the function prototype. The only argument is a float pointer to in/out array.
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(to_llvm_type<T>(s.context()))};
    // The function does not return anything.
    auto ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument(
            "Unable to create a function for the computation of the jet of Taylor derivatives with name '" + name
            + "'");
    }

    // Set the name of the function argument.
    auto in_out = f->args().begin();
    in_out->setName("in_out");

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Load the order0 values of the state variables from the input pointer.
    const auto t_order0 = taylor_load_values(s, in_out, boost::numeric_cast<std::uint32_t>(n_eq), batch_size);

    // Calculate the jet of derivatives.
    auto diff_arr
        = taylor_calculate_jet<T>(s, f, t_order0, dc, boost::numeric_cast<std::uint32_t>(n_eq),
                                  boost::numeric_cast<std::uint32_t>(n_uvars), order, batch_size, high_accuracy);

    // Write the derivatives to in_out.
    // TODO overflow checking.
    llvm_loop_u32(
        s, f, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
        [&](llvm::Value *cur_order) {
            for (std::uint32_t sv_idx = 0; sv_idx < n_eq; ++sv_idx) {
                // Load the vector.
                auto vec_val = taylor_load_diff(s, diff_arr, boost::numeric_cast<std::uint32_t>(n_uvars), cur_order,
                                                builder.getInt32(sv_idx));

                // Index in the output array: n_eq * batch_size * cur_order + sv_idx * batch_size.
                auto out_idx = builder.CreateAdd(
                    builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(n_eq * batch_size)), cur_order),
                    builder.getInt32(sv_idx * batch_size));

                // Store it into in_out.
                auto out_ptr = builder.CreateInBoundsGEP(in_out, {out_idx});
                store_vector_to_memory(builder, out_ptr, vec_val, batch_size);
            }
        });

    // Finish off the function.
    s.builder().CreateRetVoid();

    // Verify it.
    s.verify_function(f);

    // Run the optimisation pass.
    s.optimise();

    return dc;
}

} // namespace

} // namespace detail

std::vector<expression> taylor_add_jet_dbl(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                           std::uint32_t order, std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_jet_impl<double>(s, name, std::move(sys), order, batch_size, high_accuracy);
}

std::vector<expression> taylor_add_jet_ldbl(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                            std::uint32_t order, std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_jet_impl<long double>(s, name, std::move(sys), order, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_jet_f128(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                            std::uint32_t order, std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_jet_impl<mppp::real128>(s, name, std::move(sys), order, batch_size, high_accuracy);
}

#endif

std::vector<expression> taylor_add_jet_dbl(llvm_state &s, const std::string &name,
                                           std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                           std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_jet_impl<double>(s, name, std::move(sys), order, batch_size, high_accuracy);
}

std::vector<expression> taylor_add_jet_ldbl(llvm_state &s, const std::string &name,
                                            std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                            std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_jet_impl<long double>(s, name, std::move(sys), order, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_jet_f128(llvm_state &s, const std::string &name,
                                            std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                            std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_jet_impl<mppp::real128>(s, name, std::move(sys), order, batch_size, high_accuracy);
}

#endif

namespace detail
{

namespace
{

// Helper to compute max(x_v, abs(y_v)) in the Taylor stepper implementation.
llvm::Value *taylor_step_maxabs(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto x_t = llvm::cast<llvm::VectorType>(x_v->getType())->getElementType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v), y_scalars = vector_to_scalars(builder, y_v);

        // Execute the heyoka_maxabs128() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_maxabs128", llvm::Type::getFP128Ty(s.context()), {x_scalars[i], y_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        // Compute abs(b).
        auto abs_y_v = llvm_invoke_intrinsic(s, "llvm.fabs", {y_v->getType()}, {y_v});
        // Return max(a, abs(b)).
        return llvm_invoke_intrinsic(s, "llvm.maxnum", {x_v->getType()}, {x_v, abs_y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to compute min(x_v, abs(y_v)) in the Taylor stepper implementation.
llvm::Value *taylor_step_minabs(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto x_t = llvm::cast<llvm::VectorType>(x_v->getType())->getElementType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v), y_scalars = vector_to_scalars(builder, y_v);

        // Execute the heyoka_minabs128() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_minabs128", llvm::Type::getFP128Ty(s.context()), {x_scalars[i], y_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        // Compute abs(b).
        auto abs_y_v = llvm_invoke_intrinsic(s, "llvm.fabs", {y_v->getType()}, {y_v});
        // Return min(a, abs(b)).
        return llvm_invoke_intrinsic(s, "llvm.minnum", {x_v->getType()}, {x_v, abs_y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to compute min(x_v, y_v) in the Taylor stepper implementation.
llvm::Value *taylor_step_min(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto x_t = llvm::cast<llvm::VectorType>(x_v->getType())->getElementType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v), y_scalars = vector_to_scalars(builder, y_v);

        // Execute the heyoka_minnum128() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_minnum128", llvm::Type::getFP128Ty(s.context()), {x_scalars[i], y_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        return llvm_invoke_intrinsic(s, "llvm.minnum", {x_v->getType()}, {x_v, y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to compute pow(x_v, y_v) in the Taylor stepper implementation.
llvm::Value *taylor_step_pow(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto x_t = llvm::cast<llvm::VectorType>(x_v->getType())->getElementType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v), y_scalars = vector_to_scalars(builder, y_v);

        // Execute the pow() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (decltype(x_scalars.size()) i = 0; i < x_scalars.size(); ++i) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_pow128", llvm::Type::getFP128Ty(s.context()), {x_scalars[i], y_scalars[i]},
                // NOTE: in theory we may add ReadNone here as well,
                // but for some reason, at least up to LLVM 10,
                // this causes strange codegen issues. Revisit
                // in the future.
                {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn}));
        }

        // Reconstruct the return value as a vector.
        return scalars_to_vector(builder, res_scalars);
    } else {
#endif
        return llvm_invoke_intrinsic(s, "llvm.pow", {x_v->getType()}, {x_v, y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to run the Estrin scheme on the polynomial
// whose coefficients are stored in cf_vec, with evaluation
// value h. This will consume cf_vec.
// https://en.wikipedia.org/wiki/Estrin%27s_scheme
tfp taylor_run_estrin(llvm_state &s, std::vector<tfp> &cf_vec, tfp h)
{
    assert(!cf_vec.empty());

    if (cf_vec.size() == std::numeric_limits<decltype(cf_vec.size())>::max()) {
        throw std::overflow_error("Overflow error in taylor_run_estrin()");
    }

    while (cf_vec.size() != 1u) {
        // Fill in the vector of coefficients for the next iteration.
        std::vector<tfp> new_cf_vec;

        for (decltype(cf_vec.size()) i = 0; i < cf_vec.size(); i += 2u) {
            if (i + 1u == cf_vec.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing coefficient.
                new_cf_vec.push_back(cf_vec[i]);
            } else {
                new_cf_vec.push_back(tfp_add(s, cf_vec[i], tfp_mul(s, cf_vec[i + 1u], h)));
            }
        }

        // Replace the vector of coefficients
        // with the new one.
        new_cf_vec.swap(cf_vec);

        // Update h if we are not at the last iteration.
        if (cf_vec.size() != 1u) {
            h = tfp_mul(s, h, h);
        }
    }

    return cf_vec[0];
}

template <typename T, typename U>
auto taylor_add_adaptive_step_impl(llvm_state &s, const std::string &name, U sys, T tol, std::uint32_t batch_size,
                                   bool high_accuracy)
{
    using std::ceil;
    using std::exp;
    using std::log;

    if (s.is_compiled()) {
        throw std::invalid_argument("An adaptive Taylor stepper cannot be added to an llvm_state after compilation");
    }

    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size of a Taylor stepper cannot be zero");
    }

    if (!isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(
            "The tolerance in an adaptive Taylor stepper must be finite and positive, but it is " + li_to_string(tol)
            + " instead");
    }

    // Determine the order from the tolerance.
    // NOTE: minimum order is 2.
    auto order_f = std::max(T(2), ceil(-log(tol) / 2 + 1));
    if (high_accuracy) {
        // Add 10% more order in high accuracy mode.
        order_f += order_f * (T(10) / 100);
    }

    if (!detail::isfinite(order_f)) {
        throw std::invalid_argument(
            "The computation of the Taylor order in an adaptive Taylor stepper produced non-finite values");
    }
    // NOTE: static cast is safe because we know that T is at least
    // a double-precision IEEE type.
    if (order_f > static_cast<T>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("The computation of the Taylor order in an adaptive Taylor stepper resulted "
                                  "in an overflow condition");
    }
    const auto order = static_cast<std::uint32_t>(order_f);

    // NOTE: in high accuracy mode we need
    // to disable fast math flags in the builder.
    std::optional<fm_disabler> fmd;
    if (high_accuracy) {
        fmd.emplace(s);
    }

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    auto dc = taylor_decompose(std::move(sys));

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    auto &builder = s.builder();

    // Prepare the function prototype. The arguments are:
    // - pointer to the current state vector (read & write),
    // - pointer to the array of max timesteps (read & write).
    // These pointers cannot overlap.
    std::vector<llvm::Type *> fargs(2, llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument("Unable to create a function for an adaptive Taylor stepper with name '" + name
                                    + "'");
    }

    // Set the name/attributes of the function argument.
    auto state_ptr = f->args().begin();
    state_ptr->setName("state_ptr");
    state_ptr->addAttr(llvm::Attribute::NoCapture);
    state_ptr->addAttr(llvm::Attribute::NoAlias);

    auto h_ptr = state_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Load the order zero derivatives from the input pointer.
    auto order0_arr = taylor_load_values_as_tfp<T>(s, &*state_ptr, n_eq, batch_size, high_accuracy);

    // Compute the norm infinity of the state vector.
    auto max_abs_state = create_constant_vector(builder, codegen<T>(s, number{0.}), batch_size);
    for (std::uint32_t i = 0; i < n_eq; ++i) {
        max_abs_state = taylor_step_maxabs(s, max_abs_state, tfp_to_vector(s, order0_arr[i]));
    }

    // Compute the jet of derivatives at the given order.
    auto diff_arr
        = taylor_compute_jet<T>(s, std::move(order0_arr), dc, n_eq, n_uvars, order, batch_size, high_accuracy);

    // Determine the norm infinity of the derivatives
    // at orders order and order - 1.
    auto max_abs_diff_o = create_constant_vector(builder, codegen<T>(s, number{0.}), batch_size);
    auto max_abs_diff_om1 = create_constant_vector(builder, codegen<T>(s, number{0.}), batch_size);
    for (std::uint32_t i = 0; i < n_eq; ++i) {
        max_abs_diff_o = taylor_step_maxabs(s, max_abs_diff_o,
                                            tfp_to_vector(s, taylor_load_derivative(diff_arr, i, order, n_uvars)));
        max_abs_diff_om1 = taylor_step_maxabs(
            s, max_abs_diff_om1, tfp_to_vector(s, taylor_load_derivative(diff_arr, i, order - 1u, n_uvars)));
    }

    // Determine if we are in absolute or relative tolerance mode.
    auto tol_v = create_constant_vector(builder, codegen<T>(s, number{tol}), batch_size);
    auto abs_or_rel = builder.CreateFCmpOLE(builder.CreateFMul(tol_v, max_abs_state), tol_v);

    // Estimate rho at orders order - 1 and order.
    auto num_rho = builder.CreateSelect(
        abs_or_rel, create_constant_vector(builder, codegen<T>(s, number{1.}), batch_size), max_abs_state);
    auto rho_o = taylor_step_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_o),
                                 create_constant_vector(builder, codegen<T>(s, number{T(1) / order}), batch_size));
    auto rho_om1
        = taylor_step_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_om1),
                          create_constant_vector(builder, codegen<T>(s, number{T(1) / (order - 1u)}), batch_size));

    // Take the minimum.
    auto rho_m = taylor_step_min(s, rho_o, rho_om1);

    // Copmute the safety factor.
    const auto rhofac = 1 / (exp(T(1)) * exp(T(1))) * exp((T(-7) / T(10)) / (order - 1u));

    // Determine the step size.
    auto h = builder.CreateFMul(rho_m, create_constant_vector(builder, codegen<T>(s, number{rhofac}), batch_size));

    // Ensure that the step size does not exceed the limit.
    auto max_h_vec = load_vector_from_memory(builder, h_ptr, batch_size);
    h = taylor_step_minabs(s, h, max_h_vec);

    // Handle backwards propagation.
    auto backward
        = builder.CreateFCmpOLT(max_h_vec, create_constant_vector(builder, codegen<T>(s, number{0.}), batch_size));
    auto h_fac = builder.CreateSelect(backward, create_constant_vector(builder, codegen<T>(s, number{-1.}), batch_size),
                                      create_constant_vector(builder, codegen<T>(s, number{1.}), batch_size));
    h = builder.CreateFMul(h_fac, h);

    // Run the time stepper for each variable, store the results.
    for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
        std::vector<tfp> cf_vec;

        if (order == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Overflow error in an adaptive Taylor stepper: the order is too high");
        }
        for (std::uint32_t o = 0; o <= order; ++o) {
            cf_vec.push_back(taylor_load_derivative(diff_arr, var_idx, o, n_uvars));
        }

        auto new_state = taylor_run_estrin(s, cf_vec, tfp_from_vector(s, h, high_accuracy));

        if (var_idx > std::numeric_limits<std::uint32_t>::max() / batch_size) {
            throw std::overflow_error("Overflow error in an adaptive Taylor stepper: too many variables");
        }
        store_vector_to_memory(builder, builder.CreateInBoundsGEP(state_ptr, builder.getInt32(var_idx * batch_size)),
                               tfp_to_vector(s, new_state), batch_size);
    }

    // Store the timesteps that were used.
    store_vector_to_memory(builder, h_ptr, h, batch_size);

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(name);

    // Run the optimisation pass.
    s.optimise();

    return dc;
}

} // namespace

} // namespace detail

std::vector<expression> taylor_add_adaptive_step_dbl(llvm_state &s, const std::string &name,
                                                     std::vector<expression> sys, double tol, std::uint32_t batch_size,
                                                     bool high_accuracy)
{
    return detail::taylor_add_adaptive_step_impl<double>(s, name, std::move(sys), tol, batch_size, high_accuracy);
}

std::vector<expression> taylor_add_adaptive_step_ldbl(llvm_state &s, const std::string &name,
                                                      std::vector<expression> sys, long double tol,
                                                      std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_adaptive_step_impl<long double>(s, name, std::move(sys), tol, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_adaptive_step_f128(llvm_state &s, const std::string &name,
                                                      std::vector<expression> sys, mppp::real128 tol,
                                                      std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_adaptive_step_impl<mppp::real128>(s, name, std::move(sys), tol, batch_size,
                                                                high_accuracy);
}

#endif

std::vector<expression> taylor_add_adaptive_step_dbl(llvm_state &s, const std::string &name,
                                                     std::vector<std::pair<expression, expression>> sys, double tol,
                                                     std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_adaptive_step_impl<double>(s, name, std::move(sys), tol, batch_size, high_accuracy);
}

std::vector<expression> taylor_add_adaptive_step_ldbl(llvm_state &s, const std::string &name,
                                                      std::vector<std::pair<expression, expression>> sys,
                                                      long double tol, std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_adaptive_step_impl<long double>(s, name, std::move(sys), tol, batch_size, high_accuracy);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_adaptive_step_f128(llvm_state &s, const std::string &name,
                                                      std::vector<std::pair<expression, expression>> sys,
                                                      mppp::real128 tol, std::uint32_t batch_size, bool high_accuracy)
{
    return detail::taylor_add_adaptive_step_impl<mppp::real128>(s, name, std::move(sys), tol, batch_size,
                                                                high_accuracy);
}

#endif

} // namespace heyoka
