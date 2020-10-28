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
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
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

#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Pass.h>
#include <llvm/Support/Casting.h>
#include <llvm/Transforms/Vectorize.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/math_wrappers.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

// Helper to produce a unique string for the type t.
// This is used in certain AD implementations
// to avoid potential clashing in function names.
std::string taylor_mangle_suffix(llvm::Type *t)
{
    assert(t != nullptr);

    if (auto v_t = llvm::dyn_cast<llvm::VectorType>(t)) {
        // If the type is a vector, get the name of the element type
        // and append the vector size.
        return llvm_type_name(v_t->getElementType()) + "_" + li_to_string(v_t->getNumElements());
    } else {
        // Otherwise just return the type name.
        return llvm_type_name(t);
    }
}

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

    const auto orig_dc_size = v_ex.size();

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

    get_logger()->info("Taylor decomposition CSE reduced the size from {} to {}", orig_dc_size, retval.size());

    return retval;
}

// Perform a topological sort on a graph representation
// of the Taylor decomposition. This can improve performance
// by grouping together operations that can be performed in parallel.
// NOTE: the original decomposition dc is already topologically sorted,
// in the sense that the definitions of the u variables are already
// ordered according to dependency. However, because the original decomposition
// comes from a depth-first search, it has the tendency to group together
// expressions which are dependent on each other. By doing another topological
// sort, this time based on breadth-first search, we determine another valid
// sorting in which independent operations tend to be clustered together. This
// results in a measurable performance improvement in non-compact mode (~15%
// on the outer_ss benchmarks),
// however it does not seem to have an effect in compact mode. Perhaps
// in the future we can consider some options:
//
// - make this extra sorting deactivatable with a kw arg,
// - do the decomposition in breadth-first fashion directly,
//   thus avoiding this extra sorting.
auto taylor_sort_dc(std::vector<expression> &dc, std::vector<expression>::size_type n_eq)
{
    // A Taylor decomposition is supposed
    // to have n_eq variables at the beginning,
    // n_eq variables at the end and possibly
    // extra variables in the middle
    assert(dc.size() >= n_eq * 2u);

    // The graph type that we will use for the topological sorting.
    using graph_t = boost::adjacency_list<boost::vecS,           // std::vector for list of adjacent vertices
                                          boost::vecS,           // std::vector for the list of vertices
                                          boost::bidirectionalS, // directed graph with efficient access
                                                                 // to in-edges
                                          boost::no_property,    // no vertex properties
                                          boost::no_property,    // no edge properties
                                          boost::no_property,    // no graph properties
                                          boost::listS           // std::list for of the graph's edge list
                                          >;

    graph_t g;

    // Add the root node.
    const auto root_v = boost::add_vertex(g);

    // Add the nodes corresponding to the state variables.
    for (decltype(n_eq) i = 0; i < n_eq; ++i) {
        auto v = boost::add_vertex(g);

        // Add a dependency on the root node.
        boost::add_edge(root_v, v, g);
    }

    // Add the rest of the u variables.
    for (decltype(n_eq) i = n_eq; i < dc.size() - n_eq; ++i) {
        auto v = boost::add_vertex(g);

        // Fetch the list of variables in the current expression.
        const auto vars = get_variables(dc[i]);

        if (vars.empty()) {
            // The current expression does not contain
            // any variable: make it depend on the root
            // node. This means that in the topological
            // sort below, the current u var will appear
            // immediately after the state variables.
            boost::add_edge(root_v, v, g);
        } else {
            // Mark the current u variable as depending on all the
            // variables in the current expression.
            for (const auto &var : vars) {
                // Extract the index.
                const auto idx = uname_to_index(var);

                // Add the dependency.
                // NOTE: add +1 because the i-th vertex
                // corresponds to the (i-1)-th u variable
                // due to the presence of the root node.
                boost::add_edge(boost::vertex(idx + 1u, g), v, g);
            }
        }
    }

    assert(boost::num_vertices(g) - 1u == dc.size() - n_eq);

    // Run the BF topological sort on the graph. This is Kahn's algorithm:
    // https://en.wikipedia.org/wiki/Topological_sorting

    // The result of the sort.
    std::vector<decltype(dc.size())> v_idx;

    // Temp variable used to sort a list of edges in the loop below.
    std::vector<boost::graph_traits<graph_t>::edge_descriptor> tmp_edges;

    // The set of all nodes with no incoming edge.
    std::deque<decltype(dc.size())> tmp;
    // The root node has no incoming edge.
    tmp.push_back(0);

    // Main loop.
    while (!tmp.empty()) {
        // Pop the first element from tmp
        // and append it to the result.
        const auto v = tmp.front();
        tmp.pop_front();
        v_idx.push_back(v);

        // Fetch all the out edges of v and sort them according
        // to the target vertex.
        // NOTE: the sorting is important to ensure that all the state
        // variables are insered into v_idx in the correct order.
        const auto e_range = boost::out_edges(v, g);
        tmp_edges.assign(e_range.first, e_range.second);
        std::sort(tmp_edges.begin(), tmp_edges.end(),
                  [&g](const auto &e1, const auto &e2) { return boost::target(e1, g) < boost::target(e2, g); });

        // For each out edge of v:
        // - eliminate it;
        // - check if the target vertex of the edge
        //   has other incoming edges;
        // - if it does not, insert it into tmp.
        for (auto &e : tmp_edges) {
            // Fetch the target of the edge.
            const auto t = boost::target(e, g);

            // Remove the edge.
            boost::remove_edge(e, g);

            // Get the range of vertices connecting to t.
            const auto iav = boost::inv_adjacent_vertices(t, g);

            if (iav.first == iav.second) {
                // t does not have any incoming edges, add it to tmp.
                tmp.push_back(t);
            }
        }
    }

    assert(v_idx.size() == boost::num_vertices(g));
    assert(boost::num_edges(g) == 0u);

    // Adjust v_idx: remove the index of the root node,
    // decrease by one all other indices, insert the final
    // n_eq indices.
    for (decltype(v_idx.size()) i = 0; i < v_idx.size() - 1u; ++i) {
        v_idx[i] = v_idx[i + 1u] - 1u;
    }
    v_idx.resize(boost::numeric_cast<decltype(v_idx.size())>(dc.size()));
    std::iota(v_idx.data() + dc.size() - n_eq, v_idx.data() + dc.size(), dc.size() - n_eq);

    // Create the remapping dictionary.
    std::unordered_map<std::string, std::string> remap;
    for (decltype(v_idx.size()) i = n_eq; i < v_idx.size() - n_eq; ++i) {
        if (v_idx[i] != i) {
            remap.emplace("u_" + li_to_string(v_idx[i]), "u_" + li_to_string(i));
        }
    }

    // Do the remap.
    for (auto it = dc.data() + n_eq; it != dc.data() + dc.size(); ++it) {
        rename_variables(*it, remap);
    }

    // Reorder the decomposition.
    std::vector<expression> retval;
    for (auto idx : v_idx) {
        retval.push_back(std::move(dc[idx]));
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

    u_vars_defs = detail::taylor_sort_dc(u_vars_defs, n_eq);

#if !defined(NDEBUG)
    // Verify the reordered decomposition.
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

    u_vars_defs = detail::taylor_sort_dc(u_vars_defs, n_eq);

#if !defined(NDEBUG)
    // Verify the reordered decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
#endif

    return u_vars_defs;
}

namespace detail
{

template <typename T>
template <typename U>
void taylor_adaptive_impl<T>::finalise_ctor_impl(U sys, std::vector<T> state, T time, T tol, bool high_accuracy,
                                                 bool compact_mode)
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

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Add the stepper function.
    std::tie(m_dc, m_order)
        = taylor_add_adaptive_step<T>(m_llvm, "step", std::move(sys), tol, 1, high_accuracy, compact_mode);

    // Run the jit.
    m_llvm.compile();

    // Fetch the stepper.
    m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(const taylor_adaptive_impl &other)
    // NOTE: make a manual copy of all members, apart from the function pointer.
    : m_state(other.m_state), m_time(other.m_time), m_llvm(other.m_llvm), m_dim(other.m_dim), m_dc(other.m_dc),
      m_order(other.m_order)
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
// and the integration timestep that was used.
template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step_impl(T max_delta_t)
{
#if !defined(NDEBUG)
    // NOTE: this is the only precondition on max_delta_t.
    using std::isnan;
    assert(!isnan(max_delta_t));
#endif

    // Invoke the stepper.
    auto h = max_delta_t;
    m_step_f(m_state.data(), &h);

    // Update the time.
    m_time += h;

    // Check if the time or the state vector are non-finite at the
    // end of the timestep.
    if (!detail::isfinite(m_time)
        || std::any_of(m_state.cbegin(), m_state.cend(), [](const auto &x) { return !detail::isfinite(x); })) {
        return std::tuple{taylor_outcome::err_nf_state, h};
    }

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
    // Check the current time.
    if (!detail::isfinite(m_time)) {
        throw std::invalid_argument("Cannot invoke the propagate_until() function of an adaptive Taylor integrator if "
                                    "the current time is not finite");
    }

    // Check the final time.
    if (!detail::isfinite(t)) {
        throw std::invalid_argument(
            "A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

    // Initial values for the counters,
    // the min/max abs of the integration
    // timesteps, and min/max Taylor orders.
    // NOTE: iter_counter is for keeping track of the max_steps
    // limits, step_counter counts the number of timesteps performed
    // with a non-zero h. Most of the time these two quantities
    // will be identical, apart from corner cases.
    std::size_t iter_counter = 0, step_counter = 0;
    T min_h = std::numeric_limits<T>::infinity(), max_h(0);

    while (true) {
        // NOTE: t - m_time is guaranteed not to be nan: t is never non-finite,
        // and at the first iteration we have checked above the value of m_time.
        // At successive iterations, we know that m_time must be finite because
        // otherwise we would have exited the loop when checking res.
        const auto [res, h] = step_impl(t - m_time);

        if (res != taylor_outcome::success && res != taylor_outcome::time_limit) {
            // Something went wrong in the propagation of the timestep, exit.
            return std::tuple{res, min_h, max_h, step_counter};
        }

        // Update the number of iterations.
        ++iter_counter;

        // Update the number of steps.
        step_counter += static_cast<std::size_t>(h != 0);

        // Break out if the time limit is reached,
        // *before* updating the min_h/max_h values.
        if (res == taylor_outcome::time_limit) {
            return std::tuple{taylor_outcome::time_limit, min_h, max_h, step_counter};
        }

        // Update min_h/max_h.
        using std::abs;
        const auto abs_h = abs(h);
        min_h = std::min(min_h, abs_h);
        max_h = std::max(max_h, abs_h);

        // Check the iteration limit.
        if (max_steps != 0u && iter_counter == max_steps) {
            return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter};
        }
    }
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

template <typename T>
std::uint32_t taylor_adaptive_impl<T>::get_order() const
{
    return m_order;
}

template <typename T>
std::uint32_t taylor_adaptive_impl<T>::get_dim() const
{
    return m_dim;
}

// Explicit instantiation of the implementation classes/functions.
template class taylor_adaptive_impl<double>;
template void taylor_adaptive_impl<double>::finalise_ctor_impl(std::vector<expression>, std::vector<double>, double,
                                                               double, bool, bool);
template void taylor_adaptive_impl<double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                               std::vector<double>, double, double, bool, bool);

template class taylor_adaptive_impl<long double>;
template void taylor_adaptive_impl<long double>::finalise_ctor_impl(std::vector<expression>, std::vector<long double>,
                                                                    long double, long double, bool, bool);
template void taylor_adaptive_impl<long double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                                    std::vector<long double>, long double, long double,
                                                                    bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_impl<mppp::real128>;
template void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(std::vector<expression>,
                                                                      std::vector<mppp::real128>, mppp::real128,
                                                                      mppp::real128, bool, bool);
template void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                                      std::vector<mppp::real128>, mppp::real128,
                                                                      mppp::real128, bool, bool);

#endif

} // namespace detail

namespace detail
{

template <typename T>
template <typename U>
void taylor_adaptive_batch_impl<T>::finalise_ctor_impl(U sys, std::vector<T> states, std::uint32_t batch_size,
                                                       std::vector<T> times, T tol, bool high_accuracy,
                                                       bool compact_mode)
{
    // Init the data members.
    m_batch_size = batch_size;
    m_states = std::move(states);
    m_times = std::move(times);

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

    if (!detail::isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is " + li_to_string(tol)
            + " instead");
    }

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Add the stepper function.
    std::tie(m_dc, m_order)
        = taylor_add_adaptive_step<T>(m_llvm, "step", std::move(sys), tol, m_batch_size, high_accuracy, compact_mode);

    // Run the jit.
    m_llvm.compile();

    // Fetch the stepper.
    m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));

    // Prepare the temp vectors.
    m_pinf.resize(m_batch_size, std::numeric_limits<T>::infinity());
    m_minf.resize(m_batch_size, -std::numeric_limits<T>::infinity());
    m_delta_ts.resize(m_batch_size);
    m_ts_count.resize(boost::numeric_cast<decltype(m_ts_count.size())>(m_batch_size));
    m_min_abs_h.resize(m_batch_size);
    m_max_abs_h.resize(m_batch_size);
    m_tmp_res.resize(boost::numeric_cast<decltype(m_tmp_res.size())>(m_batch_size));
    m_cur_max_delta_ts.resize(m_batch_size);
    m_pfor_ts.resize(m_batch_size);
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &other)
    // NOTE: make a manual copy of all members, apart from the function pointers.
    : m_batch_size(other.m_batch_size), m_states(other.m_states), m_times(other.m_times), m_llvm(other.m_llvm),
      m_dim(other.m_dim), m_dc(other.m_dc), m_order(other.m_order), m_pinf(other.m_pinf), m_minf(other.m_minf),
      m_delta_ts(other.m_delta_ts), m_ts_count(other.m_ts_count), m_min_abs_h(other.m_min_abs_h),
      m_max_abs_h(other.m_max_abs_h), m_tmp_res(other.m_tmp_res), m_cur_max_delta_ts(other.m_cur_max_delta_ts),
      m_pfor_ts(other.m_pfor_ts)
{
    m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
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
// The function will write to res a pair for each state
// vector, containing a flag describing the outcome of the integration
// and the integration timestep that was used.
template <typename T>
void taylor_adaptive_batch_impl<T>::step_impl(std::vector<std::tuple<taylor_outcome, T>> &res,
                                              const std::vector<T> &max_delta_ts)
{
    // Check preconditions.
    assert(max_delta_ts.size() == m_batch_size);
    assert(std::none_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
        using std::isnan;
        return isnan(x);
    }));

    // Prepare res.
    res.resize(m_batch_size);

    // Copy max_delta_ts to the tmp buffer.
    std::copy(max_delta_ts.begin(), max_delta_ts.end(), m_delta_ts.begin());

    // Invoke the stepper.
    m_step_f(m_states.data(), m_delta_ts.data());

    // Helper to check if the state vector of a batch element
    // contains a non-finite value.
    auto check_nf_batch = [this](std::uint32_t batch_idx) {
        for (std::uint32_t i = 0; i < m_dim; ++i) {
            if (!detail::isfinite(m_states[i * m_batch_size + batch_idx])) {
                return true;
            }
        }
        return false;
    };

    // Update the times and write out the result.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        // The timestep that was actually used for
        // this batch element.
        const auto h = m_delta_ts[i];
        m_times[i] += h;

        if (!detail::isfinite(m_times[i]) || check_nf_batch(i)) {
            // Either the new time or state contain non-finite values,
            // return an error condition.
            res[i] = std::tuple{taylor_outcome::err_nf_state, h};
        } else {
            res[i] = std::tuple{h == max_delta_ts[i] ? taylor_outcome::time_limit : taylor_outcome::success, h};
        }
    }
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step(std::vector<std::tuple<taylor_outcome, T>> &res)
{
    return step_impl(res, m_pinf);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step_backward(std::vector<std::tuple<taylor_outcome, T>> &res)
{
    return step_impl(res, m_minf);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::propagate_for(std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &res,
                                                  const std::vector<T> &delta_ts, std::size_t max_steps)
{
    // Check the dimensionality of delta_ts.
    if (delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of time intervals specified in a Taylor integrator in batch mode: the batch size is "
            + std::to_string(m_batch_size) + ", but the number of specified time intervals is "
            + std::to_string(delta_ts.size()));
    }

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_pfor_ts[i] = m_times[i] + delta_ts[i];
    }

    propagate_until(res, m_pfor_ts, max_steps);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::propagate_until(std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &res,
                                                    const std::vector<T> &ts, std::size_t max_steps)
{
    // Check the dimensionality of ts.
    if (ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of time limits specified in a Taylor integrator in batch mode: the batch size is "
            + std::to_string(m_batch_size) + ", but the number of specified time limits is "
            + std::to_string(ts.size()));
    }

    // Check the current times.
    if (std::any_of(m_times.begin(), m_times.end(), [](const auto &t) { return !detail::isfinite(t); })) {
        throw std::invalid_argument(
            "Cannot invoke the propagate_until() function of an adaptive Taylor integrator in batch mode if "
            "one of the current times is not finite");
    }

    // Check the final times.
    if (std::any_of(ts.begin(), ts.end(), [](const auto &t) { return !detail::isfinite(t); })) {
        throw std::invalid_argument("A non-finite time was passed to the propagate_until() function of an adaptive "
                                    "Taylor integrator in batch mode");
    }

    // Prepare res.
    res.resize(m_batch_size);

    // Reset the counters and the min/max abs(h) vectors.
    std::size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: ts[i] - m_times[i] is guaranteed not to be nan: ts[i] is never non-finite,
        // and at the first iteration we have checked above the value of m_times.
        // At successive iterations, we know that m_times[i] must be finite because
        // otherwise we would have exited the loop when checking res.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_cur_max_delta_ts[i] = ts[i] - m_times[i];
        }

        // Run the integration timestep.
        step_impl(m_tmp_res, m_cur_max_delta_ts);

        // Check if the integration timestep produced an error condition.
        if (std::any_of(m_tmp_res.begin(), m_tmp_res.end(), [](const auto &tup) {
                return std::get<0>(tup) != taylor_outcome::success && std::get<0>(tup) != taylor_outcome::time_limit;
            })) {
            break;
        }

        // Update the iteration counter.
        ++iter_counter;

        // Update the local step counters.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // NOTE: the local step counters increase only if we integrated
            // for a non-zero time.
            m_ts_count[i] += static_cast<std::size_t>(std::get<1>(m_tmp_res[i]) != 0);
        }

        // Break out if we have reached the time limit for all
        // batch elements.
        if (std::all_of(m_tmp_res.begin(), m_tmp_res.end(),
                        [](const auto &tup) { return std::get<0>(tup) == taylor_outcome::time_limit; })) {
            break;
        }

        // Update min_h/max_h.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // Don't update if we reached the time limit.
            if (std::get<0>(m_tmp_res[i]) == taylor_outcome::time_limit) {
                continue;
            }

            using std::abs;
            const auto abs_h = abs(std::get<1>(m_tmp_res[i]));
            m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
            m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
        }

        // Check the iteration limit.
        if (max_steps != 0u && iter_counter == max_steps) {
            break;
        }
    }

    // Assemble the return value.
    if (max_steps != 0u && iter_counter == max_steps) {
        // We reached the max_steps limit: if the last integration step was successful
        // return step_limit, otherwise time_limit.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            res[i] = std::tuple{std::get<0>(m_tmp_res[i]) == taylor_outcome::success ? taylor_outcome::step_limit
                                                                                     : taylor_outcome::time_limit,
                                m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }
    } else {
        // We exited either because of an error, or because all the batch
        // elements reached the time limits. In this case, we just use the
        // outcome of the last timestep.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            res[i] = std::tuple{std::get<0>(m_tmp_res[i]), m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }
    }
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

template <typename T>
std::uint32_t taylor_adaptive_batch_impl<T>::get_order() const
{
    return m_order;
}

template <typename T>
std::uint32_t taylor_adaptive_batch_impl<T>::get_dim() const
{
    return m_dim;
}

// Explicit instantiation of the batch implementation classes.
template class taylor_adaptive_batch_impl<double>;
template void taylor_adaptive_batch_impl<double>::finalise_ctor_impl(std::vector<expression>, std::vector<double>,
                                                                     std::uint32_t, std::vector<double>, double, bool,
                                                                     bool);
template void taylor_adaptive_batch_impl<double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                                     std::vector<double>, std::uint32_t,
                                                                     std::vector<double>, double, bool, bool);

template class taylor_adaptive_batch_impl<long double>;
template void taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(std::vector<expression>,
                                                                          std::vector<long double>, std::uint32_t,
                                                                          std::vector<long double>, long double, bool,
                                                                          bool);
template void
taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                            std::vector<long double>, std::uint32_t,
                                                            std::vector<long double>, long double, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_batch_impl<mppp::real128>;
template void taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(std::vector<expression>,
                                                                            std::vector<mppp::real128>, std::uint32_t,
                                                                            std::vector<mppp::real128>, mppp::real128,
                                                                            bool, bool);
template void
taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                              std::vector<mppp::real128>, std::uint32_t,
                                                              std::vector<mppp::real128>, mppp::real128, bool, bool);

#endif

} // namespace detail

namespace detail
{

// Helper to fetch the derivative of order 'order' of the u variable at index u_idx from the
// derivative array 'arr'. The total number of u variables is n_uvars.
llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &arr, std::uint32_t u_idx, std::uint32_t order,
                               std::uint32_t n_uvars)
{
    // Sanity check.
    assert(u_idx < n_uvars);

    // Compute the index.
    const auto idx = static_cast<decltype(arr.size())>(order) * n_uvars + u_idx;
    assert(idx < arr.size());

    return arr[idx];
}

// Load the derivative of order 'order' of the u variable u_idx from the array of Taylor derivatives diff_arr.
// n_uvars is the total number of u variables.
llvm::Value *taylor_c_load_diff(llvm_state &s, llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                                llvm::Value *u_idx)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    auto ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx)});

    return builder.CreateLoad(ptr);
}

namespace
{

// Store the value val as the derivative of order 'order' of the u variable u_idx
// into the array of Taylor derivatives diff_arr. n_uvars is the total number of u variables.
void taylor_c_store_diff(llvm_state &s, llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                         std::uint32_t u_idx, llvm::Value *val)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    auto ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), builder.getInt32(u_idx))});

    builder.CreateStore(val, ptr);
}

// Compute the derivative of order "order" of a state variable.
// ex is the formula for the first-order derivative of the state variable (which
// is either a u variable or a number), n_uvars the number of variables in
// the decomposition, arr the array containing the derivatives of all u variables
// up to order - 1.
template <typename T>
llvm::Value *taylor_compute_sv_diff(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size)
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
                return builder.CreateFDiv(
                    ret, vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size));
            } else if constexpr (std::is_same_v<type, number>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0. No need for normalization as the only
                // nonzero value that can be produced here is the first-order
                // derivative.
                return vector_splat(builder, codegen<T>(s, (order == 1u) ? v : number{0.}), batch_size);
            } else {
                assert(false);
                return nullptr;
            }
        },
        ex.value());
}

// Compute the derivative of order "order" of a state variable.
// ex is the formula for the first-order derivative of the state variable (which
// is either a u variable or a number), n_uvars the number of variables in
// the decomposition, arr the array containing the derivatives of all u variables
// up to order - 1.
template <typename T>
llvm::Value *taylor_c_compute_sv_diff(llvm_state &s, const expression &ex, llvm::Value *diff_arr, std::uint32_t n_uvars,
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
                auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, builder.CreateSub(order, builder.getInt32(1)),
                                              builder.getInt32(u_idx));

                // We have to divide the derivative by 'order'
                // to get the normalised derivative of the state variable.
                return builder.CreateFDiv(
                    ret, vector_splat(builder, builder.CreateUIToFP(order, to_llvm_type<T>(s.context())), batch_size));
            } else if constexpr (std::is_same_v<type, number>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0. No need for normalization as the only
                // nonzero value that can be produced here is the first-order
                // derivative.
                auto cmp_cond = builder.CreateICmpEQ(order, builder.getInt32(1));
                return builder.CreateSelect(cmp_cond, vector_splat(builder, codegen<T>(s, v), batch_size),
                                            vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
            } else {
                assert(false);
                return nullptr;
            }
        },
        ex.value());
}

// Helper function to compute the jet of Taylor derivatives up to a given order. n_eq
// is the number of equations/variables in the ODE sys, dc its Taylor decomposition,
// n_uvars the total number of u variables in the decomposition.
// order is the max derivative order desired, batch_size the batch size.
// order0 contains the zero order derivatives of the state variables.
//
// The return value is the jet of derivatives of the state variables up to order 'order'.
//
// NOTE: at one point we had another version of this function which would return a variant
// containing either the diff array for all uvars (compact mode) or a std::vector containing the jet
// of derivatives for the state variables like now. This allowed us to interleave load instructions
// with computations later (e.g., in the determination of rho and in the Taylor polynomial
// evaluation functions). This did not seem to make a difference, performance-wise, but it might
// be useful to remember about this. See tree state @ 7476630eb5a1d6ac6204035093faecdd1f6d7da5.
template <typename T>
auto taylor_compute_jet(llvm_state &s, std::vector<llvm::Value *> order0, const std::vector<expression> &dc,
                        std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                        bool compact_mode)
{
    assert(order0.size() == n_eq);
    assert(n_eq > 0u);
    assert(order > 0u);

    // Make sure we can represent a size of n_uvars * order + n_eq as a 32-bit
    // unsigned integer. This is the total number of derivatives we will have to compute
    // and store.
    if (n_uvars > std::numeric_limits<std::uint32_t>::max() / order
        || n_uvars * order > std::numeric_limits<std::uint32_t>::max() - n_eq) {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives");
    }

    std::vector<llvm::Value *> retval;

    if (compact_mode) {
        auto &builder = s.builder();

        // Prepare the array that will contain the jet of derivatives.
        // We will be storing all the derivatives of the u variables
        // up to order 'order - 1', plus the derivatives of order
        // 'order' of the state variables only.
        // NOTE: the array size is specified as a 64-bit integer in the
        // LLVM API.
        auto array_type = llvm::ArrayType::get(order0[0]->getType(), n_uvars * order + n_eq);
        // NOTE: fetch a pointer to the first element of the array.
        auto diff_arr = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type, 0, "diff_arr"),
                                                  {builder.getInt32(0), builder.getInt32(0)});

        // Copy over the order0 derivatives of the state variables.
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            taylor_c_store_diff(s, diff_arr, n_uvars, builder.getInt32(0), i, order0[i]);
        }

        // Run the init for the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            auto val = taylor_c_u_init<T>(s, dc[i], diff_arr, batch_size);
            taylor_c_store_diff(s, diff_arr, n_uvars, builder.getInt32(0), i, val);
        }

        // Compute all derivatives up to order 'order - 1'.
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(order), [&](llvm::Value *cur_order) {
            // Begin with the state variables.
            // NOTE: the derivatives of the state variables
            // are at the end of the decomposition vector.
            for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
                taylor_c_store_diff(s, diff_arr, n_uvars, cur_order, i - n_uvars,
                                    taylor_c_compute_sv_diff<T>(s, dc[i], diff_arr, n_uvars, cur_order, batch_size));
            }

            // Now the other u variables.
            for (auto i = n_eq; i < n_uvars; ++i) {
                taylor_c_store_diff(s, diff_arr, n_uvars, cur_order, i,
                                    taylor_c_diff<T>(s, dc[i], diff_arr, n_uvars, cur_order, i, batch_size));
            }
        });

        // Compute the last-order derivatives for the state variables.
        for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
            taylor_c_store_diff(
                s, diff_arr, n_uvars, builder.getInt32(order), i - n_uvars,
                taylor_c_compute_sv_diff<T>(s, dc[i], diff_arr, n_uvars, builder.getInt32(order), batch_size));
        }

        // Build the return value.
        for (std::uint32_t o = 0; o <= order; ++o) {
            for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
                retval.push_back(
                    taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(o), builder.getInt32(var_idx)));
            }
        }

        return retval;
    } else {
        // Init the derivatives array with the order 0 of the state variables.
        auto diff_arr(std::move(order0));

        // Compute the order-0 derivatives of the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            diff_arr.push_back(taylor_u_init<T>(s, dc[i], diff_arr, batch_size));
        }

        // Compute the derivatives order by order, starting from 1 to order excluded.
        // We will compute the highest derivatives of the state variables separately
        // in the last step.
        for (std::uint32_t cur_order = 1; cur_order < order; ++cur_order) {
            // Begin with the state variables.
            // NOTE: the derivatives of the state variables
            // are at the end of the decomposition vector.
            for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
                diff_arr.push_back(taylor_compute_sv_diff<T>(s, dc[i], diff_arr, n_uvars, cur_order, batch_size));
            }

            // Now the other u variables.
            for (auto i = n_eq; i < n_uvars; ++i) {
                diff_arr.push_back(taylor_diff<T>(s, dc[i], diff_arr, n_uvars, cur_order, i, batch_size));
            }
        }

        // Compute the last-order derivatives for the state variables.
        for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
            diff_arr.push_back(taylor_compute_sv_diff<T>(s, dc[i], diff_arr, n_uvars, order, batch_size));
        }

        assert(diff_arr.size() == static_cast<decltype(diff_arr.size())>(n_uvars) * order + n_eq);

        // Extract the derivatives of the state variables from diff_arr.
        for (std::uint32_t o = 0; o <= order; ++o) {
            for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
                retval.push_back(taylor_fetch_diff(diff_arr, var_idx, o, n_uvars));
            }
        }

        return retval;
    }
}

// Given an input pointer 'in', load the first n * batch_size values in it as n vectors
// with size batch_size.
template <typename T>
auto taylor_load_values(llvm_state &s, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    // Overflow check.
    if (n > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("Overflow while loading Taylor values");
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

// NOTE: in compact mode, care must be taken when adding multiple jet functions to the same llvm state
// with the same floating-point type, batch size and number of u variables. The potential issue there
// is that when the first jet is added, the compact mode AD functions are created and then optimised.
// The optimisation pass might alter the functions in a way that makes them incompatible with subsequent
// uses in the second jet (e.g., an argument might be removed from the signature because it is a
// compile-time constant). A workaround to avoid issues is to set the optimisation level to zero
// in the state, add the 2 jets and then run a single optimisation pass.
// NOTE: document this eventually.
template <typename T, typename U>
auto taylor_add_jet_impl(llvm_state &s, const std::string &name, U sys, std::uint32_t order, std::uint32_t batch_size,
                         bool, bool compact_mode)
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

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    auto dc = taylor_decompose(std::move(sys));

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    // Prepare the function prototype. The only argument is a float pointer to in/out array.
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(to_llvm_type<T>(s.context()))};
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(s.builder().getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
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
    s.builder().SetInsertPoint(bb);

    // Load the order zero derivatives from the input pointer.
    auto order0_arr = taylor_load_values<T>(s, in_out, n_eq, batch_size);

    // Compute the jet of derivatives.
    auto diff_arr = taylor_compute_jet<T>(s, std::move(order0_arr), dc, n_eq, n_uvars, order, batch_size, compact_mode);

    // Write the derivatives to in_out.
    // NOTE: overflow checking. We need to be able to index into the jet array (size n_eq * (order + 1) * batch_size)
    // using uint32_t.
    if (order == std::numeric_limits<std::uint32_t>::max()
        || (order + 1u) > std::numeric_limits<std::uint32_t>::max() / batch_size
        || n_eq > std::numeric_limits<std::uint32_t>::max() / ((order + 1u) * batch_size)) {
        throw std::overflow_error("An overflow condition was detected while adding a Taylor jet");
    }
    for (decltype(diff_arr.size()) cur_order = 1; cur_order <= order; ++cur_order) {
        for (std::uint32_t j = 0; j < n_eq; ++j) {
            // Index in the jet of derivatives.
            const auto arr_idx = cur_order * n_eq + j;
            assert(arr_idx < diff_arr.size());
            const auto val = diff_arr[arr_idx];

            // Index in the output array.
            const auto out_idx = n_eq * batch_size * cur_order + j * batch_size;
            auto out_ptr
                = s.builder().CreateInBoundsGEP(in_out, {s.builder().getInt32(static_cast<std::uint32_t>(out_idx))});
            store_vector_to_memory(s.builder(), out_ptr, val);
        }
    }

    // Finish off the function.
    s.builder().CreateRetVoid();

    // Verify it.
    s.verify_function(f);

    // Run the optimisation pass.
    if (batch_size > 1u) {
        // In vector mode, add a pass to vectorize load/stores.
        // This is useful to ensure that the
        // pattern adopted in load_vector_from_memory() and
        // store_vector_to_memory() is translated to
        // vectorized store/load instructions.
        std::vector<std::unique_ptr<llvm::Pass>> passes;
        passes.push_back(std::unique_ptr<llvm::Pass>(llvm::createLoadStoreVectorizerPass()));

        s.optimise(std::move(passes));
    } else {
        s.optimise();
    }

    return dc;
}

} // namespace

} // namespace detail

std::vector<expression> taylor_add_jet_dbl(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                           std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                           bool compact_mode)
{
    return detail::taylor_add_jet_impl<double>(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
}

std::vector<expression> taylor_add_jet_ldbl(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                            std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                            bool compact_mode)
{
    return detail::taylor_add_jet_impl<long double>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                    compact_mode);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_jet_f128(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                            std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                            bool compact_mode)
{
    return detail::taylor_add_jet_impl<mppp::real128>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                      compact_mode);
}

#endif

std::vector<expression> taylor_add_jet_dbl(llvm_state &s, const std::string &name,
                                           std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                           std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_jet_impl<double>(s, name, std::move(sys), order, batch_size, high_accuracy, compact_mode);
}

std::vector<expression> taylor_add_jet_ldbl(llvm_state &s, const std::string &name,
                                            std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                            std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_jet_impl<long double>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                    compact_mode);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_jet_f128(llvm_state &s, const std::string &name,
                                            std::vector<std::pair<expression, expression>> sys, std::uint32_t order,
                                            std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_jet_impl<mppp::real128>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                      compact_mode);
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
    auto x_t = x_v->getType()->getScalarType();

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
                s, "heyoka_maxabs128", x_t, {x_scalars[i], y_scalars[i]},
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
    auto x_t = x_v->getType()->getScalarType();

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
                s, "heyoka_minabs128", x_t, {x_scalars[i], y_scalars[i]},
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
    auto x_t = x_v->getType()->getScalarType();

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
                s, "heyoka_minnum128", x_t, {x_scalars[i], y_scalars[i]},
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
    auto x_t = x_v->getType()->getScalarType();

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
        // If we are operating on SIMD vectors, try to see if we have a sleef
        // function available for pow().
        if (auto vec_t = llvm::dyn_cast<llvm::VectorType>(x_v->getType())) {
            // NOTE: if sfn ends up empty, we will be falling through
            // below and use the LLVM intrinsic instead.
            if (const auto sfn = sleef_function_name(s.context(), "pow", vec_t->getElementType(),
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x_v, y_v},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s, "llvm.pow", {x_v->getType()}, {x_v, y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Helper to compute abs(x_v) in the Taylor stepper implementation.
llvm::Value *taylor_step_abs(llvm_state &s, llvm::Value *x_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector argument.
    auto x_t = x_v->getType()->getScalarType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        // NOTE: for __float128 we cannot use the intrinsic, we need
        // to call an external function.
        auto &builder = s.builder();

        // Convert the vector arguments to scalars.
        auto x_scalars = vector_to_scalars(builder, x_v);

        // Execute the heyoka_abs128() function on the scalar values and store
        // the results in res_scalars.
        std::vector<llvm::Value *> res_scalars;
        for (auto x_scal : x_scalars) {
            res_scalars.push_back(llvm_invoke_external(
                s, "heyoka_abs128", x_t, {x_scal},
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
        return llvm_invoke_intrinsic(s, "llvm.fabs", {x_v->getType()}, {x_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

// Run the Horner scheme on multiple polynomials at a time, with evaluation point h. Each element of cf_vecs
// contains the list of coefficients of a polynomial.
std::vector<llvm::Value *> taylor_run_multihorner(llvm_state &s, const std::vector<std::vector<llvm::Value *>> &cf_vecs,
                                                  llvm::Value *h)
{
    // Preconditions: cf_vecs is not empty, and it contains polynomials
    // of degree at least 1, all with the same degree.
    assert(!cf_vecs.empty());
    assert(!cf_vecs[0].empty());
    assert(std::all_of(cf_vecs.begin() + 1, cf_vecs.end(),
                       [&cf_vecs](const auto &v) { return v.size() == cf_vecs[0].size(); }));

    auto &builder = s.builder();

    // Number of terms in each polynomial (i.e., degree + 1).
    const auto nterms = cf_vecs[0].size();

    // Init the return value, filling it with the values of the
    // coefficients of the highest-degree monomial in each polynomial.
    std::vector<llvm::Value *> retval;
    for (const auto &v : cf_vecs) {
        retval.push_back(v.back());
    }

    // Run the Horner scheme simultaneously for all polynomials.
    for (decltype(cf_vecs[0].size()) i = 1; i < nterms; ++i) {
        for (decltype(cf_vecs.size()) j = 0; j < cf_vecs.size(); ++j) {
            retval[j] = builder.CreateFAdd(cf_vecs[j][nterms - i - 1u], builder.CreateFMul(retval[j], h));
        }
    }

    return retval;
}

// As above, but instead of the Horner scheme use a compensated summation over the naive evaluation
// of monomials.
template <typename T>
std::vector<llvm::Value *> taylor_run_ceval(llvm_state &s, const std::vector<std::vector<llvm::Value *>> &cf_vecs,
                                            llvm::Value *h, std::uint32_t batch_size)
{
    // Preconditions: cf_vecs is not empty, and it contains polynomials
    // of degree at least 1, all with the same degree.
    assert(!cf_vecs.empty());
    assert(!cf_vecs[0].empty());
    assert(std::all_of(cf_vecs.begin() + 1, cf_vecs.end(),
                       [&cf_vecs](const auto &v) { return v.size() == cf_vecs[0].size(); }));

    auto &builder = s.builder();

    // Number of terms in each polynomial (i.e., degree + 1).
    const auto nterms = cf_vecs[0].size();

    // Init the return values with the order-0 monomials, and the running
    // compensations with zero.
    std::vector<llvm::Value *> retval, comp;
    for (const auto &v : cf_vecs) {
        retval.push_back(v[0]);
        comp.push_back(vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
    }

    // Evaluate and sum.
    auto cur_h = h;
    for (decltype(cf_vecs[0].size()) i = 1; i < nterms; ++i) {
        for (decltype(cf_vecs.size()) j = 0; j < cf_vecs.size(); ++j) {
            // Evaluate the current monomial.
            auto tmp = builder.CreateFMul(cf_vecs[j][i], cur_h);

            // Compute the quantities for the compensation.
            auto y = builder.CreateFSub(tmp, comp[j]);
            auto t = builder.CreateFAdd(retval[j], y);

            // Update the compensation and the return value.
            comp[j] = builder.CreateFSub(builder.CreateFSub(t, retval[j]), y);
            retval[j] = t;
        }

        // Update the power of h, if we are not at the last iteration.
        if (i != nterms - 1u) {
            cur_h = builder.CreateFMul(cur_h, h);
        }
    }

    return retval;
}

// NOTE: in compact mode, care must be taken when adding multiple stepper functions to the same llvm state
// with the same floating-point type, batch size and number of u variables. The potential issue there
// is that when the first stepper is added, the compact mode AD functions are created and then optimised.
// The optimisation pass might alter the functions in a way that makes them incompatible with subsequent
// uses in the second stepper (e.g., an argument might be removed from the signature because it is a
// compile-time constant). A workaround to avoid issues is to set the optimisation level to zero
// in the state, add the 2 steppers and then run a single optimisation pass.
// NOTE: document this eventually.
// NOTE: this is not an issue in the Taylor integrators, where we are certain that only 1 stepper
// is ever added to the LLVM state.
template <typename T, typename U>
auto taylor_add_adaptive_step_impl(llvm_state &s, const std::string &name, U sys, T tol, std::uint32_t batch_size,
                                   bool high_accuracy, bool compact_mode)
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
    auto order_f = ceil(-log(tol) / 2 + 1);
    if (!detail::isfinite(order_f)) {
        throw std::invalid_argument(
            "The computation of the Taylor order in an adaptive Taylor stepper produced a non-finite value");
    }
    // NOTE: min order is 2.
    order_f = std::max(T(2), order_f);

    // NOTE: static cast is safe because we know that T is at least
    // a double-precision IEEE type.
    if (order_f > static_cast<T>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("The computation of the Taylor order in an adaptive Taylor stepper resulted "
                                  "in an overflow condition");
    }
    const auto order = static_cast<std::uint32_t>(order_f);

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
    auto order0_arr = taylor_load_values<T>(s, state_ptr, n_eq, batch_size);

    // Compute the norm infinity of the state vector.
    auto max_abs_state = taylor_step_abs(s, order0_arr[0]);
    for (std::uint32_t i = 1; i < n_eq; ++i) {
        max_abs_state = taylor_step_maxabs(s, max_abs_state, order0_arr[i]);
    }

    // Compute the jet of derivatives at the given order.
    // NOTE: in taylor_compute_jet() we ensure that n_uvars * order + n_eq
    // is representable as a 32-bit unsigned integer. Thus, we can always
    // index into diff_arr using std::uint32_t.
    auto diff_arr = taylor_compute_jet<T>(s, std::move(order0_arr), dc, n_eq, n_uvars, order, batch_size, compact_mode);

    // Determine the norm infinity of the derivatives
    // at orders order and order - 1.
    auto max_abs_diff_o = taylor_step_abs(s, diff_arr[order * n_eq]);
    auto max_abs_diff_om1 = taylor_step_abs(s, diff_arr[(order - 1u) * n_eq]);
    for (std::uint32_t i = 1; i < n_eq; ++i) {
        max_abs_diff_o = taylor_step_maxabs(s, max_abs_diff_o, diff_arr[order * n_eq + i]);
        max_abs_diff_om1 = taylor_step_maxabs(s, max_abs_diff_om1, diff_arr[(order - 1u) * n_eq + i]);
    }

    // Determine if we are in absolute or relative tolerance mode.
    auto abs_or_rel
        = builder.CreateFCmpOLE(max_abs_state, vector_splat(builder, codegen<T>(s, number{1.}), batch_size));

    // Estimate rho at orders order - 1 and order.
    auto num_rho
        = builder.CreateSelect(abs_or_rel, vector_splat(builder, codegen<T>(s, number{1.}), batch_size), max_abs_state);
    auto rho_o = taylor_step_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_o),
                                 vector_splat(builder, codegen<T>(s, number{T(1) / order}), batch_size));
    auto rho_om1 = taylor_step_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_om1),
                                   vector_splat(builder, codegen<T>(s, number{T(1) / (order - 1u)}), batch_size));

    // Take the minimum.
    auto rho_m = taylor_step_min(s, rho_o, rho_om1);

    // Compute the scaling + safety factor.
    const auto rhofac = exp((T(-7) / T(10)) / (order - 1u)) / (exp(T(1)) * exp(T(1)));

    // Determine the step size in absolute value.
    auto h = builder.CreateFMul(rho_m, vector_splat(builder, codegen<T>(s, number{rhofac}), batch_size));

    // Ensure that the step size does not exceed the limit in absolute value.
    auto max_h_vec = load_vector_from_memory(builder, h_ptr, batch_size);
    h = taylor_step_minabs(s, h, max_h_vec);

    // Handle backwards propagation.
    auto backward = builder.CreateFCmpOLT(max_h_vec, vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
    auto h_fac = builder.CreateSelect(backward, vector_splat(builder, codegen<T>(s, number{-1.}), batch_size),
                                      vector_splat(builder, codegen<T>(s, number{1.}), batch_size));
    h = builder.CreateFMul(h_fac, h);

    // Build the Taylor polynomials that need to be evaluated for the propagation.
    std::vector<std::vector<llvm::Value *>> cf_vecs;
    for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
        cf_vecs.emplace_back();
        auto &cf_vec = cf_vecs.back();

        for (std::uint32_t o = 0; o <= order; ++o) {
            cf_vec.push_back(diff_arr[o * n_eq + var_idx]);
        }
    }

    // Evaluate the Taylor polynomials, producing the updated state of the system.
    auto new_states
        = high_accuracy ? taylor_run_ceval<T>(s, cf_vecs, h, batch_size) : taylor_run_multihorner(s, cf_vecs, h);

    // Store the new state.
    for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
        if (var_idx > std::numeric_limits<std::uint32_t>::max() / batch_size) {
            throw std::overflow_error("Overflow error in an adaptive Taylor stepper: too many variables");
        }
        store_vector_to_memory(builder, builder.CreateInBoundsGEP(state_ptr, builder.getInt32(var_idx * batch_size)),
                               new_states[var_idx]);
    }

    // Store the timesteps that were used.
    store_vector_to_memory(builder, h_ptr, h);

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass.
    if (batch_size > 1u) {
        // In vector mode, add a pass to vectorize load/stores.
        // This is useful to ensure that the
        // pattern adopted in load_vector_from_memory() and
        // store_vector_to_memory() is translated to
        // vectorized store/load instructions.
        std::vector<std::unique_ptr<llvm::Pass>> passes;
        passes.push_back(std::unique_ptr<llvm::Pass>(llvm::createLoadStoreVectorizerPass()));

        s.optimise(std::move(passes));
    } else {
        s.optimise();
    }

    return std::tuple{std::move(dc), order};
}

} // namespace

} // namespace detail

std::tuple<std::vector<expression>, std::uint32_t> taylor_add_adaptive_step_dbl(llvm_state &s, const std::string &name,
                                                                                std::vector<expression> sys, double tol,
                                                                                std::uint32_t batch_size,
                                                                                bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_adaptive_step_impl<double>(s, name, std::move(sys), tol, batch_size, high_accuracy,
                                                         compact_mode);
}

std::tuple<std::vector<expression>, std::uint32_t>
taylor_add_adaptive_step_ldbl(llvm_state &s, const std::string &name, std::vector<expression> sys, long double tol,
                              std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_adaptive_step_impl<long double>(s, name, std::move(sys), tol, batch_size, high_accuracy,
                                                              compact_mode);
}

#if defined(HEYOKA_HAVE_REAL128)

std::tuple<std::vector<expression>, std::uint32_t>
taylor_add_adaptive_step_f128(llvm_state &s, const std::string &name, std::vector<expression> sys, mppp::real128 tol,
                              std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_adaptive_step_impl<mppp::real128>(s, name, std::move(sys), tol, batch_size, high_accuracy,
                                                                compact_mode);
}

#endif

std::tuple<std::vector<expression>, std::uint32_t>
taylor_add_adaptive_step_dbl(llvm_state &s, const std::string &name, std::vector<std::pair<expression, expression>> sys,
                             double tol, std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_adaptive_step_impl<double>(s, name, std::move(sys), tol, batch_size, high_accuracy,
                                                         compact_mode);
}

std::tuple<std::vector<expression>, std::uint32_t>
taylor_add_adaptive_step_ldbl(llvm_state &s, const std::string &name,
                              std::vector<std::pair<expression, expression>> sys, long double tol,
                              std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_adaptive_step_impl<long double>(s, name, std::move(sys), tol, batch_size, high_accuracy,
                                                              compact_mode);
}

#if defined(HEYOKA_HAVE_REAL128)

std::tuple<std::vector<expression>, std::uint32_t>
taylor_add_adaptive_step_f128(llvm_state &s, const std::string &name,
                              std::vector<std::pair<expression, expression>> sys, mppp::real128 tol,
                              std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    return detail::taylor_add_adaptive_step_impl<mppp::real128>(s, name, std::move(sys), tol, batch_size, high_accuracy,
                                                                compact_mode);
}

#endif

namespace detail
{

namespace
{

// Add a function for updating the state of an ODE system, given its jet of derivatives up to order 'order'.
// n_vars is the total number of state variables. The function will take as input a read/write pointer
// to the current jet of derivatives, and a const pointer to the timestep (or timesteps in batch mode).
// It will then update the order 0 of the jet of derivatives (i.e., the state vector) via the evaluation
// of the Taylor polynomials for each state variable.
template <typename T>
auto taylor_add_state_updater_impl(llvm_state &s, const std::string &name, std::uint32_t n_vars, std::uint32_t order,
                                   std::uint32_t batch_size, bool high_accuracy)
{
    // NOTE: these are ensured by the fact that this function is always
    // called after the successful addition of a jet function, which
    // checks for these.
    // NOTE: the Taylor jet function also verifies that we can index
    // into the jet of derivatives without overflow.
    assert(n_vars > 0u);
    assert(order > 0u);
    assert(batch_size > 0u);

    auto &builder = s.builder();

    // Prepare the function prototype. The arguments are:
    // - pointer to the jet of derivatives (read and write),
    // - pointer to the array of timesteps (read only).
    // These pointers cannot overlap.
    std::vector<llvm::Type *> fargs(2, llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument("Unable to create a function for a Taylor state updater with name '" + name + "'");
    }

    // Set the name/attributes of the function argument.
    auto jet_ptr = f->args().begin();
    jet_ptr->setName("jet_ptr");
    jet_ptr->addAttr(llvm::Attribute::NoCapture);
    jet_ptr->addAttr(llvm::Attribute::NoAlias);

    auto h_ptr = jet_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);
    h_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Load the coefficients of the Taylor polynomials from jet_ptr.
    std::vector<std::vector<llvm::Value *>> cf_vecs;
    for (std::uint32_t i = 0; i < n_vars; ++i) {
        cf_vecs.emplace_back();
        auto &cf_vec = cf_vecs.back();

        for (std::uint32_t o = 0; o <= order; ++o) {
            // NOTE: overflow check done already in the jet function.
            auto ptr = builder.CreateInBoundsGEP(jet_ptr, {builder.getInt32(o * n_vars * batch_size + i * batch_size)});
            cf_vec.push_back(load_vector_from_memory(builder, ptr, batch_size));
        }
    }

    // Load the integration timestep(s).
    auto h = load_vector_from_memory(builder, h_ptr, batch_size);

    // Evaluate the Taylor polynomials, producing the updated state of the system.
    // NOTE: the asserts at the top ensure that the preconditions of these functions
    // are satisfied.
    auto new_states
        = high_accuracy ? taylor_run_ceval<T>(s, cf_vecs, h, batch_size) : taylor_run_multihorner(s, cf_vecs, h);

    // Store the new state.
    for (std::uint32_t i = 0; i < n_vars; ++i) {
        // NOTE: overflow check done already in the jet function.
        store_vector_to_memory(builder, builder.CreateInBoundsGEP(jet_ptr, builder.getInt32(i * batch_size)),
                               new_states[i]);
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // NOTE: don't run the optimisation pass here, it will be
    // run from the main function below.
}

// RAII helper to temporarily set the opt level to 0 in an llvm_state.
struct opt_disabler {
    llvm_state &m_s;
    unsigned m_orig_opt_level;

    explicit opt_disabler(llvm_state &s) : m_s(s), m_orig_opt_level(s.opt_level())
    {
        // Disable optimisations.
        m_s.opt_level() = 0;
    }
    ~opt_disabler()
    {
        // Restore the original optimisation level.
        m_s.opt_level() = m_orig_opt_level;
    }
};

template <typename T, typename U>
auto taylor_add_custom_step_impl(llvm_state &s, const std::string &name, U sys, std::uint32_t order,
                                 std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
{
    // Record the number of equations/variables.
    // NOTE: the Taylor decomposition will make sure to throw
    // if n_vars ends up being zero.
    const auto n_vars = boost::numeric_cast<std::uint32_t>(sys.size());

    // Temporarily disable optimisations in s.
    std::optional<opt_disabler> od;
    od.emplace(s);

    // Add the function for the computation of the jet of derivatives.
    auto dc = taylor_add_jet_impl<T>(s, name + "_jet", std::move(sys), order, batch_size, high_accuracy, compact_mode);

    // Add the function for the state update.
    taylor_add_state_updater_impl<T>(s, name + "_updater", n_vars, order, batch_size, high_accuracy);

    // Restore the original optimisation level in s.
    od.reset();

    // Run the optimisation pass.
    if (batch_size > 1u) {
        // In vector mode, add a pass to vectorize load/stores.
        // This is useful to ensure that the
        // pattern adopted in load_vector_from_memory() and
        // store_vector_to_memory() is translated to
        // vectorized store/load instructions.
        std::vector<std::unique_ptr<llvm::Pass>> passes;
        passes.push_back(std::unique_ptr<llvm::Pass>(llvm::createLoadStoreVectorizerPass()));

        s.optimise(std::move(passes));
    } else {
        s.optimise();
    }

    return dc;
}

} // namespace

} // namespace detail

std::vector<expression> taylor_add_custom_step_dbl(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                                   std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                                   bool compact_mode)
{
    return detail::taylor_add_custom_step_impl<double>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                       compact_mode);
}

std::vector<expression> taylor_add_custom_step_ldbl(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                                    std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                                    bool compact_mode)
{
    return detail::taylor_add_custom_step_impl<long double>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                            compact_mode);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_custom_step_f128(llvm_state &s, const std::string &name, std::vector<expression> sys,
                                                    std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                                    bool compact_mode)
{
    return detail::taylor_add_custom_step_impl<mppp::real128>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                              compact_mode);
}

#endif

std::vector<expression> taylor_add_custom_step_dbl(llvm_state &s, const std::string &name,
                                                   std::vector<std::pair<expression, expression>> sys,
                                                   std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                                   bool compact_mode)
{
    return detail::taylor_add_custom_step_impl<double>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                       compact_mode);
}

std::vector<expression> taylor_add_custom_step_ldbl(llvm_state &s, const std::string &name,
                                                    std::vector<std::pair<expression, expression>> sys,
                                                    std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                                    bool compact_mode)
{
    return detail::taylor_add_custom_step_impl<long double>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                            compact_mode);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> taylor_add_custom_step_f128(llvm_state &s, const std::string &name,
                                                    std::vector<std::pair<expression, expression>> sys,
                                                    std::uint32_t order, std::uint32_t batch_size, bool high_accuracy,
                                                    bool compact_mode)
{
    return detail::taylor_add_custom_step_impl<mppp::real128>(s, name, std::move(sys), order, batch_size, high_accuracy,
                                                              compact_mode);
}

#endif

} // namespace heyoka
