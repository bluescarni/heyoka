// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <locale>
#include <numeric>
#include <optional>
#include <ostream>
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
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/math_wrappers.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
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
    // must be binary operators or functions whose arguments
    // are either variables in the u_n form,
    // where n < i, or numbers.
    for (auto i = n_eq; i < dc.size() - n_eq; ++i) {
        std::visit(
            [i](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func> || std::is_same_v<type, binary_operator>) {
                    auto check_arg = [i](const auto &arg) {
                        if (auto p_var = std::get_if<variable>(&arg.value())) {
                            assert(p_var->name().rfind("u_", 0) == 0);
                            assert(uname_to_index(p_var->name()) < i);
                        } else if (std::get_if<number>(&arg.value()) == nullptr) {
                            assert(false);
                        }
                    };

                    for (const auto &arg : v.args()) {
                        check_arg(arg);
                    }
                } else {
                    assert(false);
                }
            },
            dc[i].value());
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
// NOTE: on Windows apparently it is necessary to declare that
// these instantiations are meant to be dll-exported.
template class taylor_adaptive_impl<double>;
template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<double>::finalise_ctor_impl(std::vector<expression>,
                                                                                 std::vector<double>, double, double,
                                                                                 bool, bool);
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_impl<double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>, std::vector<double>,
                                                 double, double, bool, bool);

template class taylor_adaptive_impl<long double>;
template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<long double>::finalise_ctor_impl(std::vector<expression>,
                                                                                      std::vector<long double>,
                                                                                      long double, long double, bool,
                                                                                      bool);
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_impl<long double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                      std::vector<long double>, long double, long double, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_impl<mppp::real128>;
template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(std::vector<expression>,
                                                                                        std::vector<mppp::real128>,
                                                                                        mppp::real128, mppp::real128,
                                                                                        bool, bool);
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                        std::vector<mppp::real128>, mppp::real128, mppp::real128, bool,
                                                        bool);

#endif

} // namespace detail

namespace detail
{

template <typename T>
template <typename U>
void taylor_adaptive_batch_impl<T>::finalise_ctor_impl(U sys, std::vector<T> state, std::uint32_t batch_size,
                                                       std::vector<T> time, T tol, bool high_accuracy,
                                                       bool compact_mode)
{
    // Init the data members.
    m_batch_size = batch_size;
    m_state = std::move(state);
    m_time = std::move(time);

    // Check input params.
    if (m_batch_size == 0u) {
        throw std::invalid_argument("The batch size in an adaptive Taylor integrator cannot be zero");
    }

    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !detail::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() % m_batch_size != 0u) {
        throw std::invalid_argument("Invalid size detected in the initialization of an adaptive Taylor "
                                    "integrator: the state vector has a size of "
                                    + std::to_string(m_state.size()) + ", which is not a multiple of the batch size ("
                                    + std::to_string(m_batch_size) + ")");
    }

    if (m_state.size() / m_batch_size != sys.size()) {
        throw std::invalid_argument("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                                    "integrator: the state vector has a dimension of "
                                    + std::to_string(m_state.size() / m_batch_size)
                                    + ", while the number of equations is " + std::to_string(sys.size()));
    }

    if (m_time.size() != m_batch_size) {
        throw std::invalid_argument("Invalid size detected in the initialization of an adaptive Taylor "
                                    "integrator: the time vector has a size of "
                                    + std::to_string(m_time.size()) + ", which is not equal to the batch size ("
                                    + std::to_string(m_batch_size) + ")");
    }

    if (std::any_of(m_time.begin(), m_time.end(), [](const auto &x) { return !detail::isfinite(x); })) {
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

    // NOTE: default init of these vectors is fine.
    m_step_res.resize(boost::numeric_cast<decltype(m_step_res.size())>(m_batch_size));
    m_prop_res.resize(boost::numeric_cast<decltype(m_prop_res.size())>(m_batch_size));

    m_ts_count.resize(boost::numeric_cast<decltype(m_ts_count.size())>(m_batch_size));
    m_min_abs_h.resize(m_batch_size);
    m_max_abs_h.resize(m_batch_size);
    m_cur_max_delta_ts.resize(m_batch_size);
    m_pfor_ts.resize(m_batch_size);
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &other)
    // NOTE: make a manual copy of all members, apart from the function pointers.
    : m_batch_size(other.m_batch_size), m_state(other.m_state), m_time(other.m_time), m_llvm(other.m_llvm),
      m_dim(other.m_dim), m_dc(other.m_dc), m_order(other.m_order), m_pinf(other.m_pinf), m_minf(other.m_minf),
      m_delta_ts(other.m_delta_ts), m_step_res(other.m_step_res), m_prop_res(other.m_prop_res),
      m_ts_count(other.m_ts_count), m_min_abs_h(other.m_min_abs_h), m_max_abs_h(other.m_max_abs_h),
      m_cur_max_delta_ts(other.m_cur_max_delta_ts), m_pfor_ts(other.m_pfor_ts)
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
const std::vector<std::tuple<taylor_outcome, T>> &
taylor_adaptive_batch_impl<T>::step_impl(const std::vector<T> &max_delta_ts)
{
    // Check preconditions.
    assert(max_delta_ts.size() == m_batch_size);
    assert(std::none_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
        using std::isnan;
        return isnan(x);
    }));

    // Sanity check.
    assert(m_step_res.size() == m_batch_size);

    // Copy max_delta_ts to the tmp buffer.
    std::copy(max_delta_ts.begin(), max_delta_ts.end(), m_delta_ts.begin());

    // Invoke the stepper.
    m_step_f(m_state.data(), m_delta_ts.data());

    // Helper to check if the state vector of a batch element
    // contains a non-finite value.
    auto check_nf_batch = [this](std::uint32_t batch_idx) {
        for (std::uint32_t i = 0; i < m_dim; ++i) {
            if (!detail::isfinite(m_state[i * m_batch_size + batch_idx])) {
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
        m_time[i] += h;

        if (!detail::isfinite(m_time[i]) || check_nf_batch(i)) {
            // Either the new time or state contain non-finite values,
            // return an error condition.
            m_step_res[i] = std::tuple{taylor_outcome::err_nf_state, h};
        } else {
            m_step_res[i] = std::tuple{h == max_delta_ts[i] ? taylor_outcome::time_limit : taylor_outcome::success, h};
        }
    }

    return m_step_res;
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T>> &taylor_adaptive_batch_impl<T>::step()
{
    return step_impl(m_pinf);
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T>> &taylor_adaptive_batch_impl<T>::step_backward()
{
    return step_impl(m_minf);
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T>> &
taylor_adaptive_batch_impl<T>::step(const std::vector<T> &max_delta_ts)
{
    // Check the dimensionality of max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is "
            + std::to_string(m_batch_size) + ", but the number of specified timesteps is "
            + std::to_string(max_delta_ts.size()));
    }

    // Make sure no values in max_delta_ts are nan.
    if (std::any_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
            using std::isnan;
            return isnan(x);
        })) {
        throw std::invalid_argument(
            "Cannot invoke the step() function of an adaptive Taylor integrator in batch mode if "
            "one of the max timesteps is nan");
    }

    return step_impl(max_delta_ts);
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &
taylor_adaptive_batch_impl<T>::propagate_for(const std::vector<T> &delta_ts, std::size_t max_steps)
{
    // Check the dimensionality of delta_ts.
    if (delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of time intervals specified in a Taylor integrator in batch mode: the batch size is "
            + std::to_string(m_batch_size) + ", but the number of specified time intervals is "
            + std::to_string(delta_ts.size()));
    }

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_pfor_ts[i] = m_time[i] + delta_ts[i];
    }

    return propagate_until(m_pfor_ts, max_steps);
}

template <typename T>
const std::vector<std::tuple<taylor_outcome, T, T, std::size_t>> &
taylor_adaptive_batch_impl<T>::propagate_until(const std::vector<T> &ts, std::size_t max_steps)
{
    // Check the dimensionality of ts.
    if (ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of time limits specified in a Taylor integrator in batch mode: the batch size is "
            + std::to_string(m_batch_size) + ", but the number of specified time limits is "
            + std::to_string(ts.size()));
    }

    // Check the current times.
    if (std::any_of(m_time.begin(), m_time.end(), [](const auto &t) { return !detail::isfinite(t); })) {
        throw std::invalid_argument(
            "Cannot invoke the propagate_until() function of an adaptive Taylor integrator in batch mode if "
            "one of the current times is not finite");
    }

    // Check the final times.
    if (std::any_of(ts.begin(), ts.end(), [](const auto &t) { return !detail::isfinite(t); })) {
        throw std::invalid_argument("A non-finite time was passed to the propagate_until() function of an adaptive "
                                    "Taylor integrator in batch mode");
    }

    // Reset the counters and the min/max abs(h) vectors.
    std::size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: ts[i] - m_time[i] is guaranteed not to be nan: ts[i] is never non-finite,
        // and at the first iteration we have checked above the value of m_time.
        // At successive iterations, we know that m_time[i] must be finite because
        // otherwise we would have exited the loop when checking m_step_res.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_cur_max_delta_ts[i] = ts[i] - m_time[i];
        }

        // Run the integration timestep.
        step_impl(m_cur_max_delta_ts);

        // Check if the integration timestep produced an error condition.
        if (std::any_of(m_step_res.begin(), m_step_res.end(), [](const auto &tup) {
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
            m_ts_count[i] += static_cast<std::size_t>(std::get<1>(m_step_res[i]) != 0);
        }

        // Break out if we have reached the time limit for all
        // batch elements.
        if (std::all_of(m_step_res.begin(), m_step_res.end(),
                        [](const auto &tup) { return std::get<0>(tup) == taylor_outcome::time_limit; })) {
            break;
        }

        // Update min_h/max_h.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // Don't update if we reached the time limit.
            if (std::get<0>(m_step_res[i]) == taylor_outcome::time_limit) {
                continue;
            }

            using std::abs;
            const auto abs_h = abs(std::get<1>(m_step_res[i]));
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
        // We exited because we reached the max_steps limit: if the last integration step was successful
        // return step_limit, otherwise time_limit.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_prop_res[i]
                = std::tuple{std::get<0>(m_step_res[i]) == taylor_outcome::success ? taylor_outcome::step_limit
                                                                                   : taylor_outcome::time_limit,
                             m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }
    } else {
        // We exited either because of an error, or because all the batch
        // elements reached the time limits. In this case, we just use the
        // outcome of the last timestep.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_prop_res[i] = std::tuple{std::get<0>(m_step_res[i]), m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }
    }

    return m_prop_res;
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
std::uint32_t taylor_adaptive_batch_impl<T>::get_batch_size() const
{
    return m_batch_size;
}

template <typename T>
std::uint32_t taylor_adaptive_batch_impl<T>::get_dim() const
{
    return m_dim;
}

// Explicit instantiation of the batch implementation classes.
template class taylor_adaptive_batch_impl<double>;
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_batch_impl<double>::finalise_ctor_impl(std::vector<expression>, std::vector<double>, std::uint32_t,
                                                       std::vector<double>, double, bool, bool);
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_batch_impl<double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                       std::vector<double>, std::uint32_t, std::vector<double>, double,
                                                       bool, bool);

template class taylor_adaptive_batch_impl<long double>;
template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(std::vector<expression>,
                                                                                            std::vector<long double>,
                                                                                            std::uint32_t,
                                                                                            std::vector<long double>,
                                                                                            long double, bool, bool);
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(std::vector<std::pair<expression, expression>>,
                                                            std::vector<long double>, std::uint32_t,
                                                            std::vector<long double>, long double, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_batch_impl<mppp::real128>;
template HEYOKA_DLL_PUBLIC void
taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(std::vector<expression>, std::vector<mppp::real128>,
                                                              std::uint32_t, std::vector<mppp::real128>, mppp::real128,
                                                              bool, bool);
template HEYOKA_DLL_PUBLIC void
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
                         llvm::Value *u_idx, llvm::Value *val)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    auto ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx)});

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

// Function to split the central part of the decomposition (i.e., the definitions of the u variables
// that do not represent state variables) into parallelisable segments. Within a segment,
// the definition of a u variable does not depend on any u variable defined within that segment.
std::vector<std::vector<expression>> taylor_segment_dc(const std::vector<expression> &dc, std::uint32_t n_eq)
{
    // Helper that takes in input the definition ex of a u variable, and returns
    // in output the list of indices of the u variables on which ex depends.
    auto udef_args_indices = [](const expression &ex) -> std::vector<std::uint32_t> {
        return std::visit(
            [](const auto &v) -> std::vector<std::uint32_t> {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func> || std::is_same_v<type, binary_operator>) {
                    std::vector<std::uint32_t> retval;

                    for (const auto &arg : v.args()) {
                        if (auto p_var = std::get_if<variable>(&arg.value())) {
                            retval.push_back(uname_to_index(p_var->name()));
                        } else if (!std::holds_alternative<number>(arg.value())) {
                            throw std::invalid_argument(
                                "Invalid argument encountered in an element of a Taylor decomposition: the "
                                "argument is not a variable or a number");
                        }
                    }

                    return retval;
                } else {
                    throw std::invalid_argument("Invalid expression encountered in a Taylor decomposition: the "
                                                "expression is not a function or a binary operator");
                }
            },
            ex.value());
    };

    // Init the return value.
    std::vector<std::vector<expression>> s_dc;

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

        const auto &ex = dc[i];

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
        s_dc.back().push_back(ex);
    }

#if !defined(NDEBUG)
    // Verify s_dc.

    decltype(dc.size()) counter = 0;
    for (const auto &s : s_dc) {
        // No segment can be empty.
        assert(!s.empty());

        for (const auto &ex : s) {
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

    assert(counter == dc.size() - n_eq * 2u);
#endif

    return s_dc;
}

// Small helper to compute the size of a global array.
std::uint32_t taylor_c_gl_arr_size(llvm::Value *v)
{
    assert(llvm::isa<llvm::GlobalVariable>(v));

    return boost::numeric_cast<std::uint32_t>(
        llvm::cast<llvm::ArrayType>(llvm::cast<llvm::PointerType>(v->getType())->getElementType())->getNumElements());
}

// Helper to construct the global arrays needed for the computation of the
// derivatives of the state variables. The return value is a set
// of 4 arrays:
// - the indices of the state variables whose derivative is a u variable, paired to
// - the indices of the u variables appearing in the derivatives, and
// - the indices of the state variables whose derivative is a constant, paired to
// - the values of said constants.
template <typename T>
auto taylor_c_make_sv_diff_globals(llvm_state &s, const std::vector<expression> &dc, std::uint32_t n_uvars)
{
    auto &context = s.context();
    auto &builder = s.builder();
    auto &module = s.module();

    // Build iteratively the output values as vectors of constants.
    std::vector<llvm::Constant *> var_indices, vars, num_indices, nums;

    // NOTE: the derivatives of the state variables are at the end of the decomposition.
    for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
        std::visit(
            [&](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    // NOTE: remove from i the n_uvars offset to get the
                    // true index of the state variable.
                    var_indices.push_back(builder.getInt32(i - n_uvars));
                    vars.push_back(builder.getInt32(uname_to_index(v.name())));
                } else if constexpr (std::is_same_v<type, number>) {
                    num_indices.push_back(builder.getInt32(i - n_uvars));
                    nums.push_back(llvm::cast<llvm::Constant>(codegen<T>(s, v)));
                } else {
                    assert(false);
                }
            },
            dc[i].value());
    }

    assert(var_indices.size() == vars.size());
    assert(num_indices.size() == nums.size());

    // Turn the vectors into global read-only LLVM arrays.
    auto var_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(var_indices.size()));

    auto var_indices_arr = llvm::ConstantArray::get(var_arr_type, var_indices);
    auto g_var_indices = new llvm::GlobalVariable(module, var_indices_arr->getType(), true,
                                                  llvm::GlobalVariable::InternalLinkage, var_indices_arr);

    auto vars_arr = llvm::ConstantArray::get(var_arr_type, vars);
    auto g_vars
        = new llvm::GlobalVariable(module, vars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, vars_arr);

    auto num_indices_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(num_indices.size()));
    auto num_indices_arr = llvm::ConstantArray::get(num_indices_arr_type, num_indices);
    auto g_num_indices = new llvm::GlobalVariable(module, num_indices_arr->getType(), true,
                                                  llvm::GlobalVariable::InternalLinkage, num_indices_arr);

    auto nums_arr_type
        = llvm::ArrayType::get(to_llvm_type<T>(context), boost::numeric_cast<std::uint64_t>(nums.size()));
    auto nums_arr = llvm::ConstantArray::get(nums_arr_type, nums);
    auto g_nums
        = new llvm::GlobalVariable(module, nums_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, nums_arr);

    return std::array{g_var_indices, g_vars, g_num_indices, g_nums};
}

// Helper to compute and store the derivatives of the state variables in compact mode at order 'order'.
// sv_diff_gl is the set of arrays produced by taylor_c_make_sv_diff_globals(), which contain
// the indices/constants necessary for the computation.
template <typename T>
void taylor_c_compute_sv_diffs(llvm_state &s, const std::array<llvm::GlobalVariable *, 4> &sv_diff_gl,
                               llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                               std::uint32_t batch_size)
{
    auto &builder = s.builder();
    auto &context = s.context();

    // Recover the number of state variables whose derivatives are given
    // by u variables and numbers.
    const auto n_vars = taylor_c_gl_arr_size(sv_diff_gl[0]);
    const auto n_nums = taylor_c_gl_arr_size(sv_diff_gl[2]);

    // Handle the u variables definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_vars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        auto sv_idx = builder.CreateLoad(builder.CreateInBoundsGEP(sv_diff_gl[0], {builder.getInt32(0), cur_idx}));

        // Fetch the index of the u variable.
        auto u_idx = builder.CreateLoad(builder.CreateInBoundsGEP(sv_diff_gl[1], {builder.getInt32(0), cur_idx}));

        // Fetch from diff_arr the derivative of order 'order - 1' of the u variable u_idx.
        auto ret = taylor_c_load_diff(s, diff_arr, n_uvars, builder.CreateSub(order, builder.getInt32(1)), u_idx);

        // We have to divide the derivative by 'order' in order
        // to get the normalised derivative of the state variable.
        ret = builder.CreateFDiv(
            ret, vector_splat(builder, builder.CreateUIToFP(order, to_llvm_type<T>(context)), batch_size));

        // Store the derivative.
        taylor_c_store_diff(s, diff_arr, n_uvars, order, sv_idx, ret);
    });

    // Handle the number definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_nums), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        auto sv_idx = builder.CreateLoad(builder.CreateInBoundsGEP(sv_diff_gl[2], {builder.getInt32(0), cur_idx}));

        // Fetch the constant.
        auto num = builder.CreateLoad(builder.CreateInBoundsGEP(sv_diff_gl[3], {builder.getInt32(0), cur_idx}));

        // If the first-order derivative is being requested,
        // do the codegen for the constant itself, otherwise
        // return 0. No need for normalization as the only
        // nonzero value that can be produced here is the first-order
        // derivative.
        auto cmp_cond = builder.CreateICmpEQ(order, builder.getInt32(1));
        auto ret = builder.CreateSelect(cmp_cond, vector_splat(builder, num, batch_size),
                                        vector_splat(builder, codegen<T>(s, number{0.}), batch_size));

        // Store the derivative.
        taylor_c_store_diff(s, diff_arr, n_uvars, order, sv_idx, ret);
    });
}

// Helper to convert the arguments of the definition of a u variable
// into a vector of variants. u variables will be converted to their indices,
// numbers will be unchanged.
auto taylor_udef_to_variants(const expression &ex)
{
    return std::visit(
        [](const auto &v) -> std::vector<std::variant<std::uint32_t, number>> {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func> || std::is_same_v<type, binary_operator>) {
                std::vector<std::variant<std::uint32_t, number>> retval;

                for (const auto &arg : v.args()) {
                    if (auto p_var = std::get_if<variable>(&arg.value())) {
                        retval.emplace_back(uname_to_index(p_var->name()));
                    } else if (auto p_num = std::get_if<number>(&arg.value())) {
                        retval.emplace_back(*p_num);
                    } else {
                        throw std::invalid_argument(
                            "Invalid argument encountered in an element of a Taylor decomposition: the "
                            "argument is not a variable or a number");
                    }
                }

                return retval;
            } else {
                throw std::invalid_argument("Invalid expression encountered in a Taylor decomposition: the "
                                            "expression is not a function or a binary operator");
            }
        },
        ex.value());
}

// Helper to convert a vector of variants into a variant of vectors.
// All elements of v must be of the same type, and v cannot be empty.
template <typename... T>
auto taylor_c_vv_transpose(const std::vector<std::variant<T...>> &v)
{
    assert(!v.empty());

    // Init the return value based on the type
    // of the first element of v.
    auto retval = std::visit(
        [size = v.size()](const auto &x) {
            using type = detail::uncvref_t<decltype(x)>;

            std::vector<type> tmp;
            tmp.reserve(boost::numeric_cast<decltype(tmp.size())>(size));
            tmp.push_back(x);

            return std::variant<std::vector<T>...>(std::move(tmp));
        },
        v[0]);

    // Append the other values from v.
    for (decltype(v.size()) i = 1; i < v.size(); ++i) {
        std::visit(
            [&retval](const auto &x) {
                std::visit(
                    [&x](auto &vv) {
                        // The value type of retval.
                        using scal_t = typename detail::uncvref_t<decltype(vv)>::value_type;

                        // The type of the current element of v.
                        using x_t = detail::uncvref_t<decltype(x)>;

                        if constexpr (std::is_same_v<scal_t, x_t>) {
                            vv.push_back(x);
                        } else {
                            throw std::invalid_argument(
                                "Inconsistent types detected while building the transposed sets of "
                                "arguments for the Taylor derivative functions in compact mode");
                        }
                    },
                    retval);
            },
            v[i]);
    }

    return retval;
}

// Functions the create the arguments generators for the functions that compute
// the Taylor derivatives in compact mode. The generators are created from vectors
// of either u var indices or floating-point constants.
std::function<llvm::Value *(llvm::Value *)> taylor_c_make_arg_gen_vidx(llvm_state &s,
                                                                       const std::vector<std::uint32_t> &ind)
{
    assert(!ind.empty());

    auto &builder = s.builder();

    // Check if all indices in ind are the same.
    if (std::all_of(ind.begin() + 1, ind.end(), [&ind](const auto &n) { return n == ind[0]; })) {
        // If all indices are the same, don't construct an array, just always return
        // the same value.
        return [num = builder.getInt32(ind[0])](llvm::Value *) -> llvm::Value * { return num; };
    }

    // Check if ind consists of consecutive indices.
    bool are_consecutive = true;
    auto prev_ind = ind[0];
    for (decltype(ind.size()) i = 1; i < ind.size(); ++i) {
        if (ind[i] != prev_ind + 1u) {
            are_consecutive = false;
            break;
        }
        prev_ind = ind[i];
    }

    if (are_consecutive) {
        // If ind consists of consecutive indices, we can replace
        // the index array with a simple offset computation.
        return [&s, start_idx = builder.getInt32(ind[0])](llvm::Value *cur_call_idx) -> llvm::Value * {
            return s.builder().CreateAdd(start_idx, cur_call_idx);
        };
    }

    auto &module = s.module();

    // Generate the array of indices as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    for (const auto &val : ind) {
        tmp_c_vec.push_back(builder.getInt32(val));
    }

    // Create the array type.
    auto arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(ind.size()));
    assert(arr_type != nullptr);

    // Create the constant array as a global read-only variable.
    auto const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr);
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto gvar = new llvm::GlobalVariable(module, const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage,
                                         const_arr);

    // Return the generator.
    return [gvar, &s](llvm::Value *cur_call_idx) -> llvm::Value * {
        auto &builder = s.builder();

        return builder.CreateLoad(builder.CreateInBoundsGEP(gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

template <typename T>
std::function<llvm::Value *(llvm::Value *)> taylor_c_make_arg_gen_vc(llvm_state &s, const std::vector<number> &vc)
{
    assert(!vc.empty());

    // Check if all the numbers are the same.
    // NOTE: the current implementation of operator== for number will return false
    // if the internal number types differ, even if their values are equal. Thus this
    // simplification will not trigger if the ODE expressions contain different
    // number types. Perhaps in the future we can change the operator== implementation
    // to take into account the values only, but then we need to adapt the hasher as well.
    if (std::all_of(vc.begin() + 1, vc.end(), [&vc](const auto &n) { return n == vc[0]; })) {
        // If all constants are the same, don't construct an array, just always return
        // the same value.
        return [num = codegen<T>(s, vc[0])](llvm::Value *) -> llvm::Value * { return num; };
    }

    auto &module = s.module();

    // Generate the array of constants as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    for (const auto &val : vc) {
        tmp_c_vec.push_back(llvm::cast<llvm::Constant>(codegen<T>(s, val)));
    }

    // Create the array type.
    auto arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(vc.size()));
    assert(arr_type != nullptr);

    // Create the constant array as a global read-only variable.
    auto const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr);
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto gvar = new llvm::GlobalVariable(module, const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage,
                                         const_arr);

    // Return the generator.
    return [gvar, &s](llvm::Value *cur_call_idx) -> llvm::Value * {
        auto &builder = s.builder();

        return builder.CreateLoad(builder.CreateInBoundsGEP(gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

// For each segment in s_dc, this function will return a vector containing a dict mapping an LLVM function
// f for the computation of a Taylor derivative to a size and a vector of std::functions. For example, one entry
// in the return value will read something like:
// {f : (2, [g_0, g_1, g_2])}
// The meaning in this example is that the arity of f is 3 and it will be called with 2 different
// sets of arguments. The g_i functions are expected to be called with input argument j in [0, 1]
// to yield the value of the i-th function argument for f at the j-th invocation.
template <typename T>
auto taylor_build_function_maps(llvm_state &s, const std::vector<std::vector<expression>> &s_dc, std::uint32_t n_eq,
                                std::uint32_t n_uvars, std::uint32_t batch_size)
{
    // Init the return value.
    std::vector<std::unordered_map<llvm::Function *,
                                   std::pair<std::uint32_t, std::vector<std::function<llvm::Value *(llvm::Value *)>>>>>
        retval;

    // Variable to keep track of the u variable
    // on whose definition we are operating.
    auto cur_u_idx = n_eq;
    for (const auto &seg : s_dc) {
        // This structure maps an LLVM function to sets of arguments
        // with which the function is to be called. For instance, if function
        // f(x, y, z) is to be called as f(a, b, c) and f(d, e, f), then tmp_map
        // will contain {f : [[a, b, c], [d, e, f]]}.
        // After construction, we have verified that for each function
        // in the map the sets of arguments have all the same size.
        std::unordered_map<llvm::Function *, std::vector<std::vector<std::variant<std::uint32_t, number>>>> tmp_map;

        for (const auto &ex : seg) {
            // Get the function for the computation of the derivative.
            auto func = taylor_c_diff_func<T>(s, ex, n_uvars, batch_size);

            // Insert the function into tmp_map.
            const auto [it, is_new_func] = tmp_map.try_emplace(func);

            assert(is_new_func || !it->second.empty());

            // Convert the variables/constants in the current dc
            // element into a set of indices/constants.
            const auto cdiff_args = taylor_udef_to_variants(ex);

            if (!is_new_func && it->second.back().size() - 1u != cdiff_args.size()) {
                throw std::invalid_argument("Inconsistent arity detected in a Taylor derivative function in compact "
                                            "mode: the same function is being called with both "
                                            + std::to_string(it->second.back().size() - 1u) + " and "
                                            + std::to_string(cdiff_args.size()) + " arguments");
            }

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
        std::unordered_map<llvm::Function *, std::vector<std::variant<std::vector<std::uint32_t>, std::vector<number>>>>
            tmp_map_transpose;
        for (const auto &[func, vv] : tmp_map) {
            assert(!vv.empty());

            // Add the function.
            const auto [it, ins_status] = tmp_map_transpose.try_emplace(func);
            assert(ins_status);

            const auto n_calls = vv.size();
            const auto n_args = vv[0].size();
            // NOTE: n_args must be at least 1 because the u idx
            // is prepended to the actual function arguments in
            // the tmp_map entries.
            assert(n_args >= 1u);

            for (decltype(vv[0].size()) i = 0; i < n_args; ++i) {
                // Build the vector of values corresponding
                // to the current argument index.
                std::vector<std::variant<std::uint32_t, number>> tmp_c_vec;
                for (decltype(vv.size()) j = 0; j < n_calls; ++j) {
                    tmp_c_vec.push_back(vv[j][i]);
                }

                // Turn tmp_c_vec (a vector of variants) into a variant
                // of vectors, and insert the result.
                it->second.push_back(taylor_c_vv_transpose(tmp_c_vec));
            }
        }

        // Add a new entry in retval for the current segment.
        retval.emplace_back();
        auto &a_map = retval.back();

        for (const auto &[func, vv] : tmp_map_transpose) {
            assert(!vv.empty());

            // Add the function.
            const auto [it, ins_status] = a_map.try_emplace(func);
            assert(ins_status);

            // Set the number of calls for this function.
            it->second.first
                = std::visit([](const auto &x) { return boost::numeric_cast<std::uint32_t>(x.size()); }, vv[0]);
            assert(it->second.first > 0u);

            // Create the g functions for each argument.
            for (const auto &v : vv) {
                it->second.second.push_back(std::visit(
                    [&s](const auto &x) {
                        using type = detail::uncvref_t<decltype(x)>;

                        if constexpr (std::is_same_v<type, std::vector<std::uint32_t>>) {
                            return taylor_c_make_arg_gen_vidx(s, x);
                        } else {
                            return taylor_c_make_arg_gen_vc<T>(s, x);
                        }
                    },
                    v));
            }
        }
    }

    return retval;
}

// Helper for the computation of a jet of derivatives in compact mode,
// used in taylor_compute_jet() below.
template <typename T>
llvm::Value *taylor_compute_jet_compact_mode(llvm_state &s, llvm::Value *order0, const std::vector<expression> &dc,
                                             std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order,
                                             std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Split dc into segments.
    const auto s_dc = taylor_segment_dc(dc, n_eq);

    // Generate the function maps.
    const auto f_maps = taylor_build_function_maps<T>(s, s_dc, n_eq, n_uvars, batch_size);

    // Generate the global arrays for the computation of the derivatives
    // of the state variables.
    const auto sv_diff_gl = taylor_c_make_sv_diff_globals<T>(s, dc, n_uvars);

    // Prepare the array that will contain the jet of derivatives.
    // We will be storing all the derivatives of the u variables
    // up to order 'order - 1', plus the derivatives of order
    // 'order' of the state variables only.
    // NOTE: the array size is specified as a 64-bit integer in the
    // LLVM API.
    // NOTE: fp_type is the original, scalar floating-point type.
    // It will be turned into a vector type (if necessary) by
    // make_vector_type() below.
    auto fp_type = llvm::cast<llvm::PointerType>(order0->getType())->getElementType();
    auto array_type = llvm::ArrayType::get(make_vector_type(fp_type, batch_size), n_uvars * order + n_eq);

    // Make the global array and fetch a pointer to its first element.
    // NOTE: we use a global array rather than a local one here because
    // its size can grow quite large, which can lead to stack overflow issues.
    // This has of course consequences in terms of thread safety, which
    // we will have to document.
    auto diff_arr = builder.CreateInBoundsGEP(make_global_zero_array(s.module(), array_type),
                                              {builder.getInt32(0), builder.getInt32(0)});

    // Copy over the order-0 derivatives of the state variables.
    // NOTE: overflow checking is already done in the parent function.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
        // Fetch the pointer from order0.
        auto ptr = builder.CreateInBoundsGEP(order0, {builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))});

        // Load as a vector.
        auto vec = load_vector_from_memory(builder, ptr, batch_size);

        // Store into diff_arr.
        builder.CreateStore(vec, builder.CreateInBoundsGEP(diff_arr, {cur_var_idx}));
    });

    // Helper to compute and store the derivatives of order cur_order
    // of the u variables which are not state variables.
    auto compute_u_diffs = [&](llvm::Value *cur_order) {
        for (const auto &map : f_maps) {
            for (const auto &p : map) {
                // The LLVM function for the computation of the
                // derivative in compact mpde.
                const auto &func = p.first;

                // The number of func calss.
                const auto ncalls = p.second.first;

                // The generators for the arguments of func.
                const auto &gens = p.second.second;

                assert(ncalls > 0u);
                assert(!gens.empty());
                assert(std::all_of(gens.begin(), gens.end(), [](const auto &f) { return static_cast<bool>(f); }));

                // Loop over the number of calls.
                llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(ncalls), [&](llvm::Value *cur_call_idx) {
                    // Create the u variable index from the first generator.
                    auto u_idx = gens[0](cur_call_idx);

                    // Initialise the vector of arguments with which func must be called:
                    // - current Taylor order,
                    // - u index of the variable,
                    // - array of derivatives.
                    std::vector<llvm::Value *> args{cur_order, u_idx, diff_arr};

                    // Create the other arguments via the generators.
                    for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                        args.push_back(gens[i](cur_call_idx));
                    }

                    // Calculate the derivative and store the result.
                    taylor_c_store_diff(s, diff_arr, n_uvars, cur_order, u_idx, builder.CreateCall(func, args));
                });
            }
        }
    };

    // Compute the order-0 derivatives (i.e., the initial values)
    // for all u variables which are not state variables.
    compute_u_diffs(builder.getInt32(0));

    // Compute all derivatives up to order 'order - 1'.
    llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(order), [&](llvm::Value *cur_order) {
        // State variables first.
        taylor_c_compute_sv_diffs<T>(s, sv_diff_gl, diff_arr, n_uvars, cur_order, batch_size);

        // The other u variables.
        compute_u_diffs(cur_order);
    });

    // Compute the last-order derivatives for the state variables.
    taylor_c_compute_sv_diffs<T>(s, sv_diff_gl, diff_arr, n_uvars, builder.getInt32(order), batch_size);

    // Return the array of derivatives of the u variables.
    return diff_arr;
}

// Given an input pointer 'in', load the first n * batch_size values in it as n vectors
// with size batch_size. If batch_size is 1, the values will be loaded as scalars.
auto taylor_load_values(llvm_state &s, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    auto &builder = s.builder();

    std::vector<llvm::Value *> retval;
    for (std::uint32_t i = 0; i < n; ++i) {
        // Fetch the pointer from in.
        // NOTE: overflow checking is done in the parent function.
        auto ptr = builder.CreateInBoundsGEP(in, {builder.getInt32(i * batch_size)});

        // Load the value in vector mode.
        retval.push_back(load_vector_from_memory(builder, ptr, batch_size));
    }

    return retval;
}

// Helper function to compute the jet of Taylor derivatives up to a given order. n_eq
// is the number of equations/variables in the ODE sys, dc its Taylor decomposition,
// n_uvars the total number of u variables in the decomposition.
// order is the max derivative order desired, batch_size the batch size.
// order0 is a pointer to an array of (at least) n_eq * batch_size scalar elements
// containing the derivatives of order 0.
//
// The return value is a variant containing either:
// - in compact mode, the array containing the derivatives of all u variables,
// - otherwise, the jet of derivatives of the state variables up to order 'order'.
template <typename T>
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_compute_jet(llvm_state &s, llvm::Value *order0, const std::vector<expression> &dc, std::uint32_t n_eq,
                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size, bool compact_mode)
{
    assert(batch_size > 0u);
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

    // We also need to be able to index up to n_eq * batch_size in order0.
    if (n_eq > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives");
    }

    if (compact_mode) {
        return taylor_compute_jet_compact_mode<T>(s, order0, dc, n_eq, n_uvars, order, batch_size);
    } else {
        // Init the derivatives array with the order 0 of the state variables.
        auto diff_arr = taylor_load_values(s, order0, n_eq, batch_size);

        // Compute the order-0 derivatives of the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            diff_arr.push_back(taylor_diff<T>(s, dc[i], diff_arr, n_uvars, 0, i, batch_size));
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
        std::vector<llvm::Value *> retval;
        for (std::uint32_t o = 0; o <= order; ++o) {
            for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
                retval.push_back(taylor_fetch_diff(diff_arr, var_idx, o, n_uvars));
            }
        }

        return retval;
    }
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

    auto &builder = s.builder();

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    auto dc = taylor_decompose(std::move(sys));

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    // Prepare the function prototype. The first argument is a float pointer to the in/out array,
    // the second argument a const float pointer to the pars. These arrays cannot overlap.
    std::vector<llvm::Type *> fargs(2, llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())));
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
    in_out->addAttr(llvm::Attribute::NoCapture);
    in_out->addAttr(llvm::Attribute::NoAlias);

    auto par_ptr = in_out + 1;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr);
    s.builder().SetInsertPoint(bb);

    // Compute the jet of derivatives.
    auto diff_variant = taylor_compute_jet<T>(s, in_out, dc, n_eq, n_uvars, order, batch_size, compact_mode);

    // Write the derivatives to in_out.
    // NOTE: overflow checking. We need to be able to index into the jet array (size n_eq * (order + 1) * batch_size)
    // using uint32_t.
    if (order == std::numeric_limits<std::uint32_t>::max()
        || (order + 1u) > std::numeric_limits<std::uint32_t>::max() / batch_size
        || n_eq > std::numeric_limits<std::uint32_t>::max() / ((order + 1u) * batch_size)) {
        throw std::overflow_error("An overflow condition was detected while adding a Taylor jet");
    }

    if (compact_mode) {
        auto diff_arr = std::get<llvm::Value *>(diff_variant);

        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                      [&](llvm::Value *cur_order) {
                          llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_idx) {
                              // Compute the index in diff_arr.
                              // NOTE: in compact mode diff_arr contains the derivatives for *all* uvars,
                              // hence the indexing is cur_order * n_uvars + cur_idx.
                              auto arr_idx
                                  = builder.CreateAdd(builder.CreateMul(cur_order, builder.getInt32(n_uvars)), cur_idx);

                              // Load the derivative value from diff_arr.
                              auto diff_val = builder.CreateLoad(builder.CreateInBoundsGEP(diff_arr, {arr_idx}));

                              // Compute the index in the output pointer.
                              auto out_idx
                                  = builder.CreateAdd(builder.CreateMul(builder.getInt32(n_eq * batch_size), cur_order),
                                                      builder.CreateMul(cur_idx, builder.getInt32(batch_size)));

                              // Store into in_out.
                              store_vector_to_memory(builder, builder.CreateInBoundsGEP(in_out, {out_idx}), diff_val);
                          });
                      });
    } else {
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_variant);

        for (decltype(diff_arr.size()) cur_order = 1; cur_order <= order; ++cur_order) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                // Index in the jet of derivatives.
                // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
                // state variables (not all u vars), hence the indexing is cur_order * n_eq + j.
                const auto arr_idx = cur_order * n_eq + j;
                assert(arr_idx < diff_arr.size());
                const auto val = diff_arr[arr_idx];

                // Index in the output array.
                const auto out_idx = n_eq * batch_size * cur_order + j * batch_size;
                auto out_ptr
                    = builder.CreateInBoundsGEP(in_out, {builder.getInt32(static_cast<std::uint32_t>(out_idx))});
                store_vector_to_memory(builder, out_ptr, val);
            }
        }
    }

    // Finish off the function.
    builder.CreateRetVoid();

    // Verify it.
    s.verify_function(f);

    // Run the optimisation pass.
    s.optimise();

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

// Run the Horner scheme to propagate an ODE state via the evaluation of the Taylor polynomials.
// diff_var contains either the derivatives for all u variables (in compact mode) or only
// for the state variables (non-compact mode). The evaluation point (i.e., the timestep)
// is h. The evaluation is run in parallel over the polynomials of all the state
// variables.
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_multihorner(llvm_state &s, const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_var,
                       llvm::Value *h, std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                       bool compact_mode)
{
    auto &builder = s.builder();

    if (compact_mode) {
        // Compact mode.
        auto diff_arr = std::get<llvm::Value *>(diff_var);

        // Create the array storing the results of the evaluation.
        auto array_type = llvm::ArrayType::get(pointee_type(diff_arr), n_eq);
        auto res_arr
            = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type), {builder.getInt32(0), builder.getInt32(0)});

        // Init the return value, filling it with the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from diff_arr.
            auto val = builder.CreateLoad(builder.CreateInBoundsGEP(
                diff_arr, {builder.CreateAdd(builder.getInt32(order * n_uvars), cur_var_idx)}));

            // Store it in res_arr.
            builder.CreateStore(val, builder.CreateInBoundsGEP(res_arr, {cur_var_idx}));
        });

        // Run the evaluation.
        llvm_loop_u32(
            s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                    // Load the current poly coeff from diff_arr.
                    // NOTE: the index is (order - cur_order) * n_uvars + cur_var_idx.
                    auto cf = builder.CreateLoad(builder.CreateInBoundsGEP(
                        diff_arr,
                        {builder.CreateAdd(builder.CreateMul(builder.CreateSub(builder.getInt32(order), cur_order),
                                                             builder.getInt32(n_uvars)),
                                           cur_var_idx)}));

                    // Accumulate in res_arr.
                    auto res_ptr = builder.CreateInBoundsGEP(res_arr, {cur_var_idx});
                    builder.CreateStore(builder.CreateFAdd(cf, builder.CreateFMul(builder.CreateLoad(res_ptr), h)),
                                        res_ptr);
                });
            });

        return res_arr;
    } else {
        // Non-compact mode.
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_var);

        // Init the return value, filling it with the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        std::vector<llvm::Value *> res_arr;
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            res_arr.push_back(diff_arr[(n_eq * order) + i]);
        }

        // Run the Horner scheme simultaneously for all polynomials.
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                res_arr[j] = builder.CreateFAdd(diff_arr[(order - i) * n_eq + j], builder.CreateFMul(res_arr[j], h));
            }
        }

        return res_arr;
    }
}

// Same as the previous function, but here the data is always coming in as a
// pointer to scalar FP values representing the derivatives of the state variables.
// The same pointer is also used for output. Hence, the internal logic and indexing are
// different.
void taylor_run_multihorner_state_updater(llvm_state &s, llvm::Value *jet_ptr, llvm::Value *h, std::uint32_t n_eq,
                                          std::uint32_t order, std::uint32_t batch_size, bool compact_mode)
{
    auto &builder = s.builder();

    if (compact_mode) {
        // Compact mode.

        // Create the array storing the result of the evaluation.
        auto array_type = llvm::ArrayType::get(make_vector_type(pointee_type(jet_ptr), batch_size), n_eq);
        auto res_arr
            = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type), {builder.getInt32(0), builder.getInt32(0)});

        // Init res_arr, filling it with the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from jet_ptr.
            // NOTE: the index is order * n_eq * batch_size + cur_var_idx * batch_size.
            // NOTE: overflow checking was done in taylor_add_jet_impl().
            auto ptr = builder.CreateInBoundsGEP(
                jet_ptr, {builder.CreateAdd(builder.getInt32(order * n_eq * batch_size),
                                            builder.CreateMul(builder.getInt32(batch_size), cur_var_idx))});
            auto val = load_vector_from_memory(builder, ptr, batch_size);

            // Store it in res_arr.
            builder.CreateStore(val, builder.CreateInBoundsGEP(res_arr, {cur_var_idx}));
        });

        // Run the evaluation.
        llvm_loop_u32(
            s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                    // Load the current poly coeff from jet_ptr.
                    // NOTE: the index is (order - cur_order) * n_eq * batch_size + cur_var_idx * batch_size.
                    auto cf_ptr = builder.CreateInBoundsGEP(
                        jet_ptr,
                        {builder.CreateAdd(builder.CreateMul(builder.CreateSub(builder.getInt32(order), cur_order),
                                                             builder.getInt32(n_eq * batch_size)),
                                           builder.CreateMul(cur_var_idx, builder.getInt32(batch_size)))});
                    auto cf = load_vector_from_memory(builder, cf_ptr, batch_size);

                    // Accumulate in res_arr.
                    auto res_ptr = builder.CreateInBoundsGEP(res_arr, {cur_var_idx});
                    builder.CreateStore(builder.CreateFAdd(cf, builder.CreateFMul(builder.CreateLoad(res_ptr), h)),
                                        res_ptr);
                });
            });

        // Copy the result to jet_ptr.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            auto val = builder.CreateLoad(builder.CreateInBoundsGEP(res_arr, {cur_var_idx}));
            store_vector_to_memory(
                builder,
                builder.CreateInBoundsGEP(jet_ptr, {builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))}),
                val);
        });
    } else {
        // Non-compact mode.

        // Create the array of results, initially containing the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        std::vector<llvm::Value *> res_arr;
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            auto ptr
                = builder.CreateInBoundsGEP(jet_ptr, {builder.getInt32(order * n_eq * batch_size + i * batch_size)});
            res_arr.push_back(load_vector_from_memory(builder, ptr, batch_size));
        }

        // Run the evaluation.
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                auto ptr = builder.CreateInBoundsGEP(
                    jet_ptr, {builder.getInt32((order - i) * n_eq * batch_size + j * batch_size)});
                res_arr[j] = builder.CreateFAdd(load_vector_from_memory(builder, ptr, batch_size),
                                                builder.CreateFMul(res_arr[j], h));
            }
        }

        // Write the result to jet_ptr.
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            store_vector_to_memory(builder, builder.CreateInBoundsGEP(jet_ptr, {builder.getInt32(batch_size * i)}),
                                   res_arr[i]);
        }
    }
}

// Same as taylor_run_multihorner(), but instead of the Horner scheme this implementation uses
// a compensated summation over the naive evaluation of monomials.
template <typename T>
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_ceval(llvm_state &s, const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_var, llvm::Value *h,
                 std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size, bool,
                 bool compact_mode)
{
    auto &builder = s.builder();

    if (compact_mode) {
        // Compact mode.
        auto diff_arr = std::get<llvm::Value *>(diff_var);

        // Create the arrays storing the results of the evaluation and the running compensations.
        auto array_type = llvm::ArrayType::get(pointee_type(diff_arr), n_eq);
        auto res_arr
            = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type), {builder.getInt32(0), builder.getInt32(0)});
        auto comp_arr
            = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type), {builder.getInt32(0), builder.getInt32(0)});

        // Init res_arr with the order-0 monomials, and the running
        // compensations with zero.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from diff_arr.
            auto val = builder.CreateLoad(builder.CreateInBoundsGEP(diff_arr, {cur_var_idx}));

            // Store it in res_arr.
            builder.CreateStore(val, builder.CreateInBoundsGEP(res_arr, {cur_var_idx}));

            // Zero-init the element in comp_arr.
            builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size),
                                builder.CreateInBoundsGEP(comp_arr, {cur_var_idx}));
        });

        // Init the running updater for the powers of h.
        auto cur_h = builder.CreateAlloca(h->getType());
        builder.CreateStore(h, cur_h);

        // Run the evaluation.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                      [&](llvm::Value *cur_order) {
                          // Load the current power of h.
                          auto cur_h_val = builder.CreateLoad(cur_h);

                          llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                              // Evaluate the current monomial.
                              // NOTE: the index is cur_order * n_uvars + cur_var_idx.
                              auto cf = builder.CreateLoad(builder.CreateInBoundsGEP(
                                  diff_arr, {builder.CreateAdd(builder.CreateMul(cur_order, builder.getInt32(n_uvars)),
                                                               cur_var_idx)}));
                              auto tmp = builder.CreateFMul(cf, cur_h_val);

                              // Compute the quantities for the compensation.
                              auto comp_ptr = builder.CreateInBoundsGEP(comp_arr, {cur_var_idx});
                              auto res_ptr = builder.CreateInBoundsGEP(res_arr, {cur_var_idx});
                              auto y = builder.CreateFSub(tmp, builder.CreateLoad(comp_ptr));
                              auto cur_res = builder.CreateLoad(res_ptr);
                              auto t = builder.CreateFAdd(cur_res, y);

                              // Update the compensation and the return value.
                              builder.CreateStore(builder.CreateFSub(builder.CreateFSub(t, cur_res), y), comp_ptr);
                              builder.CreateStore(t, res_ptr);
                          });

                          // Update the value of h.
                          builder.CreateStore(builder.CreateFMul(cur_h_val, h), cur_h);
                      });

        return res_arr;
    } else {
        // Non-compact mode.
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_var);

        // Init the return values with the order-0 monomials, and the running
        // compensations with zero.
        std::vector<llvm::Value *> res_arr, comp_arr;
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            res_arr.push_back(diff_arr[i]);
            comp_arr.push_back(vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
        }

        // Evaluate and sum.
        auto cur_h = h;
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                // Evaluate the current monomial.
                auto tmp = builder.CreateFMul(diff_arr[i * n_eq + j], cur_h);

                // Compute the quantities for the compensation.
                auto y = builder.CreateFSub(tmp, comp_arr[j]);
                auto t = builder.CreateFAdd(res_arr[j], y);

                // Update the compensation and the return value.
                comp_arr[j] = builder.CreateFSub(builder.CreateFSub(t, res_arr[j]), y);
                res_arr[j] = t;
            }

            // Update the power of h.
            cur_h = builder.CreateFMul(cur_h, h);
        }

        return res_arr;
    }
}

// Same as the previous function, but here the data is always coming in as a
// pointer to scalar FP values representing the derivatives of the state variables.
// The same pointer is also used for output. Hence, the internal logic and indexing are
// different.
template <typename T>
void taylor_run_ceval_state_updater(llvm_state &s, llvm::Value *jet_ptr, llvm::Value *h, std::uint32_t n_eq,
                                    std::uint32_t order, std::uint32_t batch_size, bool, bool compact_mode)
{
    auto &builder = s.builder();

    if (compact_mode) {
        // Compact mode.

        // Create the array storing the results of the evaluation and the running compensations.
        auto array_type = llvm::ArrayType::get(make_vector_type(pointee_type(jet_ptr), batch_size), n_eq);
        auto res_arr
            = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type), {builder.getInt32(0), builder.getInt32(0)});
        auto comp_arr
            = builder.CreateInBoundsGEP(builder.CreateAlloca(array_type), {builder.getInt32(0), builder.getInt32(0)});

        // Init res_arr with the order-0 monomials, and the running
        // compensations with zero.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from jet_ptr.
            // NOTE: the index is cur_var_idx * batch_size.
            // NOTE: overflow checking was done in taylor_add_jet_impl().
            auto ptr
                = builder.CreateInBoundsGEP(jet_ptr, {builder.CreateMul(builder.getInt32(batch_size), cur_var_idx)});
            auto val = load_vector_from_memory(builder, ptr, batch_size);

            // Store it in res_arr.
            builder.CreateStore(val, builder.CreateInBoundsGEP(res_arr, {cur_var_idx}));

            // Zero-init the element in comp_arr.
            builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size),
                                builder.CreateInBoundsGEP(comp_arr, {cur_var_idx}));
        });

        // Init the running updater for the powers of h.
        auto cur_h = builder.CreateAlloca(h->getType());
        builder.CreateStore(h, cur_h);

        // Run the evaluation.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                      [&](llvm::Value *cur_order) {
                          // Load the current power of h.
                          auto cur_h_val = builder.CreateLoad(cur_h);

                          llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                              // Evaluate the current monomial.
                              // NOTE: the index is cur_order * n_eq * batch_size + cur_var_idx * batch_size.
                              auto cf_ptr = builder.CreateInBoundsGEP(
                                  jet_ptr,
                                  {builder.CreateAdd(builder.CreateMul(cur_order, builder.getInt32(n_eq * batch_size)),
                                                     builder.CreateMul(cur_var_idx, builder.getInt32(batch_size)))});
                              auto cf = load_vector_from_memory(builder, cf_ptr, batch_size);
                              auto tmp = builder.CreateFMul(cf, cur_h_val);

                              // Compute the quantities for the compensation.
                              auto comp_ptr = builder.CreateInBoundsGEP(comp_arr, {cur_var_idx});
                              auto res_ptr = builder.CreateInBoundsGEP(res_arr, {cur_var_idx});
                              auto y = builder.CreateFSub(tmp, builder.CreateLoad(comp_ptr));
                              auto cur_res = builder.CreateLoad(res_ptr);
                              auto t = builder.CreateFAdd(cur_res, y);

                              // Update the compensation and the result.
                              builder.CreateStore(builder.CreateFSub(builder.CreateFSub(t, cur_res), y), comp_ptr);
                              builder.CreateStore(t, res_ptr);
                          });

                          // Update the value of h.
                          builder.CreateStore(builder.CreateFMul(cur_h_val, h), cur_h);
                      });

        // Copy the result to jet_ptr.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            auto val = builder.CreateLoad(builder.CreateInBoundsGEP(res_arr, {cur_var_idx}));
            store_vector_to_memory(
                builder,
                builder.CreateInBoundsGEP(jet_ptr, {builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))}),
                val);
        });
    } else {
        // Non-compact mode.

        // Init the results with the order-0 monomials, and the running
        // compensations with zero.
        std::vector<llvm::Value *> res_arr, comp_arr;
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            auto ptr = builder.CreateInBoundsGEP(jet_ptr, {builder.getInt32(i * batch_size)});
            res_arr.push_back(load_vector_from_memory(builder, ptr, batch_size));

            comp_arr.push_back(vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
        }

        // Evaluate and sum.
        auto cur_h = h;
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                auto cf_ptr
                    = builder.CreateInBoundsGEP(jet_ptr, {builder.getInt32(i * n_eq * batch_size + j * batch_size)});
                auto cf = load_vector_from_memory(builder, cf_ptr, batch_size);
                auto tmp = builder.CreateFMul(cf, cur_h);

                // Compute the quantities for the compensation.
                auto y = builder.CreateFSub(tmp, comp_arr[j]);
                auto t = builder.CreateFAdd(res_arr[j], y);

                // Update the compensation and the result.
                comp_arr[j] = builder.CreateFSub(builder.CreateFSub(t, res_arr[j]), y);
                res_arr[j] = t;
            }

            // Update the power of h.
            cur_h = builder.CreateFMul(cur_h, h);
        }

        // Write the result to jet_ptr.
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            store_vector_to_memory(builder, builder.CreateInBoundsGEP(jet_ptr, {builder.getInt32(batch_size * i)}),
                                   res_arr[i]);
        }
    }
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
    auto &context = s.context();

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

    // Compute the jet of derivatives at the given order.
    // NOTE: in taylor_compute_jet() we ensure that n_uvars * order + n_eq
    // is representable as a 32-bit unsigned integer.
    auto diff_variant = taylor_compute_jet<T>(s, state_ptr, dc, n_eq, n_uvars, order, batch_size, compact_mode);

    llvm::Value *max_abs_state, *max_abs_diff_o, *max_abs_diff_om1;

    if (compact_mode) {
        auto diff_arr = std::get<llvm::Value *>(diff_variant);

        // Compute the norm infinity of the state vector and the norm infinity of the derivatives
        // at orders order and order - 1.
        max_abs_state = builder.CreateAlloca(to_llvm_vector_type<T>(context, batch_size));
        max_abs_diff_o = builder.CreateAlloca(to_llvm_vector_type<T>(context, batch_size));
        max_abs_diff_om1 = builder.CreateAlloca(to_llvm_vector_type<T>(context, batch_size));

        // Initialise with the abs(derivatives) of the first state variable at orders 0, 'order' and 'order - 1'.
        builder.CreateStore(
            taylor_step_abs(s, builder.CreateLoad(builder.CreateInBoundsGEP(diff_arr, {builder.getInt32(0)}))),
            max_abs_state);
        // NOTE: in compact mode diff_arr contains the derivatives for *all* uvars,
        // hence the indexing is order * n_uvars.
        builder.CreateStore(taylor_step_abs(s, builder.CreateLoad(builder.CreateInBoundsGEP(
                                                   diff_arr, {builder.getInt32(order * n_uvars)}))),
                            max_abs_diff_o);
        builder.CreateStore(taylor_step_abs(s, builder.CreateLoad(builder.CreateInBoundsGEP(
                                                   diff_arr, {builder.getInt32((order - 1u) * n_uvars)}))),
                            max_abs_diff_om1);

        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n_eq), [&](llvm::Value *cur_idx) {
            builder.CreateStore(taylor_step_maxabs(s, builder.CreateLoad(max_abs_state),
                                                   builder.CreateLoad(builder.CreateInBoundsGEP(diff_arr, {cur_idx}))),
                                max_abs_state);
            builder.CreateStore(
                taylor_step_maxabs(s, builder.CreateLoad(max_abs_diff_o),
                                   builder.CreateLoad(builder.CreateInBoundsGEP(
                                       diff_arr, {builder.CreateAdd(builder.getInt32(order * n_uvars), cur_idx)}))),
                max_abs_diff_o);
            builder.CreateStore(
                taylor_step_maxabs(
                    s, builder.CreateLoad(max_abs_diff_om1),
                    builder.CreateLoad(builder.CreateInBoundsGEP(
                        diff_arr, {builder.CreateAdd(builder.getInt32((order - 1u) * n_uvars), cur_idx)}))),
                max_abs_diff_om1);
        });

        // Load the values for later use.
        max_abs_state = builder.CreateLoad(max_abs_state);
        max_abs_diff_o = builder.CreateLoad(max_abs_diff_o);
        max_abs_diff_om1 = builder.CreateLoad(max_abs_diff_om1);
    } else {
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_variant);

        // Compute the norm infinity of the state vector and the norm infinity of the derivatives
        // at orders order and order - 1.
        max_abs_state = taylor_step_abs(s, diff_arr[0]);
        // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
        // state variables (not all u vars), hence the indexing is order * n_eq.
        max_abs_diff_o = taylor_step_abs(s, diff_arr[order * n_eq]);
        max_abs_diff_om1 = taylor_step_abs(s, diff_arr[(order - 1u) * n_eq]);
        for (std::uint32_t i = 1; i < n_eq; ++i) {
            max_abs_state = taylor_step_maxabs(s, max_abs_state, diff_arr[i]);
            max_abs_diff_o = taylor_step_maxabs(s, max_abs_diff_o, diff_arr[order * n_eq + i]);
            max_abs_diff_om1 = taylor_step_maxabs(s, max_abs_diff_om1, diff_arr[(order - 1u) * n_eq + i]);
        }
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

    // Evaluate the Taylor polynomials, producing the updated state of the system.
    auto new_state_var
        = high_accuracy
              ? taylor_run_ceval<T>(s, diff_variant, h, n_eq, n_uvars, order, batch_size, high_accuracy, compact_mode)
              : taylor_run_multihorner(s, diff_variant, h, n_eq, n_uvars, order, batch_size, compact_mode);

    // Store the new state.
    // NOTE: no need to perform overflow check on n_eq * batch_size,
    // as in taylor_compute_jet() we already checked.
    if (compact_mode) {
        auto new_state = std::get<llvm::Value *>(new_state_var);

        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            auto val = builder.CreateLoad(builder.CreateInBoundsGEP(new_state, {cur_var_idx}));
            store_vector_to_memory(
                builder,
                builder.CreateInBoundsGEP(state_ptr, builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))),
                val);
        });
    } else {
        const auto &new_state = std::get<std::vector<llvm::Value *>>(new_state_var);

        assert(new_state.size() == n_eq);

        for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
            store_vector_to_memory(builder,
                                   builder.CreateInBoundsGEP(state_ptr, builder.getInt32(var_idx * batch_size)),
                                   new_state[var_idx]);
        }
    }

    // Store the timesteps that were used.
    store_vector_to_memory(builder, h_ptr, h);

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass.
    s.optimise();

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
                                   std::uint32_t batch_size, bool high_accuracy, bool compact_mode)
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

    // Load the integration timestep(s).
    auto h = load_vector_from_memory(builder, h_ptr, batch_size);

    // Evaluate the Taylor polynomials, producing the updated state of the system and
    // storing the result into jet_ptr.
    if (high_accuracy) {
        taylor_run_ceval_state_updater<T>(s, jet_ptr, h, n_vars, order, batch_size, high_accuracy, compact_mode);
    } else {
        taylor_run_multihorner_state_updater(s, jet_ptr, h, n_vars, order, batch_size, compact_mode);
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
    taylor_add_state_updater_impl<T>(s, name + "_updater", n_vars, order, batch_size, high_accuracy, compact_mode);

    // Restore the original optimisation level in s.
    od.reset();

    // Run the optimisation pass.
    s.optimise();

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

namespace detail
{

namespace
{

// Implementation of the streaming operator for the scalar integrators.
template <typename T>
std::ostream &taylor_adaptive_stream_impl(std::ostream &os, const taylor_adaptive_impl<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    oss << "Taylor order: " << ta.get_order() << '\n';
    oss << "Dimension   : " << ta.get_dim() << '\n';
    oss << "Time        : " << ta.get_time() << '\n';
    oss << "State       : [";
    for (decltype(ta.get_state().size()) i = 0; i < ta.get_state().size(); ++i) {
        oss << ta.get_state()[i];
        if (i != ta.get_state().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";

    return os << oss.str();
}

// Implementation of the streaming operator for the batch integrators.
template <typename T>
std::ostream &taylor_adaptive_batch_stream_impl(std::ostream &os, const taylor_adaptive_batch_impl<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    oss << "Taylor order: " << ta.get_order() << '\n';
    oss << "Dimension   : " << ta.get_dim() << '\n';
    oss << "Batch size  : " << ta.get_batch_size() << '\n';
    oss << "Time        : [";
    for (decltype(ta.get_time().size()) i = 0; i < ta.get_time().size(); ++i) {
        oss << ta.get_time()[i];
        if (i != ta.get_time().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";
    oss << "State       : [";
    for (decltype(ta.get_state().size()) i = 0; i < ta.get_state().size(); ++i) {
        oss << ta.get_state()[i];
        if (i != ta.get_state().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";

    return os << oss.str();
}

} // namespace

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_impl<double> &ta)
{
    return taylor_adaptive_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_impl<long double> &ta)
{
    return taylor_adaptive_stream_impl(os, ta);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_impl<mppp::real128> &ta)
{
    return taylor_adaptive_stream_impl(os, ta);
}

#endif

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch_impl<double> &ta)
{
    return taylor_adaptive_batch_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch_impl<long double> &ta)
{
    return taylor_adaptive_batch_stream_impl(os, ta);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch_impl<mppp::real128> &ta)
{
    return taylor_adaptive_batch_stream_impl(os, ta);
}

#endif

} // namespace detail

std::ostream &operator<<(std::ostream &os, taylor_outcome oc)
{
    switch (oc) {
        case taylor_outcome::success:
            os << "success";
            break;
        case taylor_outcome::step_limit:
            os << "step_limit";
            break;
        case taylor_outcome::time_limit:
            os << "time_limit";
            break;
        case taylor_outcome::err_nf_state:
            os << "err_nf_state";
            break;
    }

    return os;
}

} // namespace heyoka
