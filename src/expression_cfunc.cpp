// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <deque>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
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

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
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
#include <heyoka/detail/fast_unordered.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
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

std::optional<std::vector<expression>::size_type> decompose(funcptr_map<std::vector<expression>::size_type> &func_map,
                                                            const expression &ex, std::vector<expression> &dc)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        return fptr->decompose(func_map, dc);
    } else {
        return {};
    }
}

// Helper to verify a function decomposition.
void verify_function_dec(const std::vector<expression> &orig, const std::vector<expression> &dc,
                         std::vector<expression>::size_type nvars,
                         // NOTE: this flags establishes if parameters are allowed
                         // in the initial definitions of the u variables.
                         bool allow_pars)
{
    using idx_t = std::vector<expression>::size_type;

    // Cache the number of outputs.
    const auto nouts = orig.size();

    assert(dc.size() >= nouts);

    // The first nvars expressions of u variables
    // must be just variables or possibly parameters.
    for (idx_t i = 0; i < nvars; ++i) {
        assert(std::holds_alternative<variable>(dc[i].value())
               || (allow_pars && std::holds_alternative<param>(dc[i].value())));
    }

    // From nvars to dc.size() - nouts, the expressions
    // must be functions whose arguments
    // are either variables in the u_n form,
    // where n < i, or numbers/params.
    for (auto i = nvars; i < dc.size() - nouts; ++i) {
        std::visit(
            [i](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    for (const auto &arg : v.args()) {
                        if (auto *p_var = std::get_if<variable>(&arg.value())) {
                            assert(p_var->name().rfind("u_", 0) == 0);
                            assert(uname_to_index(p_var->name()) < i);
                        } else if (std::get_if<number>(&arg.value()) == nullptr
                                   && std::get_if<param>(&arg.value()) == nullptr) {
                            assert(false); // LCOV_EXCL_LINE
                        }
                    }
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].value());
    }

    // From dc.size() - nouts to dc.size(), the expressions
    // must be either variables in the u_n form, where n < dc.size() - nouts,
    // or numbers/params.
    for (auto i = dc.size() - nouts; i < dc.size(); ++i) {
        std::visit(
            [&dc, nouts](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    assert(v.name().rfind("u_", 0) == 0);
                    assert(uname_to_index(v.name()) < dc.size() - nouts);
                } else if constexpr (!std::is_same_v<type, number> && !std::is_same_v<type, param>) {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].value());
    }

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of the original variables or other u variables,
    // and store it in subs_map.
    for (idx_t i = 0; i < dc.size() - nouts; ++i) {
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i], subs_map));
    }

    // Reconstruct the function components
    // and compare them to the original ones.
    for (auto i = dc.size() - nouts; i < dc.size(); ++i) {
        assert(subs(dc[i], subs_map) == orig[i - (dc.size() - nouts)]);
    }
}

// Simplify a function decomposition by removing
// common subexpressions.
std::vector<expression> function_decompose_cse(std::vector<expression> &v_ex,
                                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                               std::vector<expression>::size_type nvars,
                                               std::vector<expression>::size_type nouts)
{
    using idx_t = std::vector<expression>::size_type;

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Cache the original size for logging later.
    const auto orig_size = v_ex.size();

    // A function decomposition is supposed
    // to have nvars variables at the beginning,
    // nouts variables at the end and possibly
    // extra variables in the middle.
    assert(v_ex.size() >= nouts + nvars);

    // Init the return value.
    std::vector<expression> retval;

    // expression -> idx map. This will end up containing
    // all the unique expressions from v_ex, and it will
    // map them to their indices in retval (which will
    // in general differ from their indices in v_ex).
    fast_umap<expression, idx_t, std::hash<expression>> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // The first nvars definitions are just renaming
    // of the original variables into u variables.
    for (idx_t i = 0; i < nvars; ++i) {
        retval.push_back(std::move(v_ex[i]));

        // NOTE: the u vars that correspond to the original
        // variables are never simplified,
        // thus map them onto themselves.
        [[maybe_unused]] const auto res = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Handle the u variables which do not correspond to the original variables.
    for (auto i = nvars; i < v_ex.size() - nouts; ++i) {
        auto &ex = v_ex[i];

        // Rename the u variables in ex.
        ex = rename_variables(ex, uvars_rename);

        if (auto it = ex_map.find(ex); it == ex_map.end()) {
            // This is the first occurrence of ex in the
            // decomposition. Add it to retval.
            retval.push_back(ex);

            // Add ex to ex_map, mapping it to
            // the index it corresponds to in retval
            // (let's call it j).
            ex_map.emplace(std::move(ex), retval.size() - 1u);

            // Update uvars_rename. This will ensure that
            // occurrences of the variable 'u_i' in the next
            // elements of v_ex will be renamed to 'u_j'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", retval.size() - 1u));
            assert(res.second);
        } else {
            // ex is redundant. This means
            // that it already appears in retval at index
            // it->second. Don't add anything to retval,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", it->second));
            assert(res.second); // LCOV_EXCL_LINE
        }
    }

    // Handle the definitions of the outputs at the end of the decomposition.
    // We just need to ensure that
    // the u variables in their definitions are renamed with
    // the new indices.
    for (auto i = v_ex.size() - nouts; i < v_ex.size(); ++i) {
        auto &ex = v_ex[i];

        // NOTE: here we expect only vars, numbers or params.
        assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
               || std::holds_alternative<param>(ex.value()));

        ex = rename_variables(ex, uvars_rename);

        retval.push_back(std::move(ex));
    }

    get_logger()->debug("function CSE reduced decomposition size from {} to {}", orig_size, retval.size());
    get_logger()->trace("function CSE runtime: {}", sw);

    return retval;
}

// Perform a topological sort on a graph representation
// of a function decomposition. This can improve performance
// by grouping together operations that can be performed in parallel,
// and it also makes compact mode much more effective by creating
// clusters of subexpressions which can be evaluated in parallel.
// NOTE: the original decomposition dc is already topologically sorted,
// in the sense that the definitions of the u variables are already
// ordered according to dependency. However, because the original decomposition
// comes from a depth-first search, it has the tendency to group together
// expressions which are dependent on each other. By doing another topological
// sort, this time based on breadth-first search, we determine another valid
// sorting in which independent operations tend to be clustered together.
std::vector<expression> function_sort_dc(std::vector<expression> &dc,
                                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                         std::vector<expression>::size_type nvars,
                                         std::vector<expression>::size_type nouts)
{
    // A function decomposition is supposed
    // to have nvars variables at the beginning,
    // nouts variables at the end and possibly
    // extra variables in the middle.
    assert(dc.size() >= nouts + nvars);

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

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

    // Add the nodes corresponding to the original variables.
    for (decltype(nvars) i = 0; i < nvars; ++i) {
        auto v = boost::add_vertex(g);

        // Add a dependency on the root node.
        boost::add_edge(root_v, v, g);
    }

    // Add the rest of the u variables.
    for (decltype(nvars) i = nvars; i < dc.size() - nouts; ++i) {
        auto v = boost::add_vertex(g);

        // Fetch the list of variables in the current expression.
        const auto vars = get_variables(dc[i]);

        if (vars.empty()) {
            // The current expression does not contain
            // any variable: make it depend on the root
            // node. This means that in the topological
            // sort below, the current u var will appear
            // immediately after the original variables.
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

    assert(boost::num_vertices(g) - 1u == dc.size() - nouts);

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
        // NOTE: the sorting is important to ensure that all the original
        // variables are inserted into v_idx in the correct order.
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
    // nouts indices.
    for (decltype(v_idx.size()) i = 0; i < v_idx.size() - 1u; ++i) {
        v_idx[i] = v_idx[i + 1u] - 1u;
    }
    v_idx.resize(boost::numeric_cast<decltype(v_idx.size())>(dc.size()));
    std::iota(v_idx.data() + dc.size() - nouts, v_idx.data() + dc.size(), dc.size() - nouts);

    // Create the remapping dictionary.
    std::unordered_map<std::string, std::string> remap;
    // NOTE: the u vars that correspond to the original
    // variables were inserted into v_idx in the original
    // order, thus they are not re-sorted and they do not
    // need renaming.
    for (decltype(v_idx.size()) i = 0; i < nvars; ++i) {
        assert(v_idx[i] == i);
        [[maybe_unused]] const auto res = remap.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }
    // Establish the remapping for the u variables that are not
    // original variables.
    for (decltype(v_idx.size()) i = nvars; i < v_idx.size() - nouts; ++i) {
        [[maybe_unused]] const auto res = remap.emplace(fmt::format("u_{}", v_idx[i]), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Do the remap for the definitions of the u variables and of the components.
    for (auto *it = dc.data() + nvars; it != dc.data() + dc.size(); ++it) {
        // Remap the expression.
        *it = rename_variables(*it, remap);
    }

    // Reorder the decomposition.
    std::vector<expression> retval;
    retval.reserve(v_idx.size());
    for (auto idx : v_idx) {
        retval.push_back(std::move(dc[idx]));
    }

    get_logger()->trace("function topological sort runtime: {}", sw);

    return retval;
}

} // namespace detail

// Decomposition with automatic deduction of variables.
std::pair<std::vector<expression>, std::vector<expression>::size_type>
function_decompose(const std::vector<expression> &v_ex_)
{
    if (v_ex_.empty()) {
        throw std::invalid_argument("Cannot decompose a function with no outputs");
    }

    const auto vars = get_variables(v_ex_);

    // Cache the number of variables.
    const auto nvars = vars.size();

    // Cache the number of outputs.
    const auto nouts = v_ex_.size();

    // Create the map for renaming the variables to u_i.
    // The renaming will be done in alphabetical order.
    std::unordered_map<std::string, std::string> repl_map;
    {
        decltype(vars.size()) var_idx = 0;
        for (const auto &var : vars) {
            [[maybe_unused]] const auto eres = repl_map.emplace(var, fmt::format("u_{}", var_idx++));
            assert(eres.second);
        }
    }

    // Transform sums into subs.
    auto v_ex = detail::sum_to_sub(v_ex_);

    // Split sums.
    v_ex = detail::split_sums_for_decompose(v_ex);

    // Transform sums into sum_sqs if possible.
    v_ex = detail::sums_to_sum_sqs_for_decompose(v_ex);

    // Transform prods into divs.
    v_ex = detail::prod_to_div_llvm_eval(v_ex);

    // Split prods.
    // NOTE: 8 is the same value as for split_sums_for_decompose().
    v_ex = detail::split_prods_for_decompose(v_ex, 8u);

    // Unfix.
    // NOTE: unfix is the last step, as we want to keep expressions
    // fixed in the previous preprocessing steps.
    v_ex = unfix(v_ex);

#if !defined(NDEBUG)

    // Save copy for checking in debug mode.
    const auto v_ex_verify = v_ex;

#endif

    // Rename the variables in the original function.
    v_ex = rename_variables(v_ex, repl_map);

    // Init the decomposition. It begins with a list
    // of the original variables of the function.
    std::vector<expression> ret;
    ret.reserve(nvars);
    for (const auto &var : vars) {
        ret.emplace_back(var);
    }

    // Prepare the outputs vector.
    std::vector<expression> outs;
    outs.reserve(nouts);

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition on each component of the function.
    detail::funcptr_map<std::vector<expression>::size_type> func_map;
    for (const auto &ex : v_ex) {
        // Decompose the current component.
        if (const auto dres = detail::decompose(func_map, ex, ret)) {
            // NOTE: if the component was decomposed
            // (that is, it is not constant or a single variable),
            // then the output is a u variable.
            // NOTE: all functions are forced to return
            // a non-empty dres
            // in the func API, so the only entities that
            // can return an empty dres are const/params or
            // variables.
            outs.emplace_back(fmt::format("u_{}", *dres));
        } else {
            assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
                   || std::holds_alternative<param>(ex.value()));

            outs.push_back(ex);
        }
    }

    assert(outs.size() == nouts);

    // Append the definitions of the outputs.
    ret.insert(ret.end(), outs.begin(), outs.end());

    detail::get_logger()->trace("function decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    detail::verify_function_dec(v_ex_verify, ret, nvars);

#endif

    // Simplify the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    ret = detail::function_decompose_cse(ret, nvars, nouts);

#if !defined(NDEBUG)

    // Verify the simplified decomposition.
    detail::verify_function_dec(v_ex_verify, ret, nvars);

#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    ret = detail::function_sort_dc(ret, nvars, nouts);

#if !defined(NDEBUG)

    // Verify the reordered decomposition.
    detail::verify_function_dec(v_ex_verify, ret, nvars);

#endif

    // NOTE: static_cast is fine, as we know that ret contains at least nvars elements.
    return std::make_pair(std::move(ret), static_cast<std::vector<expression>::size_type>(nvars));
}

// Function decomposition from with explicit list of input variables.
std::vector<expression> function_decompose(const std::vector<expression> &v_ex_, const std::vector<expression> &vars)
{
    if (v_ex_.empty()) {
        throw std::invalid_argument("Cannot decompose a function with no outputs");
    }

    // Sanity check vars. We need to ensure that:
    // - all the expressions in vars are variables
    //   and there are no duplicates,
    // - all the variables appearing in v_ex_
    //   are present in vars.
    // Note that vars is allowed to contain extra variables
    // (that is, variables which are not present in v_ex_).

    // A set to check for duplicates in vars.
    std::unordered_set<std::string> var_set;
    // This set will contain all the variables in v_ex_.
    std::unordered_set<std::string> v_ex_vars;

    for (const auto &ex : vars) {
        if (const auto *var_ptr = std::get_if<variable>(&ex.value())) {
            // Check if this is a duplicate variable.
            if (auto res = var_set.emplace(var_ptr->name()); !res.second) {
                // Duplicate, error out.
                throw std::invalid_argument(fmt::format("Error in the decomposition of a function: the variable '{}' "
                                                        "appears in the user-provided list of variables twice",
                                                        var_ptr->name()));
            }
        } else {
            throw std::invalid_argument(fmt::format("Error in the decomposition of a function: the "
                                                    "user-provided list of variables contains the expression '{}', "
                                                    "which is not a variable",
                                                    ex));
        }
    }

    // Build v_ex_vars.
    const auto detected_vars = get_variables(v_ex_);
    v_ex_vars.insert(detected_vars.begin(), detected_vars.end());

    // Check that all variables in v_ex_vars appear in var_set.
    for (const auto &var : v_ex_vars) {
        if (var_set.find(var) == var_set.end()) {
            throw std::invalid_argument(
                fmt::format("Error in the decomposition of a function: the variable '{}' "
                            "appears in the function but not in the user-provided list of variables",
                            var));
        }
    }

    // Cache the number of variables.
    const auto nvars = vars.size();

    // Cache the number of outputs.
    const auto nouts = v_ex_.size();

    // Create the map for renaming the variables to u_i.
    // The renaming will be done following the order of vars.
    std::unordered_map<std::string, std::string> repl_map;
    for (decltype(vars.size()) i = 0; i < nvars; ++i) {
        [[maybe_unused]] const auto eres
            = repl_map.emplace(std::get<variable>(vars[i].value()).name(), fmt::format("u_{}", i));
        assert(eres.second);
    }

    // Transform sums into subs.
    auto v_ex = detail::sum_to_sub(v_ex_);

    // Split sums.
    v_ex = detail::split_sums_for_decompose(v_ex);

    // Transform sums into sum_sqs if possible.
    v_ex = detail::sums_to_sum_sqs_for_decompose(v_ex);

    // Transform prods into divs.
    v_ex = detail::prod_to_div_llvm_eval(v_ex);

    // Split prods.
    // NOTE: 8 is the same value as for split_sums_for_decompose().
    v_ex = detail::split_prods_for_decompose(v_ex, 8u);

    // Unfix.
    // NOTE: unfix is the last step, as we want to keep expressions
    // fixed in the previous preprocessing steps.
    v_ex = unfix(v_ex);

#if !defined(NDEBUG)

    // Save copy for checking in debug mode.
    const auto v_ex_verify = v_ex;

#endif

    // Rename the variables in the original function.
    v_ex = rename_variables(v_ex, repl_map);

    // Init the decomposition. It begins with a list
    // of the original variables of the function.
    std::vector<expression> ret;
    ret.reserve(nvars);
    for (const auto &var : vars) {
        ret.push_back(var);
    }

    // Prepare the outputs vector.
    std::vector<expression> outs;
    outs.reserve(nouts);

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition on each component of the function.
    detail::funcptr_map<std::vector<expression>::size_type> func_map;
    for (const auto &ex : v_ex) {
        // Decompose the current component.
        if (const auto dres = detail::decompose(func_map, ex, ret)) {
            // NOTE: if the component was decomposed
            // (that is, it is not constant or a single variable),
            // then the output is a u variable.
            // NOTE: all functions are forced to return
            // a non-empty dres
            // in the func API, so the only entities that
            // can return an empty dres are const/params or
            // variables.
            outs.emplace_back(fmt::format("u_{}", *dres));
        } else {
            assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
                   || std::holds_alternative<param>(ex.value()));

            outs.push_back(ex);
        }
    }

    assert(outs.size() == nouts);

    // Append the definitions of the outputs.
    ret.insert(ret.end(), outs.begin(), outs.end());

    detail::get_logger()->trace("function decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)

    // Verify the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    detail::verify_function_dec(v_ex_verify, ret, nvars);

#endif

    // Simplify the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    ret = detail::function_decompose_cse(ret, nvars, nouts);

#if !defined(NDEBUG)

    // Verify the simplified decomposition.
    detail::verify_function_dec(v_ex_verify, ret, nvars);

#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: nvars is implicitly converted to std::vector<expression>::size_type here.
    // This is fine, as the decomposition must contain at least nvars items.
    ret = detail::function_sort_dc(ret, nvars, nouts);

#if !defined(NDEBUG)

    // Verify the reordered decomposition.
    detail::verify_function_dec(v_ex_verify, ret, nvars);

#endif

    return ret;
}

namespace detail
{

namespace
{

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void add_cfunc_nc_mode(llvm_state &s, llvm::Type *fp_t, llvm::Value *out_ptr, llvm::Value *in_ptr, llvm::Value *par_ptr,
                       llvm::Value *time_ptr, llvm::Value *stride, const std::vector<expression> &dc,
                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                       std::uint32_t nvars, std::uint32_t nuvars, std::uint32_t batch_size, bool high_accuracy)
{
    auto &builder = s.builder();

    // The array containing the evaluation of the decomposition.
    std::vector<llvm::Value *> eval_arr;

    // Fetch the type for external loading.
    auto *ext_fp_t = llvm_ext_type(fp_t);

    // Init it by loading the input values from in_ptr.
    for (std::uint32_t i = 0; i < nvars; ++i) {
        auto *ptr
            = builder.CreateInBoundsGEP(ext_fp_t, in_ptr, builder.CreateMul(stride, to_size_t(s, builder.getInt32(i))));
        eval_arr.push_back(ext_load_vector_from_memory(s, fp_t, ptr, batch_size));
    }

    // Evaluate the elementary subexpressions in the decomposition.
    for (std::uint32_t i = nvars; i < nuvars; ++i) {
        assert(std::holds_alternative<func>(dc[i].value()));

        eval_arr.push_back(std::get<func>(dc[i].value())
                               .llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size, high_accuracy));
    }

    // Write the outputs.
    for (decltype(dc.size()) i = nuvars; i < dc.size(); ++i) {
        // Index of the current output.
        const auto out_idx = static_cast<std::uint32_t>(i - nuvars);

        // Compute the pointer to write to.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr,
                                              builder.CreateMul(stride, to_size_t(s, builder.getInt32(out_idx))));

        std::visit(
            [&](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    // Fetch the index of the variable.
                    const auto u_idx = uname_to_index(v.name());
                    assert(u_idx < eval_arr.size());

                    // Fetch the corresponding value from eval_arr and store it.
                    ext_store_vector_to_memory(s, ptr, eval_arr[u_idx]);
                } else if constexpr (std::is_same_v<type, number>) {
                    // Codegen the number and store it.
                    ext_store_vector_to_memory(s, ptr, vector_splat(builder, llvm_codegen(s, fp_t, v), batch_size));
                } else if constexpr (std::is_same_v<type, param>) {
                    // Codegen the parameter and store it.
                    ext_store_vector_to_memory(s, ptr, cfunc_nc_param_codegen(s, v, batch_size, fp_t, par_ptr, stride));
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].value());
    }
}

// Function to split the central part of a function decomposition (i.e., the definitions of the u variables
// that do not represent original variables) into parallelisable segments. Within a segment,
// the definition of a u variable does not depend on any u variable defined within that segment.
// NOTE: the segments in the return value will contain shallow copies of the
// expressions in dc.
std::vector<std::vector<expression>> function_segment_dc(const std::vector<expression> &dc,
                                                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                         std::uint32_t nvars, std::uint32_t nuvars)
{
    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Helper that takes in input the definition ex of a u variable, and returns
    // in output the list of indices of the u variables on which ex depends.
    auto udef_args_indices = [](const expression &ex) -> std::vector<std::uint32_t> {
        return std::visit(
            [](const auto &v) -> std::vector<std::uint32_t> {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    std::vector<std::uint32_t> retval;

                    for (const auto &arg : v.args()) {
                        std::visit(
                            [&retval](const auto &x) {
                                using tp = uncvref_t<decltype(x)>;

                                if constexpr (std::is_same_v<tp, variable>) {
                                    retval.push_back(uname_to_index(x.name()));
                                } else if constexpr (!std::is_same_v<tp, number> && !std::is_same_v<tp, param>) {
                                    // LCOV_EXCL_START
                                    throw std::invalid_argument(
                                        "Invalid argument encountered in an element of a function decomposition: the "
                                        "argument is not a variable or a number/param");
                                    // LCOV_EXCL_STOP
                                }
                            },
                            arg.value());
                    }

                    return retval;
                } else {
                    // LCOV_EXCL_START
                    throw std::invalid_argument("Invalid expression encountered in a function decomposition: the "
                                                "expression is not a function");
                    // LCOV_EXCL_STOP
                }
            },
            ex.value());
    };

    // Init the return value.
    std::vector<std::vector<expression>> s_dc;

    // cur_limit_idx is initially the index of the first
    // u variable which is not an original variable.
    auto cur_limit_idx = nvars;
    for (std::uint32_t i = nvars; i < nuvars; ++i) {
        // NOTE: at the very first iteration of this for loop,
        // no block has been created yet. Do it now.
        if (i == nvars) {
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
            // less than counter + nvars (which is the starting
            // index of the block).
            const auto u_indices = udef_args_indices(ex);
            assert(std::all_of(u_indices.begin(), u_indices.end(),
                               [idx_limit = counter + nvars](auto idx) { return idx < idx_limit; }));
        }

        // Update the counter.
        counter += s.size();
    }

    assert(counter == nuvars - nvars);
#endif

    get_logger()->debug("cfunc decomposition N of segments: {}", s_dc.size());
    get_logger()->trace("cfunc decomposition segment runtime: {}", sw);

    return s_dc;
}

auto cfunc_build_function_maps(llvm_state &s, llvm::Type *fp_t, const std::vector<std::vector<expression>> &s_dc,
                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                               std::uint32_t nvars, std::uint32_t batch_size, bool high_accuracy)
{
    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Init the return value.
    // NOTE: use maps with name-based comparison for the functions. This ensures that the order in which these
    // functions are invoked is always the same. If we used directly pointer
    // comparisons instead, the order could vary across different executions and different platforms. The name
    // mangling we do when creating the function names should ensure that there are no possible name collisions.
    std::vector<
        std::map<llvm::Function *, std::pair<std::uint32_t, std::vector<std::function<llvm::Value *(llvm::Value *)>>>,
                 llvm_func_name_compare>>
        retval;

    // Variable to keep track of the u variable
    // on whose definition we are operating.
    auto cur_u_idx = nvars;
    for (const auto &seg : s_dc) {
        // This structure maps an LLVM function to sets of arguments
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
            // Get the evaluation function.
            auto *func = std::get<heyoka::func>(ex.value()).llvm_c_eval_func(s, fp_t, batch_size, high_accuracy);

            // Insert the function into tmp_map.
            const auto [it, is_new_func] = tmp_map.try_emplace(func);

            assert(is_new_func || !it->second.empty()); // LCOV_EXCL_LINE

            // Convert the variables/constants in the current dc
            // element into a set of indices/constants.
            const auto c_args = udef_to_variants(ex, {});

            // LCOV_EXCL_START
            if (!is_new_func && it->second.back().size() - 1u != c_args.size()) {
                throw std::invalid_argument(
                    fmt::format("Inconsistent arity detected in a compiled function in compact "
                                "mode: the same function is being called with both {} and {} arguments",
                                it->second.back().size() - 1u, c_args.size()));
            }
            // LCOV_EXCL_STOP

            // Add the new set of arguments.
            it->second.emplace_back();
            // Add the idx of the u variable.
            it->second.back().emplace_back(cur_u_idx);
            // Add the actual function arguments.
            it->second.back().insert(it->second.back().end(), c_args.begin(), c_args.end());

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

        // Add a new entry in retval for the current segment.
        retval.emplace_back();
        auto &a_map = retval.back();

        for (const auto &[func, vv] : tmp_map_transpose) {
            // NOTE: vv.size() is now the number of arguments. We know it cannot
            // be zero because the evaluation functions
            // in compact mode always have at least 1 argument (i.e., the index
            // of the u variable which is being evaluated).
            assert(!vv.empty()); // LCOV_EXCL_LINE

            // Add the function.
            const auto [it, ins_status] = a_map.try_emplace(func);
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
    }

    get_logger()->trace("cfunc build function maps runtime: {}", sw);

    // LCOV_EXCL_START
    // Log a breakdown of the return value in trace mode.
    if (get_logger()->should_log(spdlog::level::trace)) {
        std::vector<std::vector<std::uint32_t>> fm_bd;

        for (const auto &m : retval) {
            fm_bd.emplace_back();

            for (const auto &p : m) {
                fm_bd.back().push_back(p.second.first);
            }
        }

        get_logger()->trace("cfunc function maps breakdown: {}", fm_bd);
    }
    // LCOV_EXCL_STOP

    return retval;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void cfunc_c_store_eval(llvm_state &s, llvm::Type *fp_vec_t, llvm::Value *eval_arr, llvm::Value *idx, llvm::Value *val)
{
    auto &builder = s.builder();

    auto *ptr = builder.CreateInBoundsGEP(fp_vec_t, eval_arr, idx);

    builder.CreateStore(val, ptr);
}

} // namespace

llvm::Value *cfunc_c_load_eval(llvm_state &s, llvm::Type *fp_vec_t, llvm::Value *eval_arr, llvm::Value *idx)
{
    auto &builder = s.builder();

    auto *ptr = builder.CreateInBoundsGEP(fp_vec_t, eval_arr, idx);

    return builder.CreateLoad(fp_vec_t, ptr);
}

namespace
{

// Helper to construct the global arrays needed for the evaluation of a compiled
// function in compact mode. The first part of the
// return value is a set of 6 arrays:
// - the indices of the outputs which are u variables, paired to
// - the indices of said u variables, and
// - the indices of the outputs which are constants, paired to
// - the values of said constants, and
// - the indices of the outputs which are params, paired to
// - the indices of the params.
// The second part of the return value is a boolean flag that will be true if
// all outputs are u variables, false otherwise.
std::pair<std::array<llvm::GlobalVariable *, 6>, bool>
cfunc_c_make_output_globals(llvm_state &s, llvm::Type *fp_t, const std::vector<expression> &dc, std::uint32_t nuvars)
{
    auto &context = s.context();
    auto &builder = s.builder();
    auto &md = s.module();

    // Build iteratively the output values as vectors of constants.
    std::vector<llvm::Constant *> var_indices, vars, num_indices, nums, par_indices, pars;

    // Keep track of how many outputs are u variables.
    std::uint32_t n_out_vars = 0;

    // NOTE: the definitions of the outputs are at the end of the decomposition.
    for (auto i = nuvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
        std::visit(
            [&](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    ++n_out_vars;
                    // NOTE: remove from i the nuvars offset to get the
                    // true index of the output.
                    var_indices.push_back(builder.getInt32(i - nuvars));
                    vars.push_back(builder.getInt32(uname_to_index(v.name())));
                } else if constexpr (std::is_same_v<type, number>) {
                    num_indices.push_back(builder.getInt32(i - nuvars));
                    nums.push_back(llvm::cast<llvm::Constant>(llvm_codegen(s, fp_t, v)));
                } else if constexpr (std::is_same_v<type, param>) {
                    par_indices.push_back(builder.getInt32(i - nuvars));
                    pars.push_back(builder.getInt32(v.idx()));
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].value());
    }

    // Flag to signal that all outputs are u variables.
    assert(dc.size() >= nuvars); // LCOV_EXCL_LINE
    const auto all_out_vars = (n_out_vars == (dc.size() - nuvars));

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

    return std::pair{std::array{g_var_indices, g_vars, g_num_indices, g_nums, g_par_indices, g_pars}, all_out_vars};
}

// Small helper to compute the size of a global array.
std::uint32_t cfunc_c_gl_arr_size(llvm::Value *v)
{
    return boost::numeric_cast<std::uint32_t>(
        llvm::cast<llvm::ArrayType>(llvm::cast<llvm::GlobalVariable>(v)->getValueType())->getNumElements());
}

// Helper to write the outputs of a compiled function in compact mode.
// cout_gl is the return value of cfunc_c_make_output_globals(), which contains
// the indices/constants necessary for the computation.
void cfunc_c_write_outputs(llvm_state &s, llvm::Type *fp_scal_t, llvm::Value *out_ptr,
                           const std::pair<std::array<llvm::GlobalVariable *, 6>, bool> &cout_gl,
                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                           llvm::Value *eval_arr, llvm::Value *par_ptr, llvm::Value *stride, std::uint32_t batch_size)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    // Fetch the global arrays and
    // the all_out_vars flag.
    const auto &out_gl = cout_gl.first;
    const auto all_out_vars = cout_gl.second;

    auto &builder = s.builder();

    // Recover the number of outputs which are
    // u variables, numbers and params.
    const auto n_vars = cfunc_c_gl_arr_size(out_gl[0]);
    const auto n_nums = cfunc_c_gl_arr_size(out_gl[2]);
    const auto n_pars = cfunc_c_gl_arr_size(out_gl[4]);

    // Fetch the type for external loading.
    auto *ext_fp_t = llvm_ext_type(fp_scal_t);

    // Fetch the vector type.
    auto *fp_vec_t = make_vector_type(fp_scal_t, batch_size);

    // Handle the u variable outputs.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_vars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the output.
        // NOTE: if all outputs are u variables, there's
        // no need to lookup the index in the global array (which will just contain
        // a range).
        auto *out_idx = all_out_vars
                            ? cur_idx
                            : builder.CreateLoad(builder.getInt32Ty(),
                                                 builder.CreateInBoundsGEP(out_gl[0]->getValueType(), out_gl[0],
                                                                           {builder.getInt32(0), cur_idx}));

        // Fetch the index of the u variable.
        auto *u_idx
            = builder.CreateLoad(builder.getInt32Ty(), builder.CreateInBoundsGEP(out_gl[1]->getValueType(), out_gl[1],
                                                                                 {builder.getInt32(0), cur_idx}));

        // Fetch from eval_arr the value of the u variable u_idx.
        auto *ret = cfunc_c_load_eval(s, fp_vec_t, eval_arr, u_idx);

        // Compute the pointer into out_ptr.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.CreateMul(stride, to_size_t(s, out_idx)));

        // Store ret.
        ext_store_vector_to_memory(s, ptr, ret);
    });

    // Handle the number definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_nums), [&](llvm::Value *cur_idx) {
        // Fetch the index of the output.
        auto *out_idx
            = builder.CreateLoad(builder.getInt32Ty(), builder.CreateInBoundsGEP(out_gl[2]->getValueType(), out_gl[2],
                                                                                 {builder.getInt32(0), cur_idx}));

        // Fetch the constant.
        auto *num = builder.CreateLoad(
            fp_scal_t, builder.CreateInBoundsGEP(out_gl[3]->getValueType(), out_gl[3], {builder.getInt32(0), cur_idx}));

        // Splat it out.
        auto *ret = vector_splat(builder, num, batch_size);

        // Compute the pointer into out_ptr.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.CreateMul(stride, to_size_t(s, out_idx)));

        // Store ret.
        ext_store_vector_to_memory(s, ptr, ret);
    });

    // Handle the param definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_pars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the output.
        auto *out_idx
            = builder.CreateLoad(builder.getInt32Ty(), builder.CreateInBoundsGEP(out_gl[4]->getValueType(), out_gl[4],
                                                                                 {builder.getInt32(0), cur_idx}));

        // Fetch the index of the param.
        auto *par_idx
            = builder.CreateLoad(builder.getInt32Ty(), builder.CreateInBoundsGEP(out_gl[5]->getValueType(), out_gl[5],
                                                                                 {builder.getInt32(0), cur_idx}));

        // Load the parameter value from the array.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, par_ptr, builder.CreateMul(stride, to_size_t(s, par_idx)));
        auto *ret = ext_load_vector_from_memory(s, fp_scal_t, ptr, batch_size);

        // Compute the pointer into out_ptr.
        ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.CreateMul(stride, to_size_t(s, out_idx)));

        // Store ret.
        ext_store_vector_to_memory(s, ptr, ret);
    });
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void add_cfunc_c_mode(llvm_state &s, llvm::Type *fp_type, llvm::Value *out_ptr, llvm::Value *in_ptr,
                      llvm::Value *par_ptr, llvm::Value *time_ptr, llvm::Value *stride,
                      const std::vector<expression> &dc, std::uint32_t nvars,
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                      std::uint32_t nuvars, std::uint32_t batch_size, bool high_accuracy)
{
    auto &builder = s.builder();
    auto &md = s.module();

    // Fetch the type for external loading.
    auto *ext_fp_t = llvm_ext_type(fp_type);

    // Split dc into segments.
    const auto s_dc = function_segment_dc(dc, nvars, nuvars);

    // Generate the function maps.
    const auto f_maps = cfunc_build_function_maps(s, fp_type, s_dc, nvars, batch_size, high_accuracy);

    // Log the runtime of IR construction in trace mode.
    spdlog::stopwatch sw;

    // Generate the global arrays used to write the outputs at the
    // end of the computation.
    const auto cout_gl = cfunc_c_make_output_globals(s, fp_type, dc, nuvars);

    // Prepare the array that will contain the evaluation of all the
    // elementary subexpressions.
    // NOTE: the array size is specified as a 64-bit integer in the
    // LLVM API.
    // NOTE: fp_type is the original, scalar floating-point type.
    // It will be turned into a vector type (if necessary) by
    // make_vector_type() below.
    auto *fp_vec_type = make_vector_type(fp_type, batch_size);
    auto *array_type = llvm::ArrayType::get(fp_vec_type, nuvars);

    // Make the global array and fetch a pointer to its first element.
    // NOTE: we use a global array rather than a local one here because
    // its size can grow quite large, which can lead to stack overflow issues.
    // This has of course consequences in terms of thread safety, which
    // we will have to document.
    auto *eval_arr_gvar = make_global_zero_array(md, array_type);
    auto *eval_arr = builder.CreateInBoundsGEP(array_type, eval_arr_gvar, {builder.getInt32(0), builder.getInt32(0)});

    // Compute the size in bytes of eval_arr.
    const auto eval_arr_size = get_size(md, array_type);

    // NOTE: eval_arr is used as temporary storage for the current function,
    // but it is declared as a global variable in order to avoid stack overflow.
    // This creates a situation in which LLVM cannot elide stores into eval_arr
    // (even if it figures out a way to avoid storing intermediate results into
    // eval_arr) because LLVM must assume that some other function may
    // use these stored values later. Thus, we declare via an intrinsic that the
    // lifetime of eval_arr begins here and ends at the end of the function,
    // so that LLVM can assume that any value stored in it cannot be possibly
    // used outside this function.
    builder.CreateLifetimeStart(eval_arr, builder.getInt64(eval_arr_size));

    // Copy over the values of the variables.
    // NOTE: overflow checking is already done in the parent function.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(nvars), [&](llvm::Value *cur_var_idx) {
        // Fetch the pointer from in_ptr.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, in_ptr, builder.CreateMul(stride, to_size_t(s, cur_var_idx)));

        // Load as a vector.
        auto *vec = ext_load_vector_from_memory(s, fp_type, ptr, batch_size);

        // Store into eval_arr.
        cfunc_c_store_eval(s, fp_vec_type, eval_arr, cur_var_idx, vec);
    });

    // Helper to evaluate a block.
    // func is the LLVM function for evaluation in the block,
    // ncalls the number of times it must be called and gens the generators for the
    // function arguments.
    auto block_eval = [&](llvm::Function *func, const auto &ncalls, const auto &gens) {
        // LCOV_EXCL_START
        assert(ncalls > 0u);
        assert(!gens.empty());
        assert(std::all_of(gens.begin(), gens.end(), [](const auto &f) { return static_cast<bool>(f); }));
        // LCOV_EXCL_STOP

        // Loop over the number of calls.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(ncalls), [&](llvm::Value *cur_call_idx) {
            // Create the u variable index from the first generator.
            auto u_idx = gens[0](cur_call_idx);

            // Initialise the vector of arguments with which func must be called. The following
            // initial arguments are always present:
            // - eval array,
            // - pointer to the param values,
            // - pointer to the time value(s),
            // - stride.
            std::vector<llvm::Value *> args{u_idx, eval_arr, par_ptr, time_ptr, stride};

            // Create the other arguments via the generators.
            for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                args.push_back(gens[i](cur_call_idx));
            }

            // Evaluate and store the result.
            cfunc_c_store_eval(s, fp_vec_type, eval_arr, u_idx, builder.CreateCall(func, args));
        });
    };

    // Evaluate all elementary subexpressions by iterating
    // over all segments and blocks.
    for (const auto &map : f_maps) {
        for (const auto &p : map) {
            block_eval(p.first, p.second.first, p.second.second);
        }
    }

    // Write the results to the output pointer.
    cfunc_c_write_outputs(s, fp_type, out_ptr, cout_gl, eval_arr, par_ptr, stride, batch_size);

    // End the lifetime of eval_arr.
    builder.CreateLifetimeEnd(eval_arr, builder.getInt64(eval_arr_size));

    get_logger()->trace("cfunc IR creation compact mode runtime: {}", sw);
}

// NOTE: add_cfunc() will add two functions, one called 'name'
// and the other called 'name' + '.strided'. The first function
// indexes into the input/output/par buffers contiguously (that it,
// it assumes the input/output/par scalar/vector values are stored one
// after the other without "holes" between them).
// The second function has an extra trailing argument, the stride
// value, which indicates the distance between consecutive
// input/output/par values in the buffers. The stride is measured in the number
// of *scalar* values between input/output/par values.
// For instance, for a batch size of 1 and a stride value of 3,
// the input scalar values will be read from indices 0, 3, 6, 9, ...
// in the input array. For a batch size of 2 and a stride value of 3,
// the input vector values (of size 2) will be read from indices
// [0, 1], [3, 4], [6, 7], [9, 10], ... in the input array.
template <typename T, typename F>
auto add_cfunc_impl(llvm_state &s, const std::string &name, const F &fn, std::uint32_t batch_size, bool high_accuracy,
                    bool compact_mode, bool parallel_mode, [[maybe_unused]] long long prec)
{
    if (s.is_compiled()) {
        throw std::invalid_argument("A compiled function cannot be added to an llvm_state after compilation");
    }

    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size of a compiled function cannot be zero");
    }

    if (parallel_mode && !compact_mode) {
        throw std::invalid_argument("Parallel mode can only be enabled in conjunction with compact mode");
    }

    if (parallel_mode) {
        throw std::invalid_argument("Parallel mode has not been implemented yet");
    }

#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        const auto sprec = boost::numeric_cast<mpfr_prec_t>(prec);

        if (sprec < mppp::real_prec_min() || sprec > mppp::real_prec_max()) {
            throw std::invalid_argument(fmt::format("An invalid precision value of {} was passed to add_cfunc() (the "
                                                    "value must be in the [{}, {}] range)",
                                                    sprec, mppp::real_prec_min(), mppp::real_prec_max()));
        }
    }

#endif

    // Decompose the function and cache the number of vars and outputs.
    // NOTE: nvars and nouts are already safely cast to 32-bit integers.
    auto [dc, nvars, nouts] = [&fn]() {
        if constexpr (std::is_same_v<F, std::vector<expression>>) {
            auto dec_res = function_decompose(fn);

            return std::make_tuple(std::move(dec_res.first), boost::numeric_cast<std::uint32_t>(dec_res.second),
                                   boost::numeric_cast<std::uint32_t>(fn.size()));
        } else {
            return std::make_tuple(function_decompose(fn.first, fn.second),
                                   boost::numeric_cast<std::uint32_t>(fn.second.size()),
                                   boost::numeric_cast<std::uint32_t>(fn.first.size()));
        }
    }();

    // Determine the number of u variables.
    // NOTE: this is also safely cast to a 32-bit integer.
    assert(dc.size() >= nouts); // LCOV_EXCL_LINE
    const auto nuvars = boost::numeric_cast<std::uint32_t>(dc.size() - nouts);

    // NOTE: due to the presence of the stride argument, we will be always
    // indexing into the input, output and parameter arrays via size_t.
    // Hence, we don't need here the same overflow checking we need to perform
    // in the integrators, as we assume that any array allocated from C++
    // can't have a size larger than the max size_t.

    auto &builder = s.builder();
    auto &context = s.context();
    auto &md = s.module();

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // Prepare the arguments:
    //
    // - a write-only float pointer to the outputs,
    // - a const float pointer to the inputs,
    // - a const float pointer to the pars,
    // - a const float pointer to the time value(s),
    // - the stride.
    //
    // The pointer arguments cannot overlap.
    auto *fp_t = [&]() {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::is_same_v<T, mppp::real>) {
            return llvm_type_like(s, mppp::real{mppp::real_kind::zero, static_cast<mpfr_prec_t>(prec)});
        } else {
#endif
            return to_llvm_type<T>(context);
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    }();
    auto *ext_fp_t = llvm_ext_type(fp_t);
    std::vector<llvm::Type *> fargs(4, llvm::PointerType::getUnqual(ext_fp_t));
    fargs.push_back(to_llvm_type<std::size_t>(context));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Append ".strided" to the function name.
    const auto sname = name + ".strided";
    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, sname, &md);
    // NOTE: a cfunc cannot call itself recursively.
    f->addFnAttr(llvm::Attribute::NoRecurse);

    // Set the names/attributes of the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *in_ptr = out_ptr + 1;
    in_ptr->setName("in_ptr");
    in_ptr->addAttr(llvm::Attribute::NoCapture);
    in_ptr->addAttr(llvm::Attribute::NoAlias);
    in_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *par_ptr = out_ptr + 2;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr = out_ptr + 3;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *stride = out_ptr + 4;
    stride->setName("stride");

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    if (compact_mode) {
        add_cfunc_c_mode(s, fp_t, out_ptr, in_ptr, par_ptr, time_ptr, stride, dc, nvars, nuvars, batch_size,
                         high_accuracy);
    } else {
        add_cfunc_nc_mode(s, fp_t, out_ptr, in_ptr, par_ptr, time_ptr, stride, dc, nvars, nuvars, batch_size,
                          high_accuracy);
    }

    // Finish off the function.
    builder.CreateRetVoid();

    // Verify it.
    s.verify_function(f);

    // Store the strided function pointer for use later.
    auto *f_strided = f;

    // Build the version of the function with stride == 1.
    fargs.pop_back();
    ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &md);

    // Set the names/attributes of the function arguments.
    out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    in_ptr = out_ptr + 1;
    in_ptr->setName("in_ptr");
    in_ptr->addAttr(llvm::Attribute::NoCapture);
    in_ptr->addAttr(llvm::Attribute::NoAlias);
    in_ptr->addAttr(llvm::Attribute::ReadOnly);

    par_ptr = out_ptr + 2;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    time_ptr = out_ptr + 3;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Invoke the strided function with stride == batch_size.
    auto *fcall = builder.CreateCall(f_strided,
                                     {out_ptr, in_ptr, par_ptr, time_ptr, to_size_t(s, builder.getInt32(batch_size))});
    // NOTE: forcibly inline the function call. This will increase
    // compile time, but we want to make sure that the non-strided
    // version of the compiled function is optimised as much as possible,
    // so we accept the tradeoff.
#if LLVM_VERSION_MAJOR >= 14
    fcall->addFnAttr(llvm::Attribute::AlwaysInline);
#else
    {
        auto attrs = fcall->getAttributes();
        attrs = attrs.addAttribute(context, llvm::AttributeList::FunctionIndex, llvm::Attribute::AlwaysInline);
        fcall->setAttributes(attrs);
    }
#endif

    // Finish off the function.
    builder.CreateRetVoid();

    // Verify it.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return dc;
}

} // namespace

template <typename T>
std::vector<expression> add_cfunc(llvm_state &s, const std::string &name, const std::vector<expression> &v_ex,
                                  std::uint32_t batch_size, bool high_accuracy, bool compact_mode, bool parallel_mode,
                                  long long prec)
{
    return detail::add_cfunc_impl<T>(s, name, v_ex, batch_size, high_accuracy, compact_mode, parallel_mode, prec);
}

template <typename T>
std::vector<expression> add_cfunc(llvm_state &s, const std::string &name, const std::vector<expression> &v_ex,
                                  const std::vector<expression> &vars, std::uint32_t batch_size, bool high_accuracy,
                                  bool compact_mode, bool parallel_mode, long long prec)
{
    return detail::add_cfunc_impl<T>(s, name, std::make_pair(std::cref(v_ex), std::cref(vars)), batch_size,
                                     high_accuracy, compact_mode, parallel_mode, prec);
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<double>(llvm_state &, const std::string &,
                                                                     const std::vector<expression> &, std::uint32_t,
                                                                     bool, bool, bool, long long);
template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<double>(llvm_state &, const std::string &,
                                                                     const std::vector<expression> &,
                                                                     const std::vector<expression> &, std::uint32_t,
                                                                     bool, bool, bool, long long);

template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<long double>(llvm_state &, const std::string &,
                                                                          const std::vector<expression> &,
                                                                          std::uint32_t, bool, bool, bool, long long);
template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<long double>(llvm_state &, const std::string &,
                                                                          const std::vector<expression> &,
                                                                          const std::vector<expression> &,
                                                                          std::uint32_t, bool, bool, bool, long long);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<mppp::real128>(llvm_state &, const std::string &,
                                                                            const std::vector<expression> &,
                                                                            std::uint32_t, bool, bool, bool, long long);
template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<mppp::real128>(llvm_state &, const std::string &,
                                                                            const std::vector<expression> &,
                                                                            const std::vector<expression> &,
                                                                            std::uint32_t, bool, bool, bool, long long);

#endif

#if defined(HEYOKA_HAVE_REAL)

template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<mppp::real>(llvm_state &, const std::string &,
                                                                         const std::vector<expression> &, std::uint32_t,
                                                                         bool, bool, bool, long long);
template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<mppp::real>(llvm_state &, const std::string &,
                                                                         const std::vector<expression> &,
                                                                         const std::vector<expression> &, std::uint32_t,
                                                                         bool, bool, bool, long long);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
