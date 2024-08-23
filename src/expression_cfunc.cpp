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
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
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
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <oneapi/tbb/parallel_invoke.h>

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
#include <heyoka/detail/debug.hpp>
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

    // NOTE: the next check can be quite heavy, skip it if requested.
    if (!edb_enabled()) {
        return;
    }

    // For each u variable, expand its definition
    // in terms of the original variables or other u variables,
    // and store it in subs_map.
    std::unordered_map<std::string, expression> subs_map;
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
    auto *ext_fp_t = make_external_llvm_type(fp_t);

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
            []<typename T>(const T &v) -> std::vector<std::uint32_t> {
                if constexpr (std::is_same_v<T, func>) {
                    std::vector<std::uint32_t> retval;

                    for (const auto &arg : v.args()) {
                        std::visit(
                            [&retval]<typename U>(const U &x) {
                                if constexpr (std::is_same_v<U, variable>) {
                                    retval.push_back(uname_to_index(x.name()));
                                } else if constexpr (!std::is_same_v<U, number> && !std::is_same_v<U, param>) {
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
        // no segment has been created yet. Do it now.
        if (i == nvars) {
            assert(s_dc.empty());
            s_dc.emplace_back();
        } else {
            assert(!s_dc.empty());
        }

        const auto &ex = dc[i];

        // Determine the u indices on which ex depends.
        const auto u_indices = udef_args_indices(ex);

        if (std::ranges::any_of(u_indices, [cur_limit_idx](auto idx) { return idx >= cur_limit_idx; })) {
            // The current expression depends on one or more variables
            // within the current segment. Start a new segment and
            // update cur_limit_idx with the start index of the new segment.
            s_dc.emplace_back();
            cur_limit_idx = i;
        }

        // Append ex to the current segment.
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
            // u variables in the current segment must be
            // less than counter + nvars (which is the starting
            // index of the segment).
            const auto u_indices = udef_args_indices(ex);
            assert(std::ranges::all_of(u_indices, [idx_limit = counter + nvars](auto idx) { return idx < idx_limit; }));
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
    const auto n_vars = gl_arr_size(out_gl[0]);
    const auto n_nums = gl_arr_size(out_gl[2]);
    const auto n_pars = gl_arr_size(out_gl[4]);

    // Fetch the type for external loading.
    auto *ext_fp_t = make_external_llvm_type(fp_scal_t);

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
    auto *ext_fp_t = make_external_llvm_type(fp_type);

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
    auto block_eval = [&](llvm::Function *func, std::uint32_t ncalls, const auto &gens) {
        // LCOV_EXCL_START
        assert(ncalls > 0u);
        assert(!gens.empty());
        assert(std::all_of(gens.begin(), gens.end(), [](const auto &f) { return static_cast<bool>(f); }));
        // LCOV_EXCL_STOP

        // We will be manually unrolling loops if ncalls is small enough.
        // This seems to help with compilation times.
        constexpr auto max_unroll_n = 5u;

        if (ncalls > max_unroll_n) {
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
        } else {
            // The manually-unrolled version of the above.
            for (std::uint32_t idx = 0; idx < ncalls; ++idx) {
                auto *cur_call_idx = builder.getInt32(idx);
                auto u_idx = gens[0](cur_call_idx);
                std::vector<llvm::Value *> args{u_idx, eval_arr, par_ptr, time_ptr, stride};

                for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                    args.push_back(gens[i](cur_call_idx));
                }

                cfunc_c_store_eval(s, fp_vec_type, eval_arr, u_idx, builder.CreateCall(func, args));
            }
        }
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

// NOTE: in strided mode, the compiled function has an extra trailing argument, the stride
// value, which indicates the distance between consecutive
// input/output/par values in the buffers. The stride is measured in the number
// of *scalar* values between input/output/par values.
// For instance, for a batch size of 1 and a stride value of 3,
// the input scalar values will be read from indices 0, 3, 6, 9, ...
// in the input array. For a batch size of 2 and a stride value of 3,
// the input vector values (of size 2) will be read from indices
// [0, 1], [3, 4], [6, 7], [9, 10], ... in the input array.
//
// In non-strided mode, the compiled function indexes into the
// input/output/par buffers contiguously (that is,
// it assumes the input/output/par scalar/vector values are stored one
// after the other without "holes" between them).
//
// NOTE: there is a bunch of boilerplate logic overlap here with make_multi_cfunc(). Make sure to
// coordinate changes between the two functions.
template <typename T, typename F>
auto add_cfunc_impl(llvm_state &s, const std::string &name, const F &fn, std::uint32_t batch_size, bool high_accuracy,
                    bool compact_mode, bool parallel_mode, [[maybe_unused]] long long prec, bool strided)
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
    auto dc = function_decompose(fn.first, fn.second);
    const auto nvars = boost::numeric_cast<std::uint32_t>(fn.second.size());
    const auto nouts = boost::numeric_cast<std::uint32_t>(fn.first.size());

    // Determine the number of u variables.
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
    // - the stride (if requested).
    //
    // The pointer arguments cannot overlap.

    // Fetch the internal and external types.
    auto *fp_t = to_internal_llvm_type<T>(s, prec);
    auto *ext_fp_t = make_external_llvm_type(fp_t);
    std::vector<llvm::Type *> fargs(4, llvm::PointerType::getUnqual(ext_fp_t));

    if (strided) {
        fargs.push_back(to_external_llvm_type<std::size_t>(context));
    }

    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE

    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &md);
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

    llvm::Value *stride = nullptr;
    if (strided) {
        stride = out_ptr + 4;
        stride->setName("stride");
    } else {
        stride = to_size_t(s, builder.getInt32(batch_size));
    }

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

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return dc;
}

} // namespace

template <typename T>
std::vector<expression> add_cfunc(llvm_state &s, const std::string &name, const std::vector<expression> &v_ex,
                                  const std::vector<expression> &vars, std::uint32_t batch_size, bool high_accuracy,
                                  bool compact_mode, bool parallel_mode, long long prec, bool strided)
{
    return detail::add_cfunc_impl<T>(s, name, std::make_pair(std::cref(v_ex), std::cref(vars)), batch_size,
                                     high_accuracy, compact_mode, parallel_mode, prec, strided);
}

// Explicit instantiations.
#define HEYOKA_ADD_CFUNC_INST(T)                                                                                       \
    template HEYOKA_DLL_PUBLIC std::vector<expression> add_cfunc<T>(                                                   \
        llvm_state &, const std::string &, const std::vector<expression> &, const std::vector<expression> &,           \
        std::uint32_t, bool, bool, bool, long long, bool);

HEYOKA_ADD_CFUNC_INST(float)
HEYOKA_ADD_CFUNC_INST(double)
HEYOKA_ADD_CFUNC_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ADD_CFUNC_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_ADD_CFUNC_INST(mppp::real)

#endif

#undef HEYOKA_ADD_CFUNC_INST

namespace
{

// Implementation of the compact-mode evaluation of a compiled function split over several driver functions
// implemented in distinct llvm_state objects.
//
// states is the current list of states (to which more will be added by this function), and the last state
// in the list is the "main" state. main_fp_t is the internal scalar floating-point type as defined in the main state.
// s_dc is the segmented decomposition of the function to be compiled.
// base_name is the name of the main function from which the drivers are to be invoked. main_eval_arr,
// main_par_ptr, main_time_ptr and main_stride are, respectively, the pointer to the evaluation tape,
// the pointer to the parameter values, the pointer to time coordinate(s) and the stride - these are all
// defined in the main state and they are passed to the driver functions invocations.
template <typename SDC>
void multi_cfunc_evaluate_segments(llvm::Type *main_fp_t, std::list<llvm_state> &states, const SDC &s_dc,
                                   std::uint32_t nvars, std::uint32_t batch_size, bool high_accuracy,
                                   const std::string &base_name, llvm::Value *main_eval_arr, llvm::Value *main_par_ptr,
                                   llvm::Value *main_time_ptr, llvm::Value *main_stride)
{
    assert(!states.empty()); // LCOV_EXCL_LINE
    auto &main_state = states.back();

    // Structure used to log, in trace mode, the breakdown of each segment.
    // For each segment, this structure contains the number of invocations
    // of each evaluation function in the segment. It will be unused if we are not tracing.
    std::vector<std::vector<std::uint32_t>> segment_bd;

    // Are we tracing?
    const auto is_tracing = get_logger()->should_log(spdlog::level::trace);

    // List of evaluation functions in a segment.
    //
    // This map contains a list of functions for the compact-mode evaluation of elementary subexpressions.
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

    // Push back a new state and use it as initial current state.
    states.push_back(main_state.make_similar());
    auto *cur_state = &states.back();

    // Index of the state we are currently operating on,
    // relative to the original number of states.
    boost::safe_numerics::safe<unsigned> cur_state_idx = 0;

    // Is the stride value a constant?
    const auto const_stride = llvm::isa<llvm::ConstantInt>(main_stride);

    // Helper to create and return the prototype of a driver function in the state s.
    auto make_driver_proto = [&base_name, const_stride](llvm_state &s, unsigned cur_idx) {
        auto &builder = s.builder();
        auto &md = s.module();
        auto &ctx = s.context();

        // The arguments to the driver are:
        // - a pointer to the tape,
        // - pointers to par and time,
        // - the stride (if not a constant).
        std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(ctx), llvm::PointerType::getUnqual(ctx),
                                        llvm::PointerType::getUnqual(ctx)};
        if (!const_stride) {
            fargs.push_back(to_external_llvm_type<std::size_t>(ctx));
        }

        // The driver does not return anything.
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        assert(ft != nullptr); // LCOV_EXCL_LINE

        // Now create the driver.
        const auto cur_name = fmt::format("{}.driver_{}", base_name, cur_idx);
        auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, cur_name, &md);
        // NOTE: the driver cannot call itself recursively.
        f->addFnAttr(llvm::Attribute::NoRecurse);

        // Add the arguments' attributes.
        // NOTE: no aliasing is assumed between the pointer
        // arguments.
        auto *eval_arr_arg = f->args().begin();
        eval_arr_arg->setName("eval_arr_ptr");
        eval_arr_arg->addAttr(llvm::Attribute::NoCapture);
        eval_arr_arg->addAttr(llvm::Attribute::NoAlias);

        auto *par_ptr_arg = eval_arr_arg + 1;
        par_ptr_arg->setName("par_ptr");
        par_ptr_arg->addAttr(llvm::Attribute::NoCapture);
        par_ptr_arg->addAttr(llvm::Attribute::NoAlias);
        par_ptr_arg->addAttr(llvm::Attribute::ReadOnly);

        auto *time_ptr_arg = eval_arr_arg + 2;
        time_ptr_arg->setName("time_ptr");
        time_ptr_arg->addAttr(llvm::Attribute::NoCapture);
        time_ptr_arg->addAttr(llvm::Attribute::NoAlias);
        time_ptr_arg->addAttr(llvm::Attribute::ReadOnly);

        if (!const_stride) {
            auto *stride_arg = eval_arr_arg + 3;
            stride_arg->setName("stride");
        }

        return f;
    };

    // Helper to invoke a driver function from the main state.
    auto main_invoke_driver
        = [&main_state, main_eval_arr, main_par_ptr, main_time_ptr, main_stride, const_stride](llvm::Function *f) {
              std::vector fargs = {main_eval_arr, main_par_ptr, main_time_ptr};
              if (!const_stride) {
                  fargs.push_back(main_stride);
              }

              main_state.builder().CreateCall(f, fargs);
          };

    // Add the driver declaration to the main state, and invoke it.
    main_invoke_driver(make_driver_proto(main_state, cur_state_idx));

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
    auto cur_u_idx = nvars;

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

            // Add the driver declaration to the main state, and invoke it.
            main_invoke_driver(make_driver_proto(main_state, cur_state_idx));

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

        // This structure maps a function to sets of arguments with which the function
        // is to be called. For instance, if function f(x, y, z) is to be called as
        // f(a, b, c) and f(d, e, f), then tmp_map will contain {f : [[a, b, c], [d, e, f]]}.
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
            auto *func
                = std::get<heyoka::func>(ex.value()).llvm_c_eval_func(*cur_state, fp_t, batch_size, high_accuracy);

            // Insert the function into tmp_map.
            const auto [it, is_new_func] = tmp_map.try_emplace(func);

            assert(is_new_func || !it->second.empty()); // LCOV_EXCL_LINE

            // Convert the variables/constants in the current dc
            // element into a set of indices/constants.
            const auto c_args = udef_to_variants(ex, {});

            // LCOV_EXCL_START
            if (!is_new_func && it->second.back().size() - 1u != c_args.size()) [[unlikely]] {
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

        // Create the seg_f_list_t for the current segment.
        seg_f_list_t seg_map;

        for (const auto &[func, vv] : tmp_map_transpose) {
            // NOTE: vv.size() is now the number of arguments. We know it cannot
            // be zero because the evaluation functions
            // in compact mode always have at least 1 argument (i.e., the index
            // of the u variable which is being evaluated).
            assert(!vv.empty()); // LCOV_EXCL_LINE

            // Add the function.
            const auto [it, ins_status] = seg_map.try_emplace(func);
            assert(ins_status); // LCOV_EXCL_LINE

            // Set the number of calls for this function.
            it->second.first
                = std::visit([](const auto &x) { return boost::numeric_cast<std::uint32_t>(x.size()); }, vv[0]);
            assert(it->second.first > 0u); // LCOV_EXCL_LINE

            // Create the generators for each argument.
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
        auto *eval_arr = driver_f->args().begin();
        auto *par_ptr = driver_f->args().begin() + 1;
        auto *time_ptr = driver_f->args().begin() + 2;

        // NOTE: the stride is not an argument, if constant.
        llvm::Value *stride = nullptr;
        if (const_stride) {
            // LCOV_EXCL_START
            // NOTE: make sure that the bit width of the constant stride argument
            // is the correct one, so that llvm::ConstantInt::get() produces a
            // constant integer of the correct type.
            assert(llvm::cast<llvm::ConstantInt>(main_stride)->getValue().getBitWidth()
                   == static_cast<unsigned>(std::numeric_limits<std::size_t>::digits));
            // LCOV_EXCL_STOP

            stride
                = llvm::ConstantInt::get(cur_state->context(), llvm::cast<llvm::ConstantInt>(main_stride)->getValue());
        } else {
            stride = driver_f->args().begin() + 3;
        }

        // Generate the code for the evaluation of all blocks in the segment.
        for (const auto &[func, fpair] : seg_map) {
            const auto &[ncalls, gens] = fpair;

            // LCOV_EXCL_START
            assert(ncalls > 0u);
            assert(!gens.empty());
            assert(std::ranges::all_of(gens, [](const auto &f) { return static_cast<bool>(f); }));
            // LCOV_EXCL_STOP

            // We will be manually unrolling loops if ncalls is small enough.
            // This seems to help with compilation times.
            constexpr auto max_unroll_n = 5u;

            if (ncalls > max_unroll_n) {
                // Loop over the number of calls.
                llvm_loop_u32(*cur_state, cur_builder.getInt32(0), cur_builder.getInt32(ncalls),
                              [&](llvm::Value *cur_call_idx) {
                                  // Create the u variable index from the first generator.
                                  auto *u_idx = gens[0](cur_call_idx);

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
                                  cfunc_c_store_eval(*cur_state, fp_vec_type, eval_arr, u_idx,
                                                     cur_builder.CreateCall(func, args));
                              });
            } else {
                // The manually-unrolled version of the above.
                for (std::uint32_t idx = 0; idx < ncalls; ++idx) {
                    auto *cur_call_idx = cur_builder.getInt32(idx);
                    auto u_idx = gens[0](cur_call_idx);
                    std::vector<llvm::Value *> args{u_idx, eval_arr, par_ptr, time_ptr, stride};

                    for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                        args.push_back(gens[i](cur_call_idx));
                    }

                    cfunc_c_store_eval(*cur_state, fp_vec_type, eval_arr, u_idx, cur_builder.CreateCall(func, args));
                }
            }
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
        get_logger()->trace("make_multi_cfunc() function maps breakdown: {}", segment_bd);
    }
    // LCOV_EXCL_STOP
}

std::array<std::size_t, 2> add_multi_cfunc_impl(llvm::Type *fp_t, std::list<llvm_state> &states, llvm::Value *out_ptr,
                                                llvm::Value *in_ptr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                                llvm::Value *stride, const std::vector<expression> &dc,
                                                std::uint32_t nvars,
                                                // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                std::uint32_t nuvars, std::uint32_t batch_size, bool high_accuracy,
                                                const std::string &base_name, llvm::Value *eval_arr)
{
    // Fetch the main state, module, etc.
    auto &main_state = states.back();
    auto &main_md = main_state.module();
    auto &main_builder = main_state.builder();

    // Fetch the fp types for the main state.
    // NOTE: cloning is safe here, as even though this function is being invoked
    // in parallel from multiple threads, we have made sure that each invocation
    // gets its own cloned copy of fp_t.
    auto *main_fp_t = llvm_clone_type(main_state, fp_t);
    auto *main_ext_fp_t = make_external_llvm_type(main_fp_t);
    auto *fp_vec_type = make_vector_type(main_fp_t, batch_size);

    // Split dc into segments.
    const auto s_dc = function_segment_dc(dc, nvars, nuvars);

    // Generate the global arrays used to write the outputs at the
    // end of the computation.
    const auto cout_gl = cfunc_c_make_output_globals(main_state, main_fp_t, dc, nuvars);

    // Total required size in bytes for the tape.
    const auto sz = boost::safe_numerics::safe<std::size_t>(get_size(main_md, fp_vec_type)) * nuvars;

    // Tape alignment.
    const auto al = boost::numeric_cast<std::size_t>(get_alignment(main_md, fp_vec_type));

    // NOTE: eval_arr is used as temporary storage for the current function,
    // but it is provided externally from dynamically-allocated memory in order to avoid stack overflow.
    // This creates a situation in which LLVM cannot elide stores into eval_arr
    // (even if it figures out a way to avoid storing intermediate results into
    // eval_arr) because LLVM must assume that some other function may
    // use these stored values later. Thus, we declare via an intrinsic that the
    // lifetime of eval_arr begins here and ends at the end of the function,
    // so that LLVM can assume that any value stored in it cannot be possibly
    // used outside this function.
    main_builder.CreateLifetimeStart(eval_arr, main_builder.getInt64(sz));

    // Copy over the values of the variables.
    llvm_loop_u32(main_state, main_builder.getInt32(0), main_builder.getInt32(nvars), [&](llvm::Value *cur_var_idx) {
        // Fetch the pointer from in_ptr.
        auto *ptr = main_builder.CreateInBoundsGEP(main_ext_fp_t, in_ptr,
                                                   main_builder.CreateMul(stride, to_size_t(main_state, cur_var_idx)));

        // Load as a vector.
        auto *vec = ext_load_vector_from_memory(main_state, main_fp_t, ptr, batch_size);

        // Store into eval_arr.
        cfunc_c_store_eval(main_state, fp_vec_type, eval_arr, cur_var_idx, vec);
    });

    // Generate the code for the evaluation of all segments.
    multi_cfunc_evaluate_segments(main_fp_t, states, s_dc, nvars, batch_size, high_accuracy, base_name, eval_arr,
                                  par_ptr, time_ptr, stride);

    // Write the results to the output pointer.
    cfunc_c_write_outputs(main_state, main_fp_t, out_ptr, cout_gl, eval_arr, par_ptr, stride, batch_size);

    // End the lifetime of eval_arr.
    main_builder.CreateLifetimeEnd(eval_arr, main_builder.getInt64(sz));

    return {sz, al};
}

std::tuple<llvm_multi_state, std::vector<expression>, std::vector<std::array<std::size_t, 2>>>
make_multi_cfunc_impl(llvm::Type *fp_t, const llvm_state &tplt, const std::string &name,
                      const std::vector<expression> &fn, const std::vector<expression> &vars, std::uint32_t batch_size,
                      bool high_accuracy, bool parallel_mode)
{
    if (batch_size == 0u) [[unlikely]] {
        throw std::invalid_argument("The batch size of a compiled function cannot be zero");
    }

    if (parallel_mode) [[unlikely]] {
        throw std::invalid_argument("Parallel mode has not been implemented yet");
    }

    if (name.empty()) [[unlikely]] {
        throw std::invalid_argument("A non-empty function name is required when invoking make_multi_cfunc()");
    }

    // Decompose the function and cache the number of vars and outputs.
    auto dc = function_decompose(fn, vars);
    const auto nvars = boost::numeric_cast<std::uint32_t>(vars.size());
    const auto nouts = boost::numeric_cast<std::uint32_t>(fn.size());

    // Determine the number of u variables.
    assert(dc.size() >= nouts); // LCOV_EXCL_LINE
    const auto nuvars = boost::numeric_cast<std::uint32_t>(dc.size() - nouts);

    // NOTE: due to the presence of the stride argument, we will be always
    // indexing into the input, output and parameter arrays via size_t.
    // Hence, we don't need here the same overflow checking we need to perform
    // in the integrators, as we assume that any array allocated from C++
    // can't have a size larger than the max size_t.

    // Init the states lists.
    // NOTE: we use lists here because it is convenient to have
    // pointer/reference stability when iteratively constructing
    // the set of states.
    std::vector<std::list<llvm_state>> states_lists;
    // NOTE: if the batch size is 1, we build 2 cfuncs. Otherwise,
    // we build 3.
    if (batch_size == 1u) {
        states_lists.resize(2);
    } else {
        states_lists.resize(3);
    }

    // Init the tape size/alignment requirements vector.
    std::vector<std::array<std::size_t, 2>> tape_size_align;
    // NOTE: if the batch size is 1, we only record the size/alignment
    // requirements of the scalar tape. Otherwise, we also record
    // the size/alignment requirements of the batch-mode tape.
    if (batch_size == 1u) {
        tape_size_align.resize(1);
    } else {
        tape_size_align.resize(2);
    }

    // NOTE: this is ugly, but needed. Cloning an LLVM type into another
    // context is not a thread-safe operation as we might be poking into
    // the context of the original type. Thus, we first make 2 or 3 clones
    // of fp_t each associated to a different llvm_state without any multithreading,
    // and then we use these clones for further cloning while parallel invoking
    // create_cfunc().
    std::vector<std::pair<llvm_state, llvm::Type *>> fp_t_clones;
    fp_t_clones.reserve(3);
    for (auto i = 0; i < (batch_size == 1u ? 2 : 3); ++i) {
        // Create a new state and clone fp_t into it.
        auto new_state = tplt.make_similar();
        auto *new_fp_t = llvm_clone_type(new_state, fp_t);

        fp_t_clones.emplace_back(std::move(new_state), new_fp_t);
    }

    // Helper to create a single cfunc.
    auto create_cfunc = [&states_lists, &tape_size_align, &tplt, &name, &dc = std::as_const(dc), nvars, nuvars,
                         high_accuracy,
                         &fp_t_clones = std::as_const(fp_t_clones)](bool strided, std::uint32_t cur_batch_size) {
        // NOTE: the batch unstrided variant is not supposed to be requested.
        assert(strided || cur_batch_size == 1u);

        // Which list of states are we operating on?
        auto sidx = 0u;
        if (cur_batch_size == 1u) {
            sidx = strided ? 1 : 0;
        } else {
            sidx = 2;
        }

        // Fetch the list of states.
        auto &states = states_lists[sidx];

        assert(states.empty());

        // Fetch the local cloned fp_t.
        auto *loc_fp_t = fp_t_clones[sidx].second;

        // Add a new state and fetch it.
        states.push_back(tplt.make_similar());
        auto &s = states.back();

        // Fetch builder/context/module for the new state.
        auto &builder = s.builder();
        auto &context = s.context();
        auto &md = s.module();

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // Prepare the arguments:
        //
        // - a write-only pointer to the outputs,
        // - a read-only pointer to the inputs,
        // - a read-only pointer to the pars,
        // - a read-only pointer to the time value(s),
        // - a read/write pointer to the tape storage,
        // - the stride (if requested).
        //
        // The pointer arguments cannot overlap.
        std::vector<llvm::Type *> fargs(5, llvm::PointerType::getUnqual(context));

        if (strided) {
            fargs.push_back(to_external_llvm_type<std::size_t>(context));
        }

        // The function does not return anything.
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        assert(ft != nullptr); // LCOV_EXCL_LINE

        // Create the function prototype.
        const auto cur_name
            = fmt::format("{}.{}.batch_size_{}", name, strided ? "strided" : "unstrided", cur_batch_size);
        auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, cur_name, &md);
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

        auto *tape_ptr = out_ptr + 4;
        tape_ptr->setName("tape_ptr");
        tape_ptr->addAttr(llvm::Attribute::NoCapture);
        tape_ptr->addAttr(llvm::Attribute::NoAlias);

        llvm::Value *stride = nullptr;
        if (strided) {
            stride = out_ptr + 5;
            stride->setName("stride");
        } else {
            stride = to_size_t(s, builder.getInt32(cur_batch_size));
        }

        // Create a new basic block to start insertion into.
        auto *bb = llvm::BasicBlock::Create(context, "entry", f);
        assert(bb != nullptr); // LCOV_EXCL_LINE
        builder.SetInsertPoint(bb);

        // Create the body of the function.
        const auto tape_sa = add_multi_cfunc_impl(loc_fp_t, states, out_ptr, in_ptr, par_ptr, time_ptr, stride, dc,
                                                  nvars, nuvars, cur_batch_size, high_accuracy, cur_name, tape_ptr);

        // Add the size/alignment requirements for the tape storage.
        // NOTE: there's no difference in requirements between strided and
        // unstrided variants. Assign only is strided mode in order to avoid data races.
        if (strided) {
            tape_size_align[cur_batch_size > 1u] = tape_sa;
        }

        // Finish off the function.
        builder.CreateRetVoid();

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    };

    // Log the runtime of IR construction in trace mode.
    spdlog::stopwatch sw;

    // Build the compiled functions.
    //
    // NOTE: in order to further parallelise the creation of the individual functions, we should:
    //
    // - do a first pass analyzing the decomposition in order to decide where to initiate
    //   the construction of the drivers,
    // - construct the drivers in parallel, perhaps pushing back the states as they are constructed
    //   into a thread-safe tbb vector.
    //
    // At the moment though it looks like the practical gains from such further parallelisation
    // would not be worth it, perhaps we can reconsider in the future. It is also not clear how
    // to deal with thread-unsafe type cloning in this hypothetical scenario.
    if (batch_size == 1u) {
        oneapi::tbb::parallel_invoke([&create_cfunc]() { create_cfunc(false, 1); },
                                     [&create_cfunc]() { create_cfunc(true, 1); });
    } else {
        oneapi::tbb::parallel_invoke([&create_cfunc]() { create_cfunc(false, 1); },
                                     [&create_cfunc]() { create_cfunc(true, 1); },
                                     [&create_cfunc, batch_size]() { create_cfunc(true, batch_size); });
    }

    // Consolidate all the state lists into a single one.
    states_lists[0].splice(states_lists[0].end(), states_lists[1]);
    if (batch_size > 1u) {
        states_lists[0].splice(states_lists[0].end(), states_lists[2]);
    }

    get_logger()->trace("make_multi_cfunc() IR creation runtime: {}", sw);

    // NOTE: in C++23 we could use std::ranges::views::as_rvalue instead of
    // the custom transform:
    //
    // https://en.cppreference.com/w/cpp/ranges/as_rvalue_view
    return std::make_tuple(
        llvm_multi_state(states_lists[0] | std::views::transform([](auto &s) -> auto && { return std::move(s); })),
        std::move(dc), std::move(tape_size_align));
}

} // namespace

// This function will compile several versions of the input function fn, with input variables vars, in compact mode.
//
// The compiled functions are implemented across several llvm_states which are collated together and returned as
// a single llvm_multi_state (this is the first element of the return tuple). If batch_size is 1,
// then 2 compiled functions are created - a scalar strided and a scalar unstrided version.
// If batch size is > 1, then an additional batch-mode strided compiled function is returned.
// The function names are created using "name" as base name and then mangling in the strided/unstrided
// property and the batch size.
//
// The second element of the return tuple is the decomposition of fn.
//
// The third element of the return tuple is a vector of pairs, each pair containing the size and alignment requirements
// for the externally-provided storage for the evaluation tape. If batch_size is 1, then only a single
// pair is returned, representing the size/alignment requirements for the scalar-mode evaluation tape.
// If batch_size > 1, then an additional pair is appended representing the size/alignment requirements
// for the batch-mode evaluation tape.
//
// NOTE: there is a bunch of boilerplate logic overlap here with add_cfunc_impl(). Make sure to
// coordinate changes between the two functions.
template <typename T>
std::tuple<llvm_multi_state, std::vector<expression>, std::vector<std::array<std::size_t, 2>>>
make_multi_cfunc(llvm_state tplt, const std::string &name, const std::vector<expression> &fn,
                 const std::vector<expression> &vars, std::uint32_t batch_size, bool high_accuracy, bool parallel_mode,
                 long long prec)
{
#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        const auto sprec = boost::numeric_cast<mpfr_prec_t>(prec);

        if (sprec < mppp::real_prec_min() || sprec > mppp::real_prec_max()) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("An invalid precision value of {} was passed to make_multi_cfunc() (the "
                            "value must be in the [{}, {}] range)",
                            sprec, mppp::real_prec_min(), mppp::real_prec_max()));
        }
    }

#endif

    // Fetch the internal scalar fp type from the template state. We will be cloning
    // this throughout the rest of the implementation.
    auto *fp_t = to_internal_llvm_type<T>(tplt, prec);

    return make_multi_cfunc_impl(fp_t, tplt, name, fn, vars, batch_size, high_accuracy, parallel_mode);
}

// Explicit instantiations.
#define HEYOKA_MAKE_MULTI_CFUNC_INST(T)                                                                                \
    template HEYOKA_DLL_PUBLIC                                                                                         \
        std::tuple<llvm_multi_state, std::vector<expression>, std::vector<std::array<std::size_t, 2>>>                 \
        make_multi_cfunc<T>(llvm_state, const std::string &, const std::vector<expression> &,                          \
                            const std::vector<expression> &, std::uint32_t, bool, bool, long long);

HEYOKA_MAKE_MULTI_CFUNC_INST(float)
HEYOKA_MAKE_MULTI_CFUNC_INST(double)
HEYOKA_MAKE_MULTI_CFUNC_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_MAKE_MULTI_CFUNC_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_MAKE_MULTI_CFUNC_INST(mppp::real)

#endif

#undef HEYOKA_MAKE_MULTI_CFUNC_INST

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
