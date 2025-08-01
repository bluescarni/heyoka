// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/container/deque.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <oneapi/tbb/blocked_range.h>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/num_identity.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/tbb_isolated.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/exp.hpp>
#include <heyoka/math/log.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: this function will return a pair containing:
//
// - the mangled name and
// - the list of LLVM argument types
//
// for the function implementing the Taylor derivative in compact mode of the mathematical function
// called "name". The mangled name is assembled from "name", the types of the arguments args, the number
// of uvars and the scalar or vector floating-point type in use (which depends on fp_t and batch_size).
//
// NOTE: the values in args are inconsequential, only the types matter.
std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args(llvm::LLVMContext &context, llvm::Type *fp_t, const std::string &name,
                             // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                             std::uint32_t n_uvars, std::uint32_t batch_size,
                             const std::vector<std::variant<variable, number, param>> &args,
                             std::uint32_t n_hidden_deps)
{
    assert(fp_t != nullptr);
    assert(n_uvars > 0u);

    // LCOV_EXCL_START

    // Check that 'name' does not contain periods ".". Periods are used
    // to separate fields in the mangled name.
    if (boost::contains(name, ".")) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Cannot generate a mangled name for the compact mode Taylor derivative of the mathematical "
                        "function '{}': the function name cannot contain '.' symbols",
                        name));
    }

    // LCOV_EXCL_STOP

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the external type corresponding to fp_t.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Init the name.
    auto fname = fmt::format("heyoka.taylor_c_diff.{}.", name);

    // Init the vector of arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array (pointer to val_t),
    // - par ptr (pointer to external scalar),
    // - time ptr (pointer to external scalar).
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), llvm::PointerType::getUnqual(ext_fp_t),
                                    llvm::PointerType::getUnqual(ext_fp_t)};

    // Add the mangling and LLVM arg types for the argument types. Also, detect if
    // we have variables in the arguments.
    bool with_var = false;
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        // Detect variable.
        if (std::holds_alternative<variable>(args[i])) {
            with_var = true;
        }

        // Name mangling.
        fname += std::visit([](const auto &v) { return cm_mangle(v); }, args[i]);

        // Add the arguments separator, if we are not at the
        // last argument.
        if (i != args.size() - 1u) {
            fname += '_';
        }

        // Add the LLVM function argument type.
        fargs.push_back(std::visit(
            [&](const auto &v) -> llvm::Type * {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::same_as<type, number>) {
                    // For numbers, the argument is passed as a scalar
                    // floating-point value.
                    return val_t->getScalarType();
                } else if constexpr (std::is_same_v<type, variable> || std::is_same_v<type, param>) {
                    // For vars and params, the argument is an index
                    // in an array.
                    return llvm::Type::getInt32Ty(context);
                } else {
                    // LCOV_EXCL_START
                    assert(false);
                    throw;
                    // LCOV_EXCL_STOP
                }
            },
            args[i]));
    }

    // Close the argument list with a ".".
    // NOTE: this will result in a ".." in the name
    // if the function has zero arguments.
    fname += '.';

    // If we have variables in the arguments, add mangling
    // for n_uvars.
    if (with_var) {
        fname += fmt::format("n_uvars_{}.", n_uvars);
    } else {
        // NOTE: make sure we put something in this field even
        // if we do not have variables in the arguments. This ensures
        // that a mangled function name has a fixed number of fields, and that
        // n_uvars_{} cannot possibly be interpreted as the mangled
        // name of a floating-point type.
        fname += "no_uvars.";
    }

    // Finally, add the mangling for the floating-point type.
    fname += llvm_mangle_type(val_t);

    // Fill in the hidden dependency arguments. These are all indices.
    fargs.insert(fargs.end(), boost::numeric_cast<decltype(fargs.size())>(n_hidden_deps),
                 llvm::Type::getInt32Ty(context));

    return std::make_pair(std::move(fname), std::move(fargs));
}

llvm::Value *taylor_codegen_numparam(llvm_state &s, llvm::Type *fp_t, const number &num, llvm::Value *,
                                     std::uint32_t batch_size)
{
    return vector_splat(s.builder(), llvm_codegen(s, fp_t, num), batch_size);
}

llvm::Value *taylor_codegen_numparam(llvm_state &s, llvm::Type *fp_t, const param &p, llvm::Value *par_ptr,
                                     std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(llvm::isa<llvm::PointerType>(par_ptr->getType()));
    assert(!llvm::cast<llvm::PointerType>(par_ptr->getType())->isVectorTy());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the external type corresponding to fp_t.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Determine the index into the parameter array.
    // LCOV_EXCL_START
    if (p.idx() > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("Overflow detected in the computation of the index into a parameter array");
    }
    // LCOV_EXCL_STOP
    const auto arr_idx = static_cast<std::uint32_t>(p.idx() * batch_size);

    // Compute the pointer to load from.
    auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, par_ptr, builder.getInt32(arr_idx));

    // Load.
    return ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
}

// Codegen helpers for number/param for use in the generic c_diff implementations.
llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &s, llvm::Type *, const number &, llvm::Value *n, llvm::Value *,
                                            std::uint32_t batch_size)
{
    return vector_splat(s.builder(), n, batch_size);
}

llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &s, llvm::Type *fp_t, const param &, llvm::Value *p,
                                            llvm::Value *par_ptr, std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(llvm::isa<llvm::PointerType>(par_ptr->getType()));
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the external type corresponding to fp_t.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Fetch the pointer into par_ptr.
    // NOTE: the overflow check is done when constructing the integrator.
    auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, par_ptr, builder.CreateMul(p, builder.getInt32(batch_size)));

    return ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
}

// Helper to fetch the derivative of order 'order' of the u variable at index u_idx from the
// derivative array 'arr'. The total number of u variables is n_uvars.
llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &arr, std::uint32_t u_idx, std::uint32_t order,
                               std::uint32_t n_uvars)
{
    // Sanity check.
    assert(u_idx < n_uvars);

    // Compute the index.
    const auto idx = (static_cast<decltype(arr.size())>(order) * n_uvars) + u_idx;
    assert(idx < arr.size());

    return arr[idx];
}

// Load the derivative of order 'order' of the u variable u_idx from the array of Taylor derivatives diff_arr.
// n_uvars is the total number of u variables.
llvm::Value *taylor_c_load_diff(llvm_state &s, llvm::Type *val_t, llvm::Value *diff_arr, std::uint32_t n_uvars,
                                llvm::Value *order, llvm::Value *u_idx)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    auto *ptr = builder.CreateInBoundsGEP(
        val_t, diff_arr, builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx));

    return builder.CreateLoad(val_t, ptr);
}

// Store the value val as the derivative of order 'order' of the u variable u_idx
// into the array of Taylor derivatives diff_arr. n_uvars is the total number of u variables.
void taylor_c_store_diff(llvm_state &s, llvm::Type *val_t, llvm::Value *diff_arr, std::uint32_t n_uvars,
                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         llvm::Value *order, llvm::Value *u_idx, llvm::Value *val)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    auto *ptr = builder.CreateInBoundsGEP(
        val_t, diff_arr, builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx));

    builder.CreateStore(val, ptr);
}

namespace
{

// Simplify a Taylor decomposition by removing common subexpressions.
//
// NOTE: the hidden deps are not considered for CSE purposes, only the actual subexpressions.
auto taylor_decompose_cse(const taylor_dc_t &v_ex, const std::vector<std::uint32_t> &sv_funcs_dc,
                          const taylor_dc_t::size_type n_eq)
{
    using idx_t = taylor_dc_t::size_type;

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Cache the original size for logging later.
    const auto orig_size = v_ex.size();

    // A Taylor decomposition is supposed to have n_eq variables at the beginning, n_eq variables at the end and
    // possibly extra variables in the middle.
    assert(v_ex.size() >= n_eq * 2u);

    // Init the new decomposition.
    taylor_dc_t new_dc;

    // expression -> idx map. This will end up containing all the unique expressions from v_ex, and it will
    // map them to their indices in new_dc (which will in general differ from their indices in v_ex).
    //
    // NOTE: use std::map (rather than an unordered map) for the usual reason that comparison-based
    // containers can perform better than hashing on account of the fact that comparison does not
    // need to traverse the entire expression.
    std::map<expression, idx_t> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // NOTE: these are caches used in the renaming of the expressions in v_ex.
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    // The first n_eq definitions are just renaming
    // of the state variables into u variables.
    for (idx_t i = 0; i < n_eq; ++i) {
        assert(std::holds_alternative<variable>(v_ex[i].first.value()));
        // NOTE: no hidden deps allowed here.
        assert(v_ex[i].second.empty());
        new_dc.push_back(v_ex[i]);

        // NOTE: the u vars that correspond to state variables are never simplified, thus they do not need remapping.
        // However, we need them to show up in uvars_rename in case an sv func is identical to a state variable (because
        // uvars rename will be used to remap the indices in sv_funcs_dc).
        [[maybe_unused]] const auto res = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Handle the u variables which do not correspond to state variables.
    for (auto i = n_eq; i < v_ex.size() - n_eq; ++i) {
        const auto &[orig_ex, orig_deps] = v_ex[i];

        // Rename the u variables in orig_ex.
        // NOTE: the point of using rename_variables_impl() with the caches, rather than rename_variables(), is that we
        // want to avoid creating multiple copies of shared arguments.
        auto new_ex = rename_variables_impl(func_map, sargs_map, orig_ex, uvars_rename);

        if (const auto it = ex_map.find(new_ex); it == ex_map.end()) {
            // This is the first occurrence of new_ex in the
            // decomposition. Add it to new_dc.
            new_dc.emplace_back(new_ex, orig_deps);

            // Add new_ex to ex_map, mapping it to
            // the index it corresponds to in new_dc
            // (let's call it j).
            ex_map.emplace(std::move(new_ex), new_dc.size() - 1u);

            // Update uvars_rename. This will ensure that
            // occurrences of the variable 'u_i' in the next
            // elements of v_ex will be renamed to 'u_j'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", new_dc.size() - 1u));
            assert(res.second);
        } else {
            // new_ex is redundant. This means
            // that it already appears in new_dc at index
            // it->second. Don't add anything to new_dc,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", it->second));
            assert(res.second);
        }
    }

    // Handle the derivatives of the state variables at the
    // end of the decomposition. We just need to ensure that
    // the u variables in their definitions are renamed with
    // the new indices.
    for (auto i = v_ex.size() - n_eq; i < v_ex.size(); ++i) {
        const auto &[orig_ex, orig_deps] = v_ex[i];

        // NOTE: here we expect only non-func expressions and no hidden dependencies.
        assert(!std::holds_alternative<func>(orig_ex.value()));
        assert(orig_deps.empty());

        auto new_ex = rename_variables_impl(func_map, sargs_map, orig_ex, uvars_rename);

        // The new expression must not show up in ex_map.
        assert(!ex_map.contains(new_ex));

        new_dc.emplace_back(std::move(new_ex), orig_deps);
    }

    // Re-adjust all indices in the hidden dependencies in order to account
    // for the renaming of the uvars.
    for (auto &[_, deps] : new_dc) {
        for (auto &idx : deps) {
            const auto it = uvars_rename.find(fmt::format("u_{}", idx));
            assert(it != uvars_rename.end());
            idx = uname_to_index(it->second);
        }
    }

    // Same for the indices in sv_funcs_dc.
    std::vector<std::uint32_t> new_sv_funcs_dc;
    new_sv_funcs_dc.reserve(sv_funcs_dc.size());
    for (const auto idx : sv_funcs_dc) {
        const auto it = uvars_rename.find(fmt::format("u_{}", idx));
        assert(it != uvars_rename.end());
        new_sv_funcs_dc.push_back(uname_to_index(it->second));
    }

    get_logger()->debug("Taylor CSE reduced decomposition size from {} to {}", orig_size, new_dc.size());
    get_logger()->trace("Taylor CSE runtime: {}", sw);

    return std::make_pair(std::move(new_dc), std::move(new_sv_funcs_dc));
}

// Perform a topological sort on a graph representation of a Taylor decomposition. This can improve performance by
// grouping together operations that can be performed in parallel, and it also makes compact mode much more effective by
// creating clusters of subexpressions whose derivatives can be computed in parallel.
//
// NOTE: the original decomposition dc is already topologically sorted, in the sense that the definitions of the u
// variables are already ordered according to dependency. However, because the original decomposition comes from a
// depth-first search, it has the tendency to group together expressions which are dependent on each other. By doing
// another topological sort, this time based on breadth-first search, we determine another valid sorting in which
// independent operations tend to be clustered together.
auto taylor_sort_dc(const taylor_dc_t &dc, const std::vector<std::uint32_t> &sv_funcs_dc,
                    const taylor_dc_t::size_type n_eq)
{
    // A Taylor decomposition is supposed to have n_eq variables at the beginning, n_eq variables at the end and
    // possibly extra variables in the middle
    assert(dc.size() >= n_eq * 2u);

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

    // Add the nodes corresponding to the state variables.
    for (taylor_dc_t::size_type i = 0; i < n_eq; ++i) {
        const auto v = boost::add_vertex(g);

        // Add a dependency on the root node.
        boost::add_edge(root_v, v, g);
    }

    // Add the rest of the u variables.
    for (taylor_dc_t::size_type i = n_eq; i < dc.size() - n_eq; ++i) {
        const auto v = boost::add_vertex(g);

        // Fetch the list of variables in the current expression.
        //
        // NOTE: performance with large shared arguments sets here will be quite poor, as we will keep on creating new
        // deep copies of the same set of arguments over and over. This is not currently a practical concern because
        // taylor_sort_dc() is at the moment called only in Taylor integrators, which are not supported by dfun (which
        // is the only use of shared arguments we have thus far). If this becomes an issue in the future, we will have
        // to think of an alternative API that avoids the explosion in complexity.
        const auto vars = get_variables(dc[i].first);

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
    boost::container::deque<decltype(dc.size())> tmp;
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
        // variables are inserted into v_idx in the correct order.
        const auto e_range = boost::out_edges(v, g);
        tmp_edges.assign(e_range.first, e_range.second);
        std::ranges::sort(tmp_edges,
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

    // Adjust v_idx: remove the index of the root node decrease by one all other indices, insert the final n_eq indices.
    for (decltype(v_idx.size()) i = 0; i < v_idx.size() - 1u; ++i) {
        v_idx[i] = v_idx[i + 1u] - 1u;
    }
    v_idx.resize(boost::numeric_cast<decltype(v_idx.size())>(dc.size()));
    std::iota(v_idx.data() + dc.size() - n_eq, v_idx.data() + dc.size(), dc.size() - n_eq);

    // Create the remapping dictionary.
    std::unordered_map<std::string, std::string> remap;
    // NOTE: the u vars that correspond to state variables were inserted into v_idx in the original order, thus they are
    // not re-sorted and they do not need renaming. However, we need them to show up in 'remap' because sv funcs might
    // be state variables, and when remapping sv funcs below we are relying on them showing up in 'remap'.
    for (decltype(v_idx.size()) i = 0; i < n_eq; ++i) {
        assert(v_idx[i] == i);
        [[maybe_unused]] const auto res = remap.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }
    // Establish the remapping for the u variables that are not
    // state variables.
    for (decltype(v_idx.size()) i = n_eq; i < v_idx.size() - n_eq; ++i) {
        [[maybe_unused]] const auto res = remap.emplace(fmt::format("u_{}", v_idx[i]), fmt::format("u_{}", i));
        assert(res.second);
    }

    // NOTE: these are caches used in the renaming of the expressions in dc.
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    // Do the reordering and remapping of the decomposition.
    auto dc_transform_view = v_idx | std::views::transform([&dc](auto idx) -> const auto & { return dc[idx]; })
                             | std::views::transform([&func_map, &sargs_map, &remap](const auto &p) {
                                   const auto &[ex, deps] = p;

                                   // Remap the expression.
                                   // NOTE: the point of using rename_variables_impl() with the caches, rather than
                                   // rename_variables(), is that we want to avoid creating multiple copies of shared
                                   // arguments.
                                   auto new_ex = rename_variables_impl(func_map, sargs_map, ex, remap);

                                   // Remap the hidden deps.
                                   std::vector<std::uint32_t> new_deps;
                                   new_deps.reserve(deps.size());
                                   for (const auto idx : deps) {
                                       const auto it_remap = remap.find(fmt::format("u_{}", idx));
                                       assert(it_remap != remap.end());
                                       new_deps.push_back(uname_to_index(it_remap->second));
                                   }

                                   return std::make_pair(std::move(new_ex), std::move(new_deps));
                               });

    // Do the remap for sv_funcs.
    auto sv_funcs_dc_transform_view = sv_funcs_dc | std::views::transform([&remap](auto idx) {
                                          const auto it_remap = remap.find(fmt::format("u_{}", idx));
                                          assert(it_remap != remap.end());
                                          return uname_to_index(it_remap->second);
                                      });

    auto retval = std::make_pair(
        std::vector(std::ranges::begin(dc_transform_view), std::ranges::end(dc_transform_view)),
        std::vector(std::ranges::begin(sv_funcs_dc_transform_view), std::ranges::end(sv_funcs_dc_transform_view)));

    get_logger()->trace("Taylor topological sort runtime: {}", sw);

    return retval;
}

// LCOV_EXCL_START

#if !defined(NDEBUG)

// Helper to verify a Taylor decomposition.
void verify_taylor_dec(const std::vector<expression> &orig, const taylor_dc_t &dc)
{
    using idx_t = taylor_dc_t::size_type;

    const auto n_eq = orig.size();

    assert(dc.size() >= n_eq * 2u);

    // The first n_eq expressions of u variables
    // must be just variables. No hidden dependencies
    // are allowed.
    for (idx_t i = 0; i < n_eq; ++i) {
        assert(std::holds_alternative<variable>(dc[i].first.value()));
        assert(dc[i].second.empty());
    }

    // From n_eq to dc.size() - n_eq, the expressions
    // must be either:
    // - functions whose arguments
    //   are either variables in the u_n form,
    //   where n < i, or numbers/params, or
    // - numbers (this is a corner case arising
    //   if constant folding happens when adding
    //   hidden dependencies).
    // The hidden dependencies must contain indices
    // only in the [n_eq, dc.size() - n_eq) range.
    for (auto i = n_eq; i < dc.size() - n_eq; ++i) {
        std::visit(
            [i](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    for (const auto &arg : v.args()) {
                        if (auto p_var = std::get_if<variable>(&arg.value())) {
                            assert(p_var->name().rfind("u_", 0) == 0);
                            assert(uname_to_index(p_var->name()) < i);
                        } else if (std::get_if<number>(&arg.value()) == nullptr
                                   && std::get_if<param>(&arg.value()) == nullptr) {
                            assert(false);
                        }
                    }
                } else {
                    assert((std::is_same_v<type, number>));
                }
            },
            dc[i].first.value());

        for (auto idx : dc[i].second) {
            assert(idx >= n_eq);
            assert(idx < dc.size() - n_eq);

            // Hidden dep onto itself does not make any sense.
            assert(idx != i);
        }
    }

    // From dc.size() - n_eq to dc.size(), the expressions
    // must be either variables in the u_n form, where n < dc.size() - n_eq,
    // or numbers/params.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        std::visit(
            [&dc, n_eq](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    assert(v.name().rfind("u_", 0) == 0);
                    assert(uname_to_index(v.name()) < dc.size() - n_eq);
                } else if constexpr (!std::is_same_v<type, number> && !std::is_same_v<type, param>) {
                    assert(false);
                }
            },
            dc[i].first.value());

        // No hidden dependencies.
        assert(dc[i].second.empty());
    }

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of state variables or other u variables,
    // and store it in subs_map.
    for (idx_t i = 0; i < dc.size() - n_eq; ++i) {
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i].first, subs_map));
    }

    // Reconstruct the right-hand sides of the system
    // and compare them to the original ones.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        assert(subs(dc[i].first, subs_map) == orig[i - (dc.size() - n_eq)]);
    }
}

// Helper to verify the decomposition of the sv funcs.
void verify_taylor_dec_sv_funcs(const std::vector<std::uint32_t> &sv_funcs_dc, const std::vector<expression> &sv_funcs,
                                const taylor_dc_t &dc, std::vector<expression>::size_type n_eq)
{
    assert(sv_funcs.size() == sv_funcs_dc.size());

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of state variables or other u variables,
    // and store it in subs_map.
    for (decltype(dc.size()) i = 0; i < dc.size() - n_eq; ++i) {
        subs_map.emplace(fmt::format("u_{}", i), subs(dc[i].first, subs_map));
    }

    // Reconstruct the sv functions and compare them to the
    // original ones.
    for (decltype(sv_funcs.size()) i = 0; i < sv_funcs.size(); ++i) {
        assert(sv_funcs_dc[i] < dc.size());

        auto sv_func = subs(dc[sv_funcs_dc[i]].first, subs_map);
        assert(sv_func == sv_funcs[i]);
    }
}

#endif

// LCOV_EXCL_STOP

// Replace subexpressions in dc consisting
// of numbers with number identity functions.
// Number subexpressions can occur in case of
// constant folding when adding hidden dependencies.
void taylor_decompose_replace_numbers(taylor_dc_t &dc, std::vector<expression>::size_type n_eq)
{
    assert(dc.size() >= n_eq * 2u);

    for (auto i = n_eq; i < dc.size() - n_eq; ++i) {
        auto &[ex, deps] = dc[i];

        if (std::holds_alternative<number>(ex.value())) {
            ex = num_identity(ex);

            // NOTE: num_identity() is a function without
            // hidden dependencies.
            deps.clear();
        }
    }
}

// Helper to transform x**y -> exp(y*log(x)), if y is not a number.
std::vector<expression> pow_to_explog(const std::vector<expression> &v_ex)
{
    void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    const auto tfunc = [](const expression &ex) {
        const auto &f = std::get<func>(ex.value());
        const auto &args = f.args();

        if (f.extract<detail::pow_impl>() != nullptr && !std::holds_alternative<number>(args[1].value())) {
            // The function is a pow() and the exponent is not a number: transform x**y -> exp(y*log(x)).
            //
            // NOTE: do not call directly log(new_args[0]) in order to avoid constant folding when the base
            // is a number. For instance, if we have pow(2_dbl, par[0]), then we would end up computing
            // log(2) in double precision. This would result in an inaccurate result if the fp type
            // or precision in use during integration is higher than double.
            // NOTE: because the exponent is not a number, no other constant folding should take
            // place here.
            return exp(args[1] * expression{func{detail::log_impl(args[0])}});
        } else {
            // The function is not a pow(), or it is a pow() whose exponent is a number. Return
            // it unchanged.
            return ex;
        }
    };

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(detail::ex_traverse_transform_nodes(func_map, sargs_map, e, {}, tfunc));
    }

    return retval;
}

} // namespace

} // namespace detail

// Taylor decomposition from lhs and rhs
// of a system of equations.
std::pair<taylor_dc_t, std::vector<std::uint32_t>>
taylor_decompose_sys(const std::vector<std::pair<expression, expression>> &sys_,
                     const std::vector<expression> &sv_funcs_)
{
    // Cache the number of equations/variables.
    const auto n_eq = sys_.size();

    // Create the map for renaming the variables to u_i.
    // The renaming will be done following the order of the lhs
    // variables.
    std::unordered_map<std::string, std::string> repl_map;
    for (decltype(sys_.size()) i = 0; i < sys_.size(); ++i) {
        [[maybe_unused]] const auto eres
            = repl_map.emplace(std::get<variable>(sys_[i].first.value()).name(), fmt::format("u_{}", i));
        assert(eres.second);
    }

    // Store in a single vector of expressions both the rhs and the sv_funcs.
    std::vector<expression> all_ex;
    all_ex.reserve(sys_.size());
    std::ranges::transform(sys_, std::back_inserter(all_ex), &std::pair<expression, expression>::second);
    all_ex.insert(all_ex.end(), sv_funcs_.begin(), sv_funcs_.end());

    // Transform x**y -> exp(y*log(x)), if y is not a number.
    all_ex = detail::pow_to_explog(all_ex);

    // Transform sums into subs.
    all_ex = detail::sum_to_sub(all_ex);

    // Split sums.
    all_ex = detail::split_sums_for_decompose(all_ex);

    // Transform sums into sum_sqs if possible.
    all_ex = detail::sums_to_sum_sqs_for_decompose(all_ex);

    // Transform prods into divs.
    all_ex = detail::prod_to_div_taylor_diff(all_ex);

    // Split prods.
    // NOTE: split must be 2 here as the Taylor diff formulae require
    // binary multiplications.
    all_ex = detail::split_prods_for_decompose(all_ex, 2);

#if !defined(NDEBUG)

    // Save copies for checking in debug mode.
    const auto sys_rhs_verify = std::vector(all_ex.data(), all_ex.data() + sys_.size());
    const auto sv_funcs_verify = std::vector(all_ex.data() + sys_.size(), all_ex.data() + all_ex.size());

#endif

    // Rename the variables.
    all_ex = rename_variables(all_ex, repl_map);

    // Init the decomposition. It begins with a list
    // of the original lhs variables of the system.
    taylor_dc_t u_vars_defs;
    u_vars_defs.reserve(sys_.size());
    for (const auto &[lhs, rhs] : sys_) {
        u_vars_defs.emplace_back(variable{std::get<variable>(lhs.value()).name()}, std::vector<std::uint32_t>{});
    }

    // Prepare the outputs vector.
    taylor_dc_t outs;
    outs.reserve(n_eq);

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition on the equations.
    detail::void_ptr_map<const taylor_dc_t::size_type> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;
    for (std::vector<expression>::size_type i = 0; i < n_eq; ++i) {
        const auto &ex = all_ex[i];

        // Decompose the current equation.
        if (const auto dres = detail::taylor_decompose(func_map, sargs_map, ex, u_vars_defs)) {
            // NOTE: if the equation was decomposed
            // (that is, it is not constant or a single variable),
            // then the output is a u variable.
            // NOTE: all functions are forced to return dres != 0
            // in the func API, so the only entities that
            // can return dres == 0 are const/params or
            // variables.
            assert(std::holds_alternative<func>(ex.value()));
            outs.emplace_back(fmt::format("u_{}", *dres), std::vector<std::uint32_t>{});
        } else {
            assert(!std::holds_alternative<func>(ex.value()));
            outs.emplace_back(ex, std::vector<std::uint32_t>{});
        }
    }

    // Decompose sv_funcs, and write into sv_funcs_dc the index
    // of the u variable which each sv_func corresponds to.
    std::vector<std::uint32_t> sv_funcs_dc;
    sv_funcs_dc.reserve(sv_funcs_.size());
    for (std::vector<expression>::size_type i = n_eq; i < all_ex.size(); ++i) {
        const auto &sv_ex = all_ex[i];

        if (const auto *const var_ptr = std::get_if<variable>(&sv_ex.value())) {
            // The current sv_func is a variable, add its index to sv_funcs_dc.
            sv_funcs_dc.push_back(detail::uname_to_index(var_ptr->name()));
        } else if (const auto dres = detail::taylor_decompose(func_map, sargs_map, sv_ex, u_vars_defs)) {
            // The sv_func was decomposed, add to sv_funcs_dc
            // the index of the u variable which represents
            // the result of the decomposition.
            assert(std::holds_alternative<func>(sv_ex.value()));
            sv_funcs_dc.push_back(boost::numeric_cast<std::uint32_t>(*dres));
        } else [[unlikely]] {                                     // LCOV_EXCL_LINE
            assert(!std::holds_alternative<func>(sv_ex.value())); // LCOV_EXCL_LINE
            // The sv_func was not decomposed, meaning it's a const/param.
            throw std::invalid_argument(
                "The extra functions in a Taylor decomposition cannot be constants or parameters");
        }
    }
    assert(sv_funcs_dc.size() == sv_funcs_.size());

    // Append the definitions of the outputs.
    u_vars_defs.insert(u_vars_defs.end(), outs.begin(), outs.end());

    detail::get_logger()->trace("Taylor decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)
    // Verify the decomposition.
    detail::verify_taylor_dec(sys_rhs_verify, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, sv_funcs_verify, u_vars_defs, n_eq);
#endif

    // Simplify the decomposition.
    // NOTE: n_eq is implicitly converted to taylor_dc_t::size_type here. This is fine, as
    // the size of the Taylor decomposition is always > n_eq anyway.
    std::tie(u_vars_defs, sv_funcs_dc) = detail::taylor_decompose_cse(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(sys_rhs_verify, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, sv_funcs_verify, u_vars_defs, n_eq);
#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: n_eq is implicitly converted to taylor_dc_t::size_type here. This is fine, as
    // the size of the Taylor decomposition is always > n_eq anyway.
    std::tie(u_vars_defs, sv_funcs_dc) = detail::taylor_sort_dc(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the reordered decomposition.
    detail::verify_taylor_dec(sys_rhs_verify, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, sv_funcs_verify, u_vars_defs, n_eq);
#endif

    // Replace any number subexpression with an identity function.
    detail::taylor_decompose_replace_numbers(u_vars_defs, n_eq);

    return std::make_pair(std::move(u_vars_defs), std::move(sv_funcs_dc));
}

namespace detail
{

// Add a function for computing the dense output
// via polynomial evaluation.
void taylor_add_d_out_function(llvm_state &s, llvm::Type *fp_scal_t, std::uint32_t n_eq, std::uint32_t order,
                               std::uint32_t batch_size, bool high_accuracy, bool external_linkage)
{
    // LCOV_EXCL_START
    assert(n_eq > 0u);
    assert(order > 0u);
    assert(batch_size > 0u);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the external type corresponding to fp_scal_t.
    auto *ext_fp_scal_t = make_external_llvm_type(fp_scal_t);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The function arguments:
    // - the output pointer (read/write, used also for accumulation),
    // - the pointer to the Taylor coefficients (read-only),
    // - the pointer to the h values (read-only).
    // No overlap is allowed. All pointers are external.
    const std::vector<llvm::Type *> fargs(3, llvm::PointerType::getUnqual(ext_fp_scal_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm_func_create(ft, external_linkage ? llvm::Function::ExternalLinkage : llvm::Function::PrivateLinkage,
                               "d_out_f", &s.module());

    // Set the names/attributes of the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *tc_ptr = f->args().begin() + 1;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *h_ptr = f->args().begin() + 2;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);
    h_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Load the value of h.
    auto *h = ext_load_vector_from_memory(s, fp_scal_t, h_ptr, batch_size);

    if (high_accuracy) {
        auto *vector_t = make_vector_type(fp_scal_t, batch_size);

        // Create the array for storing the running compensations.
        auto *array_type = llvm::ArrayType::get(vector_t, n_eq);
        auto *comp_arr = builder.CreateInBoundsGEP(array_type, builder.CreateAlloca(array_type),
                                                   {builder.getInt32(0), builder.getInt32(0)});

        // Start by writing into out_ptr the zero-order coefficients
        // and by filling with zeroes the running compensations.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptr. The index is:
            // batch_size * (order + 1u) * cur_var_idx.
            auto *tc_idx = builder.CreateMul(builder.getInt32(batch_size * (order + 1u)), cur_var_idx);
            auto *tc = ext_load_vector_from_memory(
                s, fp_scal_t, builder.CreateInBoundsGEP(ext_fp_scal_t, tc_ptr, tc_idx), batch_size);

            // Store it in out_ptr. The index is:
            // batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
            ext_store_vector_to_memory(s, builder.CreateInBoundsGEP(ext_fp_scal_t, out_ptr, out_idx), tc);

            // Zero-init the element in comp_arr.
            builder.CreateStore(llvm_constantfp(s, vector_t, 0.),
                                builder.CreateInBoundsGEP(vector_t, comp_arr, cur_var_idx));
        });

        // Init the running updater for the powers of h.
        auto *cur_h = builder.CreateAlloca(vector_t);
        builder.CreateStore(h, cur_h);

        // Run the evaluation.
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(order + 1u), [&](llvm::Value *cur_order) {
            // Load the current power of h.
            auto *cur_h_val = builder.CreateLoad(vector_t, cur_h);

            llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                // Load the coefficient from tc_ptr. The index is:
                // batch_size * (order + 1u) * cur_var_idx + batch_size * cur_order.
                auto *tc_idx
                    = builder.CreateAdd(builder.CreateMul(builder.getInt32(batch_size * (order + 1u)), cur_var_idx),
                                        builder.CreateMul(builder.getInt32(batch_size), cur_order));
                auto *cf = ext_load_vector_from_memory(
                    s, fp_scal_t, builder.CreateInBoundsGEP(ext_fp_scal_t, tc_ptr, tc_idx), batch_size);
                auto *tmp = llvm_fmul(s, cf, cur_h_val);

                // Compute the quantities for the compensation.
                auto *comp_ptr = builder.CreateInBoundsGEP(vector_t, comp_arr, cur_var_idx);
                auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
                auto *res_ptr = builder.CreateInBoundsGEP(ext_fp_scal_t, out_ptr, out_idx);
                auto *y = llvm_fsub(s, tmp, builder.CreateLoad(vector_t, comp_ptr));
                auto *cur_res = ext_load_vector_from_memory(s, fp_scal_t, res_ptr, batch_size);
                auto *t = llvm_fadd(s, cur_res, y);

                // Update the compensation and the return value.
                builder.CreateStore(llvm_fsub(s, llvm_fsub(s, t, cur_res), y), comp_ptr);
                ext_store_vector_to_memory(s, res_ptr, t);
            });

            // Update the value of h.
            builder.CreateStore(llvm_fmul(s, cur_h_val, h), cur_h);
        });
    } else {
        // Start by writing into out_ptr the coefficients of the highest-degree
        // monomial in each polynomial.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptr. The index is:
            // batch_size * (order + 1u) * cur_var_idx + batch_size * order.
            auto *tc_idx
                = builder.CreateAdd(builder.CreateMul(builder.getInt32(batch_size * (order + 1u)), cur_var_idx),
                                    builder.getInt32(batch_size * order));
            auto *tc = ext_load_vector_from_memory(
                s, fp_scal_t, builder.CreateInBoundsGEP(ext_fp_scal_t, tc_ptr, tc_idx), batch_size);

            // Store it in out_ptr. The index is:
            // batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
            ext_store_vector_to_memory(s, builder.CreateInBoundsGEP(ext_fp_scal_t, out_ptr, out_idx), tc);
        });

        // Now let's run the Horner scheme.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                      [&](llvm::Value *cur_order) {
                          llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                              // Load the current Taylor coefficient from tc_ptr.
                              // NOTE: we are loading the coefficients backwards wrt the order, hence
                              // we specify order - cur_order.
                              // NOTE: the index is:
                              // batch_size * (order + 1u) * cur_var_idx + batch_size * (order - cur_order).
                              auto *tc_idx = builder.CreateAdd(
                                  builder.CreateMul(builder.getInt32(batch_size * (order + 1u)), cur_var_idx),
                                  builder.CreateMul(builder.getInt32(batch_size),
                                                    builder.CreateSub(builder.getInt32(order), cur_order)));
                              auto *tc = ext_load_vector_from_memory(
                                  s, fp_scal_t, builder.CreateInBoundsGEP(ext_fp_scal_t, tc_ptr, tc_idx), batch_size);

                              // Accumulate in out_ptr. The index is:
                              // batch_size * cur_var_idx.
                              auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
                              auto *out_p = builder.CreateInBoundsGEP(ext_fp_scal_t, out_ptr, out_idx);
                              auto *cur_out = ext_load_vector_from_memory(s, fp_scal_t, out_p, batch_size);
                              ext_store_vector_to_memory(s, out_p, llvm_fadd(s, tc, llvm_fmul(s, cur_out, h)));
                          });
                      });
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);
}

namespace
{

// NOTE: this is the worker function type which computes the
// Taylor derivatives for a subrange in a block.
// A block consists of ncalls invocations of the same
// Taylor derivative function with different arguments.
// Workers are created on the LLVM side when parallel mode is
// active.
//
// [begin, end) is a subrange of [0, ncalls). tape_ptr
// is a pointer to the tape of derivatives, par_ptr and
// time_ptr are pointers to the arrays of parameter value(s)
// and time value(s). order is the desired Taylor order for
// the computation of the derivatives.
using block_worker_f = void (*)(std::uint32_t begin, std::uint32_t end, void *tape_ptr, const void *par_ptr,
                                const void *time_ptr, std::uint32_t order) noexcept;

} // namespace

} // namespace detail

HEYOKA_END_NAMESPACE

// This function computes the Taylor derivatives for a segment in parallel mode. It is invoked
// from LLVM after the creation of the worker functions that compute the Taylor derivatives
// for a subrange in a block.
//
// worker_arr is the array of worker functions for the computations of the derivatives in the block
// subranges, ncalls_arr is an array containing the number of times each function in
// worker_arr must be called. Both worker_arr and ncalls_arr are arrays of size nfuncs.
// tape/par/time_ptr are pointers to the tape/parameter/time values. order is the desired Taylor order for
// the computation of the derivatives.
extern "C" HEYOKA_DLL_PUBLIC void heyoka_taylor_cm_par_segment(const heyoka::detail::block_worker_f *worker_arr,
                                                               const std::uint32_t *ncalls_arr, std::uint32_t nfuncs,
                                                               void *tape_ptr, const void *par_ptr,
                                                               const void *time_ptr, std::uint32_t order) noexcept
{
    try {
        heyoka::detail::tbb_isolated_parallel_for(
            oneapi::tbb::blocked_range<std::uint32_t>(0, nfuncs),
            [ncalls_arr, worker_arr, tape_ptr, par_ptr, time_ptr, order](const auto &func_range) {
                for (auto f_idx = func_range.begin(); f_idx != func_range.end(); ++f_idx) {
                    const auto cur_ncalls = ncalls_arr[f_idx];
                    auto *cur_f = worker_arr[f_idx];

                    heyoka::detail::tbb_isolated_parallel_for(
                        oneapi::tbb::blocked_range<std::uint32_t>(0, cur_ncalls),
                        [cur_f, tape_ptr, par_ptr, time_ptr, order](const auto &call_range) {
                            cur_f(call_range.begin(), call_range.end(), tape_ptr, par_ptr, time_ptr, order);
                        });
                }
            });
        // LCOV_EXCL_START
    } catch (const std::exception &ex) {
        heyoka::detail::get_logger()->critical("Exception caught in heyoka_taylor_cm_par_segment(): {}", ex.what());
    } catch (...) {
        heyoka::detail::get_logger()->critical("Exception caught in heyoka_taylor_cm_par_segment()");
    }
    // LCOV_EXCL_STOP
}
