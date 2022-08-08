// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <exception>
#include <initializer_list>
#include <ios>
#include <limits>
#include <locale>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <fmt/format.h>

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

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

// NOTE: precondition on name: must be conforming to LLVM requirements for
// function names, and must not contain "." (as we use it as a separator in
// the mangling scheme).
std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args_impl(llvm::LLVMContext &context, const std::string &name, llvm::Type *val_t,
                                  std::uint32_t n_uvars, const std::vector<std::variant<variable, number, param>> &args,
                                  std::uint32_t n_hidden_deps)
{
    assert(val_t != nullptr);
    assert(n_uvars > 0u);

    // Init the name.
    auto fname = fmt::format("heyoka.taylor_c_diff.{}.", name);

    // Init the vector of arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array (pointer to val_t),
    // - par ptr (pointer to scalar),
    // - time ptr (pointer to scalar).
    std::vector<llvm::Type *> fargs{
        llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context), llvm::PointerType::getUnqual(val_t),
        llvm::PointerType::getUnqual(val_t->getScalarType()), llvm::PointerType::getUnqual(val_t->getScalarType())};

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

                if constexpr (std::is_same_v<type, number>) {
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
    }

    // Finally, add the mangling for the floating-point type.
    fname += llvm_mangle_type(val_t);

    // Fill in the hidden dependency arguments. These are all indices.
    fargs.insert(fargs.end(), boost::numeric_cast<decltype(fargs.size())>(n_hidden_deps),
                 llvm::Type::getInt32Ty(context));

    return std::make_pair(std::move(fname), std::move(fargs));
}

namespace
{

template <typename T>
llvm::Value *taylor_codegen_numparam_num(llvm_state &s, const number &num, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, num), batch_size);
}

llvm::Value *taylor_codegen_numparam_par(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(llvm::isa<llvm::PointerType>(par_ptr->getType()));
    assert(!llvm::cast<llvm::PointerType>(par_ptr->getType())->isVectorTy());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Determine the index into the parameter array.
    // LCOV_EXCL_START
    if (p.idx() > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("Overflow detected in the computation of the index into a parameter array");
    }
    // LCOV_EXCL_STOP
    const auto arr_idx = static_cast<std::uint32_t>(p.idx() * batch_size);

    // Compute the pointer to load from.
    auto *ptr = builder.CreateInBoundsGEP(llvm::cast<llvm::PointerType>(par_ptr->getType())->getPointerElementType(),
                                          par_ptr, builder.getInt32(arr_idx));

    // Load.
    return load_vector_from_memory(builder, ptr, batch_size);
}

} // namespace

llvm::Value *taylor_codegen_numparam_dbl(llvm_state &s, const number &num, llvm::Value *, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_num<double>(s, num, batch_size);
}

llvm::Value *taylor_codegen_numparam_ldbl(llvm_state &s, const number &num, llvm::Value *, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_num<long double>(s, num, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_codegen_numparam_f128(llvm_state &s, const number &num, llvm::Value *, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_num<mppp::real128>(s, num, batch_size);
}

#endif

llvm::Value *taylor_codegen_numparam_dbl(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_par(s, p, par_ptr, batch_size);
}

llvm::Value *taylor_codegen_numparam_ldbl(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_par(s, p, par_ptr, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_codegen_numparam_f128(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_par(s, p, par_ptr, batch_size);
}

#endif

// Codegen helpers for number/param for use in the generic c_diff implementations.
llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &s, const number &, llvm::Value *n, llvm::Value *,
                                            std::uint32_t batch_size)
{
    return vector_splat(s.builder(), n, batch_size);
}

llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &s, const param &, llvm::Value *p, llvm::Value *par_ptr,
                                            std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(llvm::isa<llvm::PointerType>(par_ptr->getType()));
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the pointer into par_ptr.
    // NOTE: the overflow check is done in taylor_compute_jet().
    auto *ptr = builder.CreateInBoundsGEP(llvm::cast<llvm::PointerType>(par_ptr->getType())->getPointerElementType(),
                                          par_ptr, builder.CreateMul(p, builder.getInt32(batch_size)));

    return load_vector_from_memory(builder, ptr, batch_size);
}

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
    assert(llvm_depr_GEP_type_check(diff_arr, pointee_type(diff_arr))); // LCOV_EXCL_LINE
    auto *ptr
        = builder.CreateInBoundsGEP(pointee_type(diff_arr), diff_arr,
                                    builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx));

    return builder.CreateLoad(pointee_type(diff_arr), ptr);
}

// Store the value val as the derivative of order 'order' of the u variable u_idx
// into the array of Taylor derivatives diff_arr. n_uvars is the total number of u variables.
void taylor_c_store_diff(llvm_state &s, llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                         llvm::Value *u_idx, llvm::Value *val)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    assert(llvm_depr_GEP_type_check(diff_arr, pointee_type(diff_arr))); // LCOV_EXCL_LINE
    auto *ptr
        = builder.CreateInBoundsGEP(pointee_type(diff_arr), diff_arr,
                                    builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx));

    builder.CreateStore(val, ptr);
}

namespace
{

// Simplify a Taylor decomposition by removing
// common subexpressions.
// NOTE: the hidden deps are not considered for CSE
// purposes, only the actual subexpressions.
taylor_dc_t taylor_decompose_cse(taylor_dc_t &v_ex, std::vector<std::uint32_t> &sv_funcs_dc,
                                 taylor_dc_t::size_type n_eq)
{
    using idx_t = taylor_dc_t::size_type;

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Cache the original size for logging later.
    const auto orig_size = v_ex.size();

    // A Taylor decomposition is supposed
    // to have n_eq variables at the beginning,
    // n_eq variables at the end and possibly
    // extra variables in the middle.
    assert(v_ex.size() >= n_eq * 2u);

    // Init the return value.
    taylor_dc_t retval;

    // expression -> idx map. This will end up containing
    // all the unique expressions from v_ex, and it will
    // map them to their indices in retval (which will
    // in general differ from their indices in v_ex).
    std::unordered_map<expression, idx_t> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // The first n_eq definitions are just renaming
    // of the state variables into u variables.
    for (idx_t i = 0; i < n_eq; ++i) {
        assert(std::holds_alternative<variable>(v_ex[i].first.value()));
        // NOTE: no hidden deps allowed here.
        assert(v_ex[i].second.empty());
        retval.push_back(std::move(v_ex[i]));

        // NOTE: the u vars that correspond to state
        // variables are never simplified,
        // thus map them onto themselves.
        [[maybe_unused]] const auto res = uvars_rename.emplace(fmt::format("u_{}", i), fmt::format("u_{}", i));
        assert(res.second);
    }

    // Handle the u variables which do not correspond to state variables.
    for (auto i = n_eq; i < v_ex.size() - n_eq; ++i) {
        auto &[ex, deps] = v_ex[i];

        // Rename the u variables in ex.
        rename_variables(ex, uvars_rename);

        if (auto it = ex_map.find(ex); it == ex_map.end()) {
            // This is the first occurrence of ex in the
            // decomposition. Add it to retval.
            retval.emplace_back(ex, std::move(deps));

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
            assert(res.second);
        }
    }

    // Handle the derivatives of the state variables at the
    // end of the decomposition. We just need to ensure that
    // the u variables in their definitions are renamed with
    // the new indices.
    for (auto i = v_ex.size() - n_eq; i < v_ex.size(); ++i) {
        auto &[ex, deps] = v_ex[i];

        // NOTE: here we expect only vars, numbers or params,
        // and no hidden dependencies.
        assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
               || std::holds_alternative<param>(ex.value()));
        assert(deps.empty());

        rename_variables(ex, uvars_rename);

        retval.emplace_back(std::move(ex), std::move(deps));
    }

    // Re-adjust all indices in the hidden dependencies in order to account
    // for the renaming of the uvars.
    for (auto &[_, deps] : retval) {
        for (auto &idx : deps) {
            auto it = uvars_rename.find(fmt::format("u_{}", idx));
            assert(it != uvars_rename.end());
            idx = uname_to_index(it->second);
        }
    }

    // Same for the indices in sv_funcs_dc.
    for (auto &idx : sv_funcs_dc) {
        auto it = uvars_rename.find(fmt::format("u_{}", idx));
        assert(it != uvars_rename.end());
        idx = uname_to_index(it->second);
    }

    get_logger()->debug("Taylor CSE reduced decomposition size from {} to {}", orig_size, retval.size());
    get_logger()->trace("Taylor CSE runtime: {}", sw);

    return retval;
}

// Perform a topological sort on a graph representation
// of a Taylor decomposition. This can improve performance
// by grouping together operations that can be performed in parallel,
// and it also makes compact mode much more effective by creating
// clusters of subexpressions whose derivatives can be computed in
// parallel.
// NOTE: the original decomposition dc is already topologically sorted,
// in the sense that the definitions of the u variables are already
// ordered according to dependency. However, because the original decomposition
// comes from a depth-first search, it has the tendency to group together
// expressions which are dependent on each other. By doing another topological
// sort, this time based on breadth-first search, we determine another valid
// sorting in which independent operations tend to be clustered together.
auto taylor_sort_dc(taylor_dc_t &dc, std::vector<std::uint32_t> &sv_funcs_dc, taylor_dc_t::size_type n_eq)
{
    // A Taylor decomposition is supposed
    // to have n_eq variables at the beginning,
    // n_eq variables at the end and possibly
    // extra variables in the middle
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
    for (decltype(n_eq) i = 0; i < n_eq; ++i) {
        auto v = boost::add_vertex(g);

        // Add a dependency on the root node.
        boost::add_edge(root_v, v, g);
    }

    // Add the rest of the u variables.
    for (decltype(n_eq) i = n_eq; i < dc.size() - n_eq; ++i) {
        auto v = boost::add_vertex(g);

        // Fetch the list of variables in the current expression.
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
    // NOTE: the u vars that correspond to state
    // variables were inserted into v_idx in the original
    // order, thus they are not re-sorted and they do not
    // need renaming.
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

    // Do the remap for the definitions of the u variables, the
    // derivatives and the hidden deps.
    for (auto *it = dc.data() + n_eq; it != dc.data() + dc.size(); ++it) {
        // Remap the expression.
        rename_variables(it->first, remap);

        // Remap the hidden dependencies.
        for (auto &idx : it->second) {
            auto it_remap = remap.find(fmt::format("u_{}", idx));
            assert(it_remap != remap.end());
            idx = uname_to_index(it_remap->second);
        }
    }

    // Do the remap for sv_funcs.
    for (auto &idx : sv_funcs_dc) {
        auto it_remap = remap.find(fmt::format("u_{}", idx));
        assert(it_remap != remap.end());
        idx = uname_to_index(it_remap->second);
    }

    // Reorder the decomposition.
    taylor_dc_t retval;
    retval.reserve(v_idx.size());
    for (auto idx : v_idx) {
        retval.push_back(std::move(dc[idx]));
    }

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
    // must be functions whose arguments
    // are either variables in the u_n form,
    // where n < i, or numbers/params.
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
                    assert(false);
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
    // must be either variables in the u_n form, where n < i,
    // or numbers/params.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        std::visit(
            [i](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    assert(v.name().rfind("u_", 0) == 0);
                    assert(uname_to_index(v.name()) < i);
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

} // namespace

} // namespace detail

// Taylor decomposition with automatic deduction
// of variables.
// NOTE: when dealing with functions with hidden deps,
// we should consider avoiding adding hidden deps if the
// function argument(s) is a number/param: the hidden deps
// won't be used for the computation of the derivatives
// and thus they can be optimised out. Note that a straightforward
// implementation of this idea this will only work when the argument
// is a number/param, not when, e.g., the argument is par[0] + par[1] - in
// order to simplify this out, it should be recognized that the definition
// of a u variable depends only on numbers/params.
std::pair<taylor_dc_t, std::vector<std::uint32_t>> taylor_decompose(const std::vector<expression> &v_ex_,
                                                                    const std::vector<expression> &sv_funcs_)
{
    // Need to operate on copies due to in-place mutation
    // via rename_variables().
    // NOTE: this is suboptimal, as expressions which are shared
    // across different elements of v_ex/sv_funcs will be not shared any more
    // after the copy.
    auto v_ex = detail::copy(v_ex_);
    auto sv_funcs = detail::copy(sv_funcs_);

    if (v_ex.empty()) {
        throw std::invalid_argument("Cannot decompose a system of zero equations");
    }

    // Determine the variables in the system of equations.
    std::set<std::string> vars;
    for (const auto &ex : v_ex) {
        for (const auto &var : get_variables(ex)) {
            vars.emplace(var);
        }
    }
    if (vars.size() != v_ex.size()) {
        throw std::invalid_argument(fmt::format(
            "The number of deduced variables for a Taylor decomposition ({}) differs from the number of equations ({})",
            vars.size(), v_ex.size()));
    }

    // Check that the expressions in sv_funcs contain only
    // state variables.
    for (const auto &ex : sv_funcs) {
        for (const auto &var : get_variables(ex)) {
            if (vars.find(var) == vars.end()) {
                throw std::invalid_argument(
                    fmt::format("The extra functions in a Taylor decomposition contain the variable '{}', "
                                "which is not a state variable",
                                var));
            }
        }
    }

    // Cache the number of equations/variables
    // for later use.
    const auto n_eq = v_ex.size();

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

#if !defined(NDEBUG)

    // Store a copy of the original system and
    // sv_funcs for checking later.
    auto orig_v_ex = detail::copy(v_ex);
    auto orig_sv_funcs = detail::copy(sv_funcs);

#endif

    // Rename the variables in the original equations.
    for (auto &ex : v_ex) {
        rename_variables(ex, repl_map);
    }

    // Rename the variables in sv_funcs.
    for (auto &ex : sv_funcs) {
        rename_variables(ex, repl_map);
    }

    // Init the decomposition. It begins with a list
    // of the original variables of the system.
    taylor_dc_t u_vars_defs;
    u_vars_defs.reserve(vars.size());
    for (const auto &var : vars) {
        u_vars_defs.emplace_back(variable{var}, std::vector<std::uint32_t>{});
    }

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition on each equation.
    for (auto &ex : v_ex) {
        // Decompose the current equation.
        if (const auto dres = taylor_decompose(ex, u_vars_defs)) {
            // NOTE: if the equation was decomposed
            // (that is, it is not constant or a single variable),
            // we have to update the original definition
            // of the equation in v_ex
            // so that it points to the u variable
            // that now represents it.
            // NOTE: all functions are forced to return dres != 0
            // in the func API, so the only entities that
            // can return dres == 0 are const/params or
            // variables.
            ex = expression{fmt::format("u_{}", dres)};
        } else {
            assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
                   || std::holds_alternative<param>(ex.value()));
        }
    }

    // Decompose sv_funcs, and write into sv_funcs_dc the index
    // of the u variable which each sv_func corresponds to.
    std::vector<std::uint32_t> sv_funcs_dc;
    for (auto &sv_ex : sv_funcs) {
        if (const auto *var_ptr = std::get_if<variable>(&sv_ex.value())) {
            // The current sv_func is a variable, add its index to sv_funcs_dc.
            sv_funcs_dc.push_back(detail::uname_to_index(var_ptr->name()));
        } else if (const auto dres = taylor_decompose(sv_ex, u_vars_defs)) {
            // The sv_func was decomposed, add to sv_funcs_dc
            // the index of the u variable which represents
            // the result of the decomposition.
            sv_funcs_dc.push_back(boost::numeric_cast<std::uint32_t>(dres));
        } else {
            // The sv_func was not decomposed, meaning it's a const/param.
            throw std::invalid_argument(
                "The extra functions in a Taylor decomposition cannot be constants or parameters");
        }
    }
    assert(sv_funcs_dc.size() == sv_funcs.size());

    // Append the (possibly updated) definitions of the diff equations
    // in terms of u variables.
    for (auto &ex : v_ex) {
        u_vars_defs.emplace_back(std::move(ex), std::vector<std::uint32_t>{});
    }

    detail::get_logger()->trace("Taylor decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)
    // Verify the decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    // Simplify the decomposition.
    // NOTE: n_eq is implicitly converted to taylor_dc_t::size_type here. This is fine, as
    // the size of the Taylor decomposition is always > n_eq anyway.
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: n_eq is implicitly converted to taylor_dc_t::size_type here. This is fine, as
    // the size of the Taylor decomposition is always > n_eq anyway.
    u_vars_defs = detail::taylor_sort_dc(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the reordered decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    return std::make_pair(std::move(u_vars_defs), std::move(sv_funcs_dc));
}

// Taylor decomposition from lhs and rhs
// of a system of equations.
std::pair<taylor_dc_t, std::vector<std::uint32_t>>
taylor_decompose(const std::vector<std::pair<expression, expression>> &sys_, const std::vector<expression> &sv_funcs_)
{
    // Need to operate on copies due to in-place mutation
    // via rename_variables().
    // NOTE: this is suboptimal, as expressions which are shared
    // across different elements of sys/sv_funcs will be not shared any more
    // after the copy.
    auto sys = detail::copy(sys_);
    auto sv_funcs = detail::copy(sv_funcs_);

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
    // need to appear in the rhs: that is, not all variables
    // need to appear in the ODEs.

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
                        lhs_vars.push_back(v.name());
                    } else {
                        // Duplicate, error out.
                        throw std::invalid_argument(
                            fmt::format("Error in the Taylor decomposition of a system of equations: the variable '{}' "
                                        "appears in the left-hand side twice",
                                        v.name()));
                    }
                } else {
                    throw std::invalid_argument(
                        fmt::format("Error in the Taylor decomposition of a system of equations: the "
                                    "left-hand side contains the expression '{}', which is not a variable",
                                    lhs));
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
            throw std::invalid_argument(
                fmt::format("Error in the Taylor decomposition of a system of equations: the variable '{}' "
                            "appears in the right-hand side but not in the left-hand side",
                            var));
        }
    }

    // Check that the expressions in sv_funcs contain only
    // state variables.
    for (const auto &ex : sv_funcs) {
        for (const auto &var : get_variables(ex)) {
            if (lhs_vars_set.find(var) == lhs_vars_set.end()) {
                throw std::invalid_argument(
                    fmt::format("The extra functions in a Taylor decomposition contain the variable '{}', "
                                "which is not a state variable",
                                var));
            }
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
        [[maybe_unused]] const auto eres = repl_map.emplace(lhs_vars[i], fmt::format("u_{}", i));
        assert(eres.second);
    }

#if !defined(NDEBUG)

    // Store a copy of the original rhs and sv_funcs
    // for checking later.
    std::vector<expression> orig_rhs;
    orig_rhs.reserve(sys.size());
    for (const auto &[_, rhs_ex] : sys) {
        orig_rhs.push_back(copy(rhs_ex));
    }

    auto orig_sv_funcs = detail::copy(sv_funcs);

#endif

    // Rename the variables in the original equations.
    for (auto &[_, rhs_ex] : sys) {
        rename_variables(rhs_ex, repl_map);
    }

    // Rename the variables in sv_funcs.
    for (auto &ex : sv_funcs) {
        rename_variables(ex, repl_map);
    }

    // Init the decomposition. It begins with a list
    // of the original lhs variables of the system.
    taylor_dc_t u_vars_defs;
    u_vars_defs.reserve(lhs_vars.size());
    for (const auto &var : lhs_vars) {
        u_vars_defs.emplace_back(variable{var}, std::vector<std::uint32_t>{});
    }

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Run the decomposition on each equation.
    for (auto &[_, ex] : sys) {
        // Decompose the current equation.
        if (const auto dres = taylor_decompose(ex, u_vars_defs)) {
            // NOTE: if the equation was decomposed
            // (that is, it is not constant or a single variable),
            // we have to update the original definition
            // of the equation in sys
            // so that it points to the u variable
            // that now represents it.
            // NOTE: all functions are forced to return dres != 0
            // in the func API, so the only entities that
            // can return dres == 0 are const/params or
            // variables.
            ex = expression{fmt::format("u_{}", dres)};
        } else {
            assert(std::holds_alternative<variable>(ex.value()) || std::holds_alternative<number>(ex.value())
                   || std::holds_alternative<param>(ex.value()));
        }
    }

    // Decompose sv_funcs, and write into sv_funcs_dc the index
    // of the u variable which each sv_func corresponds to.
    std::vector<std::uint32_t> sv_funcs_dc;
    for (auto &sv_ex : sv_funcs) {
        if (auto *const var_ptr = std::get_if<variable>(&sv_ex.value())) {
            // The current sv_func is a variable, add its index to sv_funcs_dc.
            sv_funcs_dc.push_back(detail::uname_to_index(var_ptr->name()));
        } else if (const auto dres = taylor_decompose(sv_ex, u_vars_defs)) {
            // The sv_func was decomposed, add to sv_funcs_dc
            // the index of the u variable which represents
            // the result of the decomposition.
            sv_funcs_dc.push_back(boost::numeric_cast<std::uint32_t>(dres));
        } else {
            // The sv_func was not decomposed, meaning it's a const/param.
            throw std::invalid_argument(
                "The extra functions in a Taylor decomposition cannot be constants or parameters");
        }
    }
    assert(sv_funcs_dc.size() == sv_funcs.size());

    // Append the (possibly updated) definitions of the diff equations
    // in terms of u variables.
    for (auto &[_, rhs] : sys) {
        u_vars_defs.emplace_back(std::move(rhs), std::vector<std::uint32_t>{});
    }

    detail::get_logger()->trace("Taylor decomposition construction runtime: {}", sw);

#if !defined(NDEBUG)
    // Verify the decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    // Simplify the decomposition.
    // NOTE: n_eq is implicitly converted to taylor_dc_t::size_type here. This is fine, as
    // the size of the Taylor decomposition is always > n_eq anyway.
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    // Run the breadth-first topological sort on the decomposition.
    // NOTE: n_eq is implicitly converted to taylor_dc_t::size_type here. This is fine, as
    // the size of the Taylor decomposition is always > n_eq anyway.
    u_vars_defs = detail::taylor_sort_dc(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the reordered decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    return std::make_pair(std::move(u_vars_defs), std::move(sv_funcs_dc));
}

namespace detail
{

namespace
{

// Implementation of the streaming operator for the scalar integrators.
template <typename T>
std::ostream &taylor_adaptive_stream_impl(std::ostream &os, const taylor_adaptive<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);
    oss << std::boolalpha;

    oss << "Tolerance               : " << ta.get_tol() << '\n';
    oss << "High accuracy           : " << ta.get_high_accuracy() << '\n';
    oss << "Compact mode            : " << ta.get_compact_mode() << '\n';
    oss << "Taylor order            : " << ta.get_order() << '\n';
    oss << "Dimension               : " << ta.get_dim() << '\n';
    oss << "Time                    : " << ta.get_time() << '\n';
    oss << "State                   : [";
    for (decltype(ta.get_state().size()) i = 0; i < ta.get_state().size(); ++i) {
        oss << ta.get_state()[i];
        if (i != ta.get_state().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";

    if (!ta.get_pars().empty()) {
        oss << "Parameters              : [";
        for (decltype(ta.get_pars().size()) i = 0; i < ta.get_pars().size(); ++i) {
            oss << ta.get_pars()[i];
            if (i != ta.get_pars().size() - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";
    }

    if (ta.with_events()) {
        if (!ta.get_t_events().empty()) {
            oss << "N of terminal events    : " << ta.get_t_events().size() << '\n';
        }

        if (!ta.get_nt_events().empty()) {
            oss << "N of non-terminal events: " << ta.get_nt_events().size() << '\n';
        }
    }

    return os << oss.str();
}

// Implementation of the streaming operator for the batch integrators.
template <typename T>
std::ostream &taylor_adaptive_batch_stream_impl(std::ostream &os, const taylor_adaptive_batch<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);
    oss << std::boolalpha;

    oss << "Tolerance               : " << ta.get_tol() << '\n';
    oss << "High accuracy           : " << ta.get_high_accuracy() << '\n';
    oss << "Compact mode            : " << ta.get_compact_mode() << '\n';
    oss << "Taylor order            : " << ta.get_order() << '\n';
    oss << "Dimension               : " << ta.get_dim() << '\n';
    oss << "Batch size              : " << ta.get_batch_size() << '\n';
    oss << "Time                    : [";
    for (decltype(ta.get_time().size()) i = 0; i < ta.get_time().size(); ++i) {
        oss << ta.get_time()[i];
        if (i != ta.get_time().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";
    oss << "State                   : [";
    for (decltype(ta.get_state().size()) i = 0; i < ta.get_state().size(); ++i) {
        oss << ta.get_state()[i];
        if (i != ta.get_state().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";

    if (!ta.get_pars().empty()) {
        oss << "Parameters              : [";
        for (decltype(ta.get_pars().size()) i = 0; i < ta.get_pars().size(); ++i) {
            oss << ta.get_pars()[i];
            if (i != ta.get_pars().size() - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";
    }

    if (ta.with_events()) {
        if (!ta.get_t_events().empty()) {
            oss << "N of terminal events    : " << ta.get_t_events().size() << '\n';
        }

        if (!ta.get_nt_events().empty()) {
            oss << "N of non-terminal events: " << ta.get_nt_events().size() << '\n';
        }
    }

    return os << oss.str();
}

} // namespace

} // namespace detail

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<double> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<long double> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<mppp::real128> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

#endif

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<double> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<long double> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<mppp::real128> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

#endif

#define HEYOKA_TAYLOR_ENUM_STREAM_CASE(val)                                                                            \
    case val:                                                                                                          \
        os << #val;                                                                                                    \
        break

std::ostream &operator<<(std::ostream &os, taylor_outcome oc)
{
    switch (oc) {
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::success);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::step_limit);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::time_limit);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::err_nf_state);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::cb_stop);
        default:
            if (oc >= taylor_outcome{0}) {
                // Continuing terminal event.
                os << fmt::format("taylor_outcome::terminal_event_{} (continuing)", static_cast<std::int64_t>(oc));
            } else if (oc > taylor_outcome::success) {
                // Stopping terminal event.
                os << fmt::format("taylor_outcome::terminal_event_{} (stopping)", -static_cast<std::int64_t>(oc) - 1);
            } else {
                // Unknown value.
                os << "taylor_outcome::??";
            }
    }

    return os;
}

std::ostream &operator<<(std::ostream &os, event_direction dir)
{
    switch (dir) {
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(event_direction::any);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(event_direction::positive);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(event_direction::negative);
        default:
            // Unknown value.
            os << "event_direction::??";
    }

    return os;
}

#undef HEYOKA_TAYLOR_OUTCOME_STREAM_CASE

namespace detail
{

namespace
{

// Helper to create the callback used in the default
// constructor of a non-terminal event.
template <typename T, bool B>
auto nt_event_def_cb()
{
    if constexpr (B) {
        return [](taylor_adaptive_batch<T> &, T, int, std::uint32_t) {};
    } else {
        return [](taylor_adaptive<T> &, T, int) {};
    }
}

} // namespace

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl() : nt_event_impl(expression{}, nt_event_def_cb<T, B>())
{
}

template <typename T, bool B>
void nt_event_impl<T, B>::finalise_ctor(event_direction d)
{
    if (!callback) {
        throw std::invalid_argument("Cannot construct a non-terminal event with an empty callback");
    }

    if (d < event_direction::negative || d > event_direction::positive) {
        throw std::invalid_argument("Invalid value selected for the direction of a non-terminal event");
    }
    dir = d;
}

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl(const nt_event_impl &o) : eq(copy(o.eq)), callback(o.callback), dir(o.dir)
{
}

template <typename T, bool B>
nt_event_impl<T, B>::nt_event_impl(nt_event_impl &&) noexcept = default;

template <typename T, bool B>
nt_event_impl<T, B> &nt_event_impl<T, B>::operator=(const nt_event_impl &o)
{
    if (this != &o) {
        *this = nt_event_impl(o);
    }

    return *this;
}

template <typename T, bool B>
nt_event_impl<T, B> &nt_event_impl<T, B>::operator=(nt_event_impl &&) noexcept = default;

template <typename T, bool B>
nt_event_impl<T, B>::~nt_event_impl() = default;

template <typename T, bool B>
const expression &nt_event_impl<T, B>::get_expression() const
{
    return eq;
}

template <typename T, bool B>
const typename nt_event_impl<T, B>::callback_t &nt_event_impl<T, B>::get_callback() const
{
    return callback;
}

template <typename T, bool B>
event_direction nt_event_impl<T, B>::get_direction() const
{
    return dir;
}

template <typename T, bool B>
t_event_impl<T, B>::t_event_impl() : t_event_impl(expression{})
{
}

template <typename T, bool B>
void t_event_impl<T, B>::finalise_ctor(callback_t cb, T cd, event_direction d)
{
    using std::isfinite;

    callback = std::move(cb);

    if (!isfinite(cd)) {
        throw std::invalid_argument("Cannot set a non-finite cooldown value for a terminal event");
    }
    cooldown = cd;

    if (d < event_direction::negative || d > event_direction::positive) {
        throw std::invalid_argument("Invalid value selected for the direction of a terminal event");
    }
    dir = d;
}

template <typename T, bool B>
t_event_impl<T, B>::t_event_impl(const t_event_impl &o)
    : eq(copy(o.eq)), callback(o.callback), cooldown(o.cooldown), dir(o.dir)
{
}

template <typename T, bool B>
t_event_impl<T, B>::t_event_impl(t_event_impl &&) noexcept = default;

template <typename T, bool B>
t_event_impl<T, B> &t_event_impl<T, B>::operator=(const t_event_impl &o)
{
    if (this != &o) {
        *this = t_event_impl(o);
    }

    return *this;
}

template <typename T, bool B>
t_event_impl<T, B> &t_event_impl<T, B>::operator=(t_event_impl &&) noexcept = default;

template <typename T, bool B>
t_event_impl<T, B>::~t_event_impl() = default;

template <typename T, bool B>
const expression &t_event_impl<T, B>::get_expression() const
{
    return eq;
}

template <typename T, bool B>
const typename t_event_impl<T, B>::callback_t &t_event_impl<T, B>::get_callback() const
{
    return callback;
}

template <typename T, bool B>
event_direction t_event_impl<T, B>::get_direction() const
{
    return dir;
}

template <typename T, bool B>
T t_event_impl<T, B>::get_cooldown() const
{
    return cooldown;
}

namespace
{

// Implementation of stream insertion for the non-terminal event class.
std::ostream &nt_event_impl_stream_impl(std::ostream &os, const expression &eq, event_direction dir)
{
    os << "Event type     : non-terminal\n";
    os << "Event equation : " << eq << '\n';
    os << "Event direction: " << dir << '\n';

    return os;
}

// Implementation of stream insertion for the terminal event class.
template <typename C, typename T>
std::ostream &t_event_impl_stream_impl(std::ostream &os, const expression &eq, event_direction dir, const C &callback,
                                       const T &cooldown)
{
    os << "Event type     : terminal\n";
    os << "Event equation : " << eq << '\n';
    os << "Event direction: " << dir << '\n';
    os << "With callback  : " << (callback ? "yes" : "no") << '\n';
    os << "Cooldown       : " << (cooldown < 0 ? "auto" : fp_to_string(cooldown)) << '\n';

    return os;
}

} // namespace

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<double, false> &e)
{
    return nt_event_impl_stream_impl(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<double, true> &e)
{
    return nt_event_impl_stream_impl(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<long double, false> &e)
{
    return nt_event_impl_stream_impl(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<long double, true> &e)
{
    return nt_event_impl_stream_impl(os, e.get_expression(), e.get_direction());
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<mppp::real128, false> &e)
{
    return nt_event_impl_stream_impl(os, e.get_expression(), e.get_direction());
}

template <>
std::ostream &operator<<(std::ostream &os, const nt_event_impl<mppp::real128, true> &e)
{
    return nt_event_impl_stream_impl(os, e.get_expression(), e.get_direction());
}

#endif

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<double, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<double, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<long double, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<long double, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<mppp::real128, false> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

template <>
std::ostream &operator<<(std::ostream &os, const t_event_impl<mppp::real128, true> &e)
{
    return t_event_impl_stream_impl(os, e.get_expression(), e.get_direction(), e.get_callback(), e.get_cooldown());
}

#endif

// Explicit instantiation of the implementation classes/functions.
template class nt_event_impl<double, false>;
template class t_event_impl<double, false>;

template class nt_event_impl<double, true>;
template class t_event_impl<double, true>;

template class nt_event_impl<long double, false>;
template class t_event_impl<long double, false>;

template class nt_event_impl<long double, true>;
template class t_event_impl<long double, true>;

#if defined(HEYOKA_HAVE_REAL128)

template class nt_event_impl<mppp::real128, false>;
template class t_event_impl<mppp::real128, false>;

template class nt_event_impl<mppp::real128, true>;
template class t_event_impl<mppp::real128, true>;

#endif

// Add a function for computing the dense output
// via polynomial evaluation.
template <typename T>
void taylor_add_d_out_function(llvm_state &s, std::uint32_t n_eq, std::uint32_t order, std::uint32_t batch_size,
                               bool high_accuracy, bool external_linkage, bool optimise)
{
    // LCOV_EXCL_START
    assert(n_eq > 0u);
    assert(order > 0u);
    assert(batch_size > 0u);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();
    auto &context = s.context();

    // The function arguments:
    // - the output pointer (read/write, used also for accumulation),
    // - the pointer to the Taylor coefficients (read-only),
    // - the pointer to the h values (read-only).
    // No overlap is allowed.
    auto *fp_scal_t = to_llvm_type<T>(context);
    std::vector<llvm::Type *> fargs(3, llvm::PointerType::getUnqual(fp_scal_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(
        ft, external_linkage ? llvm::Function::ExternalLinkage : llvm::Function::InternalLinkage, "d_out_f",
        &s.module());
    // LCOV_EXCL_START
    if (f == nullptr) {
        throw std::invalid_argument(
            "Unable to create a function for the dense output in an adaptive Taylor integrator");
    }
    // LCOV_EXCL_STOP

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
    auto *h = load_vector_from_memory(builder, h_ptr, batch_size);

    if (high_accuracy) {
        auto *vector_t = make_vector_type(fp_scal_t, batch_size);

        // Create the array for storing the running compensations.
        auto array_type = llvm::ArrayType::get(vector_t, n_eq);
        auto comp_arr = builder.CreateInBoundsGEP(array_type, builder.CreateAlloca(array_type),
                                                  {builder.getInt32(0), builder.getInt32(0)});

        // Start by writing into out_ptr the zero-order coefficients
        // and by filling with zeroes the running compensations.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptr. The index is:
            // batch_size * (order + 1u) * cur_var_idx.
            auto *tc_idx = builder.CreateMul(builder.getInt32(batch_size * (order + 1u)), cur_var_idx);
            auto *tc
                = load_vector_from_memory(builder, builder.CreateInBoundsGEP(fp_scal_t, tc_ptr, tc_idx), batch_size);

            // Store it in out_ptr. The index is:
            // batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
            store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_scal_t, out_ptr, out_idx), tc);

            // Zero-init the element in comp_arr.
            builder.CreateStore(llvm::ConstantFP::get(vector_t, 0.),
                                builder.CreateInBoundsGEP(vector_t, comp_arr, {cur_var_idx}));
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
                auto *cf = load_vector_from_memory(builder, builder.CreateInBoundsGEP(fp_scal_t, tc_ptr, {tc_idx}),
                                                   batch_size);
                auto *tmp = builder.CreateFMul(cf, cur_h_val);

                // Compute the quantities for the compensation.
                auto *comp_ptr = builder.CreateInBoundsGEP(vector_t, comp_arr, cur_var_idx);
                auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
                auto *res_ptr = builder.CreateInBoundsGEP(fp_scal_t, out_ptr, out_idx);
                auto *y = builder.CreateFSub(tmp, builder.CreateLoad(vector_t, comp_ptr));
                auto *cur_res = load_vector_from_memory(builder, res_ptr, batch_size);
                auto *t = builder.CreateFAdd(cur_res, y);

                // Update the compensation and the return value.
                builder.CreateStore(builder.CreateFSub(builder.CreateFSub(t, cur_res), y), comp_ptr);
                store_vector_to_memory(builder, res_ptr, t);
            });

            // Update the value of h.
            builder.CreateStore(builder.CreateFMul(cur_h_val, h), cur_h);
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
            auto *tc
                = load_vector_from_memory(builder, builder.CreateInBoundsGEP(fp_scal_t, tc_ptr, tc_idx), batch_size);

            // Store it in out_ptr. The index is:
            // batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
            store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_scal_t, out_ptr, out_idx), tc);
        });

        // Now let's run the Horner scheme.
        llvm_loop_u32(
            s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                    // Load the current Taylor coefficient from tc_ptr.
                    // NOTE: we are loading the coefficients backwards wrt the order, hence
                    // we specify order - cur_order.
                    // NOTE: the index is:
                    // batch_size * (order + 1u) * cur_var_idx + batch_size * (order - cur_order).
                    auto *tc_idx
                        = builder.CreateAdd(builder.CreateMul(builder.getInt32(batch_size * (order + 1u)), cur_var_idx),
                                            builder.CreateMul(builder.getInt32(batch_size),
                                                              builder.CreateSub(builder.getInt32(order), cur_order)));
                    auto *tc = load_vector_from_memory(builder, builder.CreateInBoundsGEP(fp_scal_t, tc_ptr, tc_idx),
                                                       batch_size);

                    // Accumulate in out_ptr. The index is:
                    // batch_size * cur_var_idx.
                    auto *out_idx = builder.CreateMul(builder.getInt32(batch_size), cur_var_idx);
                    auto *out_p = builder.CreateInBoundsGEP(fp_scal_t, out_ptr, out_idx);
                    auto *cur_out = load_vector_from_memory(builder, out_p, batch_size);
                    store_vector_to_memory(builder, out_p, builder.CreateFAdd(tc, builder.CreateFMul(cur_out, h)));
                });
            });
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass, if requested.
    if (optimise) {
        s.optimise();
    }
}

template void taylor_add_d_out_function<double>(llvm_state &, std::uint32_t, std::uint32_t, std::uint32_t, bool, bool,
                                                bool);
template void taylor_add_d_out_function<long double>(llvm_state &, std::uint32_t, std::uint32_t, std::uint32_t, bool,
                                                     bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template void taylor_add_d_out_function<mppp::real128>(llvm_state &, std::uint32_t, std::uint32_t, std::uint32_t, bool,
                                                       bool, bool);

#endif

} // namespace detail

// NOTE: there are situations (e.g., ensemble simulations) in which
// we may end up recompiling over and over the same code for the computation
// of continuous output. Perhaps we should consider some caching of llvm states
// containing continuous output functions.
template <typename T>
void continuous_output<T>::add_c_out_function(std::uint32_t order, std::uint32_t dim, bool high_accuracy)
{
    // Overflow check: we want to be able to index into the arrays of
    // times and Taylor coefficients using 32-bit ints.
    // LCOV_EXCL_START
    if (m_tcs.size() > std::numeric_limits<std::uint32_t>::max()
        || m_times_hi.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Overflow detected while adding continuous output to a Taylor integrator");
    }
    // LCOV_EXCL_STOP

    auto &md = m_llvm_state.module();
    auto &builder = m_llvm_state.builder();
    auto &context = m_llvm_state.context();

    // Fetch the current insertion block.
    auto orig_bb = builder.GetInsertBlock();

    // Add the function for the computation of the dense output.
    detail::taylor_add_d_out_function<T>(m_llvm_state, dim, order, 1, high_accuracy, false, false);

    // Fetch it.
    auto d_out_f = md.getFunction("d_out_f");
    assert(d_out_f != nullptr); // LCOV_EXCL_LINE

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // Establish the time direction.
    const detail::dfloat<T> df_t_start(m_times_hi[0], m_times_lo[0]), df_t_end(m_times_hi.back(), m_times_lo.back());
    const auto dir = df_t_start < df_t_end;

    // The function arguments:
    // - the output pointer (read/write, used also for accumulation),
    // - the time value,
    // - the pointer to the Taylor coefficients (read-only),
    // - the pointer to the hi times (read-only),
    // - the pointer to the lo times (read-only).
    // No overlap is allowed.
    auto fp_t = detail::to_llvm_type<T>(context);
    auto ptr_t = llvm::PointerType::getUnqual(fp_t);
    std::vector<llvm::Type *> fargs{ptr_t, fp_t, ptr_t, ptr_t, ptr_t};
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "c_out", &md);
    // LCOV_EXCL_START
    if (f == nullptr) {
        throw std::invalid_argument("Unable to create a function for continuous output in a Taylor integrator");
    }
    // LCOV_EXCL_STOP

    // Set the names/attributes of the function arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto tm = f->args().begin() + 1;
    tm->setName("tm");

    auto tc_ptr = f->args().begin() + 2;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto times_ptr_hi = f->args().begin() + 3;
    times_ptr_hi->setName("times_ptr_hi");
    times_ptr_hi->addAttr(llvm::Attribute::NoCapture);
    times_ptr_hi->addAttr(llvm::Attribute::NoAlias);
    times_ptr_hi->addAttr(llvm::Attribute::ReadOnly);

    auto times_ptr_lo = f->args().begin() + 4;
    times_ptr_lo->setName("times_ptr_lo");
    times_ptr_lo->addAttr(llvm::Attribute::NoCapture);
    times_ptr_lo->addAttr(llvm::Attribute::NoAlias);
    times_ptr_lo->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Create the variable in which we will store the timestep size.
    // This is necessary because the d_out_f function requires a pointer
    // to the timestep.
    auto h_ptr = builder.CreateAlloca(fp_t);

    // Look for the index in the times vector corresponding to
    // a time greater than tm (less than tm in backwards integration).
    // This is essentially an implementation of std::upper_bound:
    // https://en.cppreference.com/w/cpp/algorithm/upper_bound
    auto tidx = builder.CreateAlloca(builder.getInt32Ty());
    auto count = builder.CreateAlloca(builder.getInt32Ty());
    auto step = builder.CreateAlloca(builder.getInt32Ty());
    auto first = builder.CreateAlloca(builder.getInt32Ty());

    // count is inited with the size of the range.
    builder.CreateStore(builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size())), count);
    // first is inited to zero.
    builder.CreateStore(builder.getInt32(0), first);

    detail::llvm_while_loop(
        m_llvm_state,
        [&]() { return builder.CreateICmpNE(builder.CreateLoad(builder.getInt32Ty(), count), builder.getInt32(0)); },
        [&]() {
            // tidx = first.
            builder.CreateStore(builder.CreateLoad(builder.getInt32Ty(), first), tidx);
            // step = count / 2.
            builder.CreateStore(
                builder.CreateUDiv(builder.CreateLoad(builder.getInt32Ty(), count), builder.getInt32(2)), step);
            // tidx = tidx + step.
            builder.CreateStore(builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), tidx),
                                                  builder.CreateLoad(builder.getInt32Ty(), step)),
                                tidx);

            // Logical condition:
            // - !(tm < *tidx), if integrating forward,
            // - !(tm > *tidx), if integrating backward.
            auto tidx_val_hi = builder.CreateLoad(
                fp_t, builder.CreateInBoundsGEP(fp_t, times_ptr_hi, builder.CreateLoad(builder.getInt32Ty(), tidx)));
            auto tidx_val_lo = builder.CreateLoad(
                fp_t, builder.CreateInBoundsGEP(fp_t, times_ptr_lo, builder.CreateLoad(builder.getInt32Ty(), tidx)));
            auto zero_val = llvm::ConstantFP::get(fp_t, 0.);
            auto cond = dir ? detail::llvm_dl_lt(m_llvm_state, tm, zero_val, tidx_val_hi, tidx_val_lo)
                            : detail::llvm_dl_gt(m_llvm_state, tm, zero_val, tidx_val_hi, tidx_val_lo);
            cond = builder.CreateNot(cond);

            detail::llvm_if_then_else(
                m_llvm_state, cond,
                [&]() {
                    // ++tidx.
                    builder.CreateStore(
                        builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), tidx), builder.getInt32(1)), tidx);
                    // first = tidx.
                    builder.CreateStore(builder.CreateLoad(builder.getInt32Ty(), tidx), first);
                    // count = count - (step + 1).
                    builder.CreateStore(
                        builder.CreateSub(
                            builder.CreateLoad(builder.getInt32Ty(), count),
                            builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), step), builder.getInt32(1))),
                        count);
                },
                [&]() {
                    // count = step.
                    builder.CreateStore(builder.CreateLoad(builder.getInt32Ty(), step), count);
                });
        });

    // NOTE: the output of the std::upper_bound algorithm
    // is in the 'first' variable.
    llvm::Value *tc_idx = builder.CreateLoad(builder.getInt32Ty(), first);

    // Normally, the TC index should be first - 1. The exceptions are:
    // - first == 0, in which case TC index is also 0,
    // - first == range size, in which case TC index is first - 2.
    // These two exceptions arise when tm is outside the range of validity
    // for the continuous output. In such cases, we will use either the first
    // or the last possible set of TCs.
    detail::llvm_if_then_else(
        m_llvm_state, builder.CreateICmpEQ(tc_idx, builder.getInt32(0)),
        []() {
            // first == 0, do nothing.
        },
        [&]() {
            detail::llvm_if_then_else(
                m_llvm_state,
                builder.CreateICmpEQ(tc_idx, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size()))),
                [&]() {
                    // first == range size.
                    builder.CreateStore(builder.CreateSub(tc_idx, builder.getInt32(2)), first);
                },
                [&]() {
                    // The normal path.
                    builder.CreateStore(builder.CreateSub(tc_idx, builder.getInt32(1)), first);
                });
        });

    // Reload tc_idx.
    tc_idx = builder.CreateLoad(builder.getInt32Ty(), first);

    // Load the time corresponding to tc_idx.
    auto start_tm_hi = builder.CreateLoad(fp_t, builder.CreateInBoundsGEP(fp_t, times_ptr_hi, tc_idx));
    auto start_tm_lo = builder.CreateLoad(fp_t, builder.CreateInBoundsGEP(fp_t, times_ptr_lo, tc_idx));

    // Compute and store the value of h = tm - start_tm.
    auto [h_hi, h_lo] = detail::llvm_dl_add(m_llvm_state, tm, llvm::ConstantFP::get(fp_t, 0.),
                                            builder.CreateFNeg(start_tm_hi), builder.CreateFNeg(start_tm_lo));
    builder.CreateStore(h_hi, h_ptr);

    // Compute the index into the Taylor coefficients array.
    tc_idx = builder.CreateMul(tc_idx, builder.getInt32(dim * (order + 1u)));

    // Invoke the d_out function.
    builder.CreateCall(d_out_f, {out_ptr, builder.CreateInBoundsGEP(fp_t, tc_ptr, tc_idx), h_ptr});

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    m_llvm_state.verify_function(f);

    // Run the optimisation pass.
    m_llvm_state.optimise();

    // Compile.
    m_llvm_state.compile();

    // Fetch the function pointer and assign it.
    m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
}

template <typename T>
continuous_output<T>::continuous_output() = default;

template <typename T>
continuous_output<T>::continuous_output(llvm_state &&s) : m_llvm_state(std::move(s))
{
}

template <typename T>
continuous_output<T>::continuous_output(const continuous_output &o)
    : m_llvm_state(o.m_llvm_state), m_tcs(o.m_tcs), m_times_hi(o.m_times_hi), m_times_lo(o.m_times_lo),
      m_output(o.m_output)
{
    // If o is valid, fetch the function pointer from the copied state.
    // Otherwise, m_f_ptr will remain null.
    if (o.m_f_ptr != nullptr) {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
continuous_output<T>::continuous_output(continuous_output &&) noexcept = default;

template <typename T>
continuous_output<T>::~continuous_output() = default;

template <typename T>
continuous_output<T> &continuous_output<T>::operator=(const continuous_output &o)
{
    if (this != &o) {
        *this = continuous_output(o);
    }

    return *this;
}

template <typename T>
continuous_output<T> &continuous_output<T>::operator=(continuous_output &&) noexcept = default;

template <typename T>
void continuous_output<T>::call_impl(T t)
{
    using std::isfinite;

    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output object");
    }

    // NOTE: run the assertions only after ensuring this
    // is a valid object.

    // LCOV_EXCL_START
#if !defined(NDEBUG)
    // m_output must not be empty.
    assert(!m_output.empty());
    // Need at least 2 time points.
    assert(m_times_hi.size() >= 2u);
    // hi/lo parts of times must have the same sizes.
    assert(m_times_hi.size() == m_times_lo.size());
#endif
    // LCOV_EXCL_STOP

    if (!isfinite(t)) {
        throw std::invalid_argument(
            fmt::format("Cannot compute the continuous output at the non-finite time {}", detail::fp_to_string(t)));
    }

    m_f_ptr(m_output.data(), t, m_tcs.data(), m_times_hi.data(), m_times_lo.data());
}

template <typename T>
const llvm_state &continuous_output<T>::get_llvm_state() const
{
    return m_llvm_state;
}

template <typename T>
void continuous_output<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_llvm_state;
    ar << m_tcs;
    ar << m_times_hi;
    ar << m_times_lo;
    ar << m_output;
}

template <typename T>
void continuous_output<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_llvm_state;
    ar >> m_tcs;
    ar >> m_times_hi;
    ar >> m_times_lo;
    ar >> m_output;

    // NOTE: if m_output is not empty, it means the archived
    // object had been initialised.
    if (m_output.empty()) {
        m_f_ptr = nullptr;
    } else {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
std::pair<T, T> continuous_output<T>::get_bounds() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output object");
    }

    return {m_times_hi[0], m_times_hi.back()};
}

template <typename T>
std::size_t continuous_output<T>::get_n_steps() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output object");
    }

    return boost::numeric_cast<std::size_t>(m_times_hi.size() - 1u);
}

template class continuous_output<double>;
template class continuous_output<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class continuous_output<mppp::real128>;

#endif

namespace detail
{

template <typename T>
std::ostream &c_out_stream_impl(std::ostream &os, const continuous_output<T> &co)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    if (co.get_output().empty()) {
        oss << "Default-constructed continuous_output";
    } else {
        const detail::dfloat<T> df_t_start(co.m_times_hi[0], co.m_times_lo[0]),
            df_t_end(co.m_times_hi.back(), co.m_times_lo.back());
        const auto dir = df_t_start < df_t_end;

        oss << "Direction : " << (dir ? "forward" : "backward") << '\n';
        oss << "Time range: "
            << (dir ? fmt::format("[{}, {})", fp_to_string(co.m_times_hi[0]), fp_to_string(co.m_times_hi.back()))
                    : fmt::format("({}, {}]", fp_to_string(co.m_times_hi.back()), fp_to_string(co.m_times_hi[0])))
            << '\n';
        oss << "N of steps: " << (co.m_times_hi.size() - 1u) << '\n';
    }

    return os << oss.str();
}

} // namespace detail

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<double> &co)
{
    return detail::c_out_stream_impl(os, co);
}

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<long double> &co)
{
    return detail::c_out_stream_impl(os, co);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<mppp::real128> &co)
{
    return detail::c_out_stream_impl(os, co);
}

#endif

#if !defined(NDEBUG)

extern "C" {

// Function to check, in debug mode, the indexing of the Taylor coefficients
// in the batch mode continuous output implementation.
HEYOKA_DLL_PUBLIC void heyoka_continuous_output_batch_tc_idx_debug(const std::uint32_t *tc_idx,
                                                                   std::uint32_t times_size,
                                                                   std::uint32_t batch_size) noexcept
{
    // LCOV_EXCL_START
    assert(batch_size != 0u);
    assert(times_size % batch_size == 0u);
    assert(times_size / batch_size >= 3u);
    // LCOV_EXCL_STOP

    for (std::uint32_t i = 0; i < batch_size; ++i) {
        assert(tc_idx[i] < times_size / batch_size - 2u); // LCOV_EXCL_LINE
    }
}
}

#endif

// Continuous output for the batch integrator.
template <typename T>
void continuous_output_batch<T>::add_c_out_function(std::uint32_t order, std::uint32_t dim, bool high_accuracy)
{
    // Overflow check: we want to be able to index into the arrays of
    // times and Taylor coefficients using 32-bit ints.
    // LCOV_EXCL_START
    if (m_tcs.size() > std::numeric_limits<std::uint32_t>::max()
        || m_times_hi.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "Overflow detected while adding continuous output to a Taylor integrator in batch mode");
    }
    // LCOV_EXCL_STOP

    auto &md = m_llvm_state.module();
    auto &builder = m_llvm_state.builder();
    auto &context = m_llvm_state.context();

    // The function arguments:
    // - the output pointer (read/write, used also for accumulation),
    // - the pointer to the target time values (read-only),
    // - the pointer to the Taylor coefficients (read-only),
    // - the pointer to the hi times (read-only),
    // - the pointer to the lo times (read-only).
    // No overlap is allowed.
    auto fp_t = detail::to_llvm_type<T>(context);
    auto fp_vec_t = detail::make_vector_type(fp_t, m_batch_size);
    auto ptr_t = llvm::PointerType::getUnqual(fp_t);
    std::vector<llvm::Type *> fargs(5, ptr_t);
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "c_out", &md);
    // LCOV_EXCL_START
    if (f == nullptr) {
        throw std::invalid_argument("Unable to create a function for continuous output in a Taylor integrator");
    }
    // LCOV_EXCL_STOP

    // Set the names/attributes of the function arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto tm_ptr = f->args().begin() + 1;
    tm_ptr->setName("tm_ptr");
    tm_ptr->addAttr(llvm::Attribute::NoCapture);
    tm_ptr->addAttr(llvm::Attribute::NoAlias);
    tm_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto tc_ptr = f->args().begin() + 2;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto times_ptr_hi = f->args().begin() + 3;
    times_ptr_hi->setName("times_ptr_hi");
    times_ptr_hi->addAttr(llvm::Attribute::NoCapture);
    times_ptr_hi->addAttr(llvm::Attribute::NoAlias);
    times_ptr_hi->addAttr(llvm::Attribute::ReadOnly);

    auto times_ptr_lo = f->args().begin() + 4;
    times_ptr_lo->setName("times_ptr_lo");
    times_ptr_lo->addAttr(llvm::Attribute::NoCapture);
    times_ptr_lo->addAttr(llvm::Attribute::NoAlias);
    times_ptr_lo->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Establish the time directions.
    auto bool_vector_t = detail::make_vector_type(builder.getInt1Ty(), m_batch_size);
    assert(bool_vector_t != nullptr); // LCOV_EXCL_LINE
    llvm::Value *dir_vec{};
    if (m_batch_size == 1u) {
        // In scalar mode, the direction is a single value.
        const detail::dfloat<T> df_t_start(m_times_hi[0], m_times_lo[0]),
            // NOTE: we load from the padding values here.
            df_t_end(m_times_hi.back(), m_times_lo.back());
        const auto dir = df_t_start < df_t_end;

        dir_vec = builder.getInt1(dir);
    } else {
        dir_vec = llvm::UndefValue::get(bool_vector_t);
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const detail::dfloat<T> df_t_start(m_times_hi[i], m_times_lo[i]),
                // NOTE: we load from the padding values here.
                df_t_end(m_times_hi[m_times_hi.size() - m_batch_size + i],
                         m_times_lo[m_times_lo.size() - m_batch_size + i]);
            const auto dir = df_t_start < df_t_end;

            dir_vec = builder.CreateInsertElement(dir_vec, builder.getInt1(dir), i);
        }
    }

    // Look for the index in the times vector corresponding to
    // a time greater than tm (less than tm in backwards integration).
    // This is essentially an implementation of std::upper_bound:
    // https://en.cppreference.com/w/cpp/algorithm/upper_bound
    auto int32_vec_t = detail::make_vector_type(builder.getInt32Ty(), m_batch_size);
    auto tidx = builder.CreateAlloca(int32_vec_t);
    auto count = builder.CreateAlloca(int32_vec_t);
    auto step = builder.CreateAlloca(int32_vec_t);
    auto first = builder.CreateAlloca(int32_vec_t);

    // count is inited with the size of the range.
    // NOTE: count includes the padding.
    builder.CreateStore(
        detail::vector_splat(builder, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size()) / m_batch_size),
                             m_batch_size),
        count);

    // first is inited to zero.
    auto zero_vec_i32 = detail::vector_splat(builder, builder.getInt32(0), m_batch_size);
    builder.CreateStore(zero_vec_i32, first);

    // Load the time value from tm_ptr.
    auto tm = detail::load_vector_from_memory(builder, tm_ptr, m_batch_size);

    // This is the vector [0, 1, 2, ..., (batch_size - 1)].
    llvm::Value *batch_offset{};
    if (m_batch_size == 1u) {
        // In scalar mode, use a single value.
        batch_offset = builder.getInt32(0);
    } else {
        batch_offset = llvm::UndefValue::get(int32_vec_t);
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            batch_offset = builder.CreateInsertElement(batch_offset, builder.getInt32(i), i);
        }
    }

    // Splatted version of the batch size.
    auto batch_splat = detail::vector_splat(builder, builder.getInt32(m_batch_size), m_batch_size);

    // Splatted versions of the base pointers for the time data.
    auto times_ptr_hi_vec = detail::vector_splat(builder, times_ptr_hi, m_batch_size);
    auto times_ptr_lo_vec = detail::vector_splat(builder, times_ptr_lo, m_batch_size);

    // fp vector of zeroes.
    auto zero_vec_fp = detail::vector_splat(builder, llvm::ConstantFP::get(fp_t, 0.), m_batch_size);

    // Vector of i32 ones.
    auto one_vec_i32 = detail::vector_splat(builder, builder.getInt32(1), m_batch_size);

    detail::llvm_while_loop(
        m_llvm_state,
        [&]() -> llvm::Value * {
            // NOTE: the condition here is that any value in count is not zero.
            auto cmp = builder.CreateICmpNE(builder.CreateLoad(int32_vec_t, count), zero_vec_i32);

            // NOTE: in scalar mode, no reduction is needed.
            return (m_batch_size == 1u) ? cmp : builder.CreateOrReduce(cmp);
        },
        [&]() {
            // tidx = first.
            builder.CreateStore(builder.CreateLoad(int32_vec_t, first), tidx);
            // step = count / 2.
            auto two_vec_i32 = detail::vector_splat(builder, builder.getInt32(2), m_batch_size);
            builder.CreateStore(builder.CreateUDiv(builder.CreateLoad(int32_vec_t, count), two_vec_i32), step);
            // tidx = tidx + step.
            builder.CreateStore(
                builder.CreateAdd(builder.CreateLoad(int32_vec_t, tidx), builder.CreateLoad(int32_vec_t, step)), tidx);

            // Compute the indices for loading the times from the pointers.
            auto tl_idx = builder.CreateAdd(builder.CreateMul(builder.CreateLoad(int32_vec_t, tidx), batch_splat),
                                            batch_offset);

            // Compute the pointers for loading the time data.
            auto tptr_hi = builder.CreateInBoundsGEP(fp_t, times_ptr_hi_vec, tl_idx);
            auto tptr_lo = builder.CreateInBoundsGEP(fp_t, times_ptr_lo_vec, tl_idx);

            // Gather the hi/lo values.
            auto tidx_val_hi = detail::gather_vector_from_memory(builder, fp_vec_t, tptr_hi);
            auto tidx_val_lo = detail::gather_vector_from_memory(builder, fp_vec_t, tptr_lo);

            // Compute the two conditions !(tm < *tidx) and !(tm > *tidx).
            auto cmp_lt
                = builder.CreateNot(detail::llvm_dl_lt(m_llvm_state, tm, zero_vec_fp, tidx_val_hi, tidx_val_lo));
            auto cmp_gt
                = builder.CreateNot(detail::llvm_dl_gt(m_llvm_state, tm, zero_vec_fp, tidx_val_hi, tidx_val_lo));

            // Select cmp_lt if integrating forward, cmp_gt when integrating backward.
            auto cond = builder.CreateSelect(dir_vec, cmp_lt, cmp_gt);

            // tidx += (1 or 0).
            builder.CreateStore(builder.CreateAdd(builder.CreateLoad(int32_vec_t, tidx),
                                                  builder.CreateSelect(cond, one_vec_i32, zero_vec_i32)),
                                tidx);

            // first = (tidx or first).
            builder.CreateStore(builder.CreateSelect(cond, builder.CreateLoad(int32_vec_t, tidx),
                                                     builder.CreateLoad(int32_vec_t, first)),
                                first);

            // count = count - (step or count).
            auto old_count = builder.CreateLoad(int32_vec_t, count);
            auto new_count = builder.CreateSub(
                old_count, builder.CreateSelect(cond, builder.CreateLoad(int32_vec_t, step), old_count));

            // count = count + (-1 or step).
            new_count = builder.CreateAdd(new_count, builder.CreateSelect(cond, builder.CreateNeg(one_vec_i32),
                                                                          builder.CreateLoad(int32_vec_t, step)));
            builder.CreateStore(new_count, count);
        });

    // NOTE: the output of the std::upper_bound algorithm
    // is in the 'first' variable.
    llvm::Value *tc_idx = builder.CreateLoad(int32_vec_t, first);

    // Normally, the TC index should be first - 1. The exceptions are:
    // - first == 0, in which case TC index is also 0,
    // - first == (range size - 1), in which case TC index is first - 2.
    // These two exceptions arise when tm is outside the range of validity
    // for the continuous output. In such cases, we will use either the first
    // or the last possible set of TCs.
    // NOTE: the second check is range size - 1 (rather than just range size
    // like in the scalar case) due to padding.
    // In order to vectorise the check, we compute:
    // tc_idx = tc_idx - (tc_idx != 0) - (tc_idx == range size - 1).
    auto tc_idx_cmp1 = builder.CreateZExt(builder.CreateICmpNE(tc_idx, zero_vec_i32), int32_vec_t);
    auto tc_idx_cmp2 = builder.CreateZExt(
        builder.CreateICmpEQ(
            tc_idx, detail::vector_splat(
                        builder, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size() / m_batch_size - 1u)),
                        m_batch_size)),
        int32_vec_t);
    tc_idx = builder.CreateSub(tc_idx, tc_idx_cmp1);
    tc_idx = builder.CreateSub(tc_idx, tc_idx_cmp2);

#if !defined(NDEBUG)

    {
        // In debug mode, invoke the index checking function.
        auto *array_t = llvm::ArrayType::get(builder.getInt32Ty(), m_batch_size);
        auto tc_idx_debug_ptr = builder.CreateInBoundsGEP(array_t, builder.CreateAlloca(array_t),
                                                          {builder.getInt32(0), builder.getInt32(0)});
        detail::store_vector_to_memory(builder, tc_idx_debug_ptr, tc_idx);
        detail::llvm_invoke_external(m_llvm_state, "heyoka_continuous_output_batch_tc_idx_debug", builder.getVoidTy(),
                                     {tc_idx_debug_ptr, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size())),
                                      builder.getInt32(m_batch_size)});
    }

#endif

    // Convert tc_idx into an index for loading from the time vectors.
    auto tc_l_idx = builder.CreateAdd(builder.CreateMul(tc_idx, batch_splat), batch_offset);

    // Load the times corresponding to tc_idx.
    auto start_tm_hi = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                         builder.CreateInBoundsGEP(fp_t, times_ptr_hi_vec, tc_l_idx));
    auto start_tm_lo = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                         builder.CreateInBoundsGEP(fp_t, times_ptr_lo_vec, tc_l_idx));

    // Compute the value of h = tm - start_tm.
    auto h = detail::llvm_dl_add(m_llvm_state, tm, zero_vec_fp, builder.CreateFNeg(start_tm_hi),
                                 builder.CreateFNeg(start_tm_lo))
                 .first;

    // Compute the base pointers in the array of TC for the computation
    // of Horner's scheme.
    tc_idx = builder.CreateAdd(
        builder.CreateMul(
            tc_idx, detail::vector_splat(builder, builder.getInt32(dim * (order + 1u) * m_batch_size), m_batch_size)),
        batch_offset);
    // NOTE: each pointer in tc_ptrs points to the Taylor coefficient of
    // order 0 for the first state variable in the timestep data block selected
    // for that batch index.
    auto tc_ptrs = builder.CreateInBoundsGEP(fp_t, tc_ptr, tc_idx);

    // Run the Horner scheme.
    if (high_accuracy) {
        // Create the array for storing the running compensations.
        auto array_type = llvm::ArrayType::get(fp_vec_t, dim);
        auto comp_arr = builder.CreateInBoundsGEP(array_type, builder.CreateAlloca(array_type),
                                                  {builder.getInt32(0), builder.getInt32(0)});

        // Start by writing into out_ptr the zero-order coefficients
        // and by filling with zeroes the running compensations.
        detail::llvm_loop_u32(m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptrs. The index is:
            // m_batch_size * (order + 1u) * cur_var_idx.
            auto *load_idx = builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx);
            auto *tcs = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                          builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));

            // Store it in out_ptr. The index is:
            // m_batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
            detail::store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx), tcs);

            // Zero-init the element in comp_arr.
            builder.CreateStore(zero_vec_fp, builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx));
        });

        // Init the running updater for the powers of h.
        auto *cur_h = builder.CreateAlloca(fp_vec_t);
        builder.CreateStore(h, cur_h);

        // Run the evaluation.
        detail::llvm_loop_u32(
            m_llvm_state, builder.getInt32(1), builder.getInt32(order + 1u), [&](llvm::Value *cur_order) {
                // Load the current power of h.
                auto *cur_h_val = builder.CreateLoad(fp_vec_t, cur_h);

                detail::llvm_loop_u32(
                    m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
                        // Load the coefficient from tc_ptrs. The index is:
                        // m_batch_size * (order + 1u) * cur_var_idx + m_batch_size * cur_order.
                        auto *load_idx = builder.CreateAdd(
                            builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx),
                            builder.CreateMul(builder.getInt32(m_batch_size), cur_order));
                        auto *cf = detail::gather_vector_from_memory(
                            builder, fp_vec_t, builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));
                        auto *tmp = builder.CreateFMul(cf, cur_h_val);

                        // Compute the quantities for the compensation.
                        auto *comp_ptr = builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx);
                        auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
                        auto *res_ptr = builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx);
                        auto *y = builder.CreateFSub(tmp, builder.CreateLoad(fp_vec_t, comp_ptr));
                        auto *cur_res = detail::load_vector_from_memory(builder, res_ptr, m_batch_size);
                        auto *t = builder.CreateFAdd(cur_res, y);

                        // Update the compensation and the return value.
                        builder.CreateStore(builder.CreateFSub(builder.CreateFSub(t, cur_res), y), comp_ptr);
                        detail::store_vector_to_memory(builder, res_ptr, t);
                    });

                // Update the value of h.
                builder.CreateStore(builder.CreateFMul(cur_h_val, h), cur_h);
            });
    } else {
        // Start by writing into out_ptr the coefficients of the highest-degree
        // monomial in each polynomial.
        detail::llvm_loop_u32(m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptrs. The index is:
            // m_batch_size * (order + 1u) * cur_var_idx + m_batch_size * order.
            auto *load_idx
                = builder.CreateAdd(builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx),
                                    builder.getInt32(m_batch_size * order));
            auto *tcs = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                          builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));

            // Store it in out_ptr. The index is:
            // m_batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
            detail::store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx), tcs);
        });

        // Now let's run the Horner scheme.
        detail::llvm_loop_u32(
            m_llvm_state, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                detail::llvm_loop_u32(
                    m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
                        // Load the current Taylor coefficients from tc_ptrs.
                        // NOTE: we are loading the coefficients backwards wrt the order, hence
                        // we specify order - cur_order.
                        // NOTE: the index is:
                        // m_batch_size * (order + 1u) * cur_var_idx + m_batch_size * (order - cur_order).
                        auto *load_idx = builder.CreateAdd(
                            builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx),
                            builder.CreateMul(builder.getInt32(m_batch_size),
                                              builder.CreateSub(builder.getInt32(order), cur_order)));
                        auto *tcs = detail::gather_vector_from_memory(
                            builder, fp_vec_t, builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));

                        // Accumulate in out_ptr. The index is:
                        // m_batch_size * cur_var_idx.
                        auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
                        auto *out_p = builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx);
                        auto *cur_out = detail::load_vector_from_memory(builder, out_p, m_batch_size);
                        detail::store_vector_to_memory(builder, out_p,
                                                       builder.CreateFAdd(tcs, builder.CreateFMul(cur_out, h)));
                    });
            });
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    m_llvm_state.verify_function(f);

    // Run the optimisation pass.
    m_llvm_state.optimise();

    // Compile.
    m_llvm_state.compile();

    // Fetch the function pointer and assign it.
    m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
}

template <typename T>
continuous_output_batch<T>::continuous_output_batch() = default;

template <typename T>
continuous_output_batch<T>::continuous_output_batch(llvm_state &&s) : m_llvm_state(std::move(s))
{
}

template <typename T>
continuous_output_batch<T>::continuous_output_batch(const continuous_output_batch &o)
    : m_batch_size(o.m_batch_size), m_llvm_state(o.m_llvm_state), m_tcs(o.m_tcs), m_times_hi(o.m_times_hi),
      m_times_lo(o.m_times_lo), m_output(o.m_output), m_tmp_tm(o.m_tmp_tm)
{
    // If o is valid, fetch the function pointer from the copied state.
    // Otherwise, m_f_ptr will remain null.
    if (o.m_f_ptr != nullptr) {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
continuous_output_batch<T>::continuous_output_batch(continuous_output_batch &&) noexcept = default;

template <typename T>
continuous_output_batch<T>::~continuous_output_batch() = default;

template <typename T>
continuous_output_batch<T> &continuous_output_batch<T>::operator=(const continuous_output_batch &o)
{
    if (this != &o) {
        *this = continuous_output_batch(o);
    }

    return *this;
}

template <typename T>
continuous_output_batch<T> &continuous_output_batch<T>::operator=(continuous_output_batch &&) noexcept = default;

template <typename T>
void continuous_output_batch<T>::call_impl(const T *t)
{
    using std::isfinite;

    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    // NOTE: run the assertions only after ensuring this
    // is a valid object.

    // LCOV_EXCL_START
#if !defined(NDEBUG)
    // The batch size must not be zero.
    assert(m_batch_size != 0u);
    // m_batch_size must divide m_output exactly.
    assert(m_output.size() % m_batch_size == 0u);
    // m_tmp_tm must be of size m_batch_size.
    assert(m_tmp_tm.size() == m_batch_size);
    // m_batch_size must divide the time and tcs vectors exactly.
    assert(m_times_hi.size() % m_batch_size == 0u);
    assert(m_tcs.size() % m_batch_size == 0u);
    // Need at least 3 time points (2 + 1 for padding).
    assert(m_times_hi.size() / m_batch_size >= 3u);
    // hi/lo parts of times must have the same sizes.
    assert(m_times_hi.size() == m_times_lo.size());
#endif
    // LCOV_EXCL_STOP

    // Copy over the times to the temp buffer and check that they are finite.
    // NOTE: this copy ensures we avoid aliasing issues with the
    // other data members.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        if (!isfinite(t[i])) {
            throw std::invalid_argument(fmt::format("Cannot compute the continuous output in batch mode "
                                                    "for the batch index {} at the non-finite time {}",
                                                    i, detail::fp_to_string(t[i])));
        }

        m_tmp_tm[i] = t[i];
    }

    m_f_ptr(m_output.data(), m_tmp_tm.data(), m_tcs.data(), m_times_hi.data(), m_times_lo.data());
}

template <typename T>
const std::vector<T> &continuous_output_batch<T>::operator()(const std::vector<T> &tm)
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    if (tm.size() != m_batch_size) {
        throw std::invalid_argument(
            fmt::format("An invalid time vector was passed to the call operator of continuous_output_batch: the "
                        "vector size is {}, but a size of {} was expected instead",
                        tm.size(), m_batch_size));
    }

    return (*this)(tm.data());
}

// NOTE: there's some overlap with the call_impl() code here.
template <typename T>
const std::vector<T> &continuous_output_batch<T>::operator()(T tm)
{
    using std::isfinite;

    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    // NOTE: run the assertions only after ensuring this
    // is a valid object.

    // LCOV_EXCL_START
#if !defined(NDEBUG)
    // The batch size must not be zero.
    assert(m_batch_size != 0u);
    // m_batch_size must divide m_output exactly.
    assert(m_output.size() % m_batch_size == 0u);
    // m_tmp_tm must be of size m_batch_size.
    assert(m_tmp_tm.size() == m_batch_size);
    // m_batch_size must divide the time and tcs vectors exactly.
    assert(m_times_hi.size() % m_batch_size == 0u);
    assert(m_tcs.size() % m_batch_size == 0u);
    // Need at least 3 time points (2 + 1 for padding).
    assert(m_times_hi.size() / m_batch_size >= 3u);
    // hi/lo parts of times must have the same sizes.
    assert(m_times_hi.size() == m_times_lo.size());
#endif
    // LCOV_EXCL_STOP

    if (!isfinite(tm)) {
        throw std::invalid_argument(fmt::format("Cannot compute the continuous output in batch mode "
                                                "at the non-finite time {}",
                                                detail::fp_to_string(tm)));
    }

    // Copy over the time to the temp buffer.
    std::fill(m_tmp_tm.begin(), m_tmp_tm.end(), tm);

    m_f_ptr(m_output.data(), m_tmp_tm.data(), m_tcs.data(), m_times_hi.data(), m_times_lo.data());

    return m_output;
}

template <typename T>
const llvm_state &continuous_output_batch<T>::get_llvm_state() const
{
    return m_llvm_state;
}

template <typename T>
std::uint32_t continuous_output_batch<T>::get_batch_size() const
{
    return m_batch_size;
}

template <typename T>
void continuous_output_batch<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_batch_size;
    ar << m_llvm_state;
    ar << m_tcs;
    ar << m_times_hi;
    ar << m_times_lo;
    ar << m_output;
    ar << m_tmp_tm;
}

template <typename T>
void continuous_output_batch<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_batch_size;
    ar >> m_llvm_state;
    ar >> m_tcs;
    ar >> m_times_hi;
    ar >> m_times_lo;
    ar >> m_output;
    ar >> m_tmp_tm;

    // NOTE: if m_output is not empty, it means the archived
    // object had been initialised.
    if (m_output.empty()) {
        m_f_ptr = nullptr;
    } else {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> continuous_output_batch<T>::get_bounds() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    std::vector<T> lb, ub;
    lb.resize(boost::numeric_cast<decltype(lb.size())>(m_batch_size));
    ub.resize(boost::numeric_cast<decltype(ub.size())>(m_batch_size));

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        lb[i] = m_times_hi[i];
        // NOTE: take into account the padding.
        ub[i] = m_times_hi[m_times_hi.size() - 2u * m_batch_size + i];
    }

    return std::make_pair(std::move(lb), std::move(ub));
}

template <typename T>
std::size_t continuous_output_batch<T>::get_n_steps() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    // NOTE: account for padding.
    return boost::numeric_cast<std::size_t>(m_times_hi.size() / m_batch_size - 2u);
}

template class continuous_output_batch<double>;
template class continuous_output_batch<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class continuous_output_batch<mppp::real128>;

#endif

namespace detail
{

template <typename T>
std::ostream &c_out_batch_stream_impl(std::ostream &os, const continuous_output_batch<T> &co)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    if (co.get_output().empty()) {
        oss << "Default-constructed continuous_output_batch";
    } else {
        const auto batch_size = co.m_batch_size;

        oss << "Directions : [";
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const detail::dfloat<T> df_t_start(co.m_times_hi[i], co.m_times_lo[i]),
                df_t_end(co.m_times_hi[co.m_times_hi.size() - 2u * batch_size + i],
                         co.m_times_lo[co.m_times_lo.size() - 2u * batch_size + i]);
            const auto dir = df_t_start < df_t_end;

            oss << (dir ? "forward" : "backward");

            if (i != batch_size - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";

        oss << "Time ranges: [";
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const detail::dfloat<T> df_t_start(co.m_times_hi[i], co.m_times_lo[i]),
                df_t_end(co.m_times_hi[co.m_times_hi.size() - 2u * batch_size + i],
                         co.m_times_lo[co.m_times_lo.size() - 2u * batch_size + i]);
            const auto dir = df_t_start < df_t_end;
            oss << (dir ? fmt::format("[{}, {})", fp_to_string(df_t_start.hi), fp_to_string(df_t_end.hi))
                        : fmt::format("({}, {}]", fp_to_string(df_t_end.hi), fp_to_string(df_t_start.hi)));

            if (i != batch_size - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";

        oss << "N of steps : " << co.get_n_steps() << '\n';
    }

    return os << oss.str();
}

} // namespace detail

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<double> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<long double> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<mppp::real128> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

#endif

} // namespace heyoka

// NOTE: this is the worker function that is invoked to compute
// in parallel all the derivatives of a block in parallel mode.
extern "C" HEYOKA_DLL_PUBLIC void heyoka_cm_par_looper(std::uint32_t ncalls,
                                                       void (*fptr)(std::uint32_t, std::uint32_t) noexcept) noexcept
{
    try {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::uint32_t>(0, ncalls),
                                  [fptr](const auto &range) { fptr(range.begin(), range.end()); });
        // LCOV_EXCL_START
    } catch (const std::exception &ex) {
        heyoka::detail::get_logger()->critical("Exception caught in the parallel mode looper: {}", ex.what());
    } catch (...) {
        heyoka::detail::get_logger()->critical("Exception caught in the parallel mode looper");
    }
    // LCOV_EXCL_STOP
}

namespace heyoka::detail
{

namespace
{

// NOTE: use typedef to minimise issues
// when mucking around with the preprocessor.
using par_f_ptr = void (*)() noexcept;

} // namespace

} // namespace heyoka::detail

// NOTE: this is the parallel invoker that gets called from LLVM
// to run multiple parallel workers within a segment at the same time, i.e.,
// to process multiple blocks within a segment concurrently.
// We need to generate multiple instantiatiation of this function
// up to the limit HEYOKA_CM_PAR_MAX_INVOKE_N defined in config.hpp.

#define HEYOKA_CM_PAR_INVOKE(_0, N, _1)                                                                                \
    extern "C" HEYOKA_DLL_PUBLIC void heyoka_cm_par_invoke_##N(                                                        \
        BOOST_PP_ENUM_PARAMS(N, heyoka::detail::par_f_ptr f)) noexcept                                                 \
    {                                                                                                                  \
        try {                                                                                                          \
            BOOST_PP_IF(BOOST_PP_SUB(N, 1), oneapi::tbb::parallel_invoke(BOOST_PP_ENUM_PARAMS(N, f)), f0());           \
        } catch (const std::exception &ex) {                                                                           \
            heyoka::detail::get_logger()->critical("Exception caught in the parallel mode invoker: {}", ex.what());    \
        } catch (...) {                                                                                                \
            heyoka::detail::get_logger()->critical("Exception caught in the parallel mode invoker");                   \
        }                                                                                                              \
    }

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_ADD(HEYOKA_CM_PAR_MAX_INVOKE_N, 1), HEYOKA_CM_PAR_INVOKE, _0)

#undef HEYOKA_CM_PAR_INVOKE
