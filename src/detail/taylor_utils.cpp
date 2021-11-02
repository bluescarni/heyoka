// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <chrono> // NOTE: needed for the spdlog stopwatch.
#include <cmath>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <ios>
#include <iterator>
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

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <spdlog/stopwatch.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

namespace
{

std::string taylor_c_diff_mangle(const variable &)
{
    return "var";
}

std::string taylor_c_diff_mangle(const number &)
{
    return "num";
}

std::string taylor_c_diff_mangle(const param &)
{
    return "par";
}

} // namespace

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
    auto fname = "heyoka.taylor_c_diff.{}."_format(name);

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
        fname += std::visit([](const auto &v) { return taylor_c_diff_mangle(v); }, args[i]);

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
                } else {
                    // For vars and params, the argument is an index
                    // in an array.
                    return llvm::Type::getInt32Ty(context);
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
        fname += "n_uvars_{}."_format(n_uvars);
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
    assert(batch_size > 0u);

    auto &builder = s.builder();

    // Determine the index into the parameter array.
    // LCOV_EXCL_START
    if (p.idx() > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("Overflow detected in the computation of the index into a parameter array");
    }
    // LCOV_EXCL_STOP
    const auto arr_idx = static_cast<std::uint32_t>(p.idx() * batch_size);

    // Compute the pointer to load from.
    auto *ptr = builder.CreateInBoundsGEP(par_ptr, {builder.getInt32(arr_idx)});

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
    auto &builder = s.builder();

    // Fetch the pointer into par_ptr.
    // NOTE: the overflow check is done in taylor_compute_jet().
    auto *ptr = builder.CreateInBoundsGEP(par_ptr, {builder.CreateMul(p, builder.getInt32(batch_size))});

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
    auto *ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx)});

    return builder.CreateLoad(ptr);
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
        [[maybe_unused]] const auto res = uvars_rename.emplace("u_{}"_format(i), "u_{}"_format(i));
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
            [[maybe_unused]] const auto res = uvars_rename.emplace("u_{}"_format(i), "u_{}"_format(retval.size() - 1u));
            assert(res.second);
        } else {
            // ex is redundant. This means
            // that it already appears in retval at index
            // it->second. Don't add anything to retval,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res = uvars_rename.emplace("u_{}"_format(i), "u_{}"_format(it->second));
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
            auto it = uvars_rename.find("u_{}"_format(idx));
            assert(it != uvars_rename.end());
            idx = uname_to_index(it->second);
        }
    }

    // Same for the indices in sv_funcs_dc.
    for (auto &idx : sv_funcs_dc) {
        auto it = uvars_rename.find("u_{}"_format(idx));
        assert(it != uvars_rename.end());
        idx = uname_to_index(it->second);
    }

    get_logger()->debug("Taylor CSE reduced decomposition size from {} to {}", orig_size, retval.size());
    get_logger()->trace("Taylor CSE runtime: {}", sw);

    return retval;
}

// Perform a topological sort on a graph representation
// of the Taylor decomposition. This can improve performance
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
        [[maybe_unused]] const auto res = remap.emplace("u_{}"_format(i), "u_{}"_format(i));
        assert(res.second);
    }
    // Establish the remapping for the u variables that are not
    // state variables.
    for (decltype(v_idx.size()) i = n_eq; i < v_idx.size() - n_eq; ++i) {
        [[maybe_unused]] const auto res = remap.emplace("u_{}"_format(v_idx[i]), "u_{}"_format(i));
        assert(res.second);
    }

    // Do the remap for the definitions of the u variables, the
    // derivatives and the hidden deps.
    for (auto *it = dc.data() + n_eq; it != dc.data() + dc.size(); ++it) {
        // Remap the expression.
        rename_variables(it->first, remap);

        // Remap the hidden dependencies.
        for (auto &idx : it->second) {
            auto it_remap = remap.find("u_{}"_format(idx));
            assert(it_remap != remap.end());
            idx = uname_to_index(it_remap->second);
        }
    }

    // Do the remap for sv_funcs.
    for (auto &idx : sv_funcs_dc) {
        auto it_remap = remap.find("u_{}"_format(idx));
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
                    auto check_arg = [i](const auto &arg) {
                        if (auto p_var = std::get_if<variable>(&arg.value())) {
                            assert(p_var->name().rfind("u_", 0) == 0);
                            assert(uname_to_index(p_var->name()) < i);
                        } else if (std::get_if<number>(&arg.value()) == nullptr
                                   && std::get_if<param>(&arg.value()) == nullptr) {
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
        subs_map.emplace("u_{}"_format(i), subs(dc[i].first, subs_map));
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
        subs_map.emplace("u_{}"_format(i), subs(dc[i].first, subs_map));
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

// A couple of helpers for deep-copying containers of expressions.
auto copy(const std::vector<expression> &v_ex)
{
    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    std::transform(v_ex.begin(), v_ex.end(), std::back_inserter(ret), [](const expression &e) { return copy(e); });

    return ret;
}

auto copy(const std::vector<std::pair<expression, expression>> &v)
{
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(v.size());

    std::transform(v.begin(), v.end(), std::back_inserter(ret), [](const auto &p) {
        return std::pair{copy(p.first), copy(p.second)};
    });

    return ret;
}

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
        throw std::invalid_argument(
            "The number of deduced variables for a Taylor decomposition ({}) differs from the number of equations ({})"_format(
                vars.size(), v_ex.size()));
    }

    // Check that the expressions in sv_funcs contain only
    // state variables.
    for (const auto &ex : sv_funcs) {
        for (const auto &var : get_variables(ex)) {
            if (vars.find(var) == vars.end()) {
                throw std::invalid_argument("The extra functions in a Taylor decomposition contain the variable '{}', "
                                            "which is not a state variable"_format(var));
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
            [[maybe_unused]] const auto eres = repl_map.emplace(var, "u_{}"_format(var_idx++));
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
            ex = expression{"u_{}"_format(dres)};
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
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    // Run the breadth-first topological sort on the decomposition.
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
                            "Error in the Taylor decomposition of a system of equations: the variable '{}' "
                            "appears in the left-hand side twice"_format(v.name()));
                    }
                } else {
                    throw std::invalid_argument(
                        "Error in the Taylor decomposition of a system of equations: the "
                        "left-hand side contains the expression '{}', which is not a variable"_format(lhs));
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
            throw std::invalid_argument("Error in the Taylor decomposition of a system of equations: the variable '{}' "
                                        "appears in the right-hand side but not in the left-hand side"_format(var));
        }
    }

    // Check that the expressions in sv_funcs contain only
    // state variables.
    for (const auto &ex : sv_funcs) {
        for (const auto &var : get_variables(ex)) {
            if (lhs_vars_set.find(var) == lhs_vars_set.end()) {
                throw std::invalid_argument("The extra functions in a Taylor decomposition contain the variable '{}', "
                                            "which is not a state variable"_format(var));
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
        [[maybe_unused]] const auto eres = repl_map.emplace(lhs_vars[i], "u_{}"_format(i));
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

    // Log the construction runtime in trace mode.
    spdlog::stopwatch sw;

    // Init the decomposition. It begins with a list
    // of the original lhs variables of the system.
    taylor_dc_t u_vars_defs;
    u_vars_defs.reserve(lhs_vars.size());
    for (const auto &var : lhs_vars) {
        u_vars_defs.emplace_back(variable{var}, std::vector<std::uint32_t>{});
    }

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
            ex = expression{"u_{}"_format(dres)};
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
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, sv_funcs_dc, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_rhs, u_vars_defs);
    detail::verify_taylor_dec_sv_funcs(sv_funcs_dc, orig_sv_funcs, u_vars_defs, n_eq);
#endif

    // Run the breadth-first topological sort on the decomposition.
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
std::ostream &taylor_adaptive_stream_impl(std::ostream &os, const taylor_adaptive_impl<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    oss << "Tolerance               : " << ta.get_tol() << '\n';
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
std::ostream &taylor_adaptive_batch_stream_impl(std::ostream &os, const taylor_adaptive_batch_impl<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    oss << "Tolerance   : " << ta.get_tol() << '\n';
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

    if (!ta.get_pars().empty()) {
        oss << "Parameters  : [";
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
                os << "taylor_outcome::terminal_event_{} (continuing)"_format(static_cast<std::int64_t>(oc));
            } else if (oc > taylor_outcome::success) {
                // Stopping terminal event.
                os << "taylor_outcome::terminal_event_{} (stopping)"_format(-static_cast<std::int64_t>(oc) - 1);
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
        return [](taylor_adaptive_batch_impl<T> &, T, int, std::uint32_t) {};
    } else {
        return [](taylor_adaptive_impl<T> &, T, int) {};
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
    os << "Cooldown       : " << (cooldown < 0 ? "auto" : "{}"_format(cooldown)) << '\n';

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

} // namespace detail

} // namespace heyoka
