// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <exception>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

expression::expression() : expression(number{0.}) {}

expression::expression(double x) : expression(number{x}) {}

expression::expression(long double x) : expression(number{x}) {}

#if defined(HEYOKA_HAVE_REAL128)

expression::expression(mppp::real128 x) : expression(number{x}) {}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression::expression(mppp::real x) : expression(number{std::move(x)}) {}

#endif

expression::expression(std::string s) : expression(variable{std::move(s)}) {}

expression::expression(number n) : m_value(std::move(n)) {}

expression::expression(variable var) : m_value(std::move(var)) {}

expression::expression(func f) : m_value(std::move(f)) {}

expression::expression(param p) : m_value(std::move(p)) {}

expression::expression(const expression &) = default;

// NOLINTNEXTLINE(bugprone-exception-escape)
expression::expression(expression &&other) noexcept : m_value(std::move(other.m_value))
{
    // NOTE: ensure other is equivalent to a
    // default-constructed expression.
    other.m_value.emplace<number>(0.);
}

expression::~expression() = default;

expression &expression::operator=(const expression &other)
{
    if (this != &other) {
        *this = expression(other);
    }

    return *this;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
expression &expression::operator=(expression &&other) noexcept
{
    if (this != &other) {
        m_value = std::move(other.m_value);
        // NOTE: ensure other is equivalent to a
        // default-constructed expression.
        other.m_value.emplace<number>(0.);
    }

    return *this;
}

const expression::value_type &expression::value() const
{
    return m_value;
}

void swap(expression &ex0, expression &ex1) noexcept
{
    std::swap(ex0.m_value, ex1.m_value);
}

namespace detail
{

namespace
{

expression copy_impl(funcptr_map<expression> &func_map, const expression &e)
{
    return std::visit(
        [&func_map](const auto &v) {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already copied the current function, fetch the copy
                    // from the cache.
                    return it->second;
                }

                // Create the new args vector by making a deep copy
                // of the original function arguments.
                std::vector<expression> new_args;
                new_args.reserve(v.args().size());
                for (const auto &orig_arg : v.args()) {
                    // NOTE: the argument needs to be copied via a recursive
                    // call to copy_impl() only if it is a func. Otherwise, a normal
                    // copy will suffice.
                    if (std::holds_alternative<func>(orig_arg.value())) {
                        new_args.push_back(copy_impl(func_map, orig_arg));
                    } else {
                        new_args.push_back(orig_arg);
                    }
                }

                // Create a copy of v with the new arguments.
                auto f_copy = v.copy(new_args);

                // Construct the return value and put it into the cache.
                auto ex = expression{std::move(f_copy)};
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ex);
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return ex;
            } else {
                return expression{v};
            }
        },
        e.value());
}

} // namespace

} // namespace detail

expression copy(const expression &e)
{
    detail::funcptr_map<expression> func_map;

    return detail::copy_impl(func_map, e);
}

std::vector<expression> copy(const std::vector<expression> &v_ex)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        ret.push_back(detail::copy_impl(func_map, ex));
    }

    return ret;
}

inline namespace literals
{

expression operator""_dbl(long double x)
{
    return expression{static_cast<double>(x)};
}

expression operator""_dbl(unsigned long long n)
{
    return expression{static_cast<double>(n)};
}

expression operator""_ldbl(long double x)
{
    return expression{x};
}

expression operator""_ldbl(unsigned long long n)
{
    return expression{static_cast<long double>(n)};
}

expression operator""_var(const char *s, std::size_t n)
{
    return expression{variable{std::string(s, n)}};
}

} // namespace literals

namespace detail
{

prime_wrapper::prime_wrapper(std::string s) : m_str(std::move(s)) {}

prime_wrapper::prime_wrapper(const prime_wrapper &) = default;

prime_wrapper::prime_wrapper(prime_wrapper &&) noexcept = default;

prime_wrapper &prime_wrapper::operator=(const prime_wrapper &) = default;

prime_wrapper &prime_wrapper::operator=(prime_wrapper &&) noexcept = default;

prime_wrapper::~prime_wrapper() = default;

// NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
std::pair<expression, expression> prime_wrapper::operator=(expression e) &&
{
    return std::pair{expression{variable{std::move(m_str)}}, std::move(e)};
}

} // namespace detail

detail::prime_wrapper prime(const expression &e)
{
    return std::visit(
        [&e](const auto &v) -> detail::prime_wrapper {
            if constexpr (std::is_same_v<variable, detail::uncvref_t<decltype(v)>>) {
                return detail::prime_wrapper{v.name()};
            } else {
                throw std::invalid_argument(
                    fmt::format("Cannot apply the prime() operator to the non-variable expression '{}'", e));
            }
        },
        e.value());
}

inline namespace literals
{

detail::prime_wrapper operator""_p(const char *s, std::size_t n)
{
    return detail::prime_wrapper{std::string(s, n)};
}

} // namespace literals

namespace detail
{

namespace
{

void get_variables(funcptr_set &func_set, std::unordered_set<std::string> &s_set, const expression &e)
{
    std::visit(
        [&func_set, &s_set](const auto &arg) {
            using type = uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = arg.get_ptr();

                if (func_set.find(f_id) != func_set.end()) {
                    // We already determined the list of variables for the
                    // current function, exit.
                    return;
                }

                // Determine the list of variables for each
                // function argument.
                for (const auto &farg : arg.args()) {
                    get_variables(func_set, s_set, farg);
                }

                // Add the id of f to the set.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            } else if constexpr (std::is_same_v<type, variable>) {
                s_set.insert(arg.name());
            }
        },
        e.value());
}

} // namespace

} // namespace detail

std::vector<std::string> get_variables(const expression &e)
{
    detail::funcptr_set func_set;

    std::unordered_set<std::string> s_set;

    detail::get_variables(func_set, s_set, e);

    // Turn the set into an ordered vector.
    std::vector retval(s_set.begin(), s_set.end());
    std::sort(retval.begin(), retval.end());

    return retval;
}

std::vector<std::string> get_variables(const std::vector<expression> &v_ex)
{
    detail::funcptr_set func_set;

    std::unordered_set<std::string> s_set;

    for (const auto &ex : v_ex) {
        detail::get_variables(func_set, s_set, ex);
    }

    // Turn the set into an ordered vector.
    std::vector retval(s_set.begin(), s_set.end());
    std::sort(retval.begin(), retval.end());

    return retval;
}

namespace detail
{

namespace
{

expression rename_variables(detail::funcptr_map<expression> &func_map, const expression &e,
                            const std::unordered_map<std::string, std::string> &repl_map)
{
    return std::visit(
        [&func_map, &repl_map](const auto &arg) {
            using type = uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already renamed variables for the current function,
                    // fetch the result from the cache.
                    return it->second;
                }

                // Prepare the new args vector by renaming variables
                // for all arguments.
                std::vector<expression> new_args;
                new_args.reserve(arg.args().size());
                for (const auto &orig_arg : arg.args()) {
                    new_args.push_back(rename_variables(func_map, orig_arg, repl_map));
                }

                // Create a copy of arg with the new arguments.
                auto tmp = arg.copy(new_args);

                // Put the return value in the cache.
                auto ret = expression{std::move(tmp)};
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            } else if constexpr (std::is_same_v<type, variable>) {
                if (auto it = repl_map.find(arg.name()); it != repl_map.end()) {
                    return expression{it->second};
                }

                // NOTE: fall through to the default case of returning a copy
                // of the original variable if no renaming took place.
            }

            return expression{arg};
        },
        e.value());
}

} // namespace

} // namespace detail

expression rename_variables(const expression &e, const std::unordered_map<std::string, std::string> &repl_map)
{
    detail::funcptr_map<expression> func_map;

    return detail::rename_variables(func_map, e, repl_map);
}

std::vector<expression> rename_variables(const std::vector<expression> &v_ex,
                                         const std::unordered_map<std::string, std::string> &repl_map)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        retval.push_back(detail::rename_variables(func_map, ex, repl_map));
    }

    return retval;
}

namespace detail
{

std::size_t hash(funcptr_map<std::size_t> &func_map, const expression &ex)
{
    return std::visit(
        [&func_map](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already computed the hash for the current
                    // function, return it.
                    return it->second;
                }

                // Compute the hash of the current function.
                auto retval = v.hash(func_map);

                // Add it to the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, retval);
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return retval;
            } else {
                return hash(v);
            }
        },
        ex.value());
}

} // namespace detail

std::size_t hash(const expression &ex)
{
    detail::funcptr_map<std::size_t> func_map;

    return detail::hash(func_map, ex);
}

namespace detail
{

namespace
{

// Exception to signal that the stream output
// for an expression has become too large.
struct output_too_long : std::exception {
};

} // namespace

// Helper to stream an expression to a stringstream, while
// checking that the number of characters written so far
// to the stream is not too large. If that is the case,
// an exception will be thrown.
void stream_expression(std::ostringstream &oss, const expression &e)
{
    if (oss.tellp() > 1000) {
        throw output_too_long{};
    }

    std::visit(
        [&oss](const auto &v) {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                v.to_stream(oss);
            } else {
                oss << v;
            }
        },
        e.value());
}

} // namespace detail

std::ostream &operator<<(std::ostream &os, const expression &e)
{
    std::ostringstream oss;

    try {
        detail::stream_expression(oss, e);
    } catch (const detail::output_too_long &) {
        oss << "...";
    }

    return os << oss.str();
}

namespace detail
{

namespace
{

// Exception to signal that the computation
// of the number of nodes resulted in overflow.
struct too_many_nodes : std::exception {
};

std::size_t get_n_nodes(funcptr_map<std::size_t> &func_map, const expression &e)
{
    return std::visit(
        [&func_map](const auto &arg) -> std::size_t {
            if constexpr (std::is_same_v<func, uncvref_t<decltype(arg)>>) {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already computed the number of nodes for the current
                    // function, return it.
                    return it->second;
                }

                boost::safe_numerics::safe<std::size_t> retval = 1;

                for (const auto &ex : arg.args()) {
                    try {
                        retval += get_n_nodes(func_map, ex);
                    } catch (...) {
                        throw too_many_nodes{};
                    }
                }

                // Store the number of nodes for the current function
                // in the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, retval);
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return retval;
            } else {
                return 1;
            }
        },
        e.value());
}

} // namespace

} // namespace detail

// NOTE: this always returns a number > 0, unless an overflow
// occurs due to the expression being too large. In such case,
// zero is returned.
std::size_t get_n_nodes(const expression &e)
{
    detail::funcptr_map<std::size_t> func_map;

    try {
        return detail::get_n_nodes(func_map, e);
    } catch (const detail::too_many_nodes &) {
        return 0;
    }
}

namespace detail
{

namespace
{

expression subs(funcptr_map<expression> &func_map, const expression &ex,
                const std::unordered_map<std::string, expression> &smap, bool canonicalise)
{
    return std::visit(
        [&func_map, &smap, canonicalise](const auto &arg) {
            using type = uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, number> || std::is_same_v<type, param>) {
                return expression{arg};
            } else if constexpr (std::is_same_v<type, variable>) {
                if (auto it = smap.find(arg.name()); it == smap.end()) {
                    return expression{arg};
                } else {
                    return it->second;
                }
            } else {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already performed substitution on the current function,
                    // fetch the result from the cache.
                    return it->second;
                }

                // Create the new args vector by running the
                // substitution on all arguments.
                std::vector<expression> new_args;
                new_args.reserve(arg.args().size());
                for (const auto &orig_arg : arg.args()) {
                    new_args.push_back(subs(func_map, orig_arg, smap, canonicalise));
                }

                // Canonicalise the new arguments vector,
                // if requested and if the function is commutative.
                if (canonicalise && arg.is_commutative()) {
                    std::stable_sort(new_args.begin(), new_args.end(), comm_ops_lt);
                }

                // Create a copy of arg with the new arguments.
                auto tmp = arg.copy(new_args);

                // Put the return value in the cache.
                auto ret = expression{std::move(tmp)};
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

expression subs(const expression &e, const std::unordered_map<std::string, expression> &smap, bool canonicalise)
{
    detail::funcptr_map<expression> func_map;

    return detail::subs(func_map, e, smap, canonicalise);
}

std::vector<expression> subs(const std::vector<expression> &v_ex,
                             const std::unordered_map<std::string, expression> &smap, bool canonicalise)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        ret.push_back(detail::subs(func_map, e, smap, canonicalise));
    }

    return ret;
}

namespace detail
{

namespace
{

expression subs(funcptr_map<expression> &func_map, const expression &ex,
                const std::unordered_map<expression, expression> &smap, bool canonicalise)
{
    if (auto it = smap.find(ex); it != smap.end()) {
        // ex is in the substitution map, return the value it maps to.
        return it->second;
    }

    return std::visit(
        [&](const auto &arg) {
            using type = uncvref_t<decltype(arg)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = arg.get_ptr();

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already performed substitution on the current function,
                    // fetch the result from the cache.
                    return it->second;
                }

                // Create the new args vector by running the
                // substitution on all arguments.
                std::vector<expression> new_args;
                new_args.reserve(arg.args().size());
                for (const auto &orig_arg : arg.args()) {
                    new_args.push_back(subs(func_map, orig_arg, smap, canonicalise));
                }

                // Canonicalise the new arguments vector,
                // if requested and if the function is commutative.
                if (canonicalise && arg.is_commutative()) {
                    std::stable_sort(new_args.begin(), new_args.end(), comm_ops_lt);
                }

                // Create a copy of arg with the new arguments.
                auto tmp = arg.copy(new_args);

                // Put the return value in the cache.
                auto ret = expression{std::move(tmp)};
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag);

                return ret;
            } else {
                // ex is not a function and it does not show
                // up in the substitution map. Thus, we can just
                // return a copy of it unchanged.
                return expression{arg};
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

expression subs(const expression &e, const std::unordered_map<expression, expression> &smap, bool canonicalise)
{
    detail::funcptr_map<expression> func_map;

    return detail::subs(func_map, e, smap, canonicalise);
}

std::vector<expression> subs(const std::vector<expression> &v_ex,
                             const std::unordered_map<expression, expression> &smap, bool canonicalise)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        ret.push_back(detail::subs(func_map, e, smap, canonicalise));
    }

    return ret;
}

namespace detail
{

namespace
{

// Pairwise reduction of a vector of expressions.
template <typename F>
expression pairwise_reduce_impl(const F &func, std::vector<expression> list)
{
    assert(!list.empty());

    // LCOV_EXCL_START
    if (list.size() == std::numeric_limits<decltype(list.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_reduce()");
    }
    // LCOV_EXCL_STOP

    while (list.size() != 1u) {
        const auto cur_size = list.size();

        // Init the new list. The size will be halved, +1 if the
        // current size is odd.
        const auto next_size = cur_size / 2u + cur_size % 2u;
        std::vector<expression> new_list(next_size);

        tbb::parallel_for(tbb::blocked_range<decltype(new_list.size())>(0, new_list.size()),
                          [&list, &new_list, cur_size, &func](const auto &r) {
                              for (auto i = r.begin(); i != r.end(); ++i) {
                                  if (i * 2u == cur_size - 1u) {
                                      // list has an odd size, and we are at the last element of list.
                                      // Just move it to new_list.
                                      new_list[i] = std::move(list.back());
                                  } else {
                                      new_list[i] = func(std::move(list[i * 2u]), std::move(list[i * 2u + 1u]));
                                  }
                              }
                          });

        new_list.swap(list);
    }

    return std::move(list[0]);
}

} // namespace

} // namespace detail

// Pairwise product.
expression pairwise_prod(const std::vector<expression> &prod)
{
    if (prod.empty()) {
        return 1_dbl;
    }

    return detail::pairwise_reduce_impl(std::multiplies{}, prod);
}

double eval_dbl(const expression &e, const std::unordered_map<std::string, double> &map,
                const std::vector<double> &pars)
{
    return std::visit([&](const auto &arg) { return eval_dbl(arg, map, pars); }, e.value());
}

long double eval_ldbl(const expression &e, const std::unordered_map<std::string, long double> &map,
                      const std::vector<long double> &pars)
{
    return std::visit([&](const auto &arg) { return eval_ldbl(arg, map, pars); }, e.value());
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 eval_f128(const expression &e, const std::unordered_map<std::string, mppp::real128> &map,
                        const std::vector<mppp::real128> &pars)
{
    return std::visit([&](const auto &arg) { return eval_f128(arg, map, pars); }, e.value());
}
#endif

void eval_batch_dbl(std::vector<double> &retval, const expression &e,
                    const std::unordered_map<std::string, std::vector<double>> &map, const std::vector<double> &pars)
{
    std::visit([&](const auto &arg) { eval_batch_dbl(retval, arg, map, pars); }, e.value());
}

std::vector<std::vector<std::size_t>> compute_connections(const expression &e)
{
    std::vector<std::vector<std::size_t>> node_connections;
    std::size_t node_counter = 0u;
    update_connections(node_connections, e, node_counter);
    return node_connections;
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const expression &e,
                        std::size_t &node_counter)
{
    std::visit([&node_connections,
                &node_counter](const auto &arg) { update_connections(node_connections, arg, node_counter); },
               e.value());
}

std::vector<double> compute_node_values_dbl(const expression &e, const std::unordered_map<std::string, double> &map,
                                            const std::vector<std::vector<std::size_t>> &node_connections)
{
    std::vector<double> node_values(node_connections.size());
    std::size_t node_counter = 0u;
    update_node_values_dbl(node_values, e, map, node_connections, node_counter);
    return node_values;
}

void update_node_values_dbl(std::vector<double> &node_values, const expression &e,
                            const std::unordered_map<std::string, double> &map,
                            const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter)
{
    std::visit([&map, &node_values, &node_connections, &node_counter](
                   const auto &arg) { update_node_values_dbl(node_values, arg, map, node_connections, node_counter); },
               e.value());
}

std::unordered_map<std::string, double> compute_grad_dbl(const expression &e,
                                                         const std::unordered_map<std::string, double> &map,
                                                         const std::vector<std::vector<std::size_t>> &node_connections)
{
    std::unordered_map<std::string, double> grad;
    auto node_values = compute_node_values_dbl(e, map, node_connections);
    std::size_t node_counter = 0u;
    update_grad_dbl(grad, e, map, node_values, node_connections, node_counter);
    return grad;
}

void update_grad_dbl(std::unordered_map<std::string, double> &grad, const expression &e,
                     const std::unordered_map<std::string, double> &map, const std::vector<double> &node_values,
                     const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter,
                     double acc)
{
    std::visit(
        [&map, &grad, &node_values, &node_connections, &node_counter, &acc](const auto &arg) {
            update_grad_dbl(grad, arg, map, node_values, node_connections, node_counter, acc);
        },
        e.value());
}

namespace detail
{

taylor_dc_t::size_type taylor_decompose(funcptr_map<taylor_dc_t::size_type> &func_map, const expression &ex,
                                        taylor_dc_t &dc)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        return fptr->taylor_decompose(func_map, dc);
    } else {
        return 0;
    }
}

} // namespace detail

llvm::Value *taylor_diff(llvm_state &s, llvm::Type *fp_t, const expression &ex, const std::vector<std::uint32_t> &deps,
                         const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                         std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size,
                         bool high_accuracy)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        return fptr->taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size, high_accuracy);
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument("Taylor derivatives can be computed only for functions");
        // LCOV_EXCL_STOP
    }
}

llvm::Function *taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, const expression &ex, std::uint32_t n_uvars,
                                   std::uint32_t batch_size, bool high_accuracy)
{
    if (const auto *fptr = std::get_if<func>(&ex.value())) {
        return fptr->taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy);
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument("Taylor derivatives in compact mode can be computed only for functions");
        // LCOV_EXCL_STOP
    }
}

namespace detail
{

// Helper to detect if ex is an integral number.
bool is_integral(const expression &ex)
{
    return std::visit(
        [](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, number>) {
                return std::visit(
                    [](const auto &x) {
                        using std::trunc;
                        using std::isfinite;

                        return isfinite(x) && x == trunc(x);
                    },
                    v.value());
            } else {
                // Not a number.
                return false;
            }
        },
        ex.value());
}

// Helper to detect if ex is a number in the form n / 2,
// where n is an odd integral value.
bool is_odd_integral_half(const expression &ex)
{
    return std::visit(
        [](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, number>) {
                return std::visit(
                    [](const auto &x) {
                        using std::trunc;
                        using std::isfinite;

                        if (!isfinite(x) || x == trunc(x)) {
                            // x is not finite, or it is already
                            // an integral value.
                            return false;
                        }

                        // NOTE: here we will be assuming that, for all supported
                        // float types, multiplication by 2 is exact.
                        // Since we are assuming IEEE floats anyway, we should be
                        // safe here.
                        // NOTE: y should never become infinity here, because this would mean
                        // that x is integral (since large float values are all integrals anyway).
                        const auto y = 2 * x;
                        return y == trunc(y);
                    },
                    v.value());
            } else {
                // Not a number.
                return false;
            }
        },
        ex.value());
}

expression par_impl::operator[](std::uint32_t idx) const
{
    return expression{param{idx}};
}

} // namespace detail

namespace detail
{

namespace
{

std::uint32_t get_param_size(detail::funcptr_set &func_set, const expression &ex)
{
    std::uint32_t retval = 0;

    std::visit(
        [&retval, &func_set](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, param>) {
                if (v.idx() == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("Overflow dected in get_n_param()");
                }

                retval = std::max(static_cast<std::uint32_t>(v.idx() + 1u), retval);
            } else if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_set.find(f_id); it != func_set.end()) {
                    // We already computed the number of params for the current
                    // function, exit.
                    return;
                }

                for (const auto &a : v.args()) {
                    retval = std::max(get_param_size(func_set, a), retval);
                }

                // Update the cache.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            }
        },
        ex.value());

    return retval;
}

} // namespace

} // namespace detail

// Determine the size of the parameter vector from the highest
// param index appearing in an expression. If the return value
// is zero, no params appear in the expression.
std::uint32_t get_param_size(const expression &ex)
{
    detail::funcptr_set func_set;

    return detail::get_param_size(func_set, ex);
}

namespace detail
{

namespace
{

void get_params(std::unordered_set<std::uint32_t> &idx_set, detail::funcptr_set &func_set, const expression &ex)
{
    std::visit(
        [&](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, param>) {
                idx_set.insert(v.idx());
            } else if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                if (auto it = func_set.find(f_id); it != func_set.end()) {
                    // We already got the params for the current function, exit.
                    return;
                }

                for (const auto &a : v.args()) {
                    get_params(idx_set, func_set, a);
                }

                // Update the cache.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

// Determine the list of parameters appearing in the
// expression ex. The result is a list of parameter
// expressions sorted according to the indices.
std::vector<expression> get_params(const expression &ex)
{
    std::unordered_set<std::uint32_t> idx_set;
    detail::funcptr_set func_set;

    // Write the indices of all parameters appearing in ex
    // into idx_set.
    detail::get_params(idx_set, func_set, ex);

    // Transform idx_set into a sorted vector.
    std::vector<std::uint32_t> idx_vec(idx_set.begin(), idx_set.end());
    std::sort(idx_vec.begin(), idx_vec.end());

    // Transform the sorted indices into a vector of
    // sorted parameter expressions.
    std::vector<expression> retval;
    retval.reserve(static_cast<decltype(retval.size())>(idx_vec.size()));
    std::transform(idx_vec.begin(), idx_vec.end(), std::back_inserter(retval), [](auto idx) { return par[idx]; });

    return retval;
}

std::vector<expression> get_params(const std::vector<expression> &v_ex)
{
    std::unordered_set<std::uint32_t> idx_set;
    detail::funcptr_set func_set;

    // Write the indices of all parameters appearing in v_ex
    // into idx_set.
    for (const auto &e : v_ex) {
        detail::get_params(idx_set, func_set, e);
    }

    // Transform idx_set into a sorted vector.
    std::vector<std::uint32_t> idx_vec(idx_set.begin(), idx_set.end());
    std::sort(idx_vec.begin(), idx_vec.end());

    // Transform the sorted indices into a vector of
    // sorted parameter expressions.
    std::vector<expression> retval;
    retval.reserve(static_cast<decltype(retval.size())>(idx_vec.size()));
    std::transform(idx_vec.begin(), idx_vec.end(), std::back_inserter(retval), [](auto idx) { return par[idx]; });

    return retval;
}

namespace detail
{

namespace
{

bool is_time_dependent(funcptr_map<bool> &func_map, const expression &ex)
{
    // - If ex is a function, check if it is time-dependent, or
    //   if any of its arguments is time-dependent,
    // - otherwise, return false.
    return std::visit(
        [&func_map](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                // Did we already determine if v is time-dependent?
                if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                // Check if the function is intrinsically time-dependent.
                bool is_tm_dep = v.is_time_dependent();
                if (!is_tm_dep) {
                    // The function does **not** intrinsically depend on time.
                    // Check its arguments.
                    for (const auto &a : v.args()) {
                        if (is_time_dependent(func_map, a)) {
                            // A time-dependent argument was found. Update
                            // is_tm_dep and break out, no point in checking
                            // the other arguments.
                            is_tm_dep = true;
                            break;
                        }
                    }
                }

                // Update the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, is_tm_dep);

                // An expression cannot contain itself.
                assert(flag);

                return is_tm_dep;
            } else {
                return false;
            }
        },
        ex.value());
}

} // namespace

} // namespace detail

// Determine if an expression is time-dependent.
bool is_time_dependent(const expression &ex)
{
    detail::funcptr_map<bool> func_map;

    return detail::is_time_dependent(func_map, ex);
}

HEYOKA_END_NAMESPACE
