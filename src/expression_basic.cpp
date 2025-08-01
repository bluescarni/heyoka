// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

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

#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

expression::expression() noexcept : expression(number{0.}) {}

expression::expression(float x) noexcept : expression(number{x}) {}

expression::expression(double x) noexcept : expression(number{x}) {}

expression::expression(long double x) noexcept : expression(number{x}) {}

#if defined(HEYOKA_HAVE_REAL128)

expression::expression(mppp::real128 x) noexcept : expression(number{x}) {}

#endif

#if defined(HEYOKA_HAVE_REAL)

expression::expression(mppp::real x) : expression(number{std::move(x)}) {}

#endif

expression::expression(std::string s) : expression(variable{std::move(s)}) {}

expression::expression(number n) : m_value(std::move(n)) {}

expression::expression(variable var) : m_value(std::move(var)) {}

expression::expression(func f) noexcept : m_value(std::move(f)) {}

expression::expression(param p) noexcept : m_value(std::move(p)) {}

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

void expression::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_value;
}

void expression::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_value;
}

const expression::value_type &expression::value() const noexcept
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

expression copy_impl(auto &func_map, auto &sargs_map, const expression &e)
{
    return ex_traverse_transform_nodes(func_map, sargs_map, e, {}, {});
}

} // namespace

} // namespace detail

expression copy(const expression &e)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    return detail::copy_impl(func_map, sargs_map, e);
}

std::vector<expression> copy(const std::vector<expression> &v_ex)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        ret.push_back(detail::copy_impl(func_map, sargs_map, ex));
    }

    return ret;
}

inline namespace literals
{

expression operator""_flt(long double x)
{
    return expression{static_cast<float>(x)};
}

expression operator""_flt(unsigned long long n)
{
    return expression{static_cast<float>(n)};
}

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

void get_variables_impl(auto &func_set, auto &sargs_set, auto &s_set, const expression &e)
{
    const auto vfunc = [&s_set](const expression &ex) {
        if (const auto *var_ptr = std::get_if<variable>(&ex.value())) {
            s_set.emplace(var_ptr->name());
        }
    };

    ex_traverse_visit_leaves(func_set, sargs_set, e, vfunc);
}

} // namespace

} // namespace detail

std::vector<std::string> get_variables(const expression &e)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;
    boost::unordered_flat_set<std::string> s_set;

    detail::get_variables_impl(func_set, sargs_set, s_set, e);

    // Turn the set into an ordered vector.
    std::vector retval(s_set.begin(), s_set.end());
    std::ranges::sort(retval);

    return retval;
}

std::vector<std::string> get_variables(const std::vector<expression> &v_ex)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;
    boost::unordered_flat_set<std::string> s_set;

    for (const auto &ex : v_ex) {
        detail::get_variables_impl(func_set, sargs_set, s_set, ex);
    }

    // Turn the set into an ordered vector.
    std::vector retval(s_set.begin(), s_set.end());
    std::ranges::sort(retval);

    return retval;
}

namespace detail
{

expression rename_variables_impl(void_ptr_map<const expression> &func_map,
                                 sargs_ptr_map<const func_args::shared_args_t> &sargs_map, const expression &e,
                                 const std::unordered_map<std::string, std::string> &repl_map)
{
    const auto rename_func = [&repl_map](const expression &ex) {
        // Check if the current expression is a variable whose name shows up
        // in repl_map. If it is, construct and return a variable expression
        // from the replacement string.
        //
        // Otherwise, just return a copy of ex.
        if (const auto *var_ptr = std::get_if<variable>(&ex.value())) {
            const auto it = repl_map.find(var_ptr->name());
            if (it != repl_map.end()) {
                return expression{it->second};
            }
        }

        return ex;
    };

    return ex_traverse_transform_nodes(func_map, sargs_map, e, rename_func, {});
}

} // namespace detail

expression rename_variables(const expression &e, const std::unordered_map<std::string, std::string> &repl_map)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    return detail::rename_variables_impl(func_map, sargs_map, e, repl_map);
}

std::vector<expression> rename_variables(const std::vector<expression> &v_ex,
                                         const std::unordered_map<std::string, std::string> &repl_map)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        retval.push_back(detail::rename_variables_impl(func_map, sargs_map, ex, repl_map));
    }

    return retval;
}

namespace detail
{

// NOTE: at this time there is no apparent need for a vectorised version of this.
// In case we ever need one, the simple function optimisation would need to be reconsidered.
// NOLINTNEXTLINE(bugprone-exception-escape)
std::size_t hash(const expression &ex) noexcept
{
    detail::void_ptr_map<const std::size_t> func_map;
    detail::sargs_ptr_map<const std::size_t> sargs_map;
    detail::traverse_stack stack;
    detail::return_stack<std::size_t> hash_stack;

    // Seed the stack.
    stack.emplace_back(&ex, false);

    while (!stack.empty()) {
        // Pop the traversal stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        if (const auto *f_ptr = std::get_if<func>(&cur_ex->value())) {
            // Function (i.e., internal) node.
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            if (visited) {
                // NOTE: if this is the second visit, we know that the the function cannot possibly be in the cache,
                // and thus we can avoid an unnecessary lookup.
                assert(!func_map.contains(f_id));
            } else if (const auto it = func_map.find(f_id); it != func_map.end()) {
                // We already computed the hash of the current function. Fetch it from the cache
                // and add it to the hash stack.
                hash_stack.emplace_back(it->second);
                continue;
            }

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now visited and computed the hash of all the children of the function node
                // (i.e., the function arguments). The hashes are at the tail end of
                // hash_stack. We will be popping and combining them in order to compute the hash
                // of the function.

                // NOTE: the hash of a function is obtained by combining the hashes of the
                // arguments with the hash of the function name.
                std::size_t seed = 0;
                const auto n_args = f.args().size();
                for (decltype(f.args().size()) i = 0; i < n_args; ++i) {
                    // NOTE: the hash stack must not be empty and its last element
                    // also cannot be empty.
                    assert(!hash_stack.empty());
                    assert(hash_stack.back());

                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    boost::hash_combine(seed, *hash_stack.back());
                    hash_stack.pop_back();
                }
                // Record the hash of the arguments.
                const auto args_hash = seed;

                // Final combination with the hash of the name.
                boost::hash_combine(seed, f.get_name());

                // Add the hashes to the caches.
                // NOTE: here we apply an optimisation: if the stack is empty,
                // we are at the end of the while loop and we won't need the cached
                // values in the future. Thus, we can avoid unnecessary heap allocations
                // if the expression we are hashing is a simple non-recursive function.
                if (!stack.empty()) {
                    func_map.emplace(f_id, seed);

                    if (shared_args) {
                        assert(!sargs_map.contains(&*shared_args));
                        sargs_map.emplace(&*shared_args, args_hash);
                    }
                }

                // Add it to hash_stack.
                // NOTE: hash_stack must not be empty and its last element
                // must be empty (it is supposed to be the empty hash we
                // pushed the first time we visited).
                assert(!hash_stack.empty());
                assert(!hash_stack.back());
                hash_stack.back().emplace(seed);
            } else {
                // It is the first time we visit this function.
                if (shared_args) {
                    // The function manages its arguments via a shared reference. Check
                    // if we already computed the hash of the arguments before.
                    if (const auto it = sargs_map.find(&*shared_args); it != sargs_map.end()) {
                        // We already have the hash of the arguments. Fetch it and combine it
                        // with the hash of the function name
                        auto seed = it->second;
                        boost::hash_combine(seed, f.get_name());

                        // Add the function to the cache and its hash value to hash_stack.
                        // NOTE: there's no point here in eliding the addition of f_id to the cache,
                        // because if we end up here it means that we already inserted some values
                        // in the caches.
                        func_map.emplace(f_id, seed);
                        hash_stack.emplace_back(seed);

                        continue;
                    }

                    // NOTE: if we arrive here, it means that the shared arguments of the function have never
                    // been hashed before. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its
                // arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &arg : f.args()) {
                    stack.emplace_back(&arg, false);
                }

                // Add an empty hash to hash_stack. We will add the real hash
                // later once we have hashed all arguments.
                hash_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            hash_stack.emplace_back(std::visit(
                []<typename T>(const T &arg) -> std::size_t {
                    if constexpr (std::same_as<func, T>) {
                        // LCOV_EXCL_START
                        assert(false);
                        return 0;
                        // LCOV_EXCL_STOP
                    } else {
                        return std::hash<T>{}(arg);
                    }
                },
                cur_ex->value()));
        }
    }

    assert(hash_stack.size() == 1u);
    assert(hash_stack.back());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return *hash_stack.back();
}

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
        [&oss]<typename T>(const T &v) {
            if constexpr (std::is_same_v<T, func>) {
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

// NOTE: this always returns a number > 0, unless an overflow
// occurs due to the expression being too large. In such case,
// zero is returned.
std::size_t get_n_nodes(const expression &e)
{
    detail::void_ptr_map<const std::size_t> func_map;
    detail::sargs_ptr_map<const std::size_t> sargs_map;
    detail::traverse_stack stack;

    boost::safe_numerics::safe<std::size_t> retval = 0;

    // Seed the stack.
    stack.emplace_back(&e, false);

    try {
        while (!stack.empty()) {
            // Pop the stack.
            const auto [cur_ex, visited] = stack.back();
            stack.pop_back();

            if (const auto *f_ptr = std::get_if<func>(&cur_ex->value())) {
                // Function (i.e., internal) node.
                const auto &f = *f_ptr;

                // Fetch the function id.
                const auto *f_id = f.get_ptr();

                if (visited) {
                    // NOTE: if this is the second visit, we know that the the function cannot possibly be in the cache,
                    // and thus we can avoid an unnecessary lookup.
                    assert(!func_map.contains(f_id));
                } else if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already computed the number of nodes for the current
                    // function, add it to retval.
                    retval += it->second;
                    continue;
                }

                // Check if the function manages its arguments via a shared reference.
                const auto shared_args = f.shared_args();

                if (visited) {
                    // This is the second time we visit the function. We can now
                    // compute its number of nodes.

                    // Count the function node itself.
                    boost::safe_numerics::safe<std::size_t> n_nodes = 1;

                    // Count the number of nodes in the arguments.
                    for (const auto &ex : f.args()) {
                        if (const auto *fptr = std::get_if<func>(&ex.value())) {
                            // The argument is a function. Its number of nodes was already
                            // computed and it must be available in the cache.
                            assert(func_map.contains(fptr->get_ptr()));
                            n_nodes += func_map.find(fptr->get_ptr())->second;
                        } else {
                            // The argument is a non-function (i.e., a leaf node). Increase
                            // the count by one.
                            ++n_nodes;
                        }
                    }

                    // Store the number of nodes for the current function
                    // in the cache.
                    func_map.emplace(f_id, n_nodes);

                    if (shared_args) {
                        // NOTE: if the function manages its arguments via a shared reference,
                        // we must make sure to record in sargs_map the number of nodes for the
                        // arguments, so that we do not have to recompute it when we encounter again
                        // the same shared reference.
                        assert(!sargs_map.contains(&*shared_args));
                        sargs_map.emplace(&*shared_args, n_nodes - 1);
                    }

                    // NOTE: by incrementing by 1 here we are accounting only for the function node
                    // itself. It is not necessary to account for the total number of nodes n_nodes
                    // because all the children nodes were already counted during visitation
                    // of the current function.
                    ++retval;
                } else {
                    // It is the first time we visit this function.
                    if (shared_args) {
                        // The function manages its arguments via a shared reference. Check
                        // if we already computed the total number of nodes for the arguments.
                        if (const auto it = sargs_map.find(&*shared_args); it != sargs_map.end()) {
                            // We already have the total number of nodes for the arguments. Fetch it and add 1
                            // to account for the function itself.
                            boost::safe_numerics::safe<std::size_t> n_nodes = 1;
                            n_nodes += it->second;

                            // Add the function to the cache.
                            func_map.emplace(f_id, n_nodes);

                            // Update retval and move on.
                            retval += n_nodes;
                            continue;
                        }

                        // NOTE: if we arrive here, it means that we have not computed the number of nodes
                        // for the shared arguments of the function yet. We thus fall through the usual
                        // visitation process.
                        ;
                    }

                    // Re-add the function to the stack with visited=true, and add all of its
                    // arguments to the stack as well.
                    stack.emplace_back(cur_ex, true);

                    for (const auto &ex : f.args()) {
                        stack.emplace_back(&ex, false);
                    }

                    // NOTE: we do not know at this stage what the total number of nodes
                    // for the current function is. This will be available the next time we
                    // pop the function from the stack. In the meantime do **not** increment retval.
                }
            } else {
                // Non-function (i.e., leaf) node.
                assert(!visited);
                ++retval;
            }
        }

        return retval;
    } catch (const std::system_error &) {
        // NOTE: this is not ideal, as in principle system_error may be thrown
        // by something other than overflow errors in boost::safe_numerics. However, as much
        // as I tried, I could not find out a way to detect if the thrown system_error
        // is coming out of boost::safe_numerics. It should be just a matter of comparing
        // the error code in the exception to the predefined error codes in boost::safe_numerics,
        // but for some reason this does not work. I suspect it has something to do with
        // the error category comparison failing because the error category of the code
        // in the exception is not the same object (same address) of the error category
        // in the predefined error codes. I don't understand why this has to be so complicated
        // but don't have the time to investigate now :(
        //
        // https://en.cppreference.com/w/cpp/error/error_code/operator_cmp

        return 0;
    }
}

namespace detail
{

expression subs_impl(void_ptr_map<const expression> &func_map, sargs_ptr_map<const func_args::shared_args_t> &sargs_map,
                     const expression &e, const std::unordered_map<std::string, expression> &smap)
{
    const auto subs_func = [&smap](const expression &ex) {
        if (const auto *var_ptr = std::get_if<variable>(&ex.value())) {
            // Variable node.
            if (const auto it = smap.find(var_ptr->name()); it != smap.end()) {
                // The variable shows up in smap. Return the corresponding expression.
                return it->second;
            }
        }

        // Non-variable node, or variable node which does *not* show up in smap.
        // Just return it unchanged.
        return ex;
    };

    return ex_traverse_transform_nodes(func_map, sargs_map, e, subs_func, {});
}

} // namespace detail

expression subs(const expression &e, const std::unordered_map<std::string, expression> &smap)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    return detail::subs_impl(func_map, sargs_map, e, smap);
}

std::vector<expression> subs(const std::vector<expression> &v_ex,
                             const std::unordered_map<std::string, expression> &smap)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        ret.push_back(detail::subs_impl(func_map, sargs_map, e, smap));
    }

    return ret;
}

namespace detail
{

expression subs_impl(void_ptr_map<const expression> &func_map, sargs_ptr_map<const func_args::shared_args_t> &sargs_map,
                     const expression &e, const std::map<expression, expression> &smap)
{
    traverse_stack stack;
    return_stack<expression> subs_stack;

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the traversal stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        // NOTE: the logic here is slightly different from the usual flow, in the sense
        // that we want the lookup into func_map to happen *before* the lookup
        // in smap. The reason is that the lookup in func_map is cheap, while the lookup
        // in smap can potentially be costly. If we already performed substitution on
        // cur_ex before, we want to take advantage of the cached result before performing
        // expression comparisons in the smap lookup.
        const auto *f_ptr = std::get_if<func>(&cur_ex->value());
        if (f_ptr != nullptr) {
            if (visited) {
                // NOTE: if this is the second visit, we know that the the function cannot possibly be in the cache,
                // and thus we can avoid an unnecessary lookup.
                assert(!func_map.contains(f_ptr->get_ptr()));
            } else if (const auto it = func_map.find(f_ptr->get_ptr()); it != func_map.end()) {
                // We already performed substitution on the current function,
                // fetch the result from the cache.
                subs_stack.emplace_back(it->second);
                continue;
            }
        }

        // Check if the current expression is in the substitution map.
        // NOTE: no need to check if this is the second visit.
        if (!visited) {
            if (const auto it = smap.find(*cur_ex); it != smap.end()) {
                // cur_ex is in the substitution map.

                // If cur_ex is a function, record the result of the substitution into func_map.
                if (f_ptr != nullptr) {
                    const auto *f_id = f_ptr->get_ptr();
                    func_map.emplace(f_id, it->second);
                }

                // Push to subs_stack the result of the substitution and move on.
                subs_stack.emplace_back(it->second);
                continue;
            }
        }

        if (f_ptr != nullptr) {
            // Function (i.e., internal) node.
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            // Check if the function manages its arguments via a shared reference.
            const auto shared_args = f.shared_args();

            if (visited) {
                // We have now visited and performed substitution on all the children of the function node
                // (i.e., the function arguments). The results are at the tail end of subs_stack. We will be
                // popping them from subs_stack and use them to initialise a new copy of the function.

                // Build the new arguments.
                std::vector<expression> new_args;
                const auto n_args = f.args().size();
                new_args.reserve(n_args);
                for (decltype(new_args.size()) i = 0; i < n_args; ++i) {
                    // NOTE: the subs stack must not be empty and its last element
                    // also cannot be empty.
                    assert(!subs_stack.empty());
                    assert(subs_stack.back());

                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    new_args.push_back(std::move(*subs_stack.back()));
                    subs_stack.pop_back();
                }

                // Create the new copy of the function.
                auto ex_copy = [&]() {
                    if (shared_args) {
                        // NOTE: if the function manages its arguments via a shared reference, we must make
                        // sure to record the new arguments in sargs_map, so that when we run again into the
                        // same shared reference we re-use the cached result.
                        auto new_sargs = std::make_shared<const std::vector<expression>>(std::move(new_args));

                        assert(!sargs_map.contains(&*shared_args));
                        sargs_map.emplace(&*shared_args, new_sargs);

                        return expression{f.make_copy_with_new_args(std::move(new_sargs))};
                    } else {
                        return expression{f.make_copy_with_new_args(std::move(new_args))};
                    }
                }();

                // Add it to the cache.
                func_map.emplace(f_id, ex_copy);

                // Add it to subs_stack.
                // NOTE: the subs stack must not be empty and its last element
                // must be empty (it is supposed to be the empty function we
                // pushed the first time we visited).
                assert(!subs_stack.empty());
                assert(!subs_stack.back());
                subs_stack.back().emplace(std::move(ex_copy));
            } else {
                // It is the first time we visit this function.
                if (shared_args) {
                    // The function manages its arguments via a shared reference. Check
                    // if we already performed substitution on the arguments before.
                    if (const auto it = sargs_map.find(&*shared_args); it != sargs_map.end()) {
                        // We performed substitution on the arguments before. Fetch the results from the cache and
                        // use them to construct a new copy of the function.
                        auto ex_copy = expression{f.make_copy_with_new_args(it->second)};

                        // Add the new function to the cache and to subs_stack.
                        func_map.emplace(f_id, ex_copy);
                        subs_stack.emplace_back(std::move(ex_copy));

                        continue;
                    }

                    // NOTE: if we arrive here, it means that we never performed substitution on the shared arguments
                    // before. We thus fall through the usual visitation process.
                    ;
                }

                // Re-add the function to the stack with visited=true, and add all of its
                // arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }

                // Add an empty function to subs_stack. The actual result of the substitution
                // will be emplaced once we have performed substitution on all arguments.
                subs_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node which does *not* show up in smap. Add it
            // unchanged to subs_stack.
            assert(!visited);
            subs_stack.emplace_back(*cur_ex);
        }
    }

    assert(subs_stack.size() == 1u);
    assert(subs_stack.back());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return std::move(*subs_stack.back());
}

} // namespace detail

// NOTE: in the subs() functions we are using a std::map, instead of
// std::unordered_map, because hashing always requires traversing
// the whole expression, while comparisons can exit early. This becomes
// important while traversing the expression "e" and checking if its internal
// subexpressions are contained in smap. With hashing, we run into a quadratic
// complexity scenario because at each step of the traversal we have again
// to traverse the entire subexpression in order to compute its hash value.
expression subs(const expression &e, const std::map<expression, expression> &smap)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    return detail::subs_impl(func_map, sargs_map, e, smap);
}

std::vector<expression> subs(const std::vector<expression> &v_ex, const std::map<expression, expression> &smap)
{
    detail::void_ptr_map<const expression> func_map;
    detail::sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        ret.push_back(detail::subs_impl(func_map, sargs_map, e, smap));
    }

    return ret;
}

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

expression par_impl::operator[](std::uint32_t idx) const
{
    return expression{param{idx}};
}

} // namespace detail

namespace detail
{

namespace
{

void get_param_size(auto &func_set, auto &sargs_set, std::uint32_t &retval, const expression &e)
{
    const auto vfunc = [&retval](const expression &ex) {
        if (const auto *par_ptr = std::get_if<param>(&ex.value())) {
            const auto tmp = static_cast<std::uint32_t>(par_ptr->idx() + boost::safe_numerics::safe<std::uint32_t>(1));
            retval = std::max(retval, tmp);
        }
    };

    ex_traverse_visit_leaves(func_set, sargs_set, e, vfunc);
}

} // namespace

} // namespace detail

// Determine the size of the parameter vector from the highest
// param index appearing in an expression. If the return value
// is zero, no params appear in the expression.
std::uint32_t get_param_size(const expression &ex)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;

    std::uint32_t retval = 0;

    detail::get_param_size(func_set, sargs_set, retval, ex);

    return retval;
}

std::uint32_t get_param_size(const std::vector<expression> &v_ex)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;

    std::uint32_t retval = 0;

    for (const auto &ex : v_ex) {
        detail::get_param_size(func_set, sargs_set, retval, ex);
    }

    return retval;
}

namespace detail
{

namespace
{

void get_params(auto &func_set, auto &sargs_set, auto &idx_set, const expression &e)
{
    const auto vfunc = [&idx_set](const expression &ex) {
        if (const auto *par_ptr = std::get_if<param>(&ex.value())) {
            idx_set.insert(par_ptr->idx());
        }
    };

    ex_traverse_visit_leaves(func_set, sargs_set, e, vfunc);
}

} // namespace

} // namespace detail

// Determine the list of parameters appearing in the
// expression ex. The result is a list of parameter
// expressions sorted according to the indices.
std::vector<expression> get_params(const expression &ex)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;

    boost::unordered_flat_set<std::uint32_t> idx_set;

    // Write the indices of all parameters appearing in ex into idx_set.
    detail::get_params(func_set, sargs_set, idx_set, ex);

    // Transform idx_set into a sorted vector of parameter expressions.
    std::vector<expression> retval;
    retval.reserve(idx_set.size());
    std::ranges::transform(idx_set, std::back_inserter(retval), [](auto idx) { return par[idx]; });
    std::ranges::sort(retval, [](const auto &e1, const auto &e2) {
        return std::get<param>(e1.value()).idx() < std::get<param>(e2.value()).idx();
    });

    return retval;
}

std::vector<expression> get_params(const std::vector<expression> &v_ex)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;

    boost::unordered_flat_set<std::uint32_t> idx_set;

    // Write the indices of all parameters appearing in v_ex
    // into idx_set.
    for (const auto &e : v_ex) {
        detail::get_params(func_set, sargs_set, idx_set, e);
    }

    // Transform idx_set into a sorted vector of parameter expressions.
    std::vector<expression> retval;
    retval.reserve(idx_set.size());
    std::ranges::transform(idx_set, std::back_inserter(retval), [](auto idx) { return par[idx]; });
    std::ranges::sort(retval, [](const auto &e1, const auto &e2) {
        return std::get<param>(e1.value()).idx() < std::get<param>(e2.value()).idx();
    });

    return retval;
}

namespace detail
{

namespace
{

bool is_time_dependent(auto &func_set, auto &sargs_set, const expression &e)
{
    const auto pred = [](const expression &ex) {
        if (const auto *fptr = std::get_if<func>(&ex.value())) {
            return fptr->is_time_dependent();
        } else {
            // NOTE: non-function expressions cannot be time-dependent.
            return false;
        }
    };

    return ex_traverse_test_any(func_set, sargs_set, e, pred);
}

} // namespace

} // namespace detail

// Determine if an expression is time-dependent.
bool is_time_dependent(const expression &ex)
{
    // NOTE: these sets will contain pointers to functions and
    // arguments sets which are *not* time-dependent.
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;

    return detail::is_time_dependent(func_set, sargs_set, ex);
}

bool is_time_dependent(const std::vector<expression> &v_ex)
{
    detail::void_ptr_set func_set;
    detail::sargs_ptr_set sargs_set;

    for (const auto &ex : v_ex) {
        if (detail::is_time_dependent(func_set, sargs_set, ex)) {
            return true;
        }
    }

    return false;
}

namespace detail
{

std::vector<expression> split_sums_for_decompose(const std::vector<expression> &v_ex)
{
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    const auto tfunc = [](const expression &ex) {
        // NOTE: split on a power of two so that the internal pairwise sums are rounded up exactly.
        constexpr std::uint32_t decompose_split = 8;

        return sum_split(ex, decompose_split);
    };

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(ex_traverse_transform_nodes(func_map, sargs_map, e, {}, tfunc));
    }

    return retval;
}

std::vector<expression> split_prods_for_decompose(const std::vector<expression> &v_ex, std::uint32_t split)
{
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    const auto tfunc = [split](const expression &ex) { return prod_split(ex, split); };

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(ex_traverse_transform_nodes(func_map, sargs_map, e, {}, tfunc));
    }

    return retval;
}

// Replace sum({square(x), square(y), ...}) with sum_sq({x, y, ...}).
std::vector<expression> sums_to_sum_sqs_for_decompose(const std::vector<expression> &v_ex)
{
    void_ptr_map<const expression> func_map;
    sargs_ptr_map<const func_args::shared_args_t> sargs_map;

    const auto tfunc = [](const expression &ex) { return sum_to_sum_sq(ex); };

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(ex_traverse_transform_nodes(func_map, sargs_map, e, {}, tfunc));
    }

    return retval;
}

// NOTE: this does not have any specific mathematical meaning, it
// is just used to impose an ordering on expressions.
bool ex_less_than(const expression &e1, const expression &e2)
{
    return std::visit(
        [](const auto &v1, const auto &v2) {
            using type1 = uncvref_t<decltype(v1)>;
            using type2 = uncvref_t<decltype(v2)>;

            if constexpr (std::is_same_v<type1, type2>) {
                // Handle the cases where v1 and v2
                // are the same type.
                if constexpr (std::is_same_v<variable, type1>) {
                    // Both arguments are variables: use lexicographic comparison.
                    return v1.name() < v2.name();
                } else if constexpr (std::is_same_v<param, type1>) {
                    // Both arguments are params: compare the indices.
                    return v1.idx() < v2.idx();
                } else if constexpr (std::is_same_v<number, type1>) {
                    // Both arguments are numbers: compare.
                    return v1 < v2;
                } else if constexpr (std::is_same_v<func, type1>) {
                    // Both arguments are functions: compare.
                    return v1 < v2;
                } else {
                    static_assert(always_false_v<type1>);
                }
            } else {
                // Handle mixed types.
                if constexpr (std::is_same_v<number, type1>) {
                    // Number is always less than non-number.
                    return true;
                } else if constexpr (std::is_same_v<func, type1>) {
                    // Function never less than non-function.
                    return false;
                } else if constexpr (std::is_same_v<variable, type1>) {
                    // Variable less than function, greater than anything elses.
                    return std::is_same_v<type2, func>;
                } else if constexpr (std::is_same_v<param, type1>) {
                    // Param greater than number, less than anything else.
                    return !std::is_same_v<type2, number>;
                } else {
                    static_assert(always_false_v<type1>);
                }
            }
        },
        e1.value(), e2.value());
}

} // namespace detail

HEYOKA_END_NAMESPACE
