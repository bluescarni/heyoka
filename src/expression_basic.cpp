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
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

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

#include <heyoka/detail/fast_unordered.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
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

// NOTE: here we define a couple of stack data structures to be used when traversing
// the nodes of an expression. We use boost::small_vector in order to avoid paying for
// heap allocations on small expressions.
constexpr std::size_t static_stack_size = 20;

using traverse_stack = boost::container::small_vector<std::pair<const expression *, bool>, static_stack_size>;

template <typename T>
using return_stack = boost::container::small_vector<std::optional<T>, static_stack_size>;

// NOTE: the idea here is to have two stacks: the usual stack ('stack') for depth-first traversal, and a stack of copies
// of the expression nodes ('copy_stack'). As we traverse the expression, we keep on pusing to copy_stack copies of the
// nodes.
//
// When we encounter a function node, we initially add an empty optional in copy_stack, since we cannot copy it yet as
// we need to copy its arguments first. As we proceed with the traversal, the second time we encounter the function
// node we are sure that all its arguments have been copied. The copies of the arguments are at the tail end of
// copy_stack. We can then proceed to pop them to construct the copy of the function node.
expression copy_impl(auto &func_map, auto &stack, auto &copy_stack, const expression &e)
{
    assert(stack.empty());
    assert(copy_stack.empty());

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the traversal stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        if (const auto *f_ptr = std::get_if<func>(&cur_ex->value())) {
            // Function (i.e., internal) node.
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            if (auto it = func_map.find(f_id); it != func_map.end()) {
                // We already copied the current function. Fetch the copy from the cache
                // and add it to the copy stack.
                assert(!visited);
                copy_stack.emplace_back(it->second);
                continue;
            }

            if (visited) {
                // We have now visited and copied all the children of the function node
                // (i.e., the function arguments). The copies are at the tail end of
                // copy_stack. We will be popping the copies from copy_stack and use them
                // to initialise a new copy of the function.
                std::vector<expression> new_args;
                const auto n_args = f.args().size();
                new_args.reserve(n_args);
                for (decltype(new_args.size()) i = 0; i < n_args; ++i) {
                    // NOTE: the copy stack must not be empty and its last element
                    // also cannot be empty.
                    assert(!copy_stack.empty());
                    assert(copy_stack.back());

                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    new_args.push_back(std::move(*copy_stack.back()));
                    copy_stack.pop_back();
                }

                // Create the new copy of the function.
                auto ex_copy = expression{f.copy(std::move(new_args))};

                // Add it to the cache.
                assert(!func_map.contains(f_id));
                func_map.emplace(f_id, ex_copy);

                // Add it to copy_stack.
                // NOTE: the copy stack must not be empty and its last element
                // must be empty (it is supposed to be the empty function we
                // pushed the first time we visited).
                assert(!copy_stack.empty());
                assert(!copy_stack.back());
                copy_stack.back().emplace(std::move(ex_copy));
            } else {
                // It is the first time we visit this function. Re-add it to the stack
                // with visited=true, and add all of its arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }

                // Add an empty copy of the function to copy_stack. We will perform
                // the actual copy once we have copied all arguments.
                copy_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);
            copy_stack.emplace_back(*cur_ex);
        }
    }

    assert(copy_stack.size() == 1u);
    assert(copy_stack.back());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto ret = std::move(*copy_stack.back());
    copy_stack.pop_back();

    return ret;
}

} // namespace

} // namespace detail

expression copy(const expression &e)
{
    detail::funcptr_map<expression> func_map;
    detail::traverse_stack stack;
    detail::return_stack<expression> copy_stack;

    return detail::copy_impl(func_map, stack, copy_stack, e);
}

std::vector<expression> copy(const std::vector<expression> &v_ex)
{
    detail::funcptr_map<expression> func_map;
    detail::traverse_stack stack;
    detail::return_stack<expression> copy_stack;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        ret.push_back(detail::copy_impl(func_map, stack, copy_stack, ex));
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

void get_variables_impl(auto &func_set, auto &stack, auto &s_set, const expression &e)
{
    assert(stack.empty());

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        if (const auto *f_ptr = std::get_if<func>(&cur_ex->value())) {
            // Function (i.e., internal) node.
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            if (auto it = func_set.find(f_id); it != func_set.end()) {
                // We already got the list of variables for the current function,
                // no need to do anything else.
                assert(!visited);
                continue;
            }

            if (visited) {
                // We have now visited all the children of the function node and determined
                // the list of variables. We just have to add f_id to the cache so that we
                // won't repeat the same computation again.
                assert(!func_set.contains(f_id));
                func_set.emplace(f_id);
            } else {
                // It is the first time we visit this function. Re-add it to the stack
                // with visited=true, and add all of its arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            if (const auto *var_ptr = std::get_if<variable>(&cur_ex->value())) {
                s_set.emplace(var_ptr->name());
            }
        }
    }
}

} // namespace

} // namespace detail

std::vector<std::string> get_variables(const expression &e)
{
    detail::funcptr_set func_set;
    detail::fast_uset<std::string> s_set;
    detail::traverse_stack stack;

    detail::get_variables_impl(func_set, stack, s_set, e);

    // Turn the set into an ordered vector.
    std::vector retval(s_set.begin(), s_set.end());
    std::ranges::sort(retval);

    return retval;
}

std::vector<std::string> get_variables(const std::vector<expression> &v_ex)
{
    detail::funcptr_set func_set;
    detail::fast_uset<std::string> s_set;
    detail::traverse_stack stack;

    for (const auto &ex : v_ex) {
        detail::get_variables_impl(func_set, stack, s_set, ex);
    }

    // Turn the set into an ordered vector.
    std::vector retval(s_set.begin(), s_set.end());
    std::ranges::sort(retval);

    return retval;
}

namespace detail
{

namespace
{

expression rename_variables_impl(auto &func_map, auto &stack, auto &rename_stack, const expression &e,
                                 const std::unordered_map<std::string, std::string> &repl_map)
{
    assert(stack.empty());
    assert(rename_stack.empty());

    // Seed the stack.
    stack.emplace_back(&e, false);

    while (!stack.empty()) {
        // Pop the traversal stack.
        const auto [cur_ex, visited] = stack.back();
        stack.pop_back();

        if (const auto *f_ptr = std::get_if<func>(&cur_ex->value())) {
            // Function (i.e., internal) node.
            const auto &f = *f_ptr;

            // Fetch the function id.
            const auto *f_id = f.get_ptr();

            if (auto it = func_map.find(f_id); it != func_map.end()) {
                // We already renamed the current function. Fetch the renamed copy from the cache
                // and add it to the rename stack.
                assert(!visited);
                rename_stack.emplace_back(it->second);
                continue;
            }

            if (visited) {
                // We have now visited and renamed all the children of the function node
                // (i.e., the function arguments). The renamed children are at the tail end of
                // rename_stack. We will be popping the copies from rename_stack and use them
                // to initialise a new copy of the function.
                std::vector<expression> new_args;
                const auto n_args = f.args().size();
                new_args.reserve(n_args);
                for (decltype(new_args.size()) i = 0; i < n_args; ++i) {
                    // NOTE: the rename stack must not be empty and its last element
                    // also cannot be empty.
                    assert(!rename_stack.empty());
                    assert(rename_stack.back());

                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    new_args.push_back(std::move(*rename_stack.back()));
                    rename_stack.pop_back();
                }

                // Create the new copy of the function.
                auto ex_copy = expression{f.copy(std::move(new_args))};

                // Add it to the cache.
                assert(!func_map.contains(f_id));
                func_map.emplace(f_id, ex_copy);

                // Add it to rename_stack.
                // NOTE: the rename stack must not be empty and its last element
                // must be empty (it is supposed to be the empty function we
                // pushed the first time we visited).
                assert(!rename_stack.empty());
                assert(!rename_stack.back());
                rename_stack.back().emplace(std::move(ex_copy));
            } else {
                // It is the first time we visit this function. Re-add it to the stack
                // with visited=true, and add all of its arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
                }

                // Add an empty copy of the function to rename_stack. We will perform
                // the actual rename once we have renamed all arguments.
                rename_stack.emplace_back();
            }
        } else {
            // Non-function (i.e., leaf) node.
            assert(!visited);

            // Check if the current expression is a variable whose name shows up
            // in repl_map. If it is not, we will fall through and push a copy
            // of cur_ex into the rename stack.
            if (const auto *var_ptr = std::get_if<variable>(&cur_ex->value())) {
                const auto it = repl_map.find(var_ptr->name());
                if (it != repl_map.end()) {
                    rename_stack.emplace_back(it->second);
                    continue;
                }
            }

            rename_stack.emplace_back(*cur_ex);
        }
    }

    assert(rename_stack.size() == 1u);
    assert(rename_stack.back());

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto ret = std::move(*rename_stack.back());
    rename_stack.pop_back();

    return ret;
}

} // namespace

} // namespace detail

expression rename_variables(const expression &e, const std::unordered_map<std::string, std::string> &repl_map)
{
    detail::funcptr_map<expression> func_map;
    detail::traverse_stack stack;
    detail::return_stack<expression> rename_stack;

    return detail::rename_variables_impl(func_map, stack, rename_stack, e, repl_map);
}

std::vector<expression> rename_variables(const std::vector<expression> &v_ex,
                                         const std::unordered_map<std::string, std::string> &repl_map)
{
    detail::funcptr_map<expression> func_map;
    detail::traverse_stack stack;
    detail::return_stack<expression> rename_stack;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &ex : v_ex) {
        retval.push_back(detail::rename_variables_impl(func_map, stack, rename_stack, ex, repl_map));
    }

    return retval;
}

namespace detail
{

// NOTE: at this time there is no apparent need for a vectorised version of this.
// In case we ever need one, remember to adapt the end of the function to pop
// the return value from hash_stack. The simple function optimisation would also need
// to be reconsidered.
// NOLINTNEXTLINE(bugprone-exception-escape)
std::size_t hash(const expression &ex) noexcept
{
    detail::funcptr_map<std::size_t> func_map;
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

            if (auto it = func_map.find(f_id); it != func_map.end()) {
                // We already computed the hash of the current function. Fetch it from the cache
                // and add it to the hash stack.
                assert(!visited);
                hash_stack.emplace_back(it->second);
                continue;
            }

            if (visited) {
                // We have now visited and computed the hash of all the children of the function node
                // (i.e., the function arguments). The hashes are at the tail end of
                // hash_stack. We will be popping and combining them in order to compute the hash
                // of the function.

                // NOTE: the hash of a function is obtained by combining the function
                // name with the hashes of the function arguments.
                auto seed = std::hash<std::string>{}(f.get_name());
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

                // Add the hash to the cache.
                // NOTE: here we apply an optimisation: if the stack is empty,
                // we are at the end of the while loop and we won't need the cached
                // value in the future. Like this, we avoid an unnecessary heap allocation
                // if the expression we are hashing is a simple non-recursive function.
                assert(!func_map.contains(f_id));
                if (!stack.empty()) {
                    func_map.emplace(f_id, seed);
                }

                // Add it to hash_stack.
                // NOTE: the hash stack must not be empty and its last element
                // must be empty (it is supposed to be the empty hash we
                // pushed the first time we visited).
                assert(!hash_stack.empty());
                assert(!hash_stack.back());
                hash_stack.back().emplace(seed);
            } else {
                // It is the first time we visit this function. Re-add it to the stack
                // with visited=true, and add all of its arguments to the stack as well.
                stack.emplace_back(cur_ex, true);

                for (const auto &ex : f.args()) {
                    stack.emplace_back(&ex, false);
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
                        assert(false);
                        return 0;
                    } else {
                        return std::hash<T>{}(arg);
                    }
                },
                cur_ex->value()));
        }
    }

    assert(hash_stack.size() == 1u);
    assert(hash_stack.back());

    // NOTE: usually we pop back the only element of hash_stack here, but because
    // we do not have a vectorised counterpart of hashing we can omit this step
    // (which is needed in order to prepare hash_stack for the next expression in
    // the vector).
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

// NOTE: this always returns a number > 0, unless an overflow
// occurs due to the expression being too large. In such case,
// zero is returned.
std::size_t get_n_nodes(const expression &e)
{
    detail::funcptr_map<std::size_t> func_map;
    detail::traverse_stack stack{{&e, false}};
    boost::safe_numerics::safe<std::size_t> retval = 0;

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

                if (auto it = func_map.find(f_id); it != func_map.end()) {
                    // We already computed the number of nodes for the current
                    // function, add it to retval.
                    assert(!visited);
                    retval += it->second;
                    continue;
                }

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
                    assert(!func_map.contains(f_id));
                    func_map.emplace(f_id, n_nodes);

                    // NOTE: by incrementing by 1 here we are accounting only for the function node
                    // itself. It is not necessary to account for the total number of nodes n_nodes
                    // because all the children nodes were already counted during visitation
                    // of the current function.
                    ++retval;
                } else {
                    // It is the first time we visit this function. Re-add it to the stack
                    // with visited=true, and add all of its arguments to the stack as well.
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

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
expression subs(funcptr_map<expression> &func_map, const expression &ex,
                const std::unordered_map<std::string, expression> &smap)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_map, &smap](const auto &arg) {
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
                    new_args.push_back(subs(func_map, orig_arg, smap));
                }

                // Create a copy of arg with the new arguments.
                auto tmp = arg.copy(std::move(new_args));

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

expression subs(const expression &e, const std::unordered_map<std::string, expression> &smap)
{
    detail::funcptr_map<expression> func_map;

    auto ret = detail::subs(func_map, e, smap);

    return ret;
}

std::vector<expression> subs(const std::vector<expression> &v_ex,
                             const std::unordered_map<std::string, expression> &smap)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        ret.push_back(detail::subs(func_map, e, smap));
    }

    return ret;
}

namespace detail
{

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
expression subs(funcptr_map<expression> &func_map, const expression &ex, const std::map<expression, expression> &smap)
{
    if (auto it = smap.find(ex); it != smap.end()) {
        // ex is in the substitution map, return the value it maps to.
        return it->second;
    }

    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_map, &smap](const auto &arg) {
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
                    new_args.push_back(subs(func_map, orig_arg, smap));
                }

                // Create a copy of arg with the new arguments.
                auto tmp = arg.copy(std::move(new_args));

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

// NOTE: in the subs() functions we are using a std::map, instead of
// std::unordered_map, because hashing always requires traversing
// the whole expression, while comparisons can exit early. This becomes
// important while traversing the expression "e" and checking if its internal
// subexpressions are contained in smap. With hashing, we run into a quadratic
// complexity scenario because at each step of the traversal we have again
// to traverse the entire subexpression in order to compute its hash value.
expression subs(const expression &e, const std::map<expression, expression> &smap)
{
    detail::funcptr_map<expression> func_map;

    auto ret = detail::subs(func_map, e, smap);

    return ret;
}

std::vector<expression> subs(const std::vector<expression> &v_ex, const std::map<expression, expression> &smap)
{
    detail::funcptr_map<expression> func_map;

    std::vector<expression> ret;
    ret.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        ret.push_back(detail::subs(func_map, e, smap));
    }

    return ret;
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

expression par_impl::operator[](std::uint32_t idx) const
{
    return expression{param{idx}};
}

} // namespace detail

namespace detail
{

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
std::uint32_t get_param_size(detail::funcptr_set &func_set, const expression &ex)
{
    std::uint32_t retval = 0;

    std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&retval, &func_set]<typename T>(const T &v) {
            if constexpr (std::same_as<T, param>) {
                retval = v.idx() + boost::safe_numerics::safe<std::uint32_t>(1);
            } else if constexpr (std::same_as<T, func>) {
                const auto f_id = v.get_ptr();

                if (func_set.contains(f_id)) {
                    // We already computed the number of params for the current
                    // function, exit.
                    // NOTE: we will be end up returning 0 as retval. This is ok
                    // because the number of params for the current function was
                    // already considered in the calculation.
                    return;
                }

                // Recursively fetch the number of params from the function arguments.
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

std::uint32_t get_param_size(const std::vector<expression> &v_ex)
{
    std::uint32_t retval = 0;

    detail::funcptr_set func_set;

    for (const auto &ex : v_ex) {
        retval = std::max(retval, detail::get_param_size(func_set, ex));
    }

    return retval;
}

namespace detail
{

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
void get_params(std::unordered_set<std::uint32_t> &idx_set, detail::funcptr_set &func_set, const expression &ex)
{
    std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
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
    std::ranges::sort(idx_vec);

    // Transform the sorted indices into a vector of
    // sorted parameter expressions.
    std::vector<expression> retval;
    retval.reserve(static_cast<decltype(retval.size())>(idx_vec.size()));
    std::ranges::transform(idx_vec, std::back_inserter(retval), [](auto idx) { return par[idx]; });

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
    std::ranges::sort(idx_vec);

    // Transform the sorted indices into a vector of
    // sorted parameter expressions.
    std::vector<expression> retval;
    retval.reserve(static_cast<decltype(retval.size())>(idx_vec.size()));
    std::ranges::transform(idx_vec, std::back_inserter(retval), [](auto idx) { return par[idx]; });

    return retval;
}

namespace detail
{

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
bool is_time_dependent(funcptr_set &func_set, const expression &ex)
{
    // - If ex is a function, check if it is time-dependent, or
    //   if any of its arguments is time-dependent,
    // - otherwise, return false.
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_set](const auto &v) {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                const auto f_id = v.get_ptr();

                // Did we already determine that v is *not* time-dependent?
                if (const auto it = func_set.find(f_id); it != func_set.end()) {
                    return false;
                }

                // Check if the function is intrinsically time-dependent.
                if (v.is_time_dependent()) {
                    return true;
                }

                // The function does *not* intrinsically depend on time.
                // Check its arguments.
                for (const auto &a : v.args()) {
                    if (is_time_dependent(func_set, a)) {
                        // A time-dependent argument was found, return true.
                        return true;
                    }
                }

                // Update the cache.
                [[maybe_unused]] const auto [_, flag] = func_set.emplace(f_id);

                // An expression cannot contain itself.
                assert(flag);

                return false;
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
    // NOTE: this will contain pointers to (sub)expressions
    // which are *not* time-dependent.
    detail::funcptr_set func_set;

    return detail::is_time_dependent(func_set, ex);
}

bool is_time_dependent(const std::vector<expression> &v_ex)
{
    detail::funcptr_set func_set;

    for (const auto &ex : v_ex) {
        if (detail::is_time_dependent(func_set, ex)) {
            return true;
        }
    }

    return false;
}

namespace detail
{

namespace
{

// NOTE: the default split value is a power of two so that the
// internal pairwise sums are rounded up exactly.
constexpr std::uint32_t decompose_split = 8u;

// NOLINTNEXTLINE(misc-no-recursion)
expression split_sums_for_decompose(funcptr_map<expression> &func_map, const expression &ex)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&](const auto &v) {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, func>) {
                const auto *f_id = v.get_ptr();

                // Check if we already split sums on ex.
                if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                // Split sums on the function arguments.
                std::vector<expression> new_args;
                new_args.reserve(v.args().size());
                for (const auto &orig_arg : v.args()) {
                    new_args.push_back(split_sums_for_decompose(func_map, orig_arg));
                }

                // Create a copy of v with the split arguments.
                auto f_copy = v.copy(std::move(new_args));

                // After having taken care of the arguments, split
                // v itself.
                auto ret = sum_split(expression{std::move(f_copy)}, decompose_split);

                // Put the return value into the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return ret;
            } else {
                return ex;
            }
        },
        ex.value());
}

} // namespace

std::vector<expression> split_sums_for_decompose(const std::vector<expression> &v_ex)
{
    funcptr_map<expression> func_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(split_sums_for_decompose(func_map, e));
    }

    return retval;
}

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
expression split_prods_for_decompose(funcptr_map<expression> &func_map, const expression &ex, std::uint32_t split)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_map, &ex, split](const auto &v) {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, func>) {
                const auto *f_id = v.get_ptr();

                // Check if we already split prods on ex.
                if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                // Split prods on the function arguments.
                std::vector<expression> new_args;
                new_args.reserve(v.args().size());
                for (const auto &orig_arg : v.args()) {
                    new_args.push_back(split_prods_for_decompose(func_map, orig_arg, split));
                }

                // Create a copy of v with the split arguments.
                auto f_copy = v.copy(std::move(new_args));

                // After having taken care of the arguments, split
                // v itself.
                auto ret = prod_split(expression{std::move(f_copy)}, split);

                // Put the return value into the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return ret;
            } else {
                return ex;
            }
        },
        ex.value());
}

} // namespace

std::vector<expression> split_prods_for_decompose(const std::vector<expression> &v_ex, std::uint32_t split)
{
    funcptr_map<expression> func_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(split_prods_for_decompose(func_map, e, split));
    }

    return retval;
}

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
expression sums_to_sum_sqs_for_decompose(funcptr_map<expression> &func_map, const expression &ex)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&](const auto &v) {
            if constexpr (std::is_same_v<uncvref_t<decltype(v)>, func>) {
                const auto *f_id = v.get_ptr();

                // Check if we already converted sums to sum_sqs on ex.
                if (const auto it = func_map.find(f_id); it != func_map.end()) {
                    return it->second;
                }

                // Convert sums to sum_sqs on the function arguments.
                std::vector<expression> new_args;
                new_args.reserve(v.args().size());
                for (const auto &orig_arg : v.args()) {
                    new_args.push_back(sums_to_sum_sqs_for_decompose(func_map, orig_arg));
                }

                // Create a copy of v with the split arguments.
                auto f_copy = v.copy(std::move(new_args));

                // After having taken care of the arguments, convert
                // v itself.
                auto ret = sum_to_sum_sq(expression{std::move(f_copy)});

                // Put the return value into the cache.
                [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
                // NOTE: an expression cannot contain itself.
                assert(flag); // LCOV_EXCL_LINE

                return ret;
            } else {
                return ex;
            }
        },
        ex.value());
}

} // namespace

// Replace sum({square(x), square(y), ...}) with sum_sq({x, y, ...}).
std::vector<expression> sums_to_sum_sqs_for_decompose(const std::vector<expression> &v_ex)
{
    funcptr_map<expression> func_map;

    std::vector<expression> retval;
    retval.reserve(v_ex.size());

    for (const auto &e : v_ex) {
        retval.push_back(sums_to_sum_sqs_for_decompose(func_map, e));
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
