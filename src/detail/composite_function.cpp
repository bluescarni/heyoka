// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <concepts>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/regex.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/composite_function.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Serialisation.
void composite_function_impl::save(boost::archive::binary_oarchive &oa, unsigned) const
{
    oa << boost::serialization::base_object<func_base>(*this);
    oa << m_ex;
}

void composite_function_impl::load(boost::archive::binary_iarchive &ia, unsigned)
{
    ia >> boost::serialization::base_object<func_base>(*this);
    ia >> m_ex;
}

namespace
{

// NOTE: certain characters have special meaning in the name of a composite function - the round brackets signal
// function invocation while the pound sign is used to enumerate the arguments. Thus, these characters are not allowed
// to appear in the original function names.
constexpr auto forbidden_composite_function_chars = R"(()#)";

// NOLINTNEXTLINE(cert-err58-cpp,bugprone-throwing-static-initialization)
const boost::regex forbidden_composite_function_regex(fmt::format(R"([{}])", forbidden_composite_function_chars));

template <typename S>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters,misc-no-recursion)
void construct_composite_function_impl(std::string &name, std::string &llvm_name, std::vector<expression> &fargs,
                                       boost::safe_numerics::safe<S> &arg_idx, const expression &ex)
{
    std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&name, &llvm_name, &fargs, &arg_idx]<typename T>(const T &v) {
            if constexpr (std::same_as<T, func>) {
                // function node.

                // Check for forbidden chars in the function names.
                if (boost::regex_search(v.get_name(), forbidden_composite_function_regex)) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "Invalid character(s) detected in the function name '{}' during the construction of "
                        "a composite function: the characters '{}' are forbidden",
                        v.get_name(), forbidden_composite_function_chars));
                }
                if (boost::regex_search(v.get_llvm_name(), forbidden_composite_function_regex)) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "Invalid character(s) detected in the llvm function name '{}' during the construction of "
                        "a composite function: the characters '{}' are forbidden",
                        v.get_llvm_name(), forbidden_composite_function_chars));
                }

                // Append the function names and open the arguments sections.
                name += v.get_name();
                name += '(';

                llvm_name += v.get_llvm_name();
                llvm_name += '(';

                // Recurse into the function arguments.
                const auto &args = v.args();
                const auto nargs = args.size();
                for (S i = 0; i < nargs; ++i) {
                    construct_composite_function_impl(name, llvm_name, fargs, arg_idx, args[i]);

                    if (i != nargs - 1u) {
                        name += ',';
                        llvm_name += ',';
                    }
                }

                // Close the arguments sections.
                name += ')';
                llvm_name += ')';
            } else {
                // Non-function node. This becomes an argument for the composite function.

                // Construct the argument identifier.
                const auto arg_str = fmt::format("#{}", static_cast<S>(arg_idx++));

                // Append it to the composite function names.
                name += arg_str;
                llvm_name += arg_str;

                // Append the argument.
                fargs.emplace_back(v);
            }
        },
        ex.value());
}

// Construct the names and arguments of a composite function from the original function expression.
//
// The original function expression is traversed in a depth-first fashion. Each time a function node is encountered, its
// name is appended to the composite function name. Each time a non-function node is encountered, it becomes an argument
// to the composite function.
//
// For instance, given the original expression sin(x + y) + z, the composite function name becomes something like
// "composite|sum(sin(sum(#0,#1)),#2)" and the arguments become [x, y, z].
//
// NOTE: here we are using a simple recursive traversal of the expression for simplicity. Composite functions are
// supposed to be created from short expressions in any case, thuse the recursive approach should be adequate.
func_base construct_composite_function(const expression &ex)
{
    if (!std::holds_alternative<func>(ex.value())) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Cannot construct a composite function from the expression '{}': the expression is not a function", ex));
    }

    // Initialise the names and arguments.
    std::string name = "composite|", llvm_name = name;
    std::vector<expression> fargs;
    // NOTE: this is used to keep track of the current composite function argument.
    boost::safe_numerics::safe<std::vector<expression>::size_type> arg_idx = 0;

    // Run the recursive depth-first traversal.
    construct_composite_function_impl(name, llvm_name, fargs, arg_idx, ex);

    return func_base(std::move(name), std::move(llvm_name), std::move(fargs));
}

} // namespace

composite_function_impl::composite_function_impl() : composite_function_impl(expression{func{null_func{}}}) {}

composite_function_impl::composite_function_impl(const expression &ex)
    : func_base(construct_composite_function(ex)), m_ex(ex)
{
}

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
llvm::Value *composite_function_llvm_evaluate_impl(llvm_state &s, const expression &ex, auto &arg_idx,
                                                   const std::vector<llvm::Value *> &args, llvm::Type *val_t,
                                                   llvm::Value *time_ptr, const bool high_accuracy)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&]<typename T>(const T &v) {
            if constexpr (std::same_as<T, func>) {
                // Function node.

                // Construct the list of llvm arguments for this function by recursing into its arguments.
                std::vector<llvm::Value *> local_args;
                local_args.reserve(v.args().size());

                for (const auto &arg : v.args()) {
                    local_args.push_back(
                        composite_function_llvm_evaluate_impl(s, arg, arg_idx, args, val_t, time_ptr, high_accuracy));
                }

                // Run llvm evaluation for the current function node.
                return v.llvm_evaluate(s, local_args, val_t, time_ptr, high_accuracy);
            } else {
                // Non-function node. Its evaluation is taken directly from the list of arguments of the composite
                // function.
                return args.at(arg_idx++);
            }
        },
        ex.value());
}

} // namespace

// llvm evaluation.
//
// The original expression is traversed depth-first. Each time a function node is encountered, a list of llvm arguments
// is constructed and then passed to the llvm_evaluate() function of the function node.
//
// NOTE: here we are using a simple recursive traversal of the expression for simplicity. Composite functions are
// supposed to be created from short expressions in any case, thuse the recursive approach should be adequate.
llvm::Value *composite_function_impl::llvm_evaluate(llvm_state &s, const std::vector<llvm::Value *> &args,
                                                    llvm::Type *val_t, llvm::Value *time_ptr,
                                                    const bool high_accuracy) const
{
    std::vector<llvm::Value *>::size_type arg_idx = 0;
    auto *ret = composite_function_llvm_evaluate_impl(s, m_ex, arg_idx, args, val_t, time_ptr, high_accuracy);

    assert(arg_idx == args.size());

    return ret;
}

expression composite_function(const expression &ex)
{
    return expression{func{composite_function_impl{ex}}};
}

} // namespace detail

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp,bugprone-throwing-static-initialization)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::composite_function_impl)
