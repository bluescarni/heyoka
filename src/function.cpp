// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Default implementation of Taylor decomposition for a function.
std::vector<expression>::size_type function_default_td(function &&f, std::vector<expression> &u_vars_defs)
{
    // NOTE: this is a generalisation of the implementation
    // for the binary operators.
    for (auto &arg : f.args()) {
        if (const auto dres = taylor_decompose_in_place(std::move(arg), u_vars_defs)) {
            arg = expression{variable{"u_" + detail::li_to_string(dres)}};
        }
    }

    u_vars_defs.emplace_back(std::move(f));

    return u_vars_defs.size() - 1u;
}

// Default implementation of u_init for a function.
template <typename T>
llvm::Value *taylor_u_init_default(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                   std::uint32_t batch_size)
{
    // Do the initialisation for the function arguments.
    std::vector<llvm::Value *> args_v;
    for (const auto &arg : f.args()) {
        args_v.push_back(taylor_u_init<T>(s, arg, arr, batch_size));
    }

    return codegen_from_values<T>(s, f, args_v);
}

} // namespace

} // namespace detail

function::function(std::vector<expression> args)
    : m_args(std::make_unique<std::vector<expression>>(std::move(args))),
      // Default implementation of Taylor decomposition.
      m_taylor_decompose_f(detail::function_default_td),
      // Default implementation of Taylor init.
      m_taylor_u_init_dbl_f(detail::taylor_u_init_default<double>),
      m_taylor_u_init_ldbl_f(detail::taylor_u_init_default<long double>)
#if defined(HEYOKA_HAVE_REAL128)
      ,
      m_taylor_u_init_f128_f(detail::taylor_u_init_default<mppp::real128>)
#endif
{
}

function::function(const function &f)
    : m_codegen_dbl_f(f.m_codegen_dbl_f), m_codegen_ldbl_f(f.m_codegen_ldbl_f),
#if defined(HEYOKA_HAVE_REAL128)
      m_codegen_f128_f(f.m_codegen_f128_f),
#endif
      m_display_name(f.m_display_name), m_args(std::make_unique<std::vector<expression>>(f.args())),
      m_diff_f(f.m_diff_f), m_eval_dbl_f(f.m_eval_dbl_f), m_eval_batch_dbl_f(f.m_eval_batch_dbl_f),
      m_eval_num_dbl_f(f.m_eval_num_dbl_f), m_deval_num_dbl_f(f.m_deval_num_dbl_f),
      m_taylor_decompose_f(f.m_taylor_decompose_f), m_taylor_u_init_dbl_f(f.m_taylor_u_init_dbl_f),
      m_taylor_u_init_ldbl_f(f.m_taylor_u_init_ldbl_f),
#if defined(HEYOKA_HAVE_REAL128)
      m_taylor_u_init_f128_f(f.m_taylor_u_init_f128_f),
#endif
      m_taylor_diff_dbl_f(f.m_taylor_diff_dbl_f), m_taylor_diff_ldbl_f(f.m_taylor_diff_ldbl_f),
#if defined(HEYOKA_HAVE_REAL128)
      m_taylor_diff_f128_f(f.m_taylor_diff_f128_f),
#endif
      m_taylor_c_diff_func_dbl_f(f.m_taylor_c_diff_func_dbl_f),
      m_taylor_c_diff_func_ldbl_f(f.m_taylor_c_diff_func_ldbl_f)
#if defined(HEYOKA_HAVE_REAL128)
      ,
      m_taylor_c_diff_func_f128_f(f.m_taylor_c_diff_func_f128_f)
#endif
{
}

function::function(function &&) noexcept = default;

function::~function() = default;

function &function::operator=(const function &f)
{
    if (this != &f) {
        *this = function(f);
    }
    return *this;
}

function &function::operator=(function &&) noexcept = default;

function::codegen_t &function::codegen_dbl_f()
{
    return m_codegen_dbl_f;
}

function::codegen_t &function::codegen_ldbl_f()
{
    return m_codegen_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

function::codegen_t &function::codegen_f128_f()
{
    return m_codegen_f128_f;
}

#endif

std::string &function::display_name()
{
    return m_display_name;
}

std::vector<expression> &function::args()
{
    assert(m_args);
    return *m_args;
}

function::diff_t &function::diff_f()
{
    return m_diff_f;
}

function::eval_dbl_t &function::eval_dbl_f()
{
    return m_eval_dbl_f;
}

function::eval_batch_dbl_t &function::eval_batch_dbl_f()
{
    return m_eval_batch_dbl_f;
}

function::eval_num_dbl_t &function::eval_num_dbl_f()
{
    return m_eval_num_dbl_f;
}

function::deval_num_dbl_t &function::deval_num_dbl_f()
{
    return m_deval_num_dbl_f;
}

function::taylor_decompose_t &function::taylor_decompose_f()
{
    return m_taylor_decompose_f;
}

function::taylor_u_init_t &function::taylor_u_init_dbl_f()
{
    return m_taylor_u_init_dbl_f;
}

function::taylor_u_init_t &function::taylor_u_init_ldbl_f()
{
    return m_taylor_u_init_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

function::taylor_u_init_t &function::taylor_u_init_f128_f()
{
    return m_taylor_u_init_f128_f;
}

#endif

function::taylor_diff_t &function::taylor_diff_dbl_f()
{
    return m_taylor_diff_dbl_f;
}

function::taylor_diff_t &function::taylor_diff_ldbl_f()
{
    return m_taylor_diff_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

function::taylor_diff_t &function::taylor_diff_f128_f()
{
    return m_taylor_diff_f128_f;
}

#endif

function::taylor_c_diff_func_t &function::taylor_c_diff_func_dbl_f()
{
    return m_taylor_c_diff_func_dbl_f;
}

function::taylor_c_diff_func_t &function::taylor_c_diff_func_ldbl_f()
{
    return m_taylor_c_diff_func_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

function::taylor_c_diff_func_t &function::taylor_c_diff_func_f128_f()
{
    return m_taylor_c_diff_func_f128_f;
}

#endif

const function::codegen_t &function::codegen_dbl_f() const
{
    return m_codegen_dbl_f;
}

const function::codegen_t &function::codegen_ldbl_f() const
{
    return m_codegen_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::codegen_t &function::codegen_f128_f() const
{
    return m_codegen_f128_f;
}

#endif

const std::string &function::display_name() const
{
    return m_display_name;
}

const std::vector<expression> &function::args() const
{
    assert(m_args);
    return *m_args;
}

const function::diff_t &function::diff_f() const
{
    return m_diff_f;
}

const function::eval_dbl_t &function::eval_dbl_f() const
{
    return m_eval_dbl_f;
}

const function::eval_batch_dbl_t &function::eval_batch_dbl_f() const
{
    return m_eval_batch_dbl_f;
}

const function::eval_num_dbl_t &function::eval_num_dbl_f() const
{
    return m_eval_num_dbl_f;
}

const function::deval_num_dbl_t &function::deval_num_dbl_f() const
{
    return m_deval_num_dbl_f;
}

const function::taylor_decompose_t &function::taylor_decompose_f() const
{
    return m_taylor_decompose_f;
}

const function::taylor_u_init_t &function::taylor_u_init_dbl_f() const
{
    return m_taylor_u_init_dbl_f;
}

const function::taylor_u_init_t &function::taylor_u_init_ldbl_f() const
{
    return m_taylor_u_init_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::taylor_u_init_t &function::taylor_u_init_f128_f() const
{
    return m_taylor_u_init_f128_f;
}

#endif

const function::taylor_diff_t &function::taylor_diff_dbl_f() const
{
    return m_taylor_diff_dbl_f;
}

const function::taylor_diff_t &function::taylor_diff_ldbl_f() const
{
    return m_taylor_diff_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::taylor_diff_t &function::taylor_diff_f128_f() const
{
    return m_taylor_diff_f128_f;
}

#endif

const function::taylor_c_diff_func_t &function::taylor_c_diff_func_dbl_f() const
{
    return m_taylor_c_diff_func_dbl_f;
}

const function::taylor_c_diff_func_t &function::taylor_c_diff_func_ldbl_f() const
{
    return m_taylor_c_diff_func_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::taylor_c_diff_func_t &function::taylor_c_diff_func_f128_f() const
{
    return m_taylor_c_diff_func_f128_f;
}

#endif

std::ostream &operator<<(std::ostream &os, const function &f)
{
    os << f.display_name() << '(';

    const auto &args = f.args();
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        os << args[i];
        if (i != args.size() - 1u) {
            os << ", ";
        }
    }

    return os << ')';
}

void swap(function &f0, function &f1) noexcept
{
    std::swap(f0.codegen_dbl_f(), f1.codegen_dbl_f());
    std::swap(f0.codegen_ldbl_f(), f1.codegen_ldbl_f());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.codegen_f128_f(), f1.codegen_f128_f());
#endif

    std::swap(f0.display_name(), f1.display_name());

    std::swap(f0.args(), f1.args());

    std::swap(f0.diff_f(), f1.diff_f());

    std::swap(f0.eval_dbl_f(), f1.eval_dbl_f());
    std::swap(f0.eval_batch_dbl_f(), f1.eval_batch_dbl_f());
    std::swap(f0.eval_num_dbl_f(), f1.eval_num_dbl_f());
    std::swap(f0.deval_num_dbl_f(), f1.deval_num_dbl_f());

    std::swap(f0.taylor_decompose_f(), f1.taylor_decompose_f());
    std::swap(f0.taylor_u_init_dbl_f(), f1.taylor_u_init_dbl_f());
    std::swap(f0.taylor_u_init_ldbl_f(), f1.taylor_u_init_ldbl_f());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.taylor_u_init_f128_f(), f1.taylor_u_init_f128_f());
#endif
    std::swap(f0.taylor_diff_dbl_f(), f1.taylor_diff_dbl_f());
    std::swap(f0.taylor_diff_ldbl_f(), f1.taylor_diff_ldbl_f());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.taylor_diff_f128_f(), f1.taylor_diff_f128_f());
#endif
    std::swap(f0.taylor_c_diff_func_dbl_f(), f1.taylor_c_diff_func_dbl_f());
    std::swap(f0.taylor_c_diff_func_ldbl_f(), f1.taylor_c_diff_func_ldbl_f());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.taylor_c_diff_func_f128_f(), f1.taylor_c_diff_func_f128_f());
#endif
}

std::size_t hash(const function &f)
{
    auto retval = std::hash<std::string>{}(f.display_name());

    retval += std::hash<bool>{}(static_cast<bool>(f.codegen_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.codegen_ldbl_f()));
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<bool>{}(static_cast<bool>(f.codegen_f128_f()));
#endif

    for (const auto &arg : f.args()) {
        retval += hash(arg);
    }

    retval += std::hash<bool>{}(static_cast<bool>(f.diff_f()));

    retval += std::hash<bool>{}(static_cast<bool>(f.eval_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.eval_batch_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.eval_num_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.deval_num_dbl_f()));

    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_decompose_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_u_init_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_u_init_ldbl_f()));
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_u_init_f128_f()));
#endif
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_diff_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_diff_ldbl_f()));
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_diff_f128_f()));
#endif
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_c_diff_func_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_c_diff_func_ldbl_f()));
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_c_diff_func_f128_f()));
#endif

    return retval;
}

std::vector<std::string> get_variables(const function &f)
{
    std::vector<std::string> ret;

    for (const auto &arg : f.args()) {
        auto tmp = get_variables(arg);
        ret.insert(ret.end(), std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
        std::sort(ret.begin(), ret.end());
        ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
    }

    return ret;
}

void rename_variables(function &f, const std::unordered_map<std::string, std::string> &repl_map)
{
    for (auto &arg_ex : f.args()) {
        rename_variables(arg_ex, repl_map);
    }
}

bool operator==(const function &f1, const function &f2)
{
    return f1.display_name() == f2.display_name()
           && f1.args() == f2.args()
           // NOTE: we have no way of comparing the content of std::function,
           // thus we just check if the std::function members contain something.
           && static_cast<bool>(f1.codegen_dbl_f()) == static_cast<bool>(f2.codegen_dbl_f())
           && static_cast<bool>(f1.codegen_ldbl_f()) == static_cast<bool>(f2.codegen_ldbl_f())
#if defined(HEYOKA_HAVE_REAL128)
           && static_cast<bool>(f1.codegen_f128_f()) == static_cast<bool>(f2.codegen_f128_f())
#endif
           && static_cast<bool>(f1.diff_f()) == static_cast<bool>(f2.diff_f())
           && static_cast<bool>(f1.eval_dbl_f()) == static_cast<bool>(f2.eval_dbl_f())
           && static_cast<bool>(f1.eval_batch_dbl_f()) == static_cast<bool>(f2.eval_batch_dbl_f())
           && static_cast<bool>(f1.eval_num_dbl_f()) == static_cast<bool>(f2.eval_num_dbl_f())
           && static_cast<bool>(f1.deval_num_dbl_f()) == static_cast<bool>(f2.deval_num_dbl_f())
           && static_cast<bool>(f1.taylor_decompose_f()) == static_cast<bool>(f2.taylor_decompose_f())
           && static_cast<bool>(f1.taylor_u_init_dbl_f()) == static_cast<bool>(f2.taylor_u_init_dbl_f())
           && static_cast<bool>(f1.taylor_u_init_ldbl_f()) == static_cast<bool>(f2.taylor_u_init_ldbl_f())
#if defined(HEYOKA_HAVE_REAL128)
           && static_cast<bool>(f1.taylor_u_init_f128_f()) == static_cast<bool>(f2.taylor_u_init_f128_f())
#endif
           && static_cast<bool>(f1.taylor_diff_dbl_f()) == static_cast<bool>(f2.taylor_diff_dbl_f())
           && static_cast<bool>(f1.taylor_diff_ldbl_f()) == static_cast<bool>(f2.taylor_diff_ldbl_f())
#if defined(HEYOKA_HAVE_REAL128)
           && static_cast<bool>(f1.taylor_diff_f128_f()) == static_cast<bool>(f2.taylor_diff_f128_f())
#endif
           && static_cast<bool>(f1.taylor_c_diff_func_dbl_f()) == static_cast<bool>(f2.taylor_c_diff_func_dbl_f())
           && static_cast<bool>(f1.taylor_c_diff_func_ldbl_f()) == static_cast<bool>(f2.taylor_c_diff_func_ldbl_f())
#if defined(HEYOKA_HAVE_REAL128)
           && static_cast<bool>(f1.taylor_c_diff_func_f128_f()) == static_cast<bool>(f2.taylor_c_diff_func_f128_f())
#endif
        ;
}

bool operator!=(const function &f1, const function &f2)
{
    return !(f1 == f2);
}

expression subs(const function &f, const std::unordered_map<std::string, expression> &smap)
{
    // NOTE: not the most efficient implementation, as we end up
    // copying arguments which we will be discarding anyway later.
    // The only alternative seems to be copying manually all the function
    // members one by one however...
    auto tmp = f;

    for (auto &arg : tmp.args()) {
        arg = subs(arg, smap);
    }

    return expression{std::move(tmp)};
}

expression diff(const function &f, const std::string &s)
{
    auto &df = f.diff_f();

    if (df) {
        return df(f.args(), s);
    } else {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide an implementation of the derivative");
    }
}

double eval_dbl(const function &f, const std::unordered_map<std::string, double> &map)
{
    auto &ef = f.eval_dbl_f();

    if (ef) {
        return ef(f.args(), map);
    } else {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide an implementation of double evaluation");
    }
}

void eval_batch_dbl(std::vector<double> &out_values, const function &f,
                    const std::unordered_map<std::string, std::vector<double>> &map)
{
    auto &ef = f.eval_batch_dbl_f();
    if (ef) {
        ef(out_values, f.args(), map);
    } else {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide an implementation of batch evaluation for doubles");
    }
}

double eval_num_dbl(const function &f, const std::vector<double> &in)
{
    auto &ef = f.eval_num_dbl_f();

    if (ef) {
        return ef(in);
    } else {
        throw std::invalid_argument(
            "The function '" + f.display_name()
            + "' does not provide an implementation for its pure numerical evaluation over doubles.");
    }
}

double deval_num_dbl(const function &f, const std::vector<double> &in, std::vector<double>::size_type d)
{
    auto &ef = f.deval_num_dbl_f();

    if (ef) {
        return ef(in, d);
    } else {
        throw std::invalid_argument(
            "The function '" + f.display_name()
            + "' does not provide an implementation for the pure numerical evaluation of its derivative over doubles.");
    }
}

void update_node_values_dbl(std::vector<double> &node_values, const function &f,
                            const std::unordered_map<std::string, double> &map,
                            const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter)
{
    const auto node_id = node_counter;
    node_counter++;
    // We have to recurse first as to make sure node_values is filled before being accessed later.
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        update_node_values_dbl(node_values, f.args()[i], map, node_connections, node_counter);
    }
    // Then we compute
    std::vector<double> in_values(f.args().size());
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        in_values[i] = node_values[node_connections[node_id][i]];
    }
    node_values[node_id] = eval_num_dbl(f, in_values);
}

void update_grad_dbl(std::unordered_map<std::string, double> &grad, const function &f,
                     const std::unordered_map<std::string, double> &map, const std::vector<double> &node_values,
                     const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter,
                     double acc)
{
    const auto node_id = node_counter;
    node_counter++;
    std::vector<double> in_values(f.args().size());
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        in_values[i] = node_values[node_connections[node_id][i]];
    }
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        auto value = deval_num_dbl(f, in_values, i);
        update_grad_dbl(grad, f.args()[i], map, node_values, node_connections, node_counter, acc * value);
    }
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const function &f,
                        std::size_t &node_counter)
{
    const auto node_id = node_counter;
    node_counter++;
    node_connections.push_back(std::vector<std::size_t>(f.args().size()));
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        node_connections[node_id][i] = node_counter;
        update_connections(node_connections, f.args()[i], node_counter);
    };
}

namespace detail
{

namespace
{

template <typename T>
llvm::Value *function_codegen_impl(llvm_state &s, const function &f)
{
    // Create the function arguments.
    std::vector<llvm::Value *> args_v;
    for (const auto &arg : f.args()) {
        args_v.push_back(codegen<T>(s, arg));
        assert(args_v.back() != nullptr);
    }

    return codegen_from_values<T>(s, f, args_v);
}

} // namespace

} // namespace detail

llvm::Value *codegen_dbl(llvm_state &s, const function &f)
{
    return detail::function_codegen_impl<double>(s, f);
}

llvm::Value *codegen_ldbl(llvm_state &s, const function &f)
{
    return detail::function_codegen_impl<long double>(s, f);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *codegen_f128(llvm_state &s, const function &f)
{
    return detail::function_codegen_impl<mppp::real128>(s, f);
}

#endif

std::vector<expression>::size_type taylor_decompose_in_place(function &&f, std::vector<expression> &u_vars_defs)
{
    auto &tdf = f.taylor_decompose_f();
    if (!tdf) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for Taylor decomposition");
    }
    return tdf(std::move(f), u_vars_defs);
}

llvm::Value *taylor_u_init_dbl(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                               std::uint32_t batch_size)
{
    auto &ti = f.taylor_u_init_dbl_f();
    if (!ti) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for double Taylor init");
    }
    return ti(s, f, arr, batch_size);
}

llvm::Value *taylor_u_init_ldbl(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                std::uint32_t batch_size)
{
    auto &ti = f.taylor_u_init_ldbl_f();
    if (!ti) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for long double Taylor init");
    }
    return ti(s, f, arr, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_u_init_f128(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                                std::uint32_t batch_size)
{
    auto &ti = f.taylor_u_init_f128_f();
    if (!ti) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for float128 Taylor init");
    }
    return ti(s, f, arr, batch_size);
}

#endif

llvm::Value *taylor_diff_dbl(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &td = f.taylor_diff_dbl_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for double Taylor diff");
    }
    return td(s, f, arr, n_uvars, order, idx, batch_size);
}

llvm::Value *taylor_diff_ldbl(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &td = f.taylor_diff_ldbl_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for long double Taylor diff");
    }
    return td(s, f, arr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_diff_f128(llvm_state &s, const function &f, const std::vector<llvm::Value *> &arr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    auto &td = f.taylor_diff_f128_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for float128 Taylor diff");
    }
    return td(s, f, arr, n_uvars, order, idx, batch_size);
}

#endif

llvm::Function *taylor_c_diff_func_dbl(llvm_state &s, const function &f, std::uint32_t n_uvars,
                                       std::uint32_t batch_size)
{
    auto &td = f.taylor_c_diff_func_dbl_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for double Taylor diff in compact mode");
    }
    return td(s, f, n_uvars, batch_size);
}

llvm::Function *taylor_c_diff_func_ldbl(llvm_state &s, const function &f, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    auto &td = f.taylor_c_diff_func_ldbl_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for long double Taylor diff in compact mode");
    }
    return td(s, f, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *taylor_c_diff_func_f128(llvm_state &s, const function &f, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    auto &td = f.taylor_c_diff_func_f128_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for float128 Taylor diff in compact mode");
    }
    return td(s, f, n_uvars, batch_size);
}

#endif

} // namespace heyoka
