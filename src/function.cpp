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
#include <iterator>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/tfp.hpp>
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

} // namespace

} // namespace detail

function::function(std::vector<expression> args)
    : m_args(std::make_unique<std::vector<expression>>(std::move(args))),
      // Default implementation of Taylor decomposition.
      m_taylor_decompose_f(detail::function_default_td)
{
}

function::function(const function &f)
    : m_disable_verify(f.m_disable_verify), m_name_dbl(f.m_name_dbl), m_name_ldbl(f.m_name_ldbl),
#if defined(HEYOKA_HAVE_REAL128)
      m_name_f128(f.m_name_f128),
#endif
      m_display_name(f.m_display_name), m_args(std::make_unique<std::vector<expression>>(f.args())),
      m_attributes_dbl(f.m_attributes_dbl), m_attributes_ldbl(f.m_attributes_ldbl),
#if defined(HEYOKA_HAVE_REAL128)
      m_attributes_f128(f.m_attributes_f128),
#endif
      m_ty_dbl(f.ty_dbl()), m_ty_ldbl(f.ty_ldbl()),
#if defined(HEYOKA_HAVE_REAL128)
      m_ty_f128(f.ty_f128()),
#endif
      m_diff_f(f.m_diff_f), m_eval_dbl_f(f.m_eval_dbl_f), m_eval_batch_dbl_f(f.m_eval_batch_dbl_f),
      m_eval_num_dbl_f(f.m_eval_num_dbl_f), m_deval_num_dbl_f(f.m_deval_num_dbl_f),
      m_taylor_decompose_f(f.m_taylor_decompose_f), m_taylor_init_batch_dbl_f(f.m_taylor_init_batch_dbl_f),
      m_taylor_init_batch_ldbl_f(f.m_taylor_init_batch_ldbl_f),
#if defined(HEYOKA_HAVE_REAL128)
      m_taylor_init_batch_f128_f(f.m_taylor_init_batch_f128_f),
#endif
      m_taylor_diff_batch_dbl_f(f.m_taylor_diff_batch_dbl_f), m_taylor_diff_batch_ldbl_f(f.m_taylor_diff_batch_ldbl_f)
#if defined(HEYOKA_HAVE_REAL128)
      ,
      m_taylor_diff_batch_f128_f(f.m_taylor_diff_batch_f128_f)
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

bool &function::disable_verify()
{
    return m_disable_verify;
}

std::string &function::name_dbl()
{
    return m_name_dbl;
}

std::string &function::name_ldbl()
{
    return m_name_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

std::string &function::name_f128()
{
    return m_name_f128;
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

std::vector<llvm::Attribute::AttrKind> &function::attributes_dbl()
{
    return m_attributes_dbl;
}

std::vector<llvm::Attribute::AttrKind> &function::attributes_ldbl()
{
    return m_attributes_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<llvm::Attribute::AttrKind> &function::attributes_f128()
{
    return m_attributes_f128;
}

#endif

function::type &function::ty_dbl()
{
    assert(m_ty_dbl >= type::internal && m_ty_dbl <= type::builtin);
    return m_ty_dbl;
}

function::type &function::ty_ldbl()
{
    assert(m_ty_ldbl >= type::internal && m_ty_ldbl <= type::builtin);
    return m_ty_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

function::type &function::ty_f128()
{
    assert(m_ty_f128 >= type::internal && m_ty_f128 <= type::builtin);
    return m_ty_f128;
}

#endif

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

function::taylor_init_batch_t &function::taylor_init_batch_dbl_f()
{
    return m_taylor_init_batch_dbl_f;
}

function::taylor_init_batch_t &function::taylor_init_batch_ldbl_f()
{
    return m_taylor_init_batch_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

function::taylor_init_batch_t &function::taylor_init_batch_f128_f()
{
    return m_taylor_init_batch_f128_f;
}

#endif

function::taylor_diff_batch_t &function::taylor_diff_batch_dbl_f()
{
    return m_taylor_diff_batch_dbl_f;
}

function::taylor_diff_batch_t &function::taylor_diff_batch_ldbl_f()
{
    return m_taylor_diff_batch_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

function::taylor_diff_batch_t &function::taylor_diff_batch_f128_f()
{
    return m_taylor_diff_batch_f128_f;
}

#endif

const bool &function::disable_verify() const
{
    return m_disable_verify;
}

const std::string &function::name_dbl() const
{
    return m_name_dbl;
}

const std::string &function::name_ldbl() const
{
    return m_name_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

const std::string &function::name_f128() const
{
    return m_name_f128;
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

const std::vector<llvm::Attribute::AttrKind> &function::attributes_dbl() const
{
    return m_attributes_dbl;
}

const std::vector<llvm::Attribute::AttrKind> &function::attributes_ldbl() const
{
    return m_attributes_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

const std::vector<llvm::Attribute::AttrKind> &function::attributes_f128() const
{
    return m_attributes_f128;
}

#endif

const function::type &function::ty_dbl() const
{
    assert(m_ty_dbl >= type::internal && m_ty_dbl <= type::builtin);
    return m_ty_dbl;
}

const function::type &function::ty_ldbl() const
{
    assert(m_ty_ldbl >= type::internal && m_ty_ldbl <= type::builtin);
    return m_ty_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::type &function::ty_f128() const
{
    assert(m_ty_f128 >= type::internal && m_ty_f128 <= type::builtin);
    return m_ty_f128;
}

#endif

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

const function::taylor_init_batch_t &function::taylor_init_batch_dbl_f() const
{
    return m_taylor_init_batch_dbl_f;
}

const function::taylor_init_batch_t &function::taylor_init_batch_ldbl_f() const
{
    return m_taylor_init_batch_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::taylor_init_batch_t &function::taylor_init_batch_f128_f() const
{
    return m_taylor_init_batch_f128_f;
}

#endif

const function::taylor_diff_batch_t &function::taylor_diff_batch_dbl_f() const
{
    return m_taylor_diff_batch_dbl_f;
}

const function::taylor_diff_batch_t &function::taylor_diff_batch_ldbl_f() const
{
    return m_taylor_diff_batch_ldbl_f;
}

#if defined(HEYOKA_HAVE_REAL128)

const function::taylor_diff_batch_t &function::taylor_diff_batch_f128_f() const
{
    return m_taylor_diff_batch_f128_f;
}

#endif

std::ostream &operator<<(std::ostream &os, const function &f)
{
    os << f.display_name() << '(';

    const auto &args = f.args();
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        os << args[i];
        if (i != args.size() - 1u) {
            os << ',';
        }
    }

    return os << ')';
}

void swap(function &f0, function &f1) noexcept
{
    std::swap(f0.disable_verify(), f1.disable_verify());
    std::swap(f0.name_dbl(), f1.name_dbl());
    std::swap(f0.name_ldbl(), f1.name_ldbl());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.name_f128(), f1.name_f128());
#endif
    std::swap(f0.display_name(), f1.display_name());
    std::swap(f0.args(), f1.args());
    std::swap(f0.attributes_dbl(), f1.attributes_dbl());
    std::swap(f0.attributes_ldbl(), f1.attributes_ldbl());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.attributes_f128(), f1.attributes_f128());
#endif
    std::swap(f0.ty_dbl(), f1.ty_dbl());
    std::swap(f0.ty_ldbl(), f1.ty_ldbl());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.ty_f128(), f1.ty_f128());
#endif

    std::swap(f0.diff_f(), f1.diff_f());

    std::swap(f0.eval_dbl_f(), f1.eval_dbl_f());
    std::swap(f0.eval_batch_dbl_f(), f1.eval_batch_dbl_f());
    std::swap(f0.eval_num_dbl_f(), f1.eval_num_dbl_f());
    std::swap(f0.deval_num_dbl_f(), f1.deval_num_dbl_f());

    std::swap(f0.taylor_decompose_f(), f1.taylor_decompose_f());
    std::swap(f0.taylor_init_batch_dbl_f(), f1.taylor_init_batch_dbl_f());
    std::swap(f0.taylor_init_batch_ldbl_f(), f1.taylor_init_batch_ldbl_f());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.taylor_init_batch_f128_f(), f1.taylor_init_batch_f128_f());
#endif
    std::swap(f0.taylor_diff_batch_dbl_f(), f1.taylor_diff_batch_dbl_f());
    std::swap(f0.taylor_diff_batch_ldbl_f(), f1.taylor_diff_batch_ldbl_f());
#if defined(HEYOKA_HAVE_REAL128)
    std::swap(f0.taylor_diff_batch_f128_f(), f1.taylor_diff_batch_f128_f());
#endif
}

std::size_t hash(const function &f)
{
    auto retval = std::hash<bool>{}(f.disable_verify());
    retval += std::hash<std::string>{}(f.name_dbl());
    retval += std::hash<std::string>{}(f.name_ldbl());
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<std::string>{}(f.name_f128());
#endif
    retval += std::hash<std::string>{}(f.display_name());

    for (const auto &arg : f.args()) {
        retval += hash(arg);
    }

    for (const auto &attr : f.attributes_dbl()) {
        retval += std::hash<llvm::Attribute::AttrKind>{}(attr);
    }

    for (const auto &attr : f.attributes_ldbl()) {
        retval += std::hash<llvm::Attribute::AttrKind>{}(attr);
    }

#if defined(HEYOKA_HAVE_REAL128)
    for (const auto &attr : f.attributes_f128()) {
        retval += std::hash<llvm::Attribute::AttrKind>{}(attr);
    }
#endif

    retval += std::hash<function::type>{}(f.ty_dbl());
    retval += std::hash<function::type>{}(f.ty_ldbl());
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<function::type>{}(f.ty_f128());
#endif

    retval += std::hash<bool>{}(static_cast<bool>(f.diff_f()));

    retval += std::hash<bool>{}(static_cast<bool>(f.eval_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.eval_batch_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.eval_num_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.deval_num_dbl_f()));

    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_decompose_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_init_batch_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_init_batch_ldbl_f()));
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_init_batch_f128_f()));
#endif
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_diff_batch_dbl_f()));
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_diff_batch_ldbl_f()));
#if defined(HEYOKA_HAVE_REAL128)
    retval += std::hash<bool>{}(static_cast<bool>(f.taylor_diff_batch_f128_f()));
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
    return f1.name_dbl() == f2.name_dbl() && f1.name_ldbl() == f2.name_ldbl()
#if defined(HEYOKA_HAVE_REAL128)
           && f1.name_f128() == f2.name_f128()
#endif
           && f1.display_name() == f2.display_name() && f1.args() == f2.args()
           && f1.attributes_dbl() == f2.attributes_dbl() && f1.attributes_ldbl() == f2.attributes_ldbl()
#if defined(HEYOKA_HAVE_REAL128)
           && f1.attributes_f128() == f2.attributes_f128()
#endif
           && f1.ty_dbl() == f2.ty_dbl() && f1.ty_ldbl() == f2.ty_ldbl()
#if defined(HEYOKA_HAVE_REAL128)
           && f1.ty_f128() == f2.ty_f128()
#endif
           // NOTE: we have no way of comparing the content of std::function,
           // thus we just check if the std::function members contain something.
           && static_cast<bool>(f1.diff_f()) == static_cast<bool>(f2.diff_f())
           && static_cast<bool>(f1.eval_dbl_f()) == static_cast<bool>(f2.eval_dbl_f())
           && static_cast<bool>(f1.eval_batch_dbl_f()) == static_cast<bool>(f2.eval_batch_dbl_f())
           && static_cast<bool>(f1.eval_num_dbl_f()) == static_cast<bool>(f2.eval_num_dbl_f())
           && static_cast<bool>(f1.deval_num_dbl_f()) == static_cast<bool>(f2.deval_num_dbl_f())
           && static_cast<bool>(f1.taylor_decompose_f()) == static_cast<bool>(f2.taylor_decompose_f())
           && static_cast<bool>(f1.taylor_init_batch_dbl_f()) == static_cast<bool>(f2.taylor_init_batch_dbl_f())
           && static_cast<bool>(f1.taylor_init_batch_ldbl_f()) == static_cast<bool>(f2.taylor_init_batch_ldbl_f())
#if defined(HEYOKA_HAVE_REAL128)
           && static_cast<bool>(f1.taylor_init_batch_f128_f()) == static_cast<bool>(f2.taylor_init_batch_f128_f())
#endif
           && static_cast<bool>(f1.taylor_diff_batch_dbl_f()) == static_cast<bool>(f2.taylor_diff_batch_dbl_f())
           && static_cast<bool>(f1.taylor_diff_batch_ldbl_f()) == static_cast<bool>(f2.taylor_diff_batch_ldbl_f())
#if defined(HEYOKA_HAVE_REAL128)
           && static_cast<bool>(f1.taylor_diff_batch_f128_f()) == static_cast<bool>(f2.taylor_diff_batch_f128_f())
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
const std::string &function_name_from_type(const function &f)
{
    if constexpr (std::is_same_v<T, double>) {
        return f.name_dbl();
    } else if constexpr (std::is_same_v<T, long double>) {
        return f.name_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return f.name_f128();
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type");
    }
}

template <typename T>
const function::type &function_ty_from_type(const function &f)
{
    if constexpr (std::is_same_v<T, double>) {
        return f.ty_dbl();
    } else if constexpr (std::is_same_v<T, long double>) {
        return f.ty_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return f.ty_f128();
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type");
    }
}

template <typename T>
const auto &function_attributes_from_type(const function &f)
{
    if constexpr (std::is_same_v<T, double>) {
        return f.attributes_dbl();
    } else if constexpr (std::is_same_v<T, long double>) {
        return f.attributes_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return f.attributes_f128();
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type");
    }
}

template <typename T>
llvm::Value *function_codegen_impl(llvm_state &s, const function &f)
{
    if (f.disable_verify()) {
        s.verify() = false;
    }

    // Create the function arguments.
    std::vector<llvm::Value *> args_v;
    for (const auto &arg : f.args()) {
        args_v.push_back(codegen<T>(s, arg));
        assert(args_v.back() != nullptr);
    }

    return function_codegen_from_values<T>(s, f, args_v);
}

} // namespace

// Function codegen with arguments passed as a vector of LLVM values. That is, the function
// arguments are *not* those stored in the function object, rather they are explicitly
// passed.
template <typename T>
llvm::Value *function_codegen_from_values(llvm_state &s, const function &f, const std::vector<llvm::Value *> &args_v)
{
    if (args_v.size() != f.args().size()) {
        throw std::invalid_argument(
            "Inconsistent arguments sizes when invoking function_codegen_from_values(): the function '"
            + f.display_name() + "' expects " + std::to_string(f.args().size()) + " argument(s), but "
            + std::to_string(args_v.size()) + " were provided instead");
    }

    llvm::Function *callee_f;
    const auto &f_name = function_name_from_type<T>(f);

    switch (function_ty_from_type<T>(f)) {
        case function::type::internal: {
            // Look up the name in the global module table.
            callee_f = s.module().getFunction(f_name);

            if (!callee_f) {
                throw std::invalid_argument("Unknown internal function: '" + f_name + "'");
            }

            if (callee_f->isDeclaration()) {
                throw std::invalid_argument("The internal function '" + f_name + "' cannot be just a declaration");
            }

            break;
        }
        case function::type::external: {
            // Look up the name in the global module table.
            callee_f = s.module().getFunction(f_name);

            if (callee_f) {
                // The function declaration exists already. Check that it is only a
                // declaration and not a definition.
                if (!callee_f->isDeclaration()) {
                    throw std::invalid_argument(
                        "Cannot call the function '" + f_name
                        + "' as an external function, because it is defined as an internal module function");
                }
            } else {
                // The function does not exist yet, make the prototype.
                std::vector<llvm::Type *> arg_types(args_v.size(), to_llvm_type<T>(s.context()));
                auto *ft = llvm::FunctionType::get(to_llvm_type<T>(s.context()), arg_types, false);
                assert(ft);
                callee_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, f_name, &s.module());
                assert(callee_f);

                // Add the function attributes.
                for (const auto &att : function_attributes_from_type<T>(f)) {
                    callee_f->addFnAttr(att);
                }
            }

            break;
        }
        default: {
            // Builtin.
            const auto intrinsic_ID = llvm::Function::lookupIntrinsicID(f_name);
            if (intrinsic_ID == 0) {
                throw std::invalid_argument("Cannot fetch the ID of the intrinsic '" + f_name + "'");
            }

            // NOTE: for generic intrinsics to work, we need to specify
            // the desired argument types. See:
            // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
            // And the docs of the getDeclaration() function.
            const std::vector<llvm::Type *> arg_types(args_v.size(), to_llvm_type<T>(s.context()));

            callee_f = llvm::Intrinsic::getDeclaration(&s.module(), intrinsic_ID, arg_types);

            if (!callee_f) {
                throw std::invalid_argument("Error getting the declaration of the intrinsic '" + f_name + "'");
            }

            if (!callee_f->isDeclaration()) {
                // It does not make sense to have a definition of a builtin.
                throw std::invalid_argument("The intrinsic '" + f_name + "' must be only declared, not defined");
            }
        }
    }

    // Check the number of arguments.
    if (callee_f->arg_size() != args_v.size()) {
        throw std::invalid_argument("Incorrect # of arguments passed while calling the function '" + f.display_name()
                                    + "': " + std::to_string(callee_f->arg_size()) + " are expected, but "
                                    + std::to_string(args_v.size()) + " were provided instead");
    }

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, args_v, "calltmp");
    assert(r != nullptr);
    // NOTE: we used to have r->setTailCall(true) here, but:
    // - when optimising, the tail call attribute is automatically
    //   added,
    // - it is not 100% clear to me whether it is always safe to enable it:
    // https://llvm.org/docs/CodeGenerator.html#tail-calls

    return r;
}

// Explicit instantiations of function_codegen_from_values().
template llvm::Value *function_codegen_from_values<double>(llvm_state &, const function &,
                                                           const std::vector<llvm::Value *> &);
template llvm::Value *function_codegen_from_values<long double>(llvm_state &, const function &,
                                                                const std::vector<llvm::Value *> &);

#if defined(HEYOKA_HAVE_REAL128)

template llvm::Value *function_codegen_from_values<mppp::real128>(llvm_state &, const function &,
                                                                  const std::vector<llvm::Value *> &);

#endif

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

llvm::Value *taylor_init_batch_dbl(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size)
{
    auto &ti = f.taylor_init_batch_dbl_f();
    if (!ti) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for double Taylor init");
    }
    return ti(s, f, arr, batch_idx, batch_size, vector_size);
}

llvm::Value *taylor_init_batch_ldbl(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                    std::uint32_t batch_size, std::uint32_t vector_size)
{
    auto &ti = f.taylor_init_batch_ldbl_f();
    if (!ti) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for long double Taylor init");
    }
    return ti(s, f, arr, batch_idx, batch_size, vector_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_init_batch_f128(llvm_state &s, const function &f, llvm::Value *arr, std::uint32_t batch_idx,
                                    std::uint32_t batch_size, std::uint32_t vector_size)
{
    auto &ti = f.taylor_init_batch_f128_f();
    if (!ti) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for float128 Taylor init");
    }
    return ti(s, f, arr, batch_idx, batch_size, vector_size);
}

#endif

llvm::Value *taylor_diff_batch_dbl(llvm_state &s, const function &f, std::uint32_t idx, std::uint32_t order,
                                   std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                   std::uint32_t batch_size, std::uint32_t vector_size,
                                   const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    auto &td = f.taylor_diff_batch_dbl_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for double Taylor diff");
    }
    return td(s, f, idx, order, n_uvars, diff_arr, batch_idx, batch_size, vector_size, cd_uvars);
}

llvm::Value *taylor_diff_batch_ldbl(llvm_state &s, const function &f, std::uint32_t idx, std::uint32_t order,
                                    std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                    std::uint32_t batch_size, std::uint32_t vector_size,
                                    const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    auto &td = f.taylor_diff_batch_ldbl_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for long double Taylor diff");
    }
    return td(s, f, idx, order, n_uvars, diff_arr, batch_idx, batch_size, vector_size, cd_uvars);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_diff_batch_f128(llvm_state &s, const function &f, std::uint32_t idx, std::uint32_t order,
                                    std::uint32_t n_uvars, llvm::Value *diff_arr, std::uint32_t batch_idx,
                                    std::uint32_t batch_size, std::uint32_t vector_size,
                                    const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    auto &td = f.taylor_diff_batch_f128_f();
    if (!td) {
        throw std::invalid_argument("The function '" + f.display_name()
                                    + "' does not provide a function for float128 Taylor diff");
    }
    return td(s, f, idx, order, n_uvars, diff_arr, batch_idx, batch_size, vector_size, cd_uvars);
}

#endif

tfp taylor_u_init_dbl(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                      bool high_accuracy)
{
    throw;
}

tfp taylor_u_init_ldbl(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                       bool high_accuracy)
{
    throw;
}

#if defined(HEYOKA_HAVE_REAL128)

tfp taylor_u_init_f128(llvm_state &s, const function &f, const std::vector<tfp> &arr, std::uint32_t batch_size,
                       bool high_accuracy)
{
    throw;
}

#endif

} // namespace heyoka
