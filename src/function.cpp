// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
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
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

function::function(std::vector<expression> args) : m_args(std::make_unique<std::vector<expression>>(std::move(args))) {}

function::function(const function &f)
    : m_disable_verify(f.m_disable_verify), m_dbl_name(f.m_dbl_name), m_ldbl_name(f.m_ldbl_name),
      m_display_name(f.m_display_name), m_args(std::make_unique<std::vector<expression>>(f.args())),
      m_attributes(f.m_attributes), m_ty(f.m_ty), m_diff_f(f.m_diff_f), m_eval_dbl_f(f.m_eval_dbl_f),
      m_eval_batch_dbl_f(f.m_eval_batch_dbl_f), m_eval_num_dbl_f(f.m_eval_num_dbl_f),
      m_deval_num_dbl_f(f.m_deval_num_dbl_f)
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

std::string &function::dbl_name()
{
    return m_dbl_name;
}

std::string &function::ldbl_name()
{
    return m_ldbl_name;
}

std::string &function::display_name()
{
    return m_display_name;
}

std::vector<expression> &function::args()
{
    assert(m_args);
    return *m_args;
}

std::vector<llvm::Attribute::AttrKind> &function::attributes()
{
    return m_attributes;
}

function::type &function::ty()
{
    assert(m_ty >= type::internal && m_ty <= type::builtin);
    return m_ty;
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

const bool &function::disable_verify() const
{
    return m_disable_verify;
}

const std::string &function::dbl_name() const
{
    return m_dbl_name;
}

const std::string &function::ldbl_name() const
{
    return m_ldbl_name;
}

const std::string &function::display_name() const
{
    return m_display_name;
}

const std::vector<expression> &function::args() const
{
    assert(m_args);
    return *m_args;
}

const std::vector<llvm::Attribute::AttrKind> &function::attributes() const
{
    return m_attributes;
}

const function::type &function::ty() const
{
    assert(m_ty >= type::internal && m_ty <= type::builtin);
    return m_ty;
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
    return f1.dbl_name() == f2.dbl_name() && f1.ldbl_name() == f2.ldbl_name() && f1.display_name() == f2.display_name()
           && f1.args() == f2.args() && f1.attributes() == f2.attributes()
           && f1.ty() == f2.ty()
           // NOTE: we have no way of comparing the content of std::function,
           // thus we just check if the std::function members contain something.
           && static_cast<bool>(f1.diff_f()) == static_cast<bool>(f2.diff_f())
           && static_cast<bool>(f1.eval_dbl_f()) == static_cast<bool>(f2.eval_dbl_f());
}

bool operator!=(const function &f1, const function &f2)
{
    return !(f1 == f2);
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

void eval_batch_dbl(const function &f, const std::unordered_map<std::string, std::vector<double>> &map,
                    std::vector<double> &out_values)
{
    auto &ef = f.eval_batch_dbl_f();
    if (ef) {
        ef(f.args(), map, out_values);
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

void update_node_values_dbl(const function &f, const std::unordered_map<std::string, double> &map,
                            std::vector<double> &node_values,
                            const std::vector<std::vector<unsigned>> &node_connections, unsigned &node_counter)
{
    const unsigned node_id = node_counter;
    node_counter++;
    // We have to recurse first as to make sure node_values is filled before being accessed later.
    for (auto i = 0u; i < f.args().size(); ++i) {
        update_node_values_dbl(f.args()[i], map, node_values, node_connections, node_counter);
    }
    // Then we compute
    std::vector<double> in_values(f.args().size());
    for (auto i = 0u; i < f.args().size(); ++i) {
        in_values[i] = node_values[node_connections[node_id][i]];
    }
    node_values[node_id] = eval_num_dbl(f, in_values);
}

void update_grad_dbl(const function &f, const std::unordered_map<std::string, double> &map, std::unordered_map<std::string, double> &grad,
                     const std::vector<double> &node_values, const std::vector<std::vector<unsigned>> &node_connections,
                     unsigned &node_counter, double acc)
{
    const unsigned node_id = node_counter;
    node_counter++;
    std::vector<double> in_values(f.args().size());
    for (auto i = 0u; i <f.args().size(); ++i) {
        in_values[i] = node_values[node_connections[node_id][i]];
    }
    for (auto i = 0u; i < f.args().size(); ++i) {
        auto value = deval_num_dbl(f, in_values, i);
        update_grad_dbl(f.args()[i], map, grad, node_values, node_connections, node_counter, acc * value);
    }
}

void update_connections(const function &f, std::vector<std::vector<unsigned>> &node_connections, unsigned &node_counter)
{
    const unsigned node_id = node_counter;
    node_counter++;
    node_connections.push_back(std::vector<unsigned>(f.args().size()));
    for (auto i = 0u; i < f.args().size(); ++i) {
        node_connections[node_id][i] = node_counter;
        update_connections(f.args()[i], node_connections, node_counter);
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
        return f.dbl_name();
    } else if constexpr (std::is_same_v<T, long double>) {
        return f.ldbl_name();
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

    llvm::Function *callee_f;
    const auto &f_name = function_name_from_type<T>(f);

    switch (f.ty()) {
        case function::type::internal: {
            // Look up the name in the global module table.
            callee_f = s.module().getFunction(f_name);

            if (!callee_f) {
                throw std::invalid_argument("Unknown internal function: '" + f_name + "'");
            }

            if (callee_f->empty()) {
                // An internal function cannot be empty (i.e., we need declaration
                // and definition).
                throw std::invalid_argument("The internal function '" + f_name + "' is empty");
            }

            break;
        }
        case function::type::external: {
            // Look up the name in the global module table.
            callee_f = s.module().getFunction(f_name);

            if (callee_f) {
                // The function declaration exists already. Check that it is only a
                // declaration and not a definition.
                if (!callee_f->empty()) {
                    throw std::invalid_argument(
                        "Cannot call the function '" + f_name
                        + "' as an external function, because it is defined as an internal module function");
                }
            } else {
                // The function does not exist yet, make the prototype.
                std::vector<llvm::Type *> arg_types(f.args().size(), to_llvm_type<T>(s.context()));
                auto *ft = llvm::FunctionType::get(to_llvm_type<T>(s.context()), arg_types, false);
                assert(ft);
                callee_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, f_name, &s.module());
                assert(callee_f);

                // Add the function attributes.
                for (const auto &att : f.attributes()) {
                    callee_f->addFnAttr(att);
                }
            }

            break;
        }
        default: {
            // Builtin.
            const auto intrinsic_ID = llvm::Function::lookupIntrinsicID(f_name);
            if (!intrinsic_ID) {
                throw std::invalid_argument("Cannot fetch the ID of the intrinsic '" + f_name + "'");
            }

            // NOTE: for generic intrinsics to work, we need to specify
            // the desired argument types. See:
            // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
            // And the docs of the getDeclaration() function.
            const std::vector<llvm::Type *> arg_types(f.args().size(), to_llvm_type<T>(s.context()));

            callee_f = llvm::Intrinsic::getDeclaration(&s.module(), intrinsic_ID, arg_types);

            if (!callee_f) {
                throw std::invalid_argument("Error getting the declaration of the intrinsic '" + f_name + "'");
            }

            if (!callee_f->empty()) {
                // It does not make sense to have a definition of a builtin.
                throw std::invalid_argument("The intrinsic '" + f_name + "' must be an empty function");
            }
        }
    }

    // Check the number of arguments.
    if (callee_f->arg_size() != f.args().size()) {
        throw std::invalid_argument("Incorrect # of arguments passed in a function call: "
                                    + std::to_string(callee_f->arg_size()) + " are expected, but "
                                    + std::to_string(f.args().size()) + " were provided instead");
    }

    // Create the function arguments.
    std::vector<llvm::Value *> args_v;
    for (const auto &arg : f.args()) {
        args_v.push_back(detail::invoke_codegen<T>(s, arg));
        assert(args_v.back() != nullptr);
    }

    auto r = s.builder().CreateCall(callee_f, args_v, "calltmp");
    assert(r != nullptr);
    // NOTE: not sure what this does exactly, but the optimized
    // IR from clang has this.
    r->setTailCall(true);

    return r;
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

} // namespace heyoka
