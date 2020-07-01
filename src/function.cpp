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
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/Attributes.h>

#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>

namespace heyoka
{

function::function(std::string s, std::vector<expression> args)
    : m_name(std::move(s)), m_display_name(m_name), m_args(std::make_unique<std::vector<expression>>(std::move(args)))
{
}

function::function(const function &f)
    : m_disable_verify(f.m_disable_verify), m_name(f.m_name), m_display_name(f.m_display_name),
      m_args(std::make_unique<std::vector<expression>>(f.args())), m_attributes(f.m_attributes), m_ty(f.m_ty),
      m_diff_f(f.m_diff_f), m_eval_dbl_f(f.m_eval_dbl_f)
{
}

function::function(function &&) noexcept = default;

function::~function() = default;

std::string &function::name()
{
    return m_name;
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

const std::string &function::name() const
{
    return m_name;
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

bool operator==(const function &f1, const function &f2)
{
    return f1.name() == f2.name() && f1.display_name() == f2.display_name() && f1.args() == f2.args()
           && f1.attributes() == f2.attributes()
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
        throw std::invalid_argument("The function '" + f.name()
                                    + "' does not provide an implementation of the derivative");
    }
}

double eval_dbl(const function &f, const std::unordered_map<std::string, double> &map)
{
    auto &ef = f.eval_dbl_f();

    if (ef) {
        return ef(f.args(), map);
    } else {
        throw std::invalid_argument("The function '" + f.name()
                                    + "' does not provide an implementation of double evaluation");
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

} // namespace heyoka
