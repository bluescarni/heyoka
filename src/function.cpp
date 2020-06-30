// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
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
#include <string>
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
      m_diff_f(f.m_diff_f)
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
    return m_ty;
}

function::diff_t &function::diff_f()
{
    return m_diff_f;
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
    return m_ty;
}

const function::diff_t &function::diff_f() const
{
    return m_diff_f;
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
           && f1.attributes() == f2.attributes() && f1.ty() == f2.ty()
           && static_cast<bool>(f1.diff_f()) == static_cast<bool>(f2.diff_f());
}

bool operator!=(const function &f1, const function &f2)
{
    return !(f1 == f2);
}

} // namespace heyoka
