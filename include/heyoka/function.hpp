// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNCTION_HPP
#define HEYOKA_FUNCTION_HPP

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Attributes.h>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC function
{
public:
    enum class type { internal, external, builtin };

    using diff_t = std::function<expression(const std::vector<expression> &, const std::string &)>;
    using eval_dbl_t
        = std::function<double(const std::vector<expression> &, const std::unordered_map<std::string, double> &)>;

private:
    bool m_disable_verify = false;
    std::string m_name, m_display_name;
    std::unique_ptr<std::vector<expression>> m_args;
    std::vector<llvm::Attribute::AttrKind> m_attributes;
    type m_ty = type::internal;
    diff_t m_diff_f;
    eval_dbl_t m_eval_dbl_f;

public:
    explicit function(std::string, std::vector<expression>);
    function(const function &);
    function(function &&) noexcept;
    ~function();

    std::string &name();
    std::string &display_name();
    std::vector<expression> &args();
    std::vector<llvm::Attribute::AttrKind> &attributes();
    type &ty();
    diff_t &diff_f();
    eval_dbl_t &eval_dbl_f();

    const std::string &name() const;
    const std::string &display_name() const;
    const std::vector<expression> &args() const;
    const std::vector<llvm::Attribute::AttrKind> &attributes() const;
    const type &ty() const;
    const diff_t &diff_f() const;
    const eval_dbl_t &eval_dbl_f() const;
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const function &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const function &);

HEYOKA_DLL_PUBLIC bool operator==(const function &, const function &);
HEYOKA_DLL_PUBLIC bool operator!=(const function &, const function &);

HEYOKA_DLL_PUBLIC expression diff(const function &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const function &, const std::unordered_map<std::string, double> &);

} // namespace heyoka

#endif
