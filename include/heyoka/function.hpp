// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
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

private:
    bool m_disable_verify = false;
    std::string m_name, m_display_name;
    std::unique_ptr<std::vector<expression>> m_args;
    std::vector<llvm::Attribute::AttrKind> m_attributes;
    type m_ty;
    diff_t m_diff_f;

public:
    explicit function(std::string, std::vector<expression>);
    function(const function &);
    function(function &&) noexcept;
    ~function();

    std::string &name();
    std::string &display_name();
    std::vector<expression> &args();

    const std::string &name() const;
    const std::string &display_name() const;
    const std::vector<expression> &args() const;
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const function &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const function &);

} // namespace heyoka

#endif
