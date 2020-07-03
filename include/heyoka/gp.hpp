// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_GP_HPP
#define HEYOKA_GP_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/function.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC random_expression
{
public:
    enum node_types { num, var, bo, u_fun, b_fun };

private:
    static std::vector<binary_operator::type> m_bos;
    static std::vector<expression (*)(expression)> m_u_funcs;
    static std::vector<expression (*)(expression, expression)> m_b_funcs;

    std::vector<std::string> m_vars;
    detail::splitmix64 m_e;

public:
    explicit random_expression(const std::vector<std::string> &, ::std::uint64_t);
    expression operator()(unsigned, unsigned, unsigned = 0u);
};

} // namespace heyoka

#endif
