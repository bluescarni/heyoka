// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/math_functions.hpp>

namespace heyoka
{

random_expression::random_expression(const std::vector<std::string> &vars, ::std::uint64_t seed)
    : m_vars(vars), m_e(seed){};

expression random_expression::operator()(unsigned min_depth, unsigned max_depth, unsigned depth) {};

std::vector<binary_operator::type> random_expression::m_bos
    = {binary_operator::type::add, binary_operator::type::sub, binary_operator::type::mul, binary_operator::type::div};
std::vector<expression (*)(expression)> random_expression::m_u_funcs = {heyoka::sin, heyoka::cos, heyoka::log};
std::vector<expression (*)(expression, expression)> random_expression::m_b_funcs = {heyoka::pow};

} // namespace heyoka
