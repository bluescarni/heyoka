// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <random>
#include <string>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{
template <typename It, typename Rng>
It random_element(It start, It end, Rng &g)
{
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}
} // namespace detail

random_expression::random_expression(const std::vector<std::string> &vars, ::std::uint64_t seed)
    : m_vars(vars), m_e(seed), m_b_funcs() {
    // These are the default blocks for a random expression.
    m_bos = {binary_operator::type::add, binary_operator::type::sub, binary_operator::type::mul, binary_operator::type::div};
    m_u_funcs = {heyoka::sin, heyoka::cos};
    m_b_funcs = {};
    };

expression random_expression::operator()(unsigned min_depth, unsigned max_depth, unsigned depth)
{
    std::uniform_real_distribution<double> rng01(0.0, 1.0);
    std::uniform_real_distribution<double> rngm11(-1.0, 1.0);

    // First we decide what node type this will be.
    node_type type;
    if (depth < min_depth) {
        // If the node depth is below the minimum desired, we force leaves (num or var) to be not selected
        double n_bos = m_bos.size();
        double n_u_fun = m_u_funcs.size();
        double n_b_fun = m_b_funcs.size();
        std::discrete_distribution<> dis({n_bos*4, n_u_fun*2, n_b_fun});
        switch (dis(m_e)) {
            case 0:
                type = node_type::bo;
                break;
            case 1:
                type = node_type::u_fun;
                break;
            case 2:
                type = node_type::b_fun;
                break;
        }
    } else if (depth >= max_depth) {
        // If the node depth is above the maximum desired, we force leaves (num or var) to be selected
        double n_var = m_vars.size();
        std::discrete_distribution<> dis({n_var*4, 1.});
        switch (dis(m_e)) {
            case 0:
                type = node_type::var;
                break;
            case 1:
                type = node_type::num;
                break;
        }
    } else {
        // else we can get anything
        double n_bos = m_bos.size();
        double n_u_fun = m_u_funcs.size();
        double n_b_fun = m_b_funcs.size();
        double n_var = m_vars.size();
        std::discrete_distribution<> dis({n_bos*8, n_u_fun*2, n_b_fun, n_var*4, 1.});
        switch (dis(m_e)) {
            case 0:
                type = node_type::bo;
                break;
            case 1:
                type = node_type::u_fun;
                break;
            case 2:
                type = node_type::b_fun;
                break;
            case 3:
                type = node_type::var;
                break;
            case 4:
                type = node_type::num;
                break;
        }    }
    // Once we know the node type we create one at random out of the user defined possible choices
    switch (type) {
        case node_type::num: {
            // We return a random number in -10, 10
            auto value = rngm11(m_e) * 10.;
            return expression{number{value}};
            break;
        }
        case node_type::var: {
            // We return one of the variables in m_vars
            auto symbol = *random_element(m_vars.begin(), m_vars.end(), m_e);
            return expression{variable{symbol}};
            break;
        }
        case node_type::bo: {
            // We return one of the binary oprators in m_bos with randomly constructed arguments
            auto bo_type = *random_element(m_bos.begin(), m_bos.end(), m_e);
            return expression{binary_operator(bo_type, this->operator()(min_depth, max_depth, depth + 1),
                                              this->operator()(min_depth, max_depth, depth + 1))};
            break;
        }
        case node_type::u_fun: {
            // We return one of the unary functions in m_u_funcs with randomly constructed argument
            auto u_f = *random_element(m_u_funcs.begin(), m_u_funcs.end(), m_e);
            return u_f(this->operator()(min_depth, max_depth, depth + 1));
            break;
        }
        case node_type::b_fun: {
            // We return one of the binary functions in m_b_funcs with randomly constructed argument
            auto b_f = *random_element(m_b_funcs.begin(), m_b_funcs.end(), m_e);
            return b_f(this->operator()(min_depth, max_depth, depth + 1),
                       this->operator()(min_depth, max_depth, depth + 1));
            break;
        }
        default:
            throw;
    }
};



} // namespace heyoka
