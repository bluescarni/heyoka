// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/math.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

template <typename It, typename Rng>
It random_element(It start, It end, Rng &g)
{
    std::uniform_int_distribution<decltype(std::distance(start, end))> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

void fetch_from_node_id_impl(expression &ex, std::size_t node_id, std::size_t &node_counter, expression *&ret)
{
    if (node_counter == node_id) {
        ret = &ex;
    } else {
        ++node_counter;
        std::visit(
            [node_id, &node_counter, &ret](auto &node) {
                using type = detail::uncvref_t<decltype(node)>;
                if constexpr (std::is_same_v<type, binary_operator>) {
                    for (auto &arg : node.args()) {
                        fetch_from_node_id_impl(arg, node_id, node_counter, ret);
                        if (ret) {
                            return;
                        }
                    }
                } else if constexpr (std::is_same_v<type, func>) {
                    for (auto [b, e] = node.get_mutable_args_it(); b != e; ++b) {
                        fetch_from_node_id_impl(*b, node_id, node_counter, ret);
                        if (ret) {
                            return;
                        }
                    }
                }
            },
            ex.value());
    }
}

void count_nodes_impl(const expression &e, std::size_t &node_counter)
{
    ++node_counter;
    std::visit(
        [&node_counter](auto &node) {
            using type = detail::uncvref_t<decltype(node)>;
            if constexpr (std::is_same_v<type, func> || std::is_same_v<type, binary_operator>) {
                for (auto &arg : node.args()) {
                    count_nodes_impl(arg, node_counter);
                }
            }
        },
        e.value());
}

} // namespace

} // namespace detail

expression_generator::expression_generator(const std::vector<std::string> &vars, splitmix64 &engine)
    : m_vars(vars), m_b_funcs(), m_e(engine)
{
    // These are the default blocks for a random expression.
    m_bos = {binary_operator::type::add, binary_operator::type::sub, binary_operator::type::mul,
             binary_operator::type::div};
    m_u_funcs = {heyoka::sin, heyoka::cos};
    m_b_funcs = {};
    m_range_dbl = 10;
    // Default weights for the probability of selecting a bo, unary, binary, a variable, a number.
    m_weights = {8., 2., 1., 4., 1.};
};

expression expression_generator::operator()(unsigned min_depth, unsigned max_depth, unsigned depth) const
{
    std::uniform_real_distribution<double> rng01(0.0, 1.0);
    std::uniform_real_distribution<double> rngm11(-1.0, 1.0);

    // First we decide what node type this will be.
    node_type type;
    if (depth < min_depth) {
        // If the node depth is below the minimum desired, we force leaves (num or var) to be not selected
        double n_bos = static_cast<double>(m_bos.size());
        double n_u_fun = static_cast<double>(m_u_funcs.size());
        double n_b_fun = static_cast<double>(m_b_funcs.size());
        std::discrete_distribution<> dis({n_bos * m_weights[0], n_u_fun * m_weights[1], n_b_fun * m_weights[2]});
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
        double n_var = static_cast<double>(m_vars.size());
        std::discrete_distribution<> dis({n_var * m_weights[3], m_weights[4]});
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
        double n_bos = static_cast<double>(m_bos.size());
        double n_u_fun = static_cast<double>(m_u_funcs.size());
        double n_b_fun = static_cast<double>(m_b_funcs.size());
        double n_var = static_cast<double>(m_vars.size());
        std::discrete_distribution<> dis(
            {n_bos * m_weights[0], n_u_fun * m_weights[1], n_b_fun * m_weights[2], n_var * m_weights[3], m_weights[4]});
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
        }
    }
    // Once we know the node type we create one at random out of the user defined possible choices
    switch (type) {
        case node_type::num: {
            // We return a random number in -m_range_dbl, m_range_dbl
            auto value = rngm11(m_e) * m_range_dbl;
            return expression{number{value}};
            break;
        }
        case node_type::var: {
            // We return one of the variables in m_vars
            auto symbol = *detail::random_element(m_vars.begin(), m_vars.end(), m_e);
            return expression{variable{symbol}};
            break;
        }
        case node_type::bo: {
            // We return one of the binary oprators in m_bos with randomly constructed arguments
            auto bo_type = *detail::random_element(m_bos.begin(), m_bos.end(), m_e);
            return expression{binary_operator(bo_type, this->operator()(min_depth, max_depth, depth + 1),
                                              this->operator()(min_depth, max_depth, depth + 1))};
            break;
        }
        case node_type::u_fun: {
            // We return one of the unary functions in m_u_funcs with randomly constructed argument
            auto u_f = *detail::random_element(m_u_funcs.begin(), m_u_funcs.end(), m_e);
            return u_f(this->operator()(min_depth, max_depth, depth + 1));
            break;
        }
        case node_type::b_fun: {
            // We return one of the binary functions in m_b_funcs with randomly constructed arguments
            auto b_f = *detail::random_element(m_b_funcs.begin(), m_b_funcs.end(), m_e);
            return b_f(this->operator()(min_depth, max_depth, depth + 1),
                       this->operator()(min_depth, max_depth, depth + 1));
            break;
        }
        default:
            throw;
    }
};

const std::vector<binary_operator::type> &expression_generator::get_bos() const
{
    return m_bos;
}
const std::vector<expression (*)(expression)> &expression_generator::get_u_funcs() const
{
    return m_u_funcs;
}
const std::vector<expression (*)(expression, expression)> &expression_generator::get_b_funcs() const
{
    return m_b_funcs;
}
const std::vector<std::string> &expression_generator::get_vars() const
{
    return m_vars;
}
const double &expression_generator::get_range_dbl() const
{
    return m_range_dbl;
}
const std::vector<double> &expression_generator::get_weights() const
{
    return m_weights;
}

void expression_generator::set_bos(const std::vector<binary_operator::type> &bos)
{
    m_bos = bos;
}
void expression_generator::set_u_funcs(const std::vector<expression (*)(expression)> &u_funcs)
{
    m_u_funcs = u_funcs;
}
void expression_generator::set_b_funcs(const std::vector<expression (*)(expression, expression)> &b_funcs)
{
    m_b_funcs = b_funcs;
}
void expression_generator::set_vars(const std::vector<std::string> &vars)
{
    m_vars = vars;
}
void expression_generator::set_range_dbl(const double &rd)
{
    m_range_dbl = rd;
}
void expression_generator::set_weights(const std::vector<double> &w)
{
    if (w.size() != 5) {
        throw std::invalid_argument(
            "The weight vector for the probablity distribution of the node type must have size "
            "5 -> (binary operator, unary functions, binary functions, variable, numbers), while I detected a size of: "
            + std::to_string(w.size()));
    }
    m_weights = w;
}

std::ostream &operator<<(std::ostream &os, const expression_generator &eg)
{
    os << "Expression Generator:";
    auto vars = eg.get_vars();

    os << "\nVariables: ";
    for (const auto &var : vars) {
        os << var << " ";
    }
    os << "\nBinary Operators: ";
    auto bos = eg.get_bos();
    for (const auto &bo : bos) {
        switch (bo) {
            case binary_operator::type::add:
                os << "+ ";
                break;
            case binary_operator::type::sub:
                os << "- ";
                break;
            case binary_operator::type::mul:
                os << "* ";
                break;
            case binary_operator::type::div:
                os << "/ ";
                break;
        }
    }
    auto u_funcs = eg.get_u_funcs();
    if (u_funcs.size()) {
        os << "\nUnary Functions: ";
        for (const auto &u_func : u_funcs) {
            os << u_func("."_var) << " ";
        }
    }
    auto b_funcs = eg.get_b_funcs();
    if (b_funcs.size()) {
        os << "\nBinary Functions: ";
        for (const auto &b_func : b_funcs) {
            os << b_func("."_var, "."_var) << " ";
        }
    }
    os << "\nRandom double constants range: ";
    os << "[-" << eg.get_range_dbl() << ", " << eg.get_range_dbl() << "]";
    os << "\nWeights:";
    os << "\n\tBinary operator: " << eg.get_weights()[0];
    os << "\n\tUnary function: " << eg.get_weights()[1];
    os << "\n\tBinary function: " << eg.get_weights()[2];
    os << "\n\tVariable: " << eg.get_weights()[3];
    os << "\n\tConstant: " << eg.get_weights()[4];

    return os << "\n";
}

// Version randomly selecting nodes during traversal (PROBABLY WILL BE REMOVED)
void mutate(expression &e, const expression_generator &generator, const double mut_p, splitmix64 &engine,
            const unsigned min_depth, const unsigned max_depth, unsigned depth)
{
    std::uniform_real_distribution<> rng01(0., 1.);
    if (rng01(engine) < mut_p) {
        e = generator(min_depth, max_depth, depth);
    } else {
        std::visit(
            [&generator, &mut_p, &min_depth, &max_depth, &depth, &engine](auto &node) {
                if constexpr (std::is_same_v<decltype(node), binary_operator &>) {
                    for (auto &branch : node.args()) {
                        mutate(branch, generator, mut_p, engine, min_depth, max_depth, depth + 1);
                    }
                } else if constexpr (std::is_same_v<decltype(node), func &>) {
                    for (auto [b, e] = node.get_mutable_args_it(); b != e; ++b) {
                        mutate(*b, generator, mut_p, engine, min_depth, max_depth, depth + 1);
                    }
                }
            },
            e.value());
    }
}
// Version targeting a node
void mutate(expression &e, std::size_t node_id, const expression_generator &generator, const unsigned min_depth,
            const unsigned max_depth)
{
    auto e_sub_ptr = fetch_from_node_id(e, node_id);
    if (!e_sub_ptr) {
        throw std::invalid_argument("The node id requested: " + std::to_string(node_id)
                                    + " was not found in the expression e1: ");
    }
    *e_sub_ptr = generator(min_depth, max_depth);
}

std::size_t count_nodes(const expression &e)
{
    std::size_t node_counter = 0u;
    detail::count_nodes_impl(e, node_counter);
    return node_counter;
}

expression *fetch_from_node_id(expression &ex, std::size_t node_id)
{
    std::size_t cur_id = 0;
    expression *ret = nullptr;

    detail::fetch_from_node_id_impl(ex, node_id, cur_id, ret);

    return ret;
}

// Crossover
void crossover(expression &e1, expression &e2, splitmix64 &engine)
{
    std::uniform_int_distribution<std::size_t> t1(0, count_nodes(e1) - 1u);
    std::uniform_int_distribution<std::size_t> t2(0, count_nodes(e2) - 1u);
    auto node_id1 = t1(engine);
    auto node_id2 = t2(engine);
    auto e2_sub_ptr = fetch_from_node_id(e1, node_id1);
    auto e1_sub_ptr = fetch_from_node_id(e2, node_id2);
    assert(e2_sub_ptr != nullptr);
    assert(e1_sub_ptr != nullptr);
    swap(*e2_sub_ptr, *e1_sub_ptr);
}

// Crossover targeting specific node_ids
void crossover(expression &e1, expression &e2, std::size_t node_id1, std::size_t node_id2)
{
    auto e2_sub_ptr = fetch_from_node_id(e1, node_id1);
    auto e1_sub_ptr = fetch_from_node_id(e2, node_id2);
    if (!e1_sub_ptr) {
        throw std::invalid_argument("The node id requested: " + std::to_string(node_id1)
                                    + " was not found in the expression e1: ");
    } else if (!e2_sub_ptr) {
        throw std::invalid_argument("The node id requested: " + std::to_string(node_id2)
                                    + " was not found in the expression e2: ");
    }
    swap(*e2_sub_ptr, *e1_sub_ptr);
}

} // namespace heyoka
