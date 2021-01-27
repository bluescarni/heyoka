// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_GP_HPP
#define HEYOKA_GP_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/splitmix64.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC expression_generator
{
public:
    enum class node_type { num, var, bo, u_fun, b_fun };

private:
    std::vector<std::string> m_vars;
    std::vector<binary_operator::type> m_bos;
    std::vector<expression (*)(expression)> m_u_funcs;
    std::vector<expression (*)(expression, expression)> m_b_funcs;
    std::vector<double> m_weights;
    double m_range_dbl;
    mutable splitmix64 m_e;

public:
    explicit expression_generator(const std::vector<std::string> &, splitmix64 &);
    expression operator()(unsigned, unsigned, unsigned = 0u) const;

    // getters
    const std::vector<binary_operator::type> &get_bos() const;
    const std::vector<expression (*)(expression)> &get_u_funcs() const;
    const std::vector<expression (*)(expression, expression)> &get_b_funcs() const;
    const std::vector<std::string> &get_vars() const;
    const double &get_range_dbl() const;
    const std::vector<double> &get_weights() const;

    // setters
    void set_bos(const std::vector<binary_operator::type> &);
    void set_u_funcs(const std::vector<expression (*)(expression)> &);
    void set_b_funcs(const std::vector<expression (*)(expression, expression)> &);
    void set_vars(const std::vector<std::string> &);
    void set_range_dbl(const double &);
    void set_weights(const std::vector<double> &);
};

// Streaming operators
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const expression_generator &);

// expression manipulators
HEYOKA_DLL_PUBLIC std::size_t count_nodes(const expression &);
HEYOKA_DLL_PUBLIC expression *fetch_from_node_id(expression &, std::size_t);
HEYOKA_DLL_PUBLIC void mutate(expression &, const expression_generator &, const double, splitmix64 &, const unsigned,
                              const unsigned, const unsigned = 0u);
HEYOKA_DLL_PUBLIC void mutate(expression &, std::size_t, const expression_generator &, const unsigned, const unsigned);
HEYOKA_DLL_PUBLIC void crossover(expression &, expression &, splitmix64 &);
HEYOKA_DLL_PUBLIC void crossover(expression &, expression &, std::size_t, std::size_t);

} // namespace heyoka

#endif
