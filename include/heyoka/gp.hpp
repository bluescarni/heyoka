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
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>

namespace heyoka
{
namespace detail
{
//void extract_subtree_impl(expression &, const expression &, const size_t, size_t &);
//void count_nodes_impl(const expression &, size_t &);
} // namespace detail

class HEYOKA_DLL_PUBLIC expression_generator
{
public:
    enum node_type { num, var, bo, u_fun, b_fun };

private:
    std::vector<binary_operator::type> m_bos;
    std::vector<expression (*)(expression)> m_u_funcs;
    std::vector<expression (*)(expression, expression)> m_b_funcs;
    std::vector<std::string> m_vars;
    mutable detail::random_engine_type m_e;

public:
    explicit expression_generator(const std::vector<std::string> &, ::std::uint64_t);
    expression operator()(unsigned, unsigned, unsigned = 0u) const;

    // getters
    const std::vector<binary_operator::type> &get_bos() const;
    const std::vector<expression (*)(expression)> &get_u_funcs() const;
    const std::vector<expression (*)(expression, expression)> &get_b_funcs() const;
    const std::vector<std::string> &get_vars() const;

    // setters
    void set_bos(const std::vector<binary_operator::type> &);
    void set_u_funcs(const std::vector<expression (*)(expression)> &);
    void set_b_funcs(const std::vector<expression (*)(expression, expression)> &);
    void set_vars(const std::vector<std::string> &);
};

// expression manipulators
HEYOKA_DLL_PUBLIC std::size_t count_nodes(const expression &);
HEYOKA_DLL_PUBLIC void mutate(expression &, const expression_generator &, const double, detail::random_engine_type &,
                              const unsigned = 0u);
HEYOKA_DLL_PUBLIC void extract_subtree(expression &, const expression &, const std::size_t);
HEYOKA_DLL_PUBLIC void inject_subtree(expression &, const expression &, const std::size_t);
HEYOKA_DLL_PUBLIC void crossover(expression &, expression &, detail::random_engine_type &);

} // namespace heyoka

#endif
