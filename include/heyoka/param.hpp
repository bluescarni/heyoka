// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PARAM_HPP
#define HEYOKA_PARAM_HPP

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC param
{
    std::uint32_t m_index;

public:
    explicit param(std::uint32_t);

    param(const param &);
    param(param &&) noexcept;

    param &operator=(const param &);
    param &operator=(param &&) noexcept;

    ~param();

    const std::uint32_t &idx() const;

    std::uint32_t &idx();
};

HEYOKA_DLL_PUBLIC void swap(param &, param &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const param &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const param &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const param &);
HEYOKA_DLL_PUBLIC void rename_variables(param &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const param &, const param &);
HEYOKA_DLL_PUBLIC bool operator!=(const param &, const param &);

HEYOKA_DLL_PUBLIC expression subs(const param &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const param &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const param &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const param &,
                                      const std::unordered_map<std::string, std::vector<double>> &,
                                      const std::vector<double> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const param &, std::size_t &);
[[noreturn]] HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const param &,
                                                           const std::unordered_map<std::string, double> &,
                                                           const std::vector<std::vector<std::size_t>> &,
                                                           std::size_t &);
[[noreturn]] HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const param &,
                                                    const std::unordered_map<std::string, double> &,
                                                    const std::vector<double> &,
                                                    const std::vector<std::vector<std::size_t>> &, std::size_t &,
                                                    double);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
taylor_decompose_in_place(param &&, std::vector<std::pair<expression, std::vector<std::uint32_t>>> &);

} // namespace heyoka

#endif
