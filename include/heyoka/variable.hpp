// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_VARIABLE_HPP
#define HEYOKA_VARIABLE_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC void swap(variable &, variable &) noexcept;

class HEYOKA_DLL_PUBLIC variable
{
    friend HEYOKA_DLL_PUBLIC void swap(variable &, variable &) noexcept;

    std::string m_name;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_name;
    }

public:
    variable();
    explicit variable(std::string);
    variable(const variable &);
    variable(variable &&) noexcept;
    ~variable();

    variable &operator=(const variable &);
    variable &operator=(variable &&) noexcept;

    [[nodiscard]] const std::string &name() const;
};

HEYOKA_DLL_PUBLIC std::size_t hash(const variable &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const variable &);

HEYOKA_DLL_PUBLIC bool operator==(const variable &, const variable &);
HEYOKA_DLL_PUBLIC bool operator!=(const variable &, const variable &);

HEYOKA_DLL_PUBLIC double eval_dbl(const variable &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> &);
HEYOKA_DLL_PUBLIC long double eval_ldbl(const variable &, const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &);

#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC mppp::real128 eval_f128(const variable &, const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &);
#endif

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const variable &,
                                      const std::unordered_map<std::string, std::vector<double>> &,
                                      const std::vector<double> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const variable &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const variable &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const variable &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_END_NAMESPACE

#endif
