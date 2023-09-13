// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PARAM_HPP
#define HEYOKA_PARAM_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
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

HEYOKA_DLL_PUBLIC void swap(param &, param &) noexcept;

class HEYOKA_DLL_PUBLIC param
{
    friend HEYOKA_DLL_PUBLIC void swap(param &, param &) noexcept;

    std::uint32_t m_index;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_index;
    }

public:
    param();

    explicit param(std::uint32_t);

    param(const param &);
    param(param &&) noexcept;

    param &operator=(const param &);
    param &operator=(param &&) noexcept;

    ~param();

    [[nodiscard]] const std::uint32_t &idx() const;
};

namespace detail
{

HEYOKA_DLL_PUBLIC std::size_t hash(const param &) noexcept;

} // namespace detail

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const param &);

HEYOKA_DLL_PUBLIC bool operator==(const param &, const param &);
HEYOKA_DLL_PUBLIC bool operator!=(const param &, const param &);

HEYOKA_DLL_PUBLIC double eval_dbl(const param &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> &);
HEYOKA_DLL_PUBLIC long double eval_ldbl(const param &, const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &);

#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC mppp::real128 eval_f128(const param &, const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &);
#endif

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

HEYOKA_END_NAMESPACE

namespace std
{

template <>
struct hash<heyoka::param> {
    size_t operator()(const heyoka::param &p) const noexcept
    {
        return heyoka::detail::hash(p);
    }
};

} // namespace std

#endif
