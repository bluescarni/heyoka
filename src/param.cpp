// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/param.hpp>

namespace heyoka
{

param::param() : param(0) {}

param::param(std::uint32_t idx) : m_index(idx) {}

param::param(const param &) = default;

param::param(param &&) noexcept = default;

param &param::operator=(const param &) = default;

param &param::operator=(param &&) noexcept = default;

param::~param() = default;

const std::uint32_t &param::idx() const
{
    return m_index;
}

std::uint32_t &param::idx()
{
    return m_index;
}

void swap(param &p0, param &p1) noexcept
{
    std::swap(p0.idx(), p1.idx());
}

std::size_t hash(const param &p)
{
    return std::hash<std::uint32_t>{}(p.idx());
}

std::ostream &operator<<(std::ostream &os, const param &p)
{
    using namespace fmt::literals;

    return os << "p{}"_format(p.idx());
}

namespace detail
{

std::vector<std::string> get_variables(const std::unordered_set<const void *> &, const param &)
{
    return {};
}

void rename_variables(const std::unordered_set<const void *> &, param &,
                      const std::unordered_map<std::string, std::string> &)
{
}

} // namespace detail

bool operator==(const param &p0, const param &p1)
{
    return p0.idx() == p1.idx();
}

bool operator!=(const param &p0, const param &p1)
{
    return !(p0 == p1);
}

expression diff(const param &, const std::string &)
{
    // NOTE: if we ever implement single-precision support,
    // this should be probably changed into 0_flt (i.e., the lowest
    // precision numerical type), so that it does not trigger
    // type promotions in numerical constants. Other similar
    // occurrences as well.
    return 0_dbl;
}

double eval_dbl(const param &p, const std::unordered_map<std::string, double> &, const std::vector<double> &pars)
{
    if (p.idx() >= pars.size()) {
        using namespace fmt::literals;

        throw std::out_of_range(
            "Index error in the double numerical evaluation of a parameter: the parameter index is {}, "
            "but the vector of parametric values has a size of only {}"_format(p.idx(), pars.size()));
    }

    return pars[static_cast<decltype(pars.size())>(p.idx())];
}

long double eval_ldbl(const param &p, const std::unordered_map<std::string, long double> &,
                      const std::vector<long double> &pars)
{
    if (p.idx() >= pars.size()) {
        using namespace fmt::literals;

        throw std::out_of_range(
            "Index error in the long double numerical evaluation of a parameter: the parameter index is {}, "
            "but the vector of parametric values has a size of only {}"_format(p.idx(), pars.size()));
    }

    return pars[static_cast<decltype(pars.size())>(p.idx())];
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 eval_f128(const param &p, const std::unordered_map<std::string, mppp::real128> &,
                        const std::vector<mppp::real128> &pars)
{
    if (p.idx() >= pars.size()) {
        using namespace fmt::literals;

        throw std::out_of_range(
            "Index error in the real 128 numerical evaluation of a parameter: the parameter index is {}, "
            "but the vector of parametric values has a size of only {}"_format(p.idx(), pars.size()));
    }

    return pars[static_cast<decltype(pars.size())>(p.idx())];
}
#endif

void eval_batch_dbl(std::vector<double> &out, const param &p,
                    const std::unordered_map<std::string, std::vector<double>> &, const std::vector<double> &pars)
{
    if (p.idx() >= pars.size()) {
        using namespace fmt::literals;

        throw std::out_of_range(
            "Index error in the batch double numerical evaluation of a parameter: the parameter index is {}, "
            "but the vector of parametric values has a size of only {}"_format(p.idx(), pars.size()));
    }

    std::fill(out.begin(), out.end(), pars[static_cast<decltype(pars.size())>(p.idx())]);
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const param &,
                        std::size_t &node_counter)
{
    node_connections.emplace_back();
    node_counter++;
}

void update_node_values_dbl(std::vector<double> &, const param &, const std::unordered_map<std::string, double> &,
                            const std::vector<std::vector<std::size_t>> &, std::size_t &)
{
    throw not_implemented_error("update_node_values_dbl() not implemented for param");
}

void update_grad_dbl(std::unordered_map<std::string, double> &, const param &,
                     const std::unordered_map<std::string, double> &, const std::vector<double> &,
                     const std::vector<std::vector<std::size_t>> &, std::size_t &, double)
{
    throw not_implemented_error("update_grad_dbl() not implemented for param");
}

} // namespace heyoka
