// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/param.hpp>

namespace heyoka
{

param::param(std::uint32_t idx) : m_index(idx) {}

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

    return os << "par[{}]"_format(p.idx());
}

std::vector<std::string> get_variables(const param &)
{
    return {};
}

void rename_variables(param &, const std::unordered_map<std::string, std::string> &) {}

bool operator==(const param &p0, const param &p1)
{
    return p0.idx() == p1.idx();
}

bool operator!=(const param &p0, const param &p1)
{
    return !(p0 == p1);
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

llvm::Value *codegen_dbl(llvm_state &s, const param &p)
{
    if (s.par_ptr() == nullptr) {
        throw std::invalid_argument("Cannot run the codegen for a param if par_ptr() is null");
    }

    auto &builder = s.builder();

    return builder.CreateLoad(builder.CreateInBoundsGEP(s.par_ptr(), {builder.getInt32(p.idx())}));
}

llvm::Value *codegen_ldbl(llvm_state &s, const param &p)
{
    // NOTE: the codegen is identical as the LLVM interface
    // is type-erased.
    return codegen_dbl(s, p);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *codegen_f128(llvm_state &s, const param &p)
{
    return codegen_dbl(s, p);
}

#endif

} // namespace heyoka
