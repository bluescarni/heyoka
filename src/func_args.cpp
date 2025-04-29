// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

void func_args::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_args;
}

void func_args::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_args;
}

func_args::func_args() = default;

namespace detail
{

namespace
{

// Implementation of the func_args constructor from a list of arguments.
auto func_args_ctor_impl(std::vector<expression> args, bool shared)
{
    using ret_t = std::variant<std::vector<expression>, func_args::shared_args_t>;

    if (shared) {
        return ret_t(std::make_shared<const std::vector<expression>>(std::move(args)));
    } else {
        return ret_t(std::move(args));
    }
}

} // namespace

} // namespace detail

func_args::func_args(std::vector<expression> args, bool shared)
    : m_args(detail::func_args_ctor_impl(std::move(args), shared))
{
}

func_args::func_args(shared_args_t args)
    : m_args(args ? std::move(args)
                  : throw std::invalid_argument("Cannot initialise a func_args instance from a null pointer"))
{
}

func_args::func_args(const func_args &) = default;

func_args::func_args(func_args &&) noexcept = default;

func_args &func_args::operator=(const func_args &) = default;

func_args &func_args::operator=(func_args &&) noexcept = default;

func_args::~func_args() = default;

const std::vector<expression> &func_args::get_args() const noexcept
{
    if (const auto *args_ptr = std::get_if<std::vector<expression>>(&m_args)) {
        return *args_ptr;
    } else {
        assert(std::get<1>(m_args));
        return *std::get<1>(m_args);
    }
}

func_args::shared_args_t func_args::get_shared_args() const noexcept
{
    if (const auto *shared_args_ptr = std::get_if<shared_args_t>(&m_args)) {
        assert(*shared_args_ptr);
        return *shared_args_ptr;
    } else {
        return {};
    }
}

HEYOKA_END_NAMESPACE
